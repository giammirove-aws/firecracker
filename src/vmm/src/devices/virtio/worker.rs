// Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//! A device-generic worker thread for one virtio queue group.
//!
//! A *queue group* is the set of virtqueues that must be processed by the same thread because they
//! share mutable state. For virtio-block that is a single queue; for virtio-net it is the RX/TX
//! pair, because the TX path synchronously re-enters the RX path to service MMDS and both
//! directions mutate the same MMDS network stack.
//!
//! The split of responsibilities is:
//!
//! * [`QueueGroupWorker`] is the device-specific half. It owns its queues, queue eventfds and
//!   backend **by value**, so the borrow checker — rather than a convention or a lock — guarantees
//!   that only the worker thread touches the data path while the device is activated.
//! * [`WorkerShell`] is the device-independent half that runs inside the thread: the lifecycle
//!   state machine plus the [`MutEventSubscriber`] glue.
//! * [`WorkerHandle`] is the VMM-side half: the control channel, the wake-up eventfd and the join
//!   handle.
//!
//! Threads are created once per device *before* activation (on both the cold-boot and the
//! snapshot-restore path) and park until [`WorkerHandle::start`] hands them their resources. That
//! keeps `clone` out of the VMM thread's seccomp filter.

use std::fmt::Debug;
use std::os::fd::RawFd;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use event_manager::{EventOps, Events, MutEventSubscriber, SubscriberOps};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::eventfd::EventFd;

use crate::EventManager;
use crate::devices::virtio::queue::QueueConfig;
use crate::devices::virtio::transport::VirtioInterrupt;
use crate::logger::{error, warn};
use crate::seccomp::{BpfProgram, apply_filter};

/// Epoll token reserved for the worker's control eventfd.
///
/// Implementors of [`QueueGroupWorker`] must not use it for their own sources.
pub(crate) const CONTROL_TOKEN: u32 = u32::MAX;

/// Errors raised while talking to a queue-group worker thread.
#[derive(Debug, thiserror::Error, displaydoc::Display)]
pub enum WorkerError {
    /// Failed to spawn a queue group worker thread: {0}
    Spawn(std::io::Error),
    /// The queue group worker control channel is disconnected
    Disconnected,
    /// Failed to notify a queue group worker: {0}
    Notify(#[from] std::io::Error),
}

/// Tokens at or above this belong to a device's inline worker, registered on the VMM event loop
/// next to the device's own events (see [`register_inline_events`]); the device forwards them to
/// the worker's [`QueueGroupWorker::process`] with this subtracted.
pub(crate) const INLINE_WORKER_TOKEN_BASE: u32 = 1 << 16;

/// Registers an inline worker's epoll sources on the VMM event loop, so that inline and threaded
/// mode watch the same list.
pub(crate) fn register_inline_events(ops: &mut EventOps, events: Vec<WorkerEvent>) {
    for event in events {
        if let Err(err) = ops.add(Events::with_data_raw(
            event.fd,
            INLINE_WORKER_TOKEN_BASE + event.token,
            event.events,
        )) {
            error!(
                "Failed to register inline worker event {}: {err}",
                event.token
            );
        }
    }
}

/// One epoll source of a worker: the fd to watch, how, and the token [`QueueGroupWorker::process`]
/// receives for it.
#[derive(Debug)]
pub(crate) struct WorkerEvent {
    pub(crate) fd: RawFd,
    pub(crate) token: u32,
    pub(crate) events: EventSet,
}

/// The data-path half of one virtio queue group, owned exclusively by its worker thread.
///
/// Implementors hold their queues, queue eventfds and backend by value. The worker thread is the
/// only owner for as long as the device is activated, so no interior locking is needed on the data
/// path.
pub(crate) trait QueueGroupWorker: Send + Sized + Debug {
    /// The data-path resources this worker owns. They are handed back on reset so that the device
    /// can be activated again without rebuilding them.
    type Resources: Send + Debug;

    /// The epoll sources this worker watches while it runs. The shell registers them on every
    /// transition into `Running` and unregisters them on every transition out, both from this one
    /// list, so the two cannot drift apart.
    fn events(&self) -> Vec<WorkerEvent>;

    /// Handles one epoll event, identified by its token.
    fn process(&mut self, token: u32);

    /// Quiesces the data path so that a consistent snapshot can be taken, leaving everything the
    /// snapshot saves in the worker. Called before the worker is handed back to the device for the
    /// snapshot; the device saves from it, and starts it again when the VM resumes.
    fn prepare_save(&mut self);

    /// Processes every queue that may have pending work, without an epoll event.
    fn kick(&mut self);

    /// Marks dirty the queue pages this worker wrote since the last call.
    fn commit_dirty(&self);

    /// Quiesces the data path and hands the resources back for a later activation.
    fn into_resources(self) -> Self::Resources;

    /// Final quiesce before the thread exits.
    ///
    /// Distinct from [`Self::into_resources`] because teardown may need to flush a backend where a
    /// reset only needs to drain it.
    fn shutdown(self);
}

/// Which half of a device owns its data-path resources.
///
/// Every device whose data path is a set of [`QueueGroupWorker`] queue groups has the same three
/// states, so they live here rather than once per device. The states describe *ownership*, not the
/// virtio device status: the transport configures the queues in place while they are still the
/// device's, they move into the workers on activation, and they come back on reset.
///
/// The configuring and threaded states always hold a `Vec`, even for a device with exactly one
/// queue group today. That is the extension axis — a device gains queue groups by pushing to these
/// vectors — and keeping it uniform is what lets the state machine itself be shared. An inline data
/// path is one queue group on the VMM thread, so it holds one worker.
///
/// `pub(crate)` only so that `persist.rs` can build the `Configuring` state on snapshot restore.
#[derive(Debug)]
pub(crate) enum DeviceState<W: QueueGroupWorker> {
    /// The transport configures the queues in place, before activation. In threaded mode the worker
    /// threads have already been spawned and are parked, waiting for the resources.
    Configuring(Vec<W::Resources>, Vec<WorkerHandle<W>>),
    /// Activated: the resources have moved into the workers.
    Active(ActiveDevice<W>),
    /// Transient placeholder, held only while the state machine moves resources around.
    Placeholder,
}

/// What the configuration half of a device can reach once it is activated.
#[derive(Debug)]
pub(crate) enum ActiveDevice<W: QueueGroupWorker> {
    /// Driven from the VMM event loop: the device still owns its one worker and calls it directly,
    /// queue configuration included. Boxed, because a worker is far larger than the threaded
    /// variant.
    Inline { worker: Box<W> },
    /// Driven from one thread per queue group, reached only through their handles.
    Threaded {
        handles: Vec<WorkerHandle<W>>,
        /// The interrupt the transport signals configuration changes on.
        ///
        /// Kept here, rather than only in the workers, because the workers are on other threads:
        /// the VMM thread has no way to reach the clone that travelled with them. In inline mode
        /// there is no such split, so the worker's own copy is the only one.
        interrupt: Arc<dyn VirtioInterrupt>,
        /// A clone of the queue configuration, taken before the queues moved into the workers.
        ///
        /// The transport keeps reading it for MMIO and PCI register accesses. This is sound
        /// because `QueueConfig` is written only before `DRIVER_OK`: the driver programs the ring
        /// addresses and `ready`, then activation happens, and after that the transport only reads.
        queue_cfg: Vec<QueueConfig>,
        /// The workers, while they are taken back from their threads for a snapshot (see
        /// [`ActiveDevice::prepare_save`]); the next [`ActiveDevice::kick`] starts them again.
        /// Empty while the threads run them.
        paused: Vec<W>,
    },
}

impl<W: QueueGroupWorker + 'static> ActiveDevice<W> {
    /// Hands the freshly built workers over, either to their threads or to the device itself.
    ///
    /// `handles` is empty exactly when the data path is inline, which serves one queue group; the
    /// caller has already checked that against its own `threaded` configuration. `queue_cfg`
    /// clones the workers' queue configuration, in transport order, for a threaded data path.
    pub(crate) fn start(
        mut workers: Vec<W>,
        handles: Vec<WorkerHandle<W>>,
        interrupt: Arc<dyn VirtioInterrupt>,
        queue_cfg: impl FnOnce(&[W]) -> Vec<QueueConfig>,
    ) -> Result<Self, WorkerError> {
        if handles.is_empty() {
            assert_eq!(workers.len(), 1, "an inline data path has one queue group");
            let worker = Box::new(workers.pop().unwrap());
            return Ok(Self::Inline { worker });
        }

        assert_eq!(
            handles.len(),
            workers.len(),
            "one worker thread must be spawned for each queue group"
        );
        let queue_cfg = queue_cfg(&workers);
        for (handle, worker) in handles.iter().zip(workers) {
            handle.start(worker)?;
        }
        Ok(Self::Threaded {
            handles,
            interrupt,
            queue_cfg,
            paused: Vec::new(),
        })
    }

    /// The one worker the device drives itself, inline. `None` in threaded mode, where the workers
    /// are on other threads and can only be reached through their handles.
    pub(crate) fn inline_worker_mut(&mut self) -> Option<&mut W> {
        match self {
            Self::Inline { worker, .. } => Some(worker.as_mut()),
            Self::Threaded { .. } => None,
        }
    }

    /// See [`Self::inline_worker_mut`].
    pub(crate) fn inline_worker(&self) -> Option<&W> {
        match self {
            Self::Inline { worker, .. } => Some(worker.as_ref()),
            Self::Threaded { .. } => None,
        }
    }

    /// Quiesces the data path so that a consistent snapshot can be taken.
    ///
    /// A snapshot is a reset followed by a restart, as crosvm's `virtio_sleep` and `virtio_wake`
    /// do, except that nothing is cleared: every worker thread quiesces its worker and hands it
    /// back whole, and parks. The device then saves from the workers it holds, and the next
    /// [`Self::kick`], when the VM resumes, starts them again. An inline worker quiesces in place
    /// because the VMM thread is already the only one that could touch it. A second snapshot of
    /// the same pause finds the workers taken already.
    ///
    /// Workers quiesce one at a time, in queue-group order, each finishing before the next starts.
    /// Devices rely on it: block flushes its backing file from its last queue only.
    pub(crate) fn prepare_save(&mut self) {
        match self {
            Self::Inline { worker, .. } => worker.prepare_save(),
            Self::Threaded {
                handles, paused, ..
            } if paused.is_empty() => {
                *paused = handles
                    .iter()
                    .map(|handle| {
                        handle
                            .hand_back()
                            .expect("an active queue group worker must answer a hand-back")
                            .expect("an active queue group worker must hand its worker back")
                    })
                    .collect();
            }
            Self::Threaded { .. } => {}
        }
    }

    /// Starts the workers taken back for a snapshot on their threads again, then asks every
    /// worker thread to look for pending queue work.
    ///
    /// Returns `false` if the data path is inline, in which case the caller has to poke the queue
    /// eventfds instead.
    pub(crate) fn kick(&mut self) -> bool {
        let Self::Threaded {
            handles, paused, ..
        } = self
        else {
            return false;
        };
        for (handle, worker) in handles.iter().zip(paused.drain(..)) {
            if let Err(err) = handle.start(worker) {
                error!("Failed to restart a queue group worker: {err}");
            }
        }
        for handle in handles.iter() {
            if let Err(err) = handle.kick() {
                error!("Failed to kick a queue group worker: {err}");
            }
        }
        true
    }

    /// Marks dirty the queue pages the workers wrote since the last call: an inline worker and the
    /// workers taken back for a snapshot directly, the running ones on their threads.
    pub(crate) fn commit_dirty(&self) {
        match self {
            Self::Inline { worker } => worker.commit_dirty(),
            Self::Threaded {
                handles, paused, ..
            } => {
                for handle in handles {
                    if let Err(err) = handle.commit_dirty() {
                        error!("Failed to commit the dirty pages of a queue group worker: {err}");
                    }
                }
                for worker in paused {
                    worker.commit_dirty();
                }
            }
        }
    }

    /// Takes the data-path resources back, and with them the worker threads, so that the device can
    /// be activated again without rebuilding either.
    pub(crate) fn into_resources(self) -> (Vec<W::Resources>, Vec<WorkerHandle<W>>) {
        match self {
            Self::Inline { worker, .. } => (vec![worker.into_resources()], Vec::new()),
            // Workers taken back for a snapshot are the device's already, and their threads are
            // parked: they hand their resources back here.
            Self::Threaded {
                handles, paused, ..
            } if !paused.is_empty() => {
                (paused.into_iter().map(W::into_resources).collect(), handles)
            }
            Self::Threaded { handles, .. } => {
                let resources = handles
                    .iter()
                    .map(|handle| {
                        handle
                            .reset()
                            .expect("an active queue group worker must answer a reset")
                            .expect("an active queue group worker must return its resources")
                    })
                    .collect();
                (resources, handles)
            }
        }
    }
}

impl<W: QueueGroupWorker + 'static> DeviceState<W> {
    /// Whether the device is activated.
    pub(crate) fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    /// The activated data path, if there is one.
    pub(crate) fn active(&self) -> Option<&ActiveDevice<W>> {
        match self {
            Self::Active(active) => Some(active),
            Self::Configuring(..) | Self::Placeholder => None,
        }
    }

    /// See [`Self::active`].
    pub(crate) fn active_mut(&mut self) -> Option<&mut ActiveDevice<W>> {
        match self {
            Self::Active(active) => Some(active),
            Self::Configuring(..) | Self::Placeholder => None,
        }
    }

    /// Asks the activated data path to look for pending queue work, without restarting it: the
    /// inline worker processes its queues in place, and every worker thread is kicked.
    ///
    /// Unlike [`ActiveDevice::kick`], this never starts the workers taken back for a snapshot:
    /// they stay with the device, and their parked threads ignore the kick.
    pub(crate) fn kick_running(&mut self) {
        match self {
            Self::Active(ActiveDevice::Inline { worker, .. }) => worker.kick(),
            Self::Active(ActiveDevice::Threaded { handles, .. }) => {
                for handle in handles.iter() {
                    if let Err(err) = handle.kick() {
                        error!("Failed to kick a queue group worker: {err}");
                    }
                }
            }
            Self::Configuring(..) | Self::Placeholder => {}
        }
    }

    /// The worker the device drives itself, which only exists once it is activated inline.
    pub(crate) fn inline_worker_mut(&mut self) -> Option<&mut W> {
        self.active_mut().and_then(ActiveDevice::inline_worker_mut)
    }

    /// See [`Self::inline_worker_mut`].
    pub(crate) fn inline_worker(&self) -> Option<&W> {
        self.active().and_then(ActiveDevice::inline_worker)
    }

    /// Spawns one parked worker thread per queue group, unless they are already running.
    ///
    /// `thread_name` names the thread of the queue group with the given index. If any thread fails
    /// to spawn, the ones before it are torn down again: a device with only some of its groups
    /// threaded has nothing to drive the rest with.
    pub(crate) fn spawn_workers(
        &mut self,
        seccomp_filter: &Arc<BpfProgram>,
        thread_name: impl Fn(usize) -> String,
    ) -> Result<(), WorkerError> {
        let Self::Configuring(resources, handles) = self else {
            return Ok(());
        };
        if !handles.is_empty() {
            assert_eq!(handles.len(), resources.len());
            return Ok(());
        }

        let mut spawned = Vec::with_capacity(resources.len());
        for group_index in 0..resources.len() {
            match WorkerHandle::spawn(seccomp_filter.clone(), thread_name(group_index)) {
                Ok(handle) => spawned.push(handle),
                Err(err) => {
                    spawned.into_iter().for_each(WorkerHandle::finish);
                    return Err(err);
                }
            }
        }
        *handles = spawned;
        Ok(())
    }

    /// Tears the data path down for good, on the way to dropping the device.
    ///
    /// Every worker gets its final quiesce — which for some backends is a flush, not just a drain —
    /// and every thread is joined. A device dropped before it was ever activated still has parked
    /// threads to join. As in [`ActiveDevice::prepare_save`], workers quiesce one at a time, in
    /// queue-group order.
    pub(crate) fn shutdown(&mut self) {
        match std::mem::replace(self, Self::Placeholder) {
            Self::Active(ActiveDevice::Inline { worker, .. }) => worker.shutdown(),
            // Workers taken back for a snapshot get their final quiesce here, on the VMM thread
            // that holds them, before their parked threads are joined.
            Self::Active(ActiveDevice::Threaded {
                handles, paused, ..
            }) => {
                paused.into_iter().for_each(W::shutdown);
                handles.into_iter().for_each(WorkerHandle::finish);
            }
            Self::Configuring(_, handles) => {
                handles.into_iter().for_each(WorkerHandle::finish);
            }
            Self::Placeholder => {}
        }
    }
}

/// The lifecycle state of a queue-group worker thread.
#[derive(Debug)]
enum WorkerState<W> {
    /// The thread is running but holds no resources: either not activated yet, or reset.
    Parked,
    /// The thread owns the resources and its epoll sources are registered. Boxed, because a worker
    /// is far larger than the other states.
    Running(Box<W>),
    /// The thread is about to exit.
    Finished,
}

/// The device-independent control vocabulary of a queue-group worker.
///
/// Every request that needs an answer carries its own reply `Sender`, so a reply can never be
/// delivered to the wrong requester and there is nothing to demultiplex on the VMM side.
enum LifecycleMsg<W: QueueGroupWorker> {
    /// Hand the resources over and start processing.
    Start(Box<W>),
    /// Quiesce the data path for a snapshot, hand the whole worker back and return to `Parked`.
    /// `Start` hands it over again.
    HandBack(Sender<Option<W>>),
    /// Process any pending queue work without waiting for an epoll event.
    Kick,
    /// Mark dirty the queue pages the worker wrote since the last call. Answered in every state;
    /// a parked thread holds no queues.
    CommitDirty(Sender<()>),
    /// Hand the resources back and return to `Parked`.
    Reset(Sender<Option<W::Resources>>),
    /// Tear the worker down and exit the thread.
    Finish,
}

/// The in-thread half of a queue-group worker: the lifecycle state machine and the epoll glue.
#[derive(Debug)]
pub(crate) struct WorkerShell<W: QueueGroupWorker> {
    state: WorkerState<W>,
    control_evt: EventFd,
    control_rx: Receiver<LifecycleMsg<W>>,
}

/// The VMM-side handle of a queue-group worker thread.
#[derive(Debug)]
pub(crate) struct WorkerHandle<W: QueueGroupWorker> {
    control_tx: Sender<LifecycleMsg<W>>,
    control_evt: EventFd,
    join: JoinHandle<()>,
}

impl<W: QueueGroupWorker + 'static> WorkerHandle<W> {
    /// Spawns a parked worker thread.
    ///
    /// `seccomp_filter` is mandatory rather than settable after the fact, so that a worker thread
    /// cannot end up unfiltered because a setter was forgotten.
    pub(crate) fn spawn(
        seccomp_filter: Arc<BpfProgram>,
        thread_name: String,
    ) -> Result<Self, WorkerError> {
        // One control eventfd, shared between the worker (which registers it on its epoll) and the
        // handle (which writes to it to wake the worker). Both refer to the same kernel object.
        let control_evt = EventFd::new(libc::EFD_NONBLOCK).map_err(WorkerError::Spawn)?;
        let handle_evt = control_evt.try_clone().map_err(WorkerError::Spawn)?;

        let (control_tx, control_rx) = channel::<LifecycleMsg<W>>();

        let join = thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                // Create the epoll instance first: `epoll_create1` is not in the worker filter.
                // Then lock the thread down, then enter the loop.
                let event_manager =
                    EventManager::new().expect("Failed to create the queue group worker epoll");

                if let Err(err) = apply_filter(&seccomp_filter) {
                    panic!("Failed to apply the seccomp filter on a queue group worker: {err}");
                }

                run_worker_loop::<W>(event_manager, control_evt, control_rx)
            })
            .map_err(WorkerError::Spawn)?;

        Ok(Self {
            control_tx,
            control_evt: handle_evt,
            join,
        })
    }

    /// Sends a control message and wakes the worker.
    fn send(&self, msg: LifecycleMsg<W>) -> Result<(), WorkerError> {
        self.control_tx
            .send(msg)
            .map_err(|_| WorkerError::Disconnected)?;
        self.control_evt.write(1)?;
        Ok(())
    }

    /// Sends a control message and blocks until the worker answers it.
    fn request<T>(
        &self,
        build: impl FnOnce(Sender<T>) -> LifecycleMsg<W>,
    ) -> Result<T, WorkerError> {
        let (reply_tx, reply_rx) = channel();
        self.send(build(reply_tx))?;
        reply_rx.recv().map_err(|_| WorkerError::Disconnected)
    }

    /// Hands the data-path resources to the worker and starts processing.
    pub(crate) fn start(&self, worker: W) -> Result<(), WorkerError> {
        self.send(LifecycleMsg::Start(Box::new(worker)))
    }

    /// Quiesces the worker for a snapshot and takes it back from the thread, which parks until
    /// [`Self::start`] hands the worker over again.
    pub(crate) fn hand_back(&self) -> Result<Option<W>, WorkerError> {
        self.request(LifecycleMsg::HandBack)
    }

    /// Asks the worker to look for pending queue work. Does not change its state.
    pub(crate) fn kick(&self) -> Result<(), WorkerError> {
        self.send(LifecycleMsg::Kick)
    }

    /// Marks dirty the queue pages the worker wrote since the last call. Does not change its state.
    pub(crate) fn commit_dirty(&self) -> Result<(), WorkerError> {
        self.request(LifecycleMsg::CommitDirty)
    }

    /// Takes the data-path resources back, so the device can be activated again.
    pub(crate) fn reset(&self) -> Result<Option<W::Resources>, WorkerError> {
        self.request(LifecycleMsg::Reset)
    }

    /// Tears the worker down and joins its thread.
    pub(crate) fn finish(self) {
        if let Err(err) = self.send(LifecycleMsg::Finish) {
            error!("Failed to ask a queue group worker to finish: {err}");
        }

        self.join.join().unwrap_or_else(|_| {
            error!("A queue group worker thread panicked during teardown");
        });
    }
}

impl<W: QueueGroupWorker> WorkerShell<W> {
    /// Registers the worker's epoll sources. Called on every transition into `Running`.
    ///
    /// A failure panics: a worker that cannot watch its queues would be running and never serve
    /// them, leaving the guest's I/O stuck with nothing but one line in the log to show for it.
    fn register_worker_events(worker: &W, ops: &mut EventOps) {
        for event in worker.events() {
            if let Err(err) = ops.add(Events::with_data_raw(event.fd, event.token, event.events)) {
                panic!(
                    "Failed to register queue group worker event {}: {err}",
                    event.token
                );
            }
        }
    }

    /// Unregisters them. Called on every transition out of `Running`.
    fn unregister_worker_events(worker: &W, ops: &mut EventOps) {
        for event in worker.events() {
            if let Err(err) = ops.remove(Events::with_data_raw(event.fd, event.token, event.events))
            {
                error!(
                    "Failed to unregister queue group worker event {}: {err}",
                    event.token
                );
            }
        }
    }

    fn register_control_event(&self, ops: &mut EventOps) {
        // Every request reaches the worker through this eventfd. A worker that cannot watch it
        // would leave each later caller blocked forever on its reply, so do not carry on.
        ops.add(Events::with_data(
            &self.control_evt,
            CONTROL_TOKEN,
            EventSet::IN,
        ))
        .expect("Failed to register the queue group worker control event");
    }

    fn process_control_event(&mut self, ops: &mut EventOps) {
        if let Err(err) = self.control_evt.read() {
            error!("Failed to read the queue group worker control event: {err}");
            return;
        }

        while let Ok(msg) = self.control_rx.try_recv() {
            self.process_control_msg(msg, ops);
            if self.is_finished() {
                break;
            }
        }
    }

    fn process_control_msg(&mut self, msg: LifecycleMsg<W>, ops: &mut EventOps) {
        match msg {
            LifecycleMsg::Start(worker) => self.start(worker, ops),
            LifecycleMsg::HandBack(reply) => self.hand_back(&reply, ops),
            LifecycleMsg::Kick => {
                if let WorkerState::Running(worker) = &mut self.state {
                    worker.kick();
                }
            }
            LifecycleMsg::CommitDirty(reply) => {
                if let WorkerState::Running(worker) = &self.state {
                    worker.commit_dirty();
                }
                if reply.send(()).is_err() {
                    error!("Failed to answer a queue group worker dirty commit");
                }
            }
            LifecycleMsg::Reset(reply) => self.reset(&reply, ops),
            LifecycleMsg::Finish => self.finish(ops),
        }
    }

    fn start(&mut self, worker: Box<W>, ops: &mut EventOps) {
        if !matches!(self.state, WorkerState::Parked) {
            warn!("Start requested while the queue group worker is not parked");
            return;
        }

        Self::register_worker_events(&worker, ops);
        self.state = WorkerState::Running(worker);
    }

    fn hand_back(&mut self, reply: &Sender<Option<W>>, ops: &mut EventOps) {
        let worker = match std::mem::replace(&mut self.state, WorkerState::Parked) {
            WorkerState::Running(mut worker) => {
                Self::unregister_worker_events(&worker, ops);
                worker.prepare_save();
                Some(*worker)
            }
            WorkerState::Parked => {
                warn!("Hand-back requested while the queue group worker is parked");
                None
            }
            WorkerState::Finished => {
                self.state = WorkerState::Finished;
                None
            }
        };

        if reply.send(worker).is_err() {
            error!("Failed to hand the queue group worker back to the device");
        }
    }

    fn reset(&mut self, reply: &Sender<Option<W::Resources>>, ops: &mut EventOps) {
        let resources = match std::mem::replace(&mut self.state, WorkerState::Parked) {
            WorkerState::Running(worker) => {
                Self::unregister_worker_events(&worker, ops);
                Some(worker.into_resources())
            }
            WorkerState::Parked => {
                warn!("Reset requested while the queue group worker is parked");
                None
            }
            WorkerState::Finished => {
                self.state = WorkerState::Finished;
                None
            }
        };

        if reply.send(resources).is_err() {
            error!("Failed to hand the resources back to the device on reset");
        }
    }

    fn finish(&mut self, ops: &mut EventOps) {
        match std::mem::replace(&mut self.state, WorkerState::Finished) {
            WorkerState::Running(worker) => {
                Self::unregister_worker_events(&worker, ops);
                worker.shutdown();
            }
            WorkerState::Parked | WorkerState::Finished => {}
        }
        self.state = WorkerState::Finished;
    }

    fn is_finished(&self) -> bool {
        matches!(self.state, WorkerState::Finished)
    }
}

impl<W: QueueGroupWorker> MutEventSubscriber for WorkerShell<W> {
    fn process(&mut self, event: Events, ops: &mut EventOps) {
        let token = event.data();
        let event_set = event.event_set();

        // TODO: also check for errors. Pending high level discussions on how we want to handle
        // errors in devices.
        if !EventSet::IN.contains(event_set) {
            warn!(
                "A queue group worker received an unsupported event set {event_set:?} on token \
                 {token}"
            );
            return;
        }

        if token == CONTROL_TOKEN {
            self.process_control_event(ops);
            return;
        }

        match &mut self.state {
            WorkerState::Running(worker) => worker.process(token),
            _ => warn!("A queue group worker is not running, dropping the event on token {token}"),
        }
    }

    fn init(&mut self, ops: &mut EventOps) {
        self.register_control_event(ops);
    }
}

/// Runs the worker thread's event loop until it is told to finish.
fn run_worker_loop<W: QueueGroupWorker + 'static>(
    mut event_manager: EventManager,
    control_evt: EventFd,
    control_rx: Receiver<LifecycleMsg<W>>,
) {
    // The `Mutex` here only satisfies `EventManager`'s subscriber type. It is thread-local to the
    // worker and never contended, so it is not a hot-path lock.
    let shell = Arc::new(Mutex::new(WorkerShell {
        state: WorkerState::Parked,
        control_evt,
        control_rx,
    }));
    let subscriber: Arc<Mutex<dyn MutEventSubscriber>> = shell.clone();
    event_manager.add_subscriber(subscriber);

    loop {
        if let Err(err) = event_manager.run() {
            error!("Queue group worker event loop error: {err:?}");
        }
        if shell
            .lock()
            .expect("Poisoned queue group worker lock")
            .is_finished()
        {
            break;
        }
    }

    // Drop the `EventManager` first so it releases the subscriber clone it holds; only then does
    // the `Arc` reach a strong count of one.
    drop(event_manager);
    let shell = Arc::try_unwrap(shell)
        .expect("Queue group worker references outlived the event loop")
        .into_inner()
        .expect("Poisoned queue group worker lock");
    assert!(matches!(shell.state, WorkerState::Finished));
}

#[cfg(test)]
pub(crate) mod tests {
    use std::os::fd::AsRawFd;
    use std::sync::atomic::{AtomicU32, Ordering};

    use super::*;

    /// Counts the lifecycle callbacks the shell made, so a test can assert on transitions without
    /// needing a real device backend.
    #[derive(Debug, Default)]
    pub(crate) struct MockCounters {
        pub processed: AtomicU32,
        pub prepared_save: AtomicU32,
        pub kicked: AtomicU32,
        pub committed_dirty: AtomicU32,
        pub shut_down: AtomicU32,
    }

    /// A [`QueueGroupWorker`] with no backend at all, used to test the generic shell.
    #[derive(Debug)]
    pub(crate) struct MockWorker {
        counters: Arc<MockCounters>,
    }

    /// What a reset hands back. The mock has no backend, so there is nothing in it.
    #[derive(Debug)]
    pub(crate) struct MockResources;

    impl QueueGroupWorker for MockWorker {
        type Resources = MockResources;

        fn events(&self) -> Vec<WorkerEvent> {
            Vec::new()
        }

        fn process(&mut self, _token: u32) {
            self.counters.processed.fetch_add(1, Ordering::Relaxed);
        }

        fn prepare_save(&mut self) {
            self.counters.prepared_save.fetch_add(1, Ordering::Relaxed);
        }

        fn kick(&mut self) {
            self.counters.kicked.fetch_add(1, Ordering::Relaxed);
        }

        fn commit_dirty(&self) {
            self.counters
                .committed_dirty
                .fetch_add(1, Ordering::Relaxed);
        }

        fn into_resources(self) -> MockResources {
            MockResources
        }

        fn shutdown(self) {
            self.counters.shut_down.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Spawns a mock worker thread and returns its handle plus the shared counters.
    fn spawn_mock() -> (WorkerHandle<MockWorker>, Arc<MockCounters>) {
        let counters = Arc::new(MockCounters::default());
        let handle =
            WorkerHandle::<MockWorker>::spawn(Arc::new(vec![]), "fc_q0_mock".to_string()).unwrap();
        (handle, counters)
    }

    #[test]
    #[should_panic(expected = "Failed to register the queue group worker control event")]
    fn test_control_event_registration_failure() {
        let (_control_tx, control_rx) = channel::<LifecycleMsg<MockWorker>>();
        let shell = Arc::new(Mutex::new(WorkerShell {
            state: WorkerState::Parked,
            control_evt: EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            control_rx,
        }));
        let mut event_manager = EventManager::new().unwrap();
        event_manager.add_subscriber(shell.clone());
        // The second registration of the same eventfd fails.
        event_manager.add_subscriber(shell);
    }

    /// A worker that watches one fd.
    #[derive(Debug)]
    struct FdWorker(RawFd);

    impl QueueGroupWorker for FdWorker {
        type Resources = ();

        fn events(&self) -> Vec<WorkerEvent> {
            vec![WorkerEvent {
                fd: self.0,
                token: 0,
                events: EventSet::IN,
            }]
        }

        fn process(&mut self, _token: u32) {}

        fn prepare_save(&mut self) {}

        fn kick(&mut self) {}

        fn commit_dirty(&self) {}

        fn into_resources(self) {}

        fn shutdown(self) {}
    }

    /// Watches one fd on the event loop it is added to, so that registering the same fd on that
    /// loop again fails.
    struct FdSquatter(RawFd);

    impl MutEventSubscriber for FdSquatter {
        fn process(&mut self, _: Events, _: &mut EventOps) {}

        fn init(&mut self, ops: &mut EventOps) {
            ops.add(Events::with_data_raw(self.0, 0, EventSet::IN))
                .unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "Failed to register queue group worker event")]
    fn test_worker_event_registration_failure() {
        // A worker that cannot watch its queue would be running and never serve it, so the
        // thread stops instead of entering `Running`.
        let queue_evt = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let (control_tx, control_rx) = channel::<LifecycleMsg<FdWorker>>();
        let control_evt = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        let shell = Arc::new(Mutex::new(WorkerShell {
            state: WorkerState::Parked,
            control_evt: control_evt.try_clone().unwrap(),
            control_rx,
        }));
        let mut event_manager = EventManager::new().unwrap();
        let queue_fd = queue_evt.as_raw_fd();
        event_manager.add_subscriber(Arc::new(Mutex::new(FdSquatter(queue_fd))));
        event_manager.add_subscriber(shell);

        control_tx
            .send(LifecycleMsg::Start(Box::new(FdWorker(queue_fd))))
            .unwrap();
        control_evt.write(1).unwrap();
        event_manager.run().unwrap();
    }

    #[test]
    fn test_worker_lifecycle_transitions() {
        let (handle, counters) = spawn_mock();

        // Requests that need the resources are answered even while the thread is parked, rather
        // than blocking the VMM thread forever.
        assert!(handle.hand_back().unwrap().is_none());
        assert!(handle.reset().unwrap().is_none());

        handle
            .start(MockWorker {
                counters: counters.clone(),
            })
            .unwrap();
        // `hand_back` round-trips, so by the time it returns the `start` before it has been
        // handled.
        let worker = handle.hand_back().unwrap().unwrap();
        assert_eq!(counters.prepared_save.load(Ordering::Relaxed), 1);

        // While the device holds the worker, the thread is parked: a second hand-back finds
        // nothing, and does not quiesce twice.
        assert!(handle.hand_back().unwrap().is_none());
        assert_eq!(counters.prepared_save.load(Ordering::Relaxed), 1);

        // Starting it again, as the VM resumes, lets the kick through.
        handle.start(worker).unwrap();
        handle.kick().unwrap();
        // Reset hands the resources back and returns the thread to `Parked`. It round-trips too,
        // so the start and the kick have been processed by then.
        assert!(handle.reset().unwrap().is_some());
        assert_eq!(counters.kicked.load(Ordering::Relaxed), 1);
        assert!(handle.hand_back().unwrap().is_none());

        handle.finish();
        // The worker was reset before teardown, so there was nothing left to shut down.
        assert_eq!(counters.shut_down.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_worker_kick_does_not_restart() {
        let (handle, counters) = spawn_mock();
        handle
            .start(MockWorker {
                counters: counters.clone(),
            })
            .unwrap();
        let _worker = handle.hand_back().unwrap().unwrap();

        // A kick arriving while the device holds the worker for a snapshot finds a parked thread,
        // and must not start anything on it.
        handle.kick().unwrap();
        assert!(handle.hand_back().unwrap().is_none());
        assert_eq!(counters.kicked.load(Ordering::Relaxed), 0);

        handle.finish();
        assert_eq!(counters.shut_down.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_worker_commit_dirty_in_every_state() {
        let (handle, counters) = spawn_mock();

        // A parked thread holds no queues: answered, nothing committed.
        handle.commit_dirty().unwrap();
        assert_eq!(counters.committed_dirty.load(Ordering::Relaxed), 0);

        // A running worker commits its own queues without leaving `Running`.
        handle
            .start(MockWorker {
                counters: counters.clone(),
            })
            .unwrap();
        handle.commit_dirty().unwrap();
        assert_eq!(counters.committed_dirty.load(Ordering::Relaxed), 1);
        handle.kick().unwrap();
        assert!(handle.reset().unwrap().is_some());
        assert_eq!(counters.kicked.load(Ordering::Relaxed), 1);

        handle.finish();
    }

    #[test]
    fn test_worker_finish_shuts_a_running_worker_down() {
        let (handle, counters) = spawn_mock();
        handle
            .start(MockWorker {
                counters: counters.clone(),
            })
            .unwrap();
        handle.finish();
        assert_eq!(counters.shut_down.load(Ordering::Relaxed), 1);
    }
}
