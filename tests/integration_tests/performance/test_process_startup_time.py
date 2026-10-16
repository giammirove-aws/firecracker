# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that the process startup time up to socket bind is within spec."""

import os
import time

import pytest

from framework.artifacts import GUEST_KERNEL_DEFAULT, pin_guest_kernel
from host_tools.cargo_build import run_seccompiler_bin

ITERATIONS = 100

# A cgroup v2 setup, as used in production: the jailer creates a cgroup for each microVM.
CGROUPS_V2 = ["memory.max=1073741824", "cpu.weight=100"]

pytestmark = pin_guest_kernel(GUEST_KERNEL_DEFAULT)


def _set_cgroups(microvm, cgroups):
    """Configure `cgroups`: "none", "v2", or "v2_clone" (v2 with --clone-into-cgroup)."""
    if cgroups == "none":
        return ""
    if not os.path.exists("/sys/fs/cgroup/cgroup.controllers"):
        pytest.skip("cgroupsv2 only")
    microvm.jailer.cgroup_ver = 2
    microvm.jailer.cgroups = CGROUPS_V2
    microvm.jailer.clone_into_cgroup = cgroups == "v2_clone"
    return f"_cgroup{cgroups}"


@pytest.mark.nonci
@pytest.mark.parametrize("cgroups", ["none", "v2", "v2_clone"])
def test_startup_time_new_pid_ns(
    microvm_factory, guest_kernel, rootfs, metrics, cgroups
):
    """
    Check startup time when jailer is spawned in a new PID namespace.
    """
    for _ in range(ITERATIONS):
        microvm = microvm_factory.build(guest_kernel, rootfs)
        # Measure this mode alone: the framework enables both by default.
        microvm.jailer.new_pid_ns = True
        microvm.jailer.daemonize = False
        suffix = _set_cgroups(microvm, cgroups)
        _test_startup_time(microvm, metrics, f"new_pid_ns{suffix}")
        microvm.kill()


@pytest.mark.nonci
@pytest.mark.parametrize("cgroups", ["none", "v2", "v2_clone"])
def test_startup_time_daemonize(
    microvm_factory, guest_kernel, rootfs, metrics, cgroups
):
    """
    Check startup time when jailer detaches Firecracker from the controlling terminal.
    """
    for _ in range(ITERATIONS):
        microvm = microvm_factory.build(guest_kernel, rootfs)
        # Measure this mode alone: the framework enables both by default.
        microvm.jailer.daemonize = True
        microvm.jailer.new_pid_ns = False
        suffix = _set_cgroups(microvm, cgroups)
        _test_startup_time(microvm, metrics, f"daemonize{suffix}")
        microvm.kill()


@pytest.mark.nonci
def test_startup_time_custom_seccomp(microvm_factory, guest_kernel, rootfs, metrics):
    """
    Check the startup time when using custom seccomp filters.
    """
    for _ in range(ITERATIONS):
        microvm = microvm_factory.build(guest_kernel, rootfs)
        _custom_filter_setup(microvm)
        _test_startup_time(microvm, metrics, "custom_seccomp")
        microvm.kill()


def _test_startup_time(microvm, metrics, test_suffix: str):
    test_start_time = time.time()
    microvm.spawn()
    microvm.basic_config(vcpu_count=2, mem_size_mib=1024)
    metrics.set_dimensions(
        {**microvm.dimensions, "performance_test": f"test_startup_time_{test_suffix}"}
    )
    microvm.start()
    datapoints = microvm.get_all_metrics()

    # The metrics should be at index 0.
    # Since metrics are flushed at InstanceStart, the first line will suffice.
    test_end_time = time.time()
    fc_metrics = datapoints[0]
    startup_time_us = fc_metrics["api_server"]["process_startup_time_us"]
    cpu_startup_time_us = fc_metrics["api_server"]["process_startup_time_cpu_us"]

    print(
        "Process startup time is: {} us ({} CPU us)".format(
            startup_time_us, cpu_startup_time_us
        )
    )

    assert cpu_startup_time_us > 0
    # Check that startup time is not a huge value (overflow)
    # This is to catch issues like the ones introduced in PR
    # https://github.com/firecracker-microvm/firecracker/pull/4305
    test_time_delta_us = (test_end_time - test_start_time) * 1000 * 1000
    assert startup_time_us < test_time_delta_us
    assert cpu_startup_time_us < test_time_delta_us

    metrics.put_metric("startup_cpu_time", cpu_startup_time_us, unit="Microseconds")
    metrics.put_metric("startup_time", startup_time_us, unit="Microseconds")


def _custom_filter_setup(test_microvm):
    bpf_path = os.path.join(test_microvm.path, "bpf.out")

    run_seccompiler_bin(bpf_path, binary_dir=test_microvm.fc_binary_path.parent)

    test_microvm.create_jailed_resource(bpf_path)
    test_microvm.jailer.extra_args.update({"seccomp-filter": "bpf.out"})
