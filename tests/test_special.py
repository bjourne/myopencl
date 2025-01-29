# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# Test for bugs, crashes, and stuff like that.
from myopencl.utils import platform_device_pairs
from pathlib import Path
from pytest import mark

import myopencl as cl

PAIRS = platform_device_pairs()

def get_device_type(device_id):
    attr = cl.DeviceInfo.CL_DEVICE_TYPE
    return cl.get_info(attr, device_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_autorun(platform_id, device_id):
    # Crashes on FPGA devices
    if get_device_type(device_id) == cl.DeviceType.CL_DEVICE_TYPE_ACCELERATOR:
        return
    ctx = cl.create_context(device_id)

    source = Path("kernels/autorun.cl").read_text("utf-8")
    prog = cl.create_program_with_source(ctx, source)
    queue = cl.create_command_queue_with_properties(ctx, device_id, [])

    cl.build_program(prog, device_id, "", True, True)

    kernel = cl.create_kernel(prog, "entry")
    ev = cl.enqueue_nd_range_kernel(queue, kernel, [1], None)
    cl.wait_for_events([ev])

    for o in [kernel, ev, queue, prog, ctx, device_id]:
        cl.release(o)
