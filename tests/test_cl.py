# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# Names:
#   * bname - buffer name
#   * c - Context
#   * ptr - C pointer
#   * ev - event
#   * ksize - kernel size
#   * nbytes - number of bytes
#   * qname - queue name
#   * x - tensor or numpy array

from myopencl.objs import Context
from myopencl.utils import platform_device_pairs
from pytest import mark

import ctypes
import myopencl as cl

PAIRS = platform_device_pairs()

########################################################################
# Tests: low-level
########################################################################
def test_release_none():
    try:
        cl.release(None)
        assert False
    except KeyError:
        assert True

@mark.parametrize("platform_id, device_id", PAIRS)
def test_get_details(platform_id, device_id):
    assert len(cl.get_details(platform_id)) > 0

@mark.parametrize("platform_id, device_id", PAIRS)
def test_release_device(platform_id, device_id):
    dev_id = cl.get_device_ids(platform_id)[0]
    cl.release(dev_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_get_queue_size(platform_id, device_id):
    ctx = cl.create_context(device_id)
    q = cl.create_command_queue_with_properties(ctx, device_id, [])
    attr = cl.CommandQueueInfo.CL_QUEUE_SIZE
    assert cl.get_info(attr, q) == 0
    cl.release(q)
    cl.release(ctx)
    cl.release(device_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_get_profiling_info(platform_id, device_id):
    ctx = cl.create_context(device_id)

    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    queue = cl.create_command_queue_with_properties(ctx, device_id, props)
    attr = cl.CommandQueueInfo.CL_QUEUE_PROPERTIES
    key = cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    assert cl.get_info(attr, queue) == key

    mem = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_READ_ONLY, 1024)

    ev = cl.enqueue_fill_buffer(queue, mem, ctypes.c_float(17), 0, 1024)
    cl.wait_for_events([ev])

    start = cl.get_info(cl.ProfilingInfo.CL_PROFILING_COMMAND_START, ev)
    end = cl.get_info(cl.ProfilingInfo.CL_PROFILING_COMMAND_END, ev)
    assert end > start

    for obj in [mem, queue, ctx, device_id]:
        cl.release(obj)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_event_err(platform_id, device_id):
    try:
        cl.wait_for_events([None])
    except cl.OpenCLError as e:
        assert e.code == cl.ErrorCode.CL_INVALID_EVENT

@mark.parametrize("platform_id, device_id", PAIRS)
def test_invalid_binary(platform_id, device_id):
    ctx = cl.create_context(device_id)
    b = bytes([1, 2, 3, 4])
    try:
        cl.create_program_with_binary(ctx, device_id, b)
    except cl.OpenCLError as e:
        assert e.code == cl.ErrorCode.CL_INVALID_BINARY

    cl.release(ctx)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_create_program_with_source(platform_id, device_id):
    ctx = cl.create_context(device_id)
    src1 = """
    __kernel void kern1() { }
    """
    src2 = """
    __kernel void kern2() {}
    """
    prog = cl.create_program_with_source(ctx, [src1, src2])
    cl.build_program(prog, device_id, "", True, True)
    cl.release(prog)
    cl.release(ctx)


########################################################################
# Tests: higher level
########################################################################
@mark.parametrize("platform_id,device_id", PAIRS)
def test_print_objs(platform_id, device_id):
    c = Context(platform_id, device_id)
    c.print()
    c.finish_and_release()


def test_from_indexes():
    for i, platform_id in enumerate(cl.get_platform_ids()):
        for j, _ in enumerate(cl.get_device_ids(platform_id)):
            c = Context.from_indexes(i, j)
            c.finish_and_release()
