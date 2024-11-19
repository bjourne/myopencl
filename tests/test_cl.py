# Copyright (C) 2024 Björn A. Lindqvist
#
# Names:
#   * bname - buffer name
#   * c - MyContext
#   * cptr - C pointer
#   * ev - event
#   * ksize - kernel size
#   * nbytes - number of bytes
#   * qname - queue name
#   * x - tensor or numpy array
from humanize import metric
from myopencl.objs import MyContext
from myopencl.utils import can_compile, is_gpu, platform_device_pairs
from pathlib import Path
from pytest import mark
from time import time

import ctypes
import myopencl as cl
import numpy as np

VECADD = Path("kernels/vecadd.cl")
BUILD_OPTS = "-cl-std=CL2.0 -cl-unsafe-math-optimizations"

PAIRS = platform_device_pairs()

########################################################################
# Utils
########################################################################<
def write_numpy_array(c, qname, bname, x):
    cptr = np.ctypeslib.as_ctypes(x)
    return c.register_input_buffer(qname, bname, x.nbytes, cptr)

def read_numpy_array(c, qname, bname, x):
    cptr = np.ctypeslib.as_ctypes(x)
    return c.read_buffer(qname, bname, x.nbytes, cptr)

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
def test_release_device(platform_id, device_id):
    dev_id = cl.get_device_ids(platform_id)[0]
    cl.release(dev_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_get_queue_size(platform_id, device_id):
    ctx = cl.create_context(device_id)
    q = cl.create_command_queue_with_properties(ctx, device_id, [])
    attr = cl.CommandQueueInfo.CL_QUEUE_SIZE
    assert cl.get_info(attr, q) == 0
    cl.release(ctx)
    cl.release(device_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_vecadd(platform_id, device_id):
    if not can_compile(device_id):
        return

    ctx = cl.create_context(device_id)
    queue = cl.create_command_queue_with_properties(ctx, device_id, [])

    el_tp = ctypes.c_float
    n_els = 10 * 1000 * 1024
    el_size = ctypes.sizeof(el_tp)
    nbytes = n_els * el_size
    mem_a = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_READ_ONLY, nbytes)
    mem_b = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_READ_ONLY, nbytes)
    mem_c = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_WRITE_ONLY, nbytes)

    ev1 = cl.enqueue_fill_buffer(queue, mem_a, el_tp(1.5), 0, nbytes)
    ev2 = cl.enqueue_fill_buffer(queue, mem_b, el_tp(2.5), 0, nbytes)
    cl.wait_for_events([ev1, ev2])

    source = VECADD.read_text("utf-8")
    prog = cl.create_program_with_source(ctx, source)
    cl.build_program(prog, device_id, "-Werror -cl-std=CL2.0", True, True)
    kern = cl.create_kernel(prog, "vecadd")

    for i, buf in enumerate([mem_a, mem_b, mem_c]):
        cl.set_kernel_arg(kern, i, buf)

    # The value is stupid on some platforms.
    max_wi_size = cl.get_info(
        cl.DeviceInfo.CL_DEVICE_MAX_WORK_ITEM_SIZES, device_id
    )[0]
    max_wi_size = min(max_wi_size, 8192)

    n_reps = 50

    st = time()
    for _ in range(n_reps):
        ev = cl.enqueue_nd_range_kernel(queue, kern, [n_els], [max_wi_size])
        cl.wait_for_events([ev])
    el_per_s = n_reps * n_els / (time() - st)
    el_per_s = metric(el_per_s)
    print(f"{el_per_s} adds/s ", end = " ")

    c = np.zeros(n_els, dtype = np.float32)
    ptr = np.ctypeslib.as_ctypes(c)
    ev = cl.enqueue_read_buffer(queue, mem_c, False, 0, nbytes, ptr)
    cl.wait_for_events([ev])
    assert np.sum(c) == n_els * 4

    cl.flush(queue)
    cl.finish(queue)
    objs = [
        mem_a, mem_b, mem_c,
        prog, kern,
        queue, ctx, device_id
    ]
    for obj in objs:
        cl.release(obj)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_ooo_queue(platform_id, device_id):
    ctx = MyContext(platform_id, device_id)

    prop_key = cl.CommandQueueInfo.CL_QUEUE_PROPERTIES
    prop_val = cl.CommandQueueProperties.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    props = [prop_key, prop_val]
    ctx.register_queue("main", props)

    attr = cl.DeviceInfo.CL_DEVICE_MAX_MEM_ALLOC_SIZE
    max_alloc = cl.get_info(attr, ctx.device_id)
    n_els = min(20 * 1024 * 1024, max_alloc // 8)
    arr = np.random.uniform(size = (n_els,)).astype(np.float32)
    cptr = np.ctypeslib.as_ctypes(arr)

    ev1 = ctx.register_input_buffer("main", "buf1", arr.nbytes, cptr)
    ev2 = ctx.register_input_buffer("main", "buf2", arr.nbytes, cptr)

    # This doesn't really work.
    statuses = {
        cl.CommandExecutionStatus.CL_COMPLETE,
        cl.CommandExecutionStatus.CL_QUEUED,
        cl.CommandExecutionStatus.CL_RUNNING,
        cl.CommandExecutionStatus.CL_SUBMITTED,
    }
    key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS

    st1 = cl.get_info(key, ev1)
    st2 = cl.get_info(key, ev2)
    assert st1 in statuses and st2 in statuses

    cl.wait_for_events([ev1, ev2])
    ctx.finish_and_release()

########################################################################
# Tests: higher level
########################################################################
@mark.parametrize("platform_id,device_id", PAIRS)
def test_objs(platform_id, device_id):
    status_key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS

    c = MyContext(platform_id, device_id)
    c.register_queue("main0", [])
    c.register_queue("main1", [])

    n_els = 10 * 1024 * 1024
    arr1 = np.random.uniform(size = (n_els,)).astype(np.float32)
    arr2 = np.zeros(n_els, dtype = np.float32)
    cptr1 = np.ctypeslib.as_ctypes(arr1)
    cptr2 = np.ctypeslib.as_ctypes(arr2)

    ev1 = c.register_input_buffer("main0", "buf", arr1.nbytes, cptr1)
    cl.wait_for_events([ev1])
    ev2 = c.read_buffer("main1", "buf", arr1.nbytes, cptr2)
    cl.wait_for_events([ev2])

    for ev in [ev1, ev2]:
        val = cl.CommandExecutionStatus.CL_COMPLETE
        assert cl.get_info(status_key, ev) == val
    assert np.array_equal(arr1, arr2)
    c.finish_and_release()

@mark.parametrize("platform_id,device_id", PAIRS)
def test_vecadd_objs(platform_id, device_id):
    n_els = 16
    A = np.random.uniform(size = (n_els,)).astype(np.float32)
    B = np.random.uniform(size = (n_els,)).astype(np.float32)
    C = np.empty_like(A)

    c = MyContext(platform_id, device_id)
    if not can_compile(c.device_id):
        c.finish_and_release()
        return

    c.register_queue("main", [])
    write_numpy_array(c, "main", "A", A)
    write_numpy_array(c, "main", "B", B)
    c.register_output_buffer("C", A.nbytes)
    c.register_program("vecadd", VECADD, BUILD_OPTS)


    c.run_kernel("main", "vecadd", "vecadd",
                 [n_els], None,
                 ["A", "B", "C"])

    ev = read_numpy_array(c, "main", "C", C)
    cl.wait_for_events([ev])
    assert np.sum(A + B - C) == 0.0

    C = np.empty_like(A)
    c.run_kernel("main", "vecadd", "vecadd_serial",
                 [1], None,
                 [(cl.cl_uint, 1), "A", "B", "C"])

    ev = read_numpy_array(c, "main", "C", C)
    cl.wait_for_events([ev])
    assert np.sum(A + B - C) == 0.0

    c.finish_and_release()

@mark.parametrize("platform_id,device_id", PAIRS)
def test_print_objs(platform_id, device_id):
    c = MyContext(platform_id, device_id)
    c.print()
    c.finish_and_release()


def test_from_indexes():
    for i, platform_id in enumerate(cl.get_platform_ids()):
        for j, _ in enumerate(cl.get_device_ids(platform_id)):
            c = MyContext.from_indexes(i, j)
            c.finish_and_release()
