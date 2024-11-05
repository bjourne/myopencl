# Copyright (C) 2024 Björn A. Lindqvist
from humanize import metric
from myopencl.objs import MyContext
from pathlib import Path
from time import time

import ctypes
import myopencl as cl
import numpy as np

PLAT_IDX = 0
VECADD = Path("kernels/vecadd.cl")

def test_release_none():
    try:
        cl.release(None)
        assert False
    except KeyError:
        assert True

def test_release_device():
    plat_id = cl.get_platform_ids()[0]
    dev_id = cl.get_device_ids(plat_id)[0]
    cl.release(dev_id)

def test_run_vecadd():
    plat_id = cl.get_platform_ids()[0]
    dev = cl.get_device_ids(plat_id)[0]

    ctx = cl.create_context(dev)
    queue = cl.create_command_queue_with_properties(ctx, dev, [])

    el_tp = ctypes.c_float
    n_els = 10 * 1000 * 1024
    el_size = ctypes.sizeof(el_tp)
    n_bytes = n_els * el_size
    mem_a = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_READ_ONLY, n_bytes)
    mem_b = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_READ_ONLY, n_bytes)
    mem_c = cl.create_buffer(ctx, cl.MemFlags.CL_MEM_WRITE_ONLY, n_bytes)

    ev1 = cl.enqueue_fill_buffer(queue, mem_a, el_tp(1.5), 0, n_bytes)
    ev2 = cl.enqueue_fill_buffer(queue, mem_b, el_tp(2.5), 0, n_bytes)
    cl.wait_for_events([ev1, ev2])

    source = VECADD.read_text("utf-8")
    prog = cl.create_program_with_source(ctx, source)
    cl.build_program(prog, dev, "-Werror -cl-std=CL2.0", True, True)
    kern = cl.create_kernel(prog, "vecadd")

    cl.set_kernel_arg(kern, 0, mem_a)
    cl.set_kernel_arg(kern, 1, mem_b)
    cl.set_kernel_arg(kern, 2, mem_c)

    max_wi_sizes = cl.get_device_info(
        dev, cl.DeviceInfo.CL_DEVICE_MAX_WORK_ITEM_SIZES
    )

    n_reps = 50

    st = time()
    for _ in range(n_reps):
        ev = cl.enqueue_nd_range_kernel(queue, kern, [n_els], [max_wi_sizes[0]])
        cl.wait_for_events([ev])
    el_per_s = n_reps * n_els / (time() - st)
    el_per_s = metric(el_per_s)
    print(f"{el_per_s} adds/s ", end = " ")

    c = np.zeros(n_els, dtype = np.float32)
    ptr = np.ctypeslib.as_ctypes(c)
    ev = cl.enqueue_read_buffer(queue, mem_c, False, 0, n_bytes, ptr)
    cl.wait_for_events([ev])
    assert np.sum(c) == n_els * 4

    cl.flush(queue)
    cl.finish(queue)
    objs = [
        mem_a, mem_b, mem_c,
        prog, kern,
        queue, ctx, dev
    ]
    for obj in objs:
        cl.release(obj)


def test_objs():
    status_key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS

    ctx = MyContext(PLAT_IDX, 0)
    ctx.add_queue("main0", [])
    ctx.add_queue("main1", [])

    n_els = 10 * 1024 * 1024
    arr1 = np.random.uniform(size = (n_els,)).astype(np.float32)
    arr2 = np.zeros(n_els, dtype = np.float32)
    c_ptr1 = np.ctypeslib.as_ctypes(arr1)
    c_ptr2 = np.ctypeslib.as_ctypes(arr2)

    ev1 = ctx.add_input_buffer("main0", "buf", arr1.nbytes, c_ptr1)
    cl.wait_for_events([ev1])
    ev2 = ctx.read_buffer("main1", "buf", arr1.nbytes, c_ptr2)
    cl.wait_for_events([ev2])

    for ev in [ev1, ev2]:
        val = cl.CommandExecutionStatus.CL_COMPLETE
        assert cl.get_event_info(ev, status_key) == val
    assert np.array_equal(arr1, arr2)
    ctx.finish_and_release()

def test_ooo_queue():
    prop_key = cl.CommandQueueInfo.CL_QUEUE_PROPERTIES
    prop_val = cl.CommandQueueProperties.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    props = [prop_key, prop_val]
    ctx = MyContext(PLAT_IDX, 0)
    ctx.add_queue("main", props)

    n_els = 10 * 1024 * 1024
    arr = np.random.uniform(size = (n_els,)).astype(np.float32)
    c_ptr = np.ctypeslib.as_ctypes(arr)

    ev1 = ctx.add_input_buffer("main", "buf1", arr.nbytes, c_ptr)
    ev2 = ctx.add_input_buffer("main", "buf2", arr.nbytes, c_ptr)

    # Since the queue is out of order both events should be running.
    statuses = {
        cl.CommandExecutionStatus.CL_SUBMITTED,
        cl.CommandExecutionStatus.CL_RUNNING
    }
    key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS
    assert (cl.get_event_info(ev1, key) in statuses and
            cl.get_event_info(ev2, key) in statuses)
    cl.wait_for_events([ev1, ev2])
    ctx.finish_and_release()
