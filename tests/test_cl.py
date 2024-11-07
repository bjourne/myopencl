# Copyright (C) 2024 Björn A. Lindqvist
from humanize import metric
from myopencl.objs import MyContext
from pathlib import Path
from time import time
from torch.nn import Conv2d
from torch.nn.functional import conv2d

import ctypes
import myopencl as cl
import numpy as np
import torch

PLAT_IDX = 1
VECADD = Path("kernels/vecadd.cl")

########################################################################
# Utils
########################################################################
def write_torch_tensor(ctx, q_name, buf_name, x):
    assert x.is_contiguous()
    assert x.is_cpu

    c_ptr = ctypes.cast(x.data_ptr(), ctypes.c_void_p)
    return ctx.create_input_buffer(q_name, buf_name, x.nbytes, c_ptr)

def read_torch_tensor(ctx, q_name, buf_name, x):
    assert x.is_contiguous()
    assert x.is_cpu

    c_ptr = ctypes.cast(x.data_ptr(), ctypes.c_void_p)
    return ctx.read_buffer(q_name, buf_name, x.nbytes, c_ptr)

def write_numpy_array(ctx, q_name, buf_name, x):
    c_ptr = np.ctypeslib.as_ctypes(x)
    return ctx.create_input_buffer(q_name, buf_name, x.nbytes, c_ptr)

def read_numpy_array(ctx, q_name, buf_name, x):
    c_ptr = np.ctypeslib.as_ctypes(x)
    return ctx.read_buffer(q_name, buf_name, x.nbytes, c_ptr)

########################################################################
# Tests: low-level
########################################################################
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

def test_get_queue_size():
    attr = cl.CommandQueueInfo.CL_QUEUE_SIZE
    for plat_id in cl.get_platform_ids():
        for dev_id in cl.get_device_ids(plat_id):
            ctx = cl.create_context(dev_id)
            q = cl.create_command_queue_with_properties(ctx, dev_id, [])
            assert cl.get_info(attr, q) == -1
            for o in [ctx, dev_id]:
                cl.release(o)

def test_run_vecadd():
    plat_id = cl.get_platform_ids()[PLAT_IDX]
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

    max_wi_sizes = cl.get_info(
        cl.DeviceInfo.CL_DEVICE_MAX_WORK_ITEM_SIZES, dev
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

def test_ooo_queue():
    prop_key = cl.CommandQueueInfo.CL_QUEUE_PROPERTIES
    prop_val = cl.CommandQueueProperties.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    props = [prop_key, prop_val]
    ctx = MyContext(PLAT_IDX, 0)
    ctx.create_queue("main", props)

    max_alloc = cl.get_info(cl.DeviceInfo.CL_DEVICE_MAX_MEM_ALLOC_SIZE, ctx.dev_id)
    n_els = min(100 * 1024 * 1024, max_alloc // 8)
    arr = np.random.uniform(size = (n_els,)).astype(np.float32)
    c_ptr = np.ctypeslib.as_ctypes(arr)

    ev1 = ctx.create_input_buffer("main", "buf1", arr.nbytes, c_ptr)
    ev2 = ctx.create_input_buffer("main", "buf2", arr.nbytes, c_ptr)

    # This doesn't really work.
    statuses = {
        cl.CommandExecutionStatus.CL_COMPLETE,
        cl.CommandExecutionStatus.CL_RUNNING,
        cl.CommandExecutionStatus.CL_SUBMITTED,
    }
    key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS
    assert (cl.get_info(key, ev1) in statuses and
            cl.get_info(key, ev2) in statuses)
    cl.wait_for_events([ev1, ev2])
    ctx.finish_and_release()

def test_torch_tensors():
    orig = torch.randn((64, 3, 3, 3, 25))
    new = torch.empty_like(orig)

    ctx = MyContext(PLAT_IDX, 0)
    ctx.create_queue("main", [])

    ev1 = write_torch_tensor(ctx, "main", "x", orig)
    ev2 = read_torch_tensor(ctx, "main", "x", new)
    cl.wait_for_events([ev1, ev2])

    assert torch.sum(new - orig) == 0.0
    ctx.finish_and_release()

########################################################################
# Tests: higher level
########################################################################
def test_objs():
    status_key = cl.EventInfo.CL_EVENT_COMMAND_EXECUTION_STATUS

    ctx = MyContext(PLAT_IDX, 0)
    ctx.create_queue("main0", [])
    ctx.create_queue("main1", [])

    n_els = 10 * 1024 * 1024
    arr1 = np.random.uniform(size = (n_els,)).astype(np.float32)
    arr2 = np.zeros(n_els, dtype = np.float32)
    c_ptr1 = np.ctypeslib.as_ctypes(arr1)
    c_ptr2 = np.ctypeslib.as_ctypes(arr2)

    ev1 = ctx.create_input_buffer("main0", "buf", arr1.nbytes, c_ptr1)
    cl.wait_for_events([ev1])
    ev2 = ctx.read_buffer("main1", "buf", arr1.nbytes, c_ptr2)
    cl.wait_for_events([ev2])

    for ev in [ev1, ev2]:
        val = cl.CommandExecutionStatus.CL_COMPLETE
        assert cl.get_info(status_key, ev) == val
    assert np.array_equal(arr1, arr2)
    ctx.finish_and_release()

def test_run_vecadd_obj():
    n_els = 16
    A = np.random.uniform(size = (n_els,)).astype(np.float32)
    B = np.random.uniform(size = (n_els,)).astype(np.float32)
    C = np.empty_like(A)

    ctx = MyContext(PLAT_IDX, 0)
    ctx.create_queue("main", [])
    write_numpy_array(ctx, "main", "A", A)
    write_numpy_array(ctx, "main", "B", B)
    ctx.create_output_buffer("C", A.nbytes)
    ctx.create_program("vecadd", VECADD)


    ctx.run_kernel("main", "vecadd", "vecadd",
                   [n_els], None,
                   ["A", "B", "C"])

    ev = read_numpy_array(ctx, "main", "C", C)
    cl.wait_for_events([ev])
    assert np.sum(A + B - C) == 0.0

    C = np.empty_like(A)
    ctx.run_kernel("main", "vecadd", "vecadd_serial",
                   [1], None,
                   [(cl.cl_uint, 1), "A", "B", "C"])

    ev = read_numpy_array(ctx, "main", "C", C)
    cl.wait_for_events([ev])
    assert np.sum(A + B - C) == 0.0

    ctx.finish_and_release()

def test_conv2d():

    ctx = MyContext(PLAT_IDX, 0)
    path = Path("kernels/conv2d.cl")
    ctx.create_program("conv2d", path)
    ctx.create_queue("main", [])

    scenarios = [
        dict(n_in = 3, n_out = 64, k_size = 3, i_h = 32, i_w = 32, padding = 1),
        dict(n_in = 3, n_out = 64, k_size = 3, i_h = 32, i_w = 32, padding = 0),
        dict(n_in = 1, n_out = 1, k_size = 3, i_h = 6, i_w = 6, padding = 0),
        dict(n_in = 64, n_out = 128, k_size = 3, i_h = 32, i_w = 32, padding = 1),
    ]
    for d in scenarios:
        n_out = d["n_out"]
        n_in = d["n_in"]
        k_size = d["k_size"]
        i_h = d["i_h"]
        i_w = d["i_w"]
        padding = d["padding"]

        filters = torch.randn(n_out, n_in, k_size, k_size)
        x = torch.randn(1, n_in, i_h, i_w)

        # Run on torch
        y_torch = conv2d(x, filters, padding = padding)

        # Run on OpenCL
        y_cl = torch.empty_like(y_torch)
        write_torch_tensor(ctx, "main", "filters", filters)
        write_torch_tensor(ctx, "main", "x", x)
        ctx.create_output_buffer("y", y_cl.nbytes)

        args = [
            (cl.cl_uint, n_out),
            (cl.cl_uint, n_in),
            (cl.cl_uint, k_size), (cl.cl_uint, k_size), "filters",
            (cl.cl_uint, i_h), (cl.cl_uint, i_w), "x",
            (cl.cl_uint, padding), "y"
        ]
        ev = ctx.run_kernel("main", "conv2d", "conv2d", [1], None, args)
        cl.wait_for_events([ev])
        ev = read_torch_tensor(ctx, "main", "y", y_cl)
        cl.wait_for_events([ev])

        diff = torch.abs(torch.sum(y_cl - y_torch))
        if diff > 0.01:
            print("== GOT ==")
            print(y_cl)
            print("== EXPECTED ==")
            print(y_torch)
        assert diff < 0.01
        ctx.release_all_buffers()
    ctx.finish_and_release()
