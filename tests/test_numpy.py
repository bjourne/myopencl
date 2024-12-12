# Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
from humanize import metric
from myopencl.objs import MyContext
from myopencl.utils import can_compile, platform_device_pairs
from numpy.lib.stride_tricks import as_strided
from numpy.linalg import norm
from pathlib import Path
from pytest import mark
from time import time

import ctypes
import myopencl as cl
import numpy as np

PAIRS = platform_device_pairs()
VECADD = Path("kernels/vecadd.cl")
BUILD_OPTS = "-cl-std=CL2.0 -cl-unsafe-math-optimizations"

def align_matrix(M, y_align, x_align):
    y, x = M.shape
    y_rem = y % y_align
    y_add = 0 if y_rem == 0 else y_align - y_rem
    x_rem = x % x_align
    x_add = 0 if x_rem == 0 else x_align - x_rem

    M2 = np.pad(M, [(0, y_add), (0, x_add)])
    return M2

def tile_matrix(M, ts_y, ts_x):
    y, x = M.shape
    assert y % ts_y == 0
    assert x % ts_x == 0
    shape = (y // ts_y, x // ts_x, ts_y, ts_x)
    strides = 4 * np.array([ts_y * x, ts_x, x, 1])
    M = as_strided(M, shape, strides)
    return np.ascontiguousarray(M)

# GPT wrote this
def untile_matrix(M):
    y, x, ts_y, ts_x = M.shape
    M = M.transpose(0, 2, 1, 3)
    return M.reshape(y * ts_y, x * ts_x)

def write_numpy_array(c, qname, bname, x):
    cptr = np.ctypeslib.as_ctypes(x)
    return c.register_input_buffer(qname, bname, x.nbytes, cptr)

def read_numpy_array(c, qname, bname, x):
    cptr = np.ctypeslib.as_ctypes(x)
    return c.read_buffer(qname, bname, x.nbytes, cptr)

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
def test_matmul(platform_id, device_id):
    ctx = MyContext(platform_id, device_id)
    if not can_compile(ctx.device_id):
        ctx.finish_and_release()
        return

    n, m, k = 2048, 2048, 2048
    ts_n, ts_m, ts_k, ts = 512, 256, 16, 16
    assert n % ts_n == 0 and m % ts_m == 0 and k % ts_k == 0

    path = Path("kernels/matmul.cl")
    opts = [
        "-cl-std=CL2.0",
        "-cl-unsafe-math-optimizations",
        "-D TS_N=%d" % ts_n,
        "-D TS_M=%d" % ts_m,
        "-D TS_K=%d" % ts_k,
        "-D TS=%d" % ts
    ]
    ctx.register_program("matmul", path, " ".join(opts))
    ctx.register_queue("main", [])

    # Setup data
    A = np.random.uniform(size = (n, m)).astype(np.float32) - 0.5
    B = np.random.uniform(size = (m, k)).astype(np.float32) - 0.5
    C_tiled = np.empty((n // ts_n, k // ts_k, ts_n, ts_k), dtype = np.float32)
    A_tiled = tile_matrix(A, ts_n, ts_m)
    B_tiled = tile_matrix(B, ts_m, ts_k)

    write_numpy_array(ctx, "main", "A", A_tiled)
    write_numpy_array(ctx, "main", "B", B_tiled)
    ctx.register_output_buffer("C", C_tiled.nbytes)

    bef = time()
    ctx.run_kernel(
        "main", "matmul", "matmul_tiled_tiled_sd",
        [1], None, [
            (cl.cl_uint, n),
            (cl.cl_uint, m),
            (cl.cl_uint, k),
            "A", "B", "C"
        ]
    )
    ev = read_numpy_array(ctx, "main", "C", C_tiled)
    cl.wait_for_events([ev])

    print('%4d/%4d/%4d: %.2f' % (n, m, k, time() - bef))
    C = untile_matrix(C_tiled)
    assert np.linalg.norm(C - A @ B) < 0.01
    ctx.finish_and_release()

@mark.parametrize("platform_id,device_id", PAIRS)
def test_im2col(platform_id, device_id):
    ctx = MyContext(platform_id, device_id)
    if not can_compile(ctx.device_id):
        ctx.finish_and_release()
        return

    # Input dimensions
    n, sy, sx, sc = 128, 6, 6, 3

    # Filter dimensions
    fy, fx, dc = 3, 3, 3

    # Padding and strides
    pad_y, pad_x = 1, 1
    stride_y, stride_x = 1, 1

    # Padded size
    py = sy + 2 * pad_y
    px = sx + 2 * pad_x

    # Destination size
    dy = (py - fy) // stride_y + 1
    dx = (px - fx) // stride_x + 1

    # Generate data
    W = np.arange(fy * fx * sc * dc, dtype = np.float32).reshape(fy * fx * sc, dc)
    X = np.arange(n * sy * sx * sc, dtype = np.float32).reshape(n, sy, sx, sc)
    X = np.pad(X, [(0, 0), (pad_y, pad_y), (pad_x, pad_x), (0, 0)])

    # Im2col
    shape = n, dy, dx, fy, fx, sc
    strides = 4 * np.array([
        py * px * sc,
        stride_y * px * sc,
        stride_x * sc,
        px * sc,
        sc,
        1
    ])
    X = as_strided(X, shape, strides).reshape(n * dy * dx, fy * fx * sc)

    # Compute expected
    Y_exp = X @ W

    # Tile and compute using OpenCL
    ts_n, ts_m, ts_k, ts = 64, 16, 16, 16
    opts = [
        "-cl-std=CL2.0",
        "-cl-unsafe-math-optimizations",
        "-D TS_N=%d" % ts_n,
        "-D TS_M=%d" % ts_m,
        "-D TS_K=%d" % ts_k,
        "-D TS=%d" % ts
    ]
    X = align_matrix(X, ts_n, ts_m)
    W = align_matrix(W, ts_m, ts_k)

    ctx.register_queue("main", [])
    write_numpy_array(ctx, "main", "X", X)
    write_numpy_array(ctx, "main", "W", W)

    n, m = X.shape
    _, k = W.shape

    args = [
        (cl.cl_uint, n), (cl.cl_uint, m), (cl.cl_uint, k),
        "X", "W", "Y"
    ]

    # Run multiple different matmul variants
    matmuls = [
        ("matmul_naive_sd", [1], None),
        ("matmul_naive_nd", [n, k], None),
        ("matmul_tiled_sd", [1], None),
        ("matmul_tiled_nd", [n, k], [ts, ts])
    ]

    Y = np.empty((n, k), dtype = np.float32)
    ctx.register_output_buffer("Y", Y.nbytes)
    path = Path("kernels/matmul.cl")
    ctx.register_program("matmul", path, " ".join(opts))
    for kname, gwork, lwork in matmuls:
        bef = time()
        ctx.run_kernel("main", "matmul", kname, gwork, lwork, args)
        ev = read_numpy_array(ctx, "main", "Y", Y)
        cl.wait_for_events([ev])
        Y_out = Y[:Y_exp.shape[0],:Y_exp.shape[1]]
        err = norm(Y_out - Y_exp)
        assert err < 0.01
        assert np.array_equal(Y_out, Y_exp)
    ctx.finish_and_release()
