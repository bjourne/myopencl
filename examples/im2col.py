# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# Incomplete example demonstrating im2col.
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv

import myopencl as cl
import numpy as np

def format_build_opts(*args, **kw):
    defs = [f"-D {k}={v}" for (k, v) in kw.items()]
    return " ".join(list(args) + defs)

def dim_fmt(dim):
    return " ".join(f"{v:4d}" for v in dim)

def init_mat(dim):
    n = dim[0] * dim[1]
    M = np.arange(n, dtype = np.float32).reshape(*dim)
    M -= n // 2
    assert M.dtype == np.float32
    return M

def tile_matrix(M, ts_y, ts_x):
    y, x = M.shape
    assert y % ts_y == 0
    assert x % ts_x == 0
    shape = (y // ts_y, x // ts_x, ts_y, ts_x)
    strides = 4 * np.array([ts_y * x, ts_x, x, 1])
    M = as_strided(np.ascontiguousarray(M), shape, strides)
    return np.ascontiguousarray(M)

def untile_matrix(M):
    y, x, ts_y, ts_x = M.shape
    M = M.transpose(0, 2, 1, 3)
    return M.reshape(y * ts_y, x * ts_x)

def main():
    plat_idx = int(argv[1])
    path = Path(argv[2])
    v_size = int(argv[3])
    pe_s = int(argv[4])
    x_scale = int(argv[5])
    N, M, K = [int(v) for v in argv[6:9]]

    a_block = np.array([pe_s ** 2, x_scale * v_size])
    b_block = np.array([a_block[1], a_block[0]])
    c_block = np.array([a_block[0], b_block[1]])

    a_size = [N, M] * a_block
    b_size = [M, K] * b_block
    c_size = [N, K] * c_block

    kvs = [
        ("PE_S", pe_s),
        ("X_SCALE", x_scale),
        ("V_SIZE", v_size),
        ("Block A", dim_fmt(a_block)),
        ("Block B", dim_fmt(b_block)),
        ("Block C", dim_fmt(c_block)),
        ("Size A", dim_fmt(a_size)),
        ("Size B", dim_fmt(b_size)),
        ("Size C", dim_fmt(c_size)),
    ]
    for k, v in kvs:
        print("%-20s: %10s" % (k, v))

    a_mat = init_mat(a_size)
    b_mat = init_mat(b_size)
    c_mat = a_mat @ b_mat

    a_mat_tiled = tile_matrix(a_mat, a_block[0], a_block[1])
    b_mat_t_tiled = tile_matrix(b_mat.T, b_block[1], b_block[0])

    c_mat_cl = np.empty(c_size[0] * c_size[1], dtype = np.float32)

    opts = format_build_opts(
        "-cl-std=CL2.0", "-Werror",
        V_SIZE = v_size,
        PE_S = pe_s,
        X_SCALE = x_scale
    )
    ctx = Context.from_indexes(plat_idx, 0)
    ctx.register_program("matmul", path, opts)

    dim_args = [(cl.cl_uint, M), (cl.cl_uint, N), (cl.cl_uint, K)]
    kernel_configs = [
        ("loadA", a_mat_tiled, cl.MemFlags.CL_MEM_READ_ONLY),
        ("loadB", b_mat_t_tiled, cl.MemFlags.CL_MEM_READ_ONLY),
        ("store", c_mat_cl, cl.MemFlags.CL_MEM_WRITE_ONLY)
    ]

    events = []
    queue_props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]

    # Create buffers and queues and launch kernels
    for name, mat, flag in kernel_configs:
        assert mat.dtype == np.float32
        nbytes = mat.nbytes
        ctx.register_buffer(name, nbytes, flag)
        ctx.register_queue(name, queue_props)
        if flag == cl.MemFlags.CL_MEM_READ_ONLY:
            cptr = np.ctypeslib.as_ctypes(mat)
            ctx.write_buffer(name, name, nbytes, cptr)
        # Launch kernel
        ev = ctx.run_kernel(
            name, "matmul", name, [1], None, [name] + dim_args
        )
        events.append(ev)
    cl.wait_for_events(events)

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for (name, mat, flag), ev in zip(kernel_configs, events):
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-10s: %6.4f" % (name, secs))

    nbytes = c_mat_cl.nbytes
    cptr = np.ctypeslib.as_ctypes(c_mat_cl)
    ev = ctx.read_buffer("store", "store", nbytes, cptr)
    cl.wait_for_events([ev])

    c_mat_cl = c_mat_cl.reshape(-1, pe_s, pe_s)
    c_mat_cl = c_mat_cl.transpose(0, 2, 1)
    c_mat_cl = c_mat_cl.reshape(N, K, c_block[0], c_block[1])
    c_mat_cl = untile_matrix(c_mat_cl)
    print(np.max(np.abs(c_mat_cl - c_mat)))

    ctx.finish_and_release()

main()
