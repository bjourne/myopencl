# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# Incomplete example demonstrating im2col.
from math import prod
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv
from torch.nn.functional import conv2d, pad

import myopencl as cl
import numpy as np
import torch

def format_build_opts(*args, **kw):
    defs = [f"-D {k}={v}" for (k, v) in kw.items()]
    return " ".join(list(args) + defs)

def dim_fmt(dim):
    return " ".join(f"{v:4d}" for v in dim)

def init_arr(*dims, mode = "default"):
    n = prod(dims)
    if mode == "default":
        M = np.arange(n, dtype = np.float32)
    else:
        M = np.ones(n, dtype = np.float32)
    M = M.reshape(*dims)
    if mode == "default":
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

def matmul_ocl(a_mat, b_mat, plat_idx, path, pe_s, x_scale, v_size):
    assert a_mat.dtype == np.float32 and b_mat.dtype == np.float32

    a_block = pe_s ** 2, x_scale * v_size
    b_block = a_block[1], a_block[0]
    c_block = a_block[0], b_block[1]

    a_size = a_mat.shape
    b_size = b_mat.shape
    c_size = a_size[0], b_size[1]

    N = a_size[0] // a_block[0]
    M = a_size[1] // a_block[1]
    K = b_size[1] // b_block[1]

    assert (N * a_block[0], M * a_block[1]) == a_size
    assert (M * b_block[0], K * b_block[1]) == b_size
    assert (N * c_block[0], K * c_block[1]) == c_size

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


    a_mat_tiled = tile_matrix(a_mat, a_block[0], a_block[1])
    b_mat_t_tiled = tile_matrix(b_mat.T, b_block[1], b_block[0])
    c_mat_cl = np.empty(c_size[0] * c_size[1], dtype = np.float32)

    opts = format_build_opts(
        "-cl-std=CL2.0", "-Werror",
        PE_S = pe_s,
        X_SCALE = x_scale,
        V_SIZE = v_size
    )
    ctx = Context.from_indexes(plat_idx, 0)
    ctx.register_program("matmul", path, opts)


    dim_args = [(cl.cl_uint, M), (cl.cl_uint, N), (cl.cl_uint, K)]
    kernel_configs = [
        ("loadA", a_mat_tiled, cl.MemFlags.CL_MEM_READ_ONLY),
        ("loadB", b_mat_t_tiled, cl.MemFlags.CL_MEM_READ_ONLY),
        ("store", c_mat_cl, cl.MemFlags.CL_MEM_WRITE_ONLY)
    ]

    # Create buffers and queues and launch kernels
    events = []
    queue_props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
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

    nbytes = c_mat_cl.nbytes
    cptr = np.ctypeslib.as_ctypes(c_mat_cl)
    ev = ctx.read_buffer("store", "store", nbytes, cptr)
    cl.wait_for_events([ev])

    c_mat_cl = c_mat_cl.reshape(-1, pe_s, pe_s)
    c_mat_cl = c_mat_cl.transpose(0, 2, 1)
    c_mat_cl = c_mat_cl.reshape(N, K, c_block[0], c_block[1])
    c_mat_cl = untile_matrix(c_mat_cl)

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for (name, mat, flag), ev in zip(kernel_configs, events):
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-10s: %6.4f" % (name, secs))
    ctx.finish_and_release()
    return c_mat_cl

def mean_rel_err(a, b):
    return np.mean(np.abs(a - b) / (np.abs(b) + 0.0001))

def im2col(X, W, padding):
    n_dim, iy_dim, ix_dim, ic_dim = X.shape
    fy_dim, fx_dim, ic_dim, oc_dim = W.shape

    X = np.pad(X, [(0, 0), (padding, padding), (padding, padding), (0, 0)])

    # Padded size
    py_dim = iy_dim + 2 * padding
    px_dim = ix_dim + 2 * padding

    # Destination size
    oy_dim = py_dim - fy_dim + 1
    ox_dim = px_dim - fx_dim + 1

    # The stride trick
    shape = n_dim, oy_dim, ox_dim, fy_dim, fx_dim, ic_dim
    strides = 4 * np.array([
        py_dim * px_dim * ic_dim,
        px_dim * ic_dim,
        ic_dim,
        px_dim * ic_dim,
        ic_dim,
        1
    ])
    X = as_strided(X, shape, strides).reshape(
        n_dim * oy_dim * ox_dim,
        fy_dim * fx_dim * ic_dim
    )
    X = X.reshape(n_dim * oy_dim * ox_dim, fy_dim * fx_dim * ic_dim)

    W = W.reshape(fy_dim * fx_dim * ic_dim, oc_dim)

    return X, W


def im2col_ocl(
        n_dim,
        oc_dim, ic_dim,
        fy_dim, fx_dim,
        iy_dim, ix_dim,
        padding,
        plat_idx, path,
        pe_s, x_scale, v_size
):
    X = init_arr(
        n_dim, iy_dim, ix_dim, ic_dim, mode = "default"
    )
    W = init_arr(
        fy_dim, fx_dim, ic_dim, oc_dim, mode = "default"
    )

    Xp, Wp = im2col(X, W, padding)

    X_torch = torch.from_numpy(X.transpose(0, 3, 1, 2))
    W_torch = torch.from_numpy(W.transpose(3, 2, 0, 1))
    Y_torch = conv2d(X_torch, W_torch, padding = padding)
    Y_torch = Y_torch.numpy().transpose(0, 2, 3, 1)
    Y_torch = Y_torch.reshape(-1, oc_dim)
    Y = Xp @ Wp
    Y_cl = matmul_ocl(Xp, Wp, plat_idx, path, pe_s, x_scale, v_size)
    print("Rel. err Y, Y_cl   : %.2f" % mean_rel_err(Y, Y_cl))
    print("Rel. err Y, Y_torch: %.2f" % mean_rel_err(Y, Y_torch))


def main():
    plat_idx = int(argv[1])
    path = Path(argv[2])
    v_size = int(argv[3])
    pe_s = int(argv[4])
    x_scale = int(argv[5])

    im2col_ocl(
        2,
        256, 64,
        3, 3,
        16, 16,
        1,
        plat_idx, path,
        pe_s, x_scale, v_size
    )

    # N, M, K = [int(v) for v in argv[6:9]]
    # a_mat = init_arr(N, M)
    # b_mat = init_arr(M, K)
    # c_mat = a_mat @ b_mat

    # c_mat_cl = matmul_ocl(
    #     a_mat, b_mat,
    #     plat_idx, path,
    #     pe_s, x_scale, v_size
    # )
    # print(mean_rel_err(c_mat, c_mat_cl))


main()
