# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# WIP example demonstrating im2col.
from math import ceil, prod
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv
from torch.nn.functional import conv2d

import myopencl as cl
import numpy as np
import torch

def mean_rel_err(a, b):
    return np.mean(np.abs(a - b) / (np.abs(b) + np.finfo(np.float32).eps))

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
    return X, W, (n_dim, oy_dim, ox_dim, oc_dim)

def matmul_cl(a_mat, b_mat, plat_idx, path, pe_s, x_scale, v_size):
    assert a_mat.dtype == np.float32 and b_mat.dtype == np.float32

    block_n = pe_s ** 2
    block_m = x_scale * v_size
    block_k = pe_s ** 2

    size_n, size_m = a_mat.shape
    by, size_k = b_mat.shape
    assert by == size_m

    N = ceil(size_n / block_n)
    M = ceil(size_m / block_m)
    K = ceil(size_k / block_k)

    # Flaw in the SA implementation
    M = max(M, 3)

    pad_n = N * block_n
    pad_m = M * block_m
    pad_k = K * block_k

    add_n = pad_n - size_n
    add_m = pad_m - size_m
    add_k = pad_k - size_k

    a_mat_pad = np.pad(a_mat, [(0, add_n), (0, add_m)])
    b_mat_pad = np.pad(b_mat, [(0, add_m), (0, add_k)])

    kvs = [
        ("PE_S", pe_s),
        ("X_SCALE", x_scale),
        ("V_SIZE", v_size),
        ("Block dims", "%4d %4d %4d" % (block_n, block_m, block_k)),
        ("Matrix dims", "%4d %4d %4d" % (size_n, size_m, size_k)),
        ("Padded dims", "%4d %4d %4d" % (pad_n, pad_m, pad_k)),
    ]
    for k, v in kvs:
        print("%-20s: %15s" % (k, v))

    a_mat_tiled = tile_matrix(a_mat_pad, block_n, block_m)
    b_mat_t_tiled = tile_matrix(b_mat_pad.T, block_k, block_m)
    c_mat = np.empty(pad_n * pad_k, dtype = np.float32)

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
        ("store", c_mat, cl.MemFlags.CL_MEM_WRITE_ONLY)
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

    nbytes = c_mat.nbytes
    cptr = np.ctypeslib.as_ctypes(c_mat)
    ev = ctx.read_buffer("store", "store", nbytes, cptr)
    cl.wait_for_events([ev])

    c_mat = c_mat.reshape(-1, pe_s, pe_s)
    c_mat = c_mat.transpose(0, 2, 1)
    c_mat = c_mat.reshape(N, K, block_n, block_k)
    c_mat = untile_matrix(c_mat)
    c_mat = c_mat[:size_n,:size_k]

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for (name, mat, flag), ev in zip(kernel_configs, events):
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-10s: %6.4f" % (name, secs))
    ctx.finish_and_release()
    return c_mat

def conv2d_cl(X, W, padding, plat_idx, path, pe_s, x_scale, v_size):
    X, W, y_shape = im2col(X, W, padding)
    Y = matmul_cl(X, W, plat_idx, path, pe_s, x_scale, v_size)
    return Y.reshape(y_shape)

# X: n, iy, ix, ic
# W: fy, fx, ic, oc
# Y: n, oy, ox, oc
def conv2d_torch(X, W, padding):
    _, _, _, oc_dim = W.shape
    X = torch.from_numpy(X.transpose(0, 3, 1, 2))
    W = torch.from_numpy(W.transpose(3, 2, 0, 1))
    Y = conv2d(X, W, padding = padding)
    Y = Y.numpy().transpose(0, 2, 3, 1)
    return Y

def im2col_cl(
        n_dim,
        oc_dim, ic_dim,
        fy_dim, fx_dim,
        iy_dim, ix_dim,
        padding,
        plat_idx, path,
        pe_s, x_scale, v_size
):
    X0 = init_arr(
        n_dim, iy_dim, ix_dim, ic_dim, mode = "default"
    )
    W1 = init_arr(
        fy_dim, fx_dim, ic_dim, oc_dim, mode = "default"
    )
    W2 = init_arr(
        fy_dim, fx_dim, oc_dim, oc_dim, mode = "default"
    )

    X1_torch = conv2d_torch(X0, W1, padding)
    X1_cl = conv2d_cl(X0, W1, padding, plat_idx, path, pe_s, x_scale, v_size)
    print("X1: Rel. err torch, cl   : %.5f" % mean_rel_err(X1_torch, X1_cl))

    X2_torch = conv2d_torch(X1_torch, W2, padding)
    X2_cl = conv2d_cl(X1_cl, W2, padding, plat_idx, path, pe_s, x_scale, v_size)
    print("X2: Rel. err torch, cl   : %.5f" % mean_rel_err(X2_torch, X2_cl))


def main():
    plat_idx = int(argv[1])
    path = Path(argv[2])
    v_size = int(argv[3])
    pe_s = int(argv[4])
    x_scale = int(argv[5])

    im2col_cl(
        2,
        32, 3,
        3, 3,
        16, 16,
        1,
        plat_idx, path,
        pe_s, x_scale, v_size
    )

main()
