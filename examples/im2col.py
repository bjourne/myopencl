# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# WIP example demonstrating im2col.
from math import ceil, prod
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv

from torch import from_numpy
from torch.nn.functional import batch_norm, conv2d

import myopencl as cl
import numpy as np
import torch

np.set_printoptions(precision=8,linewidth=200,threshold=5000,suppress=True)

EPS = np.finfo(np.float32).eps

# Passed on command line
PLAT_IDX = None
PATH = None
V_SIZE = None
PE_S = None
X_SCALE = None

def mean_rel_err(a, b):
    return np.mean(np.abs(a - b) / (np.abs(b) + EPS))

def format_build_opts(*args, **kw):
    defs = [f"-D {k}={v}" for (k, v) in kw.items()]
    return " ".join(list(args) + defs)

def dim_fmt(dim):
    return " ".join(f"{v:4d}" for v in dim)

def init_arr(*dims, mode = "default"):
    n = prod(dims)
    if mode == "default":
        M = np.arange(n, dtype = np.float32)
    elif mode == "ones":
        M = np.ones(n, dtype = np.float32)
    elif mode == "random":
        M = np.random.randn(n).astype(np.float32)
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
    M = M.reshape(y * ts_y, x * ts_x)
    return M

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

def cl_batch_norm_2d(x, mean, var, weights, bias):
    n_dim, y_dim, x_dim, c_dim = x.shape
    opts = format_build_opts(
        "-cl-std=CL2.0", "-Werror",
        PE_S = PE_S,
        X_SCALE = X_SCALE,
        V_SIZE = V_SIZE
    )
    ctx = Context.from_indexes(PLAT_IDX, 0)
    ctx.register_program("main", PATH, opts)
    ctx.register_queue("main", [])
    ro_buffers = [
        ("x", x),
        ("mean", mean),
        ("var", var),
        ("weights", weights),
        ("bias", bias)
    ]
    for name, arr in ro_buffers:
        flag = cl.MemFlags.CL_MEM_READ_ONLY
        ctx.register_buffer(name, arr.nbytes, flag)
        assert arr.dtype == np.float32
        ptr = np.ctypeslib.as_ctypes(arr)
        ctx.write_buffer("main", name, arr.nbytes, ptr)

    flag = cl.MemFlags.CL_MEM_WRITE_ONLY
    arr_y = np.empty((n_dim, y_dim, x_dim, c_dim), dtype = np.float32)
    ctx.register_buffer("y", arr_y.nbytes, flag)

    ctx.run_kernel(
        "main",
        "main",
        "batch_norm_2d",
        [1], None, [
            "x", "y", "mean", "var", "weights", "bias",
            (cl.cl_uint, n_dim),
            (cl.cl_uint, y_dim),
            (cl.cl_uint, x_dim),
            (cl.cl_uint, c_dim)
        ]
    )
    nbytes = arr_y.nbytes
    ptr = np.ctypeslib.as_ctypes(arr_y)
    ev = ctx.read_buffer("main", "y", nbytes, ptr)
    cl.wait_for_events([ev])
    ctx.finish_and_release()

    return arr_y

def cl_matmul(a_mat, b_mat):
    assert a_mat.dtype == np.float32 and b_mat.dtype == np.float32

    block_n = PE_S ** 2
    block_m = X_SCALE * V_SIZE
    block_k = PE_S ** 2

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
        ("PE_S", PE_S),
        ("X_SCALE", X_SCALE),
        ("V_SIZE", V_SIZE),
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
        PE_S = PE_S,
        X_SCALE = X_SCALE,
        V_SIZE = V_SIZE
    )
    ctx = Context.from_indexes(PLAT_IDX, 0)
    ctx.register_program("main", PATH, opts)

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
            name, "main", name, [1], None, [name] + dim_args
        )
        events.append(ev)
    cl.wait_for_events(events)

    nbytes = c_mat.nbytes
    cptr = np.ctypeslib.as_ctypes(c_mat)
    ev = ctx.read_buffer("store", "store", nbytes, cptr)
    cl.wait_for_events([ev])

    c_mat = c_mat.reshape(-1, PE_S, PE_S)
    c_mat = c_mat.transpose(0, 2, 1)
    c_mat = c_mat.reshape(N, K, block_n, block_k)
    c_mat = untile_matrix(c_mat)
    c_mat = c_mat[:size_n,:size_k]
    c_mat = np.ascontiguousarray(c_mat)

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for (name, mat, flag), ev in zip(kernel_configs, events):
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-10s: %6.4f" % (name, secs))
    ctx.finish_and_release()
    return c_mat

def cl_conv2d(X, W, padding):
    X, W, y_shape = im2col(X, W, padding)
    Y = cl_matmul(X, W)
    return Y.reshape(y_shape)

def torch_conv2d(x, w, padding):
    x = from_numpy(x.transpose(0, 3, 1, 2))
    w = from_numpy(w.transpose(3, 2, 0, 1))
    y = conv2d(x, w, padding = padding)
    y = y.numpy().transpose(0, 2, 3, 1)
    return y

def torch_batch_norm_2d(x, mean, var, weights, bias):
    x = from_numpy(x.transpose(0, 3, 1, 2))
    mean = from_numpy(mean)
    var = from_numpy(var)
    weights = from_numpy(weights)
    bias = from_numpy(bias)
    y = batch_norm(x, mean, var, weights, bias)
    y = y.numpy().transpose(0, 2, 3, 1)
    return y

def init(layers):
    for layer in layers:
        tp = layer["type"]
        params = layer["params"]
        if tp == "conv2d":
            ic_dim, oc_dim, k_size, padding = params
            weights = init_arr(k_size, k_size, ic_dim, oc_dim, mode = "random")
            yield weights, padding
        elif tp == "batch_norm_2d":
            c_dim, = params
            mean = np.random.randn(c_dim).astype(np.float32)
            var = np.random.rand(c_dim).astype(np.float32)
            weights = np.random.rand(c_dim).astype(np.float32)
            bias = np.random.randn(c_dim).astype(np.float32)
            yield mean, var, weights, bias
        else:
            assert False

def process(x, layers, data, backend):
    for l, d in zip(layers, data):
        tp = l["type"]
        fun = backend[tp]
        x = fun(x, *d)
    return x

def im2col_cl(
    n_dim,
    oc_dim, ic_dim,
    fy_dim, fx_dim,
    iy_dim, ix_dim,
    padding
):
    X = init_arr(
        n_dim, iy_dim, ix_dim, ic_dim, mode = "random"
    )
    layers = [
        {
            "type" : "conv2d",
            "params" : (ic_dim, oc_dim, 3, 1)
         },
        {
            "type" : "batch_norm_2d",
            "params" : (oc_dim,)
        }
    ]
    data = list(init(layers))
    backends = {
        "torch" : {
            "conv2d" : torch_conv2d,
            "batch_norm_2d" : torch_batch_norm_2d
        },
        "cl" : {
            "conv2d" : cl_conv2d,
            "batch_norm_2d" : cl_batch_norm_2d
        }
    }
    X_torch = process(X, layers, data, backends["torch"])
    X_cl = process(X, layers, data, backends["cl"])
    print("Rel. err torch, cl   : %12.10f" % mean_rel_err(X_torch, X_cl))


def main():
    global PLAT_IDX, PATH, V_SIZE, PE_S, X_SCALE
    PLAT_IDX = int(argv[1])
    PATH = Path(argv[2])
    V_SIZE = int(argv[3])
    PE_S = int(argv[4])
    X_SCALE = int(argv[5])

    im2col_cl(
        1,
        64, 3,
        3, 3,
        16, 16,
        1
    )

main()
