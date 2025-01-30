# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# This example implements VGG16 inference using OpenCL.
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv
from time import time
from torch import from_numpy, no_grad
from torch.nn import *

import myopencl as cl
import numpy as np

np.set_printoptions(
    precision=8,
    linewidth=200,
    threshold=5000,
    suppress=True
)

########################################################################
# Torch code
########################################################################
def build_vgg_layers(n_cls):
    VGG16_LAYERS = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ]
    n_chans_in = 3
    for v in VGG16_LAYERS:
        if type(v) == int:
            # Batch norm so no bias.
            yield Conv2d(n_chans_in, v, 3, padding=1, bias=False)
            yield BatchNorm2d(v)
            yield ReLU(True)
            n_chans_in = v
        elif v == "M":
            yield MaxPool2d(2)
        else:
            assert False
    yield Flatten()
    yield Linear(512, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, 4096)
    yield ReLU(inplace = True)
    yield Linear(4096, n_cls)


########################################################################
# NumPy code
########################################################################
def tile(x, win, step, axis):
    step = np.array(step)
    axis = np.array(axis)
    shape = np.array(x.shape)
    strides = np.array(x.strides)

    assert len(win) == len(step) == len(axis)
    assert len(win) <= len(shape)

    new_strides = np.concatenate((
        strides[:axis[0]],
        strides[axis] * step,
        strides[axis],
        strides[axis[-1] + 1:]
    ))

    n_wins = (shape[axis] - win) // step + 1
    new_shape = np.concatenate((
        shape[:axis[0]],
        n_wins,
        win,
        shape[axis[-1] + 1:]
    ))
    return as_strided(x, new_shape, new_strides)

def im2col(X, W, pad):
    n, iy, ix, ic = X.shape
    fy, fx, ic, oc = W.shape

    X = np.pad(X, [(0, 0), (pad, pad), (pad, pad), (0, 0)])
    X = tile(X, (fy, fx), (1, 1), (1, 2))

    # Destination size
    oy = iy + 2 * pad - fy + 1
    ox = ix + 2 * pad - fx + 1

    X = X.reshape(-1, fy * fx * ic)
    W = W.reshape(-1, oc)
    return X, W, (n, oy, ox, oc)

########################################################################
# Torch runner
########################################################################
def torch_run(net, x):
    # Torch wants (N, C, Y, X) tensors
    if len(x.shape) == 4:
        x = x.transpose(0, 3, 1, 2)
    x = from_numpy(x)
    y = net(x)
    y = y.numpy()
    if len(y.shape) == 4:
        y = y.transpose(0, 2, 3, 1)
    return y

########################################################################
# NumPy runner
########################################################################
def np_batch_norm2d(l, x):
    ps = l.running_mean, l.running_var, l.weight, l.bias
    ps = [p.numpy().reshape(1, 1, 1, -1) for p in ps]
    m, v, w, b = ps
    return w * (x - m) / np.sqrt(v + 1e-5) + b

def np_conv2d(l, x):
    assert not l.bias
    assert l.padding == (1, 1)
    w = l.weight.numpy()
    w = w.transpose(2, 3, 1, 0)
    x, w, y_shape = im2col(x, w, 1)
    y = x @ w
    return y.reshape(y_shape)

def np_flatten(l, x):
    return x.reshape(x.shape[0], -1)

def np_linear(l, x):
    w = l.weight.numpy()
    b = l.bias.numpy()
    return x @ w.T + b

def np_max_pool2d(l, x):
    wins = tile(x, (2, 2), (2, 2), (1, 2))
    return wins.max(axis = (3, 4))

def np_relu(l, x):
    x[x < 0] = 0
    return x

def np_run(net, x):
    handlers = {
        MaxPool2d : np_max_pool2d,
        BatchNorm2d : np_batch_norm2d,
        Conv2d : np_conv2d,
        Flatten : np_flatten,
        Linear : np_linear,
        ReLU : np_relu
    }
    for mod in net.modules():
        if isinstance(mod, Sequential):
            continue
        tp = type(mod)
        x = handlers[tp](mod, x)
    return x

########################################################################
# OpenCL utility
########################################################################
def format_build_opts(*args, **kw):
    defs = [f"-D {k}={v}" for (k, v) in kw.items()]
    return " ".join(list(args) + defs)

def write_np_arr(ctx, name, x):
    assert x.flags["C_CONTIGUOUS"]
    assert x.data.contiguous
    assert x.dtype == np.float32
    ptr = np.ctypeslib.as_ctypes(x)
    nbytes = x.nbytes
    return ctx.write_buffer("main", name, nbytes, ptr)

def read_np_arr(ctx, name, x):
    ptr = np.ctypeslib.as_ctypes(x)
    ev = ctx.read_buffer("main", name, x.nbytes, ptr)
    cl.wait_for_events([ev])

def run_kernel(ctx, name, bufs, uints):
    params = bufs + [(cl.cl_uint, i) for i in uints]
    return ctx.run_kernel("main", "main", name, [1], None, params)

########################################################################
# OpenCL layer handling
########################################################################
def cl_batch_norm2d(ctx, l, x):
    ps = l.running_mean, l.running_var, l.weight, l.bias

    write_np_arr(ctx, "x", x)
    param_bufs = ["w0", "w1", "w2", "w3"]
    for name, p in zip(param_bufs, ps):
        write_np_arr(ctx, name, p.numpy())

    run_kernel(
        ctx,
        "batch_norm2d",
        ["x", "y"] + param_bufs,
        x.shape
    )
    y = np.empty_like(x)
    read_np_arr(ctx, "y", y)
    return y

def cl_conv2d(ctx, l, x):
    assert not l.bias
    assert l.padding == (1, 1)
    w = l.weight.numpy()
    w = w.transpose(2, 3, 1, 0)
    x, w, y_shape = im2col(x, w, 1)

    n_dim, m_dim = x.shape
    _, k_dim = w.shape

    y = np.empty((n_dim, k_dim), dtype = np.float32)
    write_np_arr(ctx, "x", x)
    write_np_arr(ctx, "w0", w)
    run_kernel(ctx, "conv2d", ["x", "y", "w0"], [n_dim, m_dim, k_dim])
    read_np_arr(ctx, "y", y)
    return y.reshape(y_shape)

def cl_flatten(ctx, l, x):
    return x.reshape(x.shape[0], -1)

def cl_linear(ctx, l, x):
    w = l.weight.numpy()
    b = l.bias.numpy()
    n_dim, m_dim = x.shape
    k_dim = w.shape[0]

    y = np.empty((n_dim, k_dim), dtype = np.float32)
    write_np_arr(ctx, "x", x)
    write_np_arr(ctx, "w0", w)
    write_np_arr(ctx, "w1", b)
    run_kernel(ctx, "linear", ["x", "y", "w0", "w1"], [n_dim, m_dim, k_dim])
    read_np_arr(ctx, "y", y)
    return y

def cl_max_pool2d(ctx, l, x):
    write_np_arr(ctx, "x", x)
    n_dim, y_dim, x_dim, c_dim = x.shape
    run_kernel(ctx, "max_pool2d", ["x", "y"], [n_dim, y_dim, x_dim, c_dim, 2])
    y = np.empty((n_dim, y_dim // 2, x_dim // 2, c_dim),
                 dtype = np.float32)
    read_np_arr(ctx, "y", y)
    return y

def cl_relu(ctx, l, x):
    write_np_arr(ctx, "x", x)
    run_kernel(
        ctx,
        "relu",
        ["x", "y"],
        [x.size]
    )
    y = np.empty_like(x)
    read_np_arr(ctx, "y", y)
    return y

def cl_run(net, x, path, plat_idx):
    opts = format_build_opts(
        "-cl-std=CL2.0",
        "-Werror",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable"
    )

    ctx = Context.from_indexes(plat_idx, 0)
    ctx.register_program("main", path, opts)
    ctx.register_queue("main", [])

    buf_size = 256 * 1024**2

    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    ro_flag = cl.MemFlags.CL_MEM_READ_ONLY
    ctx.register_buffer("x", buf_size, rw_flag)
    ctx.register_buffer("y", buf_size, rw_flag)
    ctx.register_buffer("w0", buf_size, ro_flag)
    ctx.register_buffer("w1", buf_size, ro_flag)
    ctx.register_buffer("w2", buf_size, ro_flag)
    ctx.register_buffer("w3", buf_size, ro_flag)
    handlers = {
        MaxPool2d : cl_max_pool2d,
        BatchNorm2d : cl_batch_norm2d,
        Conv2d : cl_conv2d,
        Flatten : cl_flatten,
        Linear : cl_linear,
        ReLU : cl_relu
    }
    for mod in net.modules():
        assert x.dtype == np.float32
        if isinstance(mod, Sequential):
            continue
        tp = type(mod)
        x = handlers[tp](ctx, mod, x)

    ctx.finish_and_release()
    return x

def main():
    n_dim, y_dim, x_dim, c_dim = 64, 32, 32, 3

    net = Sequential(*build_vgg_layers(10))
    path = Path(argv[1])
    plat_idx = int(argv[2])

    x = np.random.rand(n_dim, y_dim, x_dim, c_dim).astype(np.float32)
    with no_grad():
        # Train batch norm layers
        torch_run(net, x)
        net.eval()
        y_torch = torch_run(net, x)
        y_np = np_run(net, x)
        bef = time()
        y_cl = cl_run(net, x, path, plat_idx)
        print("CL time", time() - bef)

    assert y_torch.shape == y_np.shape
    assert y_torch.dtype == y_np.dtype

    diff = np.abs(y_cl - y_torch)
    print(np.max(diff), np.mean(diff))
    print("Normed err: ", np.linalg.norm(y_np - y_torch))

main()
