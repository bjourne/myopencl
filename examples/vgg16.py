# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# This example implements VGG16 inference using OpenCL.
from math import ceil, prod
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from pickle import load as pickle_load
from sys import argv
from time import time

from torch import from_numpy, no_grad
from torch import load as torch_load
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

import click
import myopencl as cl
import numpy as np
import torch

V_SIZE = 8
PE_S = 4
X_SCALE = 8

# Dimensions for blocking matrix mul.
BLOCK_N = PE_S ** 2
BLOCK_M = X_SCALE * V_SIZE
BLOCK_K = PE_S ** 2

########################################################################
# NumPy code
########################################################################
def tile(x, win, step, axis):
    assert x.flags["C_CONTIGUOUS"]
    assert x.data.contiguous

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

def tile_matrix(mat, ts_y, ts_x):
    return tile(mat, (ts_y, ts_x), (ts_y, ts_x), (0, 1))

########################################################################
# Torch code
########################################################################
def load_cifar_test(data_dir, batch_size, n_cls):
    norm = Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
    t_te = Compose([ToTensor(), norm])
    cls = CIFAR10 if n_cls == 10 else CIFAR100
    d_te = cls(data_dir, False, t_te, download = True)
    l_te = DataLoader(d_te, batch_size, True, drop_last = True)
    if n_cls == 10:
        names = []
    else:
        meta = data_dir / 'cifar-100-python' / 'meta'
        with open(meta, 'rb') as f:
            d = pickle_load(f)
            names = d['fine_label_names']
    return l_te, names


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

def create_and_write_np_arr(ctx, name, x):
    ro_flag = cl.MemFlags.CL_MEM_READ_ONLY
    ctx.register_buffer(name, x.nbytes, ro_flag)
    write_np_arr(ctx, name, x)

def read_np_arr(ctx, name, x):
    ptr = np.ctypeslib.as_ctypes(x)
    ev = ctx.read_buffer("main", name, x.nbytes, ptr)
    cl.wait_for_events([ev])

def run_kernel(ctx, qname, kname, params):
    params = [(cl.cl_uint, p) if type(p) == int else p
              for p in params]
    return ctx.run_kernel(qname, "main", kname, [1], None, params)

########################################################################
# OpenCL layer handling
########################################################################
def schedule_sa_matmul(mat_w, size_n, size_m, size_k):
    assert mat_w.size == size_m * size_k
    n_blocks = ceil(size_n / BLOCK_N)
    m_blocks = max(ceil(size_m / BLOCK_M), 3)
    k_blocks = ceil(size_k / BLOCK_K)

    pad_n = n_blocks * BLOCK_N
    pad_m = m_blocks * BLOCK_M
    pad_k = k_blocks * BLOCK_K

    add_n = pad_n - size_n
    add_m = pad_m - size_m
    add_k = pad_k - size_k

    mat_w = np.pad(mat_w, [(0, add_k), (0, add_m)])
    mat_w = np.ascontiguousarray(tile_matrix(mat_w, BLOCK_K, BLOCK_M))
    assert mat_w.dtype == np.float32

    return [
        ("preproc_a", [size_n, size_m, n_blocks, m_blocks], []),
        ("sa_matmul", [n_blocks, m_blocks, k_blocks], [mat_w]),
        ("postproc_c", [size_n, size_k, n_blocks, k_blocks], []),
    ], [size_n, size_k]

def conv2d_to_cl(mod, x_shape):
    assert not mod.bias
    w = mod.weight.numpy()
    oc_dim, ic_dim, fy_dim, fx_dim = w.shape

    # oc, fy, fx, ic
    pad = mod.padding[0]
    n_dim, iy_dim, ix_dim, ic_dim = x_shape
    oy_dim = iy_dim + 2 * pad - fy_dim + 1
    ox_dim = ix_dim + 2 * pad - fx_dim + 1

    tasks = []
    args = [
        n_dim,
        iy_dim, ix_dim,
        fy_dim, fx_dim,
        ic_dim, oc_dim,
        pad
    ]
    tasks.append(("conv2d_im2col", args, []))

    size_n = n_dim * oy_dim * ox_dim
    size_m = fy_dim * fx_dim * ic_dim
    size_k = oc_dim

    w = w.transpose(0, 2, 3, 1)
    w = np.ascontiguousarray(w)
    w = w.reshape(size_k, size_m)
    tasks2, _ = schedule_sa_matmul(w, size_n, size_m, size_k)
    return tasks + tasks2, [n_dim, oy_dim, ox_dim, oc_dim]

def linear_to_cl(mod, x_shape):
    w = mod.weight.numpy()
    b = mod.bias.numpy()

    size_n, size_m = x_shape
    size_k, by = w.shape
    assert by == size_m
    assert len(b) == size_k

    tasks, x_shape = schedule_sa_matmul(w, size_n, size_m, size_k)
    tasks.append(("linear_bias", x_shape, [b]))
    return tasks, x_shape

def batch_norm2d_to_cl(mod, x_shape):
    buffers = mod.running_mean, mod.running_var, mod.weight, mod.bias
    mean, var, weight, bias = [b.numpy() for b in buffers]
    mul = weight / np.sqrt(var + 1e-5)
    add = -mean * mul + bias
    args = [prod(x_shape[:-1]), x_shape[-1]]
    return [("batch_norm2d", args, [mul, add])], x_shape

def flatten_to_cl(mod, x_shape):
    return [], (x_shape[0], prod(x_shape[1:]))

def max_pool2d_to_cl(mod, x_shape):
    n_dim, iy_dim, ix_dim, ic_dim = x_shape
    y_shape = n_dim, iy_dim // 2, ix_dim // 2, ic_dim
    return [("max_pool2d", list(x_shape) + [2], [])], y_shape

def sequential_to_cl(mod, x_shape):
    return [], x_shape

def relu_to_cl(mod, x_shape):
    return [("relu", [prod(x_shape)], [])], x_shape


# Converts params to NumPy. Every layer has type, scalar params, and
# buffer params.
def torch_to_cl_net(net, x_shape):
    handlers = {
        BatchNorm2d: batch_norm2d_to_cl,
        Conv2d : conv2d_to_cl,
        Flatten : flatten_to_cl,
        Linear : linear_to_cl,
        MaxPool2d : max_pool2d_to_cl,
        ReLU : relu_to_cl,
        Sequential : sequential_to_cl
    }
    tasks = []
    for m in net.modules():
        tp = type(m)
        tasks2, x_shape = handlers[tp](m, x_shape)
        tasks.extend(tasks2)
    return tasks, x_shape

def cl_run(cl_net, y_shape, x, source, plat_idx):
    opts = format_build_opts(
        "-cl-std=CL2.0",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
        "-Werror",
        INCLUDE_PP = 1,
        TYPE_SEL = 2,
        PE_S = PE_S,
        X_SCALE = X_SCALE,
        V_SIZE = V_SIZE
    )
    ctx = Context.from_indexes(plat_idx, 0)
    ctx.register_program("main", source, opts)
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    ctx.register_queue("main", props)
    ctx.register_queue("aux1", props)
    ctx.register_queue("aux2", props)

    buf_size = 512 * 1024**2
    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    ctx.register_buffer("x", buf_size, rw_flag)
    ctx.register_buffer("y", buf_size, rw_flag)

    for i, (_, _, buffers) in enumerate(cl_net):
        for j, buf in enumerate(buffers):
            create_and_write_np_arr(ctx, (i, j), buf)

    # Write initial input
    write_np_arr(ctx, "x", x)
    src, dst = "x", "y"

    events = []
    bef = time()
    for i, (tp, scalars, param_bufs) in enumerate(cl_net):
        param_buf_ids = [(i, j) for j in range(len(param_bufs))]
        if tp == "sa_matmul":
            n_blocks, m_blocks, k_blocks = scalars
            w = param_buf_ids[0]
            ev1 = run_kernel(ctx, "main", "load_a", [
                src, n_blocks, m_blocks, k_blocks
            ])
            ev2 = run_kernel(ctx, "aux1", "load_b", [
                w, n_blocks, m_blocks, k_blocks
            ])
            ev3 = run_kernel(ctx, "aux2", "store", [
                dst, n_blocks, m_blocks, k_blocks
            ])
            cl.wait_for_events([ev3])
            ev = ev3
        else:
            ev = run_kernel(
                ctx, "main", tp,
                [src, dst] + param_buf_ids + scalars
            )
        name = "%3d %-15s %-33s" % (i, tp, scalars)
        events.append((name, ev))
        src, dst = dst, src

    y = np.empty(y_shape, dtype = np.float32)
    read_np_arr(ctx, src, y)
    took = time() - bef

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for name, ev in events:
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-54s: %8.4f" % (name, secs))
    print("%-54s: %8.4f" % ("total", took))
    print()
    ctx.finish_and_release()
    return y

def load_cifar100_net(path):
    net = torch.load(path, weights_only = False)

    # And test data
    l_te, names = load_cifar_test(Path("/tmp/data"), 16, 100)
    x, _ = next(iter(l_te))
    x = x.numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)
    x = np.ascontiguousarray(x)
    return net, x

@click.command()
@click.argument(
    "network",
    type = click.Path(exists = True),
)
@click.argument(
    "source",
    nargs = -1,
    type = click.Path(exists = True),
)
@click.option(
    "-pi", "--platform-index",
    default = 0,
    help = "Index of platform to use."
)
def main(network, platform_index, source):
    """Loads a PyTorch network and runs inteference on it for one batch."""
    net, x = load_cifar100_net(network)

    # Run on torch
    with no_grad():
        # Train batch norm layers
        torch_run(net, x)
        net.eval()
        y_torch = torch_run(net, x)
        cl_net, y_shape = torch_to_cl_net(net, list(x.shape))

    source = [Path(s) for s in source]
    y_cl = cl_run(cl_net, y_shape, x, source[0], platform_index)

    print("cl/torch diff  :", np.sum(np.abs(y_cl - y_torch)))

    # Compare
    assert y_torch.shape == y_cl.shape
    assert y_torch.dtype == y_cl.dtype

    print(y_torch.argmax(axis = 1))
    print(y_cl.argmax(axis = 1))

    diff = np.abs(y_cl - y_torch)
    print(np.max(diff), np.mean(diff))
    print("Normed err: ", np.linalg.norm(y_cl - y_torch))

if __name__ == "__main__":
    main()
