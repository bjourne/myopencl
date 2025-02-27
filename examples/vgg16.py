# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# This example implements VGG16 inference using OpenCL.
#
# Run using:
#
#   python examples/vgg16.py vgg16-full.pth kernels/ops/*.cl --sa-dims 16 4 8
from collections import defaultdict
from math import ceil, prod
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from pickle import load as pickle_load
from sys import argv
from time import time

from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

import click
import myopencl as cl
import numpy as np
import torch

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
    x = torch.from_numpy(x)
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

def write_np_arr(ctx, qname, bname, x):
    assert x.flags["C_CONTIGUOUS"]
    assert x.data.contiguous
    assert x.dtype == np.float32
    ptr = np.ctypeslib.as_ctypes(x)
    nbytes = x.nbytes
    return ctx.write_buffer(qname, bname, nbytes, ptr)

def create_and_write_np_arr(ctx, qname, bname, x):
    ro_flag = cl.MemFlags.CL_MEM_READ_ONLY
    ctx.register_buffer(bname, x.nbytes, ro_flag)
    return write_np_arr(ctx, qname, bname, x)

def read_np_arr(ctx, qname, bname, x):
    ptr = np.ctypeslib.as_ctypes(x)
    return ctx.read_buffer(qname, bname, x.nbytes, ptr)

def run_kernel(ctx, qname, kname, params):
    params = [(cl.cl_uint, p) if type(p) == int else p
              for p in params]
    return ctx.run_kernel(qname, "main", kname, [1], None, params)

########################################################################
# OpenCL layer handling
########################################################################
def schedule_sa_matmul(mat_w, size_n, size_m, size_k, sa_dims):
    assert mat_w.size == size_m * size_k

    # Compute block dimensions
    v_size, pe_s, x_scale = sa_dims
    blk_n = pe_s ** 2
    blk_m = x_scale * v_size
    blk_k = pe_s ** 2

    n_blocks = ceil(size_n / blk_n)
    m_blocks = max(ceil(size_m / blk_m), 3)
    k_blocks = ceil(size_k / blk_k)

    pad_n = n_blocks * blk_n
    pad_m = m_blocks * blk_m
    pad_k = k_blocks * blk_k

    add_n = pad_n - size_n
    add_m = pad_m - size_m
    add_k = pad_k - size_k

    mat_w = np.pad(mat_w, [(0, add_k), (0, add_m)])
    mat_w = np.ascontiguousarray(tile_matrix(mat_w, blk_k, blk_m))
    assert mat_w.dtype == np.float32

    return [
        [("preproc_a", ["src", "dst", size_n, size_m, n_blocks, m_blocks])],
        [("load_a", ["src", n_blocks, m_blocks, k_blocks]),
         ("load_b", [mat_w, n_blocks, m_blocks, k_blocks]),
         ("store", ["dst", n_blocks, m_blocks, k_blocks])
         ],
        [("postproc_c", ["src", "dst", size_n, size_k, n_blocks, k_blocks])]
    ], [size_n, size_k]

def conv2d_to_cl(mod, x_shape, sa_dims):
    assert not mod.bias
    w = mod.weight.numpy()
    oc_dim, ic_dim, k_size, k_size = w.shape

    pad = mod.padding[0]
    n_dim, iy_dim, ix_dim, ic_dim = x_shape
    oy_dim = iy_dim + 2 * pad - k_size + 1
    ox_dim = ix_dim + 2 * pad - k_size + 1

    tasks = []
    args = [
        "src", "dst",
        n_dim,
        iy_dim, ix_dim, ic_dim,
        k_size,
        pad
    ]
    tasks.append([("conv2d_im2col", args)])

    size_n = n_dim * oy_dim * ox_dim
    size_m = k_size * k_size * ic_dim
    size_k = oc_dim

    w = w.transpose(0, 2, 3, 1)
    w = np.ascontiguousarray(w)
    w = w.reshape(size_k, size_m)
    tasks2, _ = schedule_sa_matmul(w, size_n, size_m, size_k, sa_dims)
    return tasks + tasks2, [n_dim, oy_dim, ox_dim, oc_dim]

def linear_to_cl(mod, x_shape, sa_dims):
    w = mod.weight.numpy()
    b = mod.bias.numpy()

    size_n, size_m = x_shape
    size_k, by = w.shape
    assert by == size_m
    assert len(b) == size_k

    tasks, x_shape = schedule_sa_matmul(w, size_n, size_m, size_k, sa_dims)
    tasks.append(
        [("linear_bias", ["src", "dst", b] + x_shape)]
    )
    return tasks, x_shape

def batch_norm2d_to_cl(mod, x_shape, sa_dims):
    buffers = mod.running_mean, mod.running_var, mod.weight, mod.bias
    mean, var, weight, bias = [b.numpy() for b in buffers]
    mul = weight / np.sqrt(var + 1e-5)
    add = -mean * mul + bias
    dims = [prod(x_shape[:-1]), x_shape[-1]]
    return [
        [("batch_norm2d", ["src", "dst", mul, add] + dims)]
    ], x_shape

def flatten_to_cl(mod, x_shape, sa_dims):
    return [], (x_shape[0], prod(x_shape[1:]))

def max_pool2d_to_cl(mod, x_shape, sa_dims):
    n_dim, iy_dim, ix_dim, ic_dim = x_shape
    y_shape = n_dim, iy_dim // 2, ix_dim // 2, ic_dim
    return [
        [("max_pool2d", ["src", "dst"] + list(x_shape) + [2])]
    ], y_shape

def sequential_to_cl(mod, x_shape, sa_dims):
    return [], x_shape

def relu_to_cl(mod, x_shape, sa_dims):
    return [
        [("relu", ["src", "dst", prod(x_shape)])]
    ], x_shape

# Converts params to NumPy. Every layer has type, scalar params, and
# buffer params.
def torch_to_cl_net(net, x_shape, sa_dims):
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
        tasks2, x_shape = handlers[tp](m, x_shape, sa_dims)
        tasks.extend(tasks2)
    return tasks, x_shape

def map_args(src, dst, i, j, args):
    for k, a in enumerate(args):
        tp = type(a)
        if tp == int:
            yield cl.cl_uint, a
        elif tp == np.ndarray:
            yield i, j, k
        elif a == "src":
            yield src
        elif a == "dst":
            yield dst
        else:
            assert False

def cl_run(
        cl_net,
        y_shape, x,
        source, platform_index, sa_dims
):
    v_size, pe_s, x_scale = sa_dims
    opts = format_build_opts(
        "-cl-std=CL2.0",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
        INCLUDE_PP = 1,
        TYPE_SEL = 2,
        V_SIZE = v_size,
        PE_S = pe_s,
        X_SCALE = x_scale
    )
    ctx = Context.from_indexes(platform_index, 0)
    ctx.register_program("main", source, opts)
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    ctx.register_queue("main", props)
    ctx.register_queue("aux1", props)
    ctx.register_queue("aux2", props)

    for i in range(4):
        ctx.register_queue(i, props)

    buf_size = 512 * 1024**2
    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    ctx.register_buffer("src", buf_size, rw_flag)
    ctx.register_buffer("dst", buf_size, rw_flag)

    for i, invocations in enumerate(cl_net):
        for j, (kname, args) in enumerate(invocations):

            for k, arg in enumerate(args):
                if type(arg) == np.ndarray:
                    key = i, j, k
                    create_and_write_np_arr(ctx, 0, (i, j, k), arg)

    # Write initial input
    ev = write_np_arr(ctx, 0, "src", x)
    cl.wait_for_events([ev])
    src, dst = "src", "dst"

    log = []
    bef = time()
    for i, invocations in enumerate(cl_net):
        these_evs = []
        for j, (kname, args) in enumerate(invocations):
            mapped_args = list(map_args(src, dst, i, j, args))
            ev = ctx.run_kernel(j, "main", kname, [1], None, mapped_args)
            these_evs.append(ev)
            log.append((i, j, kname, args, ev))
        cl.wait_for_events(these_evs)
        src, dst = dst, src
    took = time() - bef

    y = np.empty(y_shape, dtype = np.float32)
    ev = read_np_arr(ctx, 0, src, y)
    cl.wait_for_events([ev])

    log2 = []
    total = 0
    tally = defaultdict(float)
    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    for i, j, kname, args, ev in log:
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        int_args = [a for a in args if type(a) == int]
        log2.append((i, j, kname, int_args, secs))
        tally[kname] += secs
        if j == 0:
            total += secs
    ctx.finish_and_release()

    for i, j, kname, int_args, secs in log2:
        args = " ".join("%7d" % a for a in int_args)
        print("%2d %2d %-15s %-65s %8.3f" % (i, j, kname, args, secs))
    print()
    for kname, tot in tally.items():
        print("%-15s %8.3f" % (kname, tot))
    print("%-15s %8.3f" % ("total", total))
    print()
    print("%-10s: %8.4f" % ("total", took))
    print()

    return y

def load_cifar100_net(path, bs):
    net = torch.load(path, weights_only = False)

    # And test data
    l_te, names = load_cifar_test(Path("/tmp/data"), bs, 100)
    x, _ = next(iter(l_te))
    x = x.numpy()
    if len(x.shape) == 4:
        x = x.transpose(0, 2, 3, 1)
    x = np.ascontiguousarray(x)
    return net, x

@click.command(context_settings={'show_default': True})
@click.option(
    "-pi", "--platform-index",
    default = 0,
    help = "Index of platform to use"
)
@click.option(
    "--sa-dims",
    default = (8, 16, 8),
    nargs = 3,
    help = "Systolic array dimensions (V_SIZE, PE_S, X_SCALE)"
)
@click.option(
    "-bs", "--batch-size",
    default = 64,
    help = "Batch size"
)
@click.argument(
    "network",
    type = click.Path(exists = True),
)
@click.argument(
    "source",
    nargs = -1,
    required = 1,
    type = click.Path(exists = True),
)
def main(platform_index, sa_dims, batch_size, network, source):
    """Loads a PyTorch network and runs inteference on it for one batch."""
    net, x = load_cifar100_net(network, batch_size)

    # Run on torch
    with torch.no_grad():
        # Train batch norm layers
        torch_run(net, x)
        net.eval()
        y_torch = torch_run(net, x)
        cl_net, y_shape = torch_to_cl_net(net, list(x.shape), sa_dims)

    source = [Path(s) for s in source]
    y_cl = cl_run(
        cl_net, y_shape, x, source,
        platform_index, sa_dims
    )

    print("cl/torch diff  :", np.sum(np.abs(y_cl - y_torch)))

    # Compare
    assert y_torch.shape == y_cl.shape
    assert y_torch.dtype == y_cl.dtype

    pdiff = y_torch.argmax(axis = 1) - y_cl.argmax(axis = 1)
    print("pdiff", np.sum(pdiff))

    diff = np.abs(y_cl - y_torch)
    print(np.max(diff), np.mean(diff))
    print("Normed err: ", np.linalg.norm(y_cl - y_torch))

if __name__ == "__main__":
    main()
