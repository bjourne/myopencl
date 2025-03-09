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

# Number of channels must be divisible by this
CHAN_ALIGN = 16
MAX_N_ELS = 256*1024**2

########################################################################
# NumPy code
########################################################################
def win_count(x, k, s, p):
    return (x - k + 2 * p) // s + 1

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

def compute_blocks(shape, defs):
    adds = []
    counts = []
    for size_n, (blk, n_min) in zip(shape, defs):
        n_blocks = max(ceil(size_n / blk), n_min)
        add = n_blocks * blk - size_n
        adds.append((0, add))
        counts.append(n_blocks)
    return adds, counts

def pad_blocked(mat, defs):
    adds, _ = compute_blocks(mat.shape, defs)
    return np.pad(mat, adds)

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
    if x.ndim == 4:
        x = x.transpose(0, 3, 1, 2)
    x = torch.from_numpy(x)
    y = net(x)
    y = y.numpy()
    if y.ndim == 4:
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
    return ctx.run_kernel(qname, "main", kname, [1], None, params)

########################################################################
# OpenCL layer handling
########################################################################
def adaptive_avg_pool2d_to_cl(mod, x_shape, sa_dims):
    bs_dim, iy_dim, ix_dim, ic_dim = x_shape
    o_dim = mod.output_size
    y_shape = bs_dim, o_dim, o_dim, ic_dim
    return [
        [("adaptive_avg_pool2d", [
            "src", "dst",
            bs_dim, iy_dim, ix_dim, ic_dim,
            o_dim
        ])]
    ], y_shape

def schedule_sa_matmul(mat_w, size_n, size_m, size_k, sa_dims):
    assert mat_w.size == size_m * size_k

    # Compute block dimensions
    v_size, pe_s, x_scale = sa_dims
    blk_n = pe_s ** 2
    blk_m = x_scale * v_size
    blk_k = pe_s ** 2

    shape = [size_n, size_k, size_m]
    defs = [(blk_n, 0), (blk_k, 0), (blk_m, 3)]
    adds, [n_blocks, k_blocks, m_blocks] = compute_blocks(shape, defs)

    mat_w = np.pad(mat_w, adds[1:])
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
    assert x_shape[-1] % CHAN_ALIGN == 0
    w = mod.weight.numpy()
    w = pad_blocked(w, [
        (CHAN_ALIGN, 0),
        (CHAN_ALIGN, 0),
        (1, 0),
        (1, 0)
    ])

    oc_dim, ic_dim, k_dim, k_dim = w.shape

    assert ic_dim % CHAN_ALIGN == 0
    assert oc_dim % CHAN_ALIGN == 0

    pad = mod.padding[0]
    stride = mod.stride[0]

    bs_dim, iy_dim, ix_dim, ic_dim = x_shape
    oy_dim = win_count(iy_dim, k_dim, stride, pad)
    ox_dim = win_count(ix_dim, k_dim, stride, pad)

    size_n = bs_dim * oy_dim * ox_dim
    size_m = k_dim * k_dim * ic_dim
    size_k = oc_dim

    assert size_n * size_m < MAX_N_ELS
    assert size_n * size_k < MAX_N_ELS

    w = w.transpose(0, 2, 3, 1)
    w = np.ascontiguousarray(w)
    w = w.reshape(size_k, size_m)

    # Compute block dimensions
    v_size, pe_s, x_scale = sa_dims
    blk_n = pe_s ** 2
    blk_m = x_scale * v_size
    blk_k = pe_s ** 2

    shape = [size_n, size_k, size_m]
    defs = [(blk_n, 0), (blk_k, 0), (blk_m, 3)]
    adds, [n_blocks, k_blocks, m_blocks] = compute_blocks(shape, defs)

    w = np.pad(w, adds[1:])
    w = np.ascontiguousarray(tile_matrix(w, blk_k, blk_m))
    assert w.dtype == np.float32

    tasks = [
        [("im2col_and_tile", [
            "src", "dst",
            bs_dim, iy_dim, ix_dim, ic_dim,
            k_dim, pad, stride
        ])],
        [("load_a", ["src", n_blocks, m_blocks, k_blocks]),
         ("load_b", [w, n_blocks, m_blocks, k_blocks]),
         ("store", ["dst", n_blocks, m_blocks, k_blocks])
         ],
        [("postproc_c", ["src", "dst", size_n, size_k, n_blocks, k_blocks])]
    ]

    if mod.bias is not None:
        b = mod.bias.numpy()
        b = pad_blocked(b, [(CHAN_ALIGN, 0)])
        tasks.append(
            [("linear_bias", ["src", "dst", b, size_n, size_k, 0])]
        )

    return tasks, [bs_dim, oy_dim, ox_dim, oc_dim]


def linear_to_cl(mod, x_shape, sa_dims, relu):
    w = mod.weight.numpy()
    w = pad_blocked(w, [(CHAN_ALIGN, 0), (1, 0)])
    n_dim, m_dim = x_shape
    k_dim, bx = w.shape
    assert bx == m_dim

    tasks, x_shape = schedule_sa_matmul(w, n_dim, m_dim, k_dim, sa_dims)

    b = mod.bias
    if b is not None:
        b = b.numpy()
        b = pad_blocked(b, [(CHAN_ALIGN, 0)])
        assert len(b) == k_dim
        tasks.append(
            [("linear_bias", ["src", "dst", b] + x_shape + [relu])]
        )
    return tasks, x_shape

def linear_to_cl_no_relu(mod, x_shape, sa_dims):
    return linear_to_cl(mod, x_shape, sa_dims, 0)

def linear_to_cl_yes_relu(mod, x_shape, sa_dims):
    return linear_to_cl(mod.linear, x_shape, sa_dims, 1)

def batch_norm2d_to_cl(mod, x_shape, sa_dims, relu):
    buffers = mod.running_mean, mod.running_var, mod.weight, mod.bias
    mean, var, weight, bias = [b.numpy() for b in buffers]
    mul = weight / np.sqrt(var + 1e-5)
    add = -mean * mul + bias
    dims = [prod(x_shape[:-1]), x_shape[-1]]
    return [
        [("batch_norm2d", ["src", "dst", mul, add] + dims + [relu])]
    ], x_shape

def batch_norm2d_to_cl_no_relu(mod, x_shape, sa_dims):
    return batch_norm2d_to_cl(mod, x_shape, sa_dims, 0)

def batch_norm2d_to_cl_yes_relu(mod, x_shape, sa_dims):
    return batch_norm2d_to_cl(mod.bn, x_shape, sa_dims, 1)

def flatten_to_cl(mod, x_shape, sa_dims):
    return [], (x_shape[0], prod(x_shape[1:]))

def max_pool2d_to_cl(mod, x_shape, sa_dims):
    n_dim, iy_dim, ix_dim, c_dim = x_shape
    k_dim = mod.kernel_size
    stride = mod.stride

    oy_dim = win_count(iy_dim, k_dim, stride, 0)
    ox_dim = win_count(ix_dim, k_dim, stride, 0)
    y_shape = n_dim, oy_dim, ox_dim, c_dim
    return [
        [("max_pool2d", [
            "src", "dst",
            n_dim, iy_dim, ix_dim, c_dim,
            k_dim, stride
        ])]
    ], y_shape

def identity_to_cl(mod, x_shape, sa_dims):
    return [], x_shape

def relu_to_cl(mod, x_shape, sa_dims):
    return [
        [("relu", ["src", "dst", prod(x_shape)])]
    ], x_shape

class BatchNorm2dWithReLU:
    def __init__(self, bn):
        self.bn = bn

class LinearWithReLU:
    def __init__(self, linear):
        self.linear = linear

LAYER_HANDLERS = {
    AdaptiveAvgPool2d: adaptive_avg_pool2d_to_cl,
    BatchNorm2d: batch_norm2d_to_cl_no_relu,
    BatchNorm2dWithReLU: batch_norm2d_to_cl_yes_relu,
    Conv2d : conv2d_to_cl,
    Flatten : flatten_to_cl,
    Linear : linear_to_cl_no_relu,
    LinearWithReLU : linear_to_cl_yes_relu,
    MaxPool2d : max_pool2d_to_cl,
    ReLU : relu_to_cl,
    Sequential : identity_to_cl
}


# Converts params to NumPy. Every layer has type, scalar params, and
# buffer params.
def torch_to_cl_net(net, x_shape, sa_dims):
    modules = list(net.modules())
    fused_modules = []
    last_tp = None
    for m in modules:
        tp = type(m)
        if tp == ReLU and last_tp == BatchNorm2d:
            fused_modules[-1] = BatchNorm2dWithReLU(fused_modules[-1])
        elif tp == ReLU and last_tp == Linear:
            fused_modules[-1] = LinearWithReLU(fused_modules[-1])
        else:
            fused_modules.append(m)
        last_tp = tp

    # Temporary solution...
    fused_modules = [fm for fm in fused_modules if type(fm) in LAYER_HANDLERS]

    x_shape = list(x_shape)
    if len(x_shape) == 4:
        x_shape[-1] = ceil(x_shape[-1] / CHAN_ALIGN) * CHAN_ALIGN

    tasks = []
    for m in fused_modules:
        assert(prod(x_shape) < MAX_N_ELS)
        tp = type(m)
        fun = LAYER_HANDLERS[tp]
        tasks2, x_shape = fun(m, x_shape, sa_dims)
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
    for i in range(4):
        ctx.register_queue(i, props)

    buf_size = 4 * MAX_N_ELS
    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    ctx.register_buffer("src", buf_size, rw_flag)
    ctx.register_buffer("dst", buf_size, rw_flag)

    for i, invocations in enumerate(cl_net):
        for j, (kname, args) in enumerate(invocations):
            for k, arg in enumerate(args):
                if type(arg) == np.ndarray:
                    create_and_write_np_arr(ctx, 0, (i, j, k), arg)

    # Pad and write initial input
    if x.ndim == 4:
        x = pad_blocked(x, [(1, 0), (1, 0), (1, 0), (CHAN_ALIGN, 0)])
        assert(x.shape[-1] == CHAN_ALIGN)

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
        args = " ".join("%7s" % a for a in int_args)
        print("%2d %2d %-21s %-65s %8.3f" % (i, j, kname, args, secs))
    print()
    for kname, tot in tally.items():
        print("%-22s %8.3f" % (kname, tot))
    print("%-22s %8.3f" % ("total", total))
    print()
    print("%-22s: %8.4f" % ("total", took))
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
        cl_net, y_shape = torch_to_cl_net(net, x.shape, sa_dims)

    source = [Path(s) for s in source]
    y_cl = cl_run(
        cl_net, y_shape, x, source,
        platform_index, sa_dims
    )

    y_cl = y_cl[tuple(slice(None, e) for e in y_torch.shape)]

    # Compare
    assert y_torch.shape == y_cl.shape
    assert y_torch.dtype == y_cl.dtype

    print("cl/torch diff  :", np.sum(np.abs(y_cl - y_torch)))

    pdiff = y_torch.argmax(axis = 1) - y_cl.argmax(axis = 1)
    print("pdiff", np.sum(pdiff))

    diff = np.abs(y_cl - y_torch)
    print(np.max(diff), np.mean(diff))
    print("Normed err: ", np.linalg.norm(y_cl - y_torch))

if __name__ == "__main__":
    main()
