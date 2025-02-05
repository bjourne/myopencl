# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# This example implements VGG16 inference using OpenCL.
from math import ceil, prod
from myopencl.objs import Context
from pathlib import Path
from sys import argv
from time import time

from torch import from_numpy, no_grad
from torch import load as torch_load
from torch.nn import *
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Normalize, ToTensor

import myopencl as cl
import numpy as np
import pickle

V_SIZE = 8
PE_S = 8
X_SCALE = 8

# Dimensions for blocking matrix mul.
BLOCK_N = PE_S ** 2
BLOCK_M = X_SCALE * V_SIZE
BLOCK_K = PE_S ** 2

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
            d = pickle.load(f)
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

def run_kernel(ctx, name, bufs, uints):
    params = bufs + [(cl.cl_uint, i) for i in uints]
    return ctx.run_kernel("main", "main", name, [1], None, params)

########################################################################
# OpenCL layer handling
########################################################################

def conv2d_to_cl(mod, x_shape):
    assert not mod.bias
    w = mod.weight.numpy()
    oc_dim, ic_dim, fy_dim, fx_dim = w.shape

    # oc, fy, fx, ic
    w_t = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
    pad = mod.padding[0]
    n_dim, iy_dim, ix_dim, ic_dim = x_shape
    oy_dim = iy_dim + 2 * pad - fy_dim + 1
    ox_dim = ix_dim + 2 * pad - fx_dim + 1

    layers = []
    args = [
        n_dim,
        iy_dim, ix_dim,
        fy_dim, fx_dim,
        ic_dim, oc_dim,
        pad
    ]
    layers.append(("conv2d_im2col", args, []))

    mat_n = n_dim * oy_dim * ox_dim
    mat_m = fy_dim * fx_dim * ic_dim
    mat_k = oc_dim
    args = [mat_n, mat_m, mat_k]
    layers.append(("conv2d_matmul", args, [w_t]))
    return layers, [n_dim, oy_dim, ox_dim, oc_dim]

def get_padded_dims(n, m, k):
    n_blocks = ceil(n / BLOCK_N)
    m_blocks = max(ceil(m / BLOCK_M), 3)
    k_blocks = ceil(k / BLOCK_K)

    pad_n = n_blocks * BLOCK_N
    pad_m = m_blocks * BLOCK_M
    pad_k = k_blocks * BLOCK_K

    return pad_n, pad_m, pad_k

def linear_to_cl(mod, x_shape):
    w = mod.weight.numpy()
    b = mod.bias.numpy()
    mat_n, mat_m = x_shape

    # Note that it is transposed
    mat_k, _ = w.shape

    pad_n, pad_m, pad_k = get_padded_dims(mat_n, mat_m, mat_k)

    w = np.pad(w, [(0, pad_k - mat_k), (0, pad_m - mat_m)])
    b = np.pad(b, [(0, pad_k - mat_k)])

    layers = [
        ("linear_pad", [mat_n, mat_m, pad_n, pad_m], []),
        ("linear", [pad_n, pad_m, pad_k], [w, b]),
        ("linear_unpad", [pad_n, pad_k, mat_n, mat_k], [])
    ]
    return layers, [mat_n, mat_k]

def batch_norm2d_to_cl(mod, x_shape):
    buffers = mod.running_mean, mod.running_var, mod.weight, mod.bias
    mean, var, weight, bias = [b.numpy() for b in buffers]
    mul = weight / np.sqrt(var + 1e-5)
    add = -mean * mul + bias
    args = [prod(x_shape[:-1]), x_shape[-1]]
    return [("batch_norm2d", args, [mul, add])], x_shape


# Converts params to NumPy. Every layer has type, scalar params, and
# buffer params.
def torch_to_cl_net(net, x_shape):
    tasks = []
    for m in net.modules():
        tp = type(m)
        if tp == BatchNorm2d:
            tasks2, y_shape = batch_norm2d_to_cl(m, x_shape)
            tasks.extend(tasks2)
        elif tp == Conv2d:
            tasks2, y_shape = conv2d_to_cl(m, x_shape)
            tasks.extend(tasks2)
        elif tp == Flatten:
            y_shape = x_shape[0], prod(x_shape[1:])
        elif tp == Linear:
            tasks2, y_shape = linear_to_cl(m, x_shape)
            tasks.extend(tasks2)
        elif tp == MaxPool2d:
            tasks.append(("max_pool2d", list(x_shape) + [2], []))
            n_dim, iy_dim, ix_dim, ic_dim = x_shape
            y_shape = n_dim, iy_dim // 2, ix_dim // 2, ic_dim
        elif tp == ReLU:
            tasks.append(("relu", [prod(x_shape)], []))
            y_shape = x_shape
        elif tp == Sequential:
            y_shape = x_shape
        else:
            assert False
        x_shape = y_shape
    return tasks, x_shape

def cl_run(cl_net, y_shape, x, path, plat_idx):
    opts = format_build_opts(
        "-cl-std=CL2.0",
        "-Werror",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable"
    )
    ctx = Context.from_indexes(plat_idx, 0)
    ctx.register_program("main", path, opts)
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    ctx.register_queue("main", props)

    buf_size = 128 * 1024**2
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
        bufs = [src, dst] + [(i, j) for j in range(len(param_bufs))]
        ev = run_kernel(ctx, tp, bufs, scalars)
        name = "%3d %-15s %-33s" % (i, tp, scalars)
        cl.wait_for_events([ev])
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

N_CLS = 100

def main():
    # Build model
    # net = Sequential(*build_vgg_layers(N_CLS))
    # if len(argv) == 4:
    #     d = torch_load(argv[3], weights_only = True)
    #     d2 = {}
    #     for k, v in d.items():
    #         d2[k[len("features."):]] = v
    #     net.load_state_dict(d2)

    # # Load one batch
    # l_te, names = load_cifar_test(Path("/tmp/data"), 16, N_CLS)
    # x, y_real = next(iter(l_te))
    # x = x.numpy()
    # if len(x.shape) == 4:
    #     x = x.transpose(0, 2, 3, 1)
    # x = np.ascontiguousarray(x)

    x = np.random.randn(16, 100).astype(np.float32)
    net = Sequential(Linear(100, 16))

    mod = list(net.modules())[0][0]
    weight = mod.weight.detach().numpy()
    bias = mod.bias.detach().numpy()
    y_real = x @ weight.T + bias


    # Run on torch
    with no_grad():
        # Train batch norm layers
        torch_run(net, x)
        net.eval()
        y_torch = torch_run(net, x)
        cl_net, y_shape = torch_to_cl_net(net, list(x.shape))



    # Run on cl
    path = Path(argv[1])
    plat_idx = int(argv[2])
    y_cl = cl_run(cl_net, y_shape, x, path, plat_idx)

    # Compare
    assert y_torch.dtype == y_cl.dtype
    print(y_torch.argmax(axis = 1))
    print(y_cl.argmax(axis = 1))

    diff = np.abs(y_cl - y_torch)
    print(np.max(diff), np.mean(diff))
    print("Normed err: ", np.linalg.norm(y_cl - y_torch))

main()
