# Copyright (C) 2024-2025 Björn A. Lindqvist
from collections import namedtuple
from einops import rearrange
from myopencl.objs import Context
from myopencl.utils import can_compile, is_gpu, platform_device_pairs
from pathlib import Path
from pytest import mark
from torch.nn.functional import conv2d, pad

import ctypes
import myopencl as cl
import torch

PAIRS = platform_device_pairs()

V_SIZE = 16
K_SIZE = 3
BUILD_OPTS = " ".join([
    "-cl-std=CL2.0",
    "-cl-unsafe-math-optimizations",
    "-I kernels",
    f"-D V_SIZE={V_SIZE}",
    f"-D K_SIZE={K_SIZE}",
    f"-D DEBUG=1"
])

Conv2DSetup = namedtuple(
    "Conv2DSetup",
    ["nin", "nout", "ksize", "ih", "iw", "padding", "tol"]
)

CONV2D_SETUPS = [
    Conv2DSetup(4, 1, 3, 4, 4, 1, 0.001),
    Conv2DSetup(3, 8, 3, 4, 4, 1, 0.001),
    Conv2DSetup(8, 2, 3, 4, 4, 1, 0.001),
    Conv2DSetup(3, 64, 3, 32, 32, 1, 0.001),
    Conv2DSetup(3, 64, 3, 32, 32, 0, 0.001),
    Conv2DSetup(64, 128, 3, 16, 16, 1, 0.01),
    Conv2DSetup(2, 2, 3, 6, 6, 0, 0.0001),
    Conv2DSetup(64, 128, 3, 16, 16, 1, 0.01),
    Conv2DSetup(512, 512, 3, 8, 8, 1, 0.2)
]

########################################################################
# Utils
########################################################################
def write_torch_tensor(ctx, qname, bname, x):
    assert x.is_contiguous()
    assert x.is_cpu

    cptr = ctypes.cast(x.data_ptr(), ctypes.c_void_p)
    return ctx.register_input_buffer(qname, bname, x.nbytes, cptr)

def read_torch_tensor(ctx, q_name, buf_name, x):
    assert x.is_contiguous()
    assert x.is_cpu

    c_ptr = ctypes.cast(x.data_ptr(), ctypes.c_void_p)
    return ctx.read_buffer(q_name, buf_name, x.nbytes, c_ptr)

def compare_tensors(t1, t2, tol):
    assert t1.shape == t2.shape
    diff = torch.abs(torch.sum(t1 - t2))
    if diff > tol:
        print("== GOT ==")
        print(t1)
        print("== EXPECTED ==")
        print(t2)
    assert diff < tol

########################################################################
# Tests
########################################################################
@mark.parametrize("platform_id,device_id", PAIRS)
def test_torch_tensors(platform_id, device_id):
    orig = torch.randn((64, 3, 3, 3, 25))
    new = torch.empty_like(orig)

    ctx = Context(platform_id, device_id)
    ctx.register_queue("main", [])

    ev1 = write_torch_tensor(ctx, "main", "x", orig)
    ev2 = read_torch_tensor(ctx, "main", "x", new)
    cl.wait_for_events([ev1, ev2])

    assert torch.sum(new - orig) == 0.0
    ctx.finish_and_release()

@mark.parametrize("platform_id,device_id", PAIRS)
def test_conv2d(platform_id, device_id):
    c = Context(platform_id, device_id)
    if not can_compile(c.device_id) or is_gpu(c.device_id):
        c.finish_and_release()
        return

    path = Path("kernels/conv2d.cl")
    c.register_program("conv2d", [path], BUILD_OPTS)
    c.register_queue("main", [])

    for d in CONV2D_SETUPS:
        W = torch.randn(d.nout, d.nin, d.ksize, d.ksize)
        X = torch.randn(1, d.nin, d.ih, d.iw)

        # Run on torch
        Y_torch = conv2d(X, W, padding = d.padding)

        # Run on OpenCL
        Y_cl = torch.empty_like(Y_torch)
        write_torch_tensor(c, "main", "filters", W)
        write_torch_tensor(c, "main", "x", X)
        c.register_output_buffer("y", Y_cl.nbytes)

        args = [
            (cl.cl_uint, d.nout),
            (cl.cl_uint, d.nin),
            (cl.cl_uint, d.ksize), (cl.cl_uint, d.ksize), "filters",
            (cl.cl_uint, d.ih), (cl.cl_uint, d.iw), "x",
            (cl.cl_uint, d.padding), "y"
        ]
        ev = c.run_kernel("main", "conv2d", "conv2d", [1], None, args)
        ev = read_torch_tensor(c, "main", "y", Y_cl)
        cl.wait_for_events([ev])

        compare_tensors(Y_cl, Y_torch, d.tol)

        c.release_all_buffers()
    c.finish_and_release()

@mark.parametrize("platform_id, device_id", PAIRS)
def test_conv2d_fix(platform_id, device_id):
    c = Context(platform_id, device_id)
    if not can_compile(c.device_id) or is_gpu(c.device_id):
        c.finish_and_release()
        return

    path = Path("kernels/conv2d_fix.cl")
    c.register_program("conv2d", [path], BUILD_OPTS)
    c.register_queue("main", [])

    for d in CONV2D_SETUPS:

        dc, sc = d.nout, d.nin
        sy, sx = d.ih, d.iw
        fy, fx = d.ksize, d.ksize

        W = torch.randn(dc, sc, fy, fx)
        X = torch.randn(1, sc, sy, sx)

        # Run on torch
        Y_torch = conv2d(X, W, padding = d.padding)

        # Run on OpenCL
        n, dc, dy, dx = Y_torch.shape
        Y_cl = torch.empty(n, dy, dx, dc)

        # Rearrange into efficient format and pad
        X = rearrange(X, "n sc sy sx -> n sy sx sc").contiguous()
        W = rearrange(W, "dc sc fy fx -> dc fy fx sc").contiguous()

        rem = sc % V_SIZE
        n_pad = V_SIZE - rem if rem else 0
        X = pad(X, (0, n_pad), "constant", 0)
        W = pad(W, (0, n_pad), "constant", 0)

        write_torch_tensor(c, "main", "filters", W)
        write_torch_tensor(c, "main", "x", X)
        c.register_output_buffer("y", Y_cl.nbytes)

        args = [
            (cl.cl_uint, dc), (cl.cl_uint, sc + n_pad),
            (cl.cl_uint, fy), (cl.cl_uint, fx), "filters",
            (cl.cl_uint, sy), (cl.cl_uint, sx), "x",
            (cl.cl_uint, d.padding), "y"
        ]
        c.run_kernel("main", "conv2d", "conv2d", [1], None, args)
        ev = read_torch_tensor(c, "main", "y", Y_cl)
        cl.wait_for_events([ev])

        Y_cl = rearrange(Y_cl, "n dy dx dc -> n dc dy dx")

        compare_tensors(Y_cl, Y_torch, d.tol)
        c.release_all_buffers()
    c.finish_and_release()
