# Copyright (C) 2024 Björn A. Lindqvist
from collections import namedtuple
from myopencl.objs import MyContext
from myopencl.utils import (can_compile, is_gpu, platform_device_pairs)
from pathlib import Path
from pytest import mark
from torch.nn.functional import conv2d

import ctypes
import myopencl as cl
import torch

PAIRS = platform_device_pairs()
BUILD_OPTS = "-cl-std=CL2.0 -cl-unsafe-math-optimizations"

Conv2DSetup = namedtuple(
    "Conv2DSetup",
    ["nin", "nout", "ksize", "ih", "iw", "padding", "tol"]
)

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

########################################################################
# Tests
########################################################################
@mark.parametrize("platform_id,device_id", PAIRS)
def test_torch_tensors(platform_id, device_id):
    orig = torch.randn((64, 3, 3, 3, 25))
    new = torch.empty_like(orig)

    ctx = MyContext(platform_id, device_id)
    ctx.register_queue("main", [])

    ev1 = write_torch_tensor(ctx, "main", "x", orig)
    ev2 = read_torch_tensor(ctx, "main", "x", new)
    cl.wait_for_events([ev1, ev2])

    assert torch.sum(new - orig) == 0.0
    ctx.finish_and_release()

@mark.parametrize("platform_id,device_id", PAIRS)
def test_conv2d(platform_id, device_id):
    c = MyContext(platform_id, device_id)
    if not can_compile(c.device_id) or is_gpu(c.device_id):
        c.finish_and_release()
        return

    path = Path("kernels/conv2d.cl")
    c.register_program("conv2d", path, BUILD_OPTS)
    c.register_queue("main", [])

    scenarios = [
        Conv2DSetup(3, 64, 3, 32, 32, 1, 0.001),
        Conv2DSetup(3, 64, 3, 32, 32, 0, 0.001),
        Conv2DSetup(64, 128, 3, 16, 16, 1, 0.01),
        Conv2DSetup(1, 1, 3, 6, 6, 0, 0.0001),
        Conv2DSetup(64, 128, 3, 16, 16, 1, 0.01),
        Conv2DSetup(512, 512, 3, 8, 8, 1, 0.2)
    ]
    for d in scenarios:
        filters = torch.randn(d.nout, d.nin, d.ksize, d.ksize)
        x = torch.randn(1, d.nin, d.ih, d.iw)

        # Run on torch
        y_torch = conv2d(x, filters, padding = d.padding)

        # Run on OpenCL
        y_cl = torch.empty_like(y_torch)
        write_torch_tensor(c, "main", "filters", filters)
        write_torch_tensor(c, "main", "x", x)
        c.register_output_buffer("y", y_cl.nbytes)

        args = [
            (cl.cl_uint, d.nout),
            (cl.cl_uint, d.nin),
            (cl.cl_uint, d.ksize), (cl.cl_uint, d.ksize), "filters",
            (cl.cl_uint, d.ih), (cl.cl_uint, d.iw), "x",
            (cl.cl_uint, d.padding), "y"
        ]
        ev = c.run_kernel("main", "conv2d", "conv2d", [1], None, args)
        cl.wait_for_events([ev])
        ev = read_torch_tensor(c, "main", "y", y_cl)
        cl.wait_for_events([ev])

        diff = torch.abs(torch.sum(y_cl - y_torch))
        if diff > d.tol:
            print("== GOT ==")
            print(y_cl)
            print("== EXPECTED ==")
            print(y_torch)
        assert diff < d.tol
        c.release_all_buffers()
    c.finish_and_release()
