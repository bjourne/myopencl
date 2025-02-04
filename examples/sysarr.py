# Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
#
# Demonstrates systolic array matmul.
from math import ceil
from myopencl.objs import Context
from numpy.lib.stride_tricks import as_strided
from pathlib import Path
from sys import argv

import myopencl as cl
import numpy as np

########################################################################
# OpenCL code
########################################################################
def write_np_arr(ctx, qname, bname, x):
    assert x.flags["C_CONTIGUOUS"]
    assert x.data.contiguous
    ptr = np.ctypeslib.as_ctypes(x)
    nbytes = x.nbytes
    return ctx.write_buffer(qname, bname, nbytes, ptr)

def read_np_arr(ctx, qname, bname, x):
    ptr = np.ctypeslib.as_ctypes(x)
    ev = ctx.read_buffer(qname, bname, x.nbytes, ptr)
    cl.wait_for_events([ev])

def run_sd_kernel(ctx, qname, kname, params):
    params = [p if type(p) == str else (cl.cl_uint, p)
              for p in params]
    return ctx.run_kernel(qname, "main", kname, [1], None, params)

V_SIZE = 8
PE_S = 16
X_SCALE = 8
TYPE_SEL = 1

def format_build_opts(*args, **kw):
    defs = [f"-D {k}={v}" for (k, v) in kw.items()]
    return " ".join(list(args) + defs)

def cl_matmul(ctx, mat_a, mat_b):
    assert mat_a.dtype == mat_b.dtype

    block_n = PE_S ** 2
    block_m = X_SCALE * V_SIZE
    block_k = PE_S ** 2

    size_n, size_m = mat_a.shape
    by, size_k = mat_b.shape
    assert by == size_m

    n_blocks = ceil(size_n / block_n)
    m_blocks = max(ceil(size_m / block_m), 3)
    k_blocks = ceil(size_k / block_k)

    print("== Setup == ")
    kvs = [
        ("Block dims", (block_n, block_m, block_k)),
        ("Matrix dims", (size_n, size_m, size_k)),
        ("N blocks", (n_blocks, m_blocks, k_blocks)),
    ]
    for k, arr in kvs:
        v = " ".join("%4d" % a for a in arr)
        print("%-20s: %15s" % (k, v))
    print()

    write_np_arr(ctx, "load_a", "mat_a", mat_a)
    write_np_arr(ctx, "load_b", "mat_b", mat_b)

    dim_args = [n_blocks, m_blocks, k_blocks]
    preproc_a_args = ["mat_a", "mat_ap", size_n, size_m, n_blocks, m_blocks]
    preproc_b_args = ["mat_b", "mat_bp", size_m, size_k, m_blocks, k_blocks]
    postproc_args = ["mat_cp", "mat_c", size_n, size_k, n_blocks, k_blocks]
    tasks = [
        ("load_a", "preproc_a", preproc_a_args),
        ("load_a", "load_a", ["mat_ap"] + dim_args),
        ("load_b", "preproc_b", preproc_b_args),
        ("load_b", "load_b", ["mat_bp"] + dim_args),
        ("store", "store", ["mat_cp"] + dim_args),
        ("store", "postproc_c", postproc_args)
    ]

    evs = []
    for qname, kname, args in tasks:
        ev = run_sd_kernel(ctx, qname, kname, args)
        evs.append(ev)
    cl.wait_for_events(evs)

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    print("== Timing ==")
    for ev, (_, kname, _) in zip(evs, tasks):
        start = cl.get_info(attr_start, ev)
        end = cl.get_info(attr_end, ev)
        secs = (end - start) * 1.0e-9
        print("%-20s: %8.3f" % (kname, secs))
    print()

    c_mat = np.empty((size_n, size_k), dtype = mat_a.dtype)
    read_np_arr(ctx, "store", "mat_c", c_mat)
    return c_mat

def main():
    # Setup OpenCL context
    path = Path(argv[1])
    plat_idx = int(argv[2])

    ctx = Context.from_indexes(plat_idx, 0)
    opts = format_build_opts(
        "-cl-std=CL2.0",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
        "-Werror",
        INCLUDE_PP = 1,
        TYPE_SEL = 1,
        PE_S = PE_S,
        X_SCALE = X_SCALE,
        V_SIZE = V_SIZE
    )
    ctx.register_program("main", path, opts)

    # Register queues and buffers
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    for name in ["load_a", "load_b", "store"]:
        ctx.register_queue(name, props)

    nbytes = 128 * 1024**2
    ro_flag = cl.MemFlags.CL_MEM_READ_ONLY
    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    wo_flag = cl.MemFlags.CL_MEM_WRITE_ONLY
    ctx.register_buffer("mat_a", nbytes, ro_flag)
    ctx.register_buffer("mat_ap", nbytes, rw_flag)
    ctx.register_buffer("mat_b", nbytes, ro_flag)
    ctx.register_buffer("mat_bp", nbytes, rw_flag)
    ctx.register_buffer("mat_cp", nbytes, rw_flag)
    ctx.register_buffer("mat_c", nbytes, wo_flag)

    n_trials = 5
    tot_err = 0
    for _ in range(n_trials):
        dims = [100 + 10 * np.random.randint(50) for _ in range(3)]
        size_n, size_m, size_k = dims
        if TYPE_SEL == 1:
            mat_a = np.random.randint(-10, 10, size = (size_n, size_m))
            mat_b = np.random.randint(-10, 10, size = (size_m, size_k))
            mat_a = mat_a.astype(np.int64)
            mat_b = mat_b.astype(np.int64)
        elif TYPE_SEL == 2:
            mat_a = np.random.randn(size_n, size_m)
            mat_b = np.random.randn(size_m, size_k)
            mat_a = mat_a.astype(np.float32)
            mat_b = mat_b.astype(np.float32)
        elif TYPE_SEL == 3:
            mat_a = np.random.randint(-10, 10, size = (size_n, size_m))
            mat_b = np.random.randint(-10, 10, size = (size_m, size_k))
            mat_a = mat_a.astype(np.int32)
            mat_b = mat_b.astype(np.int32)

        mat_c = mat_a @ mat_b
        cl_mat_c = cl_matmul(ctx, mat_a, mat_b)
        err = np.sum(np.abs(mat_c - cl_mat_c))
        tot_err += err
    ctx.finish_and_release()
    mean_err = tot_err / n_trials
    print("Mean error: %.3f (%s)" % (mean_err, cl_mat_c.dtype))

main()
