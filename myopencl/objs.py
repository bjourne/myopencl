# Copyright (C) 2024 Björn A. Lindqvist
from collections import defaultdict
from myopencl.utils import INDENT_STR, pp_dict, terminal_wrapper

import ctypes
import myopencl as cl

def pp_dict_with_header(header, wrap, d):
    print(f"== {header} ==")
    pp_dict(wrap, d)

EVENT_CNT = 0
def name_event(*args):
    global EVENT_CNT
    pf = ":".join(args)
    s = f"{pf}-{EVENT_CNT:03d}"
    EVENT_CNT += 1
    return s

TYPE_STR_TO_CTYPE = {
    "float" : ctypes.c_float,
    "uint" : cl.cl_uint
}

def is_buf_arg(details):
    key = cl.KernelArgInfo.CL_KERNEL_ARG_ADDRESS_QUALIFIER
    val = cl.KernelArgAddressQualifier.CL_KERNEL_ARG_ADDRESS_GLOBAL
    return details[key] == val

def py_to_ctypes(details, py_val):
    key = cl.KernelArgInfo.CL_KERNEL_ARG_TYPE_NAME
    s = details[key]
    ct = TYPE_STR_TO_CTYPE[s]
    return ct(py_val)


class MyContext:
    def __init__(self, plat_idx, dev_idx):
        self.plat_id = cl.get_platform_ids()[plat_idx]
        self.dev_id = cl.get_device_ids(self.plat_id)[dev_idx]
        self.ctx = cl.create_context(self.dev_id)
        self.queues = {}
        self.bufs = {}
        self.events = {}
        self.programs = {}
        self.kernels = defaultdict(dict)

    def create_program(self, prog_name, path):
        source = path.read_text("utf-8")
        prog = cl.create_program_with_source(self.ctx, source)
        opts = "-Werror -cl-std=CL2.0"
        cl.build_program(prog, self.dev_id, opts, True, True)
        self.programs[prog_name] = prog

        for kern_name in cl.get_kernel_names(prog):
            kern = cl.create_kernel(prog, kern_name)
            self.kernels[prog_name][kern_name] = kern

    def set_kernel_args(self, prog_name, kern_name, py_args):
        kern = self.kernels[prog_name][kern_name]
        attr = cl.KernelArgInfo.CL_KERNEL_ARG_TYPE_NAME
        args_details = cl.get_kernel_args_details(kern)
        for i, (py_arg, d) in enumerate(zip(py_args, args_details)):
            c_arg = None
            if is_buf_arg(d) and type(py_arg) == str:
                c_arg = self.bufs[py_arg]
                cl.set_kernel_arg(kern, i, self.bufs[py_arg])
            else:
                c_arg = py_to_ctypes(d, py_arg)
            cl.set_kernel_arg(kern, i, c_arg)

    def enqueue_kernel(self, q_name, p_name, k_name,
                       global_work, local_work):
        q = self.queues[q_name]
        k = self.kernels[p_name][k_name]
        return cl.enqueue_nd_range_kernel(q, k, global_work, local_work)

    def run_kernel(self,
                   q_name, p_name, k_name,
                   gl_work, lo_work,
                   args):
        self.set_kernel_args(p_name, k_name, args)
        return self.enqueue_kernel(q_name, p_name, k_name, gl_work, lo_work)

    def create_queue(self, name, props):
        q = cl.create_command_queue_with_properties(
            self.ctx, self.dev_id, props
        )
        self.queues[name] = q

    def finish_and_release(self):
        for queue in self.queues.values():
            cl.flush(queue)
            cl.finish(queue)

        cl_objs = []
        for prog_name, kernels in self.kernels.items():
            cl_objs.extend(list(kernels.values()))

        dicts = [
            self.events,
            self.programs,
            self.bufs,
            self.queues
        ]
        for d in dicts:
            cl_objs.extend(list(d.values()))
        cl_objs.extend([self.ctx, self.dev_id])
        for cl_obj in cl_objs:
            cl.release(cl_obj)

    def print(self):
        wrap = terminal_wrapper()
        wrap.subsequent_indent = wrap.initial_indent + INDENT_STR

        data = [
            ("Platform", cl.get_details(self.plat_id)),
            ("Device", cl.get_details(self.dev_id)),
            ("Context", cl.get_details(self.ctx))
        ]
        for name, queue in self.queues.items():
            data.append((f"Queue: {name}", cl.get_details(queue)))

        for prog_name, prog in self.programs.items():
            d = cl.get_details(prog)
            d.update(cl.get_details(prog, self.dev_id))
            data.append((f"Program: {prog_name}", d))

        for prog_name, kernels in self.kernels.items():
            for kern_name, kern in kernels.items():
                data.append((f"Kernel: {kern_name}", cl.get_details(kern)))

        for name, mem in self.bufs.items():
            data.append((f"Buffer: {name}", cl.get_details(mem)))
        for name, ev in self.events.items():
            data.append((f"Event: {name}", cl.get_details(ev)))


        for header, d in data:
            pp_dict_with_header(header, wrap, d)

    # Reading and writing buffers
    def create_input_buffer(self, q_name, buf_name, n_bytes, c_ptr):
        buf = cl.create_buffer(
            self.ctx, cl.MemFlags.CL_MEM_READ_ONLY, n_bytes
        )
        self.bufs[buf_name] = buf
        q = self.queues[q_name]
        ev = cl.enqueue_write_buffer(q, buf, False, 0, n_bytes, c_ptr)

        name = name_event("w", q_name, buf_name)
        self.events[name] = ev
        return ev

    def create_output_buffer(self, b_name, n_bytes):
        buf = cl.create_buffer(
            self.ctx, cl.MemFlags.CL_MEM_WRITE_ONLY, n_bytes
        )
        self.bufs[b_name] = buf
        return buf

    def read_buffer(self, q_name, buf_name, n_bytes, c_ptr):
        q = self.queues[q_name]
        buf = self.bufs[buf_name]
        ev = cl.enqueue_read_buffer(q,  buf, False, 0, n_bytes, c_ptr)
        name = name_event("r", q_name, buf_name)
        self.events[name] = ev
        return ev
