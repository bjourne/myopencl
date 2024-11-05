# Copyright (C) 2024 Björn A. Lindqvist
from myopencl.utils import INDENT_STR, pp_dict, terminal_wrapper

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

class MyContext:
    def __init__(self, plat_idx, dev_idx):
        self.plat_id = cl.get_platform_ids()[plat_idx]
        self.dev_id = cl.get_device_ids(self.plat_id)[dev_idx]
        self.ctx = cl.create_context(self.dev_id)
        self.queues = {}
        self.bufs = {}
        self.events = {}
        self.programs = {}
        self.kernels = {}

    def create_program_and_kernels(self, name, path):
        source = path.read_text("utf-8")
        prog = cl.create_program_with_source(self.ctx, source)
        opts = "-Werror -cl-std=CL2.0"
        cl.build_program(prog, self.dev_id, opts, True, True)
        self.programs[name] = prog

        for kname in cl.get_kernel_names(prog):
            kern = cl.create_kernel(prog, kname)
            self.kernels[f"{name}/{kname}"] = kern

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
        dicts = [
            self.events,
            self.kernels,
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

        for name, prog in self.programs.items():
            d = cl.get_details(prog)
            d.update(cl.get_details(prog, self.dev_id))
            data.append((f"Program: {name}", d))
        for name, kern in self.kernels.items():
            data.append((f"Kernel: {name}", cl.get_details(kern)))
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
        q = self.queues[q_name]
        ev = cl.enqueue_write_buffer(q, buf, False, 0, n_bytes, c_ptr)
        self.bufs[buf_name] = buf
        name = name_event("w", q_name, buf_name)
        self.events[name] = ev
        return ev

    def read_buffer(self, q_name, buf_name, n_bytes, c_ptr):
        q = self.queues[q_name]
        buf = self.bufs[buf_name]
        ev = cl.enqueue_read_buffer(q,  buf, False, 0, n_bytes, c_ptr)
        name = name_event("r", q_name, buf_name)
        self.events[name] = ev
        return ev
