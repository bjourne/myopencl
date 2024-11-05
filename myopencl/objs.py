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

    def add_queue(self, name):
        q = cl.create_command_queue_with_properties(
            self.ctx, self.dev_id, []
        )
        self.queues[name] = q

    def add_input_buffer(self, q_name, buf_name, n_bytes, c_ptr):
        buf = cl.create_buffer(
            self.ctx, cl.MemFlags.CL_MEM_READ_ONLY, n_bytes
        )
        q = self.queues[q_name]
        ev = cl.enqueue_write_buffer(q, buf, False, 0, n_bytes, c_ptr)
        self.bufs[buf_name] = buf
        name = name_event("write", q_name, buf_name)
        self.events[name] = ev
        return ev

    def read_buffer(self, q_name, buf_name, n_bytes, c_ptr):
        q = self.queues[q_name]
        buf = self.bufs[buf_name]
        ev = cl.enqueue_read_buffer(q,  buf, False, 0, n_bytes, c_ptr)
        name = name_event("read", q_name, buf_name)
        self.events[name] = ev
        return ev

    def finish_release(self):
        for queue in self.queues.values():
            cl.flush(queue)
            cl.finish(queue)
        cl_objs = list(self.events.values()) \
            + list(self.bufs.values()) \
            + list(self.queues.values()) \
            + [self.ctx, self.dev_id]
        for cl_obj in cl_objs:
            cl.release(cl_obj)

    def print(self):
        wrap = terminal_wrapper()
        wrap.subsequent_indent = wrap.initial_indent + INDENT_STR

        data = [
            ("Platform", cl.get_platform_details(self.plat_id)),
            ("Device", cl.get_device_details(self.dev_id)),
            ("Context", cl.get_context_details(self.ctx))
        ]
        for name, queue in self.queues.items():
            data.append((f"Queue: {name}", cl.get_command_queue_details(queue)))

        for name, mem in self.bufs.items():
            data.append((f"Buffer: {name}", cl.get_mem_object_details(mem)))
        for name, ev in self.events.items():
            data.append((f"Event: {name}", cl.get_event_details(ev)))

        for header, d in data:
            pp_dict_with_header(header, wrap, d)
