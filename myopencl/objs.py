# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>

# Names:
#   * c - Context
#   * ptr - C pointer
#   * bname - buffer name
#   * qname - queue name
#   * gl_work, lo_work - Global and local work


from collections import defaultdict
from myopencl import MemFlags
from myopencl.utils import INDENT_STR, pp_dict, terminal_wrapper

import ctypes
import myopencl as cl

def pp_dict_with_header(header, wrap, d):
    print(f"== {header} ==")
    pp_dict(wrap, d)

EVENT_CNT = 0
def name_event(*args):
    global EVENT_CNT
    pf = ":".join(str(a) for a in args)
    s = f"{pf}-{EVENT_CNT:03d}"
    EVENT_CNT += 1
    return s

class Context:
    @classmethod
    def from_indexes(cls, platform_idx, device_idx):
        platform_id = cl.get_platform_ids()[platform_idx]
        device_id = cl.get_device_ids(platform_id)[device_idx]
        return Context(platform_id, device_id)

    def __init__(self, platform_id, device_id):
        self.platform_id = platform_id
        self.device_id = device_id
        self.context = cl.create_context(self.device_id)
        self.queues = {}
        self.buffers = {}
        self.events = {}
        self.programs = {}
        self.kernels = defaultdict(dict)

    # Registering
    def register_queue(self, name, props):
        q = cl.create_command_queue_with_properties(
            self.context, self.device_id, props
        )
        self.queues[name] = q

    def register_buffer(self, name, nbytes, flags):
        assert not name in self.buffers
        buf = cl.create_buffer(self.context, flags, nbytes)
        self.buffers[name] = buf

    def register_input_buffer(self, qname, bname, nbytes, cptr):
        self.register_buffer(bname, nbytes, MemFlags.CL_MEM_READ_ONLY)
        return self.write_buffer(qname, bname, nbytes, cptr)

    def register_output_buffer(self, bname, nbytes):
        self.register_buffer(bname, nbytes, MemFlags.CL_MEM_WRITE_ONLY)

    def register_event(self, ev, name):
        self.events[name] = ev
        return ev

    def register_program(self, pname, paths, opts):
        """Since an OpenCL program can consist of multiple source
        files, paths is a list of Path objects.
        """
        dev_id = self.device_id
        ctx = self.context

        suffix = paths[0].suffix
        data = [p.read_bytes() for p in paths]
        if suffix == ".cl":
            source = [d.decode("utf-8") for d in data]
            prog = cl.create_program_with_source(ctx, source)
        else:
            prog = cl.create_program_with_binary(ctx, dev_id, data[0])

        cl.build_program(prog, dev_id, opts, True, True)
        self.programs[pname] = prog

        names = cl.get_kernel_names(prog)
        kernels = {n : cl.create_kernel(prog, n) for n in names}
        self.kernels[pname] = kernels


    # IO
    def write_buffer(self, qname, bname, nbytes, ptr):
        buf = self.buffers[bname]
        q = self.queues[qname]
        ev = cl.enqueue_write_buffer(q, buf, False, 0, nbytes, ptr)
        return self.register_event(ev, name_event("w", qname, bname))

    def read_buffer(self, qname, bname, nbytes, ptr):
        q = self.queues[qname]
        buf = self.buffers[bname]
        ev = cl.enqueue_read_buffer(q,  buf, False, 0, nbytes, ptr)
        return self.register_event(ev, name_event("r", qname, bname))

    # Release everything
    def release_buffer(self, name):
        buf = self.buffers[name]
        cl.release(buf)
        del self.buffers[name]

    def release_all_buffers(self):
        for name in list(self.buffers.keys()):
            self.release_buffer(name)

    def finish_and_release(self):
        for queue in self.queues.values():
            cl.flush(queue)
            cl.finish(queue)
        self.release_all_buffers()
        cl_objs = []
        for prog_name, kernels in self.kernels.items():
            cl_objs.extend(list(kernels.values()))
        dicts = [
            self.events,
            self.programs,
            self.queues
        ]
        for d in dicts:
            cl_objs.extend(list(d.values()))
        cl_objs.extend([self.context, self.device_id])
        for cl_obj in cl_objs:
            cl.release(cl_obj)

    # Kernel intraction
    def enqueue_kernel(self, qname, pname, kname, gl_work, lo_work):
        q = self.queues[qname]
        k = self.kernels[pname][kname]
        # Maybe these events should be registered?
        return cl.enqueue_nd_range_kernel(q, k, gl_work, lo_work)

    def set_kernel_args(self, pname, kname, args):
        kern = self.kernels[pname][kname]
        for i, arg in enumerate(args):
            val = self.buffers.get(arg)
            if val is None:
                ctype, pyval = arg
                val = ctype(pyval)
            cl.set_kernel_arg(kern, i, val)

    def run_kernel(self, qname, pname, kname, gl_work, lo_work, args):
        self.set_kernel_args(pname, kname, args)
        return self.enqueue_kernel(qname, pname, kname, gl_work, lo_work)

    # Printing
    def print(self):
        wrap = terminal_wrapper()
        wrap.subsequent_indent = wrap.initial_indent + INDENT_STR

        data = [
            ("Platform", cl.get_details(self.platform_id)),
            ("Device", cl.get_details(self.device_id)),
            ("Context", cl.get_details(self.context))
        ]
        for name, queue in self.queues.items():
            data.append((f"Queue: {name}", cl.get_details(queue)))

        for prog_name, prog in self.programs.items():
            d = cl.get_details(prog)
            d.update(cl.get_details(prog, self.device_id))
            data.append((f"Program: {prog_name}", d))

        for prog_name, kernels in self.kernels.items():
            for kern_name, kern in kernels.items():
                data.append((f"Kernel: {kern_name}", cl.get_details(kern)))

        for name, mem in self.buffers.items():
            data.append((f"Buffer: {name}", cl.get_details(mem)))
        for name, ev in self.events.items():
            data.append((f"Event: {name}", cl.get_details(ev)))

        for header, d in data:
            pp_dict_with_header(header, wrap, d)
