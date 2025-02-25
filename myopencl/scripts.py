# Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
from myopencl.objs import Context
from myopencl.utils import INDENT_STR, pp_dict, terminal_wrapper
from pathlib import Path

import click
import myopencl as cl

def format_opts(includes, defines):
    includes = [f"-I {ip}" for ip in includes]
    defines = [f"-D {kv}" for kv in defines]
    opts = [
        "-cl-std=CL2.0",
        "-cl-kernel-arg-info"
    ] + includes + defines
    return " ".join(opts)


@click.group(
    invoke_without_command = True,
    no_args_is_help=True
)
@click.pass_context
@click.version_option(package_name = "myopencl")
def cli(ctx):
    assert ctx

@cli.command()
def list_platforms():
    """
    List OpenCL platform details.
    """
    wrapper = terminal_wrapper()
    for plat_id in cl.get_platform_ids():
        wrapper.initial_indent = ""
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        details = cl.get_details(plat_id)
        pp_dict(wrapper, details)
        wrapper.initial_indent = INDENT_STR
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        dev_ids = cl.get_device_ids(plat_id)
        for dev_id in dev_ids:
            details = cl.get_details(dev_id)
            pp_dict(wrapper, details)
            cl.release(dev_id)


@cli.command()
@click.argument(
    "filename",
    type = click.Path(exists = True)
)
@click.option(
    "-pi", "--platform-index", default = 0,
    help = "Index of platform to use"
)
@click.option(
    "-I", "includes",
    type = click.Path(exists = True, file_okay = False, dir_okay = True),
    multiple = True,
    help = "Include path",
    default = ()
)
@click.option(
    "-D", "defines",
    multiple = True,
    help = "Definition",
    default = ()
)
def build_program(filename, platform_index, includes, defines):
    """Build an OpenCL program and list its details. If the extension
    of FILENAME Is not .cl it is assumed to be a binary.
    """
    path = Path(filename)

    plat_id = cl.get_platform_ids()[platform_index]
    dev = cl.get_device_ids(plat_id)[0]
    ctx = cl.create_context(dev)

    dev_name = cl.get_info(cl.DeviceInfo.CL_DEVICE_NAME, dev)
    dev_driver = cl.get_info(cl.DeviceInfo.CL_DRIVER_VERSION, dev)
    print(f"OpenCL program: {path}")
    print(f"Device        : {dev_name}")
    print(f"Driver        : {dev_driver}")


    data = path.read_bytes()
    if path.suffix == ".cl":
        src = [data.decode("utf-8")]
        prog = cl.create_program_with_source(ctx, src)
    else:
        prog = cl.create_program_with_binary(ctx, dev, data)
    opts = format_opts(includes, defines)
    cl.build_program(prog, dev, " ".join(opts), True, True)

    wrap = terminal_wrapper()
    pp_dict(wrap, cl.get_details(ctx))
    pp_dict(wrap, cl.get_details(prog, dev))
    pp_dict(wrap, cl.get_details(prog))

    names = cl.get_kernel_names(prog)
    kernels = [cl.create_kernel(prog, name) for name in names]
    for kernel in kernels:
        wrap.initial_indent = INDENT_STR
        pp_dict(wrap, cl.get_details(kernel))

        all_details = cl.get_kernel_args_details(kernel)
        wrap.initial_indent = 2 * INDENT_STR
        for details in all_details:
            pp_dict(wrap, details)

    for kernel in kernels:
        cl.release(kernel)
    cl.release(prog)
    cl.release(ctx)
    cl.release(dev)

@cli.command(context_settings = dict(show_default = True))
@click.option(
    "-pi", "--platform-index", default = 0,
    help = "Index of platform to use"
)
@click.option(
    "-I", "includes",
    type = click.Path(exists = True, file_okay = False, dir_okay = True),
    multiple = True,
    help = "Include path",
    default = ()
)
@click.option(
    "-D", "defines",
    multiple = True,
    help = "Preprocessor defines",
    default = ()
)
@click.argument(
    "filename",
    type = click.Path(exists = True)
)
@click.argument(
    "kernel"
)
@click.argument(
    "arguments",
    nargs = -1,
    required = 1
)
def benchmark_kernel(platform_index, includes, defines, filename, kernel, arguments):
    """
    Load the OpenCL program in FILENAME and run KERNEL with specified
    ARGUMENTS
    """
    from myopencl import so
    ctx = Context.from_indexes(platform_index, 0)
    paths = [Path(filename)]
    opts = format_opts(includes, defines)
    ctx.register_program("main", paths, opts)
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    ctx.register_queue("main", props)

    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    cl_args = []
    for i, arg in enumerate(arguments):
        pf, val = arg.split(":")
        if pf == "buf":
            n_bytes = int(float(val))
            ctx.register_buffer(i, n_bytes, rw_flag)
            cl_args.append(i)
        elif pf == "uint":
            cl_args.append((cl.cl_uint, int(float(val))))
        else:
            assert False
    ev = ctx.run_kernel("main", "main", kernel, [1], None, cl_args)
    cl.wait_for_events([ev])
    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    start = cl.get_info(attr_start, ev)
    end = cl.get_info(attr_end, ev)
    secs = (end - start) * 1.0e-6
    print("%8.2f ms" % secs)
    ctx.finish_and_release()

def main():
    cli(obj={})

if __name__ == "__main__":
    main()
