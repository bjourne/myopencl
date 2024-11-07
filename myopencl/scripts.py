# Copyright (C) 2024 Björn A. Lindqvist
from myopencl.utils import INDENT_STR, pp_dict, terminal_wrapper
from pathlib import Path

import click
import myopencl as cl

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
@click.argument("filename", type = click.Path(exists = True))
@click.option(
    "-pi", "--platform-index", default = 0,
    help = "Index of platform to use."
)
@click.option(
    "-I", "include_paths",
    type = click.Path(exists = True, file_okay = False, dir_okay = True),
    multiple = True,
    help = "Include path.",
    default = ()
)
def build_program(filename, platform_index, include_paths):
    """Build an OpenCL program and list its details."""
    path = Path(filename)

    source = path.read_text("utf-8")

    plat_id = cl.get_platform_ids()[platform_index]
    dev = cl.get_device_ids(plat_id)[0]
    ctx = cl.create_context(dev)

    dev_name = cl.get_info(cl.DeviceInfo.CL_DEVICE_NAME, dev)
    dev_driver = cl.get_info(cl.DeviceInfo.CL_DRIVER_VERSION, dev)
    print(f"OpenCL program: {path}")
    print(f"Device        : {dev_name}")
    print(f"Driver        : {dev_driver}")

    prog = cl.create_program_with_source(ctx, source)
    opts = [
        "-Werror",
        "-cl-std=CL2.0",
        "-cl-kernel-arg-info"
    ] + [f"-I {ip}" for ip in include_paths]
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

def main():
    cli(obj={})

if __name__ == "__main__":
    main()
