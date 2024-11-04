# Copyright (C) Björn A. Lindqvist 2024
from enum import Enum
from humanize import naturalsize
from os import get_terminal_size
from pathlib import Path
from textwrap import TextWrapper

import click
import myopencl as cl

KEY_LEN = 40
INDENT_STR = " " * 4


BYTE_INFOS = {
    cl.DeviceInfo.CL_DEVICE_PRINTF_BUFFER_SIZE,
    cl.DeviceInfo.CL_DEVICE_MAX_MEM_ALLOC_SIZE,
    cl.DeviceInfo.CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
    cl.DeviceInfo.CL_DEVICE_LOCAL_MEM_SIZE,
    cl.DeviceInfo.CL_DEVICE_GLOBAL_MEM_SIZE,
    cl.DeviceInfo.CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
}


def pp_enum_val(wrapper, key, val):
    if key in BYTE_INFOS:
        val = naturalsize(val)
    if isinstance(val, Enum):
        val = val.name
    if isinstance(val, str) and "\n" in val:
        val = val.strip()
        val = val.split("\n")
    else:
        val = [val]

    base_fmt = "%%-%ds: %%s" % KEY_LEN
    s = base_fmt % (key.name, val[0])
    print(wrapper.fill(s))
    more_pf = " " * (KEY_LEN + 2)
    for line in val[1:]:
        print("%s%s" % (more_pf, line))


def pp_dict(wrapper, d):
    for key, val in d.items():
        pp_enum_val(wrapper, key, val)
    print()

def terminal_wrapper():
    cols, _ = get_terminal_size()
    return TextWrapper(width=cols - 4, subsequent_indent=INDENT_STR)


@click.group(
    invoke_without_command = True,
    no_args_is_help=True
)
@click.pass_context
@click.version_option(package_name = "myopencl")
def cli(ctx):
    pass

@cli.command()
def list_platforms():
    """
    List OpenCL platform details.
    """
    wrapper = terminal_wrapper()
    for plat_id in cl.get_platform_ids():
        wrapper.initial_indent = ""
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        details = cl.get_platform_details(plat_id)
        pp_dict(wrapper, details)
        wrapper.initial_indent = INDENT_STR
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        dev_ids = cl.get_device_ids(plat_id)
        for dev_id in dev_ids:
            details = cl.get_device_details(dev_id)
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

    source = path.read_text()

    plat_id = cl.get_platform_ids()[platform_index]
    dev = cl.get_device_ids(plat_id)[0]
    ctx = cl.create_context(dev)

    dev_name = cl.get_device_info(dev, cl.DeviceInfo.CL_DEVICE_NAME)
    print("OpenCL program: %s" % path)
    print("Device        : %s" % dev_name)

    prog = cl.create_program_with_source(ctx, source)

    opts = [
        '-Werror',
        '-cl-std=CL2.0'
    ] + ['-I %s' % ip for ip in include_paths]
    cl.build_program(prog, dev, " ".join(opts), True, True)

    wrapper = terminal_wrapper()
    pp_dict(wrapper, cl.get_context_details(ctx))
    pp_dict(wrapper, cl.get_program_build_details(prog, dev))
    pp_dict(wrapper, cl.get_program_details(prog))
    names = cl.get_program_info(
        prog,
        cl.ProgramInfo.CL_PROGRAM_KERNEL_NAMES
    )
    names = names.split(";")

    kernels = [cl.create_kernel(prog, name) for name in names]
    for kernel in kernels:
        wrapper.initial_indent = INDENT_STR
        pp_dict(wrapper, cl.get_kernel_details(kernel))
        n_args = cl.get_kernel_info(kernel, cl.KernelInfo.CL_KERNEL_NUM_ARGS)
        wrapper.initial_indent = 2 * INDENT_STR
        for i in range(n_args):
            pp_dict(wrapper, cl.get_kernel_arg_details(kernel, i))

    for kernel in kernels:
        cl.release(kernel)
    cl.release(prog)
    cl.release(ctx)
    cl.release(dev)

def main():
    cli(obj={})

if __name__ == "__main__":
    main()
