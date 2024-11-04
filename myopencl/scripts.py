# Copyright (C) Björn A. Lindqvist 2024
from enum import Enum
from humanize import naturalsize
from os import get_terminal_size
from textwrap import TextWrapper

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
    base_fmt = "%%-%ds: %%s" % KEY_LEN
    if isinstance(val, Enum):
        val = val.name
    s = base_fmt % (key.name, val)
    print(wrapper.fill(s))


def pp_dict(wrapper, d):
    for key, val in d.items():
        pp_enum_val(wrapper, key, val)
    print()


def list_platforms():
    cols, _ = get_terminal_size()
    wrapper = TextWrapper(width=cols - 4, subsequent_indent=INDENT_STR)
    for plat_id in cl.get_platform_ids():
        wrapper.initial_indent = ""
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        details = cl.get_platform_details(plat_id)
        pp_dict(wrapper, details)
        wrapper.initial_indent = INDENT_STR
        wrapper.subsequent_indent = wrapper.initial_indent + INDENT_STR
        for dev_id in cl.get_device_ids(plat_id):
            details = cl.get_device_details(dev_id)
            pp_dict(wrapper, details)


if __name__ == "__main__":
    list_platforms()
