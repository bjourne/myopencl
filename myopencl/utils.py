# Copyright (C) 2024 Björn A. Lindqvist
from enum import Enum
from humanize import naturalsize
from os import get_terminal_size
from sys import stdout
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

BYTE_INFO_LISTS = {
    cl.ProgramInfo.CL_PROGRAM_BINARY_SIZES
}

def pp_enum_val(wrapper, key, val):
    if key in BYTE_INFOS:
        val = naturalsize(val)
    if key in BYTE_INFO_LISTS:
        val = ', '.join(naturalsize(v) for v in val)
    if isinstance(val, Enum):
        val = val.name
    if isinstance(val, str) and "\n" in val:
        val = val.strip()
        val = val.split("\n")
    else:
        val = [val]

    base_fmt = f"%-{KEY_LEN}s: %s"
    s = base_fmt % (key.name, val[0])
    print(wrapper.fill(s))
    more_pf = " " * (KEY_LEN + 2)
    for line in val[1:]:
        print(f"{more_pf}{line}")


def pp_dict(wrapper, d):
    for key, val in d.items():
        pp_enum_val(wrapper, key, val)
    print()

def terminal_wrapper():
    cols = get_terminal_size()[0] if stdout.isatty() else 72
    return TextWrapper(width=cols - 4, subsequent_indent=INDENT_STR)
