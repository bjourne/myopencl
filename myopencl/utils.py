# Copyright (C) 2024, 2026 Björn A. Lindqvist <bjourne@gmail.com>
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

def prettify_info(k, v):
    if k in BYTE_INFOS:
        return [naturalsize(v)]
    elif k in BYTE_INFO_LISTS:
        return [', '.join(naturalsize(v0) for v0 in v)]
    elif isinstance(v, Enum):
        return [v.name]
    elif isinstance(v, str) and "\n" in v:
        return v.strip().split("\n")
    elif k == cl.PlatformInfo.CL_PLATFORM_NUMERIC_VERSION:
        if v is not None:
            v = (v >> 22, (v >> 12) & 0x3ff, v & 0x3ff)
        return [v]
    return [v]

def pp_enum_val(wrapper, key, val):
    val = prettify_info(key, val)
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
    cols = get_terminal_size()[0] if stdout.isatty() else 120
    return TextWrapper(width=cols - 4, subsequent_indent=INDENT_STR)

def can_compile(device_id):
    attr = cl.DeviceInfo.CL_DEVICE_COMPILER_AVAILABLE
    return cl.get_info(attr, device_id)

def is_gpu(device_id):
    attr = cl.DeviceInfo.CL_DEVICE_TYPE
    return cl.get_info(attr, device_id) == cl.DeviceType.CL_DEVICE_TYPE_GPU

def platform_device_pairs():
    s = []
    for p in cl.get_platform_ids():
        s.extend((p, d) for d in cl.get_device_ids(p))
    return s

def format_opts(includes, defines):
    includes = [f"-I {ip}" for ip in includes]
    defines = [f"-D {kv}" for kv in defines]
    opts = [
        "-cl-std=CL2.0",
        "-cl-kernel-arg-info"
    ] + includes + defines
    return " ".join(opts)
