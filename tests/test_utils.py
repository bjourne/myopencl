# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
import myopencl as cl

from myopencl.utils import format_opts, prettify_info

def test_prettify_info():
    attr = cl.PlatformInfo.CL_PLATFORM_NUMERIC_VERSION
    val = prettify_info(attr, 12582912)
    assert val == [(3, 0, 0)]

def test_format_opts():
    s = format_opts([], [])
    assert isinstance(s, str)
