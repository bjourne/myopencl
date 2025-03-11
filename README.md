# MyOpenCL

MyOpenCL is a small Python wrapper for OpenCL based on ctypes. I wrote
MyOpenCL primarily because I wanted a very minimal, low-level wrapper,
but also because I wanted to verify whether ctypes is good enough for
"real" wrappers. I think it is. There is not a lot of documentation
other than the [test
suites](https://github.com/bjourne/myopencl/tree/main/tests).

If you want a fully-featured binding use
[PyOpenCL](https://documen.tician.de/pyopencl/) instead.

## Installation

    git clone https://github.com/bjourne/myopencl
    cd myopencl
    pip install -U . --break-system-packages

You can use the installeed mcl-tool for diagnostics:

    $ mcl-tool list-platforms
    ...
    $ mcl-tool build-kernel kernel/path.cl

## Unit tests

    $ PYTHONPATH=. pytest -vs tests

## Future work

* There are some tricky attributes `cl.get_info` currently doesn't
  handle.
