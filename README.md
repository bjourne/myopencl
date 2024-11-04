# myopencl

myopencl is a small Python wrapper for OpenCL based on ctypes.

myopencl is a low-level wrapper and only includes the features I
need. If you want a fully-featured binding use [PyOpenCL](https://documen.tician.de/pyopencl/) instead.

## Installation

    pip install -U . --break-system-packages

You can use the mcl-tool for diagnostics:

    $ mcl-tool list-platforms
    ...
    $ mcl-tool build-kernel kernel/path.cl

## Unit tests

    $ PYTHONPATH=. pytest -vs
