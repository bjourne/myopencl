# myopencl

myopencl is a small Python wrapper for OpenCL based on ctypes.

myopencl is a low-level wrapper and only includes the features I
need. If you want a fully-featured binding use
[PyOpenCL](https://documen.tician.de/pyopencl/) instead. Look at the
[test suites](https://github.com/bjourne/myopencl/tree/main/tests) for
usage. There is no other documentation.

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

* Make `cl.get_info` also handle really tricky attributes.
