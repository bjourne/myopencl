# MyOpenCL

MyOpenCL is a small Python wrapper for OpenCL based on ctypes. I wrote
MyOpenCL primarily because I wanted a very minimal, low-level wrapper,
but also because I wanted to verify whether ctypes is good enough for
"real" wrappers.

If you want a fully-featured binding use
[PyOpenCL](https://documen.tician.de/pyopencl/) instead.

## Installation

    git clone https://github.com/bjourne/myopencl
    cd myopencl
    pip install -U . --break-system-packages

It installes `mcl-tool` to be used for diagnostics:

    $ mcl-tool list-platforms
    ...
    $ mcl-tool build-program kernel/path.cl

## Documentation

The [test suites](https://github.com/bjourne/myopencl/tree/main/tests)
serves as the main source of documentation. The structure of the
package is as follows:

* `myopencl` - core bindings
* `myopencl.objs` - a higher-level interface based around one `Context` class

## Unit tests

    $ PYTHONPATH=. pytest -vs tests

## Future work

* There are some tricky attributes `cl.get_info` currently doesn't
  handle.
