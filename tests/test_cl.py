# Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#
# Names:
#   * bname - buffer name
#   * c - MyContext
#   * cptr - C pointer
#   * ev - event
#   * ksize - kernel size
#   * nbytes - number of bytes
#   * qname - queue name
#   * x - tensor or numpy array

from myopencl.objs import MyContext
from myopencl.utils import platform_device_pairs
from pytest import mark

import myopencl as cl

PAIRS = platform_device_pairs()

########################################################################
# Tests: low-level
########################################################################
def test_release_none():
    try:
        cl.release(None)
        assert False
    except KeyError:
        assert True

@mark.parametrize("platform_id, device_id", PAIRS)
def test_release_device(platform_id, device_id):
    dev_id = cl.get_device_ids(platform_id)[0]
    cl.release(dev_id)

@mark.parametrize("platform_id, device_id", PAIRS)
def test_get_queue_size(platform_id, device_id):
    ctx = cl.create_context(device_id)
    q = cl.create_command_queue_with_properties(ctx, device_id, [])
    attr = cl.CommandQueueInfo.CL_QUEUE_SIZE
    assert cl.get_info(attr, q) == 0
    cl.release(ctx)
    cl.release(device_id)

########################################################################
# Tests: higher level
########################################################################
@mark.parametrize("platform_id,device_id", PAIRS)
def test_print_objs(platform_id, device_id):
    c = MyContext(platform_id, device_id)
    c.print()
    c.finish_and_release()


def test_from_indexes():
    for i, platform_id in enumerate(cl.get_platform_ids()):
        for j, _ in enumerate(cl.get_device_ids(platform_id)):
            c = MyContext.from_indexes(i, j)
            c.finish_and_release()
