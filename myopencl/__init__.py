# Copyright (C) Björn A. Lindqvist 2024
from ctypes import *
from enum import Enum, Flag, IntEnum


########################################################################
# Level -1: ctypes utilities
########################################################################
class Opaque(Structure):
    pass


def OPAQUE_POINTER(name="opaque"):
    class Cls(Opaque):
        pass

    Cls.__name__ = name
    ptr = POINTER(Cls)
    return ptr


########################################################################
# Level 0: Typedefs and function definitions
########################################################################
so = cdll.LoadLibrary("libOpenCL.so")

# Opaque pointers
cl_context = OPAQUE_POINTER("cl_context")
cl_platform_id = OPAQUE_POINTER("cl_platform_id")
cl_device_id = OPAQUE_POINTER("cl_device_id")
cl_command_queue = OPAQUE_POINTER("cl_command_queue")
cl_mem = OPAQUE_POINTER("cl_mem")
cl_program = OPAQUE_POINTER("cl_program")
cl_kernel = OPAQUE_POINTER("cl_kernel")
cl_event = OPAQUE_POINTER("cl_event")
cl_sampler = c_void_p

# Basic types
cl_int = c_int32
cl_uint = c_uint32
cl_ulong = c_uint64
cl_bitfield = cl_ulong


class cl_bool(cl_uint):
    pass


# Build
class cl_build_status(cl_int):
    pass


# Command queue
cl_command_queue_info = cl_uint


class cl_command_queue_properties(cl_bitfield):
    pass


class cl_command_type(cl_uint):
    pass


class cl_command_execution_status(cl_int):
    pass


# Context
cl_context_properties = c_int64
cl_context_info = cl_uint

# Device
cl_device_info = cl_uint


class cl_device_local_mem_type(cl_uint):
    pass


class cl_device_fp_config(cl_bitfield):
    pass


class cl_device_type(cl_bitfield):
    pass


class cl_device_mem_cache_type(cl_uint):
    pass


# Event
cl_event_info = cl_uint

# Kernel
cl_kernel_info = cl_uint
cl_kernel_arg_info = cl_uint
class cl_kernel_arg_address_qualifier(cl_uint):
    pass
class cl_kernel_arg_access_qualifier(cl_uint):
    pass
class cl_kernel_arg_type_qualifier(cl_uint):
    pass


# Memory
class cl_mem_flags(cl_bitfield):
    pass


class cl_mem_object_type(cl_uint):
    pass


cl_mem_info = cl_uint


# Platform
cl_platform_info = cl_uint

# Program
cl_program_info = cl_uint
cl_program_build_info = cl_uint

# Properties
cl_properties = cl_ulong
cl_queue_properties = cl_properties


# Functions here


# Command Queue
so.clCreateCommandQueueWithProperties.restype = cl_command_queue
so.clCreateCommandQueueWithProperties.argtypes = [
    cl_context,
    cl_device_id,
    POINTER(cl_queue_properties),
    POINTER(cl_int),
]

so.clEnqueueNDRangeKernel.restype = cl_int
so.clEnqueueNDRangeKernel.argtypes = [
    cl_command_queue,
    cl_kernel,
    cl_uint,
    POINTER(c_size_t),
    POINTER(c_size_t),
    POINTER(c_size_t),
    cl_uint,
    POINTER(cl_event),
    POINTER(cl_event),
]

so.clEnqueueFillBuffer.restype = cl_int
so.clEnqueueFillBuffer.argtypes = [
    cl_command_queue,
    cl_mem,
    c_void_p,
    c_size_t,
    c_size_t,
    c_size_t,
    cl_uint,
    POINTER(cl_event),
    POINTER(cl_event),
]

so.clEnqueueWriteBuffer.restype = cl_int
so.clEnqueueWriteBuffer.argtypes = [
    cl_command_queue,
    cl_mem,
    cl_bool,
    c_size_t,
    c_size_t,
    c_void_p,
    cl_uint,
    POINTER(cl_event),
    POINTER(cl_event),
]

so.clEnqueueReadBuffer.restype = cl_int
so.clEnqueueReadBuffer.argtypes = [
    cl_command_queue,
    cl_mem,
    cl_bool,
    c_size_t, c_size_t,
    c_void_p,
    cl_uint,
    POINTER(cl_event),
    POINTER(cl_event),
]

so.clFlush.restype = cl_int
so.clFlush.argtypes = [cl_command_queue]

so.clFinish.restype = cl_int
so.clFinish.argtypes = [cl_command_queue]


# Context
so.clCreateContext.restype = cl_context
so.clCreateContext.argtypes = [
    POINTER(cl_context_properties),
    cl_uint,
    POINTER(cl_device_id),
    c_void_p,
    c_void_p,
    POINTER(cl_int),
]

# Device
so.clGetDeviceIDs.restype = cl_int
so.clGetDeviceIDs.argtypes = [
    cl_platform_id,
    cl_device_type,
    cl_uint,
    POINTER(cl_device_id),
    POINTER(cl_uint),
]

# Event
so.clWaitForEvents.restype = cl_int
so.clWaitForEvents.argtypes = [cl_uint, POINTER(cl_event)]

# Kernel
so.clCreateKernel.restype = cl_kernel
so.clCreateKernel.argtypes = [cl_program, c_char_p, POINTER(cl_int)]

so.clSetKernelArg.restype = cl_int
so.clSetKernelArg.argtypes = [cl_kernel, cl_uint, c_size_t, c_void_p]

# Mem
so.clCreateBuffer.restype = cl_mem
so.clCreateBuffer.argtypes = [
    cl_context,
    cl_mem_flags,
    c_size_t,
    c_void_p,
    POINTER(cl_int),
]

# Platform
so.clGetPlatformIDs.restype = cl_int
so.clGetPlatformIDs.argtypes = [
    cl_uint, POINTER(cl_platform_id), POINTER(cl_uint)
]

# Program
so.clCreateProgramWithSource.restype = cl_program
so.clCreateProgramWithSource.argtypes = [
    cl_context,
    cl_uint,
    POINTER(c_char_p),
    POINTER(c_size_t),
    POINTER(cl_int),
]

so.clBuildProgram.restype = cl_int
so.clBuildProgram.argtypes = [
    cl_program,
    cl_uint,
    POINTER(cl_device_id),
    c_char_p,
    c_void_p,
    c_void_p,
]

# Automatically generate level 0 bindings for release functions since
# they all work the same.
TYPE_RELEASERS = {
    cl_command_queue: so.clReleaseCommandQueue,
    cl_context: so.clReleaseContext,
    cl_event: so.clReleaseEvent,
    cl_device_id: so.clReleaseDevice,
    cl_kernel: so.clReleaseKernel,
    cl_mem: so.clReleaseMemObject,
    cl_program: so.clReleaseProgram,
}

for ocl_type, ocl_fun in TYPE_RELEASERS.items():
    setattr(ocl_fun, "restype", cl_int)
    setattr(ocl_fun, "argtypes", [ocl_type])

# Automatically generate level 0 bindings for functions to get object
# info.
TYPE_INFOS = [
    (so.clGetCommandQueueInfo, [cl_command_queue, cl_command_queue_info]),
    (so.clGetDeviceInfo, [cl_device_id, cl_device_info]),
    (so.clGetEventInfo, [cl_event, cl_event_info]),
    (so.clGetKernelInfo, [cl_kernel, cl_kernel_info]),
    (so.clGetKernelArgInfo, [cl_kernel, cl_uint, cl_kernel_arg_info]),
    (so.clGetMemObjectInfo, [cl_mem, cl_mem_info]),
    (so.clGetPlatformInfo, [cl_platform_id, cl_platform_info]),
    (so.clGetProgramBuildInfo, [
        cl_program, cl_device_id, cl_program_build_info
    ]),
    (so.clGetProgramInfo, [cl_program, cl_program_info])
]
for ocl_fun, args in TYPE_INFOS:
    args = args + [c_size_t, c_void_p, POINTER(c_size_t)]
    setattr(ocl_fun, "restype", cl_int)
    setattr(ocl_fun, "argtypes", args)

########################################################################
# Level 1: Pythonic enumerations and bitfieds
########################################################################
class InfoEnum(Enum):
    def __new__(cls, val, tp):
        obj = object.__new__(cls)
        obj._value_ = val
        obj.type = tp
        return obj


class BuildStatus(Enum):
    CL_BUILD_SUCCESS = 0
    CL_BUILD_NONE = -1
    CL_BUILD_ERROR = -2
    CL_BUILD_IN_PROGRESS = -3


class CommandExecutionStatus(Enum):
    CL_COMPLETE = 0x0
    CL_RUNNING = 0x1
    CL_SUBMITTED = 0x2
    CL_QUEUED = 0x3


class CommandQueueInfo(InfoEnum):
    CL_QUEUE_CONTEXT = 0x1090, cl_context
    CL_QUEUE_DEVICE = 0x1091, cl_device_id
    CL_QUEUE_REFERENCE_COUNT = 0x1092, cl_uint
    CL_QUEUE_PROPERTIES = 0x1093, cl_command_queue_properties
    CL_QUEUE_SIZE = 0x1094, cl_uint


class CommandQueueProperties(Flag):
    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = 1 << 0
    CL_QUEUE_PROFILING_ENABLE = 1 << 1
    CL_QUEUE_ON_DEVICE = 1 << 2
    CL_QUEUE_ON_DEVICE_DEFAULT = 1 << 3


class CommandType(Enum):
    CL_COMMAND_NDRANGE_KERNEL = 0x11F0
    CL_COMMAND_TASK = 0x11F1
    CL_COMMAND_NATIVE_KERNEL = 0x11F2
    CL_COMMAND_READ_BUFFER = 0x11F3
    CL_COMMAND_WRITE_BUFFER = 0x11F4
    CL_COMMAND_FILL_BUFFER = 0x1207


class ContextInfo(InfoEnum):
    CL_CONTEXT_REFERENCE_COUNT = 0x1080, cl_uint
    CL_CONTEXT_DEVICES = 0x1081, POINTER(cl_device_id)
    CL_CONTEXT_PROPERTIES = 0x1082, POINTER(cl_context_properties)
    CL_CONTEXT_NUM_DEVICES = 0x1083, cl_uint


class DeviceFpConfig(Flag):
    CL_FP_DENORM = 1 << 0
    CL_FP_INF_NAN = 1 << 1
    CL_FP_ROUND_TO_NEAREST = 1 << 2
    CL_FP_ROUND_TO_ZERO = 1 << 3
    CL_FP_ROUND_TO_INF = 1 << 4
    CL_FP_FMA = 1 << 5
    CL_FP_SOFT_FLOAT = 1 << 6
    CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = 1 << 7


class DeviceInfo(InfoEnum):
    CL_DEVICE_TYPE = 0x1000, cl_device_type
    CL_DEVICE_VENDOR_ID = 0x1001, cl_uint
    CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002, cl_uint
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003, cl_uint
    CL_DEVICE_MAX_WORK_ITEM_SIZES = 0x1005, POINTER(c_size_t)

    CL_DEVICE_ADDRESS_BITS = 0x100D, cl_uint
    CL_DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010, cl_ulong
    CL_DEVICE_IMAGE_SUPPORT = 0x1016, cl_bool
    CL_DEVICE_MAX_PARAMETER_SIZE = 0x1017, c_size_t
    CL_DEVICE_MAX_SAMPLERS = 0x1018, cl_uint
    CL_DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019, cl_uint
    CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A, cl_uint
    CL_DEVICE_SINGLE_FP_CONFIG = 0x101B, cl_device_fp_config
    CL_DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C, cl_device_mem_cache_type
    CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D, cl_uint
    CL_DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E, cl_ulong
    CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F, cl_ulong
    CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020, cl_ulong
    CL_DEVICE_MAX_CONSTANT_ARGS = 0x1021, cl_uint
    CL_DEVICE_LOCAL_MEM_TYPE = 0x1022, cl_device_local_mem_type
    CL_DEVICE_LOCAL_MEM_SIZE = 0x1023, cl_ulong
    CL_DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024, cl_bool
    CL_DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025, c_size_t
    CL_DEVICE_ENDIAN_LITTLE = 0x1026, cl_bool
    CL_DEVICE_AVAILABLE = 0x1027, cl_bool
    CL_DEVICE_COMPILER_AVAILABLE = 0x1028, cl_bool

    CL_DEVICE_NAME = 0x102B, c_char_p
    CL_DEVICE_VENDOR = 0x102C, c_char_p
    CL_DRIVER_VERSION = 0x102D, c_char_p
    CL_DEVICE_PROFILE = 0x102E, c_char_p
    CL_DEVICE_PRINTF_BUFFER_SIZE = 0x1049, c_size_t

    CL_DEVICE_BUILT_IN_KERNELS = 0x103F, c_char_p


class DeviceLocalMemType(Enum):
    CL_LOCAL = 1
    CL_GLOBAL = 2


class DeviceMemCacheType(Enum):
    CL_NONE = 0x0
    CL_READ_ONLY_CACHE = 0x1
    CL_READ_WRITE_CACHE = 0x2


class DeviceType(Enum):
    CL_DEVICE_TYPE_DEFAULT = 1 << 0
    CL_DEVICE_TYPE_CPU = 1 << 1
    CL_DEVICE_TYPE_GPU = 1 << 2
    CL_DEVICE_TYPE_ACCELERATOR = 1 << 3
    CL_DEVICE_TYPE_CUSTOM = 1 << 4
    CL_DEVICE_TYPE_ALL = 0xFFFFFFFF


class ErrorCode(Enum):
    CL_SUCCESS = 0
    CL_DEVICE_NOT_FOUND = -1
    CL_DEVICE_NOT_AVAILABLE = -2
    CL_COMPILER_NOT_AVAILABLE = -3
    CL_MEM_OBJECT_ALLOCATION_FAILURE = -4
    CL_BUILD_PROGRAM_FAILURE = -11
    CL_DEVICE_PARTITION_FAILED = -18
    CL_KERNEL_ARG_INFO_NOT_AVAILABLE = -19
    CL_INVALID_VALUE = -30
    CL_INVALID_PLATFORM = -32
    CL_INVALID_COMMAND_QUEUE = -36
    CL_INVALID_HOST_PTR = -37
    CL_INVALID_MEM_OBJECT = -38
    CL_INVALID_PROGRAM = -44
    CL_INVALID_PROGRAM_EXECUTABLE = -45
    CL_INVALID_KERNEL_NAME = -46
    CL_INVALID_ARG_INDEX = -49
    CL_INVALID_ARG_VALUE = -50
    CL_INVALID_ARG_SIZE = -51
    CL_INVALID_WORK_DIMENSION = -53
    CL_INVALID_WORK_GROUP_SIZE = -54


class EventInfo(InfoEnum):
    CL_EVENT_COMMAND_QUEUE = 0x11D0, cl_command_queue
    CL_EVENT_COMMAND_TYPE = 0x11D1, cl_command_type
    CL_EVENT_REFERENCE_COUNT = 0x11D2, cl_uint
    CL_EVENT_COMMAND_EXECUTION_STATUS = 0x11D3, cl_command_execution_status
    CL_EVENT_CONTEXT = 0x11D4, cl_context


class MemFlags(Enum):
    CL_MEM_READ_WRITE = 1 << 0
    CL_MEM_WRITE_ONLY = 1 << 1
    CL_MEM_READ_ONLY = 1 << 2
    CL_MEM_USE_HOST_PTR = 1 << 3
    CL_MEM_ALLOC_HOST_PTR = 1 << 4
    CL_MEM_COPY_HOST_PTR = 1 << 5


class MemInfo(InfoEnum):
    CL_MEM_TYPE = 0x1100, cl_mem_object_type
    CL_MEM_FLAGS = 0x1101, cl_mem_flags
    CL_MEM_SIZE = 0x1102, c_size_t
    CL_MEM_HOST_PTR = 0x1103, c_void_p
    CL_MEM_MAP_COUNT = 0x1104, cl_uint
    CL_MEM_REFERENCE_COUNT = 0x1105, cl_uint


class MemObjectType(Enum):
    CL_MEM_OBJECT_BUFFER = 0x10F0
    CL_MEM_OBJECT_IMAGE2D = 0x10F1
    CL_MEM_OBJECT_IMAGE3D = 0x10F2
    CL_MEM_OBJECT_IMAGE2D_ARRAY = 0x10F3


class PlatformInfo(InfoEnum):
    CL_PLATFORM_PROFILE = 0x0900, c_char_p
    CL_PLATFORM_VERSION = 0x0901, c_char_p
    CL_PLATFORM_NAME = 0x0902, c_char_p
    CL_PLATFORM_VENDOR = 0x0903, c_char_p
    CL_PLATFORM_EXTENSIONS = 0x0904, c_char_p


class ProgramInfo(InfoEnum):
    CL_PROGRAM_REFERENCE_COUNT = 0x1160, cl_uint
    CL_PROGRAM_CONTEXT = 0x1161, cl_context
    CL_PROGRAM_NUM_DEVICES = 0x1162, cl_uint
    CL_PROGRAM_DEVICES = 0x1163, POINTER(cl_device_id)
    CL_PROGRAM_SOURCE = 0x1164, c_char_p
    CL_PROGRAM_BINARY_SIZES = 0x1165, POINTER(c_size_t)
    CL_PROGRAM_NUM_KERNELS = 0x1167, c_size_t
    CL_PROGRAM_KERNEL_NAMES = 0x1168, c_char_p


class KernelInfo(InfoEnum):
    CL_KERNEL_FUNCTION_NAME = 0x1190, c_char_p
    CL_KERNEL_NUM_ARGS = 0x1191, cl_uint
    CL_KERNEL_REFERENCE_COUNT = 0x1192, cl_uint
    CL_KERNEL_CONTEXT = 0x1193, cl_context
    CL_KERNEL_PROGRAM = 0x1194, cl_program
    CL_KERNEL_ATTRIBUTES = 0x1195, c_char_p

class KernelArgInfo(InfoEnum):
    CL_KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196, cl_kernel_arg_address_qualifier
    CL_KERNEL_ARG_ACCESS_QUALIFIER = 0x1197, cl_kernel_arg_access_qualifier
    CL_KERNEL_ARG_TYPE_NAME = 0x1198, c_char_p
    CL_KERNEL_ARG_TYPE_QUALIFIER = 0x1199, cl_kernel_arg_type_qualifier
    CL_KERNEL_ARG_NAME = 0x119A, c_char_p

class KernelArgAccessQualifier(Enum):
    CL_KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0
    CL_KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1
    CL_KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2
    CL_KERNEL_ARG_ACCESS_NONE = 0x11A3

class KernelArgAddressQualifier(Enum):
    CL_KERNEL_ARG_ADDRESS_GLOBAL = 0x119B
    CL_KERNEL_ARG_ADDRESS_LOCAL = 0x119C
    CL_KERNEL_ARG_ADDRESS_CONSTANT = 0x119D
    CL_KERNEL_ARG_ADDRESS_PRIVATE = 0x119E

class KernelArgTypeQualifier(Flag):
    CL_KERNEL_ARG_TYPE_NONE = 0
    CL_KERNEL_ARG_TYPE_CONST = 1 << 0
    CL_KERNEL_ARG_TYPE_RESTRICT = 1 << 1
    CL_KERNEL_ARG_TYPE_VOLATILE = 1 << 2
    CL_KERNEL_ARG_TYPE_PIPE = 1 << 3

class ProgramBuildInfo(InfoEnum):
    CL_PROGRAM_BUILD_STATUS = 0x1181, cl_build_status
    CL_PROGRAM_BUILD_OPTIONS = 0x1182, c_char_p
    CL_PROGRAM_BUILD_LOG = 0x1183, c_char_p



cl_type_to_python_type = {
    cl_bool: bool,
    cl_build_status: BuildStatus,
    cl_command_execution_status: CommandExecutionStatus,
    cl_command_queue_properties: CommandQueueProperties,
    cl_command_type: CommandType,
    cl_device_mem_cache_type: DeviceMemCacheType,
    cl_device_fp_config: DeviceFpConfig,
    cl_device_local_mem_type: DeviceLocalMemType,
    cl_device_type: DeviceType,
    cl_kernel_arg_access_qualifier : KernelArgAccessQualifier,
    cl_kernel_arg_address_qualifier : KernelArgAddressQualifier,
    cl_kernel_arg_type_qualifier : KernelArgTypeQualifier,
    cl_mem_flags: MemFlags,
    cl_mem_object_type: MemObjectType,
}

TYPE_INFO_GETTERS = {
    (cl_command_queue,) : (so.clGetCommandQueueInfo, CommandQueueInfo),
    (cl_context,) : (so.clGetContextInfo, ContextInfo),
    (cl_device_id,) : (so.clGetDeviceInfo, DeviceInfo),
    (cl_event,) : (so.clGetEventInfo, EventInfo),
    (cl_kernel,) : (so.clGetKernelInfo, KernelInfo),
    (cl_kernel, int) : (so.clGetKernelArgInfo, KernelArgInfo),
    (cl_mem,) : (so.clGetMemObjectInfo, MemInfo),
    (cl_platform_id,) : (so.clGetPlatformInfo, PlatformInfo),
    (cl_program,) : (so.clGetProgramInfo, ProgramInfo),
    (cl_program, cl_device_id) : (so.clGetProgramBuildInfo, ProgramBuildInfo)
}

OPTIONAL_INFO = {
    ErrorCode.CL_INVALID_VALUE : {
        CommandQueueInfo.CL_QUEUE_SIZE : -1
    },
    ErrorCode.CL_INVALID_COMMAND_QUEUE : {
        CommandQueueInfo.CL_QUEUE_SIZE : -1
    },
    ErrorCode.CL_INVALID_PROGRAM_EXECUTABLE : {
        ProgramInfo.CL_PROGRAM_NUM_KERNELS : 0,
        ProgramInfo.CL_PROGRAM_KERNEL_NAMES : ""
    }
}

########################################################################
# Level 2: Functional API
########################################################################
class OpenCLError(Exception):
    def __init__(self, code):
        super().__init__(f"{code} {code.value}")
        self.code = code


def check(err):
    err = ErrorCode(err)
    if err != ErrorCode.CL_SUCCESS:
        raise OpenCLError(err)


def check_last(fun, *args):
    err = cl_int()
    ret = fun(*args, err)
    check(err.value)
    return ret


def cl_info_to_py(tp, buf):
    if tp == c_char_p:
        val = cast(buf, tp).value.decode("utf-8")
    elif hasattr(tp, "contents"):
        to_type = tp._type_
        if issubclass(to_type, Opaque):
            return tp.from_buffer(buf)
        n_el = sizeof(buf) // sizeof(to_type)
        tp_buf = cast(buf, POINTER(to_type * n_el)).contents
        val = list(tp_buf)
    else:
        val = tp.from_buffer(buf).value
    py_tp = cl_type_to_python_type.get(tp)
    val = py_tp(val) if py_tp else val
    return val


# Avoids repeating some tedious code
def size_and_fill(cl_fun, cl_size_tp, cl_el_tp, *args):
    n = cl_size_tp()
    check(cl_fun(*args, 0, None, byref(n)))
    buf = (cl_el_tp * n.value)()
    check(cl_fun(*args, n, buf, None))
    return buf

def get_info(attr, *args):
    tps = tuple(type(a) for a in args)
    cl_fun, info_enum = TYPE_INFO_GETTERS[tps]
    args = args + (attr.value,)
    try:
        buf = size_and_fill(cl_fun, c_size_t, c_byte, *args)
    except OpenCLError as e:
        for ec, opt_attrs in OPTIONAL_INFO.items():
            if e.code == ec and attr in opt_attrs:
                return opt_attrs[attr]
        raise e
    return cl_info_to_py(attr.type, buf)

# Command queue

# This function is bonkers
def create_command_queue_with_properties(ctx, dev, lst):
    n_props = len(lst) + 1
    props = (cl_queue_properties * n_props)()
    for i, val in enumerate(lst):
        if isinstance(val, Enum):
            val = val.value
        props[i] = val
    props[-1] = 0
    return check_last(so.clCreateCommandQueueWithProperties, ctx, dev, props)


def enqueue_nd_range_kernel(queue, kern, global_work, local_work):
    work_dim = len(global_work)

    gl_work = (c_size_t * work_dim)(*global_work)
    lo_work = None
    if local_work:
        lo_work = (c_size_t * work_dim)(*local_work)

    ev = cl_event()
    check(
        so.clEnqueueNDRangeKernel(
            queue, kern, work_dim, None, gl_work, lo_work, 0, None, byref(ev)
        )
    )
    return ev


def enqueue_fill_buffer(queue, mem, pattern, offset, size):
    ev = cl_event()
    check(
        so.clEnqueueFillBuffer(
            queue,
            mem,
            byref(pattern),
            sizeof(pattern),
            offset,
            size,
            0,
            None,
            byref(ev),
        )
    )
    return ev


def enqueue_write_buffer(queue, mem, blocking_write, offset, size, ptr):
    ev = cl_event()
    check(
        so.clEnqueueWriteBuffer(
            queue, mem, blocking_write, offset, size, ptr, 0, None, byref(ev)
        )
    )
    return ev

def enqueue_read_buffer(queue, mem, blocking_read, offset, size, ptr):
    ev = cl_event()
    check(
        so.clEnqueueReadBuffer(
            queue, mem, blocking_read, offset, size, ptr, 0, None, byref(ev)
        )
    )
    return ev


def flush(queue):
    check(so.clFlush(queue))


def finish(queue):
    check(so.clFinish(queue))


# Context
def create_context(dev_id):
    return check_last(so.clCreateContext, None, 1, byref(dev_id), None, None)


# Device
def get_device_ids(plat_id):
    dev_type = DeviceType.CL_DEVICE_TYPE_ALL.value
    return size_and_fill(
        so.clGetDeviceIDs, cl_uint, cl_device_id, plat_id, dev_type
    )


# Event
def wait_for_events(evs):
    n = len(evs)
    evs = (cl_event * n)(*evs)
    check(so.clWaitForEvents(n, evs))


# Kernel
def create_kernel(prog, name):
    name = create_string_buffer(name.encode("utf-8"))
    return check_last(so.clCreateKernel, prog, name)

def set_kernel_arg(kern, i, arg):
    check(so.clSetKernelArg(kern, i, sizeof(arg), byref(arg)))


# Mem
def create_buffer(ctx, flags, n_bytes):
    return check_last(so.clCreateBuffer, ctx, flags.value, n_bytes, None)


# Platform
def get_platform_ids():
    return size_and_fill(so.clGetPlatformIDs, cl_uint, cl_platform_id)



# Program
def create_program_with_source(ctx, src):
    strings = (c_char_p * 1)(src.encode("utf-8"))
    lengths = (c_size_t * 1)(len(src))
    return check_last(so.clCreateProgramWithSource, ctx, 1, strings, lengths)


def build_program(prog, dev, opts, throw, print_log):
    opts = create_string_buffer(opts.encode("utf-8"))
    err = so.clBuildProgram(prog, 1, pointer(dev), opts, None, None)
    if err != 0 and print_log:
        attr = ProgramBuildInfo.CL_PROGRAM_BUILD_LOG
        log = get_program_build_info(prog, dev, attr)
        print(log)
    if throw:
        check(err)

def release(obj):
    rel_fun = TYPE_RELEASERS[type(obj)]
    check(rel_fun(obj))


########################################################################
# Level 3: Holistic functions that calls more than one OpenCL function
# or calls the same OpenCL function in a loop
########################################################################
def get_details(*args):
    tps = tuple(type(a) for a in args)
    _, info_enum = TYPE_INFO_GETTERS[tps]
    return {k : get_info(k, *args) for k in info_enum}

def get_kernel_names(prog):
    names = get_info(
        ProgramInfo.CL_PROGRAM_KERNEL_NAMES,
        prog
    )
    return [n for n in names.split(";") if n]
