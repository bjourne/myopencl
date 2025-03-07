#include "kernels/ops/utils.h"

#if IS_AOC==1
#define STORE(a, i, v)       __pipelined_store(&a[(i)], (v))
#define LOAD(a, i)           __pipelined_load(&a[i])
#else
#define STORE(a, i, v)       a[(i)] = v
#define LOAD(a, i)           a[(i)]
#endif

#define CHUNK       32

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
relu(
    global const float * restrict X,
    global float * restrict Y,
    uint n
) {
    for (uint i = 0; i < n / CHUNK; i++) {
#pragma unroll
        for (uint j = 0; j < CHUNK; j++) {
            float v = LOAD(X, CHUNK * i + j);
            v = MAX(v, 0);
            STORE(Y, CHUNK * i + j, v);
        }
    }
}
