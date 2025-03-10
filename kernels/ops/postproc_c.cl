#include "kernels/ops/sysarr.h"
#include "kernels/ops/utils.h"

#if IS_AOC==1
#define STORE(a, i, v)       __pipelined_store(&a[(i)], (v))
#define LOAD(a, i)           __pipelined_load(&a[i])
#else
#define STORE(a, i, v)       a[(i)] = v
#define LOAD(a, i)           a[(i)]
#endif

#define BLOCK_K_CHUNKS      (BLOCK_K / CHAN_ALIGN)

// Untranspose, untile, and unpad.
// This variant (double unroll, scalar, pipe+pipe) = 10.6 ms @
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
postproc_c(
    global const float* restrict X,
    global float* restrict Y,
    uint n_dim, uint k_dim
) {

    ASSERT(k_dim % CHAN_ALIGN == 0);
    ASSERT(BLOCK_K % CHAN_ALIGN == 0);
    uint n_blocks = ALIGN_TO(n_dim, BLOCK_N);
    uint k_blocks = ALIGN_TO(k_dim, BLOCK_K);
    uint k_dim_chunks = k_dim / CHAN_ALIGN;

#pragma ivdep
    for (uint n = 0; n < n_dim; n++) {
        uint n0 = n / BLOCK_N;
        uint n1 = n % BLOCK_N;
#pragma ivdep
        for (uint k0 = 0; k0 < k_blocks; k0++) {
            uint src_base = IDX4D(
                n_blocks, k_blocks, BLOCK_N, BLOCK_K_CHUNKS,
                n0, k0, n1, 0
            );
            uint dst_base = n * k_dim_chunks + BLOCK_K_CHUNKS * k0;
            float tmp[PE_S][PE_S];
#pragma unroll
            for (uint k1 = 0; k1 < PE_S; k1++) {
#pragma unroll
                for (uint k2 = 0; k2 < PE_S; k2++) {
                    uint ofs = IDX2D(PE_S, PE_S, k1, k2);
                    float v = LOAD(X, CHAN_ALIGN * src_base + ofs);
                    tmp[k2][k1] = v;
                }
            }
#pragma unroll
            for (uint k1 = 0; k1 < PE_S; k1++) {
#pragma unroll
                for (uint k2 = 0; k2 < PE_S; k2++) {
                    uint ofs = IDX2D(PE_S, PE_S, k1, k2);
                    float v = tmp[k1][k2];
                    STORE(Y, CHAN_ALIGN * dst_base + ofs, v);
                }
            }
        }
    }
}
