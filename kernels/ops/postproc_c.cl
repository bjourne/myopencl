#include "kernels/ops/sysarr.h"
#include "kernels/ops/utils.h"

#if IS_AOC==1
#define STORE(a, i, v)       __pipelined_store(&a[(i)], (v))
#define LOAD(a, i)           __pipelined_load(&a[i])
#else
#define STORE(a, i, v)       a[(i)] = v
#define LOAD(a, i)           a[(i)]
#endif

// Untranspose, untile, and unpad.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
postproc_c(
    global type* restrict X,
    global type* restrict Y,
    uint n_dim, uint k_dim,
    uint n_blocks, uint k_blocks
) {
    n_blocks = ALIGN_TO(n_dim, BLOCK_N);
    k_blocks = ALIGN_TO(k_dim, BLOCK_K);

    for (uint n = 0; n < n_dim; n++) {
        uint n0 = n / BLOCK_N;
        uint n1 = n % BLOCK_N;
        for (uint k0 = 0; k0 < k_blocks; k0++) {
            uint src_base = IDX4D(
                n_blocks, k_blocks, BLOCK_N, BLOCK_K,
                n0, k0, n1, 0);
            float tmp[PE_S][PE_S];
#pragma unroll
            for (uint k1 = 0; k1 < PE_S; k1++) {
#pragma unroll
                for (uint k2 = 0; k2 < PE_S; k2++) {
                    uint src_ofs = IDX2D(PE_S, PE_S, k1, k2);
                    tmp[k2][k1] = LOAD(X, src_base + src_ofs);
                }
            }
            uint dst_base = n * k_dim + BLOCK_K * k0;
#pragma unroll
            for (uint k1 = 0; k1 < PE_S; k1++) {
#pragma unroll
                for (uint k2 = 0; k2 < PE_S; k2++) {
                    uint dst_ofs = IDX2D(PE_S, PE_S, k1, k2);
                    STORE(Y, dst_base + dst_ofs, tmp[k1][k2]);
                }
            }
        }
    }
}
