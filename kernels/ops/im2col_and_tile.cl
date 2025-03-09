// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// Kernel that fuses im2col with tiling
#include "kernels/ops/utils.h"
#include "kernels/ops/sysarr.h"

#define CHAN_SIZE       16
#define BLOCK_M_CHUNKS  (BLOCK_M / CHAN_SIZE)

#if IS_AOC==1
#define STORE(a, i, v)       __pipelined_store(&a[(i)], (v))
#define LOAD(a, i)           __burst_coalesced_load(&a[i])
#else
#define STORE(a, i, v)       a[(i)] = v
#define LOAD(a, i)           a[(i)]
#endif

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
im2col_and_tile(
    global const float * restrict X,
    global float * restrict Y,
    uint bs_dim,
    uint iy_dim, uint ix_dim, uint c_dim,
    uint k_dim, uint pad, uint stride
) {
    ASSERT(BLOCK_M % CHAN_SIZE == 0);
    ASSERT(c_dim % CHAN_SIZE == 0);

    uint oy_dim = WIN_COUNT(iy_dim, k_dim, stride, pad);
    uint ox_dim = WIN_COUNT(ix_dim, k_dim, stride, pad);

    // Unpadded size of the output matrix
    uint n_dim = bs_dim * oy_dim * ox_dim;
    uint m_dim = k_dim * k_dim * c_dim;

    uint n_blocks = ALIGN_TO(n_dim, BLOCK_N);
    uint m_blocks = MAX(ALIGN_TO(m_dim, BLOCK_M), 3);

    uint c_dim_chunks = c_dim / CHAN_SIZE;
    uint m_dim_chunks = m_dim / CHAN_SIZE;

    uint n_els = n_blocks * m_blocks * BLOCK_N * BLOCK_M;
#pragma unroll CHAN_SIZE
    for (uint i = 0; i < n_els; i++) {
        STORE(Y, i, 0);
    }
#pragma ivdep
    for (uint bs = 0; bs < bs_dim; bs++) {
#pragma ivdep
        for (uint oy = 0; oy < oy_dim; oy++) {
#pragma ivdep
            for (uint ox = 0; ox < ox_dim; ox++) {
#pragma ivdep
                for (uint fy = 0; fy < k_dim; fy++) {
#pragma ivdep
                    for (uint fx = 0; fx < k_dim; fx++) {
#pragma ivdep
                        for (uint c = 0; c < c_dim_chunks; c++) {
                            int iy = stride * oy + fy - pad;
                            int ix = stride * ox + fx - pad;
                            float v[CHAN_SIZE] = {0};
                            if (
                                0 <= iy && iy < iy_dim &&
                                0 <= ix && ix < ix_dim
                            ) {
                                uint src = IDX4D(
                                    bs_dim, iy_dim, ix_dim, c_dim_chunks,
                                    bs, iy, ix, c
                                );
#pragma unroll CHAN_SIZE
                                for (uint i = 0; i < CHAN_SIZE; i++)   {
                                    v[i] = LOAD(X, CHAN_SIZE * src + i );
                                }
                            }
                            uint n = IDX3D(bs_dim, oy_dim, ox_dim, bs, oy, ox);
                            uint m = IDX3D(k_dim, k_dim, c_dim_chunks, fy, fx, c);
                            uint n0 = n / BLOCK_N;
                            uint n1 = n % BLOCK_N;
                            uint m0 = m / BLOCK_M_CHUNKS;
                            uint m1 = m % BLOCK_M_CHUNKS;
                            uint dst = IDX4D(
                                n_blocks, m_blocks,
                                BLOCK_N, BLOCK_M_CHUNKS,
                                n0, m0, n1, m1
                            );
#pragma unroll CHAN_SIZE
                            for (uint i = 0; i < CHAN_SIZE; i++) {
                                STORE(Y, CHAN_SIZE * dst + i, v[i]);
                            }
                        }
                    }
                }
            }
        }
    }
}
