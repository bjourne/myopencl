// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// Kernel that fuses im2col with tiling
#include "kernels/ops/utils.h"
#include "kernels/ops/sysarr.h"

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
im2col_and_tile(
    global const float * restrict X,
    global float * restrict Y,
    uint bs_dim,
    uint iy_dim, uint ix_dim, uint c_dim,
    uint k_dim,
    uint pad
) {

    uint oy_dim = iy_dim + 2 * pad - k_dim + 1;
    uint ox_dim = ix_dim + 2 * pad - k_dim + 1;

    // Not padded
    uint size_n = bs_dim * iy_dim * ix_dim;
    uint size_m = k_dim * k_dim * c_dim;

    uint n_blocks = ALIGN_TO(size_n, BLOCK_N);
    uint m_blocks = MAX(ALIGN_TO(size_m, BLOCK_M), 3);

    uint n_els = n_blocks * m_blocks * BLOCK_N * BLOCK_M;
#pragma unroll 16
    for (uint i = 0; i < n_els; i++) {
        Y[i] = 0;
    }
    for (uint bs = 0; bs < bs_dim; bs++) {
        for (uint oy = 0; oy < oy_dim; oy++) {
            for (uint ox = 0; ox < ox_dim; ox++) {
#pragma ivdep
                for (uint fy = 0; fy < k_dim; fy++) {
#pragma ivdep
                    for (uint fx = 0; fx < k_dim; fx++) {
#pragma ivdep
#pragma unroll 16
                        for (uint c = 0; c < c_dim; c++) {
                            int iy = oy + fy - pad;
                            int ix = ox + fx - pad;
                            float v = 0;
                            if (
                                0 <= iy && iy < iy_dim &&
                                0 <= ix && ix < ix_dim
                            ) {
                                uint src = IDX4D(
                                    bs_dim, iy_dim, ix_dim,
                                    c_dim, bs, iy, ix, c
                                );
                                v = X[src];
                            }
                            uint n = IDX3D(bs_dim, oy_dim, ox_dim, bs, oy, ox);
                            uint m = IDX3D(k_dim, k_dim, c_dim, fy, fx, c);
                            uint n0 = n / BLOCK_N;
                            uint m0 = m / BLOCK_M;
                            uint n1 = n % BLOCK_N;
                            uint m1 = m % BLOCK_M;
                            uint dst = IDX4D(
                                n_blocks, m_blocks, BLOCK_N, BLOCK_M,
                                n0, m0, n1, m1
                            );
                            Y[dst] = v;
                        }
                    }
                }
            }
        }
    }
}
