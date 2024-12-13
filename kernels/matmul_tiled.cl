// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>

#define TS_A (TS_N * TS_M)
#define TS_B (TS_M * TS_K)
#define TS_C (TS_N * TS_K)

// | TS_N | TS_M | TS_K | TIME |
// |------|------|------|------|
// |  32  |  32  | 16   | 1.34 |
// |  32  |  32  | 32   | 2.31 |
// |  32  |  32  | 64   | 5.02 |
// |  64  |  32  | 64   | 5.02 |
// |  64  |  64  | 8    | 1.33 |
// |  64  |  64  | 16   | 1.27 |
// |  64  |  64  | 64   | 4.97 |
// |  64  |  64  | 128  | 4.92 |
// | 256  | 256  | 16   | 1.19 |
// | 256  | 512  | 16   | 1.43 |
// | 512  | 256  | 16   | 1.17 |
// | 512  | 512  | 16   | 1.43 |

// This kernel requires input and output data to be tiled.
__attribute__((max_work_group_size(1,1,1)))
__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
matmul_sd(
    uint N, uint M, uint K,
    __global const float * restrict A,
    __global const float * restrict B,
    __global float * restrict C
) {
    for (uint i = 0; i < N * K; i++) {
        C[i] = 0.0;
    }
    uint n_n0 = N / TS_N;
    uint n_m0 = M / TS_M;
    uint n_k0 = K / TS_K;
    for (uint n0 = 0; n0 < n_n0; n0++) {
        for (uint m0 = 0; m0 < n_m0; m0++) {
            float l_A[TS_A];
            int a_addr = n0 * n_m0 * TS_A + m0 * TS_A;
            for (uint i = 0; i < TS_A; i++) {
                l_A[i] = A[a_addr + i];
            }
            for (uint k0 = 0; k0 < n_k0; k0++) {
                __private float l_B[TS_B];
                __private float l_C[TS_C] = {0};
                int b_addr = m0 * n_k0 * TS_B + k0 * TS_B;
                for (uint i = 0; i < TS_B; i++) {
                    l_B[i] = B[b_addr + i];
                }
                for (uint n1 = 0; n1 < TS_N; n1++) {
                    for (uint m1 = 0; m1 < TS_M; m1++) {
                        float a = l_A[n1 * TS_M + m1];
                        for (uint k1 = 0; k1 < TS_K; k1++) {
                            float b = l_B[TS_K * m1 + k1];
                            l_C[TS_K * n1 + k1] += a * b;
                        }
                    }
                }
                int c_addr = n0 * n_k0 * TS_C + k0 * TS_C;
                for (uint i = 0; i < TS_C; i++) {
                    C[c_addr + i] += l_C[i];
                }
            }
        }
    }
}
