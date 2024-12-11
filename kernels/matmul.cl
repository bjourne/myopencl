// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>

// NxM * MxK = NxK
__attribute__((max_work_group_size(1,1,1)))
__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
matmul_tiled(
    uint N, uint M, uint K,
    __global const float * restrict A,
    __global const float * restrict B,
    __global float * restrict C
) {
    for (uint i = 0; i < N * K; i++) {
        C[i] = 0.0;
    }
    for (uint n0 = 0; n0 < N / TS_N; n0++) {
        for (uint m0 = 0; m0 < M / TS_M; m0++) {
            for (uint k0 = 0; k0 < K / TS_K; k0++) {

                float l_A[TS_N][TS_M];
                float l_B[TS_M][TS_K];
                float l_C[TS_N][TS_K];
                for (uint y = 0; y < TS_N; y++) {
                    for (uint x = 0; x < TS_M; x++) {
                        l_A[y][x] = A[M * (TS_N * n0 + y) + TS_M * m0 + x];
                    }
                }
                for (uint y = 0; y < TS_M; y++) {
                    for (uint x = 0; x < TS_K; x++) {
                        l_B[y][x] = B[K * (TS_M * m0 + y) + TS_K * k0 + x];
                    }
                }
                for (uint y = 0; y < TS_N; y++) {
                    for (uint x = 0; x < TS_K; x++) {
                        l_C[y][x] = C[K * (TS_N * n0 + y) + TS_K * k0 + x];
                    }
                }
                for (uint n1 = 0; n1 < TS_N; n1++) {
                    for (uint m1 = 0; m1 < TS_M; m1++) {
                        float a = l_A[n1][m1];
                        for (uint k1 = 0; k1 < TS_K; k1++) {
                            float b = l_B[m1][k1];
                            l_C[n1][k1] += a * b;
                        }
                    }
                }
                for (uint y = 0; y < TS_N; y++) {
                    for (uint x = 0; x < TS_K; x++) {
                        C[K * (TS_N * n0 + y) + TS_K * k0 + x] = l_C[y][x];
                    }
                }
            }
        }
    }
}

__attribute__((max_work_group_size(1,1,1)))
__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
matmul_naive(
    uint N, uint M, uint K,
    __global const float * restrict A,
    __global const float * restrict B,
    __global float * restrict C
) {
    for (uint i = 0; i < N * K; i++) {
        C[i] = 0.0;
    }
    for (uint n = 0; n < N; n++) {
        for (uint m = 0; m < M; m++) {
            float v = A[M * n + m];
            for (uint k = 0; k < K; k++) {
                C[K * n + k] += v * B[K * m + k];
            }
        }
    }
}
