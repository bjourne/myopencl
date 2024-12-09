// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#define TS    32

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
    for (uint n0 = 0; n0 < N / TS; n0++) {
        for (uint m0 = 0; m0 < M / TS; m0++) {
            for (uint k0 = 0; k0 < K / TS; k0++) {

                float l_A[TS][TS];
                float l_B[TS][TS];
                float l_C[TS][TS];
                for (uint y = 0; y < TS; y++) {
                    for (uint x = 0; x < TS; x++) {
                        l_A[y][x] = A[M * (TS * n0 + y) + TS * m0 + x];
                        l_B[y][x] = B[K * (TS * m0 + y) + TS * k0 + x];
                        l_C[y][x] = C[K * (TS * n0 + y) + TS * k0 + x];
                    }
                }

                for (uint n1 = 0; n1 < TS; n1++) {
                    for (uint m1 = 0; m1 < TS; m1++) {
                        float a = l_A[n1][m1];
                        for (uint k1 = 0; k1 < TS; k1++) {
                            float b = l_B[m1][k1];
                            l_C[n1][k1] += a * b;
                        }
                    }
                }
                for (uint y = 0; y < TS; y++) {
                    for (uint x = 0; x < TS; x++) {
                        C[K * (TS * n0 + y) + TS * k0 + x] = l_C[y][x];
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
    for (uint n = 0; n < N; n++) {
        for (uint m = 0; m < M; m++) {
            float v = A[M * n + m];
            for (uint k = 0; k < K; k++) {
                C[K * n + k] += v * B[K * m + k];
            }
        }
    }
}
