// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
//
// NxM * MxK = NxK
__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
matmul(
    uint N, uint M, uint K,
    __global const float * restrict X,
    __global const float * restrict W,
    __global float * restrict Y
) {
    for (uint i = 0; i < N * K; i++) {
        Y[i] = 0.0;
    }
    for (uint n = 0; n < N; n++) {
        for (uint m = 0; m < M; m++) {
            float v = X[M * n + m];
            for (uint k = 0; k < K; k++) {
                Y[K * n + k] += v * W[K * m + k];
            }
        }
    }
}
