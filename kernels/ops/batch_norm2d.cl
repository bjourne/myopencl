// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// The affine transform has to be precomputed on the host:
//
//      mul = weight / sqrt(var + 1e-5)
//      add = -mean * mul + bias
#include "kernels/ops/utils.h"

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
batch_norm2d(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict mul,
    global const float * restrict add,
    uint n_dim, uint c_dim, uint relu
) {

    for (uint n = 0; n < n_dim; n++) {
        for (uint c = 0; c < c_dim; c++) {
            uint i = c_dim * n + c;
            Y[i] = MAYBE_RELU(mul[c] * X[i] + add[c], relu);
        }
    }
}
