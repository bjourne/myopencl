// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// Basic AdaptiveAvgPool2d
#include "kernels/ops/utils.h"

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
adaptive_avg_pool2d(
    global const float * restrict X,
    global float * restrict Y,
    uint n_dim, uint y_dim, uint x_dim, uint c_dim,
    uint o_dim
) {
    float stride_y = (float)y_dim / (float)o_dim;
    float stride_x = (float)x_dim / (float)o_dim;
    for (uint n = 0; n < n_dim; n++) {
        for (uint oy = 0; oy < o_dim; oy++) {
            for (uint ox = 0; ox < o_dim; ox++) {
                uint y0 = floor(stride_y * oy);
                uint x0 = floor(stride_x * ox);
                uint y1 = ceil(stride_y * (oy + 1));
                uint x1 = ceil(stride_x * (ox + 1));
                uint cnt = (y1 - y0) * (x1 - x0);
                for (uint c = 0; c < c_dim; c++) {
                    float sum = 0.0;
                    for (uint y = y0; y < y1; y++) {
                        for (uint x = x0; x < x1; x++) {
                            uint a = IDX4D(
                                n_dim, y_dim, x_dim, c_dim,
                                n, y, x, c
                            );
                            sum += X[a];
                        }
                    }
                    uint a = IDX4D(n_dim, o_dim, o_dim, c_dim, n, oy, ox, c);
                    Y[a] = sum / cnt;
                }
            }
        }
    }
}
