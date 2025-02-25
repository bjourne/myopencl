#include "kernels/ops/utils.h"

// X:   (n, iy, ix, ic)
// Y:   (n * oy * ox, fy * fx * ic)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
conv2d_im2col(
    global const float * restrict X,
    global float * restrict Y,
    uint n_dim,
    uint iy_dim, uint ix_dim, uint c_dim,
    uint k_size,
    uint pad
) {
    uint oy_dim = iy_dim + 2 * pad - k_size + 1;
    uint ox_dim = ix_dim + 2 * pad - k_size + 1;
    for (uint n = 0; n < n_dim; n++) {
        for (uint oy = 0; oy < oy_dim; oy++) {
            for (uint ox = 0; ox < ox_dim; ox++) {
#pragma ivdep
                for (uint fy = 0; fy < k_size; fy++) {
#pragma ivdep
                    for (uint fx = 0; fx < k_size; fx++) {
                        int iy = oy + fy - pad;
                        int ix = ox + fx - pad;
#pragma ivdep
#pragma unroll 16
                        for (uint c = 0; c < c_dim; c++) {
                            float v = 0;
                            if (iy >= 0 && iy < iy_dim && ix >= 0 && ix < ix_dim) {
                                uint src = IDX4D(n_dim, iy_dim, ix_dim, c_dim,
                                                 n, iy, ix, c);
                                v = X[src];
                            }
                            uint dst = IDX6D(n_dim, oy_dim, ox_dim, k_size, k_size, c_dim,
                                             n, oy, ox, fy, fx, c);
                            Y[dst] = v;
                        }
                    }
                }
            }
        }
    }
}
