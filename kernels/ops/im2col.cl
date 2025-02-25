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
    uint iy_dim, uint ix_dim, uint ic_dim,
    uint fy_dim, uint fx_dim,
    uint pad
) {
    uint oy_dim = iy_dim + 2 * pad - fy_dim + 1;
    uint ox_dim = ix_dim + 2 * pad - fx_dim + 1;
    for (uint n = 0; n < n_dim; n++) {
        for (uint oy = 0; oy < oy_dim; oy++) {
            for (uint ox = 0; ox < ox_dim; ox++) {
                for (uint fy = 0; fy < fy_dim; fy++) {
                    for (uint fx = 0; fx < fx_dim; fx++) {
                        for (uint ic = 0; ic < ic_dim; ic++) {
                            int iy = oy + fy - pad;
                            int ix = ox + fx - pad;
                            float v = 0;
                            if (iy >= 0 && iy < iy_dim && ix >= 0 && ix < ix_dim) {
                                uint src = IDX4D(n_dim, iy_dim, ix_dim, ic_dim,
                                                 n, iy, ix, ic);
                                v = X[src];
                            }
                            uint dst = IDX6D(n_dim, oy_dim, ox_dim, fy_dim, fx_dim, ic_dim,
                                             n, oy, ox, fy, fx, ic);
                            Y[dst] = v;
                        }
                    }
                }
            }
        }
    }
}
