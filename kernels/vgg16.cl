// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// This OpenCL program contains kernels for inferring VGG16 and other
// networks.

////////////////////////////////////////////////////////////////////////
// Macro utility
////////////////////////////////////////////////////////////////////////
#define IDX4D(ad, bd, cd, dd, a, b, c, d) \
    ((a) * (bd) * (cd) * (dd) + (b) * (cd) * (dd) + (c) * (dd) + (d))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))

// The affine transform has to be precomputed on the host:
//
//      mul = weight / sqrt(var + 1e-5)
//      add = -mean * mul + bias
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
batch_norm2d(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict mul,
    global const float * restrict add,
    uint n_dim, uint c_dim
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint c = 0; c < c_dim; c++) {
            uint i = c_dim * n + c;
            Y[i] = mul[c] * X[i] + add[c];
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
linear_bias(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict B,
    uint n_dim, uint k_dim
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint k = 0; k < k_dim; k++) {
            Y[k_dim * n + k] = B[k] + X[k_dim * n + k];
        }
    }
}


// X:   (n, iy, ix, ic)
// Y:   (n * oy * ox, fy * fx * ic)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
conv2d_im2col(
    global const float * restrict X,
    global float * restrict Y,
    uint n_dim,
    uint iy_dim, uint ix_dim,
    uint fy_dim, uint fx_dim,
    uint ic_dim, uint oc_dim,
    uint pad
) {
    uint oy_dim = iy_dim + 2 * pad - fy_dim + 1;
    uint ox_dim = ix_dim + 2 * pad - fx_dim + 1;
    uint at = 0;
    for (uint n = 0; n < n_dim; n++) {
        for (uint oy = 0; oy < oy_dim; oy++) {
            for (uint ox = 0; ox < ox_dim; ox++) {
                for (uint fy = 0; fy < fy_dim; fy++) {
                    for (uint fx = 0; fx < fx_dim; fx++) {
                        int iy = oy + fy - pad;
                        int ix = ox + fx - pad;
                        for (uint ic = 0; ic < ic_dim; ic++) {
                            float v = 0;
                            if (iy >= 0 && iy < iy_dim && ix >= 0&& ix < ix_dim) {
                                uint idx = IDX4D(n_dim, iy_dim, ix_dim, ic_dim,
                                                 n, iy, ix, ic);
                                v = X[idx];
                            }
                            Y[at] = v;
                            at++;
                        }
                    }
                }
            }
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
relu(
    global const float * restrict X,
    global float * restrict Y,
    uint n) {

    for (uint i = 0; i < n; i++) {
        Y[i] = max(X[i], 0.0f);
    }
}

#define N_CHANS_MAX     1024

// X    : (n, y, x, c)
// Y    : (n, y / k_size, x / k_size, c)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
max_pool2d(
    global const float * restrict X,
    global float * restrict Y,
    uint n_dim, uint y_dim, uint x_dim, uint c_dim,
    uint k_size
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint y0 = 0; y0 < y_dim / k_size; y0++) {
            for (uint x0 = 0; x0 < x_dim / k_size; x0++) {
                float max[N_CHANS_MAX];
                for (uint c = 0; c < c_dim; c++) {
                    max[c] = FLT_MIN;
                }
                for (uint y1 = 0; y1 < k_size; y1++) {
                    for (uint x1 = 0; x1 < k_size; x1++) {
                        for (uint c = 0; c < c_dim; c++) {
                            uint addr = IDX4D(
                                n_dim, y_dim, x_dim, c_dim,
                                n, k_size * y0 + y1, k_size * x0 + x1, c
                            );
                            max[c] = MAX(X[addr], max[c]);
                        }
                    }
                }
                for (uint c = 0; c < c_dim; c++) {
                    uint addr = IDX4D(
                        n_dim, y_dim / k_size, x_dim / k_size, c_dim,
                        n, y0, x0, c
                    );
                    Y[addr] = max[c];
                }
            }
        }
    }
}
