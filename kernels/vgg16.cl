// Copyright (C) 2025 Björn A. Lindqvist <bjourne@gmail.com>
//
// This OpenCL code implements functions necessary for inferring
// vgg16.

// Max nr of supported channels.
#define N_CHANS_MAX     1024

////////////////////////////////////////////////////////////////////////
// Macro utility
////////////////////////////////////////////////////////////////////////
#define IDX4D(ad, bd, cd, dd, a, b, c, d) \
    ((a) * (bd) * (cd) * (dd) + (b) * (cd) * (dd) + (c) * (dd) + (d))
#define MAX(a, b)       ((a) > (b) ? (a) : (b))

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
batch_norm2d(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict mean,
    global const float * restrict var,
    global const float * restrict weights,
    global const float * restrict bias,
    uint n_dim, uint y_dim, uint x_dim, uint c_dim
) {
    // Precompute affine transform. weight[i] / sqrt(var[i] + eps)
    float mul[N_CHANS_MAX];
    float add[N_CHANS_MAX];
    for (uint c = 0; c < c_dim; c++) {
        mul[c] = weights[c] / sqrt(var[c] + 1e-5);
        add[c] = -mean[c] * mul[c] + bias[c];
    }
    for (uint n = 0; n < n_dim; n++) {
        for (uint y = 0; y < y_dim; y++) {
            for (uint x = 0; x < x_dim; x++) {
                for (uint c = 0; c < c_dim; c++) {
                    uint i = IDX4D(n_dim, y_dim, x_dim, c_dim,
                                   n, y, x, c);
                    Y[i] = mul[c] * X[i] + add[c];
                }
            }
        }
    }
}

// X    : (n, m)
// W_t  : (k, m)
// Y    : (n, k)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
linear(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict W_t,
    global const float * restrict B,
    uint n_dim, uint m_dim, uint k_dim
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint k = 0; k < k_dim; k++) {
            float acc = B[k];
            for (uint m = 0; m < m_dim; m++) {
                float w_val = W_t[k * m_dim + m];
                acc += X[m_dim * n + m] * w_val;
            }
            Y[k_dim * n + k] = acc;
        }
    }
}

// Note that the weights W_t are transposed.
//
// X    : (n, iy, ix, ic)
// Xp   : (n * oy * ox, fy * fx * ic)
// W_t  : (oc_dim, fy * fx * ic)
// Y    : (n, oy, ox, oc)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
conv2d(
    global const float * restrict X,
    global float * restrict Xp,
    global float * restrict Y,
    global const float * restrict W_t,
    uint n_dim,
    uint iy_dim, uint ix_dim,
    uint fy_dim, uint fx_dim,
    uint ic_dim, uint oc_dim,
    uint pad
)   {
    // First do im2col and store in Xp
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
                            Xp[at] = v;
                            at++;
                        }
                    }
                }
            }
        }
    }
    uint mat_n = n_dim * oy_dim * ox_dim;
    uint mat_m = fy_dim * fx_dim * ic_dim;
    uint mat_k = oc_dim;

    // Then multiply Xp * W
    for (uint n = 0; n < mat_n; n++) {
        for (uint k = 0; k < mat_k; k++) {
            float acc = 0;
            for (uint m = 0; m < mat_m; m++) {
                acc += Xp[mat_m * n + m] * W_t[mat_m * k + m];
            }
            Y[mat_k * n + k] = acc;
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
