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
linear_pad(
    global const float * restrict X,
    global float * restrict Y,
    uint src_y, uint src_x,
    uint dst_y, uint dst_x
) {
    for (uint y = 0; y < dst_y; y++) {
        for (uint x = 0; x < dst_x; x++) {
            float v = 0;
            if (y < src_y && x < src_x) {
                v = X[src_x * y + x];
            }
            Y[dst_x * y + x] = v;
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
linear_unpad(
    global const float * restrict X,
    global float * restrict Y,
    uint src_y, uint src_x,
    uint dst_y, uint dst_x
) {
    for (uint y = 0; y < dst_y; y++) {
        for (uint x = 0; x < dst_x; x++) {
            Y[dst_x * y + x] = X[src_x * y + x];
        }
    }
}

// X    : (n, m)
// W_t  : (k, m)
// Y    : (n, k)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
linear_matmul(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict W_t,
    uint n_dim, uint m_dim, uint k_dim
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint k = 0; k < k_dim; k++) {
            float acc = 0;
            for (uint m = 0; m < m_dim; m++) {
                float w_val = W_t[k * m_dim + m];
                acc += X[m_dim * n + m] * w_val;
            }
            Y[k_dim * n + k] = acc;
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


// Note that the weights W_t are transposed.
//
// X    : (n, m)
// W_t  : (k, m)
// Y    : (n, k)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
conv2d_matmul(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict W_t,
    uint mat_n, uint mat_m, uint mat_k
) {
    for (uint n = 0; n < mat_n; n++) {
        for (uint k = 0; k < mat_k; k++) {
            float acc = 0;
            for (uint m = 0; m < mat_m; m++) {
                acc += X[mat_m * n + m] * W_t[mat_m * k + m];
            }
            Y[mat_k * n + k] = acc;
        }
    }
}

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
conv2d_full(
    global const float * restrict X,
    global float * restrict Y,
    global float * restrict T,
    global const float * restrict W_t,
    uint n_dim,
    uint iy_dim, uint ix_dim,
    uint fy_dim, uint fx_dim,
    uint ic_dim, uint oc_dim,
    uint pad
) {
    // // First do im2col and store in Xp
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
                            T[at] = v;
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

    // Then multiply T * W
    for (uint n = 0; n < mat_n; n++) {
        for (uint k = 0; k < mat_k; k++) {
            float acc = 0;
            for (uint m = 0; m < mat_m; m++) {
                acc += T[mat_m * n + m] * W_t[mat_m * k + m];
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
