// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>

#define DEBUG 0

static inline void
assert_impl(
    const __constant char *fun, int line,
    const __constant char *cond_str, bool cond
) {
    #if DEBUG == 1
    if (!cond) {
        printf("%10s, line %3d: %s FAIL!\n", fun, line, cond_str);
    }
    #endif
}

#define STRINGIFY(expr) #expr
#define ASSERT(cond)    assert_impl(__func__, __LINE__, STRINGIFY(cond), cond)

static inline uint
idx4d(uint d0, uint d1, uint d2, uint d3,
       uint i0, uint i1, uint i2, uint i3) {
    return d1 * d2 * d3 * i0 + d2 * d3 * i1 + d3 * i2 + i3;
}

static inline uint
idx3d(uint d0, uint d1, uint d2,
       uint i0, uint i1, uint i2) {
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2);
    return d1 * d2 * i0 + d2 * i1 + i2;
}

static inline float
get_4d(__global const float * restrict D,
       uint d0, uint d1, uint d2, uint d3,
       uint i0, uint i1, uint i2, uint i3) {
    return D[idx4d(d0, d1, d2, d3, i0, i1, i2, i3)];
}

static inline float
get3d(__global const float * restrict D,
       uint d0, uint d1, uint d2,
       uint i0, uint i1, uint i2) {
    return D[idx3d(d0, d1, d2, i0, i1, i2)];
}

static inline void
set3d(
    __global float * restrict D,
    uint d0, uint d1, uint d2,
    uint i0, uint i1, uint i2,
    float val) {
    D[idx3d(d0, d1, d2, i0, i1, i2)] = val;
}

// [id][cyx] : i[cyx]_dim)

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
conv2d(
    uint dc_dim, uint ic_dim,
    uint fy_dim, uint fx_dim,
    __global const float * restrict F,
    uint iy_dim, uint ix_dim,
    __global const float * restrict S,
    uint padding,
    __global float * restrict D
) {
    // Destination image height and width
    uint dy_dim = iy_dim + 2 * padding - fy_dim + 1;
    uint dx_dim = ix_dim + 2 * padding - fx_dim + 1;

    uint d_n = dc_dim * dy_dim * dx_dim;
    uint i_n = ic_dim * iy_dim * ix_dim;

    ASSERT(i_n <= 65536);
    ASSERT(d_n <= 65536);

    // Local image buffers
    __private float LS[65536];
    for (uint i = 0; i < i_n; i++) {
        LS[i] = S[i];
    }
    __private float LD[65536] = {0.0};

    for (uint dc = 0; dc < dc_dim; dc++) {

        // Read filter into local memory
        __private float LF[512][3][3];
        for (uint ci = 0; ci < ic_dim; ci++) {
            for (uint fy = 0; fy < fy_dim; fy++) {
                for (uint fx = 0; fx < fx_dim; fx++) {
                    LF[ci][fy][fx] = F[idx4d(dc_dim, ic_dim, fy_dim, fx_dim,
                                             dc, ci, fy, fx)];
                }
            }
        }

        for (uint ic = 0; ic < ic_dim; ic++) {
            for (uint dy = 0; dy < dy_dim; dy++) {
                for (uint dx = 0; dx < dx_dim; dx++) {
                    uint daddr = idx3d(dc_dim, dy_dim, dx_dim, dc, dy, dx);
                    float acc = LD[daddr];
                    for (uint fy = 0; fy < fy_dim; fy++) {
                        for (uint fx = 0; fx < fx_dim; fx++) {
                            int ay = dy + fy - padding;
                            int ax = dx + fx - padding;
                            float s = 0;
                            if (ay >= 0 && ay < iy_dim && ax >= 0 && ax < ix_dim) {
                                uint addr = idx3d(ic_dim, iy_dim, ix_dim, ic, ay, ax);
                                s = S[addr];
                            }
                            float w = LF[ic][fy][fx];
                            acc += s * w;
                        }
                    }
                    LD[daddr] = acc;
                }
            }
        }
    }
    for (uint i = 0; i < d_n; i++) {
        D[i] = LD[i];
    }
}
