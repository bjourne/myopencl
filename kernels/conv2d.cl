// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>

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
idx_4d(uint d0, uint d1, uint d2, uint d3,
       uint i0, uint i1, uint i2, uint i3) {
    return d1 * d2 * d3 * i0 + d2 * d3 * i1 + d3 * i2 + i3;
}

static inline uint
idx_3d(uint d0, uint d1, uint d2,
       uint i0, uint i1, uint i2) {
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2);
    return d1 * d2 * i0 + d2 * i1 + i2;
}

static inline float
get_4d(const float *D,
       uint d0, uint d1, uint d2, uint d3,
       uint i0, uint i1, uint i2, uint i3) {
    return D[idx_4d(d0, d1, d2, d3, i0, i1, i2, i3)];
}

static inline float
get_3d(const float *D,
       uint d0, uint d1, uint d2,
       uint i0, uint i1, uint i2) {
    return D[idx_3d(d0, d1, d2, i0, i1, i2)];
}

static inline void
set_3d(
    float *D,
    uint d0, uint d1, uint d2,
    uint i0, uint i1, uint i2,
    float val) {
    D[idx_3d(d0, d1, d2, i0, i1, i2)] = val;
}

__kernel void
conv2d(
    uint n_out, uint n_in, uint f_height, uint f_width,
    __global const float * restrict F,
    uint i_height, uint i_width,
    __global const float * restrict S,
    uint padding,
    __global float * restrict D
) {
    uint d_height = 2 * padding + 1 + i_height - f_height;
    uint d_width = 2 * padding + 1 + i_width - f_width;
    uint d_n_el = n_out * d_height * d_width;

    // Initialize output
    for (uint i = 0; i < d_n_el; i++) {
        D[i] = 0.0;
    }

    int iy_start = -padding;
    int iy_end = i_height + padding - f_height + 1;
    int ix_start = -padding;
    int ix_end = i_width + padding - f_width + 1;
    for (uint co = 0; co < n_out; co++) {
        for (uint ci = 0; ci < n_in; ci++) {
            for (int iy = iy_start; iy < iy_end; iy++) {
                for (int ix = ix_start; ix < ix_end; ix++) {
                    int dy = iy - iy_start;
                    int dx = ix - ix_start;
                    ASSERT(dy >= 0 && dx >= 0);

                    float acc = get_3d(D, n_out, d_height, d_width, co, dy, dx);
                    uint ay_start = max(iy, 0);
                    uint ay_end = min(iy + f_height, i_height);
                    uint ax_start = max(ix, 0);
                    uint ax_end = min(ix + f_width, i_width);
                    for (uint ay = ay_start; ay < ay_end; ay++) {
                        for (uint ax = ax_start; ax < ax_end; ax++) {
                            float s = get_3d(S, n_in, i_height, i_width, ci, ay, ax);
                            float w = get_4d(
                                F, n_out, n_in, f_height, f_width, co, ci,
                                ay - iy, ax - ix);
                            acc += s * w;
                        }
                    }
                    set_3d(D, n_out, d_height, d_width, co, dy, dx, acc);
                }
            }
        }
    }
}
