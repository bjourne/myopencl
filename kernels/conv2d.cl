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
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2 && i3 < d3);
    return d1 * d2 * d3 * i0 + d2 * d3 * i1 + d3 * i2 + i3;
}

static inline uint
idx3d(uint d0, uint d1, uint d2,
      uint i0, uint i1, uint i2) {
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2);
    return d1 * d2 * i0 + d2 * i1 + i2;
}

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
conv2d(
    uint dc_dim, uint sc_dim,
    uint fy_dim, uint fx_dim,
    __global const float * restrict F,
    uint sy_dim, uint sx_dim,
    __global const float * restrict S,
    uint padding,
    __global float * restrict D
) {
    // Destination image height and width
    uint dy_dim = sy_dim + 2 * padding - fy_dim + 1;
    uint dx_dim = sx_dim + 2 * padding - fx_dim + 1;

    uint dn = dc_dim * dy_dim * dx_dim;
    uint sn = sc_dim * sy_dim * sx_dim;

    ASSERT(sn <= 65536);
    ASSERT(dn <= 65536);

    // Local image buffers
    __private float LS[65536];
    for (uint i = 0; i < sn; i++) {
        LS[i] = S[i];
    }
    __private float LD[65536];
    for (uint i = 0; i < dn; i++) {
        LD[i] = 0.0f;
    }
    for (uint dc = 0; dc < dc_dim; dc++) {
        // Read filter into local memory
        __private float LF[512][3][3];
        for (uint sc = 0; sc < sc_dim; sc++) {
            for (uint fy = 0; fy < fy_dim; fy++) {
                for (uint fx = 0; fx < fx_dim; fx++) {
                    LF[sc][fy][fx] = F[idx4d(dc_dim, sc_dim, fy_dim, fx_dim,
                                             dc, sc, fy, fx)];
                }
            }
        }
        for (uint sc = 0; sc < sc_dim; sc++) {
            for (uint dy = 0; dy < dy_dim; dy++) {
                for (uint dx = 0; dx < dx_dim; dx++) {
                    uint d_addr = idx3d(dc_dim, dy_dim, dx_dim, dc, dy, dx);
                    float acc = LD[d_addr];
                    for (uint fy = 0; fy < fy_dim; fy++) {
                        #pragma unroll
                        for (uint fx = 0; fx < 8; fx++) {
                            int sy = dy + fy - padding;
                            int sx = dx + fx - padding;
                            if (sy >= 0 && sy < sy_dim && sx >= 0 && sx < sx_dim && fx < fx_dim) {
                                uint s_addr = idx3d(sc_dim, sy_dim, sx_dim, sc, sy, sx);
                                float v = LS[s_addr];
                                float w = LF[sc][fy][fx];
                                acc += v * w;
                            }
                        }
                    }
                    /// Add batch norm
                    acc = (acc - mean) / std;
                    LD[d_addr] = acc;
                }
            }
        }
    }
    for (uint i = 0; i < dn; i++) {
        D[i] = LD[i];
    }
}
