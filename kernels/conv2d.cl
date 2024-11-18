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

#define S_MAX   65536
#define D_MAX   65536
#define F_MAX   8192

__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
conv2d(
uint dc_dim, uint sc_dim,
    uint fy_dim, uint fx_dim,
    __global const float * restrict F,
    uint sy_dim, uint sx_dim,
    __global const float * restrict S,
    uint pad,
    __global float * restrict D
) {

    // Padded source height and width
    uint py_dim = sy_dim + 2 * pad;
    uint px_dim = sx_dim + 2 * pad;

    // Destination image height and width
    uint dy_dim = py_dim - fy_dim + 1;
    uint dx_dim = px_dim - fx_dim + 1;

    uint dn = dc_dim * dy_dim * dx_dim;
    uint sn = sc_dim * sy_dim * sx_dim;
    uint fn = sc_dim * fy_dim * fx_dim;

    ASSERT(sn <= S_MAX);
    ASSERT(dn <= D_MAX);
    ASSERT(fn <= F_MAX);

    // Read padded image into local memory
    __private float LS[S_MAX];
    for (uint sc = 0; sc < sc_dim; sc++) {
        for (uint py = 0; py < py_dim; py++) {
            for (uint px = 0; px < px_dim; px++) {
                int sy = py - pad;
                int sx = px - pad;
                uint d_addr = idx3d(sc_dim, py_dim, px_dim, sc, py, px);
                float v = 0;
                if (0 <= sy && sy < sy_dim && 0 <= sx && sx < sx_dim) {
                    uint s_addr = idx3d(sc_dim, sy_dim, sx_dim, sc, sy, sx);
                    v = S[s_addr];
                }
                LS[d_addr] = v;
            }
        }
    }

    __private float LD[D_MAX];
    for (uint i = 0; i < dn; i++) {
        LD[i] = 0.0f;
    }
    __private float LF[F_MAX];
    for (uint dc = 0; dc < dc_dim; dc++) {
        // Read filter into local memory
        for (uint sc = 0; sc < sc_dim; sc++) {
            for (uint fy = 0; fy < fy_dim; fy++) {
                for (uint fx = 0; fx < fx_dim; fx++) {
                    uint l_addr = idx3d(sc_dim, fy_dim, fx_dim, sc, fy, fx);
                    uint g_addr = idx4d(dc_dim, sc_dim, fy_dim, fx_dim, dc, sc, fy, fx);
                    LF[l_addr] = F[g_addr];
                }
            }
        }
        for (uint dy = 0; dy < dy_dim; dy++) {
            for (uint dx = 0; dx < dx_dim; dx++) {
                uint d_addr = idx3d(dc_dim, dy_dim, dx_dim, dc, dy, dx);
                float acc = LD[d_addr];
                for (uint sc = 0; sc < sc_dim; sc++) {
                    for (uint fy = 0; fy < fy_dim; fy++) {
                        for (uint fx = 0; fx < fx_dim; fx++) {
                            int sy = dy + fy;
                            int sx = dx + fx;
                            uint s_addr = idx3d(sc_dim, py_dim, px_dim, sc, sy, sx);
                            uint f_addr = idx3d(sc_dim, fy_dim, fx_dim, sc, fy, fx);
                            float v = LS[s_addr];
                            float w = LF[f_addr];
                            acc += v * w;
                        }
                    }
                }
                LD[d_addr] = acc;
            }
        }
    }
    for (uint i = 0; i < dn; i++) {
        D[i] = LD[i];
    }
}
