// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#define DEBUG 0
#include "utils.cl"

#define S_MAX   65536
#define D_MAX   65536
#define F_MAX   8192

#define WIDTH 16

#if WIDTH==8

#define vload   vload8
#define vstore  vstore8
typedef float8 vfloat;

#elif WIDTH==16

#define vload   vload16
#define vstore  vstore16
typedef float16 vfloat;

#endif

#define REMAIN(v, m)   ((v) & ((m) - 1))
#define ALIGN_TO(v, m) REMAIN(v, m) ? ((v) + (m) - REMAIN(v, m)) : (v)

// F: dc fy fx sc (OK)
// S: n sy sx sc (OK)
// D: n dy dx dc
__attribute__((uses_global_work_offset(0)))
__attribute__((max_global_work_dim(0)))
__kernel void
conv2d(uint dc_dim, uint sc_dim,
       uint fy_dim, uint fx_dim,
       __global const float * restrict F,
       uint sy_dim, uint sx_dim,
       __global const float * restrict S,
       uint pad,
       __global float * restrict D) {

    // Channels in local image buffer
    uint sc_dim_alg = ALIGN_TO(sc_dim, WIDTH);

    ASSERT(sc_dim_alg >= sc_dim);

    // Padded source height and width
    uint py_dim = sy_dim + 2 * pad;
    uint px_dim = sx_dim + 2 * pad;

    // Destination image height and width
    uint dy_dim = py_dim - fy_dim + 1;
    uint dx_dim = px_dim - fx_dim + 1;

    uint dn = dc_dim * dy_dim * dx_dim;
    uint fn = sc_dim_alg * fy_dim * fx_dim;
    uint sn = sc_dim_alg * py_dim * px_dim;

    ASSERT(dn <= D_MAX);
    ASSERT(fn <= F_MAX);
    ASSERT(sn <= S_MAX);

    // Read padded image into local memory
    __private float LS[S_MAX];
    for (uint py = 0; py < py_dim; py++) {
        for (uint px = 0; px < px_dim; px++) {
            for (uint sc = 0; sc < sc_dim_alg; sc++) {
                int sy = py - pad;
                int sx = px - pad;
                float v = 0;
                if (0 <= sy && sy < sy_dim &&
                    0 <= sx && sx < sx_dim &&
                    sc < sc_dim) {
                    uint s_addr = idx3d(sy_dim, sx_dim, sc_dim, sy, sx, sc);
                    v = S[s_addr];
                }
                uint d_addr = idx3d(py_dim, px_dim, sc_dim_alg, py, px, sc);
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
        for (uint fy = 0; fy < fy_dim; fy++) {
            for (uint fx = 0; fx < fx_dim; fx++) {
                for (uint sc = 0; sc < sc_dim_alg; sc++) {
                    float v = 0;
                    if (sc < sc_dim) {
                        uint f_addr = idx4d(dc_dim, fy_dim, fx_dim, sc_dim,
                                            dc, fy, fx, sc);
                        v = F[f_addr];
                    }
                    uint lf_addr = idx3d(fy_dim, fx_dim, sc_dim_alg,
                                         fy, fx, sc);
                    LF[lf_addr] = v;
                }
            }
        }

        for (uint dy = 0; dy < dy_dim; dy++) {
            for (uint dx = 0; dx < dx_dim; dx++) {
                vfloat acc = 0;
                for (uint fy = 0; fy < fy_dim; fy++) {
                    for (uint fx = 0; fx < fx_dim; fx++) {
                        uint sy = dy + fy;
                        uint sx = dx + fx;
                        for (uint sc = 0; sc < sc_dim_alg; sc += WIDTH) {
                            uint s_addr = idx3d(py_dim, px_dim, sc_dim_alg, sy, sx, sc);
                            uint f_addr = idx3d(fy_dim, fx_dim, sc_dim_alg, fy, fx, sc);

                            vfloat fv = vload(f_addr / WIDTH, LF);
                            vfloat sv = vload(s_addr / WIDTH, LS);
                            acc += fv * sv;
                        }
                    }
                }
                uint d_addr = idx3d(dy_dim, dx_dim, dc_dim, dy, dx, dc);
                vfloat v = vload(d_addr / WIDTH, LD) + acc;
                vstore(v, d_addr / WIDTH, LD);
            }
        }
    }
    for (uint i = 0; i < dn; i++) {
        D[i] = LD[i];
    }
}
