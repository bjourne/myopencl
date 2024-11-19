// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#define DEBUG 0
#include "utils.cl"

#define S_MAX   65536
#define D_MAX   65536
#define F_MAX   8192

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
        uint addr = idx4d(dc_dim, sc_dim, fy_dim, fx_dim, dc, 0, 0, 0);
        for (uint i = 0; i < fn; i++) {
            LF[i] = F[addr + i];
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
