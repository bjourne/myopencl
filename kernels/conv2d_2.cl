// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#define DEBUG 0
#include "utils.cl"

#define S_MAX   65536
#define D_MAX   65536
#define F_MAX   8192

#if WIDTH==4

#define vload   vload4
#define vstore  vstore4
typedef float4 vfloat;

#elif WIDTH==8

#define vload   vload8
#define vstore  vstore8
typedef float8 vfloat;

#elif WIDTH==16

#define vload   vload16
#define vstore  vstore16
typedef float16 vfloat;

#else

#error "Define WIDTH!"

#endif

#define REMAIN(v, m)   ((v) & ((m) - 1))
#define ALIGN_TO(v, m) REMAIN(v, m) ? ((v) + (m) - REMAIN(v, m)) : (v)

// sc must be a multiple of WIDTH
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

    ASSERT(sc_dim % WIDTH == 0);

    // Padded source height and width
    uint py_dim = sy_dim + 2 * pad;
    uint px_dim = sx_dim + 2 * pad;

    // Destination image height and width
    uint dy_dim = py_dim - fy_dim + 1;
    uint dx_dim = px_dim - fx_dim + 1;

    uint dn = dc_dim * dy_dim * dx_dim;
    uint fn = sc_dim * fy_dim * fx_dim;
    uint sn = sc_dim * py_dim * px_dim;

    ASSERT(dn <= D_MAX);
    ASSERT(fn <= F_MAX);
    ASSERT(sn <= S_MAX);

    // Read padded image into local memory
    __private float LS[S_MAX];
    for (uint py = 0; py < py_dim; py++) {
        for (uint px = 0; px < px_dim; px++) {
            for (uint sc = 0; sc < sc_dim; sc++) {
                int sy = py - pad;
                int sx = px - pad;
                float v = 0;
                if (0 <= sy && sy < sy_dim && 0 <= sx && sx < sx_dim) {
                    uint s_addr = idx3d(sy_dim, sx_dim, sc_dim, sy, sx, sc);
                    v = S[s_addr];
                }
                uint d_addr = idx3d(py_dim, px_dim, sc_dim, py, px, sc);
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
        uint addr = idx4d(dc_dim, fy_dim, fx_dim, sc_dim, dc, 0, 0, 0);
        for (uint i = 0; i < fn; i++) {
            LF[i] = F[addr + i];
        }
        for (uint dy = 0; dy < dy_dim; dy++) {
            for (uint dx = 0; dx < dx_dim; dx++) {
                vfloat acc = 0;
                for (uint fy = 0; fy < fy_dim; fy++) {
                    for (uint fx = 0; fx < fx_dim; fx++) {
                        uint sy = dy + fy;
                        uint sx = dx + fx;
                        for (uint sc = 0; sc < sc_dim; sc += WIDTH) {
                            uint s_addr = idx3d(py_dim, px_dim, sc_dim, sy, sx, sc);
                            uint f_addr = idx3d(fy_dim, fx_dim, sc_dim, fy, fx, sc);

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
