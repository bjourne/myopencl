// Copyright (C) 2024-2025 Björn A. Lindqvist <bjourne@gmail.com>
#include "utils.cl"

#define REG_LEN                     64

#define S_MAX   65536
#define D_MAX   65536
#define F_MAX   8192

#if V_SIZE==4

#define vload   vload4
#define vstore  vstore4
typedef float4 vfloat;

#elif V_SIZE==8

#define vload   vload8
#define vstore  vstore8
typedef float8 vfloat;

#elif V_SIZE==16

#define vload   vload16
#define vstore  vstore16
typedef float16 vfloat;

#define vreduce(v)  (v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7] \
    + v[8] + v[9] + v[10] + v[11] + v[12] + v[13] + v[14] + v[15])

#else

#error "Define V_SIZE!"

#endif

// sc must be a multiple of V_SIZE
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

    ASSERT(sc_dim % V_SIZE == 0);

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
    ASSERT(fy_dim == K_SIZE && fx_dim == K_SIZE);
    ASSERT(sc_dim % V_SIZE == 0);

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

#pragma ivdep
        for (uint dy = 0; dy < dy_dim; dy++) {
#pragma ivdep
            for (uint dx = 0; dx < dx_dim; dx++) {

                vfloat acc[REG_LEN] = {0};
#pragma ii 1
                for (uint i = 0; i < K_SIZE * sc_dim; i += V_SIZE) {
#pragma unroll
                    for (uint fy = 0; fy < 3; fy++) {
                        uint f_addr = K_SIZE * sc_dim * fy + i;
                        uint s_addr = px_dim * sc_dim * (dy + fy) + sc_dim * dx + i;

                        vfloat fv = vload(f_addr / V_SIZE, LF);
                        vfloat sv = vload(s_addr / V_SIZE, LS);

                        vfloat sum = acc[0] + fv * sv;
#pragma unroll
                        for (uint j = 0; j < REG_LEN - 1; j++) {
                            acc[j] = acc[j + 1];
                        }
                        acc[REG_LEN - 1] = sum;
                    }
                }
                uint d_addr = idx3d(dy_dim, dx_dim, dc_dim, dy, dx, dc);
                vfloat acc0 = 0;
#pragma unroll
                for (uint i = 0; i < REG_LEN; i++) {
                    acc0 += acc[i];
                }
                LD[d_addr] += vreduce(acc0);
            }
        }
    }
    for (uint i = 0; i < dn; i++) {
        D[i] = LD[i];
    }
}
