#include "kernels/ops/utils.h"

#define N_CHANS_MAX     1024
#define REG_SIZE        2

#if IS_AOC==1
#define STORE(a, i, v)       __pipelined_store(&a[(i)], (v))
#define LOAD(a, i)           __pipelined_load(&a[i])
#else
#define STORE(a, i, v)       a[(i)] = v
#define LOAD(a, i)           a[(i)]
#endif

// 12 ms with fmax, 12 ms @ 525 MHz with scalars
//
// X    : (n, y, x, c)
// Y    : (n, y / k_size, x / k_size, c)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
max_pool2d(
    global const chan_vfloat * restrict X,
    global chan_vfloat * restrict Y,
    uint n_dim, uint y_dim, uint x_dim, uint c_dim,
    uint k_dim
) {
    ASSERT(c_dim % CHAN_ALIGN == 0);
    uint c_dim_chunks = c_dim / CHAN_ALIGN;
    for (uint n = 0; n < n_dim; n++) {
        for (uint y0 = 0; y0 < y_dim / k_dim; y0++) {
            for (uint x0 = 0; x0 < x_dim / k_dim; x0++) {
#pragma ivdep
                for (uint c = 0; c < c_dim_chunks; c++) {
                    chan_vfloat reg[REG_SIZE] = {FLT_MIN};
                    for (uint y1 = 0; y1 < k_dim; y1++) {
                        for (uint x1 = 0; x1 < k_dim; x1++) {
                            uint addr = IDX4D(
                                n_dim, y_dim, x_dim, c_dim_chunks,
                                n, k_dim * y0 + y1, k_dim * x0 + x1,
                                0
                            );
                            chan_vfloat v = LOAD(X, addr + c);
                            chan_vfloat top = reg[0];
#pragma unroll
                            for (uint i = 0; i < REG_SIZE - 1; i++) {
                                reg[i] = reg[i + 1];
                            }
                            reg[REG_SIZE - 1] = fmax(top, v);
                        }
                    }
                    uint addr = IDX4D(
                        n_dim, y_dim / k_dim, x_dim / k_dim, c_dim_chunks,
                        n, y0, x0, c
                    );
                    chan_vfloat max = FLT_MIN;
#pragma unroll
                    for (uint i = 0; i < REG_SIZE; i++) {
                        max = fmax(reg[i], max);
                    }
                    STORE(Y, addr, max);
                }
            }
        }
    }
}
