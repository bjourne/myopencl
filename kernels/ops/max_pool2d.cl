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
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
max_pool2d(
    global const chan_vfloat * restrict X,
    global chan_vfloat * restrict Y,
    uint n_dim, uint iy_dim, uint ix_dim, uint c_dim,
    uint k_dim, uint stride
) {
    ASSERT(c_dim % CHAN_ALIGN == 0);
    uint c_dim_chunks = c_dim / CHAN_ALIGN;

    uint oy_dim = WIN_COUNT(iy_dim, k_dim, stride, 0);
    uint ox_dim = WIN_COUNT(ix_dim, k_dim, stride, 0);
    for (uint n = 0; n < n_dim; n++) {
        for (uint oy = 0; oy < oy_dim; oy++) {
            for (uint ox = 0; ox < ox_dim; ox++) {
#pragma ivdep
                for (uint c = 0; c < c_dim_chunks; c++) {
                    // No idea why OpenCL doesn't like initializer
                    // expressions.
                    chan_vfloat reg[REG_SIZE];
#pragma unroll
                    for (uint i = 0; i < REG_SIZE; i++) {
                        reg[i] = -1e20;
                    }
                    for (uint y = 0; y < k_dim; y++) {
                        for (uint x = 0; x < k_dim; x++) {
                            uint iy = stride * oy + y;
                            uint ix = stride * ox + x;
                            uint addr = IDX4D(
                                n_dim, iy_dim, ix_dim, c_dim_chunks,
                                n, iy, ix, c
                            );
                            chan_vfloat v = LOAD(X, addr);
                            chan_vfloat top = reg[0];
#pragma unroll
                            for (uint i = 0; i < REG_SIZE - 1; i++) {
                                reg[i] = reg[i + 1];
                            }
                            reg[REG_SIZE - 1] = fmax(top, v);
                        }
                    }
                    uint addr = IDX4D(
                        n_dim, oy_dim, ox_dim, c_dim_chunks,
                        n, oy, ox, c
                    );
                    chan_vfloat max = -1e30;
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
