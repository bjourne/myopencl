#include "kernels/ops/utils.h"
#include "kernels/ops/sysarr.h"

__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
preproc_a(
    global const type* restrict A,
    global type* restrict Ap,
    uint size_n, uint size_m,
    uint N, uint M
) {
    // Pad and tile the A matrix.
    for (uint n0 = 0; n0 < N; n0++) {
        for (uint m0 = 0; m0 < M; m0++) {
            for (int n1 = 0; n1 < BLOCK_N; n1++) {
                for (uint m1 = 0; m1 < BLOCK_M; m1++) {
                    uint n = n0 * BLOCK_N + n1;
                    uint m = m0 * BLOCK_M + m1;
                    type v = 0;
                    if (n < size_n && m < size_m) {
                        uint src = IDX2D(size_n, size_m, n, m);
                        v = A[src];
                    }
                    uint dst = IDX4D(N, M, BLOCK_N, BLOCK_M, n0, m0, n1, m1);
                    Ap[dst] = v;
                }
            }
        }
    }
}
