#include "kernels/ops/sysarr.h"
#include "kernels/ops/utils.h"

// Cp:  (N, K, BLOCK_N, PE_S, PE_S)
// C:   (N, K)
//
// Untiles and unpads the C matrix.
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
postproc_c(
    global type* restrict Cp,
    global type* restrict C,
    uint size_n, uint size_k,
    uint N, uint K
) {
    for (uint n0 = 0; n0 < N; n0++) {
        for (uint k0 = 0; k0 < K; k0++) {
            for (uint n1 = 0; n1 < BLOCK_N; n1++) {
                for (uint k1 = 0; k1 < PE_S; k1++) {
                    for (uint k2 = 0; k2 < PE_S; k2++) {
                        uint src = IDX5D(N, K, BLOCK_N, PE_S, PE_S,
                                         n0, k0, n1, k1, k2);
                        uint dst_n = BLOCK_N * n0 + n1;
                        uint dst_k = BLOCK_K * k0 + k2 * PE_S + k1;
                        if (dst_n < size_n && dst_k < size_k) {
                            uint dst = IDX2D(size_n, size_k, dst_n, dst_k);
                            C[dst] = Cp[src];
                        }
                    }
                }
            }
        }
    }
}
