__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
linear_bias(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict B,
    uint n_dim, uint k_dim
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint k = 0; k < k_dim; k++) {
            Y[k_dim * n + k] = B[k] + X[k_dim * n + k];
        }
    }
}
