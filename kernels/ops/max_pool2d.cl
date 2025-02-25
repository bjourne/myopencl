#define N_CHANS_MAX     1024

// X    : (n, y, x, c)
// Y    : (n, y / k_size, x / k_size, c)
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
max_pool2d(
    global const float * restrict X,
    global float * restrict Y,
    uint n_dim, uint y_dim, uint x_dim, uint c_dim,
    uint k_size
) {
    for (uint n = 0; n < n_dim; n++) {
        for (uint y0 = 0; y0 < y_dim / k_size; y0++) {
            for (uint x0 = 0; x0 < x_dim / k_size; x0++) {
                float max[N_CHANS_MAX];
                for (uint c = 0; c < c_dim; c++) {
                    max[c] = FLT_MIN;
                }
                for (uint y1 = 0; y1 < k_size; y1++) {
                    for (uint x1 = 0; x1 < k_size; x1++) {
                        for (uint c = 0; c < c_dim; c++) {
                            uint addr = IDX4D(
                                n_dim, y_dim, x_dim, c_dim,
                                n, k_size * y0 + y1, k_size * x0 + x1, c
                            );
                            max[c] = MAX(X[addr], max[c]);
                        }
                    }
                }
                for (uint c = 0; c < c_dim; c++) {
                    uint addr = IDX4D(
                        n_dim, y_dim / k_size, x_dim / k_size, c_dim,
                        n, y0, x0, c
                    );
                    Y[addr] = max[c];
                }
            }
        }
    }
}
