// The affine transform has to be precomputed on the host:
//
//      mul = weight / sqrt(var + 1e-5)
//      add = -mean * mul + bias
__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
batch_norm2d(
    global const float * restrict X,
    global float * restrict Y,
    global const float * restrict mul,
    global const float * restrict add,
    uint n_dim, uint c_dim
) {

    for (uint n = 0; n < n_dim; n++) {
        for (uint c = 0; c < c_dim; c++) {
            uint i = c_dim * n + c;
            Y[i] = mul[c] * X[i] + add[c];
        }
    }
}
