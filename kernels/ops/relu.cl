__attribute__((max_global_work_dim(0)))
__attribute__((uses_global_work_offset(0)))
kernel void
relu(
    global const float * restrict X,
    global float * restrict Y,
    uint n
) {
    for (uint i = 0; i < n; i++) {
        Y[i] = max(X[i], 0.0f);
    }
}
