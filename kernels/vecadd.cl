__kernel void vecadd(
    __global const float * restrict a,
    __global const float * restrict b,
    __global float * restrict c
) {
    size_t i = get_global_id(0);
    float av = a[i], bv = b[i];
    c[i] = av + bv;
}

__kernel void vecadd_serial(
    uint N,
    __global const int * restrict a,
    __global const int * restrict b,
    __global int * restrict c
) {
    for (uint i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
