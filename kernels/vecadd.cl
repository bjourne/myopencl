__kernel void vecadd(__global const int *a, __global const int *b, __global int *c) {
    size_t i = get_global_id(0);
    c[i] = a[i] + b[i];
}

__kernel void vecadd2(uint N, __global const int *a, __global const int *b, __global int *c) {
    for (uint i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
