// Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
#ifndef __UTILS_CL__
#define __UTILS_CL__

#define STRINGIFY(expr) #expr

#ifndef DEBUG
#error "Set DEBUG to 0 or 1"
#endif

#if DEBUG==1

static inline void
assert_impl(
    const __constant char *fun, int line,
    const __constant char *cond_str, bool cond
) {
    if (!cond) {
        printf("%10s, line %3d: %s FAIL!\n", fun, line, cond_str);
    }
}

#define ASSERT(cond)    assert_impl(__func__, __LINE__, STRINGIFY(cond), cond)
#else
#define ASSERT(cond)
#endif

static inline uint
idx4d(uint d0, uint d1, uint d2, uint d3,
      uint i0, uint i1, uint i2, uint i3) {
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2 && i3 < d3);
    return d1 * d2 * d3 * i0 + d2 * d3 * i1 + d3 * i2 + i3;
}

static inline uint
idx3d(uint d0, uint d1, uint d2,
      uint i0, uint i1, uint i2) {
    ASSERT(i0 < d0 && i1 < d1 && i2 < d2);
    return d1 * d2 * i0 + d2 * i1 + i2;
}

#endif
