#ifndef __UTILS_H__
#define __UTILS_H__

#define MAX(a, b)       ((a) > (b) ? (a) : (b))

#define IDX2D(ad, bd, a, b) ((a) * (bd) + (b))
#define IDX3D(ad, bd, cd, a, b, c) ((a) * (bd) * (cd) + (b) * (cd) + (c))
#define IDX4D(ad, bd, cd, dd, a, b, c, d) \
    ((a) * (bd) * (cd) * (dd) + (b) * (cd) * (dd) + (c) * (dd) + (d))
#define IDX5D(ad, bd, cd, dd, ed, a, b, c, d, e)    \
    ((a) * (bd) * (cd) * (dd) * (ed) +              \
     (b) * (cd) * (dd) * (ed) +                     \
     (c) * (dd) * (ed) +                            \
     (d) * (ed) +                                   \
     (e))
#define IDX6D(ad, bd, cd, dd, ed, fd, a, b, c, d, e, f) \
    ((a) * (bd) * (cd) * (dd) * (ed) * (fd) +           \
     (b) * (cd) * (dd) * (ed) * (fd) +                  \
     (c) * (dd) * (ed) * (fd) +                         \
     (d) * (ed) * (fd) +                                \
     (e) * (fd) +                                       \
     (f))

#endif
