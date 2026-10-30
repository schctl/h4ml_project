// Single-precision natural log using only float arithmetic.
// Algorithm: FreeBSD msun e_logf.c (from Sun fdlibm), reduced-degree polynomial, error < 1 ulp:
//   ====================================================
//   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//   Developed at SunPro, a Sun Microsystems, Inc. business.
//   Permission to use, copy, modify, and distribute this software is freely granted,
//   provided that this notice is preserved.
//   ====================================================
// newlib's logf on this toolchain promotes to software double (__aeabi_dmul etc.), ~2.4k cycles per call.
// Also compiled on the host by python/test_fastlogf.py to measure its error against numpy.
#pragma once
#include <stdint.h>
#include <string.h>

static inline float fast_logf(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f, two25 = 3.355443200e+07f;
  const float Lg1 = 0.66666662693f, Lg2 = 0.40000972152f, Lg3 = 0.28498786688f, Lg4 = 0.24279078841f;
  float hfsq, f, s, z, R, w, t1, t2, dk;
  int32_t k, ix, i, j;
  memcpy(&ix, &x, 4);
  k = 0;
  if (ix < 0x00800000) {                       // x < 2**-126
    if ((ix & 0x7fffffff) == 0) return -1.0f / 0.0f;  // log(+-0) = -inf
    if (ix < 0) return (x - x) / 0.0f;         // log(-#) = NaN
    k -= 25;
    x *= two25;                                // subnormal: scale up
    memcpy(&ix, &x, 4);
  }
  if (ix >= 0x7f800000) return x + x;
  k += (ix >> 23) - 127;
  ix &= 0x007fffff;
  i = (ix + (0x95f64 << 3)) & 0x800000;
  {
    int32_t t = ix | (i ^ 0x3f800000);         // normalize x or x/2
    memcpy(&x, &t, 4);
  }
  k += (i >> 23);
  f = x - 1.0f;
  if ((0x007fffff & (0x8000 + ix)) < 0xc000) {  // -2**-9 <= f < 2**-9
    if (f == 0.0f) {
      if (k == 0) return 0.0f;
      dk = (float)k;
      return dk * ln2_hi + dk * ln2_lo;
    }
    R = f * f * (0.5f - 0.33333333333333333f * f);
    if (k == 0) return f - R;
    dk = (float)k;
    return dk * ln2_hi - ((R - dk * ln2_lo) - f);
  }
  s = f / (2.0f + f);
  dk = (float)k;
  z = s * s;
  i = ix - (0x6147a << 3);
  w = z * z;
  j = (0x6b851 << 3) - ix;
  t1 = w * (Lg2 + w * Lg4);
  t2 = z * (Lg1 + w * Lg3);
  i |= j;
  R = t2 + t1;
  if (i > 0) {
    hfsq = 0.5f * f * f;
    if (k == 0) return f - (hfsq - s * (hfsq + R));
    return dk * ln2_hi - ((hfsq - (s * (hfsq + R) + dk * ln2_lo)) - f);
  }
  if (k == 0) return f - s * (f - R);
  return dk * ln2_hi - ((s * (f - R) - dk * ln2_lo) - f);
}
