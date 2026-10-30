// Feature extraction per docs/feature_spec.md (reference: python/features.py).
#pragma once
#include <stdint.h>

#define DSP_N_FFT 1024
#define DSP_N_BANDS 64

void dsp_init();
// 0 = baseline (newlib logf/sqrtf), 1 = optimized (float-only log, VSQRT). Same spec, see fastlogf.h.
extern int g_dsp_opt;
int dsp_spec_version();
// x: DSP_N_FFT int16 samples at 4 kHz. out: DSP_N_BANDS standardized features.
// fft_us receives the time spent in arm_rfft_fast_f32 alone.
void dsp_features(const int16_t *x, float *out, uint32_t *fft_us);

struct DspProfile {
  uint32_t prep_us, fft_us, maglog_us, band_us;
};
// Same computation as dsp_features, with per-stage timing.
void dsp_features_profiled(const int16_t *x, float *out, DspProfile *p);
