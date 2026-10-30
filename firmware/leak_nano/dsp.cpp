#include "dsp.h"

#include <Arduino.h>
#include <arm_math.h>
#include <math.h>

#include "fastlogf.h"
#include "feature_data.h"

static_assert(FD_N_FFT == DSP_N_FFT && FD_N_BANDS == DSP_N_BANDS, "feature_data.h does not match dsp.h");

static arm_rfft_fast_instance_f32 rfft;
static float buf_in[DSP_N_FFT];
static float buf_out[DSP_N_FFT];
static float logmag[DSP_N_FFT / 2 + 1];
int g_dsp_opt = 1;

static inline float vsqrt(float x) {
  float r;
  __asm__("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x));
  return r;
}

void dsp_init() { arm_rfft_fast_init_f32(&rfft, DSP_N_FFT); }
int dsp_spec_version() { return FD_SPEC_VERSION; }

void dsp_features_profiled(const int16_t *x, float *out, DspProfile *p) {
  uint32_t t0 = micros();
  // DC removal from the exact integer sum (spec step 3).
  int32_t sum = 0;
  for (int i = 0; i < DSP_N_FFT; i++) sum += x[i];
  const float mean = (float)((double)sum / (DSP_N_FFT * 32768.0));
  if (g_dsp_opt) {
    for (int i = 0; i < DSP_N_FFT; i++) buf_in[i] = ((float)x[i] * (1.0f / 32768.0f) - mean) * FD_HANN[i];
  } else {
    for (int i = 0; i < DSP_N_FFT; i++) buf_in[i] = ((float)x[i] / 32768.0f - mean) * FD_HANN[i];
  }

  uint32_t t1 = micros();
  arm_rfft_fast_f32(&rfft, buf_in, buf_out, 0);  // clobbers buf_in
  uint32_t t2 = micros();

  // Packed output: [Re X0, Re X(N/2), Re X1, Im X1, ...].
  if (g_dsp_opt) {
    logmag[0] = fast_logf(fabsf(buf_out[0]) + FD_EPS);
    logmag[DSP_N_FFT / 2] = fast_logf(fabsf(buf_out[1]) + FD_EPS);
    for (int k = 1; k < DSP_N_FFT / 2; k++) {
      const float re = buf_out[2 * k], im = buf_out[2 * k + 1];
      logmag[k] = fast_logf(vsqrt(re * re + im * im) + FD_EPS);
    }
  } else {
    logmag[0] = logf(fabsf(buf_out[0]) + FD_EPS);
    logmag[DSP_N_FFT / 2] = logf(fabsf(buf_out[1]) + FD_EPS);
    for (int k = 1; k < DSP_N_FFT / 2; k++) {
      const float re = buf_out[2 * k], im = buf_out[2 * k + 1];
      logmag[k] = logf(sqrtf(re * re + im * im) + FD_EPS);
    }
  }
  uint32_t t3 = micros();

  for (int b = 0; b < DSP_N_BANDS; b++) {
    float acc = 0.0f;
    for (int k = FD_BAND_EDGES[b]; k < FD_BAND_EDGES[b + 1]; k++) acc += logmag[k];
    const float f = acc / (float)(FD_BAND_EDGES[b + 1] - FD_BAND_EDGES[b]);
    out[b] = (f - FD_MU[b]) / FD_SIGMA[b];
  }
  uint32_t t4 = micros();
  p->prep_us = t1 - t0;
  p->fft_us = t2 - t1;
  p->maglog_us = t3 - t2;
  p->band_us = t4 - t3;
}

void dsp_features(const int16_t *x, float *out, uint32_t *fft_us) {
  DspProfile p;
  dsp_features_profiled(x, out, &p);
  *fft_us = p.fft_us;
}
