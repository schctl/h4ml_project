#include "mlp.h"

#include <arm_math.h>
#include <math.h>
#include <string.h>

#include "model_data.h"

static_assert(MD_N_CLASSES <= MLP_MAX_CLASSES, "too many classes");

int g_mlp_opt = 1;
int mlp_n_classes() { return MD_N_CLASSES; }
int mlp_n_in() { return MD_N_IN; }
const char *mlp_model_id() { return MD_MODEL_ID; }

static int32_t clamp(int32_t v, int32_t lo, int32_t hi) { return v < lo ? lo : (v > hi ? hi : v); }

// ---------------- requantization (TFLite fixed-point rescaling) ----------------
// Scales an int32 accumulator by (mult * 2^shift) exactly as TFLite's MultiplyByQuantizedMultiplier does.
static int32_t sat_rounding_doubling_high_mul(int32_t a, int32_t b) {
  if (a == b && a == INT32_MIN) return INT32_MAX;
  const int64_t ab = (int64_t)a * (int64_t)b;
  const int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
  return (int32_t)((ab + nudge) / (1LL << 31));
}

static int32_t rounding_divide_by_pot(int32_t x, int exponent) {
  const int32_t mask = (int32_t)((1LL << exponent) - 1);
  const int32_t remainder = x & mask;
  const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
  return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

static int32_t requantize(int32_t x, int32_t mult, int shift) {
  const int left = shift > 0 ? shift : 0;
  const int right = shift > 0 ? 0 : -shift;
  return rounding_divide_by_pot(sat_rounding_doubling_high_mul(x * (1 << left), mult), right);
}

// ---------------- one dense layer, plain C ----------------
static void dense_f32(const LayerF32 &L, const float *in, float *out) {
  for (int j = 0; j < L.n_out; j++) {
    float sum = L.b[j];
    for (int i = 0; i < L.n_in; i++) sum += L.w[j * L.n_in + i] * in[i];
    out[j] = (L.relu && sum < 0.0f) ? 0.0f : sum;  // ReLU
  }
}

static void dense_i8(const LayerI8 &L, const int8_t *in, int8_t *out) {
  for (int j = 0; j < L.n_out; j++) {
    const int8_t *w = L.w + j * L.n_in;  // weights of output neuron j
    int32_t sum = L.b[j];
    for (int i = 0; i < L.n_in; i++) {
      sum += ((int32_t)in[i] + L.in_offset) * (int32_t)w[i];
    }
    const int c = L.per_channel ? j : 0;
    // rescale to the int8 output scale, then clamp (this also applies ReLU)
    const int32_t y = requantize(sum, L.mult[c], L.shift[c]) + L.out_zp;
    out[j] = (int8_t)clamp(y, L.act_min, L.act_max);
  }
}

// ---------------- one dense layer, optimized for the Cortex-M4 (same results) ----------------
// Float: four independent accumulators (the M4's VFMA takes 3 cycles and does not pipeline).
__attribute__((optimize("O2", "fp-contract=off"))) static float dot_f32_x4(const float *a, const float *b, int n) {
  float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; i++) s0 += a[i] * b[i];
  return (s0 + s1) + (s2 + s3);
}

// int8: SXTB16 splits 4 bytes into two int16 pairs, SMLAD does two 16x16 MACs per instruction.
__attribute__((optimize("O2"))) static int32_t dot_q7_simd(const int8_t *a, const int8_t *b, int n) {
  int32_t acc = 0;
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    uint32_t va, vb;
    memcpy(&va, a + i, 4);
    memcpy(&vb, b + i, 4);
    acc = __SMLAD(__SXTB16(va), __SXTB16(vb), acc);
    acc = __SMLAD(__SXTB16(__ROR(va, 8)), __SXTB16(__ROR(vb, 8)), acc);
  }
  for (; i < n; i++) acc += (int32_t)a[i] * (int32_t)b[i];
  return acc;
}

float mlp_dot_f32(const float *a, const float *b, int n) { return dot_f32_x4(a, b, n); }

static void dense_f32_fast(const LayerF32 &L, const float *in, float *out) {
  for (int j = 0; j < L.n_out; j++) {
    const float sum = dot_f32_x4(L.w + j * L.n_in, in, L.n_in) + L.b[j];
    out[j] = (L.relu && sum < 0.0f) ? 0.0f : sum;
  }
}

static void dense_i8_fast(const LayerI8 &L, const int8_t *in, int8_t *out) {
  for (int j = 0; j < L.n_out; j++) {
    // sum((in + off) * w) + b == sum(in * w) + (b + off * sum(w)); b_folded is precomputed at export.
    const int32_t sum = dot_q7_simd(in, L.w + j * L.n_in, L.n_in) + L.b_folded[j];
    const int c = L.per_channel ? j : 0;
    out[j] = (int8_t)clamp(requantize(sum, L.mult[c], L.shift[c]) + L.out_zp, L.act_min, L.act_max);
  }
}

// ---------------- whole network ----------------
static float act_a[MD_MAX_WIDTH], act_b[MD_MAX_WIDTH];
static int8_t q_a[MD_MAX_WIDTH], q_b[MD_MAX_WIDTH];

void mlp_f32(const float *x, float *logits) {
  const float *in = x;
  for (int l = 0; l < MD_N_LAYERS; l++) {
    float *out = (l == MD_N_LAYERS - 1) ? logits : ((l & 1) ? act_b : act_a);
    if (g_mlp_opt) dense_f32_fast(MD_F32[l], in, out);
    else dense_f32(MD_F32[l], in, out);
    in = out;
  }
}

void mlp_quantize_input(const float *x, int8_t *q) {
  for (int i = 0; i < MD_N_IN; i++) {
    int32_t v = (int32_t)roundf(x[i] / MD_IN_SCALE) + MD_IN_ZP;
    q[i] = (int8_t)clamp(v, -128, 127);
  }
}

void mlp_i8(const int8_t *q_in, int8_t *q_out) {
  const int8_t *in = q_in;
  for (int l = 0; l < MD_N_LAYERS; l++) {
    int8_t *out = (l == MD_N_LAYERS - 1) ? q_out : ((l & 1) ? q_b : q_a);
    if (g_mlp_opt) dense_i8_fast(MD_I8[l], in, out);
    else dense_i8(MD_I8[l], in, out);
    in = out;
  }
}

float mlp_dequant_output(int8_t q) { return ((int32_t)q - MD_OUT_ZP) * MD_OUT_SCALE; }
