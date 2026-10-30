// Hand-written float32 and int8 MLPs. Weights come from the generated model_data.h.
#pragma once
#include <stdint.h>

#define MLP_MAX_CLASSES 8

// 0 = baseline scalar loops, 1 = CMSIS-DSP dot products (+ folded int8 input offset). int8 stays bit-exact.
extern int g_mlp_opt;
int mlp_n_classes();
int mlp_n_in();
const char *mlp_model_id();
// float32 reference network; logits[n_classes].
void mlp_f32(const float *x, float *logits);
// Quantize standardized features to the int8 input tensor (TFLite QUANTIZE semantics).
void mlp_quantize_input(const float *x, int8_t *q);
// int8 network following TFLite's FULLY_CONNECTED reference kernel; q_out[n_classes].
void mlp_i8(const int8_t *q_in, int8_t *q_out);
float mlp_dequant_output(int8_t q);
float mlp_dot_f32(const float *a, const float *b, int n);  // exposed for the 'U' microbenchmark
