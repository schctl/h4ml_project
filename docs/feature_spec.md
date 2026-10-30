# Feature specification (single source of truth)

Spec version: **2** (`FeatureConfig.spec_version`). Reference implementation: `python/features.py`.
Firmware implementation: `firmware/leak_nano/dsp.cpp`. Parity is checked by the self-test vectors
(`python/export.py` → `firmware/leak_nano/testvec.h`, the `S` command).

## 1. Input decoding
- The dataset `.raw` files are headerless little-endian int32 PCM at 8000 Hz (from the dataset's reader snippet).
- In all 122 hydrophone files the low 16 bits of every sample are zero (`results/dataset_summary.json`,
  `all_low16_zero`). So the data is 16-bit audio, left-justified in 32 bits.
- **Scaling:** `x16 = x32 >> 16`. This is exact, with no gain and no information loss. Samples at ±32767 are
  clipped in the source recordings (44 files have some clipped samples; the worst, LO_LC_T_N, has 4.3%).
  They are kept as-is.

## 2. Resampling (host only)
- 8000 Hz → 4000 Hz with `scipy.signal.resample_poly(x, 1, 2)`. This is a polyphase FIR with scipy's default
  Kaiser-window anti-alias filter, applied to the float64 int16 signal, then rounded to nearest and clipped
  to int16.
- Justification: the dataset paper states leak frequencies are below 1 kHz [CITATION NEEDED: Aghashahi et al.
  2023]. Our band-energy check (`results/signal_check.csv`) shows that almost all energy is below 100 Hz,
  and that above 1 kHz the RMS is around 10 counts.
- E4 compares this against no resampling.

## 3. Windowing
- `N = 1024` samples (256 ms at 4 kHz). The hop is 1024 (non-overlapping windows), and the incomplete
  tail is dropped.
- **DC removal:** `mean = float32( sum_int64(x16) / (N * 32768.0) )`. The sum is exact, the division is done
  in double, and the result is rounded once to float32. Then `x = float32(x16) / 32768 - mean`.
- **Window:** periodic Hann, `w[n] = 0.5 - 0.5 cos(2πn/N)`, computed in float64 and stored as float32. The
  firmware uses the same table, generated into `feature_data.h`.

## 4. Spectrum
- Real FFT of the windowed float32 frame, giving 513 bins with spacing 3.906 Hz. The magnitude is `|X[k]|`.
  There is no FFT normalization.
  - Host: `numpy.fft.rfft` (float32 in NumPy 2).
  - Firmware: CMSIS-DSP `arm_rfft_fast_f32`, Arduino_CMSIS-DSP 5.7.0. Its packed output (`[Re X0, Re X512, Re X1, Im X1, ...]`) is unpacked.
- **Log:** `L[k] = ln(|X[k]| + 1e-6)`.

## 5. Band pooling
- There are 64 bands over FFT bins `[6, 513)`, which is 23.4 Hz to 2000 Hz. Band `b` is the **mean of `L[k]`**
  over bins `[e[b], e[b+1])`.
- **Low cut:** bins below 20 Hz (bins 0–5) are excluded. That region holds most of the signal energy (slow
  pressure fluctuations), and it differs between pipe networks and sessions. The aim is to stop that
  shortcut from dominating. E1 includes a no-low-cut ablation (`f_low = 0`, bins 1–512).
- **Edges:** greedy geometric spacing.
  - Start at `e[0] = 6`.
  - For band `i`, with `B = 64 - i` bands left, take `r = (513/e[i])^(1/B)` and
    `width = max(1, round(e[i]·r - e[i]))`. The last edge is forced to 513.
  - The result is single-bin bands for bins 6–22 (the first 17 bands), growing to 33 bins at the top. The table is generated into
    `FD_BAND_EDGES`.

## 6. Standardization
- `z = (f - μ) / σ`, per band, with σ floored at 1e-6. μ and σ come from the **training windows of the model
  being evaluated only**. Validation windows used for early stopping are excluded.
- For the deployed model, μ and σ are baked into the firmware as `FD_MU`, `FD_SIGMA`.

## 7. int8 input quantization (int8 model only)
- `q = clip(round_half_away_from_zero(float32(z) / float32(s_in)) + zp_in, -128, 127)`. `s_in` and `zp_in`
  come from the int8 TFLite model's input tensor. The same formula is used in numpy (`export.py`) and in C
  (`mlp.cpp`).

## Changes log
- v1 → v2: DC removal changed from a float32 `mean()` to the exact integer-sum form above, so the host and
  firmware compute a bit-identical mean. All reported results use v2.
