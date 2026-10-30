"""Export the deployed model to TFLite (float32, int8 PTQ), generate firmware headers and test vectors,
and check host-side agreement (Keras vs TFLite vs numpy emulation of the firmware int8 kernel).

Outputs:
  results/models/deploy_binary_{f32,int8}.tflite
  firmware/leak_nano/{feature_data.h, model_data.h, testvec.h}
  results/export_info.json, results/int8_agreement_host.json
  data/cache/hil_windows.npz   (all windows' int16 frames + host reference outputs, for HIL streaming)
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

from dataset import CACHE, build
from features import DEFAULT, band_edges, hann
from load import ROOT
from train import DEPLOY, MODELS, RES

FW = ROOT / "firmware" / "leak_nano"
N_TV = 16
TOL_FEAT = 1e-3
TOL_LOGIT = 1e-4


# ---------------- quantization helpers (mirror firmware/leak_nano/mlp.cpp) ----------------
def round_half_away(v):
    t = np.trunc(v)
    return t + np.where(np.abs(v - t) >= 0.5, np.sign(v), 0)


def quantize_input(z, scale, zp):
    q = round_half_away(z.astype(np.float32) / np.float32(scale)).astype(np.int64) + zp
    return np.clip(q, -128, 127).astype(np.int8)


def quantize_multiplier(m):
    """TFLite QuantizeMultiplier (double -> Q31 multiplier + shift)."""
    if m == 0.0:
        return 0, 0
    q, shift = math.frexp(m)
    qf = int(math.floor(q * (1 << 31) + 0.5))
    if qf == (1 << 31):
        qf //= 2
        shift += 1
    if shift < -31:
        return 0, 0
    return qf, shift


def srdhm(a, b):
    """SaturatingRoundingDoublingHighMul, vectorized over int64 arrays."""
    ab = a.astype(np.int64) * np.int64(b) if np.isscalar(b) else a.astype(np.int64) * b.astype(np.int64)
    nudge = np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    s = ab + nudge
    res = np.where(s >= 0, s // (1 << 31), -((-s) // (1 << 31)))  # C division truncates toward zero
    return np.where((a == -(1 << 31)) & (b == -(1 << 31)), (1 << 31) - 1, res)


def rdbpot(x, exp):
    mask = (np.int64(1) << exp) - 1
    rem = x & mask
    thr = (mask >> 1) + (x < 0)
    return (x >> exp) + (rem > thr)


def mbqm(x, mult, shift):
    left = np.maximum(shift, 0)
    right = np.maximum(-shift, 0)
    return rdbpot(srdhm(x * (np.int64(1) << left), mult), right)


def i8_forward(q_in, layers):
    a = q_in.astype(np.int64)
    for L in layers:
        acc = (a + L["in_offset"]) @ L["w"].astype(np.int64).T + L["b"].astype(np.int64)
        mult = L["mult"] if L["per_channel"] else L["mult"][0]
        shift = L["shift"] if L["per_channel"] else L["shift"][0]
        acc = mbqm(acc, np.asarray(mult, np.int64), np.asarray(shift, np.int64)) + L["out_zp"]
        a = np.clip(acc, L["act_min"], L["act_max"])
    return a.astype(np.int8)


# ---------------- TFLite ----------------
def convert(model, rep):
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    f32 = c.convert()
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = lambda: ([r[None, :]] for r in rep)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    return f32, c.convert()


def interp(content, ref=True):
    kw = dict(experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF) if ref else {}
    it = tf.lite.Interpreter(model_content=content, **kw)
    it.allocate_tensors()
    return it


def run_tflite(it, X):
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    it.resize_tensor_input(i["index"], [len(X), X.shape[1]])
    it.allocate_tensors()
    it.set_tensor(i["index"], X)
    it.invoke()
    return it.get_tensor(o["index"]).copy()


def extract_int8_layers(content, relu_flags):
    it = interp(content)
    tensors = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()
    fcs = [op for op in ops if op["op_name"] == "FULLY_CONNECTED"]
    assert len(fcs) == len(relu_flags) and len(ops) == len(fcs), [op["op_name"] for op in ops]
    layers = []
    for op, relu in zip(fcs, relu_flags):
        ti, tw, tb = (tensors[k] for k in op["inputs"])
        to = tensors[op["outputs"][0]]
        s_in = float(ti["quantization_parameters"]["scales"][0])
        zp_in = int(ti["quantization_parameters"]["zero_points"][0])
        s_w = ti_w = tw["quantization_parameters"]["scales"].astype(np.float64)
        assert np.all(tw["quantization_parameters"]["zero_points"] == 0)
        s_out = float(to["quantization_parameters"]["scales"][0])
        zp_out = int(to["quantization_parameters"]["zero_points"][0])
        per_channel = len(s_w) > 1
        mults, shifts = zip(*[quantize_multiplier(np.float64(np.float32(s_in)) * np.float64(np.float32(s)) /
                                                  np.float64(np.float32(s_out))) for s in ti_w])
        layers.append(dict(
            n_in=int(tw["shape"][1]), n_out=int(tw["shape"][0]),
            w=it.get_tensor(tw["index"]).astype(np.int8), b=it.get_tensor(tb["index"]).astype(np.int32),
            in_scale=s_in, in_zp=zp_in, in_offset=-zp_in, out_scale=s_out, out_zp=zp_out,
            w_scales=s_w.tolist(), per_channel=per_channel, mult=list(mults), shift=list(shifts),
            act_min=max(-128, zp_out) if relu else -128, act_max=127))
    return layers


# ---------------- C emitters ----------------
def cf(v):
    t = f"{float(np.float32(v)):.9g}"
    if not any(ch in t for ch in ".en"):
        t += ".0"
    return t + "f"


def carr(ctype, name, vals, per_line=12, fmt=str):
    vals = list(vals)
    lines = [", ".join(fmt(v) for v in vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return f"static const {ctype} {name}[{len(vals)}] = {{\n  " + ",\n  ".join(lines) + "\n};\n"


def emit_feature_data(mu, sigma):
    e = band_edges(DEFAULT)
    s = ["// GENERATED by python/export.py -- do not edit.", "#pragma once", "#include <stdint.h>",
         f"#define FD_N_FFT {DEFAULT.n_fft}", f"#define FD_N_BANDS {DEFAULT.n_bands}",
         f"#define FD_SPEC_VERSION {DEFAULT.spec_version}", f"#define FD_EPS {cf(DEFAULT.eps)}", ""]
    s.append(carr("uint16_t", "FD_BAND_EDGES", e.tolist(), 16))
    s.append(carr("float", "FD_HANN", hann(DEFAULT.n_fft), 8, cf))
    s.append(carr("float", "FD_MU", mu, 8, cf))
    s.append(carr("float", "FD_SIGMA", sigma, 8, cf))
    (FW / "feature_data.h").write_text("\n".join(s))


def emit_model_data(keras_layers, i8_layers, model_id, n_classes):
    s = ["// GENERATED by python/export.py -- do not edit.", "#pragma once", "#include <stdint.h>", "",
         "struct LayerF32 { int n_in, n_out; const float *w, *b; int relu; };",
         "struct LayerI8 { int n_in, n_out; const int8_t *w; const int32_t *b, *b_folded; const int32_t *mult;",
         "                 const int8_t *shift; int per_channel; int32_t in_offset, out_zp, act_min, act_max; };", "",
         f'#define MD_MODEL_ID "{model_id}"', f"#define MD_N_IN {keras_layers[0][0].shape[0]}",
         f"#define MD_N_CLASSES {n_classes}", f"#define MD_N_LAYERS {len(keras_layers)}",
         f"#define MD_MAX_WIDTH {max(w.shape[1] for w, _, _ in keras_layers)}",
         f"#define MD_IN_SCALE {cf(i8_layers[0]['in_scale'])}", f"#define MD_IN_ZP {i8_layers[0]['in_zp']}",
         f"#define MD_OUT_SCALE {cf(i8_layers[-1]['out_scale'])}", f"#define MD_OUT_ZP {i8_layers[-1]['out_zp']}", ""]
    f32_rows, i8_rows = [], []
    for l, ((w, b, relu), L) in enumerate(zip(keras_layers, i8_layers)):
        s.append(carr("float", f"MD_W{l}", w.T.reshape(-1), 8, cf))  # Keras kernel is [in, out] -> [out][in]
        s.append(carr("float", f"MD_B{l}", b, 8, cf))
        s.append(carr("int8_t", f"MD_QW{l}", L["w"].reshape(-1), 24))
        s.append(carr("int32_t", f"MD_QB{l}", L["b"], 12))
        folded = L["b"].astype(np.int64) + L["in_offset"] * L["w"].astype(np.int64).sum(axis=1)
        assert np.all(np.abs(folded) < 2**31)
        s.append(carr("int32_t", f"MD_QBF{l}", folded, 12))
        s.append(carr("int32_t", f"MD_QM{l}", L["mult"], 8))
        s.append(carr("int8_t", f"MD_QS{l}", L["shift"], 16))
        f32_rows.append(f"  {{{w.shape[0]}, {w.shape[1]}, MD_W{l}, MD_B{l}, {int(relu)}}}")
        i8_rows.append(f"  {{{L['n_in']}, {L['n_out']}, MD_QW{l}, MD_QB{l}, MD_QBF{l}, MD_QM{l}, MD_QS{l}, {int(L['per_channel'])}, "
                       f"{L['in_offset']}, {L['out_zp']}, {L['act_min']}, {L['act_max']}}}")
    s.append("static const LayerF32 MD_F32[MD_N_LAYERS] = {\n" + ",\n".join(f32_rows) + "\n};")
    s.append("static const LayerI8 MD_I8[MD_N_LAYERS] = {\n" + ",\n".join(i8_rows) + "\n};\n")
    (FW / "model_data.h").write_text("\n".join(s))


def emit_testvec(frames, feat, logits, qin, qout):
    n, nc = len(frames), logits.shape[1]
    s = ["// GENERATED by python/export.py -- do not edit.", "#pragma once", "#include <stdint.h>",
         f"#define TV_N {n}", f"#define TV_TOL_FEAT {TOL_FEAT}f", f"#define TV_TOL_LOGIT {TOL_LOGIT}f", ""]
    def block(ctype, name, arr, fmt, per_line):
        rows = []
        for r in arr:
            vals = [fmt(v) for v in np.ravel(r)]
            rows.append("  {" + ",\n   ".join(", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)) + "}")
        dims = "".join(f"[{d}]" for d in arr.shape)
        return f"static const {ctype} {name}{dims} = {{\n" + ",\n".join(rows) + "\n};\n"
    s.append(block("int16_t", "TV_X", frames, str, 16))
    s.append(block("float", "TV_FEAT", feat, cf, 8))
    s.append(block("float", "TV_LOGITS", logits, cf, 8))
    s.append(block("int8_t", "TV_QIN", qin, str, 32))
    s.append(block("int8_t", "TV_QOUT", qout, str, 32))
    (FW / "testvec.h").write_text("\n".join(s))


def main():
    model = tf.keras.models.load_model(MODELS / "deploy_binary.keras")
    st = np.load(MODELS / "deploy_binary_std.npz")
    mu, sigma, train_idx = st["mu"], st["sigma"], st["train_idx"]
    d = build(DEFAULT, keep_frames=True)
    Z = ((d["feat"] - mu) / sigma).astype(np.float32)

    rng = np.random.default_rng(0)
    rep_idx = rng.choice(train_idx, size=min(2000, len(train_idx)), replace=False)
    f32_c, i8_c = convert(model, Z[rep_idx])
    (MODELS / "deploy_binary_f32.tflite").write_bytes(f32_c)
    (MODELS / "deploy_binary_int8.tflite").write_bytes(i8_c)
    model_id = hashlib.sha1(i8_c).hexdigest()[:12]

    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    keras_layers = [(l.kernel.numpy(), l.bias.numpy(), l.activation.__name__ == "relu") for l in dense]
    i8_layers = extract_int8_layers(i8_c, [r for _, _, r in keras_layers])
    nc = keras_layers[-1][0].shape[1]

    # Host reference outputs for every window.
    keras_logits = model.predict(Z, verbose=0, batch_size=2048)
    tfl_f32 = run_tflite(interp(f32_c), Z)
    L0 = i8_layers[0]
    qin = quantize_input(Z, L0["in_scale"], L0["in_zp"])
    tfl_i8_ref = run_tflite(interp(i8_c, ref=True), qin)
    tfl_i8_default = run_tflite(interp(i8_c, ref=False), qin)
    emu_i8 = i8_forward(qin, i8_layers)

    test = (d["topology"] == DEPLOY["topology"]) & (d["condition"] == DEPLOY["held_out"])
    y = d["y_binary"]

    def acc_ba(pred, m):
        from sklearn.metrics import balanced_accuracy_score
        return dict(accuracy=float(np.mean(pred[m] == y[m])), balanced_accuracy=float(balanced_accuracy_score(y[m], pred[m])))

    deq = lambda q: (q.astype(np.float32) - i8_layers[-1]["out_zp"]) * i8_layers[-1]["out_scale"]
    agree = dict(
        n_windows_all=int(len(Z)), n_windows_test=int(test.sum()),
        tflite_f32_vs_keras_max_abs_logit_diff=float(np.max(np.abs(tfl_f32 - keras_logits))),
        tflite_f32_vs_keras_pred_agree=float(np.mean(tfl_f32.argmax(1) == keras_logits.argmax(1))),
        emu_vs_tflite_ref_int8_exact_elements=float(np.mean(emu_i8 == tfl_i8_ref)),
        emu_vs_tflite_ref_int8_exact_windows=float(np.mean(np.all(emu_i8 == tfl_i8_ref, axis=1))),
        tflite_default_vs_ref_int8_exact_elements=float(np.mean(tfl_i8_default == tfl_i8_ref)),
        tflite_default_vs_ref_max_abs_diff=int(np.max(np.abs(tfl_i8_default.astype(int) - tfl_i8_ref))),
        int8_vs_f32_pred_agree_all=float(np.mean(tfl_i8_ref.argmax(1) == tfl_f32.argmax(1))),
        int8_dequant_vs_f32_max_abs_logit_diff=float(np.max(np.abs(deq(tfl_i8_ref) - tfl_f32))),
        test_accuracy=dict(keras_f32=acc_ba(keras_logits.argmax(1), test), tflite_f32=acc_ba(tfl_f32.argmax(1), test),
                           tflite_int8=acc_ba(tfl_i8_ref.argmax(1), test)),
        note="int8 inputs are quantized in numpy with the firmware's formula (float32 divide, round half away "
             "from zero) and fed to the int8-I/O model. 'ref' = BUILTIN_REF kernels; 'default' = interpreter's "
             "default resolver (may use XNNPACK).",
    )
    (RES / "int8_agreement_host.json").write_text(json.dumps(agree, indent=1))

    # Test vectors: N_TV windows from the held-out test set, both classes, spread across recordings.
    tv = []
    for c in (0, 1):
        idx = np.flatnonzero(test & (y == c))
        tv += idx[np.linspace(0, len(idx) - 1, N_TV // 2).astype(int)].tolist()
    tv = np.array(tv)
    emit_feature_data(mu, sigma)
    emit_model_data(keras_layers, i8_layers, model_id, nc)
    emit_testvec(d["frames"][tv], Z[tv], tfl_f32[tv], qin[tv], tfl_i8_ref[tv])

    np.savez(CACHE / "hil_windows.npz", frames=d["frames"], y=y, test=test, recording=d["recording"],
             condition=d["condition"], topology=d["topology"], z=Z, tfl_f32=tfl_f32, qin=qin,
             tfl_i8=tfl_i8_ref, out_scale=i8_layers[-1]["out_scale"], out_zp=i8_layers[-1]["out_zp"])
    info = dict(model_id=model_id, deploy=DEPLOY, feature_config=DEFAULT.as_dict(),
                params=int(model.count_params()), tflite_f32_bytes=len(f32_c), tflite_int8_bytes=len(i8_c),
                n_rep=int(len(rep_idx)), test_vector_indices=tv.tolist(),
                int8_layers=[{k: v for k, v in L.items() if k not in ("w", "b")} for L in i8_layers])
    (RES / "export_info.json").write_text(json.dumps(info, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps(agree, indent=1))
    print("model_id", model_id, "params", info["params"], "per_channel", [L["per_channel"] for L in i8_layers])
    return 0


if __name__ == "__main__":
    sys.exit(main())
