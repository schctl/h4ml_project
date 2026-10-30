"""Course-aligned analyses of the deployed model (all measured / computed, nothing estimated):

1. Classification metrics with leak = positive: precision, recall (sensitivity), specificity, F1, ROC-AUC and a
   threshold sweep, for host TFLite and for the on-device outputs recorded in results/hil_serial_test_windows.csv.
   Also E1b pooled metrics from the per-seed confusion matrices.
2. Model complexity: parameters and MACs per layer, FP32 vs INT8 weight storage, measured effective MAC/s on
   the Nano (MACs / measured median MLP latency).
3. ReLU activation sparsity on the test windows: the MACs a zero-skipping engine could avoid.
4. Pruning sensitivity: global magnitude pruning without fine-tuning, test balanced accuracy.

Writes results/extra_metrics.json and results/threshold_sweep.csv.
"""
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

from load import ROOT
from train import MODELS, RES


def binary_metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), accuracy=float((tp + tn) / len(y)),
                precision=float(prec), recall=float(rec), specificity=float(spec), f1=float(f1),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)))


def from_cm(cm):
    (tn, fp), (fn, tp) = cm
    prec, rec, spec = tp / (tp + fp), tp / (tp + fn), tn / (tn + fp)
    return dict(precision=prec, recall=rec, specificity=spec, f1=2 * prec * rec / (prec + rec))


def main():
    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    test = h["test"]
    y = h["y"][test]
    out = {}

    # ---- 1. metrics, host and device ----
    host_score = (h["tfl_f32"][:, 1] - h["tfl_f32"][:, 0])[test]   # logit margin for "leak"
    host_q = (h["tfl_i8"][:, 1].astype(int) - h["tfl_i8"][:, 0])[test]
    rows = list(csv.DictReader(open(RES / "hil_serial_test_windows.csv")))
    dev_idx = np.array([int(r["idx"]) for r in rows])
    dev_y = np.array([int(r["y"]) for r in rows])
    dev_score = np.array([float(r["logit1"]) - float(r["logit0"]) for r in rows])
    dev_q = np.array([int(r["q1"]) - int(r["q0"]) for r in rows])
    dev_pf = np.array([int(r["pred_f32"]) for r in rows])
    dev_pi = np.array([int(r["pred_i8"]) for r in rows])
    out["deployed_test"] = dict(
        n=int(len(y)), positive_class="leak",
        host_tflite_f32=dict(**binary_metrics(y, (host_score > 0).astype(int)), roc_auc=float(roc_auc_score(y, host_score))),
        host_tflite_int8=dict(**binary_metrics(y, (host_q > 0).astype(int)), roc_auc=float(roc_auc_score(y, host_q))),
        nano_f32=dict(**binary_metrics(dev_y, dev_pf), roc_auc=float(roc_auc_score(dev_y, dev_score)), n=int(len(dev_y))),
        nano_int8=dict(**binary_metrics(dev_y, dev_pi), roc_auc=float(roc_auc_score(dev_y, dev_q)), n=int(len(dev_y))),
        note="argmax decision = threshold 0 on the logit margin (leak - no_leak). int8 AUC uses the integer margin, "
             "which has ties; it is a ranking metric only.")
    # threshold sweep on the on-device float logit margin
    sweep = []
    for thr in np.round(np.arange(-6, 6.01, 0.5), 2):
        m = binary_metrics(dev_y, (dev_score > thr).astype(int))
        sweep.append(dict(threshold=float(thr), **{k: m[k] for k in ("precision", "recall", "specificity", "f1", "balanced_accuracy")}))
    with open(RES / "threshold_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0]))
        w.writeheader()
        w.writerows(sweep)
    out["threshold_sweep_source"] = "results/threshold_sweep.csv (Nano float32 logit margin)"

    # E1b pooled metrics, mean over seeds
    e1b = json.loads((RES / "e1b_binary_loco.json").read_text())
    out["e1b_metrics"] = {}
    for topo in ("LO", "BR"):
        ms = [from_cm(r["test"]["confusion"]) for r in e1b[topo]["runs"]]
        out["e1b_metrics"][topo] = {k: dict(mean=float(np.mean([m[k] for m in ms])), std=float(np.std([m[k] for m in ms])))
                                    for k in ms[0]}

    # ---- 2. complexity ----
    model = tf.keras.models.load_model(MODELS / "deploy_binary.keras")
    dense = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    layers = []
    for l in dense:
        n_in, n_out = l.kernel.shape
        layers.append(dict(n_in=int(n_in), n_out=int(n_out), weights=int(n_in * n_out), biases=int(n_out),
                           macs=int(n_in * n_out)))
    params = sum(L["weights"] + L["biases"] for L in layers)
    macs = sum(L["macs"] for L in layers)
    weights = sum(L["weights"] for L in layers)
    biases = sum(L["biases"] for L in layers)
    nb = json.loads((RES / "nano_bench.json").read_text())
    S = {(s["ble_on"], s["opt"]): s for s in nb["settings"]}
    eff = {}
    for (b, o), s in S.items():
        for k in ("mlp_f32_us", "mlp_i8_us"):
            eff[f"ble{b}_opt{o}_{k[:-3]}"] = dict(median_us=s["latency_us"][k]["median"],
                                                    macs_per_s=macs / (s["latency_us"][k]["median"] * 1e-6),
                                                    cycles_per_mac=64e6 * s["latency_us"][k]["median"] * 1e-6 / macs)
    out["complexity"] = dict(layers=layers, params=params, macs_per_inference=macs,
                             storage_bytes=dict(fp32_all=4 * params, int8_weights_int32_bias=weights + 4 * biases,
                                                fp32_weights=4 * weights, int8_weights=weights),
                             nano_effective_throughput=eff,
                             note="MACs count multiply-accumulates in the three dense layers (bias adds and ReLU "
                                  "excluded, as in the course's convention). cycles_per_mac assumes the 64 MHz core "
                                  "clock reported by the firmware and includes loop overhead and requantization.")

    # ---- 3. ReLU sparsity -> zero-skipping potential ----
    Z = h["z"][test]
    a = Z
    zero = []
    skip_macs = 0.0
    for li, l in enumerate(dense[:-1]):
        a = np.maximum(a @ l.kernel.numpy() + l.bias.numpy(), 0)
        frac0 = float(np.mean(a == 0))
        zero.append(dict(layer=li, units=int(a.shape[1]), frac_zero=frac0))
        skip_macs += np.mean(np.sum(a == 0, axis=1)) * dense[li + 1].kernel.shape[1]
    out["relu_sparsity"] = dict(layers=zero, mean_skippable_macs=float(skip_macs),
                                skippable_frac_of_total=float(skip_macs / macs),
                                note="A zero activation makes all MACs that consume it skippable in the next layer "
                                     "(zero-skipping). Input features are standardized and dense, so layer 1 MACs "
                                     "cannot be skipped this way.")

    # ---- 4. pruning sensitivity (no fine-tuning) ----
    W = [l.get_weights() for l in dense]
    allw = np.concatenate([w[0].ravel() for w in W])
    prune = []
    for frac in (0.0, 0.25, 0.5, 0.75, 0.9):
        thr = np.quantile(np.abs(allw), frac) if frac > 0 else -1
        for l, (k, b) in zip(dense, W):
            l.set_weights([np.where(np.abs(k) > thr, k, 0.0), b])
        pred = model.predict(Z, verbose=0, batch_size=2048).argmax(1)
        nz = sum(int(np.count_nonzero(l.kernel.numpy())) for l in dense)
        prune.append(dict(pruned_frac=frac, nonzero_weights=nz, test_balanced_accuracy=float(balanced_accuracy_score(y, pred))))
    for l, w in zip(dense, W):
        l.set_weights(w)
    out["pruning"] = dict(results=prune, note="Global magnitude pruning of dense kernels, biases kept, no fine-tuning; "
                                             "deployed float model, held-out test windows (host Keras).")
    (RES / "extra_metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("deployed_test", "e1b_metrics")}, indent=1)[:3000])
    print(json.dumps({k: out[k] for k in ("complexity", "relu_sparsity", "pruning")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
