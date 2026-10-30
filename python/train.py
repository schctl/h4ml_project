"""Host training and evaluation for E1-E4.

  python train.py e1|e2|e3|e4|all [--seeds 5]

Each experiment writes results/<exp>.json with per-seed metrics. The deployed model (fixed before any
E1b result was seen) is E1b, topology LO, held-out condition T_N, seed 0; saved to results/models/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from dataset import LEAK_TYPES, build, subset
from features import DEFAULT, FeatureConfig, Standardizer
from load import ROOT

RES = ROOT / "results"
MODELS = RES / "models"
DEPLOY = dict(exp="e1b", topology="LO", held_out="T_N", seed=0)


def build_model(n_in, n_out):
    l2 = tf.keras.regularizers.l2(1e-4)  # small weight penalty
    return tf.keras.Sequential([
        tf.keras.Input((n_in,)),
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=l2),
        tf.keras.layers.Dense(n_out),  # 2 outputs (logits); the larger one wins
    ])


def val_split(d, y, rng):
    """Hold out one experiment (both sensors) per class from the training pool for early stopping."""
    val = np.zeros(len(y), bool)
    for c in np.unique(y):
        exps = np.unique(d["experiment"][y == c])
        val |= d["experiment"] == rng.choice(exps)
    return val


def fit(Xtr, ytr, Xval, yval, n_out, seed):
    tf.keras.utils.set_random_seed(seed)
    m = build_model(Xtr.shape[1], n_out)
    m.compile(tf.keras.optimizers.Adam(1e-3),
              tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    cw = compute_class_weight("balanced", classes=np.arange(n_out), y=ytr)
    hist = m.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=200, batch_size=64, verbose=0,
                 class_weight=dict(enumerate(cw)),
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)])
    return m, dict(epochs_run=len(hist.history["loss"]),
                   best_epoch=int(np.argmin(hist.history["val_loss"])) + 1,
                   class_weights=cw.tolist())


def metrics(y, pred, d, class_names):
    k = len(class_names)
    cm = confusion_matrix(y, pred, labels=np.arange(k))
    recall = {class_names[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i in range(k)}
    out = dict(n=int(len(y)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
               accuracy=float(np.mean(y == pred)), recall=recall, confusion=cm.tolist())
    per_cond = {}
    for c in np.unique(d["condition"]):
        m = d["condition"] == c
        cmc = confusion_matrix(y[m], pred[m], labels=np.arange(k))
        per_cond[str(c)] = dict(n=int(m.sum()), accuracy=float(np.mean(y[m] == pred[m])),
                                recall={class_names[i]: (float(cmc[i, i] / cmc[i].sum()) if cmc[i].sum() else None)
                                        for i in range(k)})
    out["per_condition"] = per_cond
    # Recording-level majority vote (each recording = one hydrophone file).
    recs = np.unique(d["recording"])
    ry = np.array([np.bincount(y[d["recording"] == r]).argmax() for r in recs])
    rp = np.array([np.bincount(pred[d["recording"] == r], minlength=k).argmax() for r in recs])
    out["recording_vote"] = dict(n=int(len(recs)), balanced_accuracy=float(balanced_accuracy_score(ry, rp)),
                                 accuracy=float(np.mean(ry == rp)),
                                 confusion=confusion_matrix(ry, rp, labels=np.arange(k)).tolist())
    return out


def run_split(d, tr, te, label_key, class_names, seed, save_as=None, return_pred=False):
    y = d[label_key]
    rng = np.random.default_rng(seed)
    dtr = subset(d, tr)
    val = val_split(dtr, y[tr], rng)
    std = Standardizer().fit(d["feat"][tr][~val])
    Xtr, ytr = std(d["feat"][tr][~val]), y[tr][~val]
    Xval, yval = std(d["feat"][tr][val]), y[tr][val]
    t0 = time.time()
    m, info = fit(Xtr, ytr, Xval, yval, len(class_names), seed)
    info["train_seconds"] = time.time() - t0
    Xte = std(d["feat"][te])
    pred = np.argmax(m.predict(Xte, verbose=0, batch_size=1024), axis=1)
    res = dict(seed=seed, n_train=int(len(ytr)), n_val=int(len(yval)),
               val_experiments=sorted(set(dtr["experiment"][val].tolist())), **info,
               test=metrics(y[te], pred, subset(d, te), class_names))
    if return_pred:
        res["pred"] = pred
    if save_as:
        MODELS.mkdir(parents=True, exist_ok=True)
        m.save(MODELS / f"{save_as}.keras")
        np.savez(MODELS / f"{save_as}_std.npz", mu=std.mu, sigma=std.sigma,
                 train_idx=np.flatnonzero(tr)[~val])
    return res


def summarize(runs):
    ba = [r["test"]["balanced_accuracy"] for r in runs]
    rv = [r["test"]["recording_vote"]["balanced_accuracy"] for r in runs]
    return dict(balanced_accuracy_mean=float(np.mean(ba)), balanced_accuracy_std=float(np.std(ba)),
                balanced_accuracy_min=float(np.min(ba)), balanced_accuracy_max=float(np.max(ba)),
                recording_vote_ba_mean=float(np.mean(rv)), n_seeds=len(runs))


def cross_topology(d, label_key, class_names, seeds, tag):
    out = {}
    for train_topo, test_topo in (("LO", "BR"), ("BR", "LO")):
        tr, te = d["topology"] == train_topo, d["topology"] == test_topo
        runs = []
        for s in seeds:
            runs.append(run_split(d, tr, te, label_key, class_names, s))
            print(f"  {tag} {train_topo}->{test_topo} seed {s}: BA={runs[-1]['test']['balanced_accuracy']:.3f}",
                  flush=True)
        out[f"{train_topo}_to_{test_topo}"] = dict(summary=summarize(runs), runs=runs)
    return out


def loco(d, label_key, class_names, seeds, tag, save_deploy=False):
    """Leave-one-condition-out within each topology. Each fold holds out every experiment (all classes,
    both sensors) of one flow/background condition; fold predictions are pooled per seed."""
    out = {}
    y = d[label_key]
    for topo in ("LO", "BR"):
        in_topo = d["topology"] == topo
        conds = sorted(np.unique(d["condition"][in_topo]))
        runs = []
        for s in seeds:
            pred = np.full(len(y), -1)
            folds = {}
            for c in conds:
                te = in_topo & (d["condition"] == c)
                tr = in_topo & ~te
                save = ("deploy_binary" if save_deploy and (topo, c, s) ==
                        (DEPLOY["topology"], DEPLOY["held_out"], DEPLOY["seed"]) else None)
                r = run_split(d, tr, te, label_key, class_names, s, save_as=save, return_pred=True)
                pred[te] = r.pop("pred")
                folds[str(c)] = r
            pooled = metrics(y[in_topo], pred[in_topo], subset(d, in_topo), class_names)
            runs.append(dict(seed=s, test=pooled, folds=folds))
            print(f"  {tag} {topo} seed {s}: pooled BA={pooled['balanced_accuracy']:.3f}", flush=True)
        out[topo] = dict(conditions=conds, summary=summarize(runs), runs=runs)
    return out


def time_split(d, label_key, class_names, seeds, train_frac=0.7):
    """Within-recording split: first 70% of each recording trains, last 30% tests, with a 1-window gap.
    H1/H2 of an experiment are split at the same relative time, so both sides contain every experiment."""
    n_rec = {r: (d["recording"] == r).sum() for r in np.unique(d["recording"])}
    n = np.array([n_rec[r] for r in d["recording"]])
    cut = np.floor(train_frac * n).astype(int)
    tr = d["win_idx"] < cut - 1
    te = d["win_idx"] >= cut
    runs = []
    for s in seeds:
        runs.append(run_split(d, tr, te, label_key, class_names, s))
        print(f"  e2 seed {s}: BA={runs[-1]['test']['balanced_accuracy']:.3f}", flush=True)
    return dict(train_frac=train_frac, summary=summarize(runs), runs=runs)


def save(name, obj):
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(obj, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["e1", "e1b", "e2", "e3", "e4", "all"])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    seeds = list(range(a.seeds))
    tf.config.experimental.enable_op_determinism()
    bin_names = ["no_leak", "leak"]
    todo = ["e1", "e1b", "e2", "e3", "e4"] if a.exp == "all" else [a.exp]
    d = build(DEFAULT)
    if "e1" in todo:
        r = dict(feature_config=DEFAULT.as_dict(), main=cross_topology(d, "y_binary", bin_names, seeds, "e1"))
        cfg0 = FeatureConfig(f_low=0.0)
        r["ablation_no_lowcut"] = dict(feature_config=cfg0.as_dict(),
                                       **cross_topology(build(cfg0), "y_binary", bin_names, seeds, "e1-nolowcut"))
        save("e1_binary_cross_topology", r)
    if "e1b" in todo:
        save("e1b_binary_loco", dict(feature_config=DEFAULT.as_dict(), deploy=DEPLOY,
                                     **loco(d, "y_binary", bin_names, seeds, "e1b", save_deploy=True)))
    if "e2" in todo:
        save("e2_binary_time_split", dict(feature_config=DEFAULT.as_dict(),
                                          **time_split(d, "y_binary", bin_names, seeds)))
    if "e3" in todo:
        save("e3_leaktype", dict(feature_config=DEFAULT.as_dict(), class_names=LEAK_TYPES,
                                 cross_topology=cross_topology(d, "y_type", LEAK_TYPES, seeds, "e3-xtopo"),
                                 loco=loco(d, "y_type", LEAK_TYPES, seeds, "e3-loco")))
    if "e4" in todo:
        cfg8 = FeatureConfig(fs=8000, n_fft=2048, hop=2048, f_high=4000.0)
        save("e4_samplerate", dict(
            note="E1b (LOCO) protocol. Both configs use 256 ms windows and 64 bands from 20 Hz; "
                 "8 kHz bands extend to 4 kHz.",
            fs4000=dict(feature_config=DEFAULT.as_dict(), **loco(d, "y_binary", bin_names, seeds, "e4-4k")),
            fs8000=dict(feature_config=cfg8.as_dict(), **loco(build(cfg8), "y_binary", bin_names, seeds, "e4-8k"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
