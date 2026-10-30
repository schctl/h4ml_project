"""Diagnostics for the cross-topology result: is it a pipeline bug or a domain shift?
Writes results/diag_shift.json."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from dataset import build
from features import Standardizer, FeatureConfig
from load import ROOT

out = {}
for name, cfg in [("lowcut20", FeatureConfig()), ("nolowcut", FeatureConfig(f_low=0.0))]:
    d = build(cfg); X, y = d["feat"], d["y_binary"]
    r = {}
    for model_name, mk in [("logreg", lambda: LogisticRegression(max_iter=2000, class_weight="balanced")),
                           ("rf", lambda: RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=0))]:
        for tr_t in ("LO", "BR"):
            tr = d["topology"] == tr_t; te = ~tr
            s = Standardizer().fit(X[tr]); m = mk().fit(s(X[tr]), y[tr])
            r[f"{model_name}_train_{tr_t}_trainBA"] = float(balanced_accuracy_score(y[tr], m.predict(s(X[tr]))))
            r[f"{model_name}_{tr_t}_to_other_BA"] = float(balanced_accuracy_score(y[te], m.predict(s(X[te]))))
            # within topology: 6-fold group CV on experiments (H1/H2 together)
            bas = []
            for a, b in GroupKFold(6).split(X[tr], y[tr], d["experiment"][tr]):
                Xa, Xb = X[tr][a], X[tr][b]; s2 = Standardizer().fit(Xa)
                m2 = mk().fit(s2(Xa), y[tr][a]); yb = y[tr][b]
                if len(np.unique(yb)) == 2: bas.append(balanced_accuracy_score(yb, m2.predict(s2(Xb))))
            r[f"{model_name}_within_{tr_t}_groupcv_BA_mean"] = float(np.mean(bas)); r[f"{model_name}_within_{tr_t}_nfolds"] = len(bas)
    # mean feature profile per topology x class, to show the shift
    prof = {f"{t}_{c}": d["feat"][(d["topology"] == t) & (d["y_binary"] == c)].mean(0).round(2).tolist()
            for t in ("LO", "BR") for c in (0, 1)}
    r["mean_profile_band_0_8_16_32_48_63"] = {k: [v[i] for i in (0, 8, 16, 32, 48, 63)] for k, v in prof.items()}
    out[name] = r
(ROOT / "results" / "diag_shift.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
