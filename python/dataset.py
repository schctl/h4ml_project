"""Window-level dataset: frames, raw features and metadata for every recording, cached per FeatureConfig."""
import hashlib
import json

import numpy as np

from features import DEFAULT, frame, load_signal, window_features
from load import ROOT, recordings

CACHE = ROOT / "data" / "cache"
LEAK_TYPES = ["NL", "CC", "LC", "GL", "OL"]


def _key(cfg):
    return hashlib.sha1(json.dumps(cfg.as_dict(), sort_keys=True).encode()).hexdigest()[:10]


def build(cfg=DEFAULT, keep_frames=False):
    """Returns dict of arrays, one row per window."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"windows_{_key(cfg)}{'_frames' if keep_frames else ''}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    cols = {k: [] for k in ("feat", "frames", "leak", "topology", "flow", "background", "sensor",
                            "experiment", "recording", "win_idx", "t_frac")}
    for r in recordings():
        fr = frame(load_signal(r["path"], cfg), cfg)
        n = len(fr)
        cols["feat"].append(window_features(fr, cfg))
        if keep_frames:
            cols["frames"].append(fr)
        for k in ("leak", "topology", "flow", "background", "sensor", "experiment"):
            cols[k].append(np.full(n, r[k]))
        cols["recording"].append(np.full(n, f"{r['experiment']}_{r['sensor']}"))
        cols["win_idx"].append(np.arange(n))
        cols["t_frac"].append((np.arange(n) + 0.5) / n)
    out = {k: np.concatenate(v) for k, v in cols.items() if v}
    out["condition"] = np.char.add(np.char.add(out["flow"], "_"), out["background"])
    out["y_binary"] = (out["leak"] != "NL").astype(np.int32)
    out["y_type"] = np.array([LEAK_TYPES.index(s) for s in out["leak"]], dtype=np.int32)
    np.savez(path, **out)
    return out


def subset(d, mask):
    return {k: v[mask] for k, v in d.items()}
