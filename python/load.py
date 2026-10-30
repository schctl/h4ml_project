"""Dataset loading and inventory for the Aghashahi et al. hydrophone leak dataset.

Files are headerless little-endian int32 PCM at 8 kHz (per the dataset's reader
snippet). Run as a script to write results/dataset_inventory.csv and
results/dataset_summary.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HYD_DIR = ROOT / "data" / "raw" / "hydrophone" / "Hydrophone"
FS = 8000

NAME_RE = re.compile(r"^(BR|LO)_(CC|LC|GL|OL|NL)_(0\.18 LPS|0\.47 LPS|ND|Transient)_(N|NN)_(H1|H2)$")
FLOW_CODE = {"0.18 LPS": "0.18", "0.47 LPS": "0.47", "ND": "ND", "Transient": "T"}


def read_raw(path):
    """Return the int32 samples of a .raw file."""
    return np.fromfile(path, dtype="<i4")


def to_int16(x32):
    """Exact int32 -> int16 conversion; valid only when the low 16 bits are zero."""
    return (x32 >> 16).astype(np.int16)


def parse_name(stem):
    m = NAME_RE.match(stem)
    if not m:
        return None
    topo, leak, flow, bg, sensor = m.groups()
    return dict(topology=topo, leak=leak, flow=FLOW_CODE[flow], background=bg, sensor=sensor,
                experiment=f"{topo}_{leak}_{FLOW_CODE[flow]}_{bg}")


def recordings():
    """List of dicts (one per hydrophone leak/no-leak file) with metadata and path."""
    out = []
    for p in sorted(HYD_DIR.rglob("*.raw")):
        meta = parse_name(p.stem)
        if meta is None:
            continue
        meta["path"] = p
        out.append(meta)
    return out


def background_files():
    return sorted((HYD_DIR / "Background Noise").glob("*.raw"))


def inventory():
    import pandas as pd

    rows, unparsed = [], []
    for p in sorted(HYD_DIR.rglob("*.raw")):
        x = read_raw(p)
        meta = parse_name(p.stem)
        if meta is None:
            if "Background Noise" not in p.parts:
                unparsed.append(p.name)
            meta = dict(topology="", leak="BG", flow="", background="", sensor=p.stem[-2:],
                        experiment="background")
        low16_zero = bool(np.all((x & 0xFFFF) == 0))
        x16 = (x >> 16).astype(np.float64)
        rows.append(dict(
            file=str(p.relative_to(HYD_DIR)), **{k: meta[k] for k in
            ("topology", "leak", "flow", "background", "sensor", "experiment")},
            bytes=p.stat().st_size, n_samples=len(x), duration_s=len(x) / FS,
            low16_all_zero=low16_zero, int16_min=int(x16.min()), int16_max=int(x16.max()),
            n_clipped=int(np.sum(np.abs(x16) >= 32767)), dc_mean=float(x16.mean()),
            rms_ac=float(np.std(x16)),
        ))
    df = pd.DataFrame(rows)

    # H1 vs H2 pairing check per experiment.
    pairs = []
    for exp, g in df[df.experiment != "background"].groupby("experiment"):
        s = set(g.sensor)
        h1 = g[g.sensor == "H1"]
        h2 = g[g.sensor == "H2"]
        same_len = len(h1) == 1 and len(h2) == 1 and int(h1.n_samples.iloc[0]) == int(h2.n_samples.iloc[0])
        identical = None
        corr = None
        if len(h1) == 1 and len(h2) == 1:
            a = read_raw(HYD_DIR / h1.file.iloc[0]) >> 16
            b = read_raw(HYD_DIR / h2.file.iloc[0]) >> 16
            n = min(len(a), len(b))
            identical = bool(len(a) == len(b) and np.array_equal(a, b))
            corr = float(np.corrcoef(a[:n].astype(float), b[:n].astype(float))[0, 1])
        pairs.append(dict(experiment=exp, sensors=sorted(s), same_length=same_len,
                          identical=identical, corr_h1_h2=corr))
    return df, pairs, unparsed


def main():
    df, pairs, unparsed = inventory()
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    df.to_csv(res / "dataset_inventory.csv", index=False)
    rec = df[df.experiment != "background"]
    summary = dict(
        n_files_total=int(len(df)),
        n_recordings=int(len(rec)),
        n_background=int((df.experiment == "background").sum()),
        n_experiments=int(rec.experiment.nunique()),
        unparsed_names=unparsed,
        counts_by_topology_leak=rec.groupby(["topology", "leak"]).size().unstack(fill_value=0).to_dict(),
        conditions=sorted(set(zip(rec.flow, rec.background))),
        counts_by_condition={f"{f}_{b}": int(n) for (f, b), n in rec.groupby(["flow", "background"]).size().items()},
        all_low16_zero=bool(df.low16_all_zero.all()),
        duration_s=dict(min=float(df.duration_s.min()), max=float(df.duration_s.max()),
                        median=float(df.duration_s.median())),
        n_files_with_clipping=int((df.n_clipped > 0).sum()),
        binary_files=dict(leak=int((rec.leak != "NL").sum()), no_leak=int((rec.leak == "NL").sum())),
        h1_h2_pairs=pairs,
        h1_h2_length_mismatch=[p["experiment"] for p in pairs if not p["same_length"]],
        h1_h2_identical=[p["experiment"] for p in pairs if p["identical"]],
    )
    (res / "dataset_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "h1_h2_pairs"}, indent=1, default=str))
    print(df.groupby(["leak", "flow", "background"]).agg(rms=("rms_ac", "median"),
                                                          dur=("duration_s", "median")).round(1).to_string())
    print("corr H1/H2: min %.3f median %.3f max %.3f" % tuple(
        np.nanpercentile([p["corr_h1_h2"] for p in pairs], [0, 50, 100])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
