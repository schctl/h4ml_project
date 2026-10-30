"""Per-file clipping and band-energy check (informs the feature spec). Writes results/signal_check.csv."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import welch
sys.path.insert(0, str(Path(__file__).parent))
from load import recordings, background_files, read_raw, to_int16, FS, ROOT

BANDS = [(0, 20), (20, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 4000)]
rows = []
items = [(r["path"], r) for r in recordings()] + [(p, dict(experiment="background", leak="BG", flow="", background="", sensor=p.stem[-2:])) for p in background_files()]
for p, r in items:
    x = to_int16(read_raw(p)).astype(np.float64)
    f, pxx = welch(x - x.mean(), fs=FS, nperseg=4096)
    df_ = f[1] - f[0]
    row = dict(experiment=r["experiment"], leak=r["leak"], flow=r["flow"], background=r["background"], sensor=r["sensor"],
               n_clip=int(np.sum(np.abs(x) >= 32767)), frac_clip=float(np.mean(np.abs(x) >= 32767)))
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        row[f"rms_{lo}_{hi}"] = float(np.sqrt(pxx[m].sum() * df_))
    rows.append(row)
d = pd.DataFrame(rows)
d.to_csv(ROOT / "results" / "signal_check.csv", index=False)
print("clip fraction: files>0 =", int((d.n_clip > 0).sum()), " max frac = %.4f" % d.frac_clip.max(), " files>1%% = %d" % int((d.frac_clip > 0.01).sum()))
print(d[d.n_clip > 0].groupby(["leak"]).size().to_string())
cols = [c for c in d.columns if c.startswith("rms_")]
print(d.groupby(["leak", "flow", "background"])[cols].median().round(0).to_string())
