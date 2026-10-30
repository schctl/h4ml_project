"""Host check of firmware/leak_nano/fastlogf.h against numpy float32 log (ulp error). Writes results/fastlogf_check.json."""
import ctypes, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
src = '#include "fastlogf.h"\nvoid run(const float*x, float*y, int n){for(int i=0;i<n;i++) y[i]=fast_logf(x[i]);}\n'
with tempfile.TemporaryDirectory() as d:
    c = Path(d) / "t.c"; so = Path(d) / "t.so"; c.write_text(src)
    subprocess.run(["gcc", "-O2", "-ffp-contract=off", "-fPIC", "-shared", "-I", str(ROOT / "firmware/leak_nano"), str(c), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    rng = np.random.default_rng(0)
    # Range seen by the firmware: |X| + 1e-6 for |X| in [0, ~600]; plus a log-uniform sweep.
    x = np.concatenate([np.exp(rng.uniform(np.log(1e-6), np.log(1e3), 2_000_000)), rng.uniform(1e-6, 2.0, 1_000_000)]).astype(np.float32)
    y = np.empty_like(x)
    P = ctypes.POINTER(ctypes.c_float)
    lib.run(x.ctypes.data_as(P), y.ctypes.data_as(P), len(x))
ref = np.log(x.astype(np.float64))                       # exact reference
ref32 = np.log(x)                                         # numpy float32 (what features.py uses)
ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
err_ulp = np.abs(y.astype(np.float64) - ref) / ulp
res = dict(n=int(len(x)), max_ulp_vs_exact=float(err_ulp.max()), mean_ulp_vs_exact=float(err_ulp.mean()),
           frac_equal_numpy_float32=float(np.mean(y == ref32)), max_abs_diff_numpy_float32=float(np.max(np.abs(y - ref32))),
           numpy_float32_max_ulp_vs_exact=float((np.abs(ref32.astype(np.float64) - ref) / ulp).max()),
           note="Compiled with host gcc -O2 -ffp-contract=off (no FMA), matching the M4F's separate VMUL/VADD.")
(ROOT / "results" / "fastlogf_check.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
