"""E6 Nano side: latency of baseline vs optimized kernels, with BLE off/on, over USB serial.

  python host/bench_nano.py [--n 200]

For each (opt, ble) setting: self-test, per-stage feature profile ('P'), and n streamed test windows
(on-device micros() timings returned in each result). Writes results/nano_bench.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serial_stream import Link, stats

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ble", default="01", help="BLE settings to measure, e.g. 0, 1 or 01; results are merged into "
                                                "results/nano_bench.json (settings from the same firmware build only)")
    a = ap.parse_args()
    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    idx = np.flatnonzero(h["test"])[: a.n]
    link = Link(a.port)
    import re
    m = re.search(r"Sketch uses (\d+)", (ROOT / "results" / "build_output.txt").read_text())
    out = dict(info=link.command("I"), mem_boot=link.command("M"), settings=[], flash_bytes=int(m.group(1)) if m else None)
    for ble in a.ble:
        link.command("X", ble.encode())
        for opt in ("0", "1"):
            link.command("O", opt.encode())
            st = link.command("S", timeout=30)
            prof = link.command("P", timeout=60)
            link.command("R")
            rows = [link.infer(k, h["frames"][i]) for k, i in enumerate(idx)]
            rows = [r for r in rows if r is not None]
            agree_f32 = float(np.mean([r["pred_f32"] == int(np.argmax(h["tfl_f32"][i])) for r, i in zip(rows, idx)]))
            exact_i8 = float(np.mean([np.array_equal(r["q_out"], h["tfl_i8"][i]) for r, i in zip(rows, idx)]))
            rec = dict(ble_on=int(ble), opt=int(opt), selftest=st, profile_us=prof, n_results=len(rows),
                       pred_f32_agree_host=agree_f32, i8_exact_windows_vs_host=exact_i8,
                       counters=link.command("G"),
                       latency_us={k: stats([r[k] for r in rows]) for k in ("feat_us", "fft_us", "mlp_f32_us", "mlp_i8_us")})
            out["settings"].append(rec)
            print(f"ble={ble} opt={opt} selftest={st['selftest']} n={len(rows)} "
                  + " ".join(f"{k}={v['median']:.0f}" for k, v in rec["latency_us"].items()), flush=True)
    out["mem_after"] = link.command("M")
    link.command("X", b"1")
    link.command("O", b"1")
    path = ROOT / "results" / "nano_bench.json"
    if a.ble != "01" and path.exists():
        prev = json.loads(path.read_text())
        same_build = prev.get("info", {}).get("built") == out["info"]["built"] and prev.get("flash_bytes") == out.get("flash_bytes")
        if same_build:
            keep = [x for x in prev["settings"] if str(x["ble_on"]) not in a.ble]
            out["settings"] = sorted(keep + out["settings"], key=lambda x: (x["ble_on"], x["opt"]))
    (path).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["mem_after"]))


if __name__ == "__main__":
    main()
