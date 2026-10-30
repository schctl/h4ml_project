"""HIL over USB serial: self-test, then stream windows to the Nano and compare with host TFLite outputs.

  python host/serial_stream.py [--port /dev/ttyACM0] [--subset all|test] [--keep-ble]

Writes results/hil_serial.json (summary) and results/hil_serial_windows.csv (per window).
Stop-and-wait: one window in flight, so the host round-trip time is also measured.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import serial

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


class Link:
    # The mbed core's USB-CDC RX path drops bytes under unpaced 2 KB bursts, so windows are written in
    # `piece`-byte writes separated by `gap_s` (see README / report, E6 notes).
    def __init__(self, port, piece=512, gap_s=0.005):
        self.piece, self.gap_s = piece, gap_s
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.p = proto.Parser()
        self.txt = proto.TextAssembler()
        time.sleep(0.2)
        self.s.reset_input_buffer()

    def frames(self, timeout):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            data = self.s.read(self.s.in_waiting or 1)
            if data:
                for f in self.p.feed(data):
                    yield f

    def command(self, c, args=b"", timeout=5.0, retries=3):
        # A reply can be lost on the serial link while BLE is active, so a timed-out command is re-sent.
        for _ in range(retries):
            self.s.write(proto.cmd(c, args))
            self.txt.parts = {}
            for h, payload in self.frames(timeout):
                if h[0] == proto.FT_TEXT:
                    obj = self.txt.add(h, payload)
                    if obj is not None:
                        return obj
        raise TimeoutError(f"no reply to command {c!r}")

    def infer(self, wid, samples, timeout=2.0):
        t0 = time.perf_counter()
        for fr in proto.window_frames(wid, samples, 2048):
            for i in range(0, len(fr), self.piece):
                self.s.write(fr[i:i + self.piece])
                if self.gap_s:
                    time.sleep(self.gap_s)
        for h, payload in self.frames(timeout):
            if h[0] == proto.FT_RESULT:
                r = proto.parse_result(payload)
                if r["window_id"] == wid:
                    r["rtt_us"] = (time.perf_counter() - t0) * 1e6
                    return r
        return None


def stats(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), mean=float(v.mean()), std=float(v.std()), min=float(v.min()),
                median=float(np.median(v)), p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--subset", choices=["all", "test"], default="all")
    ap.add_argument("--keep-ble", action="store_true", help="leave BLE advertising on during the test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--piece", type=int, default=512)
    ap.add_argument("--gap-ms", type=float, default=5.0)
    a = ap.parse_args()

    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    idx = np.arange(len(h["frames"])) if a.subset == "all" else np.flatnonzero(h["test"])
    link = Link(a.port, a.piece, a.gap_ms / 1000)
    info = link.command("I")
    print("info", info)
    selftest = link.command("S", timeout=20)
    print("selftest", selftest)
    if not a.keep_ble:
        print(link.command("X", b"0"))
    link.command("R")
    mem_before = link.command("M")

    rows, missing = [], []
    t_start = time.monotonic()
    for n, i in enumerate(idx):
        r = link.infer(int(i) & 0xFFFF, h["frames"][i])
        if r is None:
            missing.append(int(i))
            continue
        r["idx"] = int(i)
        rows.append(r)
        if n % 1000 == 0:
            print(f"  {n}/{len(idx)}", flush=True)
    wall = time.monotonic() - t_start
    counters = link.command("G")
    mem_after = link.command("M")
    if not a.keep_ble:
        link.command("X", b"1")

    ii = np.array([r["idx"] for r in rows])
    dev_logits = np.array([r["logits_f32"] for r in rows], np.float32)
    dev_q = np.array([r["q_out"] for r in rows], np.int8)
    dev_pf = np.array([r["pred_f32"] for r in rows])
    dev_pi = np.array([r["pred_i8"] for r in rows])
    host_f = h["tfl_f32"][ii]
    host_q = h["tfl_i8"][ii]
    y = h["y"][ii]
    test = h["test"][ii]

    def acc(pred, m):
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        return dict(n=int(m.sum()), accuracy=float(np.mean(pred[m] == y[m])),
                    balanced_accuracy=float(balanced_accuracy_score(y[m], pred[m])),
                    confusion=confusion_matrix(y[m], pred[m], labels=[0, 1]).tolist())

    summary = dict(
        port=a.port, subset=a.subset, host_write_piece_bytes=a.piece, host_write_gap_ms=a.gap_ms, ble_during_test=bool(a.keep_ble), info=info, selftest=selftest,
        n_sent=int(len(idx)), n_results=len(rows), n_missing=len(missing), wall_s=wall,
        counters=counters, mem_before=mem_before, mem_after=mem_after,
        agreement=dict(
            f32_pred_agree=float(np.mean(dev_pf == host_f.argmax(1))),
            f32_logit_max_abs_diff=float(np.max(np.abs(dev_logits - host_f))),
            f32_logit_mean_abs_diff=float(np.mean(np.abs(dev_logits - host_f))),
            i8_exact_elements=float(np.mean(dev_q == host_q)),
            i8_exact_windows=float(np.mean(np.all(dev_q == host_q, axis=1))),
            i8_max_abs_diff=int(np.max(np.abs(dev_q.astype(int) - host_q))),
            i8_pred_agree=float(np.mean(dev_pi == host_q.argmax(1))),
            n_windows_i8_mismatch=int(np.sum(~np.all(dev_q == host_q, axis=1))),
        ),
        device_test_accuracy=dict(f32=acc(dev_pf, test), int8=acc(dev_pi, test)),
        host_test_accuracy=dict(f32=acc(host_f.argmax(1), test), int8=acc(host_q.argmax(1), test)),
        latency_us={k: stats([r[k] for r in rows]) for k in ("feat_us", "fft_us", "mlp_f32_us", "mlp_i8_us", "rtt_us")},
    )
    tag = f"_{a.tag}" if a.tag else ""
    (RES / f"hil_serial{tag}.json").write_text(json.dumps(summary, indent=1))
    with open(RES / f"hil_serial{tag}_windows.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "y", "test", "pred_f32", "pred_i8", "logit0", "logit1", "q0", "q1", "host_q0", "host_q1",
                    "feat_us", "fft_us", "mlp_f32_us", "mlp_i8_us", "rtt_us"])
        for r, hq in zip(rows, host_q):
            w.writerow([r["idx"], int(h["y"][r["idx"]]), int(h["test"][r["idx"]]), r["pred_f32"], r["pred_i8"],
                        *r["logits_f32"], *r["q_out"], *hq.tolist(), r["feat_us"], r["fft_us"], r["mlp_f32_us"],
                        r["mlp_i8_us"], round(r["rtt_us"], 1)])
    print(json.dumps({k: summary[k] for k in ("n_results", "n_missing", "agreement", "device_test_accuracy")}, indent=1))
    print(json.dumps(summary["latency_us"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
