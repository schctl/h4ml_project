"""Live demo: stream held-out recordings to the Nano in real time and show its decisions.

  python host/live_demo.py [--port /dev/ttyACM0] [--speed 1.0] [--windows 0]

Each 256 ms window of a recording is sent to the Nano over USB serial; the Nano computes the features, runs the
int8 network, returns its decision and lights its RGB LED (red = leak, green = no leak). The window shows:
  top    - the 64 band features of the recent windows (computed on the desktop for display only),
  middle - the leak probability returned by the Nano, with the true label shaded,
  bottom - the current recording, the Nano's decision and its on-device processing time.
The recordings are the held-out test set of the deployed model (never used for training).
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "python"))
from features import DEFAULT, window_features  # noqa: E402
from serial_stream import Link  # noqa: E402

WINDOW_S = DEFAULT.n_fft / DEFAULT.fs  # 0.256 s
ORDER = ["LO_NL_T_N_H1", "LO_OL_T_N_H1", "LO_NL_T_N_H2", "LO_GL_T_N_H2", "LO_CC_T_N_H1", "LO_LC_T_N_H2"]
NAMES = {"NL": "no leak", "CC": "circumferential crack", "LC": "longitudinal crack", "GL": "gasket leak",
         "OL": "orifice leak"}
HISTORY = 120  # windows shown (about 30 s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--speed", type=float, default=1.0, help="1.0 = real time (one window every 256 ms)")
    ap.add_argument("--windows", type=int, default=0, help="stop after this many windows (0 = all)")
    ap.add_argument("--snapshot", default="", help="save the window as an image when finished, then exit")
    a = ap.parse_args()

    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    std = np.load(ROOT / "results" / "models" / "deploy_binary_std.npz")
    seq = []
    for r in ORDER:
        idx = np.sort(np.flatnonzero(h["test"] & (h["recording"] == r)))
        seq += [(r, int(i)) for i in idx]
    if a.windows:
        seq = seq[: a.windows]

    link = Link(a.port)
    link.command("X", b"0")  # BLE off: cleaner serial link
    link.command("O", b"1")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(11, 7.2))
    fig.canvas.manager.set_window_title("Leak detection on Arduino Nano 33 BLE")
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, 0.9], hspace=0.62)
    ax_f = fig.add_subplot(gs[0])
    ax_p = fig.add_subplot(gs[1])
    ax_s = fig.add_subplot(gs[2])
    ax_s.axis("off")
    fig.suptitle("Water-leak detection running on an Arduino Nano 33 BLE (live)", fontsize=13, fontweight="bold")

    feats = np.full((64, HISTORY), np.nan)
    img = ax_f.imshow(feats, aspect="auto", origin="lower", cmap="gray_r", vmin=-2.5, vmax=2.5,
                      extent=(-HISTORY * WINDOW_S, 0, 0, 64))
    ax_f.set_title("Model input: 64 frequency-band features per 256 ms window", fontsize=10)
    ax_f.set_ylabel("Band (low → high freq.)")
    ax_f.set_xlabel("Time (s)")

    probs = np.full(HISTORY, np.nan)
    truth = np.full(HISTORY, np.nan)
    t_axis = (np.arange(HISTORY) - HISTORY + 1) * WINDOW_S
    truth_fill = [ax_p.fill_between(t_axis, 0, np.nan_to_num(truth), color="0.85", step="mid", lw=0)]
    (line,) = ax_p.plot(t_axis, probs, color="black", linewidth=1.6)
    ax_p.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax_p.set_ylim(-0.03, 1.03)
    ax_p.set_xlim(t_axis[0], t_axis[-1])
    ax_p.set_title("Leak probability from the Nano (shaded = recording is truly a leak)", fontsize=10)
    ax_p.set_ylabel("P(leak)")
    ax_p.set_xlabel("Time (s)")
    status = ax_s.text(0.0, 0.55, "", fontsize=13, family="monospace", va="center")
    stats = ax_s.text(0.0, 0.0, "", fontsize=10, va="center", color="0.25")
    plt.show(block=False)

    n_ok = n_done = 0
    t_next = time.perf_counter()
    try:
        for k, (rec, i) in enumerate(seq):
            res = link.infer(k & 0xFFFF, h["frames"][i])
            if res is None:
                continue
            f = (window_features(h["frames"][i][None, :], DEFAULT)[0] - std["mu"]) / std["sigma"]
            p = 1.0 / (1.0 + np.exp(-(res["logits_f32"][1] - res["logits_f32"][0])))
            y = int(h["y"][i])
            n_done += 1
            n_ok += int(res["pred_i8"] == y)

            feats = np.roll(feats, -1, axis=1)
            feats[:, -1] = f
            probs = np.roll(probs, -1)
            probs[-1] = p
            truth = np.roll(truth, -1)
            truth[-1] = y
            img.set_data(feats)
            line.set_ydata(probs)
            truth_fill[0].remove()
            truth_fill[0] = ax_p.fill_between(t_axis, 0, np.nan_to_num(truth), color="0.85", step="mid", lw=0, zorder=0)
            kind = NAMES[rec.split("_")[1]]
            decision = "LEAK   " if res["pred_i8"] == 1 else "no leak"
            status.set_text(f"Recording: {rec} ({kind})   |   Nano decision: {decision}   |   "
                            f"true: {'leak' if y else 'no leak'}")
            total_ms = (res["feat_us"] + res["mlp_i8_us"]) / 1000
            stats.set_text(f"On-device time this window: {total_ms:.2f} ms (features {res['feat_us'] / 1000:.2f} ms, "
                           f"int8 network {res['mlp_i8_us'] / 1000:.2f} ms) of the 256 ms window   |   "
                           f"correct so far: {n_ok}/{n_done}")
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

            t_next += WINDOW_S / a.speed
            pause = t_next - time.perf_counter()
            if pause > 0:
                plt.pause(pause)
            if not plt.fignum_exists(fig.number):
                break
    finally:
        link.command("X", b"1")
    print(f"done: {n_ok}/{n_done} windows correct")
    if a.snapshot:
        fig.savefig(a.snapshot, dpi=150)
        print("saved", a.snapshot)
    elif plt.fignum_exists(fig.number):
        plt.show()


if __name__ == "__main__":
    main()
