"""E7: BLE link test (host = central via bleak/BlueZ, Nano = peripheral).

  python host/ble_stream.py [--port /dev/ttyACM0] [--burst 300] [--realtime-s 60] [--chunk 0]

BLE carries only data (window chunks -> RX characteristic, results <- notify). Control and counters go
over USB serial at the same time, so the Nano's view (sequence errors, write arrival times) can be read
without disturbing the link.

Tests:
  burst    : send windows as fast as possible with credit-based flow control (<= --inflight windows
             awaiting a result). An unthrottled flood dropped the link and hung the Nano (see report).
  realtime : send one window every 256 ms (the real-time rate of 4 kHz int16 audio = 8000 B/s payload).
             A window is late if its result arrives > 256 ms after its last chunk was written, dropped if
             no result arrives within 2 s.
Writes results/ble_link.json (+ results/ble_realtime_windows.csv).
"""
import argparse
import asyncio
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto
from serial_stream import Link, stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RX_UUID = "6c6b0002-5a1e-4b1a-9c55-6e616e6f3333"
RES_UUID = "6c6b0003-5a1e-4b1a-9c55-6e616e6f3333"
WIN_PERIOD_S = 1024 / 4000


def host_env():
    def sh(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"error: {e}"
    import bleak
    return dict(os=platform.platform(), python=platform.python_version(),
                bleak=getattr(bleak, "__version__", None) or sh("../.venv/bin/pip show bleak | grep Version"),
                bluetoothctl=sh("bluetoothctl --version"), bluetoothd=sh("/usr/lib/bluetooth/bluetoothd --version"),
                adapter_usb=sh("lsusb | grep -i bluetooth"), controller=sh("bluetoothctl show | head -3"),
                hci_version=sh("hciconfig -a 2>/dev/null | grep -E 'HCI Version|LMP Version'"))


class Session:
    def __init__(self, client, chunk):
        self.c = client
        self.chunk = chunk
        self.results = {}
        self.parser_errors = 0

    def on_notify(self, _, data):
        t = time.perf_counter()
        if len(data) < proto.OVERHEAD or data[:2] != proto.MAGIC:
            self.parser_errors += 1
            return
        h = proto.HDR.unpack_from(data, 2)
        body = bytes(data[2:-2])
        if int.from_bytes(data[-2:], "little") != proto.crc16(body) or h[0] != proto.FT_RESULT:
            self.parser_errors += 1
            return
        r = proto.parse_result(body[proto.HDR.size:])
        r["t_result"] = t
        self.results[r["window_id"]] = r

    async def send_window(self, wid, samples):
        for fr in proto.window_frames(wid, samples, self.chunk):
            await self.c.write_gatt_char(RX_UUID, fr, response=False)
        return time.perf_counter()


async def run(a):
    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    test_idx = np.flatnonzero(h["test"])
    link = Link(a.port)
    print("info", link.command("I"))
    link.command("X", b"1")

    dev = await BleakScanner.find_device_by_name("LeakNano", timeout=15)
    if dev is None:
        raise SystemExit("LeakNano not found while scanning")
    out = dict(host_env=host_env(), device_address=dev.address)
    async with BleakClient(dev) as client:
        try:
            await client._backend._acquire_mtu()  # BlueZ: needed for mtu_size to reflect the negotiated value
        except Exception as e:  # noqa: BLE001
            out["mtu_acquire_error"] = repr(e)
        mtu = client.mtu_size
        chunk = (a.chunk if a.chunk else mtu - 3) - proto.OVERHEAD  # payload bytes per frame
        out.update(att_mtu=mtu, chunk_payload_bytes=chunk - chunk % 2,
                   frames_per_window=int(np.ceil(2048 / (chunk - chunk % 2))))
        print("MTU", mtu, "chunk payload", out["chunk_payload_bytes"], "frames/window", out["frames_per_window"])
        s = Session(client, chunk)
        await client.start_notify(RES_UUID, s.on_notify)
        await asyncio.sleep(1.0)

        # ---- burst ----
        link.command("R")
        n = a.burst
        ids = [int(test_idx[i % len(test_idx)]) for i in range(n)]
        t0 = time.perf_counter()
        for k, i in enumerate(ids):
            # credit-based flow control: at most `inflight` windows without a result
            t_wait = time.perf_counter()
            while k - len(s.results) >= a.inflight and time.perf_counter() - t_wait < 2.0:
                await asyncio.sleep(0.001)
            await s.send_window(k, h["frames"][i])
        t_sent = time.perf_counter()
        await asyncio.sleep(3.0)
        got = [k for k in range(n) if k in s.results]
        t_last = max((s.results[k]["t_result"] for k in got), default=t_sent)
        burst_counters = link.command("G")
        out["burst"] = dict(
            max_inflight_windows=a.inflight, n_windows=n, n_results=len(got), n_dropped=n - len(got),
            send_s=t_sent - t0, payload_bytes_sent=n * 2048,
            write_rate_payload_Bps=n * 2048 / (t_sent - t0),
            goodput_payload_Bps=len(got) * 2048 / (t_last - t0),
            nano_counters=burst_counters)
        print("burst", json.dumps(out["burst"]))
        arr = []
        link.s.write(proto.cmd("A"))
        parts = {}
        for hh, payload in link.frames(5.0):
            if hh[0] == proto.FT_ARRIVALS:
                parts[hh[2]] = payload
                if len(parts) == hh[3]:
                    break
        for k in sorted(parts):
            arr += list(np.frombuffer(parts[k], "<u4"))
        if arr:
            gaps = np.diff(np.array(arr, np.int64)) / 1000.0  # ms
            big = gaps[gaps > 1.0]
            out["burst"]["write_arrival_gaps_ms"] = dict(
                n=int(len(gaps)), frac_gt_1ms=float(np.mean(gaps > 1.0)),
                gt_1ms_median=float(np.median(big)) if len(big) else None,
                gt_1ms_p10=float(np.percentile(big, 10)) if len(big) else None,
                gt_1ms_hist_0p5ms={f"{b:.1f}": int(c) for b, c in zip(*np.unique(np.round(big * 2) / 2, return_counts=True))}
                if len(big) else {})
        s.results.clear()

        # ---- real-time ----
        link.command("R")
        n_rt = int(a.realtime_s / WIN_PERIOD_S)
        rows = []
        t_base = time.perf_counter()
        for k in range(n_rt):
            target = t_base + k * WIN_PERIOD_S
            await asyncio.sleep(max(0.0, target - time.perf_counter()))
            t_start = time.perf_counter()
            t_done = await s.send_window(k, h["frames"][int(test_idx[k % len(test_idx)])])
            rows.append(dict(k=k, idx=int(test_idx[k % len(test_idx)]), start_lag_ms=(t_start - target) * 1e3,
                             send_ms=(t_done - t_start) * 1e3, t_done=t_done))
        await asyncio.sleep(2.0)
        rt_counters = link.command("G")
        lat, late, dropped = [], 0, 0
        for r in rows:
            res = s.results.get(r["k"])
            if res is None or res["t_result"] - r["t_done"] > 2.0:
                r["latency_ms"] = None
                dropped += 1
                continue
            r["latency_ms"] = (res["t_result"] - r["t_done"]) * 1e3
            r["pred_f32"], r["pred_i8"] = res["pred_f32"], res["pred_i8"]
            r["agree_f32"] = int(res["pred_f32"] == int(np.argmax(h["tfl_f32"][r["idx"]])))
            lat.append(r["latency_ms"])
            late += r["latency_ms"] > WIN_PERIOD_S * 1e3
        wall = rows[-1]["t_done"] - t_base
        out["realtime"] = dict(
            duration_s=a.realtime_s, n_windows=n_rt, n_dropped=dropped, n_late=int(late),
            required_payload_Bps=2048 / WIN_PERIOD_S, achieved_payload_Bps=n_rt * 2048 / wall,
            schedule_overrun_windows=int(sum(r["start_lag_ms"] > WIN_PERIOD_S * 1e3 for r in rows)),
            send_ms=stats([r["send_ms"] for r in rows]), start_lag_ms=stats([r["start_lag_ms"] for r in rows]),
            result_latency_ms=stats(lat) if lat else None,
            f32_pred_agree_with_host=float(np.mean([r["agree_f32"] for r in rows if "agree_f32" in r])) if lat else None,
            nano_counters=rt_counters)
        print("realtime", json.dumps({k: v for k, v in out["realtime"].items() if k != "nano_counters"}))
        await client.stop_notify(RES_UUID)

    out["parser_errors"] = s.parser_errors
    tag = f"_{a.tag}" if a.tag else ""
    (RES / f"ble_link{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    with open(RES / f"ble_realtime{tag}_windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "idx", "start_lag_ms", "send_ms", "latency_ms", "pred_f32", "pred_i8", "agree_f32"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--burst", type=int, default=300)
    ap.add_argument("--realtime-s", type=float, default=60.0)
    ap.add_argument("--chunk", type=int, default=0, help="frame size in bytes incl. overhead (0 = MTU-3)")
    ap.add_argument("--inflight", type=int, default=4, help="burst test: max windows awaiting a result")
    ap.add_argument("--tag", default="")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
