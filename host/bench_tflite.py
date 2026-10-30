"""E6 host side: per-window latency of feature extraction and TFLite float32/int8 inference (batch 1).

  python host/bench_tflite.py [--n 2000] [--threads 1]

The host is a reference point, not an edge platform. Writes results/host_bench.json.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import numpy as np
import tensorflow as tf

from features import DEFAULT, window_features
sys.path.insert(0, str(ROOT / "host"))
from serial_stream import stats  # noqa: E402


def cpu_info():
    try:
        model = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:  # noqa: BLE001
        model = platform.processor()
    gov = subprocess.run("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", shell=True,
                         capture_output=True, text=True).stdout.strip()
    return dict(cpu=model, logical_cpus=os.cpu_count(), governor=gov, os=platform.platform(),
                python=platform.python_version(), numpy=np.__version__, tensorflow=tf.__version__)


def time_each(fn, items, warmup=50):
    for x in items[:warmup]:
        fn(x)
    out = []
    for x in items:
        t0 = time.perf_counter_ns()
        fn(x)
        out.append((time.perf_counter_ns() - t0) / 1000.0)
    return stats(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    h = np.load(ROOT / "data" / "cache" / "hil_windows.npz")
    idx = np.flatnonzero(h["test"])
    idx = np.resize(idx, a.n)
    frames = h["frames"][idx]
    z = h["z"][idx]
    q = h["qin"][idx]
    models = ROOT / "results" / "models"
    st = np.load(models / "deploy_binary_std.npz")

    res = dict(env=cpu_info(), n=a.n, threads=a.threads, unit="us per window (batch 1)")
    res["features_numpy"] = time_each(lambda f: (window_features(f[None, :], DEFAULT) - st["mu"]) / st["sigma"], list(frames))

    for name, fname, X in (("f32", "deploy_binary_f32.tflite", z), ("int8", "deploy_binary_int8.tflite", q)):
        content = (models / fname).read_bytes()
        res[f"tflite_{name}_bytes"] = len(content)
        for resolver, kw in (("default", {}),
                             ("builtin_ref", dict(experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF))):
            it = tf.lite.Interpreter(model_content=content, num_threads=a.threads, **kw)
            it.allocate_tensors()
            i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

            def run(x, it=it, i=i, o=o):
                it.set_tensor(i, x[None, :])
                it.invoke()
                return it.get_tensor(o)

            res[f"tflite_{name}_{resolver}"] = time_each(run, list(X))
    (ROOT / "results" / "host_bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
