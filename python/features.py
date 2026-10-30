"""Reference implementation of docs/feature_spec.md. The firmware must match this.

All parameters live in FeatureConfig; the default is the deployed configuration.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import resample_poly

from load import read_raw, to_int16


@dataclass(frozen=True)
class FeatureConfig:
    fs: int = 4000          # rate after resampling (8000 = no resampling, used by E4)
    n_fft: int = 1024       # window length in samples (256 ms at 4 kHz)
    hop: int = 1024         # non-overlapping windows
    n_bands: int = 64
    f_low: float = 20.0     # low cut in Hz (0 = keep everything except DC)
    f_high: float = 2000.0  # upper edge of the top band
    eps: float = 1e-6
    spec_version: int = 2   # bump when the computation changes (invalidates the feature cache)

    def as_dict(self):
        return asdict(self)


DEFAULT = FeatureConfig()


def resample_int16(x16, fs_out):
    """8 kHz int16 -> fs_out int16, polyphase with scipy's default Kaiser anti-alias FIR."""
    if fs_out == 8000:
        return x16
    assert 8000 % fs_out == 0
    y = resample_poly(x16.astype(np.float64), 1, 8000 // fs_out)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def load_signal(path, cfg=DEFAULT):
    return resample_int16(to_int16(read_raw(path)), cfg.fs)


def frame(x16, cfg=DEFAULT):
    """(n_windows, n_fft) int16, dropping the incomplete tail."""
    n = 1 + (len(x16) - cfg.n_fft) // cfg.hop
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop * np.arange(n)[:, None]
    return x16[idx]


def band_edges(cfg=DEFAULT):
    """Integer FFT-bin edges (n_bands + 1). Band b covers bins [e[b], e[b+1])."""
    df = cfg.fs / cfg.n_fft
    lo = max(1, int(np.ceil(cfg.f_low / df)))
    hi = int(np.floor(cfg.f_high / df)) + 1
    e = [lo]
    for i in range(cfg.n_bands):
        cur = e[-1]
        bands_left = cfg.n_bands - i
        if bands_left == 1:
            e.append(hi)
            break
        r = (hi / cur) ** (1.0 / bands_left)
        width = max(1, int(round(cur * r - cur)))
        e.append(min(cur + width, hi - (bands_left - 1)))
    e = np.array(e, dtype=np.int32)
    assert len(e) == cfg.n_bands + 1 and np.all(np.diff(e) >= 1)
    return e


def hann(n):
    """Periodic Hann window."""
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)


def window_features(frames16, cfg=DEFAULT, dtype=np.float32):
    """Raw (unstandardized) log-band features, shape (n_windows, n_bands)."""
    # remove the DC (average) value; the mean is computed exactly from the integer sum
    n = frames16.shape[1]
    mean = (frames16.sum(axis=1, dtype=np.int64) / (n * 32768.0)).astype(dtype)
    x = frames16.astype(dtype) / dtype(32768.0)
    x = x - mean[:, None]
    x = x * hann(cfg.n_fft).astype(dtype)
    # FFT -> magnitude -> log
    mag = np.abs(np.fft.rfft(x, axis=1))
    logmag = np.log(mag + cfg.eps)
    # average the log-magnitude inside each band (via a running sum)
    e = band_edges(cfg)
    zeros = np.zeros((len(x), 1))
    c = np.concatenate([zeros, np.cumsum(logmag, axis=1)], axis=1)
    band_sum = c[:, e[1:]] - c[:, e[:-1]]
    return (band_sum / np.diff(e)).astype(np.float32)


class Standardizer:
    def __init__(self, mu=None, sigma=None):
        self.mu, self.sigma = mu, sigma

    def fit(self, f):
        self.mu = f.mean(axis=0).astype(np.float32)
        self.sigma = np.maximum(f.std(axis=0), 1e-6).astype(np.float32)
        return self

    def __call__(self, f):
        return ((f - self.mu) / self.sigma).astype(np.float32)
