"""NumPy physical-frequency power descriptors shared by probes and FD loading."""
from __future__ import annotations

import math

import numpy as np


def spectral_features(signal, fs, bins, max_hz):
    """Integrate a relative Hann periodogram into equal-width physical Hz bins."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    if (len(signal) < 8 or not np.isfinite(signal).all() or not math.isfinite(fs) or fs <= 0
            or bins < 2 or not math.isfinite(max_hz) or max_hz <= 0):
        raise ValueError("Power spectrum requires >=8 finite points, positive fs/max_hz and bins >=2")
    if max_hz > fs / 2 + 1e-6:
        raise ValueError(f"Requested {max_hz:g}Hz exceeds this channel's Nyquist {fs / 2:g}Hz")
    centered = signal - signal.mean()
    if np.mean(centered ** 2) <= 1e-24 * max(np.mean(signal ** 2), np.finfo(np.float64).tiny):
        return np.zeros(bins)
    power = np.abs(np.fft.rfft(centered * np.hanning(len(signal)))) ** 2
    power[1:-1 if len(signal) % 2 == 0 else None] *= 2.
    freqs = np.fft.rfftfreq(len(signal), d=1. / fs)
    edges = np.r_[0., (freqs[1:] + freqs[:-1]) / 2., fs / 2.]
    total = power.sum()
    if total <= np.finfo(np.float64).tiny:
        return np.zeros(bins)
    cumulative = np.r_[0., np.cumsum(power / total)]
    grid = np.linspace(0., max_hz, bins + 1)
    bands = np.diff(np.interp(grid, edges, cumulative)).clip(min=0.)
    return np.log1p(bins * bands)
