"""Cached external-model inputs and causal, trajectory-isolated RUL histories."""
from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset
from tqdm import tqdm

from ..data import H5Index, PHMH5Dataset
from ..fixed_split import digest, record_id
from .registry import GENERIC_MODELS

CACHE_VERSION = 2


def resize_signal(signal, points):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[-1] == points:
        return signal.copy()
    divisor = math.gcd(signal.shape[-1], points)
    return resample_poly(signal, points // divisor, signal.shape[-1] // divisor, axis=-1).astype(np.float32)


def descriptors(signal):
    """16 time/spectral descriptors per channel, computed from the native window."""
    x = np.asarray(signal, dtype=np.float64)
    mean = x.mean(-1)
    std = x.std(-1).clip(1e-8)
    centered = (x - mean[:, None]) / std[:, None]
    rms = np.sqrt(np.square(x).mean(-1)).clip(1e-8)
    absolute = np.abs(x).mean(-1).clip(1e-8)
    peak = np.abs(x).max(-1)
    power = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    power = power / power.sum(-1, keepdims=True).clip(1e-12)
    freq = np.linspace(0, 1, power.shape[-1])
    centroid = (power * freq).sum(-1)
    spread = np.sqrt((power * (freq - centroid[:, None]) ** 2).sum(-1))
    bands = [part.sum(-1) for part in np.array_split(power, 4, axis=-1)]
    values = [mean, std, rms, peak, np.ptp(x, axis=-1), (centered**3).mean(-1),
              (centered**4).mean(-1), peak / rms, rms / absolute,
              peak / absolute, centroid, spread, *bands]
    return np.nan_to_num(np.stack(values, axis=-1).reshape(-1), posinf=0, neginf=0).astype(np.float32)


def cache_key(index, task):
    files = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns] for p in index.files}
    return digest({"version": CACHE_VERSION, "task": task, "files": files,
                   "records": [(record_id(r), r.label, r.target, r.sample_period) for r in index]})


class SignalCache:
    """Memory maps reopen lazily in each DataLoader worker, including Windows spawn."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "complete.json").read_text(encoding="utf-8"))
        self._arrays = {}

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(self.directory / f"{name}.npy", mmap_mode="r")
        return self._arrays[name]

    def __getstate__(self):
        return {**self.__dict__, "_arrays": {}}


def build_cache(index, task, cache_root, progress=True):
    key = cache_key(index, task)
    root = Path(cache_root) / f"{task}_{key[:24]}"
    if (root / "complete.json").exists():
        cache = SignalCache(root)
        if cache.meta["key"] != key:
            raise ValueError("Cache identity mismatch")
        return cache
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive build prevents concurrent matrix jobs corrupting a shared cache.
    lock = root / "building.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Cache is being prepared: {root}. Prepare once before parallel jobs; remove building.lock only after a crashed builder has stopped.")
    os.close(fd)
    raw = PHMH5Dataset(index, mode="eval", robust_normalize=False)
    try:
        points, channels = (4096, 1) if task == "fd" else (256, 2)
        waves = np.lib.format.open_memmap(root / "waves.npy", mode="w+", dtype="float32", shape=(len(index), channels, points))
        features = (np.lib.format.open_memmap(root / "features.npy", mode="w+", dtype="float32", shape=(len(index), 32))
                    if task == "rul" else None)
        moment_waves = (np.lib.format.open_memmap(root / "moment_waves.npy", mode="w+", dtype="float32", shape=(len(index), 2, 2048))
                        if task == "rul" else None)
        for i in tqdm(range(len(index)), desc=f"Cache {task}", disable=not progress, unit="window"):
            # Read only waveforms: no RPM, labels or targets are input features.
            ref = index.records[i]
            signal, _ = raw._load_signal(index.files[ref.path], ref.row)
            if not np.isfinite(signal).all():
                raise ValueError(f"Non-finite waveform: {ref.path}, row={ref.row}")
            if task == "rul" and signal.shape[0] != 2:
                raise ValueError("XJTU baselines require both vibration channels")
            signal = signal[:channels]
            if features is not None:
                features[i] = descriptors(signal)
                moment_waves[i] = resize_signal(signal, 2048)
            else:
                # Per-window/channel normalization, also used for FD inference.
                signal = signal - signal.mean(-1, keepdims=True)
                signal /= signal.std(-1, keepdims=True).clip(1e-6)
            waves[i] = resize_signal(signal, points)
        waves.flush()
        if features is not None:
            features.flush()
            moment_waves.flush()
        (root / "complete.json").write_text(json.dumps({"key": key, "version": CACHE_VERSION,
            "records": len(index), "channels": channels, "points": points}, indent=2), encoding="utf-8")
        return SignalCache(root)
    finally:
        raw.close()
        lock.unlink(missing_ok=True)


def positions(index, subset):
    lookup = {record_id(ref): i for i, ref in enumerate(index)}
    return np.array([lookup[record_id(ref)] for ref in subset], dtype=np.int64)


def fit_rul_scaler(cache, train_positions):
    """Only training trajectories contribute to means/scales."""
    result = {}
    for name in ("features", "waves", "moment_waves"):
        total = total2 = None
        count = 0
        for start in range(0, len(train_positions), 1024):
            x = np.asarray(cache.array(name)[train_positions[start:start + 1024]], dtype=np.float64)
            if name.endswith("waves"):
                x = x.transpose(0, 2, 1).reshape(-1, 2)
            subtotal, subtotal2 = x.sum(0), np.square(x).sum(0)
            total = subtotal if total is None else total + subtotal
            total2 = subtotal2 if total2 is None else total2 + subtotal2
            count += len(x)
        mean = total / count
        scale = np.sqrt(np.maximum(total2 / count - mean**2, 0)).clip(1e-6)
        result[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
    return result


class FDInputs(Dataset):
    def __init__(self, cache, row_positions, labels):
        self.cache, self.positions = cache, np.asarray(row_positions)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        x = np.array(self.cache.array("waves")[self.positions[i]], copy=True)
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.long), int(i)


class RULInputs(Dataset):
    def __init__(self, index, cache, row_positions, scaler, model, history=30, history_stride=1, negatives=4):
        self.cache, self.positions, self.scaler, self.model = cache, np.asarray(row_positions), scaler, model
        self.targets = np.array([r.target for r in index], dtype=np.float32)
        self.history, self.history_stride, self.negatives = history, history_stride, negatives
        groups = defaultdict(list)
        for p in self.positions:
            groups[index.records[p].group_id].append(int(p))
        self.groups, self.ends = {}, {}
        for group, items in groups.items():
            items.sort(key=lambda p: (index.records[p].row, Path(index.records[p].path).name))
            self.groups[group] = np.asarray(items, dtype=np.int64)
            for ordinal, p in enumerate(items):
                self.ends[p] = (group, ordinal)

    def __len__(self):
        return len(self.positions)

    def _input(self, position):
        group, end = self.ends[int(position)]
        # Replicate the first observed window; never take future rows or another bearing.
        steps = np.maximum(end - np.arange(self.history - 1, -1, -1) * self.history_stride, 0)
        selected = self.groups[group][steps]
        name = "features" if self.model in {"da_rul", "dual_mixer"} else "waves"
        if self.model in (*GENERIC_MODELS, "moment_community"):
            selected = np.asarray([position])  # Original waveform encoder operates per acquisition window.
            name = "moment_waves"
        x = np.array(self.cache.array(name)[selected], copy=True)
        mean, scale = [np.asarray(self.scaler[name][k], dtype=np.float32) for k in ("mean", "scale")]
        if name.endswith("waves"):
            x = (x - mean[None, :, None]) / scale[None, :, None]
        else:
            x = (x - mean) / scale
        return x[0] if self.model in (*GENERIC_MODELS, "moment_community") else x

    def _negative_positions(self, position):
        group, ordinal = self.ends[int(position)]
        items = self.groups[group]
        offsets = np.arange(len(items))
        keep = np.abs(offsets - ordinal) > 0.25 * len(items)
        candidates = items[keep]
        if not len(candidates):
            raise ValueError("FSGRI needs distinct temporally separated training windows per bearing")
        # Same Gaussian-threshold strategy as the supplied C-MAPSS sampler.
        center = ordinal / max(1, len(items) - 1) * 8 - 4
        values = np.exp(-0.5 * ((np.linspace(-4, 4, len(candidates)) - center) / 0.3) ** 2)
        probability = np.exp(values - values.max())
        probability /= probability.sum()
        return np.random.choice(candidates, self.negatives, replace=len(candidates) < self.negatives, p=probability)

    def __getitem__(self, i):
        p = int(self.positions[i])
        if self.model == "dual_mixer" and self.negatives:
            choices = np.r_[p, self._negative_positions(p)]
            x = np.stack([self._input(int(v)) for v in choices])
            y = self.targets[choices].copy()
        else:
            x, y = self._input(p), self.targets[p]
        return torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32), int(i)
