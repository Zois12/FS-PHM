"""Production single-window RUL readout, without experiment-run dependencies."""
from __future__ import annotations

from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .architecture import PHMTaskModel
from .data import PHMH5Dataset, collate_batch
from .training import _move_batch, _rotation_kwargs

AMPLITUDE_NAMES = ("log_rms", "log_std", "log_peak_to_peak", "log_mean_absolute",
                   "log_crest_factor", "log1p_kurtosis")


def amplitude_features(signal):
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("Expected at least 8 finite vibration points")
    centered = x - x.mean()
    std = np.sqrt(np.mean(centered ** 2))
    rms = np.sqrt(np.mean(x ** 2))
    peak, absolute = np.abs(x).max(), np.abs(x).mean()
    kurtosis = np.mean((centered / max(std, 1e-12)) ** 4)
    return np.r_[np.log(np.maximum([rms, std, np.ptp(x), absolute,
                                   peak / max(rms, 1e-12)], 1e-12)), np.log1p(kurtosis)]


def balanced_weights(index):
    """Equal conditions, then equal bearings; no target values are consulted."""
    counts = Counter(r.group_id for r in index)
    conditions = defaultdict(set)
    condition_by_group = {}
    for ref in index:
        name = Path(ref.path).stem
        if "_Bearing" not in name:
            raise ValueError(f"Expected XJTU operating-condition filename: {name}")
        condition = name.split("_Bearing", 1)[0]
        conditions[condition].add(ref.group_id)
        condition_by_group[ref.group_id] = condition
    weights = torch.tensor([1. / (len(conditions[condition_by_group[r.group_id]]) * counts[r.group_id])
                            for r in index], dtype=torch.float64)
    return weights / weights.sum()


class RULWindowDataset(PHMH5Dataset):
    """Stream raw channels, retain amplitude, and bound per-worker open handles."""
    def __init__(self, index, signal_scale=None):
        super().__init__(index, mode="eval", robust_normalize=False)
        self.names = list(index.files[index.records[0].path].signal_keys)
        if len(self.names) != 2:
            raise ValueError("This RUL readout expects two named vibration channels")
        for ref in index:
            info = index.files[ref.path]
            if list(info.signal_keys) != self.names or info.n_channels != 2:
                raise ValueError("Inconsistent vibration channel schema")
        self.signal_scale = np.asarray(signal_scale if signal_scale is not None else [1., 1.],
                                       dtype=np.float32)
        if self.signal_scale.shape != (2,) or not np.isfinite(self.signal_scale).all() or (self.signal_scale <= 0).any():
            raise ValueError("Expected two finite positive TRAIN channel scales")
        self._handles = OrderedDict()

    def _handle(self, path):
        if path not in self._handles:
            if len(self._handles) >= 4:
                _, old = self._handles.popitem(last=False)
                old.close()
            self._handles[path] = h5py.File(path, "r", rdcc_nbytes=64 * 1024**2,
                                          rdcc_nslots=10007)
        return self._handles[path]

    def __getstate__(self):
        state = super().__getstate__()
        state["_handles"] = OrderedDict()
        return state

    def __getitem__(self, position):
        sample = super().__getitem__(position)
        raw = sample["signal"]
        sample["amplitude"] = np.concatenate([amplitude_features(x) for x in raw]).astype(np.float32)
        center = np.median(raw[:, :sample["length"]], axis=1, keepdims=True)
        sample["signal"] = ((raw - center) / self.signal_scale[:, None]).astype(np.float32)
        return sample


def rul_collate(samples):
    batch = collate_batch(samples)
    batch["amplitude"] = torch.from_numpy(np.stack([s["amplitude"] for s in samples]))
    return batch


def data_loader(dataset, workers, device, batch_size=256):
    return DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False,
                      persistent_workers=False, pin_memory=str(device).startswith("cuda"),
                      collate_fn=rul_collate)


def fit_normalization(train, workers, device):
    """Refit historical estimators on TRAIN only; no waveform disk cache."""
    dataset = RULWindowDataset(train)
    values = []
    try:
        for batch in tqdm(data_loader(dataset, workers, device, 512), desc="TRAIN amplitude statistics", unit="batch"):
            values.append(batch["amplitude"])
    finally:
        dataset.close()
    amp = torch.cat(values).double()
    weights = balanced_weights(train)
    scales = []
    for channel in amp[:, 1::6].T:
        ordered, positions = channel.sort()
        cumulative = weights[positions].cumsum(0)
        median = ordered[torch.searchsorted(cumulative, cumulative[-1] * .5)]
        scales.append(float(median.exp().clamp_min(1e-6)))
    return {"signal_scale": scales, "amplitude_mean": amp.mean(0).float().tolist(),
            "amplitude_scale": amp.std(0, unbiased=False).clamp_min(1e-6).float().tolist(),
            "channels": dataset.names,
            "estimator": "TRAIN-only condition/bearing-weighted median(log_std), exp; amplitude mean/std"}


def pool_statistics(hidden, token_mask):
    h, mask = hidden.float(), token_mask.unsqueeze(-1)
    count = mask.sum(1).clamp_min(1)
    mean = h.masked_fill(~mask, 0).sum(1) / count
    variance = (h - mean[:, None]).square().masked_fill(~mask, 0).sum(1) / count
    maximum = h.masked_fill(~mask, -torch.inf).amax(1)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, (variance + 1e-6).sqrt(), maximum), -1)


class RULFusionHead(nn.Module):
    """Historical 111,073-parameter head at embedding width 256."""
    def __init__(self, embed_dim, amplitude_dim=12, dropout=.1):
        super().__init__()
        self.token_projection = nn.Sequential(nn.LayerNorm(embed_dim * 3),
                                              nn.Linear(embed_dim * 3, 128), nn.GELU())
        self.amplitude_projection = nn.Sequential(nn.Linear(amplitude_dim, 32), nn.GELU())
        self.regressor = nn.Sequential(nn.LayerNorm(160), nn.Dropout(dropout),
                                       nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, stats, standardized_amplitude):
        fused = torch.cat((self.token_projection(stats),
                           self.amplitude_projection(standardized_amplitude.clamp(-8, 8))), -1)
        return self.regressor(fused).float().squeeze(-1).sigmoid()


class FrozenRULModel(PHMTaskModel):
    def __init__(self, foundation, normalization, dropout=.1):
        # Retain the historical task-wrapper initialization/RNG consumption.
        super().__init__(foundation, "rul", adaptation="linear", lora_rank=8, lora_alpha=16)
        self.register_buffer("amplitude_mean", torch.tensor(normalization["amplitude_mean"], dtype=torch.float32))
        self.register_buffer("amplitude_scale", torch.tensor(normalization["amplitude_scale"], dtype=torch.float32))
        self.register_buffer("signal_scale", torch.tensor(normalization["signal_scale"], dtype=torch.float32))
        self.head = RULFusionHead(foundation.config.embed_dim, len(self.amplitude_mean), foundation.config.dropout)
        self.head.regressor[1].p = dropout
        for parameter in self.foundation.parameters():
            parameter.requires_grad_(False)
        self.foundation.eval()

    def train(self, mode=True):
        super().train(mode)
        self.foundation.eval()
        return self

    def forward(self, signals, lengths, channel_mask, sample_periods, amplitude, **rotation):
        with torch.no_grad():
            encoded = self.foundation.encode(signals, lengths, channel_mask, sample_periods,
                                              mask_ratio=0., **rotation)
            stats = pool_statistics(encoded["hidden"], encoded["token_mask"])
        amp = (amplitude.float() - self.amplitude_mean) / self.amplitude_scale
        return {"prediction": self.head(stats, amp)}


@torch.inference_mode()
def extract_features(index, foundation, normalization, workers, device, batch_size=256):
    """Feature tensors reside in RAM; TEST is extracted only after fitting."""
    dataset = RULWindowDataset(index, normalization["signal_scale"])
    if dataset.names != normalization["channels"]:
        raise ValueError("Inference channel order differs from TRAIN")
    stats = torch.empty(len(index), foundation.config.embed_dim * 3)
    amp = torch.empty(len(index), 12)
    position = 0
    foundation.eval()
    try:
        for raw in tqdm(data_loader(dataset, workers, device, batch_size), desc="Frozen features in RAM", unit="batch"):
            batch = _move_batch(raw, device)
            encoded = foundation.encode(batch["signals"], batch["lengths"], batch["channel_mask"],
                                        batch["sample_periods"], mask_ratio=0., **_rotation_kwargs(batch))
            values = pool_statistics(encoded["hidden"], encoded["token_mask"])
            n = len(values)
            stats[position:position+n] = values.cpu()
            amp[position:position+n] = raw["amplitude"]
            position += n
    finally:
        dataset.close()
    if position != len(index) or not torch.isfinite(stats).all() or not torch.isfinite(amp).all():
        raise ValueError("Invalid extracted RUL features")
    return {"stats": stats, "amp": amp,
            "targets": torch.tensor([r.target for r in index], dtype=torch.float32)}
