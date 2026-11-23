"""Raw multichannel caches and masked collation for the paper-derived model."""
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ...data import PHMH5Dataset
from ...fixed_split import digest, record_id
from ..data import SignalCache, resize_signal


def build_cache(index, cache_root, points=2048, progress=True):
    used = {r.path for r in index}
    files = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns,
                 info.n_channels, list(info.signal_keys)] for p, info in index.files.items() if p in used}
    key = digest({"kind": "rmgpt_raw_multichannel_v1", "points": points, "files": files,
                  "rows": [(record_id(r), r.task, r.sample_period) for r in index]})
    root = Path(cache_root) / ("rmgpt_raw_" + key[:24])
    if (root / "complete.json").is_file():
        result = SignalCache(root)
        if result.meta["key"] != key:
            raise ValueError("Cache identity mismatch")
        return result
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "building.lock"
    try:
        handle = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Cache build already active/incomplete: {root}; stop its builder before removing its lock")
    os.close(handle)
    dataset = PHMH5Dataset(index, mode="eval", robust_normalize=False)
    try:
        offsets = [0]
        for ref in index:
            offsets.append(offsets[-1] + index.files[ref.path].n_channels)
        waves = np.lib.format.open_memmap(root / "waves.npy", mode="w+", dtype="float32",
                                         shape=(offsets[-1], points))
        for i in tqdm(range(len(index)), desc="Cache RmGPT raw", unit="window", disable=not progress):
            ref = index.records[i]
            x, _ = dataset._load_signal(index.files[ref.path], ref.row)
            if not np.isfinite(x).all():
                raise ValueError(f"Nonfinite waveform: {ref.path}:{ref.row}")
            waves[offsets[i]:offsets[i + 1]] = resize_signal(x, points)
        waves.flush()
        np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        metadata = {"key": key, "records": len(index), "points": points, "labels_loaded": False,
                    "max_channels": max(np.diff(offsets)).item(), "normalization": "none"}
        (root / "complete.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return SignalCache(root)
    finally:
        dataset.close()
        lock.unlink(missing_ok=True)


class Inputs(Dataset):
    def __init__(self, cache, positions=None, targets=None, task=None):
        self.cache = cache
        self.positions = np.arange(cache.meta["records"]) if positions is None else np.asarray(positions)
        self.targets, self.task = targets, task

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, i):
        p = self.positions[i]
        offsets = self.cache.array("offsets")
        x = torch.from_numpy(np.array(self.cache.array("waves")[offsets[p]:offsets[p + 1]], copy=True))
        return x, None if self.targets is None else self.targets[i], i, self.task


def collate(rows):
    channels = max(x.shape[0] for x, _, _, _ in rows)
    points = rows[0][0].shape[-1]
    x = torch.zeros(len(rows), channels, points)
    mask = torch.zeros(len(rows), channels, dtype=torch.bool)
    for i, (wave, _, _, _) in enumerate(rows):
        x[i, :len(wave)] = wave
        mask[i, :len(wave)] = True
    batch = {"x": x, "channels": mask, "indices": torch.tensor([row[2] for row in rows])}
    if rows[0][1] is not None:
        batch["y"] = torch.tensor([row[1] for row in rows],
                                  dtype=torch.long if rows[0][3] == "fd" else torch.float32)
    return batch
