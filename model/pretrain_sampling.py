"""Label-free, reproducible task/dataset quotas for PHM pretraining."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import PureWindowsPath

import torch
from torch.utils.data import Sampler


def allocate(total, probabilities):
    """Largest-remainder rounding, with stable alphabetical tie breaking."""
    raw = {key: total * value for key, value in probabilities.items()}
    counts = {key: math.floor(value) for key, value in raw.items()}
    ordered = sorted(raw, key=lambda key: (-(raw[key] - counts[key]), key))
    for key in ordered[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def sampling_summary(index, mode, fd_fraction, samples_per_epoch, world_size, batch_size, accumulation):
    if mode not in {"records", "task-dataset-balanced"}:
        raise ValueError("Invalid pretraining sampling mode")
    if not math.isfinite(fd_fraction) or not 0 < fd_fraction < 1:
        raise ValueError("--fd-sampling-fraction must be finite and strictly between 0 and 1")
    if samples_per_epoch < 0 or min(world_size, batch_size, accumulation) < 1:
        raise ValueError("Invalid sample budget, world size, batch size or accumulation")
    if not index.records:
        raise ValueError("No pretraining records after exclusions")
    counts = Counter(f"{r.task}:{r.dataset}" for r in index)
    by_task = defaultdict(list)
    for key in sorted(counts):
        task = key.split(":", 1)[0]
        if task not in {"fd", "rul"}:
            raise ValueError("Sampling requires FD/RUL task metadata")
        by_task[task].append(key)
    probabilities = {}
    if mode == "records":
        probabilities = {key: n / len(index) for key, n in counts.items()}
    else:
        for task, keys in by_task.items():
            mass = 1.0 if len(by_task) == 1 else (fd_fraction if task == "fd" else 1 - fd_fraction)
            probabilities.update({key: mass / len(keys) for key in keys})
    requested = samples_per_epoch or len(index)
    local_count = math.ceil(requested / world_size)
    effective = local_count * world_size
    task_mass = {task: sum(probabilities[key] for key in keys) for task, keys in by_task.items()}
    quotas = {}
    if mode == "task-dataset-balanced":
        for task, task_count in allocate(effective, task_mass).items():
            keys = by_task[task]
            quotas.update(allocate(task_count, {key: 1 / len(keys) for key in keys}))
        if any(n == 0 for n in quotas.values()):
            raise ValueError("Epoch sample budget is too small to sample every task/dataset bucket")
    identity = hashlib.sha256()
    # Order matters for indexed sampling and resume. Supervision is never used.
    for r in index:
        info = index.files[r.path]
        identity.update((json.dumps([r.task, r.dataset, PureWindowsPath(r.path).name, r.group_id,
                                    r.row, r.sample_period, info.signal_length,
                                    info.n_channels, list(info.signal_keys)], ensure_ascii=True) + "\n").encode())
    batches = math.ceil(local_count / batch_size)
    return {
        "mode": mode, "fd_fraction_requested": fd_fraction if mode != "records" else None,
        "inventory_hash": identity.hexdigest(), "available_records": len(index),
        "world_size": world_size, "requested_samples_per_epoch": requested,
        "effective_samples_per_epoch": effective, "samples_per_rank": local_count,
        "batches_per_rank": batches, "optimizer_steps_per_epoch": math.ceil(batches / accumulation),
        "draw_method": "shuffle_without_replacement_with_ddp_padding" if mode == "records" and not samples_per_epoch
                       else "with_replacement",
        "buckets": {key: {"available_records": counts[key],
                          "natural_fraction": counts[key] / len(index),
                          "target_fraction": probabilities[key],
                          "global_quota": quotas.get(key)} for key in sorted(counts)},
        "task_target_fractions": task_mass,
        "hash_scope": "ordered row identities and schema; no waveform bytes or labels",
    }


class PretrainSampler(Sampler[int]):
    """Make one global draw plan per epoch, then stride-shard it over ranks.

    Every rank uses the same generator seed, never rank-specific RNG state.
    Sampling with replacement preserves original RUL trajectory adjacency in the
    dataset index; both positives and next windows come from the excluded index.
    """
    def __init__(self, index, summary, seed=42, rank=0):
        if not 0 <= rank < summary["world_size"]:
            raise ValueError("Invalid sampler rank")
        self.summary, self.seed, self.rank, self.epoch = summary, seed, rank, 0
        self.size = len(index)
        buckets = defaultdict(list)
        for i, ref in enumerate(index):
            buckets[f"{ref.task}:{ref.dataset}"].append(i)
        self.buckets = {key: torch.tensor(indices, dtype=torch.long) for key, indices in buckets.items()}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.summary["samples_per_rank"]

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.summary["effective_samples_per_epoch"]
        if self.summary["mode"] == "records":
            indices = torch.randint(self.size, (total,), generator=generator)
        else:
            pieces = []
            for key, data in self.summary["buckets"].items():
                pool = self.buckets[key]
                draws = torch.randint(len(pool), (data["global_quota"],), generator=generator)
                pieces.append(pool[draws])
            indices = torch.cat(pieces)
            indices = indices[torch.randperm(total, generator=generator)]
        return iter(indices[self.rank::self.summary["world_size"]].tolist())


def check_sampling_resume(saved_config, summary, args):
    prior = saved_config.get("pretraining") if isinstance(saved_config, dict) else None
    if prior is None:
        if summary["mode"] != "records" or args.samples_per_epoch:
            raise ValueError("Legacy checkpoint has no sampling metadata; start a new run to change sampling")
        return
    if prior["sampling"] != summary:
        raise ValueError("Resume requires identical pretraining sampling, index and step budget")
    for key in ("seed", "epochs", "batch_size", "grad_accumulation"):
        if prior["args"].get(key) != getattr(args, key):
            raise ValueError(f"Resume must preserve --{key.replace('_', '-')}; start a new run for a different experiment")
