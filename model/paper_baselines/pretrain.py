"""Mixed FD/RUL label-free pretraining of comparison signal encoders."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import FewShotSelection, H5Index, PHMH5Dataset
from ..fixed_split import digest, record_id
from ..pretrain_sampling import PretrainSampler, sampling_summary
from ..run_logging import append_event
from .data import SignalCache, resize_signal
from .protocol import save_json
from .runner import atomic_checkpoint, restore_rng, seed_all, seed_worker
from .ssl_models import SSL_MODELS, MaskedPretrainer


def build_source_index(args):
    excluded = {name.upper() for name in args.exclude_dataset}
    excluded.update(name.upper() for name in args.holdout_datasets)
    groups = set()
    for path in args.few_shot_manifest:
        selection = FewShotSelection.load(path)
        excluded.update(ref.dataset.upper() for ref in selection.selected)
        groups.update(selection.excluded_group_ids)
    index = H5Index.build(data_root=args.data_root, task="all", roots=args.root, datasets=args.datasets,
                          exclude_datasets=excluded, exclude_groups=groups, exclude_paths=args.exclude_path,
                          include_supervision=False, strict=not args.skip_bad_files,
                          show_progress=not args.no_progress)
    if args.max_records and len(index) > args.max_records:
        chosen = sorted(random.Random(args.seed).sample(range(len(index)), args.max_records))
        index = H5Index([index.records[i] for i in chosen], index.files)
    if not len(index):
        raise ValueError("No pretraining rows after dataset/path/manifest exclusions")
    if any(r.label is not None or r.target is not None for r in index):
        raise ValueError("Pretraining index must not contain supervision")
    return index, sorted(excluded)


def waveform_cache(index, cache_root, progress=True):
    """Store all vibration channels; no label/target-dependent cache identity."""
    used_paths = {r.path for r in index}
    files = {p: [Path(p).stat().st_size, Path(p).stat().st_mtime_ns,
                  list(info.signal_keys), info.n_channels] for p, info in index.files.items()
             if p in used_paths}
    key = digest({"kind": "unlabeled_waveform_v1", "files": files,
                  "rows": [(record_id(r), r.task, r.sample_period) for r in index]})
    root = Path(cache_root) / f"ssl_{key[:24]}"
    if (root / "complete.json").exists():
        cache = SignalCache(root)
        if cache.meta["key"] != key:
            raise ValueError("Pretraining cache identity mismatch")
        return cache
    root.mkdir(parents=True, exist_ok=True)
    lock = root / "building.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Pretraining cache is being built: {root}; prepare once before parallel training")
    os.close(fd)
    raw = PHMH5Dataset(index, mode="eval", robust_normalize=False)
    try:
        offsets = [0]
        for ref in index:
            offsets.append(offsets[-1] + index.files[ref.path].n_channels)
        waves = np.lib.format.open_memmap(root / "waves.npy", mode="w+", dtype="float32", shape=(offsets[-1], 2048))
        for row in tqdm(range(len(index)), desc="Cache unlabeled", unit="window", disable=not progress):
            ref = index.records[row]
            x, _ = raw._load_signal(index.files[ref.path], ref.row)
            if not np.isfinite(x).all():
                raise ValueError(f"Non-finite waveform: {ref.path}, row={ref.row}")
            x = resize_signal(x, 2048)
            x = (x - x.mean(-1, keepdims=True)) / x.std(-1, keepdims=True).clip(1e-6)
            waves[offsets[row]:offsets[row + 1]] = x
        waves.flush()
        np.save(root / "offsets.npy", np.asarray(offsets, dtype=np.int64))
        save_json(root / "complete.json", {"key": key, "records": len(index), "channel_windows": offsets[-1],
                                           "points": 2048, "labels_loaded": False})
        return SignalCache(root)
    finally:
        raw.close()
        lock.unlink(missing_ok=True)


class UnlabeledInputs(Dataset):
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return self.cache.meta["records"]

    def __getitem__(self, i):
        offsets = self.cache.array("offsets")
        # One observed vibration channel per row draw; both FD/RUL share a
        # channel-independent encoder. All real channels remain eligible.
        position = np.random.randint(offsets[i], offsets[i + 1])
        return torch.from_numpy(np.array(self.cache.array("waves")[position:position + 1], copy=True))


def source_identity(name, args):
    root = Path(__file__).parent
    paths = [root / p for p in ("pretrain.py", "ssl_models.py", "adapters.py", "data.py", "runner.py", "registry.py")]
    paths.extend((root.parent / "data.py", root.parent / "pretrain_sampling.py"))
    if name == "moment_community":
        paths.append(root / "hf_configs" / (args.hf_backbone.replace("/", "__") + ".json"))
    if name in {"cnn_lstm", "transformer", "resnet18"}:
        paths.append(root.parent / "comparisons" / "architectures.py")
    else:
        paths.append(root / "vendor" / "sources.json")
    return {str(p.relative_to(root.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def train_one(name, args, index, excluded, cache, sampling):
    seed_all(args.seed)
    directory = Path(args.output) / name
    settings = {key: getattr(args, key) for key in ("epochs", "batch_size", "seed", "weight_decay", "warmup_steps",
                "mask_ratio", "mask_policy", "spectral_weight", "contrastive_weight", "temperature", "hf_backbone", "hf_init", "amp")}
    settings["lr"] = args.lr if args.lr is not None else (1e-5 if name == "moment_community" else 3e-4)
    identity = {"model_name": name, "settings": settings, "cache_key": cache.meta["key"],
                "sampling": sampling, "excluded_datasets": excluded, "code": source_identity(name, args)}
    run_id = digest(identity)
    config = {**identity, "run_id": run_id, "kind": "paper_baseline_pretrain_v1", "args": vars(args),
              "dataset_names": sorted({r.dataset for r in index}), "records": len(index),
              "labels_loaded": False, "objective": "masked_time_spectrum_infonce",
              "method_note": "PHM mixed-corpus pretraining adaptation; not the original UniFault training recipe",
              "initialization": "hf_pretrained" if name == "moment_community" and args.hf_init == "pretrained" else "scratch"}
    last = directory / "last.pt"
    saved = None
    if args.resume and last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        if saved["config"]["run_id"] != run_id:
            raise ValueError("Cannot resume pretraining: corpus, code or training configuration changed")
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Existing pretraining output: {directory}; use --resume or a new output")
    if saved and saved["epoch"] == args.epochs:
        print(f"Already complete: {last}", flush=True)
        return
    model = MaskedPretrainer(name, args.hf_backbone, "random" if saved else args.hf_init).to(args.device)
    # Classification heads do not participate in SSL and are always reset downstream.
    for key, parameter in model.named_parameters():
        if ".head." in key:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=settings["lr"], weight_decay=args.weight_decay)
    start = 0
    if saved:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(args.device)
        start = saved["epoch"]
        restore_rng(saved)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "config.json", config)
    dataset = UnlabeledInputs(cache)
    append_event(directory / "metrics.jsonl", {"event": "start", "model": name, "start_epoch": start + 1,
                  "epochs": args.epochs, "records": len(index), "datasets": config["dataset_names"], "excluded_datasets": excluded})
    total_steps = args.epochs * sampling["batches_per_rank"]
    warmup = min(args.warmup_steps, max(0, total_steps - 1))
    for epoch in range(start, args.epochs):
        model.train()
        sampler = None
        if args.sampling != "records" or args.samples_per_epoch:
            sampler = PretrainSampler(index, sampling, args.seed)
            sampler.set_epoch(epoch)
        batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                             num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                             worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed + epoch))
        elapsed, totals, seen = time.monotonic(), Counter(), 0
        progress = tqdm(batches, desc=f"{name} pretrain {epoch + 1}/{args.epochs}", unit="batch", disable=args.no_progress)
        for batch_id, x in enumerate(progress):
            step = epoch * len(batches) + batch_id + 1
            # Schedule each update from its preceding completed step, keeping
            # the last update nonzero even in a one-step diagnostic run.
            factor = (step / warmup if step <= warmup else
                      0.5 * (1 + math.cos(math.pi * (step - warmup - 1) / max(1, total_steps - warmup))))
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            x = x.to(args.device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and args.device.startswith("cuda")):
                loss, components = model(x, args.mask_ratio, args.mask_policy, args.spectral_weight, args.contrastive_weight, args.temperature)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {name} pretraining loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            totals["loss"] += loss.item() * len(x)
            for key, value in components.items():
                totals[key] += value.item() * len(x)
            seen += len(x)
            progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        metrics = {"event": "epoch", "epoch": epoch + 1, **{k: v / seen for k, v in totals.items()},
                   "lr": optimizer.param_groups[0]["lr"], "records": seen, "elapsed_seconds": time.monotonic() - elapsed}
        atomic_checkpoint(last, model, optimizer, epoch + 1, config)
        append_event(directory / "metrics.jsonl", metrics)
        print(json.dumps(metrics), flush=True)
        del batches
    atomic_checkpoint(directory / "final.pt", model, optimizer, args.epochs, config)


def parser():
    p = argparse.ArgumentParser(description="Label-free mixed FD/RUL pretraining for comparison encoders")
    p.add_argument("--models", nargs="+", choices=SSL_MODELS, default=["unifault", "moment_community"])
    p.add_argument("--data-root", default="splitData")
    p.add_argument("--root", action="append")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--holdout-datasets", nargs="+", default=["CWRU", "UO", "XJTU-SY"], help="Entire downstream datasets held out; default covers all three comparison targets")
    p.add_argument("--exclude-dataset", action="append", default=[])
    p.add_argument("--exclude-path", action="append", default=[])
    p.add_argument("--few-shot-manifest", action="append", default=[])
    p.add_argument("--skip-bad-files", action="store_true")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--sampling", choices=["records", "task-dataset-balanced"], default="task-dataset-balanced")
    p.add_argument("--fd-sampling-fraction", type=float, default=0.5)
    p.add_argument("--samples-per-epoch", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--mask-policy", choices=["random", "last"], default="random")
    p.add_argument("--spectral-weight", type=float, default=0.5)
    p.add_argument("--contrastive-weight", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--hf-backbone", choices=["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "t5-small", "t5-base", "t5-large"], default="google/flan-t5-small")
    p.add_argument("--hf-init", choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache-dir", default="runs/paper_cache")
    p.add_argument("--output", default="runs/paper_pretrain_10ep")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.epochs, args.batch_size) < 1 or min(args.workers, args.max_records, args.samples_per_epoch, args.warmup_steps) < 0:
        raise ValueError("Invalid epoch/batch/worker/sample settings")
    if not math.isfinite(args.mask_ratio) or not 0 < args.mask_ratio < 1 or not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Invalid mask ratio or temperature")
    if any(not math.isfinite(getattr(args, k)) or getattr(args, k) < 0 for k in ("weight_decay", "spectral_weight", "contrastive_weight")):
        raise ValueError("Loss weights and weight decay must be finite and nonnegative")
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("LR must be finite and positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run comparison pretraining with one process per GPU, not torchrun")
    args.models = list(dict.fromkeys(args.models))
    if not args.prepare_only:
        required = {"einops"} if "unifault" in args.models else set()
        if "moment_community" in args.models:
            required.add("transformers")
        for package in sorted(required):
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"Missing {package}; install model/paper_baselines/requirements.txt")
    args.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(torch.device(args.device))
        if args.amp and not torch.cuda.is_bf16_supported():
            raise ValueError("--amp requires BF16 support")
    index, excluded = build_source_index(args)
    sampling = sampling_summary(index, args.sampling, args.fd_sampling_fraction, args.samples_per_epoch, 1, args.batch_size, 1)
    summary = {"dataset_counts": dict(Counter(r.dataset for r in index)), "excluded_datasets": excluded,
               "sampling": sampling, "labels_loaded": False}
    save_json(Path(args.output) / "data_summary.json", summary, immutable=True)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    cache = waveform_cache(index, args.cache_dir, not args.no_progress)
    if args.prepare_only:
        return
    for name in args.models:
        train_one(name, args, index, excluded, cache, sampling)


if __name__ == "__main__":
    main()
