"""Next-patch RmGPT pretraining, retaining the current PHM corpus and budget."""
import argparse
from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path
import time

import torch
from tqdm import tqdm

from ...fixed_split import digest
from ..pretrain import build_source_index
from ..protocol import save_json
from ...pretrain_sampling import PretrainSampler, sampling_summary
from .architecture import RmGPT, RmGPTConfig
from .common import (atomic_checkpoint, device_setup, loader, parameter_counts, print_event,
                     restore_training, saved_or_restart, seed_all, source_identity)
from .data import Inputs, build_cache


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="splitData")
    p.add_argument("--root", action="append")
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--holdout-datasets", nargs="+", default=["CWRU", "UO", "XJTU-SY"])
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
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--points", type=int, default=2048)
    p.add_argument("--patch", type=int, default=256)
    p.add_argument("--max-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache-dir", default="runs/paper_cache")
    p.add_argument("--output", default="runs/rmgpt_paper_v2/pretrain")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    device_setup(args)
    if args.warmup_steps < 0 or args.max_records < 0:
        raise ValueError("Warmup and max-records must be nonnegative")
    seed_all(args.seed)
    architecture = RmGPTConfig(points=args.points, patch=args.patch, width=args.width, layers=args.layers,
                               heads=args.heads, max_channels=args.max_channels, dropout=args.dropout)
    index, excluded = build_source_index(args)
    sampling = sampling_summary(index, args.sampling, args.fd_sampling_fraction,
                                args.samples_per_epoch, 1, args.batch_size, 1)
    cache = build_cache(index, args.cache_dir, architecture.points, not args.no_progress)
    if cache.meta["max_channels"] > architecture.max_channels:
        raise ValueError("Increase --max-channels to cover all sensor channels")
    settings = {key: getattr(args, key) for key in ("epochs", "batch_size", "seed", "lr", "weight_decay", "warmup_steps", "amp")}
    identity = {"kind": "rmgpt_paper_v2_pretrain", "model_name": "rmgpt_paper_v2",
                "architecture": asdict(architecture), "settings": settings, "cache_key": cache.meta["key"],
                "sampling": sampling, "excluded_datasets": excluded, "code": source_identity()}
    config = {**identity, "run_id": digest(identity), "args": vars(args), "labels_loaded": False,
              "dataset_names": sorted({r.dataset for r in index}), "objective": "next_final_patch_mse",
              "dataset_counts": dict(Counter(r.dataset for r in index)), "initialization": "random_signal_transformer"}
    if args.prepare_only:
        # Preparation uses a separate directory so it cannot contaminate training.
        save_json(Path(args.output + "_prepare") / "data_summary.json", config, immutable=True)
        return
    output = Path(args.output)
    saved = saved_or_restart(output, args.resume)
    if saved and saved["config"]["run_id"] != config["run_id"]:
        raise ValueError("Resume requires identical architecture, corpus, code and training budget")
    if saved and saved["epoch"] == args.epochs:
        print(f"Already complete: {output / 'last.pt'}", flush=True)
        return
    model = RmGPT(architecture).to(args.device)
    model.fault_tokens.requires_grad_(False)
    for name in ("task_tokens", "magnitude_projection", "phase_projection"):
        getattr(model.backbone, name).requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    if saved:
        restore_training(saved, model, optimizer, args.device)
        start = saved["epoch"]
    config["parameters"] = parameter_counts(model)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "config.json", config)
    print_event(output / "metrics.jsonl", {"event": "start", "start_epoch": start + 1,
                "parameters": config["parameters"], "records": len(index), "datasets": config["dataset_names"]})
    dataset = Inputs(cache)
    total_steps = args.epochs * sampling["batches_per_rank"]
    warmup = min(args.warmup_steps, max(0, total_steps - 1))
    for epoch in range(start, args.epochs):
        model.train()
        sampler = None
        if args.sampling != "records" or args.samples_per_epoch:
            sampler = PretrainSampler(index, sampling, args.seed)
            sampler.set_epoch(epoch)
        batches = loader(dataset, args, epoch, shuffle=True, sampler=sampler)
        began, total, seen = time.monotonic(), 0.0, 0
        progress = tqdm(batches, desc=f"RmGPT pretrain {epoch + 1}/{args.epochs}", unit="batch", disable=args.no_progress)
        for i, batch in enumerate(progress):
            step = epoch * len(batches) + i + 1
            factor = step / warmup if step <= warmup else 0.5 * (
                1 + math.cos(math.pi * (step - warmup - 1) / max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            optimizer.zero_grad(set_to_none=True)
            x, channels = batch["x"].to(args.device), batch["channels"].to(args.device)
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=args.amp and x.is_cuda):
                loss, _ = model.pretrain_loss(x, channels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
            seen += len(x)
            progress.set_postfix(loss=f"{loss.item():.4f}")
        atomic_checkpoint(output / "last.pt", model, optimizer, epoch + 1, config)
        print_event(output / "metrics.jsonl", {"event": "epoch", "epoch": epoch + 1, "loss": total / seen,
                    "lr": optimizer.param_groups[0]["lr"], "records": seen, "elapsed_seconds": time.monotonic() - began})
        del batches
    atomic_checkpoint(output / "final.pt", model, optimizer, args.epochs, config)


if __name__ == "__main__":
    main()
