"""Full-data FD training with a class-stratified, group-level test split."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .architecture import FoundationConfig, PHMFoundationModel, PHMTaskModel, load_foundation_checkpoint
from .data import FD_DATASETS, H5Index, PHMH5Dataset
from .run_logging import append_event
from .training import (
    build_grad_scaler,
    build_loader,
    cleanup_distributed,
    evaluate_task,
    load_training_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_task_epoch,
)


def split_groups_by_class(index: H5Index, test_fraction: float, seed: int) -> tuple[H5Index, H5Index, list[int]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    group_labels: dict[str, int] = {}
    for ref in index.records:
        if ref.label is None:
            raise ValueError("Every FD record must have a label")
        label = int(ref.label)
        previous = group_labels.setdefault(ref.group_id, label)
        if previous != label:
            raise ValueError(f"FD group {ref.group_id!r} contains multiple labels; cannot stratify by group")

    class_groups: dict[int, list[str]] = defaultdict(list)
    for group_id, label in group_labels.items():
        class_groups[label].append(group_id)
    rng = random.Random(seed)
    test_groups: set[str] = set()
    for label in sorted(class_groups):
        groups = sorted(class_groups[label])
        if len(groups) < 2:
            raise ValueError(f"FD class {label} needs at least two independent groups for a held-out test")
        rng.shuffle(groups)
        test_count = min(max(1, round(len(groups) * test_fraction)), len(groups) - 1)
        test_groups.update(groups[:test_count])

    labels = sorted(class_groups)
    label_ids = {label: class_id for class_id, label in enumerate(labels)}
    train = [replace(ref, label=label_ids[int(ref.label)]) for ref in index.records if ref.group_id not in test_groups]
    test = [replace(ref, label=label_ids[int(ref.label)]) for ref in index.records if ref.group_id in test_groups]
    return H5Index(train, index.files), H5Index(test, index.files), labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train FD on all windows from stratified groups; test once on held-out groups")
    parser.add_argument("--dataset", choices=FD_DATASETS, required=True)
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--root", action="append", help="Explicit H5 root; repeatable")
    parser.add_argument("--sample-period", type=float)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument("--checkpoint", help="Optional pretrained foundation checkpoint")
    parser.add_argument("--resume", help="Resume this full-data FD run from last.pt")
    parser.add_argument("--output", default="runs/fd_full_80_20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--architecture-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--physical-patch-seconds", type=float, default=0.004)
    parser.add_argument("--local-attention-window", type=int, default=9)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.grad_accumulation < 1:
        raise ValueError("epochs, batch sizes, and grad accumulation must be positive")
    context = setup_distributed(args.device)
    seed_everything(args.seed, context.rank)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    index = H5Index.build(
        data_root=args.data_root, task="fd", roots=args.root, datasets=[args.dataset],
        sample_period=args.sample_period, strict=not args.skip_bad_files,
        show_progress=args.progress and context.is_main,
    )
    if not index.records:
        raise RuntimeError(f"No FD records found for {args.dataset}")
    train_index, test_index, original_labels = split_groups_by_class(index, args.test_fraction, args.seed)
    train_counts = Counter(int(ref.label) for ref in train_index.records)
    test_counts = Counter(int(ref.label) for ref in test_index.records)
    split_summary = {
        "dataset": args.dataset,
        "split_unit": "group_stratified_by_class",
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_groups": sorted({ref.group_id for ref in train_index.records}),
        "test_groups": sorted({ref.group_id for ref in test_index.records}),
        "num_classes": len(original_labels),
        "classes": [
            {"class_id": class_id, "original_label": label,
             "train_records": train_counts[class_id], "test_records": test_counts[class_id]}
            for class_id, label in enumerate(original_labels)
        ],
    }

    train_dataset = PHMH5Dataset(train_index, mode="finetune", robust_normalize=True)
    test_dataset = PHMH5Dataset(test_index, mode="eval", robust_normalize=True)
    train_loader, train_sampler = build_loader(
        train_dataset, args.batch_size, shuffle=True, workers=args.workers, context=context, seed=args.seed
    )
    test_loader, _ = build_loader(
        test_dataset, args.eval_batch_size, shuffle=False, workers=args.workers,
        context=context, seed=args.seed, distributed_eval=True,
    )

    model_config = FoundationConfig(
        embed_dim=args.embed_dim, patch_size=args.patch_size, patch_stride=args.patch_stride,
        layers=args.layers, attention_heads=args.attention_heads, dropout=args.dropout,
        feature_version=args.architecture_version,
        physical_patch_seconds=args.physical_patch_seconds,
        local_attention_window=args.local_attention_window,
    )
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_config = saved.get("config", {}) if isinstance(saved, dict) else {}
        saved_model = saved_config.get("model") if isinstance(saved_config, dict) else None
        if not isinstance(saved_model, dict) or saved_config.get("split") != split_summary:
            raise ValueError("--resume requires the same dataset, labels, and group split")
        allowed = {key: value for key, value in saved_model.items() if key in FoundationConfig.__annotations__}
        allowed.setdefault("feature_version", 1)
        foundation = PHMFoundationModel(FoundationConfig(**allowed))
    elif args.checkpoint:
        foundation = load_foundation_checkpoint(args.checkpoint, device=context.device)
    else:
        foundation = PHMFoundationModel(model_config)
    model_config = foundation.config
    task_model = PHMTaskModel(foundation, task="fd", num_classes=len(original_labels), adaptation="full").to(context.device)
    optimizer = make_optimizer(task_model, args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accumulation)
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_steps)
    scaler = build_grad_scaler(args.amp and args.amp_dtype == "fp16" and context.device.type == "cuda")
    start_epoch = 0
    global_step = 0
    train_metrics: dict[str, float] = {}
    if args.resume:
        completed_epoch, global_step, train_metrics = load_training_checkpoint(
            args.resume, task_model, optimizer, scheduler, scaler, device=context.device
        )
        start_epoch = completed_epoch + 1
    wrapped_model = (
        DistributedDataParallel(
            task_model, device_ids=[context.local_rank] if context.device.type == "cuda" else None
        ) if context.is_distributed else task_model
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": asdict(model_config), "args": vars(args), "split": split_summary}
    metrics_path = output_dir / "metrics.jsonl"
    session_id = uuid.uuid4().hex[:12]
    if context.is_main:
        (output_dir / "config.json").write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_dir / "split_summary.json").write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        append_event(metrics_path, {
            "event": "start", "session_id": session_id, "start_epoch": start_epoch + 1,
            "total_epochs": args.epochs, "train_records": len(train_index),
            "test_records": len(test_index), "batches_per_epoch": len(train_loader),
            "resume": args.resume,
        })

    progress = (
        tqdm(total=(args.epochs - start_epoch) * len(train_loader), desc="FD train",
             unit="batch", dynamic_ncols=True, mininterval=1.0)
        if args.progress and context.is_main and start_epoch < args.epochs else None
    )
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.monotonic()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if progress is not None:
                progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", phase="train")

                def on_batch(loss: float) -> None:
                    progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.4f}", refresh=False)
                    progress.update()
            else:
                on_batch = None
            train_metrics = train_task_epoch(
                wrapped_model, train_loader, optimizer, scheduler, scaler, context, task="fd",
                accumulation_steps=args.grad_accumulation, grad_clip=args.grad_clip,
                amp=args.amp, amp_dtype=args.amp_dtype, progress_callback=on_batch,
            )
            global_step += steps_per_epoch
            if context.is_main:
                if progress is not None:
                    progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", phase="save")
                save_checkpoint(
                    output_dir / "last.pt", task_model, optimizer, scheduler, scaler,
                    epoch, global_step, run_config, train_metrics,
                )
                append_event(metrics_path, {
                    "event": "epoch", "session_id": session_id, "epoch": epoch + 1,
                    "global_step": global_step,
                    "elapsed_seconds": round(time.monotonic() - epoch_started, 3),
                    "lr": optimizer.param_groups[0]["lr"], **train_metrics,
                })
                message = json.dumps({"epoch": epoch + 1, **train_metrics}, ensure_ascii=False)
                tqdm.write(message) if progress is not None else print(message, flush=True)
    finally:
        if progress is not None:
            progress.close()

    test_progress = (
        tqdm(total=len(test_loader), desc="FD test", unit="batch", dynamic_ncols=True)
        if args.progress and context.is_main else None
    )
    try:
        test_metrics = evaluate_task(
            task_model, test_loader, "fd", context,
            progress_callback=test_progress.update if test_progress is not None else None,
        )
    finally:
        if test_progress is not None:
            test_progress.close()
    if context.is_main:
        result = {"epoch": args.epochs, **{f"test_{key}": value for key, value in test_metrics.items()}}
        (output_dir / "test_metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_checkpoint(
            output_dir / "final.pt", task_model, optimizer, scheduler, scaler,
            args.epochs - 1, global_step, run_config, {**train_metrics, **result},
        )
        append_event(metrics_path, {"event": "test", "session_id": session_id, **result})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
