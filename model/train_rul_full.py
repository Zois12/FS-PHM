"""Full-data RUL training with a reproducible trajectory or window test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from .architecture import FoundationConfig, PHMFoundationModel, PHMTaskModel, load_foundation_checkpoint, normalize_backbone_ablation
from .data import H5Index, PHMH5Dataset, RUL_DATASETS
from .run_logging import append_event
from .training import (
    build_grad_scaler,
    build_loader,
    cleanup_distributed,
    evaluate_task,
    load_training_checkpoint,
    make_finetune_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_task_epoch,
)


def split_trajectories(index: H5Index, test_fraction: float, seed: int) -> tuple[H5Index, H5Index]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    groups = sorted({ref.group_id for ref in index.records})
    if len(groups) < 2:
        raise ValueError("At least two independent RUL trajectories are required for an 80/20 split")
    random.Random(seed).shuffle(groups)
    test_count = min(max(1, round(len(groups) * test_fraction)), len(groups) - 1)
    test_groups = set(groups[:test_count])
    train = [ref for ref in index.records if ref.group_id not in test_groups]
    test = [ref for ref in index.records if ref.group_id in test_groups]
    return H5Index(train, index.files), H5Index(test, index.files)


def split_rul_data(index: H5Index, test_fraction: float, seed: int, unit: str = "trajectory"):
    if unit == "trajectory":
        return split_trajectories(index, test_fraction, seed)
    if unit != "window":
        raise ValueError("RUL split unit must be trajectory or window")
    if not 0 < test_fraction < 1 or len(index) < 2:
        raise ValueError("Window split requires at least two rows and test-fraction between 0 and 1")
    # Stable identity order makes the same seed independent of indexing order
    # and absolute root relocation. The split uses no target values.
    records = sorted(index.records, key=lambda ref: (ref.group_id, ref.row))
    keys = [(ref.group_id, ref.row) for ref in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Window split requires unique group/row identities")
    positions = list(range(len(records)))
    random.Random(seed).shuffle(positions)
    count = min(max(1, round(len(records) * test_fraction)), len(records) - 1)
    test_positions = set(positions[:count])
    return (H5Index([r for i, r in enumerate(records) if i not in test_positions], index.files),
            H5Index([r for i, r in enumerate(records) if i in test_positions], index.files))


def window_split_identity(train, test):
    def fingerprint(index):
        payload = [(r.dataset, r.group_id, r.row, r.target, r.sample_period)
                   for r in sorted(index.records, key=lambda ref: (ref.group_id, ref.row))]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return {"train_rows_hash": fingerprint(train), "test_rows_hash": fingerprint(test),
            "split_algorithm": "global_random_windows_v1"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RUL on a fixed 80/20 trajectory or random window split")
    parser.add_argument("--dataset", choices=RUL_DATASETS, required=True)
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--root", action="append", help="Explicit H5 root; repeatable")
    parser.add_argument("--target-key", default="rul_norm")
    parser.add_argument("--sample-period", type=float)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-unit", choices=("trajectory", "window"), default="trajectory",
                        help="trajectory holds out bearings; window randomly holds out rows across the entire dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-reference", help="Existing RUL run whose exact split must match; permits hash-verified H5 root relocation")
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument("--checkpoint", help="Optional pretrained foundation checkpoint")
    parser.add_argument("--random-backbone", action="store_true",
                        help="Use --checkpoint architecture/config only, with random foundation weights")
    parser.add_argument("--resume", help="Resume this full-data RUL run from last.pt")
    parser.add_argument("--adaptation", choices=("full", "lora", "linear"), default=None,
                        help="Default full for new runs; resume inherits the saved mode")
    parser.add_argument("--lora-rank", type=int, default=None, help="LoRA rank; inherit foundation config (default 8)")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha; inherit foundation config (default 16)")
    parser.add_argument("--output", default="runs/rul_full_80_20")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=None,
                        help="Optional separate LR for LoRA/unfrozen foundation; --lr controls the RUL head")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--architecture-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--physical-patch-seconds", type=float, default=0.004)
    parser.add_argument("--local-attention-window", type=int, default=9)
    parser.add_argument("--backbone-ablation", type=normalize_backbone_ablation, default=None,
                        help="Contribution knockout; default inherits checkpoint (none for new models)")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rul-loss", choices=("smooth_l1", "mse", "mae"), default="smooth_l1")
    parser.add_argument("--weibull-loss-weight", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.random_backbone and not (args.checkpoint or args.resume):
        raise ValueError("--random-backbone requires --checkpoint as an architecture reference, or --resume")
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.grad_accumulation < 1:
        raise ValueError("epochs, batch sizes, and grad accumulation must be positive")
    context = setup_distributed(args.device)
    seed_everything(args.seed, context.rank)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    index = H5Index.build(
        data_root=args.data_root, task="rul", roots=args.root, datasets=[args.dataset],
        target_key=args.target_key, sample_period=args.sample_period,
        strict=not args.skip_bad_files, show_progress=args.progress and context.is_main,
    )
    if not index.records:
        raise RuntimeError(f"No RUL records found for {args.dataset}")
    if any(ref.target is None for ref in index.records):
        raise ValueError("Every RUL record must have a finite target")
    group_aliases = {}
    if args.resume or args.split_reference:
        from .rul_controls import load_reference, restore_index

        reference = (torch.load(args.resume, map_location="cpu", weights_only=True)["config"]
                     if args.resume else load_reference(args.split_reference))
        for key in ("dataset", "target_key", "test_fraction", "seed", "split_unit"):
            if getattr(args, key) != reference["split"].get(key):
                raise ValueError(f"Current {key} differs from the saved split")
        train_index, test_index, split_report = restore_index(index, reference)
        group_aliases = split_report["group_id_aliases"]
    else:
        train_index, test_index = split_rul_data(index, args.test_fraction, args.seed, args.split_unit)
    split_summary = {
        "dataset": args.dataset,
        "target_key": args.target_key,
        "split_unit": args.split_unit,
        "seed": args.seed,
        "test_fraction": args.test_fraction,
        "train_records": len(train_index),
        "test_records": len(test_index),
        "train_groups": sorted({ref.group_id for ref in train_index.records}),
        "test_groups": sorted({ref.group_id for ref in test_index.records}),
    }
    if args.split_unit == "window":
        split_summary.update(window_split_identity(train_index, test_index))

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
            raise ValueError("--resume requires the same dataset, target, and saved split")
        allowed = {key: value for key, value in saved_model.items() if key in FoundationConfig.__annotations__}
        allowed.setdefault("feature_version", 1)
        foundation = PHMFoundationModel(FoundationConfig(**allowed))
        if args.backbone_ablation is not None and args.backbone_ablation != foundation.config.backbone_ablation:
            raise ValueError("--resume must preserve backbone-ablation")
        saved_args = saved_config.get("args", {})
        args.random_backbone = saved_args.get("random_backbone", False)
        args.checkpoint = saved_args.get("checkpoint", args.checkpoint)
        # Old full-data checkpoints always used full adaptation. Never switch
        # parameter sets while restoring an optimizer state.
        saved_adaptation = saved_args.get("adaptation") or "full"
        saved_rank = saved_args.get("lora_rank") or foundation.config.lora_rank
        saved_alpha = saved_args.get("lora_alpha")
        if saved_alpha is None:
            saved_alpha = foundation.config.lora_alpha
        for key, value in (("adaptation", saved_adaptation), ("lora_rank", saved_rank),
                           ("lora_alpha", saved_alpha), ("backbone_lr", saved_args.get("backbone_lr"))):
            if getattr(args, key) is not None and getattr(args, key) != value:
                raise ValueError(f"--resume must preserve {key}; start a new run from --checkpoint to change adaptation")
            setattr(args, key, value)
        del saved
    elif args.checkpoint:
        foundation = load_foundation_checkpoint(args.checkpoint, device="cpu", load_weights=not args.random_backbone)
    else:
        foundation = PHMFoundationModel(model_config)
    args.backbone_ablation = args.backbone_ablation or foundation.config.backbone_ablation
    foundation.configure_ablation(args.backbone_ablation)
    model_config = foundation.config
    args.adaptation = args.adaptation or "full"
    args.lora_rank = args.lora_rank if args.lora_rank is not None else model_config.lora_rank
    args.lora_alpha = args.lora_alpha if args.lora_alpha is not None else model_config.lora_alpha
    if args.lora_rank < 1 or not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0:
        raise ValueError("lora-rank and lora-alpha must be positive")
    task_model = PHMTaskModel(foundation, task="rul", adaptation=args.adaptation,
                             lora_rank=args.lora_rank, lora_alpha=args.lora_alpha).to(context.device)
    parameter_counts = {
        "total": sum(p.numel() for p in task_model.parameters()),
        "trainable": sum(p.numel() for p in task_model.parameters() if p.requires_grad),
        "lora": sum(p.numel() for n, p in task_model.named_parameters() if p.requires_grad and "lora_" in n),
        "head": sum(p.numel() for p in task_model.head.parameters() if p.requires_grad),
    }
    optimizer = make_finetune_optimizer(task_model, args.lr, args.weight_decay, args.backbone_lr)
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
    run_config = {"model": asdict(model_config), "args": vars(args), "split": split_summary,
                  "parameter_counts": parameter_counts, "split_group_aliases": group_aliases}
    metrics_path = output_dir / "metrics.jsonl"
    session_id = uuid.uuid4().hex[:12]
    if context.is_main:
        print(json.dumps({"adaptation": args.adaptation, "parameter_counts": parameter_counts}), flush=True)
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
            "adaptation": args.adaptation, "parameter_counts": parameter_counts,
            "foundation_initialization": "random" if args.random_backbone or not args.checkpoint else "pretrained",
        })

    progress = (
        tqdm(total=(args.epochs - start_epoch) * len(train_loader), desc="RUL train",
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
                wrapped_model, train_loader, optimizer, scheduler, scaler, context, task="rul",
                accumulation_steps=args.grad_accumulation, grad_clip=args.grad_clip,
                amp=args.amp, amp_dtype=args.amp_dtype, rul_loss_kind=args.rul_loss,
                weibull_loss_weight=args.weibull_loss_weight, progress_callback=on_batch,
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
        tqdm(total=len(test_loader), desc="RUL test", unit="batch", dynamic_ncols=True)
        if args.progress and context.is_main else None
    )
    try:
        test_metrics = evaluate_task(
            task_model, test_loader, "rul", context,
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
    train_dataset.close()
    test_dataset.close()
    cleanup_distributed(context)


if __name__ == "__main__":
    main()
