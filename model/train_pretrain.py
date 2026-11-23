"""Distributed self-supervised pretraining entry point.

Run from the repository root, for example:
  torchrun --standalone --nproc_per_node=4 -m model.train_pretrain \
      --data-root splitData --few-shot-manifest runs/cwru_1shot.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on the data-only workstation
    raise SystemExit(
        "PyTorch is required for training. Install model/requirements.txt on the GPU server."
    ) from exc
from torch.nn.parallel import DistributedDataParallel

from .architecture import FoundationConfig, PHMFoundationModel, load_foundation_checkpoint, normalize_backbone_ablation
from .data import FewShotSelection, H5Index, PHMH5Dataset, write_index_summary
from .losses import PretrainingLossConfig
from .run_logging import append_event
from .pretrain_sampling import PretrainSampler, check_sampling_resume, sampling_summary
from .training import (
    build_grad_scaler,
    build_loader,
    cleanup_distributed,
    load_training_checkpoint,
    make_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_pretraining_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHM Physics-MoE masked pretraining")
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--task", choices=("all", "fd", "rul"), default="all")
    parser.add_argument("--root", action="append", help="Explicit H5 file/directory root; repeatable")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to include")
    parser.add_argument("--target-key", default="rul_norm")
    parser.add_argument("--sample-period", type=float, default=None)
    parser.add_argument(
        "--few-shot-manifest",
        help="Exclude the manifest target dataset and selected groups from this pretraining source index",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        help="Exclude an entire dataset (repeatable); manifests also hold out their target datasets",
    )
    parser.add_argument(
        "--exclude-manifest-datasets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With a few-shot manifest, hold out its complete target dataset (default: true)",
    )
    parser.add_argument("--exclude-path", action="append", default=[])
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Optional deterministic cap after indexing, useful for a server smoke test (0 = all)",
    )
    parser.add_argument("--output", default="runs/phm_pretrain")
    parser.add_argument("--sampling", choices=("records", "task-dataset-balanced"), default="records",
                        help="Natural record shuffle, or label-free task quotas and equal datasets within each task")
    parser.add_argument("--fd-sampling-fraction", type=float, default=0.5,
                        help="FD share when balanced sampling includes both tasks (default 0.5); RUL receives the rest")
    parser.add_argument("--samples-per-epoch", type=int, default=0,
                        help="Global draw budget, 0 uses indexed row count; positive values sample with replacement")
    parser.add_argument("--prepare-only", action="store_true", help="Write index/sampling/step summaries without loading a model or training")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--patch-stride", type=int, default=128)
    parser.add_argument("--architecture-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--backbone-ablation", type=normalize_backbone_ablation, default="none",
                        help="Apply fixed knockouts throughout pretraining; join multiple names with +")
    parser.add_argument("--physical-patch-seconds", type=float, default=0.004)
    parser.add_argument("--local-attention-window", type=int, default=9)
    parser.add_argument("--channel-fusion", choices=("mean", "attention"), default="mean")
    parser.add_argument("--zero-channel-policy", choices=("keep", "drop-zero"), default="keep")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask-ratio", type=float, default=0.4)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument("--monotonicity-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.epochs, args.batch_size, args.grad_accumulation) < 1 or args.samples_per_epoch < 0:
        raise ValueError("Epochs, batch size and accumulation must be positive; sample budget must be nonnegative")
    context = setup_distributed(args.device)
    seed_everything(args.seed, context.rank)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    selection = FewShotSelection.load(args.few_shot_manifest) if args.few_shot_manifest else None
    if selection and args.task in {"fd", "rul"} and selection.task != args.task:
        raise ValueError(f"Manifest task {selection.task} does not match --task {args.task}")
    excluded_groups = selection.excluded_group_ids if selection else []
    excluded_datasets = set(args.exclude_dataset)
    if selection and args.exclude_manifest_datasets:
        excluded_datasets.update(ref.dataset for ref in selection.selected)
    index = H5Index.build(
        data_root=args.data_root,
        task=args.task,
        roots=args.root,
        datasets=args.datasets,
        target_key=args.target_key,
        sample_period=args.sample_period,
        exclude_paths=args.exclude_path,
        exclude_groups=excluded_groups,
        exclude_datasets=excluded_datasets,
        strict=not args.skip_bad_files,
        include_supervision=False,
        show_progress=args.progress and context.is_main,
    )
    if args.max_records > 0 and len(index.records) > args.max_records:
        rng = random.Random(args.seed)
        index = H5Index(rng.sample(index.records, args.max_records), index.files)
    summary = sampling_summary(index, args.sampling, args.fd_sampling_fraction,
                               args.samples_per_epoch, context.world_size, args.batch_size, args.grad_accumulation)
    summary["optimizer_steps_total"] = summary["optimizer_steps_per_epoch"] * args.epochs
    summary["seed"] = args.seed
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        check_sampling_resume(saved.get("config", {}), summary, args)
    output_dir = Path(args.output)
    if args.sampling != "records" and not args.resume and output_dir.exists() and any(output_dir.glob("*.pt")):
        raise ValueError("Balanced pretraining needs a new output directory; use --resume only for the same balanced experiment")
    if context.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_index_summary(index, output_dir / "data_summary.json")
        (output_dir / "sampling_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"sampling": summary}, ensure_ascii=False), flush=True)
    if args.prepare_only:
        cleanup_distributed(context)
        return
    sampler_override = (
        PretrainSampler(index, summary, args.seed, context.rank)
        if args.sampling != "records" or args.samples_per_epoch else None
    )
    dataset = PHMH5Dataset(index, mode="pretrain", robust_normalize=True, include_metadata=True)
    loader, sampler = build_loader(
        dataset, args.batch_size, shuffle=True, workers=args.workers, context=context, seed=args.seed,
        sampler_override=sampler_override,
    )
    if len(loader) == 0:
        raise RuntimeError("No pretraining batches were found after exclusions")

    model_config = FoundationConfig(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        layers=args.layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        mask_ratio=args.mask_ratio,
        feature_version=args.architecture_version,
        physical_patch_seconds=args.physical_patch_seconds,
        local_attention_window=args.local_attention_window,
        channel_fusion=args.channel_fusion,
        zero_channel_policy=args.zero_channel_policy,
        backbone_ablation=args.backbone_ablation,
    )
    base_model = (
        load_foundation_checkpoint(args.resume, device="cpu").to(context.device)
        if args.resume else PHMFoundationModel(model_config).to(context.device)
    )
    model_config = base_model.config
    if model_config.backbone_ablation != args.backbone_ablation:
        raise ValueError("Resume must preserve the pretrained backbone ablation")
    args.channel_fusion = model_config.channel_fusion
    args.zero_channel_policy = model_config.zero_channel_policy
    optimizer = make_optimizer(base_model, args.lr, args.weight_decay)
    steps_per_epoch = math.ceil(len(loader) / max(args.grad_accumulation, 1))
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_steps)
    scaler = build_grad_scaler(args.amp and args.amp_dtype == "fp16" and context.device.type == "cuda")
    start_epoch = 0
    global_step = 0
    if args.resume:
        start_epoch, global_step, _ = load_training_checkpoint(
            args.resume, base_model, optimizer, scheduler, scaler, device=context.device
        )
        start_epoch += 1
    model = (
        DistributedDataParallel(base_model, device_ids=[context.local_rank] if context.device.type == "cuda" else None)
        if context.is_distributed
        else base_model
    )
    loss_config = PretrainingLossConfig(
        spectral_weight=args.spectral_weight,
        monotonicity_weight=args.monotonicity_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": asdict(model_config), "loss": asdict(loss_config), "args": vars(args)}
    # Preserve the flat FoundationConfig contract used by downstream loaders.
    checkpoint_config = {**asdict(model_config), "pretraining": {"sampling": summary, "args": vars(args)}}
    session_id = uuid.uuid4().hex[:12]
    metrics_path = output_dir / "metrics.jsonl"
    if context.is_main:
        (output_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        append_event(metrics_path, {
            "event": "start", "session_id": session_id, "start_epoch": start_epoch + 1,
            "total_epochs": args.epochs, "records": len(dataset), "batches_per_epoch": len(loader),
            "resume": args.resume,
            "sampling": summary,
        })

    progress = (
        tqdm(
            total=(args.epochs - start_epoch) * len(loader),
            desc="Pretrain",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        if args.progress and context.is_main and start_epoch < args.epochs else None
    )
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.monotonic()
            if sampler is not None:
                sampler.set_epoch(epoch)
            if progress is not None:
                progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", phase="train")

                def on_batch(loss: float) -> None:
                    progress.set_postfix(
                        epoch=f"{epoch + 1}/{args.epochs}", phase="train",
                        loss=f"{loss:.4f}", refresh=False,
                    )
                    progress.update()
            else:
                on_batch = None
            metrics = train_pretraining_epoch(
                model,
                loader,
                optimizer,
                scheduler,
                scaler,
                context,
                loss_config,
                mask_ratio=args.mask_ratio,
                accumulation_steps=args.grad_accumulation,
                grad_clip=args.grad_clip,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                progress_callback=on_batch,
                track_sampling=True,
            )
            if context.is_main:
                if progress is not None:
                    progress.set_postfix(
                        epoch=f"{epoch + 1}/{args.epochs}", phase="save",
                        loss=f"{metrics['loss']:.4f}",
                    )
                save_checkpoint(
                    output_dir / "last.pt",
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step + steps_per_epoch,
                    checkpoint_config,
                    metrics,
                )
                save_checkpoint(
                    output_dir / f"epoch_{epoch + 1:04d}.pt",
                    base_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step + steps_per_epoch,
                    checkpoint_config,
                    metrics,
                )
                append_event(metrics_path, {
                    "event": "epoch", "session_id": session_id, "epoch": epoch + 1,
                    "global_step": global_step + steps_per_epoch,
                    "elapsed_seconds": round(time.monotonic() - epoch_started, 3),
                    "lr": optimizer.param_groups[0]["lr"], **metrics,
                })
                message = json.dumps({"epoch": epoch + 1, **metrics}, ensure_ascii=False)
                if progress is not None:
                    tqdm.write(message)
                else:
                    print(message)
            global_step += steps_per_epoch
    finally:
        if progress is not None:
            progress.close()
        dataset.close()
    cleanup_distributed(context)


if __name__ == "__main__":
    main()

