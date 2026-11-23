"""FD/RUL few-shot or full-data adaptation entry point.

Examples:
  python -m model.make_few_shot --task fd --datasets CWRU --shots 1 --output runs/cwru_1shot.json
  python -m model.train_finetune --task fd --datasets CWRU \
      --few-shot-manifest runs/cwru_1shot.json --checkpoint runs/phm_pretrain/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm
from typing import Iterable, List, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on the data-only workstation
    raise SystemExit(
        "PyTorch is required for training. Install model/requirements.txt on the GPU server."
    ) from exc
from torch.nn.parallel import DistributedDataParallel

from .architecture import FoundationConfig, PHMFoundationModel, PHMTaskModel, load_foundation_checkpoint, normalize_backbone_ablation
from .data import FewShotSelection, H5Index, PHMH5Dataset, SampleRef, records_from_selection, select_few_shot
from .run_logging import append_event
from .fixed_split import prepare_fixed, save_split
from .inner_validation import prepare_inner, save_immutable, write_validation_report
from .paired_sampling import FDPairBatchSampler
from .power_spectrum import validate_power_checkpoint, checkpoint_gate_options
from .confidence_gate import MODE, gate_head
from .training import (
    build_grad_scaler,
    build_loader,
    cleanup_distributed,
    evaluate_task,
    initialize_fd_prototypes,
    initialize_fd_power_spectrum,
    load_training_checkpoint,
    make_finetune_optimizer,
    make_scheduler,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    train_task_epoch,
    preserve_rng_state,
    measure_support_gate,
)


def _group_split(index: H5Index, validation_fraction: float, seed: int) -> Tuple[H5Index, H5Index]:
    groups = sorted({ref.group_id for ref in index.records})
    rng = random.Random(seed)
    rng.shuffle(groups)
    validation_count = min(max(1, int(round(len(groups) * validation_fraction))), max(len(groups) - 1, 0))
    validation_groups = set(groups[:validation_count])
    train_records = [ref for ref in index.records if ref.group_id not in validation_groups]
    validation_records = [ref for ref in index.records if ref.group_id in validation_groups]
    return H5Index(train_records, index.files), H5Index(validation_records, index.files)


def _remap_fd_labels(train: H5Index, validation: H5Index) -> Tuple[H5Index, H5Index, int]:
    refs = list(train.records) + list(validation.records)
    keys = sorted({(ref.dataset, int(ref.label)) for ref in refs if ref.label is not None})
    mapping = {key: index for index, key in enumerate(keys)}

    def remap(index: H5Index) -> H5Index:
        return H5Index(
            [
                replace(ref, label=mapping[(ref.dataset, int(ref.label))])
                if ref.label is not None
                else ref
                for ref in index.records
            ],
            index.files,
        )

    return remap(train), remap(validation), len(mapping)


def _optional_lr(value: str) -> float | None:
    return None if value.lower() == "none" else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PHM FD/RUL downstream adaptation")
    parser.add_argument("--task", choices=("fd", "rul"), required=True)
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--root", action="append", help="Explicit H5 file/directory root; repeatable")
    parser.add_argument("--datasets", nargs="+", help="Dataset names; use one target dataset for a normal benchmark")
    parser.add_argument("--target-key", default="rul_norm")
    parser.add_argument("--sample-period", type=float, default=None)
    parser.add_argument("--few-shot-manifest")
    parser.add_argument("--split-manifest", help="Fixed FD train-pool/test manifest: train for fixed epochs, test once at the end")
    parser.add_argument("--inner-fold", help="Internal validation fold restricted to --split-manifest's training pool")
    parser.add_argument("--validation-every", type=int, default=1,
                        help="Internal validation interval; final epoch is always evaluated")
    parser.add_argument("--shots", type=int, help="Select K support windows when no manifest is supplied")
    parser.add_argument("--shot-unit", choices=("window", "group", "trajectory", "file"), default="window")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-eval", type=int, default=20000)
    parser.add_argument("--skip-bad-files", action="store_true")
    parser.add_argument("--checkpoint", help="Pretraining foundation checkpoint")
    parser.add_argument("--random-backbone", action="store_true",
                        help="Use --checkpoint architecture/config only; randomly initialize foundation weights for a matched control")
    parser.add_argument("--resume", help="Downstream checkpoint to resume")
    parser.add_argument("--output", default="runs/phm_finetune")
    parser.add_argument("--adaptation", choices=("lora", "linear", "full"), default="lora")
    parser.add_argument("--backbone-ablation", type=normalize_backbone_ablation, default=None,
                        help="Optional contribution knockout after loading the same pretrained backbone; saved/restored in model config")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--fd-head", choices=("linear", "cosine"), default="linear")
    parser.add_argument("--fd-pooling", choices=("mean", "mean-max"), default="mean")
    parser.add_argument("--fd-envelope", action=argparse.BooleanOptionalAction, default=False,
                        help="Add a zero-initialized whole-window Hilbert envelope spectrum residual for FD")
    parser.add_argument("--envelope-bins", type=int, default=128)
    parser.add_argument("--fd-power-spectrum", action=argparse.BooleanOptionalAction, default=False,
                        help="Add support-ridge-initialized per-channel power-spectrum logits to the backbone logits")
    parser.add_argument("--power-bins", type=int, default=256)
    parser.add_argument("--power-max-hz", type=float, default=6000.)
    parser.add_argument("--power-ridge-alpha", type=float, default=1.)
    parser.add_argument("--power-logit-scale", type=float, default=10.)
    parser.add_argument("--confidence-gate", action=argparse.BooleanOptionalAction, default=None,
                        help="Enabled by default with --fd-power-spectrum; --no-confidence-gate keeps constant spectral weight")
    parser.add_argument("--gate-threshold", type=float, default=.02,
                        help="Detached per-sample softmax standard-deviation threshold (default: 0.02)")
    parser.add_argument("--gate-alpha-low", type=float, default=.01,
                        help="Spectral coefficient below threshold; high coefficient is --power-logit-scale (10)")
    parser.add_argument("--envelope-max-hz", type=float, default=1000.0)
    parser.add_argument("--envelope-mode", choices=("broadband", "multiband"), default="broadband",
                        help="Multiband: broadband plus three fixed Nyquist-relative carrier bands with learned fusion; requires --fd-envelope")
    parser.add_argument("--fd-head-dropout", type=float, default=0.1,
                        help="Linear FD head dropout only; independent of pretrained backbone dropout")
    parser.add_argument("--channel-fusion", choices=("mean", "attention"), default=None,
                        help="Inherit checkpoint/default mean, or add shared trainable channel scoring")
    parser.add_argument("--zero-channel-policy", choices=("keep", "drop-zero"), default=None,
                        help="Inherit checkpoint/default keep; opt in to excluding exactly zero normalized channels")
    parser.add_argument("--cosine-scale", type=float, default=10.0)
    parser.add_argument("--prototype-init", action=argparse.BooleanOptionalAction, default=True,
                        help="Initialize cosine head using training support features (ignored for linear)")
    parser.add_argument("--best-metric", choices=("accuracy", "f1", "rmse"), default=None,
                        help="Default: accuracy for new FD runs, rmse for RUL; resume preserves saved criterion")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save epoch_NNNN.pt every N epochs; 0 keeps only normal last/best/final checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fd-sampling", choices=("random", "paired"), default="random",
                        help="paired: same-class pairs, preferring different support groups; FD only")
    parser.add_argument("--fd-contrastive-weight", type=float, default=0.0,
                        help="Cross-group supervised contrastive weight on final FD features; default disabled")
    parser.add_argument("--fd-contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=_optional_lr, default=8e-5,
                        help="Backbone LR (default 8e-5); use 'none' for the legacy shared-LR optimizer")
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
        raise ValueError("--random-backbone needs --checkpoint as an architecture reference, or --resume")
    if args.save_every < 0:
        raise ValueError("--save-every must be nonnegative")
    if not math.isfinite(args.fd_contrastive_weight) or args.fd_contrastive_weight < 0:
        raise ValueError("--fd-contrastive-weight must be finite and nonnegative")
    if not math.isfinite(args.fd_contrastive_temperature) or args.fd_contrastive_temperature <= 0:
        raise ValueError("--fd-contrastive-temperature must be finite and positive")
    if args.task != "fd" and (args.fd_sampling != "random" or args.fd_contrastive_weight != 0):
        raise ValueError("FD paired sampling and contrastive loss are FD-only")
    if args.resume:
        resume_args = torch.load(args.resume, map_location="cpu", weights_only=True).get("config", {}).get("args", {})
        validate_power_checkpoint(resume_args)
        for key, value in checkpoint_gate_options(resume_args).items():
            setattr(args, key, value)
        args.power_fusion = resume_args.get("power_fusion", "logits")
        for key, default in (("fd_sampling", "random"), ("fd_contrastive_weight", 0.0),
                             ("fd_contrastive_temperature", 0.1)):
            if getattr(args, key) != resume_args.get(key, default):
                raise ValueError(f"Resume must preserve --{key.replace('_', '-')}; repeat the original command")
        if args.fd_sampling == "paired" or args.fd_contrastive_weight:
            for key in ("batch_size", "grad_accumulation", "seed", "epochs"):
                if getattr(args, key) != resume_args.get(key):
                    raise ValueError(f"Paired/contrastive resume must preserve {key}")
    else:
        if args.confidence_gate is None and args.task == "fd" and args.fd_power_spectrum:
            args.confidence_gate = True
        if args.confidence_gate and (args.task != "fd" or not args.fd_power_spectrum):
            raise ValueError("--confidence-gate requires --task fd --fd-power-spectrum")
        args.power_fusion = MODE if args.fd_power_spectrum else "logits"
        if not math.isfinite(args.gate_threshold) or args.gate_threshold < 0:
            raise ValueError("--gate-threshold must be finite and nonnegative")
        if not math.isfinite(args.gate_alpha_low) or not 0 <= args.gate_alpha_low <= args.power_logit_scale:
            raise ValueError("--gate-alpha-low must be between 0 and --power-logit-scale")
    bound = bool(args.split_manifest)
    internal = bool(args.inner_fold)
    fixed = bound and not internal
    if args.validation_every < 1 or (args.validation_every != 1 and not internal):
        raise ValueError("--validation-every must be positive; intervals other than 1 require --inner-fold")
    if internal and (not bound or not args.few_shot_manifest or args.shots is not None):
        raise ValueError("--inner-fold requires outer --split-manifest and --few-shot-manifest, without --shots")
    if internal:
        args.best_metric = args.best_metric or "f1"
        if args.best_metric != "f1":
            raise ValueError("Internal group validation selects by macro F1; use --best-metric f1")
    if bound and (args.task != "fd" or args.shot_unit != "window"):
        raise ValueError("--split-manifest currently supports FD window shots only")
    if fixed and args.best_metric is not None:
        raise ValueError("Fixed K-shot protocol has no validation selection; omit --best-metric")
    if bound and args.epochs < 1:
        raise ValueError("Fixed K-shot requires at least one training epoch")
    if bound and not args.resume and Path(args.output).exists() and any(Path(args.output).iterdir()):
        raise ValueError("Use a new empty output directory for a fixed-split experiment")
    if fixed and args.resume and (Path(args.output) / "test_metrics.json").exists():
        raise ValueError("This run has already tested; do not extend training after observing the test result")
    context = setup_distributed(args.device)
    seed_everything(args.seed, context.rank)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    index = H5Index.build(
        data_root=args.data_root,
        task=args.task,
        roots=args.root,
        datasets=args.datasets,
        target_key=args.target_key,
        sample_period=args.sample_period,
        strict=not args.skip_bad_files,
        show_progress=args.progress and context.is_main,
    )
    if not index.records:
        raise RuntimeError("No downstream records were found")

    selection: FewShotSelection | None = None
    protocol = None
    fixed_payload = None
    if internal:
        train_index, validation_index, selection, inner_payload, protocol = prepare_inner(
            index, args.split_manifest, args.inner_fold, args.few_shot_manifest
        )
        fixed_payload = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    elif fixed:
        train_index, validation_index, selection, fixed_payload, protocol = prepare_fixed(
            index, args.split_manifest, args.few_shot_manifest, args.shots, args.seed
        )
    elif args.few_shot_manifest:
        selection = FewShotSelection.load(args.few_shot_manifest)
        if selection.split_id:
            raise ValueError("This support manifest is bound to a fixed split; supply --split-manifest")
        if selection.task != args.task:
            raise ValueError(f"Manifest task {selection.task} does not match --task {args.task}")
        manifest_datasets = sorted({ref.dataset for ref in selection.selected})
        # A manifest defines the target domain.  Always restrict evaluation to
        # those datasets, even when a broader --datasets list was supplied.
        index = index.filter(datasets=manifest_datasets)
        train_index = records_from_selection(index, selection)
        held_out_groups = set(selection.excluded_group_ids)
        validation_index = H5Index(
            [ref for ref in index.records if ref.group_id not in held_out_groups], index.files
        )
    elif args.shots:
        selection = select_few_shot(index, args.shots, args.task, seed=args.seed, unit=args.shot_unit)
        train_index = records_from_selection(index, selection)
        held_out_groups = set(selection.excluded_group_ids)
        validation_index = H5Index(
            [ref for ref in index.records if ref.group_id not in held_out_groups], index.files
        )
    else:
        train_index, validation_index = _group_split(index, args.validation_fraction, args.seed)

    if not train_index.records:
        raise RuntimeError("The selected few-shot training split is empty")
    if not validation_index.records:
        raise RuntimeError("The validation split is empty; reserve at least one independent group")
    if args.task == "fd" and any(ref.label is None for ref in train_index.records + validation_index.records):
        raise ValueError("FD fine-tuning requires a label for every training and validation record")
    if args.task == "rul" and any(ref.target is None for ref in train_index.records + validation_index.records):
        raise ValueError("RUL fine-tuning requires a finite target for every training and validation record")
    if args.task == "fd":
        class_keys = sorted({(ref.dataset, int(ref.label)) for ref in train_index.records + validation_index.records})
        train_index, validation_index, num_classes = _remap_fd_labels(train_index, validation_index)
    else:
        num_classes = None
    if not bound and args.max_eval > 0 and len(validation_index) > args.max_eval:
        rng = random.Random(args.seed)
        capped = rng.sample(validation_index.records, args.max_eval)
        validation_index = H5Index(capped, validation_index.files)
    train_dataset = PHMH5Dataset(train_index, mode="finetune", robust_normalize=True)
    validation_dataset = PHMH5Dataset(validation_index, mode="eval", robust_normalize=True)
    pair_sampler = None
    if args.fd_sampling == "paired":
        pair_sampler = FDPairBatchSampler(train_index.records, args.batch_size, args.seed,
                                         context.world_size, context.rank)
        if args.resume:
            saved_sampling = torch.load(args.resume, map_location="cpu", weights_only=True).get("config", {}).get("fd_sampling")
            if saved_sampling != pair_sampler.summary:
                raise ValueError("Paired resume requires the same support sampling plan and world size")
        if context.is_main:
            print(json.dumps({"fd_sampling": pair_sampler.summary}, ensure_ascii=False), flush=True)
    if args.fd_contrastive_weight > 0:
        by_class = {}
        for ref in train_index.records:
            by_class.setdefault(ref.label, set()).add(ref.group_id)
        if not any(len(groups) > 1 for groups in by_class.values()):
            raise ValueError("Cross-group contrastive loss requires same-class support from multiple groups; not for 1-shot")
    train_loader, train_sampler = build_loader(
        train_dataset, args.batch_size, shuffle=True, workers=args.workers, context=context, seed=args.seed,
        batch_sampler_override=pair_sampler,
    )
    validation_loader, _ = build_loader(
        validation_dataset,
        args.eval_batch_size,
        shuffle=False,
        workers=args.workers,
        context=context,
        seed=args.seed,
        distributed_eval=True,
    )

    model_config = FoundationConfig(
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        layers=args.layers,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        feature_version=args.architecture_version,
        physical_patch_seconds=args.physical_patch_seconds,
        local_attention_window=args.local_attention_window,
    )
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved_config = saved.get("config", {}) if isinstance(saved, dict) else {}
        if saved_config.get("protocol") != protocol:
            raise ValueError("Resume requires the same fixed split, support and evaluation protocol")
        if internal:
            for key in ("epochs", "validation_every", "batch_size", "grad_accumulation", "seed", "amp", "amp_dtype"):
                if saved_config.get("args", {}).get(key) != getattr(args, key):
                    raise ValueError(f"Internal validation resume must preserve {key}")
        if fixed and (saved_config.get("args", {}).get("epochs") != args.epochs
                      or any(k.startswith("test_") for k in saved.get("metrics", {}))):
            raise ValueError("Fixed protocol resume must preserve the epoch budget and precede final testing")
        saved_model = saved_config.get("model") if isinstance(saved_config, dict) else None
        if not isinstance(saved_model, dict):
            raise ValueError("Fine-tuning checkpoint has no model configuration for --resume")
        saved_args = saved_config.get("args", {})
        if isinstance(saved_args, dict):
            if saved_args.get("task", args.task) != args.task:
                raise ValueError("--task does not match the fine-tuning checkpoint")
            for key in ("adaptation", "lora_rank", "lora_alpha"):
                if key in saved_args:
                    setattr(args, key, saved_args[key])
            args.random_backbone = saved_args.get("random_backbone", False)
            args.checkpoint = saved_args.get("checkpoint", args.checkpoint)
            # Restore head semantics, including defaults of legacy runs.
            for key, default in {
                "fd_head": "linear", "cosine_scale": 10.0, "prototype_init": True,
                "fd_pooling": "mean", "fd_head_dropout": 0.1, "backbone_lr": None,
                "fd_envelope": False, "envelope_bins": 128, "envelope_max_hz": 1000.0,
                "envelope_mode": "broadband",
                "fd_power_spectrum": False, "power_bins": 256, "power_max_hz": 6000.,
                "power_ridge_alpha": 1., "power_logit_scale": 10., "power_channel_names": None,
            }.items():
                setattr(args, key, saved_args.get(key, default))
            # Foundation architecture is saved separately. Resume cannot silently
            # change the optimizer grouping or the fusion parameter shapes.
            args.channel_fusion = saved_model.get("channel_fusion", "mean")
            args.zero_channel_policy = saved_model.get("zero_channel_policy", "keep")
            args.backbone_ablation = saved_model.get("backbone_ablation", "none")
            args.lr = saved_args.get("lr", args.lr)
            args.weight_decay = saved_args.get("weight_decay", args.weight_decay)
            if context.is_main and (
                saved_args.get("fd_augmentation", "none") != "none"
                or saved_args.get("fd_orthogonality_weight", 0.0) != 0.0
            ):
                print("Resuming weights/optimizer with plain classification loss; experimental FD augmentation and orthogonality regularization have been removed.", flush=True)
            saved_metric = saved_args.get("best_metric") or ("f1" if args.task == "fd" else "rmse")
            if args.best_metric is not None and args.best_metric != saved_metric:
                raise ValueError("--resume must preserve --best-metric; start a new run from the pretraining checkpoint to change it")
            args.best_metric = saved_metric
        saved_num_classes = saved_config.get("num_classes")
        if saved_num_classes != num_classes:
            raise ValueError("Current data split has a different number of classes than --resume")
        allowed = {key: value for key, value in saved_model.items() if key in FoundationConfig.__annotations__}
        allowed.setdefault("feature_version", 1)
        foundation = PHMFoundationModel(FoundationConfig(**allowed))
    elif args.checkpoint:
        foundation = load_foundation_checkpoint(args.checkpoint, device="cpu", strict=True,
                                               load_weights=not args.random_backbone)
    else:
        foundation = PHMFoundationModel(model_config)
    # A checkpoint is the source of truth for architecture dimensions.  Keep
    # the saved run configuration aligned when CLI defaults differ.
    args.channel_fusion = args.channel_fusion or foundation.config.channel_fusion
    args.zero_channel_policy = args.zero_channel_policy or foundation.config.zero_channel_policy
    args.backbone_ablation = args.backbone_ablation or foundation.config.backbone_ablation
    foundation.configure_ablation(args.backbone_ablation)
    foundation.tokenizer.configure_channel_fusion(args.channel_fusion, args.zero_channel_policy)
    foundation.config.channel_fusion = args.channel_fusion
    foundation.config.zero_channel_policy = args.zero_channel_policy
    model_config = foundation.config
    args.best_metric = args.best_metric or ("accuracy" if args.task == "fd" else "rmse")
    if args.best_metric not in ({"accuracy", "f1"} if args.task == "fd" else {"rmse"}):
        raise ValueError("FD best metric must be accuracy/f1; RUL best metric must be rmse")
    if args.task != "fd" and (args.fd_head != "linear" or args.fd_pooling != "mean" or args.fd_head_dropout != 0.1 or args.fd_envelope or args.fd_power_spectrum):
        raise ValueError("--fd-head, --fd-pooling and --fd-head-dropout are FD-only options")
    if args.fd_power_spectrum:
        if (args.power_bins < 2 or any(not math.isfinite(v) or v <= 0
                for v in (args.power_max_hz, args.power_ridge_alpha, args.power_logit_scale))):
            raise ValueError("Power bins must be >=2; frequency, ridge alpha and logit scale must be finite/positive")
        args.power_channel_names = train_dataset.configure_power_spectrum(
            args.power_bins, args.power_max_hz, getattr(args, "power_channel_names", None))
        validation_dataset.configure_power_spectrum(args.power_bins, args.power_max_hz, args.power_channel_names)
    task_model = PHMTaskModel(
        foundation,
        task=args.task,
        num_classes=num_classes,
        adaptation=args.adaptation,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        fd_head=args.fd_head,
        cosine_scale=args.cosine_scale,
        fd_pooling=args.fd_pooling,
        fd_head_dropout=args.fd_head_dropout,
        fd_envelope=args.fd_envelope,
        envelope_bins=args.envelope_bins,
        envelope_max_hz=args.envelope_max_hz,
        envelope_mode=args.envelope_mode,
        fd_power_spectrum=args.fd_power_spectrum,
        power_bins=args.power_bins,
        power_channels=len(args.power_channel_names) if args.fd_power_spectrum else 2,
        power_logit_scale=args.power_logit_scale,
        confidence_gate=args.confidence_gate if args.fd_power_spectrum else None,
        gate_threshold=args.gate_threshold,
        gate_alpha_low=args.gate_alpha_low,
    ).to(context.device)
    prototype_counts = None
    power_support_counts = None
    if args.fd_power_spectrum and not args.resume:
        with preserve_rng_state(context.device):
            support_loader, _ = build_loader(train_dataset, args.eval_batch_size, shuffle=False, workers=0,
                                            context=replace(context, rank=0, world_size=1), seed=args.seed)
            power_support_counts = initialize_fd_power_spectrum(task_model, support_loader, args.power_ridge_alpha)
        if context.is_main:
            print(f"Initialized power-spectrum ridge branch from training records per class: {power_support_counts}", flush=True)
    if args.task == "fd" and args.fd_head == "cosine" and args.prototype_init and not args.resume:
        # Complete training support on each rank: no padded distributed duplicates,
        # no augmentation, and no validation records involved in initialization.
        prototype_loader, _ = build_loader(
            train_dataset, args.eval_batch_size, shuffle=False, workers=0,
            context=replace(context, rank=0, world_size=1), seed=args.seed,
        )
        prototype_counts = initialize_fd_prototypes(task_model, prototype_loader, context.device)
        if context.is_main:
            print(f"Initialized cosine head from training records per class: {prototype_counts}", flush=True)
    optimizer = make_finetune_optimizer(task_model, args.lr, args.weight_decay, args.backbone_lr)
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accumulation, 1))
    scheduler = make_scheduler(optimizer, steps_per_epoch * args.epochs, args.warmup_steps)
    scaler = build_grad_scaler(args.amp and args.amp_dtype == "fp16" and context.device.type == "cuda")
    start_epoch = 0
    global_step = 0
    best_value = float("-inf") if args.task == "fd" else float("inf")
    if args.resume:
        start_epoch, global_step, previous_metrics = load_training_checkpoint(
            args.resume, task_model, optimizer, scheduler, scaler, device=context.device
        )
        start_epoch += 1
        metric_key = args.best_metric
        previous_value = previous_metrics.get(
            f"best_val_{metric_key}", previous_metrics.get(f"val_{metric_key}")
        )
        if previous_value is not None and math.isfinite(float(previous_value)):
            best_value = float(previous_value)
    wrapped_model = (
        DistributedDataParallel(
            task_model, device_ids=[context.local_rank] if context.device.type == "cuda" else None
        )
        if context.is_distributed
        else task_model
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {"model": asdict(model_config), "args": vars(args), "num_classes": num_classes}
    if gate_head(task_model) is not None:
        run_config.update(implementation=MODE,
                          gate_definition="FP32 softmax base logits; sample std correction=1; high if >=tau; detached per sample")
    if pair_sampler is not None:
        run_config["fd_sampling"] = pair_sampler.summary
    if bound:
        run_config["protocol"] = protocol
    session_id = uuid.uuid4().hex[:12]
    metrics_path = output_dir / "metrics.jsonl"
    if context.is_main:
        if bound:
            save_split(fixed_payload, output_dir / "fixed_split.json")
            selection.save(output_dir / "support.json")
            if internal:
                save_immutable(inner_payload, output_dir / "inner_fold.json")
        (output_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
        split_summary = {
            "task": args.task,
            "train_records": len(train_index),
            "validation_records": len(validation_index),
            "train_groups": len({ref.group_id for ref in train_index.records}),
            "validation_groups": len({ref.group_id for ref in validation_index.records}),
        }
        if args.task == "fd":
            train_counts = Counter(int(ref.label) for ref in train_index.records)
            validation_counts = Counter(int(ref.label) for ref in validation_index.records)
            split_summary.update({
                "num_classes": num_classes,
                "uniform_baseline_accuracy": 1.0 / num_classes,
                "majority_baseline_accuracy": max(validation_counts.values()) / len(validation_index),
                "classes": [
                    {
                        "class_id": class_id, "dataset": dataset, "original_label": label,
                        "train_records": train_counts[class_id],
                        "validation_records": validation_counts[class_id],
                    }
                    for class_id, (dataset, label) in enumerate(class_keys)
                ],
            })
        if fixed:
            split_summary.update(protocol)
            split_summary["test_groups"] = split_summary.pop("validation_groups")
            split_summary["test_records"] = split_summary.pop("validation_records")
            split_summary["validation_records"] = 0
            split_summary["validation_groups"] = 0
            for item in split_summary["classes"]:
                item["test_records"] = item.pop("validation_records")
                item["validation_records"] = 0
        elif internal:
            split_summary.update(protocol)
            split_summary.update(test_records=0, test_groups=0)
        (output_dir / "split_summary.json").write_text(
            json.dumps(split_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        append_event(metrics_path, {
            "event": "start", "session_id": session_id, "start_epoch": start_epoch + 1,
            "total_epochs": args.epochs, "batches_per_epoch": len(train_loader),
            "validation_batches_per_epoch": 0 if fixed else len(validation_loader),
            "test_batches": len(validation_loader) if fixed else 0,
            "best_metric": None if fixed else args.best_metric, "fd_head": args.fd_head,
            "validation_every": args.validation_every if internal else (0 if fixed else 1),
            "prototype_counts": prototype_counts,
            "foundation_initialization": "random" if args.random_backbone or not args.checkpoint else "pretrained",
            "fd_power_spectrum": args.fd_power_spectrum, "power_support_counts": power_support_counts,
            "power_bins": args.power_bins, "power_max_hz": args.power_max_hz,
            "power_logit_scale": args.power_logit_scale,
            "power_fusion": args.power_fusion, "confidence_gate": args.confidence_gate,
            "gate_threshold": args.gate_threshold, "gate_alpha_low": args.gate_alpha_low,
            "fd_sampling": pair_sampler.summary if pair_sampler is not None else {"name": "random"},
            "fd_contrastive_weight": args.fd_contrastive_weight,
            "fd_contrastive_temperature": args.fd_contrastive_temperature,
            "fd_pooling": args.fd_pooling, "fd_head_dropout": args.fd_head_dropout,
            "fd_envelope": args.fd_envelope, "envelope_bins": args.envelope_bins,
            "envelope_max_hz": args.envelope_max_hz,
            "envelope_mode": args.envelope_mode,
            "channel_fusion": args.channel_fusion, "zero_channel_policy": args.zero_channel_policy,
            "backbone_ablation": args.backbone_ablation,
            "optimizer_groups": [
                {"name": group.get("name", "all"), "initial_lr": group["initial_lr"],
                 "trainable_parameters": sum(p.numel() for p in group["params"])}
                for group in optimizer.param_groups
            ],
            "resume": args.resume, **{key: value for key, value in split_summary.items() if key != "classes"},
        })
        if args.task == "fd" and any(item["train_records"] == 0 for item in split_summary["classes"]):
            print("Warning: some validation classes have no training examples; see split_summary.json", flush=True)
        if gate_head(task_model) is not None:
            with preserve_rng_state(context.device):
                gate_loader, _ = build_loader(train_dataset, args.eval_batch_size, shuffle=False, workers=0,
                                             context=replace(context, rank=0, world_size=1), seed=args.seed)
                gate_initial = measure_support_gate(task_model, gate_loader, context.device)
            gate_event = {"event": "confidence_init", "session_id": session_id,
                          "foundation_initialization": "random" if args.random_backbone or not args.checkpoint else "pretrained",
                          "partition": "training_support", "resumed": bool(args.resume), **gate_initial}
            append_event(metrics_path, gate_event)
            print(json.dumps(gate_event, ensure_ascii=False), flush=True)
    progress = (
        tqdm(
            total=((args.epochs - start_epoch) * len(train_loader) +
                   sum(not fixed and (not internal or (e + 1) % args.validation_every == 0 or e + 1 == args.epochs)
                       for e in range(start_epoch, args.epochs)) * len(validation_loader)),
            desc=f"Finetune {args.task.upper()}",
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
        )
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
                    progress.set_postfix(
                        epoch=f"{epoch + 1}/{args.epochs}", phase="train",
                        loss=f"{loss:.4f}", refresh=False,
                    )
                    progress.update()
            else:
                on_batch = None
            train_metrics = train_task_epoch(
                wrapped_model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                context,
                task=args.task,
                accumulation_steps=args.grad_accumulation,
                grad_clip=args.grad_clip,
                amp=args.amp,
                amp_dtype=args.amp_dtype,
                rul_loss_kind=args.rul_loss,
                weibull_loss_weight=args.weibull_loss_weight,
                progress_callback=on_batch,
                fd_contrastive_weight=args.fd_contrastive_weight,
                fd_contrastive_temperature=args.fd_contrastive_temperature,
                fd_pair_logging=args.fd_sampling == "paired",
            )
            if task_model.fd_pool is not None:
                gates = task_model.fd_pool.gate.detach().tanh().float()
                train_metrics.update({"pool_gate_mean": gates.mean().item(), "pool_gate_abs_max": gates.abs().max().item()})
            if task_model.foundation.tokenizer.channel_fusion is not None:
                train_metrics["channel_score_norm"] = task_model.foundation.tokenizer.channel_fusion.score.detach().float().norm().item()
            if task_model.fd_envelope is not None:
                train_metrics["envelope_projection_norm"] = task_model.fd_envelope.weight.detach().float().norm().item()
                if task_model.fd_envelope.mode == "multiband":
                    for name, value in zip(("broadband", "low", "mid", "high"), task_model.fd_envelope.band_weights().detach().cpu().tolist()):
                        train_metrics[f"envelope_band_weight_{name}"] = value
            if task_model.fd_power_spectrum is not None:
                train_metrics["power_classifier_norm"] = task_model.head.weight.detach().float().norm().item()
                train_metrics["power_active_dimensions"] = int(task_model.fd_power_spectrum.active.sum().item())
            validate_now = not fixed and (not internal or (epoch + 1) % args.validation_every == 0
                                         or epoch + 1 == args.epochs)
            if progress is not None and validate_now:
                progress.set_postfix(
                    epoch=f"{epoch + 1}/{args.epochs}", phase="val",
                    loss=f"{train_metrics['loss']:.4f}",
                )
            # Evaluation uses the unwrapped module so ranks with an empty shard do
            # not enter DDP's forward-time buffer broadcasts.
            validation_metrics = {}
            if internal and validate_now:
                with preserve_rng_state(context.device):
                    validation_metrics = evaluate_task(
                        task_model, validation_loader, args.task, context,
                        progress_callback=progress.update if progress is not None else None,
                        fd_details=True,
                    )
                if context.is_main:
                    write_validation_report(output_dir, epoch + 1, validation_metrics, class_keys, protocol)
                else:
                    validation_metrics.pop("per_class", None)
                    validation_metrics.pop("confusion_matrix", None)
            elif validate_now:
                validation_metrics = evaluate_task(
                    task_model, validation_loader, args.task, context,
                    progress_callback=progress.update if progress is not None else None,
                )
            current = validation_metrics.get(args.best_metric, float("nan"))
            improved = bool(torch.isfinite(torch.tensor(current))) and (
                current > best_value if args.task == "fd" else current < best_value
            )
            if improved:
                best_value = current
            if context.is_main:
                metrics = {**train_metrics, **{f"val_{key}": value for key, value in validation_metrics.items()}}
                if not fixed and math.isfinite(best_value):
                    metrics[f"best_val_{args.best_metric}"] = best_value
                if progress is not None:
                    progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", phase="save")
                save_checkpoint(
                    output_dir / "last.pt",
                    task_model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step + steps_per_epoch,
                    run_config,
                    metrics,
                )
                if args.save_every and (epoch + 1) % args.save_every == 0:
                    # Preserve this exact training state without additional forward
                    # passes, RNG draws or test access. Snapshots sit beside the
                    # split/support files consumed by diagnose_finetune.
                    shutil.copyfile(output_dir / "last.pt", output_dir / f"epoch_{epoch + 1:04d}.pt")
                if improved and not fixed:
                    save_checkpoint(
                        output_dir / "best.pt",
                        task_model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step + steps_per_epoch,
                        run_config,
                        metrics,
                    )
                append_event(metrics_path, {
                    "event": "epoch", "session_id": session_id, "epoch": epoch + 1,
                    "global_step": global_step + steps_per_epoch,
                    "elapsed_seconds": round(time.monotonic() - epoch_started, 3),
                    "lr": optimizer.param_groups[0]["lr"], **metrics,
                    "learning_rates": {group.get("name", "all"): group["lr"] for group in optimizer.param_groups},
                })
                message = json.dumps({"epoch": epoch + 1, **metrics}, ensure_ascii=False)
                if progress is not None:
                    tqdm.write(message)
                else:
                    print(message)
            global_step += steps_per_epoch
        if internal and context.is_main:
            shutil.copyfile(output_dir / "last.pt", output_dir / "final.pt")
            final_report = json.loads((output_dir / "validation" / f"epoch_{args.epochs:04d}.json").read_text(encoding="utf-8"))
            (output_dir / "validation_metrics.json").write_text(
                json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
        if fixed:
            test_progress = tqdm(total=len(validation_loader), desc="FD fixed test", unit="batch",
                                 disable=not (args.progress and context.is_main))
            try:
                test_metrics = evaluate_task(task_model, validation_loader, "fd", context,
                                             progress_callback=test_progress.update)
            finally:
                test_progress.close()
            if context.is_main:
                result = {"epoch": args.epochs, "split_id": protocol["split_id"],
                          "support_id": protocol["support_id"],
                          **{f"test_{k}": v for k, v in test_metrics.items()}}
                save_checkpoint(output_dir / "final.pt", task_model, optimizer, scheduler, scaler,
                                args.epochs - 1, global_step, run_config, result)
                (output_dir / "test_metrics.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                append_event(metrics_path, {"event": "test", "session_id": session_id, **result})
                print(json.dumps(result, ensure_ascii=False), flush=True)
    finally:
        if progress is not None:
            progress.close()
        train_dataset.close()
        validation_dataset.close()
    cleanup_distributed(context)


if __name__ == "__main__":
    main()

