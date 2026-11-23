"""Reusable distributed training loops for PHM pretraining and fine-tuning."""

from __future__ import annotations

import json
import math
import os
import random
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from .architecture import CosineFDHead
from .data import PHMH5Dataset, collate_batch
from .losses import PretrainingLossConfig, fd_loss, pretraining_loss, rul_loss, cross_group_contrastive_loss
from .metrics import classification_metrics, rul_metrics
from .confidence_gate import gate_head


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(device_arg: str = "auto") -> DistributedContext:
    """Initialize torchrun's process group when WORLD_SIZE is greater than one."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank if world_size > 1 else 0)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return DistributedContext(rank, world_size, local_rank, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int = 0) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation rows without DistributedSampler's padding duplicates."""

    def __init__(self, dataset: PHMH5Dataset, num_replicas: int, rank: int) -> None:
        if num_replicas < 1:
            raise ValueError("num_replicas must be positive")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.indices = list(range(self.rank, len(dataset), self.num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def build_loader(
    dataset: PHMH5Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    context: DistributedContext,
    seed: int = 0,
    distributed_eval: bool = False,
    sampler_override: Optional[Sampler[int]] = None,
    batch_sampler_override: Optional[Sampler] = None,
) -> Tuple[DataLoader, Optional[Sampler[int]]]:
    if batch_sampler_override is not None:
        if sampler_override is not None:
            raise ValueError("Use only one sampler override")
        kwargs = {"batch_sampler": batch_sampler_override, "num_workers": int(workers),
                  "pin_memory": context.device.type == "cuda", "collate_fn": collate_batch}
        if workers > 0:
            kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(dataset, **kwargs), batch_sampler_override
    sampler: Optional[Sampler[int]] = sampler_override
    if sampler_override is not None:
        shuffle = False
    elif context.is_distributed:
        if distributed_eval:
            sampler = DistributedEvalSampler(dataset, context.world_size, context.rank)
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=shuffle,
                seed=seed,
            )
        shuffle = False
    kwargs: Dict[str, Any] = {
        "batch_size": int(batch_size),
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": int(workers),
        "pin_memory": context.device.type == "cuda",
        "drop_last": False,
        "collate_fn": collate_batch,
    }
    if workers > 0:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return DataLoader(dataset, **kwargs), sampler


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _rotation_kwargs(batch: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    return {
        name: batch.get(f"{prefix}{name}")
        for name in (
            "rpm_traces", "rpm_trace_valid", "shaft_angles", "shaft_angle_valid", "nominal_rpm",
        )
    }


def _autocast(device: torch.device, enabled: bool, dtype: str = "bf16"):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if dtype == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support bf16 AMP; use --amp-dtype fp16 or --no-amp")
    if dtype == "fp16":
        autocast_dtype = torch.float16
    else:
        autocast_dtype = torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True)


def _task_kwargs(batch: Mapping[str, Any]) -> Dict[str, Any]:
    values = _rotation_kwargs(batch)
    if "power_features" in batch:
        values["power_features"] = batch["power_features"]
    return values


def _reduce_scalar(value: float, context: DistributedContext) -> float:
    if not context.is_distributed:
        return float(value)
    tensor = torch.tensor(float(value), device=context.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / context.world_size).item())


def _trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def _step_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    grad_clip: float,
) -> None:
    use_scaler = scaler is not None and scaler.is_enabled()
    if use_scaler:
        scaler.unscale_(optimizer)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(_trainable_parameters(model), grad_clip)
    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _augment(signals: Tensor, lengths: Tensor, channel_mask: Tensor) -> Tensor:
    scale = torch.empty((signals.shape[0], 1, 1), device=signals.device).uniform_(0.9, 1.1)
    augmented = signals * scale + torch.randn_like(signals) * 0.01
    points = torch.arange(signals.shape[-1], device=signals.device).view(1, 1, -1)
    valid_points = points < lengths.view(-1, 1, 1)
    augmented = augmented * valid_points.to(augmented.dtype) * channel_mask.unsqueeze(-1).to(augmented.dtype)
    return augmented


def _serialize_config(config: Any) -> Any:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    return config


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    step: int,
    config: Any,
    metrics: Optional[Mapping[str, float]] = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": int(epoch),
        "step": int(step),
        "config": _serialize_config(config),
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, output)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str | torch.device = "cpu",
) -> Tuple[int, int, Dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state)
    if optimizer is not None and isinstance(payload, Mapping) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and isinstance(payload, Mapping) and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and isinstance(payload, Mapping) and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return (
        int(payload.get("epoch", 0)) if isinstance(payload, Mapping) else 0,
        int(payload.get("step", 0)) if isinstance(payload, Mapping) else 0,
        dict(payload.get("metrics", {})) if isinstance(payload, Mapping) else {},
    )


def make_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(_trainable_parameters(model), lr=float(lr), weight_decay=float(weight_decay), betas=(0.9, 0.95))


def make_finetune_optimizer(
    model: nn.Module, lr: float, weight_decay: float, backbone_lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Optional two-rate optimizer; legacy one-group state remains loadable.

    Head, FD pooling, spectral adapters and channel fusion use lr; pretrained foundation parameters
    (LoRA, or all unfrozen weights under full adaptation) use backbone_lr.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Head learning rate must be finite and positive")
    if backbone_lr is None:
        return make_optimizer(model, lr, weight_decay)
    if not math.isfinite(backbone_lr) or backbone_lr < 0:
        raise ValueError("Backbone learning rate must be finite and nonnegative")
    adapters, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("foundation.") and not name.startswith("foundation.tokenizer.channel_fusion."):
            backbone.append(parameter)
        else:
            adapters.append(parameter)
    groups = [{"params": adapters, "lr": lr, "name": "head_and_fusion"}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr, "name": "backbone"})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> Any:
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps - 1)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def train_pretraining_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    context: DistributedContext,
    loss_config: PretrainingLossConfig,
    mask_ratio: float,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    amp: bool = True,
    amp_dtype: str = "bf16",
    progress_callback: Optional[Callable[[float], None]] = None,
    track_sampling: bool = False,
) -> Dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = {"loss": 0.0, "specmae": 0.0, "monotonicity": 0.0, "contrastive": 0.0}
    steps = 0
    total_batches = len(loader)
    use_scaler = scaler is not None and scaler.is_enabled()
    sampling_counts = {}
    if track_sampling:
        sampling_counts = {f"{r.task}:{r.dataset}": 0 for r in loader.dataset.records}
    for batch_index, raw_batch in enumerate(loader):
        if track_sampling:
            for item in raw_batch["metadata"]:
                sampling_counts[f"{item['task']}:{item['dataset']}"] += 1
        batch = _move_batch(raw_batch, context.device)
        signals = batch["signals"]
        lengths = batch["lengths"]
        channel_mask = batch["channel_mask"]
        periods = batch.get("sample_periods")
        with _autocast(context.device, amp, amp_dtype):
            outputs = model(
                signals, lengths, channel_mask, periods, mask_ratio=mask_ratio,
                **_rotation_kwargs(batch),
            )
            positive = model(
                _augment(signals, lengths, channel_mask),
                lengths,
                channel_mask,
                periods,
                mask_ratio=0.0,
                **_rotation_kwargs(batch),
            )
            next_outputs = model(
                batch["next_signals"],
                batch["next_lengths"],
                batch["next_channel_mask"],
                periods,
                mask_ratio=0.0,
                **_rotation_kwargs(batch, "next_"),
            )
            losses = pretraining_loss(
                outputs,
                positive,
                loss_config,
                next_outputs=next_outputs,
                pair_valid=batch["pair_valid"],
            )
            group_start = (batch_index // max(1, accumulation_steps)) * max(1, accumulation_steps)
            group_size = min(max(1, accumulation_steps), total_batches - group_start)
            loss = losses["loss"] / max(1, group_size)
        if use_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (batch_index + 1) % max(1, accumulation_steps) == 0:
            _step_optimizer(model, optimizer, scheduler, scaler, grad_clip)
        for key in running:
            running[key] += float(losses[key].item())
        steps += 1
        if progress_callback is not None:
            progress_callback(running["loss"] / steps)
    if steps and steps % max(1, accumulation_steps):
        _step_optimizer(model, optimizer, scheduler, scaler, grad_clip)
    if steps == 0:
        return running
    metrics = {key: _reduce_scalar(value / steps, context) for key, value in running.items()}
    if track_sampling:
        keys = sorted(sampling_counts)
        totals = torch.tensor([sampling_counts[key] for key in keys], device=context.device, dtype=torch.long)
        if context.is_distributed:
            dist.all_reduce(totals)
        metrics["sampling_counts"] = dict(zip(keys, totals.cpu().tolist()))
        metrics["sampled_records"] = int(totals.sum().item())
    return metrics


@torch.no_grad()
def evaluate_task(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    context: DistributedContext,
    progress_callback: Optional[Callable[[], None]] = None,
    fd_details: bool = False,
) -> Dict[str, Any]:
    model.eval()
    gate = gate_head(model)
    if gate is not None:
        gate.reset_gate_stats()
    task = task.lower()
    all_labels = []
    all_scores = []
    all_targets = []
    all_predictions = []
    for raw_batch in loader:
        batch = _move_batch(raw_batch, context.device)
        output = model(
            batch["signals"], batch["lengths"], batch["channel_mask"], batch.get("sample_periods"),
            **_task_kwargs(batch),
        )
        if task == "fd":
            all_labels.append(batch["labels"].detach().cpu())
            all_scores.append(output["logits"].detach().cpu())
        else:
            all_targets.append(batch["targets"].detach().cpu())
            all_predictions.append(output["prediction"].detach().cpu())
        if progress_callback is not None:
            progress_callback()
    gate_metrics = gate.gate_stats(context) if gate is not None else {}
    if task == "fd":
        labels = torch.cat(all_labels).numpy() if all_labels else np.empty(0, dtype=np.int64)
        scores = torch.cat(all_scores).numpy() if all_scores else np.empty((0, 0), dtype=np.float32)
        if context.is_distributed:
            gathered = [None for _ in range(context.world_size)]
            dist.all_gather_object(gathered, (labels, scores))
            label_parts = [item[0] for item in gathered if item[0].size]
            score_parts = [item[1] for item in gathered if item[1].shape[0]]
            labels = np.concatenate(label_parts, axis=0) if label_parts else np.empty(0, dtype=np.int64)
            scores = np.concatenate(score_parts, axis=0) if score_parts else np.empty((0, 0), dtype=np.float32)
        if not labels.size:
            return {"accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan"), "recall": float("nan"), **gate_metrics}
        metrics = classification_metrics(labels, scores)
        metrics.update(gate_metrics)
        if fd_details:
            count = scores.shape[1]
            matrix = np.zeros((count, count), dtype=np.int64)
            np.add.at(matrix, (labels, scores.argmax(1)), 1)
            metrics["confusion_matrix"] = matrix.tolist()
            metrics["per_class"] = [
                {"class_id": c, "records": int(matrix[c].sum()), "correct": int(matrix[c, c]),
                 "recall": float(matrix[c, c] / matrix[c].sum()) if matrix[c].sum() else None}
                for c in range(count)]
        return metrics
    targets = torch.cat(all_targets).numpy() if all_targets else np.empty(0, dtype=np.float32)
    predictions = torch.cat(all_predictions).numpy() if all_predictions else np.empty(0, dtype=np.float32)
    if context.is_distributed:
        gathered = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered, (targets, predictions))
        target_parts = [item[0] for item in gathered if item[0].size]
        prediction_parts = [item[1] for item in gathered if item[1].size]
        targets = np.concatenate(target_parts, axis=0) if target_parts else np.empty(0, dtype=np.float32)
        predictions = np.concatenate(prediction_parts, axis=0) if prediction_parts else np.empty(0, dtype=np.float32)
    if not targets.size:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan")}
    return rul_metrics(targets, predictions)


@contextmanager
def preserve_rng_state(device):
    """Periodic diagnostic evaluation must not consume training RNG draws."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


@torch.no_grad()
def initialize_fd_power_spectrum(model: nn.Module, loader: DataLoader, alpha: float = 1.) -> list[int]:
    """Fit once from the complete unsharded training support; no validation loader."""
    device = model.fd_power_spectrum.mean.device
    features, labels = [], []
    for batch in loader:
        features.append(batch["power_features"].cpu())
        labels.append(batch["labels"].cpu())
    # Solve the small ridge system in CPU float64, then copy fitted state to the model.
    normalizer = model.fd_power_spectrum
    raw, labels = torch.cat(features), torch.cat(labels)
    normalizer.fit(raw.to(device))
    transformed = normalizer(raw.to(device)).cpu()
    # initialize_ridge accepts inputs on CPU and copies coefficients to parameters.
    model.head.initialize_ridge(transformed, labels, alpha)
    return torch.bincount(labels, minlength=len(model.head.bias)).tolist()


@torch.no_grad()
def measure_support_gate(model, loader, device):
    """Observe initialization on training support only, without consuming RNG.

    No fitting, gradient updates, threshold selection or query access occurs.
    """
    head = gate_head(model)
    states = [(module, module.training) for module in model.modules()]
    head.reset_gate_stats()
    try:
        with preserve_rng_state(device):
            model.eval()
            for raw in loader:
                batch = _move_batch(raw, device)
                model(batch["signals"], batch["lengths"], batch["channel_mask"],
                      batch.get("sample_periods"), **_task_kwargs(batch))
            return head.gate_stats()
    finally:
        for module, training in states:
            module.training = training
        head.reset_gate_stats()


@torch.no_grad()
def initialize_fd_prototypes(model: nn.Module, loader: DataLoader, device: torch.device) -> list[int]:
    """Initialize cosine directions from unaugmented TRAIN records, never query labels.

    The caller supplies the complete unsharded support loader on every rank.
    DDP subsequently broadcasts rank 0's initialized model as usual.
    """
    head = getattr(model.head, "backbone_head", model.head)
    if not isinstance(head, CosineFDHead):
        raise ValueError("prototype initialization requires a cosine FD head")
    was_training = model.training
    model.eval()
    sums = torch.zeros_like(head.weight)
    counts = torch.zeros(sums.shape[0], device=device, dtype=torch.long)
    try:
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            encoded = model.foundation.encode(
                batch["signals"], batch["lengths"], batch["channel_mask"], batch.get("sample_periods"),
                mask_ratio=0.0, **_rotation_kwargs(batch),
            )
            if getattr(model, "fd_envelope", None) is not None or getattr(model, "fd_power_spectrum", None) is not None:
                pooled, _ = model.fd_features(encoded, batch["signals"], batch["lengths"],
                                              batch["channel_mask"], batch.get("sample_periods"), batch.get("power_features"))
            else:
                pooled = model.pool_fd_features(encoded)
            features = F.normalize(pooled[:, :sums.shape[1]].float(), dim=-1)
            labels = batch["labels"]
            sums.index_add_(0, labels, features)
            counts.index_add_(0, labels, torch.ones_like(labels))
        if (counts == 0).any():
            raise ValueError("Every class needs a training sample for prototype initialization")
        if not torch.isfinite(sums).all() or (sums.norm(dim=-1) <= 1e-12).any():
            raise ValueError("Cannot initialize prototypes from nonfinite or zero class features")
        head.weight.copy_(F.normalize(sums, dim=-1))
    finally:
        model.train(was_training)
    return counts.cpu().tolist()


def train_task_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    context: DistributedContext,
    task: str,
    accumulation_steps: int = 1,
    grad_clip: float = 1.0,
    amp: bool = True,
    amp_dtype: str = "bf16",
    rul_loss_kind: str = "smooth_l1",
    weibull_loss_weight: float = 0.05,
    progress_callback: Optional[Callable[[float], None]] = None,
    fd_contrastive_weight: float = 0.0,
    fd_contrastive_temperature: float = 0.1,
    fd_pair_logging: bool = False,
) -> Dict[str, float]:
    model.train()
    gate = gate_head(model)
    if gate is not None:
        gate.reset_gate_stats()
    pair_metrics = task == "fd" and (fd_pair_logging or fd_contrastive_weight > 0)
    ce_total = contrastive_total = pair_anchors = pair_batches = 0.0
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    steps = 0
    correct = 0
    observed = 0
    envelope_ratio_sum = 0.0
    envelope_samples = 0
    total_batches = len(loader)
    use_scaler = scaler is not None and scaler.is_enabled()
    for batch_index, raw_batch in enumerate(loader):
        batch = _move_batch(raw_batch, context.device)
        with _autocast(context.device, amp, amp_dtype):
            output = model(
                batch["signals"], batch["lengths"], batch["channel_mask"], batch.get("sample_periods"),
                **_task_kwargs(batch),
            )
            if task == "fd":
                loss = fd_loss(output["logits"], batch["labels"])
                if pair_metrics:
                    ce = loss
                    group_ids = [m["group_id"] for m in raw_batch["metadata"]]
                    contrastive, valid_anchors = cross_group_contrastive_loss(
                        output["pooled"], batch["labels"], group_ids, fd_contrastive_temperature)
                    loss = ce + fd_contrastive_weight * contrastive
                    ce_total += float(ce.detach())
                    contrastive_total += float(contrastive.detach())
                    pair_anchors += valid_anchors
                    pair_batches += int(valid_anchors > 0)
            else:
                loss = rul_loss(
                    output["prediction"],
                    batch["targets"],
                    kind=rul_loss_kind,
                    weibull_shape=output["weibull_shape"],
                    weibull_scale=output["weibull_scale"],
                    weibull_weight=weibull_loss_weight,
                )
            group_start = (batch_index // max(1, accumulation_steps)) * max(1, accumulation_steps)
            group_size = min(max(1, accumulation_steps), total_batches - group_start)
            scaled_loss = loss / max(1, group_size)
        if use_scaler:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        if (batch_index + 1) % max(1, accumulation_steps) == 0:
            _step_optimizer(model, optimizer, scheduler, scaler, grad_clip)
        running += float(loss.detach().item())
        steps += 1
        if task == "fd":
            correct += int((output["logits"].detach().argmax(dim=1) == batch["labels"]).sum().item())
            observed += int(batch["labels"].numel())
            if "envelope_residual_ratio" in output:
                envelope_ratio_sum += float(output["envelope_residual_ratio"].sum().item())
                envelope_samples += int(output["envelope_residual_ratio"].numel())
        if progress_callback is not None:
            progress_callback(running / steps)
    if steps and steps % max(1, accumulation_steps):
        _step_optimizer(model, optimizer, scheduler, scaler, grad_clip)
    metrics = {"loss": _reduce_scalar(running / max(steps, 1), context)}
    if task == "fd":
        counts = torch.tensor((correct, observed), device=context.device, dtype=torch.long)
        if context.is_distributed:
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        metrics["train_accuracy"] = float(counts[0].item() / max(counts[1].item(), 1))
        if pair_metrics:
            metrics.update({"classification_loss": _reduce_scalar(ce_total / max(steps, 1), context),
                            "fd_contrastive_loss": _reduce_scalar(contrastive_total / max(steps, 1), context),
                            "fd_contrastive_weight": fd_contrastive_weight,
                            "cross_group_anchor_fraction": _reduce_scalar(pair_anchors / max(observed, 1), context),
                            "cross_group_batch_fraction": _reduce_scalar(pair_batches / max(steps, 1), context),
                            "train_sample_draws_per_rank": observed})
        if envelope_samples:
            totals = torch.tensor((envelope_ratio_sum, envelope_samples), device=context.device, dtype=torch.float64)
            if context.is_distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            metrics["envelope_residual_ratio"] = float((totals[0] / totals[1]).item())
    if gate is not None:
        metrics.update(gate.gate_stats(context))
    return metrics


def build_grad_scaler(enabled: bool) -> Any:
    # GradScaler is useful for fp16 and harmlessly disabled for bf16/CPU.
    active = bool(enabled and torch.cuda.is_available())
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)

