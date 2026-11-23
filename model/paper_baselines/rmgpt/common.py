"""Shared deterministic loading, provenance and first-epoch recovery."""
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

import torch
from torch.utils.data import DataLoader

from ..runner import atomic_checkpoint, restore_rng, seed_all, seed_worker
from .data import collate


def source_identity():
    folder = Path(__file__).parent
    paths = [folder / name for name in ("architecture.py", "data.py", "common.py", "pretrain.py", "finetune.py")]
    model_root = folder.parents[1]
    paths.extend(model_root / name for name in ("data.py", "metrics.py", "pretrain_sampling.py",
                 "paper_baselines/protocol.py", "paper_baselines/runner.py", "paper_baselines/pretrain.py"))
    return {str(p.relative_to(model_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}



def device_setup(args):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Use one process per experiment; this entry does not implement DDP")
    args.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.cuda.set_device(torch.device(args.device))
        if args.amp and not torch.cuda.is_bf16_supported():
            raise ValueError("--amp requires BF16 support")
    if min(args.epochs, args.batch_size) < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch size or workers")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("Invalid learning rate or weight decay")


def loader(dataset, args, epoch=0, shuffle=False, sampler=None):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                      persistent_workers=False, worker_init_fn=seed_worker, collate_fn=collate,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def saved_or_restart(directory, resume):
    directory = Path(directory)
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        if not resume:
            raise ValueError(f"Existing checkpoint: {checkpoint}; use --resume")
        return torch.load(checkpoint, map_location="cpu", weights_only=True)
    if directory.is_dir() and any(directory.iterdir()):
        if not resume:
            raise ValueError(f"Existing output: {directory}; use --resume")
        if any(directory.rglob("*.pt")) or (directory / "test_metrics.json").exists():
            raise ValueError(f"Missing last.pt with other weights/results present: {directory}")
        backup = directory.with_name(directory.name + "_first_epoch_backup_" + uuid.uuid4().hex)
        directory.rename(backup)
        print(f"No last.pt; previous incomplete attempt moved to {backup}. Starting epoch 1.", flush=True)
    return None


def restore_training(saved, model, optimizer, device):
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)
    restore_rng(saved)


def print_event(path, payload):
    from ...run_logging import append_event
    append_event(path, payload)
    print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
