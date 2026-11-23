"""Train/evaluate the frozen-foundation single-window RUL fusion readout."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .architecture import FoundationConfig, PHMFoundationModel, load_foundation_checkpoint
from .data import H5Index
from .fixed_split import digest
from .metrics import rul_metrics
from .rul_frozen import FrozenRULModel, balanced_weights, extract_features, fit_normalization
from .run_logging import append_event
from .train_rul_full import split_trajectories
from .training import make_scheduler, seed_everything


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="splitData/XJTU-SY_h5_01")
    p.add_argument("--data-root", default="splitData")
    p.add_argument("--checkpoint", help="Foundation checkpoint; not required for --evaluate")
    p.add_argument("--evaluate", help="Evaluate a final.pt/last.pt from this entry without the source foundation file")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output", default="runs/xjtu_frozen")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--sampling-seed", type=int, default=2026)
    p.add_argument("--test-fraction", type=float, default=.2)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--scheduler-steps", type=int, default=600)
    p.add_argument("--phase-steps", type=int, default=325)
    p.add_argument("--floor-lr", type=float, default=1e-8)
    p.add_argument("--device", default="auto")
    p.add_argument("--cpu-threads", type=int, default=1)
    return p


def atomic_save(path, payload):
    temp = Path(path).with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def report(index, prediction, output, prefix="test"):
    y = np.asarray([r.target for r in index], dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite RUL predictions")
    result = {f"{prefix}_{k}": v for k, v in rul_metrics(y, prediction).items()}
    result["groups"] = []
    for group in sorted({r.group_id for r in index}):
        selected = np.array([r.group_id == group for r in index])
        result["groups"].append({"group_id": group, "num_samples": int(selected.sum()),
                                 **rul_metrics(y[selected], prediction[selected])})
    with (output / f"{prefix}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "group_id", "path", "row", "target", "prediction"])
        for r, pred in zip(index, prediction):
            writer.writerow([r.dataset, r.group_id, r.path, r.row, r.target, float(pred)])
    (output / f"{prefix}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


@torch.no_grad()
def predict(model, features, device):
    model.eval()
    predictions = []
    for i in range(0, len(features["targets"]), 512):
        stats = features["stats"][i:i+512].to(device)
        amp = features["amp"][i:i+512].to(device)
        amp = (amp - model.amplitude_mean) / model.amplitude_scale
        predictions.append(model.head(stats, amp).cpu())
    return torch.cat(predictions).numpy()


def main(argv=None):
    a = parser().parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("One process per RUL run; run different seeds on different GPUs")
    if min(a.epochs, a.batch_size, a.eval_batch_size, a.scheduler_steps, a.phase_steps, a.cpu_threads) < 1 or a.workers < 0 or a.warmup_steps < 0:
        raise ValueError("Invalid epoch, batch, worker or scheduler settings")
    if not 0 < a.test_fraction < 1 or not 0 <= a.dropout < 1:
        raise ValueError("Invalid split fraction or dropout")
    if not math.isfinite(a.lr) or a.lr <= 0 or not math.isfinite(a.weight_decay) or a.weight_decay < 0 or not math.isfinite(a.floor_lr) or a.floor_lr < 0:
        raise ValueError("Invalid learning rate or weight decay")
    a.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(a.device))
    torch.set_num_threads(a.cpu_threads)
    torch.set_float32_matmul_precision("high")
    seed_everything(a.seed)
    output = Path(a.output)
    if a.evaluate:
        output.mkdir(parents=True, exist_ok=True)
        saved = torch.load(a.evaluate, map_location="cpu", weights_only=True)
        config = saved["config"]
        if config.get("format") != "fs_phm_rul_frozen_v1":
            raise ValueError("Use a checkpoint generated by model.train_rul_frozen")
    else:
        if not a.checkpoint:
            raise ValueError("Training requires --checkpoint")
        if output.exists() and any(output.iterdir()) and not a.resume:
            raise ValueError("Output is not empty: choose a new directory or use --resume")
        output.mkdir(parents=True, exist_ok=True)
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True) if a.resume and (output / "last.pt").exists() else None
        if a.resume and saved is None:
            raise ValueError("No last.pt to resume; choose a fresh output directory")
    index = H5Index.build(roots=[a.root], data_root=a.data_root, task="rul", datasets=["XJTU-SY"],
                          target_key="rul_norm", strict=True)
    if not len(index) or any(r.target is None or not 0 <= r.target <= 1 for r in index):
        raise ValueError("Expected finite normalized RUL labels in [0, 1]")
    if any(info.target_key != "rul_norm" for info in index.files.values()):
        raise ValueError("Every file must contain rul_norm; implicit fallback is disabled")
    split_seed = config["args"]["split_seed"] if a.evaluate else a.split_seed
    fraction = config["args"]["test_fraction"] if a.evaluate else a.test_fraction
    train, test = split_trajectories(index, fraction, split_seed)
    split = {"unit": "trajectory", "seed": split_seed, "test_fraction": fraction,
             "train_records": len(train), "test_records": len(test),
             "train_groups": sorted({r.group_id for r in train}), "test_groups": sorted({r.group_id for r in test})}
    inventory = digest([(r.dataset, r.group_id, r.row, r.target, r.sample_period) for r in index])
    if a.evaluate:
        if split != config["split"] or inventory != config["inventory"]:
            raise ValueError("Evaluation dataset/split differs from training")
        foundation = PHMFoundationModel(FoundationConfig(**config["foundation_config"]))
        model = FrozenRULModel(foundation, config["normalization"], config["args"]["dropout"])
        model.load_state_dict(saved["model"], strict=True)
        model.to(a.device)
        features = extract_features(test, model.foundation, config["normalization"], a.workers, a.device, a.eval_batch_size)
        print(json.dumps(report(test, predict(model, features, a.device), output, "evaluation"), indent=2))
        return
    normalization = saved["config"]["normalization"] if saved else fit_normalization(train, a.workers, a.device)
    foundation_hash = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    # Match the historical initialization order: seed, foundation, old wrapper, fusion head.
    seed_everything(a.seed)
    foundation = load_foundation_checkpoint(a.checkpoint)
    model = FrozenRULModel(foundation, normalization, a.dropout).to(a.device)
    config = {"format": "fs_phm_rul_frozen_v1", "args": {k: v for k, v in vars(a).items() if k not in {"resume", "evaluate", "device", "workers", "cpu_threads"}},
              "split": split, "inventory": inventory, "normalization": normalization,
              "foundation_sha256": foundation_hash, "foundation_config": asdict(foundation.config),
              "sampling": "condition/bearing balanced replacement; N draws per epoch; entire TRAIN pool eligible",
              "loss": "MAE", "precision": "FP32", "disk_feature_cache": False,
              "selection": "fixed final epoch; TEST evaluated only after fitting",
              "parameters": {"total": sum(p.numel() for p in model.parameters()),
                             "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
    config["run_id"] = digest(config)
    if saved and saved["config"]["run_id"] != config["run_id"]:
        raise ValueError("Resume configuration changed")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else []
    features = extract_features(train, model.foundation, normalization, a.workers, a.device, a.eval_batch_size)
    # Feature DataLoaders consume RNG; historical head fitting reseeded/rebuilt
    # the model after extraction. Keep that same dropout starting state.
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    stats, amp, targets = (features[k].to(a.device) for k in ("stats", "amp", "targets"))
    amp = (amp - model.amplitude_mean) / model.amplitude_scale
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(.9, .95))
    scheduler = make_scheduler(optimizer, a.scheduler_steps, a.warmup_steps)
    generator = torch.Generator().manual_seed(a.sampling_seed)
    weights = balanced_weights(train)
    start, step = 0, 0
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != "step":
                    state[key] = value.to(a.device)
        scheduler.load_state_dict(saved["scheduler"])
        generator.set_state(saved["sampling_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if a.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start, step = saved["epoch"], saved["step"]
    n = len(train)
    for epoch in range(start, a.epochs):
        model.train()
        draws = torch.multinomial(weights, n, replacement=True, generator=generator).to(a.device)
        total = 0.
        bar = tqdm(range(0, n, a.batch_size), desc=f"RUL {epoch+1}/{a.epochs}", unit="batch")
        for offset in bar:
            ids = draws[offset:offset+a.batch_size]
            optimizer.zero_grad(set_to_none=True)
            error = model.head(stats[ids], amp[ids]) - targets[ids]
            loss = error.abs().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RUL training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step >= a.phase_steps:
                for group in optimizer.param_groups:
                    group["lr"] = a.floor_lr
            total += float(error.detach().abs().sum())
            bar.set_postfix(mae=f"{float(loss):.4f}")
        state = {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                 "config": config, "epoch": epoch+1, "step": step, "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "sampling_rng": generator.get_state(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if a.device.startswith("cuda") else []}
        atomic_save(output / "last.pt", state)
        append_event(output / "metrics.jsonl", {"event": "epoch", "epoch": epoch+1, "step": step,
                                               "train_online_mae": total/n, "lr": optimizer.param_groups[0]["lr"]})
    # TEST neither fits normalization nor selects epochs or hyperparameters.
    del features, stats, amp, targets
    features = extract_features(test, model.foundation, normalization, a.workers, a.device, a.eval_batch_size)
    result = report(test, predict(model, features, a.device), output)
    state = {"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "config": config, "epoch": a.epochs, "step": step, "metrics": result}
    atomic_save(output / "final.pt", state)
    append_event(output / "metrics.jsonl", {"event": "test", **result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
