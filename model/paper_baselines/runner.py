"""Fixed-budget supervised experiments using the supplied external models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..fixed_split import digest
from ..metrics import classification_metrics, rul_metrics
from ..run_logging import append_event
from .adapters import SCMamba, build_model
from .data import FDInputs, RULInputs, build_cache, fit_rul_scaler, positions
from .protocol import build_index, fd_protocol, rul_protocol, save_json
from .registry import GENERIC_MODELS, MODELS, REGIMES, normalize_regime, trials


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def loader(dataset, args, shuffle=False, epoch=0):
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=args.device.startswith("cuda"), drop_last=False,
                      persistent_workers=args.workers > 0, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def atomic_checkpoint(path, model, optimizer, epoch, config):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    np_state = np.random.get_state()
    state = {"model_state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
        "epoch": epoch, "config": config, "torch_rng": torch.get_rng_state(),
        "numpy_rng": [np_state[0], torch.tensor(np_state[1].astype(np.int64)), *np_state[2:]],
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_rng(state):
    torch.set_rng_state(state["torch_rng"].cpu())
    rng = state["numpy_rng"]
    np.random.set_state((rng[0], rng[1].cpu().numpy().astype(np.uint32), *rng[2:]))
    random.setstate(state["python_rng"])
    if state["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])


def load_external_weights(model, path, dataset=None, expected_name=None):
    """Require complete encoder coverage; never silently accept partial random weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    pretraining = payload.get("config", {})
    managed = pretraining.get("kind") == "paper_baseline_pretrain_v1"
    if managed:
        if expected_name is not None and pretraining.get("model_name") != expected_name:
            raise ValueError("Pretraining checkpoint model does not match the downstream model")
        seen_datasets = pretraining.get("dataset_names")
        if seen_datasets is None or dataset is not None and dataset.upper() in {d.upper() for d in seen_datasets}:
            raise ValueError(f"Pretraining checkpoint does not prove downstream dataset {dataset} was excluded")
        if pretraining.get("labels_loaded") is not False:
            raise ValueError("Pretraining checkpoint does not prove label-free training")
    weights = payload.get("state_dict", payload.get("model_state_dict", payload.get("model", payload)))
    if not isinstance(weights, dict):
        raise ValueError("Checkpoint does not contain a model state_dict")
    current = model.state_dict()
    matched = {}
    for key, value in weights.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = key.removeprefix("module.")
        if managed:
            if not key.startswith("backbone."):
                continue  # Discard pretraining-only reconstruction heads.
            key = key.removeprefix("backbone.")
        candidates = [key, "network." + key, "network." + key.removeprefix("model.")]
        for target in candidates:
            if target in current and current[target].shape == value.shape:
                if ".head." not in target and not target.startswith("head."):
                    matched[target] = value
                    break
    required = {key for key in current if ".head." not in key and not key.startswith("head.")}
    missing = required - set(matched)
    if missing:
        raise ValueError(f"External checkpoint does not cover the complete backbone ({len(missing)} missing keys, e.g. {sorted(missing)[:8]})")
    model.load_state_dict(matched, strict=False)
    return {"checkpoint": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "loaded_tensors": len(matched), "head": "new downstream initialization",
            "pretraining": {key: pretraining.get(key) for key in ("kind", "model_name", "dataset_names", "excluded_datasets",
                            "labels_loaded", "objective", "run_id", "initialization")} if managed else None}


def sample_support(dataset, classes, per_class):
    chosen = []
    for k in range(classes):
        indices = np.flatnonzero(dataset.labels == k)
        order = torch.randperm(len(indices))[:min(len(indices), per_class)].numpy()
        chosen.extend(indices[order].tolist())
    values = [dataset[i] for i in chosen]
    return torch.stack([v[0] for v in values]), torch.stack([v[1] for v in values])


@torch.inference_mode()
def evaluate(model, dataset, task, args):
    model.eval()
    predictions, truth, sample_indices = [], [], []
    for x, y, indices in tqdm(loader(dataset, args), desc="Test", unit="batch", disable=args.no_progress):
        x = x.to(args.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=args.amp and args.device.startswith("cuda") and not isinstance(model, SCMamba)):
            prediction = model(x)
        prediction = prediction.float().cpu()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Non-finite test predictions; metrics were not written")
        predictions.append(prediction.numpy())
        truth.append(y.numpy())
        sample_indices.extend(indices.tolist())
    pred, target = np.concatenate(predictions), np.concatenate(truth)
    if task == "fd":
        metrics = classification_metrics(target, pred)
    else:
        pred = pred.reshape(-1)
        metrics = rul_metrics(target, pred)
    return metrics, target, pred, sample_indices


def write_predictions(path, index, task, target, prediction, indices, mapping):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        extra = [f"score_class_{label}" for label in mapping] if task == "fd" else []
        writer.writerow(["dataset", "file", "group_id", "row", "target", "prediction", *extra])
        inverse = {v: k for k, v in mapping.items()}
        for y, p, ordinal in zip(target, prediction, indices):
            ref = index.records[ordinal]
            if task == "fd":
                writer.writerow([ref.dataset, Path(ref.path).name, ref.group_id, ref.row,
                                 inverse[int(y)], inverse[int(np.argmax(p))], *p.tolist()])
            else:
                writer.writerow([ref.dataset, Path(ref.path).name, ref.group_id, ref.row, float(y), float(p)])


def run_trial(args, trial, index, cache, train_index, test_index, protocol):
    seed_all(args.seed)
    name, task = trial["model"], trial["task"]
    output = Path(args.output) / name / trial["dataset"] / trial["regime"]
    mapping = protocol.get("class_mapping", {})
    train_positions, test_positions = positions(index, train_index), positions(index, test_index)
    scaler = fit_rul_scaler(cache, train_positions) if task == "rul" else None
    if task == "fd":
        train_data = FDInputs(cache, train_positions, [mapping[r.label] for r in train_index])
        test_data = FDInputs(cache, test_positions, [mapping[r.label] for r in test_index])
    else:
        train_data = RULInputs(index, cache, train_positions, scaler, name, args.history,
                              args.history_stride, args.negatives if name == "dual_mixer" else 0)
        test_data = RULInputs(index, cache, test_positions, scaler, name, args.history, args.history_stride, negatives=0)
    if name in GENERIC_MODELS:
        architecture_path = Path(__file__).parent.parent / "comparisons" / "architectures.py"
        source = {"path": "model/comparisons/architectures.py", "kind": "local_generic_baseline",
                  "sha256": hashlib.sha256(architecture_path.read_bytes()).hexdigest()}
    else:
        source_report = Path(__file__).parent / "vendor" / "sources.json"
        if not source_report.exists():
            raise FileNotFoundError("Extract supplied sources first: python -m model.paper_baselines.prepare_sources")
        source = json.loads(source_report.read_text(encoding="utf-8"))[MODELS[name].source]
    external = args.checkpoints.get(name)
    settings = {key: getattr(args, key) for key in ("epochs", "batch_size", "seed", "history", "history_stride",
        "negatives", "weight_decay", "support_per_class", "hf_backbone", "hf_init", "scan_backend", "amp")}
    settings["lr"] = args.lr if args.lr is not None else MODELS[name].lr
    settings["weight_decay"] = (args.weight_decay if args.weight_decay is not None else
                                1e-4 if name == "unifault" else 0.01 if name == "moment_community" else 0.0)
    settings["optimizer"] = "AdamW" if name in {"unifault", "moment_community"} else "Adam"
    code = {p: hashlib.sha256((Path(__file__).parent / p).read_bytes()).hexdigest()
            for p in ("adapters.py", "data.py", "protocol.py", "runner.py", "registry.py", "compat.py")}
    if name == "moment_community":
        config_name = "hf_configs/" + args.hf_backbone.replace("/", "__") + ".json"
        code[config_name] = hashlib.sha256((Path(__file__).parent / config_name).read_bytes()).hexdigest()
    identity = {"trial": trial, "protocol": protocol, "cache_key": cache.meta["key"], "settings": settings,
                "source": source, "adapter_code": code,
                "external_sha256": hashlib.sha256(Path(external).read_bytes()).hexdigest() if external else None}
    run_id = digest(identity)
    config = {**identity, "run_id": run_id, "args": {**vars(args), "checkpoints": args.checkpoints},
              "scaler": scaler, "model_label": MODELS[name].label, "method_note": MODELS[name].note,
              "selection": "final_epoch_no_test_selection", "validation_records": 0}
    if output.exists() and any(output.iterdir()) and not (args.resume or args.evaluate):
        raise ValueError(f"Trial exists: {output}. Use --resume or choose another --output")
    saved = None
    if args.resume or args.evaluate:
        checkpoint = Path(args.evaluate) if args.evaluate else output / "last.pt"
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved["config"]["run_id"] != run_id:
                raise ValueError(f"Cannot resume/evaluate changed data, source, model or optimization budget: {checkpoint}")
        elif args.evaluate or output.exists() and any(output.iterdir()):
            raise FileNotFoundError(checkpoint)
    if saved and not args.evaluate and (output / "test_metrics.json").exists():
        result = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
        if result["run_id"] != run_id or saved["epoch"] != args.epochs:
            raise ValueError("Saved test result does not match last checkpoint")
        return result
    # A resumed MOMENT already contains all weights; avoid unnecessary HF downloads.
    model = build_model(name, task, len(mapping) if task == "fd" else 1, args.history,
                        args.scan_backend, args.hf_backbone, "random" if saved or external else args.hf_init)
    initialization = "hf_pretrained" if name == "moment_community" and args.hf_init == "pretrained" else "scratch"
    if external and not saved:
        config["external_loading"] = load_external_weights(model, external, trial["dataset"], name)
        initialization = "external_pretrained"
    if saved:
        initialization = saved["config"].get("initialization", initialization)
        if "external_loading" in saved["config"]:
            config["external_loading"] = saved["config"]["external_loading"]
    config["initialization"] = initialization
    model.to(args.device)
    optimizer_class = torch.optim.AdamW if settings["optimizer"] == "AdamW" else torch.optim.Adam
    optimizer = optimizer_class(model.parameters(), lr=settings["lr"],
                                weight_decay=0 if name == "wdcnn" else settings["weight_decay"])
    start = 0
    if saved:
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        # Move optimizer state from CPU to the selected GPU.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(args.device)
        start = saved["epoch"]
        restore_rng(saved)
    output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate:
        save_json(output / "config.json", config)
        append_event(output / "metrics.jsonl", {"event": "start", "model": name,
            "dataset": trial["dataset"], "regime": trial["regime"], "start_epoch": start + 1,
            "epochs": args.epochs, "batch_size": args.batch_size, "train_records": len(train_data),
            "test_records": len(test_data), "initialization": initialization, "run_id": run_id})
        progress = tqdm(total=max(0, args.epochs - start) * math.ceil(len(train_data) / args.batch_size),
                        desc=f"{name} {trial['dataset']} {trial['regime']}", unit="batch", disable=args.no_progress)
        try:
            for epoch in range(start, args.epochs):
                model.train()
                started, total, correct, examples = time.monotonic(), 0.0, 0, 0
                # Re-create the seeded loader per epoch so resume does not depend
                # on persistent worker RNG state from earlier epochs.
                train_loader = loader(train_data, args, shuffle=True, epoch=epoch)
                for batch_idx, (x, y, _) in enumerate(train_loader):
                    # Deterministic schedules need no separate scheduler state on resume.
                    factor = 1.0
                    if name == "unifault":
                        total_steps = len(train_loader) * args.epochs
                        warmup = min(2048, max(1, round(total_steps * 0.1)))
                        step = epoch * len(train_loader) + batch_idx + 1
                        factor = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
                    elif name == "da_rul":
                        factor = 0.98 ** epoch
                    elif name == "sc_mamba":
                        factor = 0.1 ** (epoch // 10)
                    for group in optimizer.param_groups:
                        group["lr"] = settings["lr"] * factor
                    x, y = x.to(args.device, non_blocking=True), y.to(args.device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=args.amp and args.device.startswith("cuda") and name != "sc_mamba"):
                        if name == "sc_mamba":
                            sx, sy = sample_support(train_data, len(mapping), args.support_per_class)
                            loss, prediction = model.support_loss(x, y, sx.to(args.device), sy.to(args.device))
                        elif name == "dual_mixer" and args.negatives:
                            loss, prediction = model.training_loss(x, y)
                        else:
                            prediction = model(x)
                            loss = (F.cross_entropy(prediction, y) if task == "fd" else
                                    F.mse_loss(prediction.reshape(-1), y.reshape(-1)))
                        if hasattr(model, "regularization"):
                            loss = loss + model.regularization()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}")
                    loss.backward()
                    norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    batch = len(x)
                    total, examples = total + loss.item() * batch, examples + batch
                    if task == "fd":
                        correct += int((prediction.argmax(-1) == y).sum())
                    progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss.item():.4f}", refresh=False)
                    progress.update()
                metrics = {"event": "epoch", "epoch": epoch + 1, "train_loss": total / examples,
                    "elapsed_seconds": time.monotonic() - started, "lr": optimizer.param_groups[0]["lr"],
                    "train_records": examples}
                if task == "fd":
                    metrics["train_accuracy"] = correct / examples
                atomic_checkpoint(output / "last.pt", model, optimizer, epoch + 1, config)
                append_event(output / "metrics.jsonl", metrics)
                print(json.dumps(metrics), flush=True)
                del train_loader
        finally:
            progress.close()
    if isinstance(model, SCMamba):
        model.build_support(loader(train_data, args), args.device)
    metrics, target, prediction, indices = evaluate(model, test_data, task, args)
    completed = saved["epoch"] if args.evaluate else args.epochs
    result = {**trial, "model_label": MODELS[name].label, "run_id": run_id, "epoch": completed,
        "train_records": len(train_data), "test_records": len(test_data), "initialization": initialization,
        "split_id": protocol["split_id"], "support_id": protocol.get("support_id"),
        **{f"test_{k}": v if math.isfinite(v) else None for k, v in metrics.items()}}
    stem = "evaluation" if args.evaluate else "test"
    save_json(output / f"{stem}_metrics.json", result)
    write_predictions(output / f"{stem}_predictions.csv", test_index, task, target, prediction, indices, mapping)
    append_event(output / "metrics.jsonl", {"event": stem, **result})
    if not args.evaluate:
        atomic_checkpoint(output / "final.pt", model, optimizer, args.epochs, config)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def parser():
    p = argparse.ArgumentParser(description="External paper models: CWRU/UO FD budgets and XJTU-SY full RUL")
    p.add_argument("--models", nargs="+", default=["all"], choices=["all", *MODELS])
    p.add_argument("--datasets", nargs="+", default=["CWRU", "UO", "XJTU-SY"], choices=["CWRU", "UO", "XJTU-SY"])
    p.add_argument("--regimes", nargs="+", default=["all"], type=normalize_regime, choices=["all", *REGIMES])
    p.add_argument("--data-root", default="splitData")
    p.add_argument("--cwru-root", default="splitData/CWRU_h5_12kdrive")
    p.add_argument("--uo-root", default="splitData/UO_h5")
    p.add_argument("--xjtu-root", default="splitData/XJTU-SY_h5_01")
    p.add_argument("--fd-split-unit", choices=["group", "window"], default="group")
    p.add_argument("--cwru-split")
    p.add_argument("--uo-split")
    p.add_argument("--cwru-support-template", help="Existing support manifest path containing {regime}")
    p.add_argument("--uo-support-template", help="Existing support manifest path containing {regime}")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--target-key", default="rul_norm")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="Default: each supplied implementation's baseline LR")
    p.add_argument("--weight-decay", type=float, default=None, help="Defaults: UniFault 1e-4, MOMENT .01, others 0; WDCNN has original kernel L2")
    p.add_argument("--history", type=int, default=30)
    p.add_argument("--history-stride", type=int, default=1)
    p.add_argument("--negatives", type=int, default=4, help="FSGRI negatives per anchor; 0 means Dual-Mixer backbone only")
    p.add_argument("--support-per-class", type=int, default=16, help="SC-Mamba training reference cap per class; test uses all supports")
    p.add_argument("--scan-backend", choices=["cuda", "reference"], default="cuda")
    p.add_argument("--hf-backbone", choices=["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "t5-small", "t5-base", "t5-large"], default="google/flan-t5-small")
    p.add_argument("--hf-init", choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--checkpoint", action="append", default=[], help="External pretrained checkpoint: model_name=path; repeatable")
    p.add_argument("--pretrain-root", help="Load MODEL/last.pt from comparison pretraining for UniFault, MOMENT and selected generic models")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="List trials without reading H5 or building models")
    p.add_argument("--prepare-only", action="store_true", help="Validate/save protocols and prepare shared data cache")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", help="Evaluate one saved last.pt/final.pt without training")
    p.add_argument("--cache-dir", default="runs/paper_cache")
    p.add_argument("--output", default="runs/paper_comparisons_50ep")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.evaluate:
        saved = torch.load(args.evaluate, map_location="cpu", weights_only=True)
        old_args = saved["config"]["args"]
        # Data/config identity is restored from the run; device and workers can change.
        device, workers, no_progress, evaluate_path = args.device, args.workers, args.no_progress, args.evaluate
        args = argparse.Namespace(**old_args)
        args.device, args.workers, args.no_progress, args.evaluate = device, workers, no_progress, evaluate_path
        trial = saved["config"]["trial"]
        args.models, args.datasets, args.regimes = [trial["model"]], [trial["dataset"]], [trial["regime"]]
        args.resume, args.prepare_only, args.plan_only = False, False, False
    if any(getattr(args, k) < 1 for k in ("epochs", "batch_size", "history", "history_stride", "support_per_class")) or args.workers < 0 or args.negatives < 0:
        raise ValueError("Invalid training/data-loader settings")
    if not 0 < args.test_fraction < 1 or args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        raise ValueError("Invalid test fraction or LR")
    if args.weight_decay is not None and (not math.isfinite(args.weight_decay) or args.weight_decay < 0):
        raise ValueError("weight-decay must be finite and nonnegative")
    args.models = list(MODELS) if args.models == ["all"] else list(dict.fromkeys(args.models))
    args.regimes = list(REGIMES) if args.regimes == ["all"] else list(dict.fromkeys(args.regimes))
    if "all" in args.models or "all" in args.regimes:
        raise ValueError("all must be used alone")
    args.checkpoints = dict(item.split("=", 1) for item in args.checkpoint)
    if getattr(args, "pretrain_root", None):
        from .ssl_models import SSL_MODELS
        for name in args.models:
            if name in SSL_MODELS and name not in args.checkpoints:
                path = Path(args.pretrain_root) / name / "last.pt"
                if not path.is_file():
                    raise FileNotFoundError(f"Pretrained checkpoint missing: {path}; pretrain this model first")
                args.checkpoints[name] = str(path)
        args.checkpoint = [f"{name}={path}" for name, path in args.checkpoints.items()]
    if set(args.checkpoints) - set(MODELS):
        raise ValueError("Unknown --checkpoint model name")
    plan = list(trials(args.models, args.datasets, args.regimes))
    if not plan:
        raise ValueError("Selected models do not support selected datasets")
    if args.plan_only:
        print(json.dumps({"epochs": args.epochs, "batch_size": args.batch_size, "trials": plan}, indent=2))
        return
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("Run independent comparisons on separate GPUs; this entry is one process per experiment, not torchrun")
    args.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(torch.device(args.device))
    if args.amp and args.device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise ValueError("--amp requires BF16 support; omit --amp on this GPU")
    if not args.prepare_only:
        needed = {"einops"} if set(args.models) & {"sc_mamba", "unifault"} else set()
        if "moment_community" in args.models:
            needed.add("transformers")
        if "dual_mixer" in args.models:
            needed.update(("sklearn", "matplotlib"))
        for package in sorted(needed):
            if importlib.util.find_spec(package) is None:
                raise ImportError(f"Missing {package}; install model/paper_baselines/requirements.txt")
        if "sc_mamba" in args.models and args.scan_backend == "cuda":
            if not args.device.startswith("cuda") or importlib.util.find_spec("mamba_ssm") is None:
                raise RuntimeError("SC-Mamba requires CUDA and mamba-ssm; use --scan-backend reference only for CPU verification")
    # Preflight roots for the complete requested matrix before any training starts.
    indices = {dataset: build_index(args, dataset) for dataset in dict.fromkeys(t["dataset"] for t in plan)}
    prepared = {}
    for dataset, index in indices.items():
        protocol = rul_protocol(args, index) if dataset == "XJTU-SY" else fd_protocol(args, dataset, index, args.regimes)
        cache = build_cache(index, "rul" if dataset == "XJTU-SY" else "fd", args.cache_dir, not args.no_progress)
        prepared[dataset] = protocol, cache
    Path(args.output).mkdir(parents=True, exist_ok=True)
    save_json(Path(args.output) / "plan.json", {"epochs": args.epochs, "batch_size": args.batch_size, "trials": plan})
    if args.prepare_only:
        return
    summary = Path(args.output) / "results.jsonl"
    for trial in plan:
        protocols, cache = prepared[trial["dataset"]]
        train, test, protocol = protocols[trial["regime"]]
        result = run_trial(args, trial, indices[trial["dataset"]], cache, train, test, protocol)
        # Keep all model subsets when separate invocations share a matrix output.
        rows = [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines()] if summary.exists() else []
        rows = [r for r in rows if (r["model"], r["dataset"], r["regime"]) != (trial["model"], trial["dataset"], trial["regime"])]
        rows.append(result)
        summary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
