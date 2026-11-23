"""Fixed-budget FD/RUL adaptation of the independent paper-derived RmGPT."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from ...fixed_split import digest
from ...metrics import classification_metrics, rul_metrics
from ..data import positions
from ..protocol import build_index, fd_protocol, rul_protocol, save_json
from ..registry import REGIMES, normalize_regime
from ..runner import write_predictions
from .architecture import RmGPT, RmGPTConfig
from .common import (atomic_checkpoint, device_setup, loader, parameter_counts, print_event,
                     restore_training, saved_or_restart, seed_all, source_identity)
from .data import Inputs, build_cache


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=["CWRU", "UO", "XJTU-SY"], default=["CWRU", "UO", "XJTU-SY"])
    p.add_argument("--regimes", nargs="+", choices=["all", *REGIMES], type=normalize_regime, default=["all"])
    p.add_argument("--data-root", default="splitData")
    p.add_argument("--cwru-root", default="splitData/CWRU_h5_12kdrive")
    p.add_argument("--uo-root", default="splitData/UO_h5")
    p.add_argument("--xjtu-root", default="splitData/XJTU-SY_h5_01")
    p.add_argument("--cwru-split", default=None)
    p.add_argument("--cwru-support-template", default=None)
    p.add_argument("--uo-split", default=None)
    p.add_argument("--uo-support-template", default=None)
    p.add_argument("--fd-split-unit", choices=["group", "window"], default="group")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--target-key", default="rul_norm")
    p.add_argument("--checkpoint", help="New RmGPT paper v2 pretraining last.pt; MOMENT weights are incompatible")
    p.add_argument("--adaptation", choices=["prompt", "full"], default="prompt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evaluate", help="Evaluate one downstream last.pt/final.pt using its saved configuration")
    p.add_argument("--cache-dir", default="runs/paper_cache")
    p.add_argument("--output", default="runs/rmgpt_paper_v2/downstream_prompt")
    return p


def trial_plan(args):
    regimes = list(REGIMES) if args.regimes == ["all"] else list(dict.fromkeys(args.regimes))
    if "all" in regimes:
        raise ValueError("all must be used alone")
    return [{"model": "rmgpt_paper_v2", "dataset": dataset, "task": "rul" if dataset == "XJTU-SY" else "fd", "regime": regime}
            for dataset in dict.fromkeys(args.datasets) for regime in (regimes if dataset != "XJTU-SY" else ["full"])]


def validate_pretraining(saved, dataset):
    config = saved.get("config", {})
    if config.get("kind") != "rmgpt_paper_v2_pretrain" or config.get("labels_loaded") is not False:
        raise ValueError("Expected a label-free RmGPT paper v2 pretraining checkpoint")
    names = config.get("dataset_names")
    if names is None or dataset.upper() in {name.upper() for name in names}:
        raise ValueError(f"Pretraining exposure to downstream dataset {dataset} is not allowed in this protocol")


@torch.inference_mode()
def evaluate(model, dataset, args, task):
    model.eval()
    scores, truth, indices = [], [], []
    for batch in tqdm(loader(dataset, args), desc="RmGPT test", unit="batch", disable=args.no_progress):
        x, channels = batch["x"].to(args.device), batch["channels"].to(args.device)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=args.amp and x.is_cuda):
            prediction = model(x, channels).float().cpu()
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite test predictions")
        scores.append(prediction.numpy())
        truth.append(batch["y"].numpy())
        indices.extend(batch["indices"].tolist())
    scores, truth = np.concatenate(scores), np.concatenate(truth)
    metrics = classification_metrics(truth, scores) if task == "fd" else rul_metrics(truth, scores)
    return metrics, truth, scores, indices


def run_trial(args, trial, index, cache, train, test, protocol, pretrained, architecture, checkpoint_hash):
    seed_all(args.seed)
    output = Path(args.output) / args.adaptation / trial["dataset"] / trial["regime"]
    mapping = protocol.get("class_mapping", {})
    train_targets = [mapping[r.label] for r in train] if trial["task"] == "fd" else [r.target for r in train]
    test_targets = [mapping[r.label] for r in test] if trial["task"] == "fd" else [r.target for r in test]
    train_data = Inputs(cache, positions(index, train), train_targets, trial["task"])
    test_data = Inputs(cache, positions(index, test), test_targets, trial["task"])
    settings = {key: getattr(args, key) for key in ("epochs", "batch_size", "seed", "lr", "weight_decay", "amp", "adaptation")}
    identity = {"kind": "rmgpt_paper_v2_downstream", "trial": trial, "architecture": asdict(architecture),
                "protocol": protocol, "settings": settings, "cache_key": cache.meta["key"],
                "pretraining_sha256": checkpoint_hash, "code": source_identity()}
    config = {**identity, "run_id": digest(identity), "args": vars(args),
              "selection": "final_epoch_no_test_selection", "validation_records": 0}
    saved = (torch.load(args.evaluate, map_location="cpu", weights_only=True) if args.evaluate
             else saved_or_restart(output, args.resume))
    if saved and saved["config"]["run_id"] != config["run_id"]:
        raise ValueError(f"Cannot resume/evaluate changed data, code, model or training budget: {output}")
    if saved and not args.evaluate and (output / "test_metrics.json").exists():
        result = json.loads((output / "test_metrics.json").read_text(encoding="utf-8"))
        if saved["epoch"] != args.epochs or result["run_id"] != config["run_id"]:
            raise ValueError("Saved results do not match checkpoint")
        return result
    model = RmGPT(architecture, trial["task"], len(mapping) if trial["task"] == "fd" else 1)
    if saved:
        model.load_state_dict(saved["model_state_dict"], strict=True)
    elif pretrained:
        # Same architecture, strict backbone coverage. New prototypes belong to
        # this dataset/task; they are never imported from pretraining.
        weights = {key.removeprefix("backbone."): value for key, value in pretrained["model_state_dict"].items()
                   if key.startswith("backbone.")}
        model.backbone.load_state_dict(weights, strict=True)
    model.set_adaptation(args.adaptation)
    model.to(args.device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    if saved and not args.evaluate:
        restore_training(saved, model, optimizer, args.device)
        start = saved["epoch"]
    config["parameters"] = parameter_counts(model)
    output.mkdir(parents=True, exist_ok=True)
    if not args.evaluate:
        save_json(output / "config.json", config)
        print_event(output / "metrics.jsonl", {"event": "start", **trial, "start_epoch": start + 1,
                    "parameters": config["parameters"], "train_records": len(train), "test_records": len(test)})
        progress = tqdm(total=(args.epochs - start) * math.ceil(len(train_data) / args.batch_size),
                        desc=f"RmGPT {args.adaptation} {trial['dataset']} {trial['regime']}", unit="batch", disable=args.no_progress)
        try:
            for epoch in range(start, args.epochs):
                model.train()
                total, seen, correct, began = 0.0, 0, 0, time.monotonic()
                batches = loader(train_data, args, epoch, shuffle=True)
                for batch in batches:
                    x, channels, y = (batch[key].to(args.device) for key in ("x", "channels", "y"))
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=args.amp and x.is_cuda):
                        prediction = model(x, channels)
                        loss = F.cross_entropy(prediction, y) if trial["task"] == "fd" else F.mse_loss(prediction, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite downstream loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    total += loss.item() * len(x)
                    seen += len(x)
                    if trial["task"] == "fd":
                        correct += int((prediction.argmax(-1) == y).sum())
                    progress.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss.item():.4f}")
                    progress.update()
                atomic_checkpoint(output / "last.pt", model, optimizer, epoch + 1, config)
                metrics = {"event": "epoch", "epoch": epoch + 1, "train_loss": total / seen,
                           "lr": args.lr, "elapsed_seconds": time.monotonic() - began, "train_records": seen}
                if trial["task"] == "fd":
                    metrics["train_accuracy"] = correct / seen
                print_event(output / "metrics.jsonl", metrics)
                del batches
        finally:
            progress.close()
    metrics, truth, prediction, ordinals = evaluate(model, test_data, args, trial["task"])
    result = {**trial, "adaptation": args.adaptation, "run_id": config["run_id"],
              "epoch": saved["epoch"] if args.evaluate else args.epochs, "train_records": len(train),
              "test_records": len(test), "split_id": protocol["split_id"], "support_id": protocol.get("support_id"),
              **{f"test_{key}": float(value) if math.isfinite(value) else None for key, value in metrics.items()}}
    stem = "evaluation" if args.evaluate else "test"
    save_json(output / f"{stem}_metrics.json", result)
    write_predictions(output / f"{stem}_predictions.csv", test, trial["task"], truth, prediction, ordinals, mapping)
    print_event(output / "metrics.jsonl", {"event": stem, **result})
    if not args.evaluate:
        atomic_checkpoint(output / "final.pt", model, optimizer, args.epochs, config)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    only_trial = None
    reference_config = None
    if args.evaluate:
        reference_config = torch.load(args.evaluate, map_location="cpu", weights_only=True)["config"]
        if reference_config.get("kind") != "rmgpt_paper_v2_downstream":
            raise ValueError("Expected a RmGPT paper v2 downstream checkpoint")
        device, workers, no_progress, path = args.device, args.workers, args.no_progress, args.evaluate
        args = argparse.Namespace(**reference_config["args"])
        args.device, args.workers, args.no_progress, args.evaluate = device, workers, no_progress, path
        args.resume, args.prepare_only, args.plan_only = False, False, False
        only_trial = reference_config["trial"]
    plan = [only_trial] if only_trial else trial_plan(args)
    if args.plan_only:
        print(json.dumps({"epochs": args.epochs, "batch_size": args.batch_size, "adaptation": args.adaptation, "trials": plan}, indent=2))
        return
    device_setup(args)
    if not 0 < args.test_fraction < 1:
        raise ValueError("test-fraction must be between 0 and 1")
    pretrained = None
    checkpoint_hash = None
    if args.evaluate:
        # All required weights and provenance are already inside downstream last.pt.
        architecture = RmGPTConfig(**reference_config["architecture"])
        checkpoint_hash = reference_config["pretraining_sha256"]
    elif args.checkpoint:
        pretrained = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        for dataset in {t["dataset"] for t in plan}:
            validate_pretraining(pretrained, dataset)
        architecture = RmGPTConfig(**pretrained["config"]["architecture"])
        checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    else:
        if args.adaptation == "prompt":
            raise ValueError("Prompt adaptation requires --checkpoint from the new pretraining run")
        architecture = RmGPTConfig()
    # Preflight every selected dataset before training the first trial.
    prepared = {}
    for dataset in dict.fromkeys(t["dataset"] for t in plan):
        index = build_index(args, dataset)
        regimes = list(dict.fromkeys(t["regime"] for t in plan if t["dataset"] == dataset))
        protocols = rul_protocol(args, index) if dataset == "XJTU-SY" else fd_protocol(args, dataset, index, regimes)
        cache = build_cache(index, args.cache_dir, architecture.points, not args.no_progress)
        if cache.meta["max_channels"] > architecture.max_channels:
            raise ValueError("Dataset has more channels than the pretrained architecture supports")
        prepared[dataset] = index, protocols, cache
    if not args.evaluate:
        save_json(Path(args.output) / "plan.json", {"trials": plan, "adaptation": args.adaptation})
    if args.prepare_only:
        return
    for trial in plan:
        index, protocols, cache = prepared[trial["dataset"]]
        train, test, protocol = protocols[trial["regime"]]
        result = run_trial(args, trial, index, cache, train, test, protocol,
                           pretrained, architecture, checkpoint_hash)
        if not args.evaluate:
            summary = Path(args.output) / "results.jsonl"
            rows = [json.loads(line) for line in summary.read_text(encoding="utf-8").splitlines() if line.strip()] if summary.exists() else []
            rows = [r for r in rows if (r["dataset"], r["regime"], r["adaptation"]) !=
                    (result["dataset"], result["regime"], result["adaptation"])]
            rows.append(result)
            summary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
