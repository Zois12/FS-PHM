"""Fold the fitted Source-Mean offsets into a foundation checkpoint for adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch

from .architecture import FoundationConfig, normalize_backbone_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--branch-ablation", choices=("no-fast", "no-slow", "uniform-router"),
                        help="Set a downstream ablation in the resulting foundation configuration")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    calibration = json.loads(Path(__file__).with_name("source_mean.json").read_text(encoding="utf-8"))
    offsets = calibration["offsets"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"config", "model"} <= checkpoint.keys():
        parser.error("Expected a foundation checkpoint containing config and model")
    config = checkpoint["config"]
    state = checkpoint["model"]
    if not isinstance(config, dict) or not isinstance(state, dict):
        parser.error("Expected dictionary checkpoint configuration and state")
    if "shared_router" in config:
        parser.error("Router calibration is already recorded in this checkpoint")
    if args.branch_ablation:
        existing = config.get("backbone_ablation", "none")
        requested = args.branch_ablation if existing == "none" else f"{existing}+{args.branch_ablation}"
        try:
            ablation = normalize_backbone_ablation(requested)
        except ValueError as exc:
            parser.error(str(exc))
        config = {**config, "backbone_ablation": ablation}
    layer_count = config.get("layers", FoundationConfig.layers)
    if layer_count != len(offsets):
        parser.error(f"Expected {len(offsets)} backbone layers, got {layer_count}")

    for index, offset in enumerate(offsets):
        key = f"backbone.layers.{index}.router.3.bias"
        bias = state.get(key)
        if not isinstance(bias, torch.Tensor) or bias.shape != (2,) or not bias.is_floating_point():
            parser.error(f"Missing or incompatible two-branch router bias: {key}")
        calibrated = bias.clone()
        calibrated[1].add_(offset)
        state[key] = calibrated

    result = {key: checkpoint[key] for key in ("epoch", "step") if key in checkpoint}
    result["model"] = state
    result["config"] = {**config, "shared_router": {"mode": "source_mean", **calibration}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        staged = Path(directory) / "checkpoint.pt"
        torch.save(result, staged)
        args.output.hardlink_to(staged)
    print(args.output)


if __name__ == "__main__":
    main()
