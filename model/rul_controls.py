"""Restore a full-data RUL experiment's exact saved data protocol for controls."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .data import H5Index
from .train_rul_full import build_parser, split_rul_data, window_split_identity


class RULSplitMismatch(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Data counts, targets or sample identities differ from the saved RUL split.\n"
                         + json.dumps(report, ensure_ascii=False, indent=2))


def load_reference(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8-sig"))
    saved, split = config.get("args", {}), config.get("split", {})
    if not saved.get("dataset") or not split or split.get("split_unit") not in {"window", "trajectory"}:
        raise ValueError("Reference must be a train_rul_full output directory with its saved split")
    for key in ("dataset", "target_key", "seed", "test_fraction"):
        if saved.get(key) != split.get(key):
            raise ValueError(f"Reference args and split disagree for {key}")
    if saved.get("split_unit", "trajectory") != split["split_unit"]:
        raise ValueError("Reference args and split disagree for split_unit")
    return config


def inspect_reference(config, roots=None, data_root=None, progress=True):
    """Rebuild and describe the split without fitting or reading vibration rows."""
    saved, split = config["args"], config["split"]
    index = H5Index.build(data_root=data_root or saved.get("data_root", "splitData"), task="rul",
                          roots=roots or saved.get("root"), datasets=[split["dataset"]],
                          target_key=split["target_key"], sample_period=saved.get("sample_period"),
                          strict=True, show_progress=progress)
    return inspect_index(index, config, roots, data_root)


def _split_summary(train, test, unit):
    actual = {"train_records": len(train), "test_records": len(test),
              "train_groups": sorted({r.group_id for r in train}), "test_groups": sorted({r.group_id for r in test})}
    if unit == "window":
        actual.update(window_split_identity(train, test))
    return actual


def _group_signature(group):
    parts = group.split("::", 2)
    if len(parts) < 2:
        return None
    return parts[0], parts[1].replace("\\", "/").rsplit("/", 1)[-1], tuple(parts[2:])


def _verified_group_relocation(index, split):
    """Recover historical group names only when BOTH saved row hashes verify."""
    if split["split_unit"] != "window" or not all(split.get(k) for k in ("train_rows_hash", "test_rows_hash")):
        return None
    saved_groups = set(split["train_groups"]) | set(split["test_groups"])
    current_groups = {r.group_id for r in index}
    if current_groups == saved_groups or len(current_groups) != len(saved_groups):
        return None
    def unique(groups):
        signatures = {}
        for group in groups:
            signature = _group_signature(group)
            if signature is None or signature in signatures:
                return None
            signatures[signature] = group
        return signatures
    saved, current = unique(saved_groups), unique(current_groups)
    if saved is None or current is None or saved.keys() != current.keys():
        return None
    aliases = {current[key]: saved[key] for key in saved if current[key] != saved[key]}
    relocated = H5Index([replace(r, group_id=aliases.get(r.group_id, r.group_id)) for r in index], index.files)
    train, test = split_rul_data(relocated, split["test_fraction"], split["seed"], split["split_unit"])
    actual = _split_summary(train, test, split["split_unit"])
    if any(value != split.get(key) for key, value in actual.items() if key != "split_algorithm"):
        return None
    return train, test, actual, aliases


def inspect_index(index, config, roots=None, data_root=None):
    saved, split = config["args"], config["split"]
    if not len(index) or any(r.target is None for r in index):
        raise ValueError("RUL controls require nonempty data with finite targets")
    train, test = split_rul_data(index, split["test_fraction"], split["seed"], split["split_unit"])
    actual = _split_summary(train, test, split["split_unit"])
    aliases = {}
    if any(actual[key] != split.get(key) for key in ("train_groups", "test_groups")):
        recovered = _verified_group_relocation(index, split)
        if recovered is not None:
            train, test, actual, aliases = recovered
    # Matching both row hashes proves the same declared partition even if an old
    # configuration omitted the descriptive algorithm name. Missing hashes still fail.
    differences = {key: {"saved": split.get(key), "current": value}
                   for key, value in actual.items()
                   if key != "split_algorithm" and value != split.get(key)}
    hints = []
    if any(key.endswith("records") for key in differences):
        hints.append("Window counts changed: check H5 roots, data completeness, and whether the 0.1s/0.2s variant matches the original run.")
    if any(key.endswith("groups") for key in differences):
        hints.append("Group IDs changed: compare group filenames and relative prefixes; data_root/root relocation or source/device metadata can change IDs and window ordering.")
    missing_hashes = [key for key in ("train_rows_hash", "test_rows_hash")
                      if split["split_unit"] == "window" and not split.get(key)]
    if missing_hashes:
        hints.append("The reference window split lacks row hashes. Recover the original saved split/config or original training implementation; counts and seed alone cannot verify the historical test partition.")
    elif any(key.endswith("rows_hash") for key in differences):
        hints.append("Row hashes include dataset, group_id, row, target and physical window seconds. With identical groups/counts, inspect target values, window seconds and the original split implementation.")
    report = {"matches": not differences, "differences": differences,
              "group_id_aliases": aliases,
              "group_alias_verification": "Both saved row hashes match after restoring original group names" if aliases else None,
              "saved_split": split, "current_split": {**split, **actual},
              "data_settings": {"saved_root": saved.get("root"), "current_root": roots or saved.get("root"),
                                "saved_data_root": saved.get("data_root", "splitData"),
                                "current_data_root": data_root or saved.get("data_root", "splitData"),
                                "sample_period_override": saved.get("sample_period"),
                                "current_window_seconds": sorted({r.sample_period for r in index}),
                                "current_target_keys": sorted({f.target_key for f in index.files.values()}),
                                "current_file_count": len(index.files), "current_records": len(index)},
              "hints": hints}
    return train, test, report


def restore_index(index, config):
    train, test, report = inspect_index(index, config)
    if not report["matches"]:
        raise RULSplitMismatch(report)
    return train, test, report


def restore_data(config, roots=None, data_root=None, progress=True, return_report=False):
    train, test, report = inspect_reference(config, roots, data_root, progress)
    if not report["matches"]:
        raise RULSplitMismatch(report)
    # Also record row fingerprints for trajectory splits, whose legacy runs only saved groups/counts.
    identities = window_split_identity(train, test)
    result = (train, test, {key: identities[key] for key in ("train_rows_hash", "test_rows_hash")})
    return (*result, report) if return_report else result


def random_control_args(config, output, device, workers=None, roots=None, data_root=None, progress=True):
    """Serialize the original training budget/options, changing initialization and output."""
    saved = config["args"]
    if saved.get("random_backbone") or not saved.get("checkpoint"):
        raise ValueError("Random control requires a pretrained reference with its original foundation checkpoint")
    if not Path(saved["checkpoint"]).is_file():
        raise ValueError("Original foundation checkpoint was not found; restore it at the saved checkpoint path")
    overrides = {"output": str(output), "device": device, "resume": None, "random_backbone": True,
                 "progress": progress, "skip_bad_files": False, "split_reference": None}
    if workers is not None:
        overrides["workers"] = workers
    if roots is not None:
        overrides["root"] = roots
    if data_root is not None:
        overrides["data_root"] = data_root
    command = []
    for action in build_parser()._actions:
        if action.dest == "help":
            continue
        value = overrides.get(action.dest, saved.get(action.dest, action.default))
        if value is None:
            continue
        option = action.option_strings[0]
        if isinstance(action, argparse.BooleanOptionalAction):
            command.append(option if value else next(s for s in action.option_strings if s.startswith("--no-")))
        elif isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(option)
        elif isinstance(action, argparse._AppendAction):
            for item in value:
                command.extend([option, str(item)])
        else:
            command.extend([option, str(value)])
    return command


def main():
    parser = argparse.ArgumentParser(description="Check an existing RUL split; no model loading or feature fitting")
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--root", action="append")
    parser.add_argument("--data-root")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    config = load_reference(args.reference_run)
    _, _, report = inspect_reference(config, args.root, args.data_root, progress=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
