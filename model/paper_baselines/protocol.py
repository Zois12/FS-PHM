"""Shared immutable train/test and support protocols across external models."""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..data import H5Index
from ..fixed_split import digest, load_split, make_split, prepare_fixed, record_id, save_split, select_support


def save_json(path, payload, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"Protocol/config differs from existing file: {path}; choose a new output path")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def build_index(args, dataset):
    name = {"CWRU": "cwru_root", "UO": "uo_root", "XJTU-SY": "xjtu_root"}[dataset]
    root = Path(getattr(args, name))
    if not root.exists():
        raise FileNotFoundError(f"{dataset} root is missing: {root}. Set --{name.replace('_', '-')} to the server H5 directory.")
    task = "rul" if dataset == "XJTU-SY" else "fd"
    index = H5Index.build(data_root=args.data_root, roots=[root], task=task, datasets=[dataset],
                          target_key=args.target_key, show_progress=not args.no_progress)
    if not len(index):
        raise ValueError(f"No {task} records in {root}")
    if any((r.target if task == "rul" else r.label) is None for r in index):
        raise ValueError(f"Missing or non-finite {task} supervision")
    if task == "rul" and any(i.target_key != args.target_key for i in index.files.values()):
        raise ValueError(f"Every RUL file must contain exact target key {args.target_key}; implicit per-bearing target rescaling is disabled")
    return index


def fd_protocol(args, dataset, index, regimes):
    root = Path(args.output) / "protocols" / dataset
    override = getattr(args, "cwru_split" if dataset == "CWRU" else "uo_split")
    path = Path(override) if override else root / "fixed_split.json"
    if not path.exists():
        if override:
            raise FileNotFoundError(path)
        save_split(make_split(index, args.test_fraction, args.seed, args.fd_split_unit), path)
    pool, _, split = load_split(index, path)
    # Explicit manifests define the protocol; their seed/unit may differ from
    # generation defaults, which are ignored once a manifest is supplied.
    save_split(split, root / "fixed_split.json")
    prepared = {}
    template = getattr(args, "cwru_support_template" if dataset == "CWRU" else "uo_support_template")
    for regime in regimes:
        support_path = Path(template.format(regime=regime)) if template else root / f"{regime}.json"
        if not support_path.exists():
            if template:
                raise FileNotFoundError(support_path)
            shots = int(regime[:-4]) if regime.endswith("shot") else None
            fraction = None if shots else (0.1 if regime == "10percent" else 1.0)
            selection = select_support(pool, shots, args.seed, split["split_id"], fraction)
            save_json(support_path, selection.as_dict(), immutable=True)
        train, test, selection, _, protocol = prepare_fixed(index, path, support_path)
        expected = (int(regime[:-4]), None) if regime.endswith("shot") else (0, 0.1 if regime == "10percent" else 1.0)
        if (selection.shots, selection.train_fraction) != expected:
            raise ValueError(f"Manifest budget does not match {regime}: {support_path}")
        save_json(root / f"{regime}.json", selection.as_dict(), immutable=True)
        classes = sorted({int(r.label) for r in pool})
        mapping = {label: i for i, label in enumerate(classes)}
        prepared[regime] = (train, test, {**protocol, "class_mapping": mapping,
                                        "train_records": len(train), "test_records": len(test)})
    return prepared


def rul_protocol(args, index):
    groups = sorted({r.group_id for r in index})
    if len(groups) < 2:
        raise ValueError("XJTU-SY comparison requires at least two bearings")
    path = Path(args.output) / "protocols" / "XJTU-SY" / "fixed_split.json"
    # Same trajectory split algorithm as model.train_rul_full.
    shuffled = groups.copy()
    random.Random(args.seed).shuffle(shuffled)
    count = min(len(groups) - 1, max(1, round(len(groups) * args.test_fraction)))
    test_groups = set(shuffled[:count])
    train = H5Index([r for r in index if r.group_id not in test_groups], index.files)
    test = H5Index([r for r in index if r.group_id in test_groups], index.files)
    identity = [(record_id(r), r.target, r.sample_period) for r in index]
    payload = {"version": 1, "task": "rul", "unit": "trajectory", "dataset": "XJTU-SY",
        "target_key": args.target_key, "seed": args.seed, "test_fraction": args.test_fraction,
        "inventory_hash": digest(identity), "train_groups": sorted(set(groups) - test_groups),
        "test_groups": sorted(test_groups), "train_records": len(train), "test_records": len(test)}
    payload["split_id"] = digest(payload)
    save_json(path, payload, immutable=True)
    return {"full": (train, test, payload)}
