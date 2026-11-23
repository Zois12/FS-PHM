"""FD group folds restricted to the training pool of an existing outer split."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .data import FewShotSelection, H5Index, records_from_selection
from .fixed_split import (class_key, digest, load_split, record_id,
                          reject_cwru_alias_groups, check_cwru_source_isolation)


def save_immutable(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError(f"Existing manifest differs; choose a new output directory: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_inner_folds(index, outer_split_path, folds=3, seed=2026):
    pool, outer_test, outer = load_split(index, outer_split_path)
    if outer["unit"] != "group":
        raise ValueError("Internal group validation requires an outer group split")
    reject_cwru_alias_groups(pool)
    if folds < 2:
        raise ValueError("Need at least two internal folds")
    classes = defaultdict(lambda: defaultdict(list))
    owners = {}
    for ref in pool:
        key = class_key(ref)
        group = (ref.dataset, ref.group_id)
        if owners.setdefault(group, key) != key:
            raise ValueError("A group contains multiple FD labels")
        classes[key][ref.group_id].append(ref)
    held_out = [set() for _ in range(folds)]
    for key, groups in sorted(classes.items()):
        names = sorted(groups)
        if len(names) < folds:
            raise ValueError(f"Class {key} has {len(names)} training groups, fewer than {folds} folds")
        random.Random(digest([seed, key, "inner_groups"])).shuffle(names)
        for i, group in enumerate(names):
            held_out[i % folds].update(record_id(r) for r in groups[group])
    pool_ids = {record_id(r) for r in pool}
    result = []
    for i, query in enumerate(held_out):
        payload = {"version": 1, "kind": "fd_inner_group_v1", "task": "fd",
                   "outer_split_id": outer["split_id"], "fold": i + 1, "folds": folds,
                   "seed": seed, "train_pool": sorted(pool_ids - query), "validation": sorted(query)}
        payload["fold_id"] = digest(payload)
        result.append(payload)
    return result


def load_inner_partition(index, outer_split_path, fold_path, *, check_source_isolation=True):
    pool, outer_test, outer = load_split(index, outer_split_path, check_source_isolation=check_source_isolation)
    fold = json.loads(Path(fold_path).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in fold.items() if k != "fold_id"}
    if (fold.get("version") != 1 or fold.get("kind") != "fd_inner_group_v1"
            or fold.get("task") != "fd" or fold.get("fold_id") != digest(unsigned)):
        raise ValueError("Invalid internal fold checksum/format")
    if (outer["unit"] != "group" or fold.get("outer_split_id") != outer["split_id"]
            or not 1 <= fold.get("fold", 0) <= fold.get("folds", 0) or fold.get("folds", 0) < 2):
        raise ValueError("Internal fold does not match the outer group split")
    refs = {record_id(r): r for r in pool}
    train, query = fold["train_pool"], fold["validation"]
    if (not train or not query or len(set(train)) != len(train) or len(set(query)) != len(query)
            or set(train) & set(query) or set(train) | set(query) != set(refs)):
        raise ValueError("Internal partitions must be disjoint and cover only the outer training pool")
    train = H5Index([refs[k] for k in train], index.files)
    query = H5Index([refs[k] for k in query], index.files)
    if ({class_key(r) for r in train} != {class_key(r) for r in query}
            or {(r.dataset, r.group_id) for r in train} & {(r.dataset, r.group_id) for r in query}):
        raise ValueError("Internal partitions need every class and disjoint groups")
    if check_source_isolation:
        check_cwru_source_isolation(train, query)
    # load_split enforces outer group isolation; the complete inner union equals its pool.
    return train, query, fold, outer, len(outer_test)


def balanced_group_support(pool, shots, seed, fold_id):
    if shots < 1:
        raise ValueError("shots must be positive")
    classes = defaultdict(lambda: defaultdict(list))
    for ref in pool:
        classes[class_key(ref)][ref.group_id].append(ref)
    selected = []
    for key, groups in sorted(classes.items()):
        rng = random.Random(digest([seed, key, "inner_support"]))
        queues = {g: sorted(rows, key=record_id) for g, rows in sorted(groups.items())}
        if sum(map(len, queues.values())) < shots:
            raise ValueError(f"Class {key} has fewer than {shots} candidate windows")
        for rows in queues.values():
            rng.shuffle(rows)
        chosen = []
        while len(chosen) < shots:
            names = [g for g in queues if queues[g]]
            rng.shuffle(names)
            for group in names:
                chosen.append(queues[group].pop())
                if len(chosen) == shots:
                    break
        selected.extend(chosen)
    return FewShotSelection("fd", shots, seed, "window", selected,
                            sorted({r.group_id for r in selected}), split_id=fold_id)


def prepare_inner(index, outer_split_path, fold_path, support_path, *, check_source_isolation=True):
    pool, query, fold, outer, excluded_count = load_inner_partition(
        index, outer_split_path, fold_path, check_source_isolation=check_source_isolation)
    selection = FewShotSelection.load(support_path)
    if (selection.task != "fd" or selection.unit != "window" or selection.shots < 1
            or selection.train_fraction is not None or selection.split_id != fold["fold_id"]):
        raise ValueError("Internal support must be K-shot windows bound to this fold_id")
    support = records_from_selection(pool, selection)
    expected = {class_key(r): selection.shots for r in pool}
    if (len({record_id(r) for r in support}) != len(support)
            or dict(Counter(class_key(r) for r in support)) != expected):
        raise ValueError("Internal support must contain exactly K distinct windows per class")
    for current, saved in zip(support, selection.selected):
        if record_id(current) != record_id(saved) or class_key(current) != class_key(saved):
            raise ValueError("Internal support identities/labels changed")
    protocol = {"name": "fd_inner_group_validation_v1", "evaluation_partition": "internal_validation",
                "outer_split_id": outer["split_id"], "split_id": fold["fold_id"],
                "support_id": digest(sorted(record_id(r) for r in support)), "split_unit": "group",
                "fold": fold["fold"], "folds": fold["folds"], "shots": selection.shots,
                "support_seed": selection.seed, "train_pool_records": len(pool),
                "validation_records": len(query), "excluded_outer_test_records": excluded_count,
                "selection": "internal_validation_macro_f1"}
    return support, query, selection, fold, protocol


def write_validation_report(directory, epoch, metrics, class_keys, protocol):
    details = metrics.pop("per_class")
    matrix = metrics.pop("confusion_matrix")
    for row, (dataset, label) in zip(details, class_keys):
        row.update(dataset=dataset, original_label=label)
    report = {"epoch": epoch, "evaluation_partition": "internal_validation", "protocol": protocol,
              "metrics": dict(metrics), "classes": details, "confusion_matrix": matrix}
    path = Path(directory) / "validation" / f"epoch_{epoch:04d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
