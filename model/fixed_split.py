"""Persisted FD benchmark partitions and nested K-shot support selections.

Fingerprints cover row identities and metadata, not waveform byte contents.
File basenames must be unambiguous within dataset/group/row on every machine.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

from .data import FewShotSelection, H5Index, records_from_selection


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def record_id(ref):
    return json.dumps([ref.dataset, PureWindowsPath(ref.path).name, ref.group_id, ref.row],
                      ensure_ascii=True, separators=(",", ":"))


def inventory(index):
    rows = {}
    for ref in index.records:
        if ref.task != "fd" or ref.label is None:
            raise ValueError("Fixed FD splits require labeled FD records only")
        key = record_id(ref)
        if key in rows:
            raise ValueError("Duplicate/ambiguous portable record identity in FD index")
        info = index.files[ref.path]
        rows[key] = [int(ref.label), ref.sample_period, info.signal_length,
                     info.n_channels, list(info.signal_keys)]
    if not rows:
        raise ValueError("Cannot split an empty FD index")
    return rows


def class_key(ref):
    return ref.dataset, int(ref.label)


def cwru_acquisition(ref):
    """X### denotes a CWRU acquisition even when copied under another filename."""
    if ref.dataset != "CWRU":
        return None
    matches = re.findall(r"(?:device=|_)(X\d+)(?=\.h5|$)", ref.group_id)
    return matches[-1] if matches else None


def reject_cwru_alias_groups(index):
    groups = defaultdict(set)
    for ref in index:
        source = cwru_acquisition(ref)
        if source:
            groups[source].add(ref.group_id)
    aliases = {source: sorted(names) for source, names in groups.items() if len(names) > 1}
    if aliases:
        raise ValueError(f"CWRU acquisition copied into multiple groups: {aliases}. Run model.audit_fd_data and export a clean root before creating group splits/folds.")


def check_cwru_source_isolation(left, right):
    left_sources = {cwru_acquisition(r) for r in left} - {None}
    right_sources = {cwru_acquisition(r) for r in right} - {None}
    if left_sources & right_sources:
        raise ValueError(f"CWRU acquisition overlaps both partitions despite different group names: {sorted(left_sources & right_sources)}. Audit and deduplicate the data before regenerating manifests.")


def make_split(index, test_fraction=0.2, seed=2026, unit="group"):
    if not 0 < test_fraction < 1 or unit not in {"group", "window"}:
        raise ValueError("Invalid test fraction or split unit")
    rows = inventory(index)
    if unit == "group":
        reject_cwru_alias_groups(index)
    classes = defaultdict(list)
    group_labels = {}
    for ref in index.records:
        classes[class_key(ref)].append(ref)
        group = (ref.dataset, ref.group_id)
        if unit == "group" and group_labels.setdefault(group, class_key(ref)) != class_key(ref):
            raise ValueError("A group contains multiple FD classes; cannot stratify by group")
    test_ids = set()
    for key, refs in sorted(classes.items()):
        units = sorted({r.group_id for r in refs}) if unit == "group" else sorted(record_id(r) for r in refs)
        if len(units) < 2:
            raise ValueError(f"Class {key} needs at least two {unit}s for a fixed test split")
        random.Random(digest([seed, key, "partition"])).shuffle(units)
        count = min(len(units) - 1, max(1, round(len(units) * test_fraction)))
        chosen = set(units[:count])
        test_ids.update(record_id(r) for r in refs if (r.group_id if unit == "group" else record_id(r)) in chosen)
    payload = {
        "version": 1, "task": "fd", "unit": unit, "seed": seed,
        "test_fraction": test_fraction, "datasets": sorted({r.dataset for r in index.records}),
        "inventory_hash": digest(rows),
        "train_pool": sorted(set(rows) - test_ids), "test": sorted(test_ids),
    }
    payload["split_id"] = digest(payload)
    return payload


def save_split(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A protocol file is immutable: a different split needs a different filename.
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
        raise ValueError("Split file already exists with different contents; choose a new output path")
    path.write_text(content, encoding="utf-8")


def load_split(index, path, *, check_source_isolation=True):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in payload.items() if k != "split_id"}
    if payload.get("version") != 1 or payload.get("task") != "fd" or payload.get("split_id") != digest(unsigned):
        raise ValueError("Invalid fixed split format or split_id checksum")
    index = index.filter(datasets=payload["datasets"])
    rows = inventory(index)
    if digest(rows) != payload["inventory_hash"]:
        raise ValueError("Dataset inventory changed since split creation (rows/labels/schema/window duration)")
    pool, test = payload["train_pool"], payload["test"]
    if (not pool or not test or len(set(pool)) != len(pool) or len(set(test)) != len(test)
            or set(pool) & set(test) or set(pool) | set(test) != set(rows)):
        raise ValueError("Invalid fixed split: partitions must be disjoint and cover the full index")
    refs = {record_id(r): r for r in index.records}
    pool_index = H5Index([refs[k] for k in pool], index.files)
    test_index = H5Index([refs[k] for k in test], index.files)
    if {class_key(r) for r in pool_index} != {class_key(r) for r in test_index}:
        raise ValueError("Every class must occur in both fixed partitions")
    if payload["unit"] not in {"group", "window"}:
        raise ValueError("Invalid split unit")
    if payload["unit"] == "group" and ({(r.dataset, r.group_id) for r in pool_index} & {(r.dataset, r.group_id) for r in test_index}):
        raise ValueError("Group split has train/test group overlap")
    if payload["unit"] == "group" and check_source_isolation:
        check_cwru_source_isolation(pool_index, test_index)
    return pool_index, test_index, payload


def fraction_count(size, fraction):
    if not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("train_fraction must be finite and in (0, 1]")
    return max(1, math.ceil(size * fraction))


def select_support(pool, shots, seed, split_id, train_fraction=None):
    if train_fraction is not None and shots is not None:
        raise ValueError("Specify shots or train_fraction, not both")
    if train_fraction is None and (shots is None or shots < 1):
        raise ValueError("shots must be >= 1")
    if train_fraction is not None:
        fraction_count(1, train_fraction)
    classes = defaultdict(list)
    for ref in pool:
        classes[class_key(ref)].append(ref)
    selected = []
    for key, refs in sorted(classes.items()):
        refs = sorted(refs, key=record_id)
        # Per-class permutation does not depend on K: S1 is a subset of S2, etc.
        random.Random(digest([seed, key, "support"])).shuffle(refs)
        count = shots if train_fraction is None else fraction_count(len(refs), train_fraction)
        if len(refs) < count:
            raise ValueError(f"Class {key} has only {len(refs)} train-pool windows, fewer than {shots}-shot")
        selected.extend(refs[:count])
    return FewShotSelection("fd", shots if train_fraction is None else 0, seed, "window", selected,
                            sorted({r.group_id for r in selected}), split_id=split_id,
                            train_fraction=train_fraction)


def prepare_fixed(index, split_path, manifest=None, shots=None, seed=42, *, check_source_isolation=True):
    pool, test, payload = load_split(index, split_path, check_source_isolation=check_source_isolation)
    if manifest and shots is not None:
        raise ValueError("Use either --few-shot-manifest or --shots with a fixed split, not both")
    if manifest:
        selection = FewShotSelection.load(manifest)
    elif shots is not None:
        selection = select_support(pool, shots, seed, payload["split_id"])
    else:
        raise ValueError("Fixed split requires a bound few-shot manifest or --shots K")
    if selection.task != "fd" or selection.unit != "window" or selection.split_id != payload["split_id"]:
        raise ValueError("Support manifest must be FD window shots bound to this split_id; regenerate it using --split-manifest")
    support = records_from_selection(pool, selection)
    counts = Counter(class_key(r) for r in support)
    pool_counts = Counter(class_key(r) for r in pool)
    if selection.train_fraction is None:
        if selection.shots < 1:
            raise ValueError("Support must contain exactly K distinct train-pool windows per class")
        expected = {key: selection.shots for key in pool_counts}
    else:
        if selection.shots != 0:
            raise ValueError("Fraction manifest must use shots=0, not a K-shot budget")
        expected = {key: fraction_count(n, selection.train_fraction) for key, n in pool_counts.items()}
    if len({record_id(r) for r in support}) != len(support) or dict(counts) != expected:
        raise ValueError("Support must contain exactly K or the declared fraction of distinct train-pool windows per class")
    for current, saved in zip(support.records, selection.selected):
        if class_key(current) != class_key(saved) or record_id(current) != record_id(saved):
            raise ValueError("Support manifest labels or row identities changed")
    protocol = {
        "name": "fixed_fd_kshot_v1", "split_id": payload["split_id"],
        "support_id": digest(sorted(record_id(r) for r in support)),
        "split_unit": payload["unit"], "shots": selection.shots,
        "support_seed": selection.seed, "train_pool_records": len(pool),
        "test_records": len(test), "selection": "fixed_epochs_no_target_validation",
    }
    if selection.train_fraction is not None:
        protocol.update(name="fixed_fd_fraction_v1", train_fraction=selection.train_fraction,
                        fraction_basis="per_class_train_pool", fraction_rounding="ceil_min_one")
    return support, test, selection, payload, protocol
