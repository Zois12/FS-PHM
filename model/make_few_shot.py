"""Create a reproducible, leakage-safe PHM few-shot manifest.

Examples:
  python -m model.make_few_shot --task fd --datasets CWRU --shots 1 --output runs/cwru_1shot.json
  python -m model.make_few_shot --task rul --datasets XJTU-SY --shots 2 --unit window --output runs/xjtu_2shot.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .data import H5Index, select_few_shot, write_index_summary
from .fixed_split import load_split, select_support


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a deterministic PHM few-shot manifest")
    parser.add_argument("--task", choices=("fd", "rul"), required=True)
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--root", action="append", help="Explicit H5 file/directory root; repeatable")
    parser.add_argument("--datasets", nargs="+", help="Dataset names, e.g. CWRU or XJTU-SY")
    parser.add_argument("--target-key", default="rul_norm")
    parser.add_argument("--sample-period", type=float, default=None)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--shots", type=int)
    budget.add_argument("--train-fraction", type=float,
                        help="Fraction of each class in the fixed FD train pool (0,1]; requires --split-manifest")
    parser.add_argument("--unit", choices=("window", "group", "trajectory", "file"), default="window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-manifest", help="Fixed FD train-pool/test split; select nested K-shot from its train pool only")
    parser.add_argument("--summary", help="Optional JSON index summary path")
    parser.add_argument("--skip-bad-files", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_fraction is not None and not args.split_manifest:
        raise ValueError("--train-fraction requires --split-manifest for FD")
    index = H5Index.build(
        data_root=args.data_root,
        task=args.task,
        roots=args.root,
        datasets=args.datasets,
        target_key=args.target_key,
        sample_period=args.sample_period,
        strict=not args.skip_bad_files,
    )
    if args.split_manifest:
        if args.task != "fd" or args.unit != "window":
            raise ValueError("Fixed split currently supports FD window shots only")
        pool, _, split = load_split(index, args.split_manifest)
        selection = select_support(pool, args.shots, args.seed, split["split_id"], args.train_fraction)
    else:
        selection = select_few_shot(index, args.shots, args.task, seed=args.seed, unit=args.unit)
    selection.save(args.output)
    if args.summary:
        write_index_summary(index, args.summary)
    result = {
        "manifest": str(Path(args.output).resolve()),
        "task": selection.task,
        "shots": selection.shots,
        "split_id": selection.split_id,
        "train_fraction": selection.train_fraction,
        "selected_per_class": dict(sorted(Counter(f"{r.dataset}:{r.label}" for r in selection.selected).items())),
        "selected_records": len(selection.selected),
        "excluded_groups": len(selection.excluded_group_ids),
        "datasets": sorted({ref.dataset for ref in selection.selected}),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

