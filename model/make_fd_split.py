"""Create an immutable class-stratified FD train-pool/test manifest once."""
import argparse
import json
from collections import Counter

from .data import H5Index
from .fixed_split import load_split, make_split, save_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="splitData")
    parser.add_argument("--root", action="append")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--sample-period", type=float)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--unit", choices=("group", "window"), default="group")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    index = H5Index.build(data_root=args.data_root, task="fd", roots=args.root,
                          datasets=args.datasets, sample_period=args.sample_period)
    payload = make_split(index, args.test_fraction, args.seed, args.unit)
    save_split(payload, args.output)
    pool, test, _ = load_split(index, args.output)
    counts = lambda part: dict(sorted(Counter(f"{r.dataset}:{r.label}" for r in part).items()))
    print(json.dumps({"split_id": payload["split_id"], "unit": args.unit,
                      "train_pool_records": len(pool), "test_records": len(test),
                      "actual_test_fraction": len(test) / len(index),
                      "train_pool_per_class": counts(pool), "test_per_class": counts(test)}, indent=2))


if __name__ == "__main__":
    main()
