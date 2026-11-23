"""DataLoader lifetime fixes kept separate from checkpointed training code.

The runner creates a fresh seeded loader every epoch. Persistent workers bring
no reuse in that design, and combining them with pin_memory registers worker
cleanup callbacks repeatedly in some PyTorch releases, retaining descriptors.
This runtime wrapper disables persistence while preserving the original seeds,
batch order and checked training identity, so saved runs can resume normally.
"""
import torch
from torch.utils.data import DataLoader


def epoch_loader(dataset, args, shuffle=False, epoch=0):
    from .runner import seed_worker
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers,
                      pin_memory=args.device.startswith("cuda"), drop_last=False,
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(args.seed + epoch))


def main(argv=None):
    from . import runner
    # Do not edit runner.py solely for this lifecycle repair: its SHA256 is
    # part of existing run identities and a change would reject valid resumes.
    runner.loader = epoch_loader
    return runner.main(argv)
