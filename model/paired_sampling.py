"""FD batches of same-class pairs, preferring different training groups."""
import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class FDPairBatchSampler(Sampler):
    def __init__(self, records, batch_size, seed=0, num_replicas=1, rank=0):
        if batch_size < 4 or batch_size % 2:
            raise ValueError("FD paired sampling requires an even batch-size >= 4")
        if num_replicas < 1 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid paired sampler rank/world size")
        self.groups = defaultdict(lambda: defaultdict(list))
        for i, ref in enumerate(records):
            if ref.label is None:
                raise ValueError("FD paired sampling requires labels")
            self.groups[(ref.dataset, int(ref.label))][ref.group_id].append(i)
        self.labels = sorted(self.groups)
        counts = {c: sum(map(len, self.groups[c].values())) for c in self.labels}
        if len(counts) < 2 or min(counts.values()) < 2:
            raise ValueError("FD paired sampling needs >=2 classes and >=2 support rows per class; not for 1-shot")
        self.pairs_per_batch = batch_size // 2
        if self.pairs_per_batch > len(self.labels):
            raise ValueError("Paired batch has more class pairs than available classes")
        pair_counts = [math.ceil(n / 2) for n in counts.values()]
        self.global_batches = max(math.ceil(sum(pair_counts) / self.pairs_per_batch), max(pair_counts))
        self.num_replicas, self.rank = num_replicas, rank
        self.seed, self.epoch = seed, 0
        self.summary = {"name": "class_pairs_prefer_cross_group_v1", "batch_size": batch_size,
                        "classes_per_batch": self.pairs_per_batch, "unique_support_records": sum(counts.values()),
                        "classes": len(counts), "classes_with_cross_group_support": sum(len(self.groups[c]) > 1 for c in self.labels),
                        "global_batches_before_ddp_padding": self.global_batches,
                        "batches_per_rank": len(self), "world_size": num_replicas,
                        "note": "Covers all support rows per global epoch; odd classes, batch fill and DDP padding may repeat rows. No additional labeled rows."}

    def __len__(self):
        return math.ceil(self.global_batches / self.num_replicas)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        queues, originals = {}, {}
        for c in self.labels:
            groups = {g: rows.copy() for g, rows in sorted(self.groups[c].items())}
            for rows in groups.values():
                rng.shuffle(rows)
            pairs = []
            # Largest groups first maximizes opportunities for cross-group pairs.
            while any(groups.values()):
                active = [g for g in groups if groups[g]]
                rng.shuffle(active)
                active.sort(key=lambda g: -len(groups[g]))
                first_group = active[0]
                first = groups[first_group].pop()
                if len(active) > 1:
                    second = groups[active[1]].pop()
                elif groups[first_group]:
                    second = groups[first_group].pop()
                else:
                    other = [i for g, rows in self.groups[c].items() if g != first_group for i in rows]
                    candidates = other or [i for rows in self.groups[c].values() for i in rows if i != first]
                    second = rng.choice(candidates)
                pairs.append([first, second])
            rng.shuffle(pairs)
            queues[c], originals[c] = pairs.copy(), pairs
        batches = []
        for _ in range(self.global_batches):
            labels = self.labels.copy()
            rng.shuffle(labels)
            labels.sort(key=lambda c: -len(queues[c]))
            chosen = labels[:self.pairs_per_batch]
            batch = []
            for c in chosen:
                batch.extend(queues[c].pop() if queues[c] else rng.choice(originals[c]))
            rng.shuffle(batch)
            batches.append(batch)
        if any(queues.values()):
            raise RuntimeError("Paired sampler failed to cover support")
        # Every rank has equal steps and intact local positive/negative pairs.
        target = len(self) * self.num_replicas
        batches += [batches[i % len(batches)].copy() for i in range(target - len(batches))]
        yield from batches[self.rank::self.num_replicas]
