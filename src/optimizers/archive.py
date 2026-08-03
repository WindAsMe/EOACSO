"""Diversity-preserving elite archive (strategy 5).

Maintains a bounded set of non-dominated solutions on the two objectives
(error rate, feature ratio), truncating the most crowded member (by Hamming
distance between the concatenated feature/classifier masks) when the archive
exceeds capacity. Used both to supply elite guidance to the competitive
update (strategy 2) and, after the search finishes, to report alternative
accuracy/parsimony trade-off solutions.
"""

import numpy as np


def _dominates(a, b):
    le = a["error_rate"] <= b["error_rate"] and a["feature_ratio"] <= b["feature_ratio"]
    lt = a["error_rate"] < b["error_rate"] or a["feature_ratio"] < b["feature_ratio"]
    return le and lt


class EliteArchive:
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.entries = []

    def try_insert(self, position, info):
        entry = {
            "position": position.copy(),
            "error_rate": 1.0 - info["balanced_accuracy"],
            "feature_ratio": info["feature_ratio"],
            "mask": np.concatenate([info["feature_mask"], info["clf_mask"]]),
            "info": info,
        }
        if any(_dominates(e, entry) for e in self.entries):
            return
        self.entries = [e for e in self.entries if not _dominates(entry, e)]
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self._truncate()

    def _truncate(self):
        while len(self.entries) > self.max_size:
            masks = np.stack([e["mask"] for e in self.entries])
            n = len(masks)
            dists = (masks[:, None, :] != masks[None, :, :]).sum(axis=2).astype(float)
            np.fill_diagonal(dists, np.inf)
            nearest = dists.min(axis=1)
            del self.entries[int(np.argmin(nearest))]

    def sample_elite(self, rng):
        if not self.entries:
            return None
        idx = rng.integers(len(self.entries))
        return self.entries[idx]["position"]

    def __len__(self):
        return len(self.entries)
