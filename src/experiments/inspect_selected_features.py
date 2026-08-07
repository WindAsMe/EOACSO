"""Recover which specific acoustic features the proposed method (CSO_searched_tf)
selects on its single best-performing run per dataset, for the feature-
interpretability discussion in the paper (Section 5.4's logged limitation).

`run_fs_experiment.py`'s RECORD_FIELDS never wrote `best_info["feature_mask"]`
to results/tables/*_comparison_results.csv, only the feature *count*
(feature_ratio/n_features) -- so which named features were selected cannot be
recovered from the saved comparison results. This script reruns CSO_searched_tf
with the exact (dataset, seed) of the best (lowest best_fitness) run already on
record for each dataset in results/tables/*_comparison_results.csv:
  oxford:  run=9,  seed=2476 (best_fitness=0.145988, balanced_accuracy=0.858)
  naranjo: run=15, seed=2776 (best_fitness=0.143750, balanced_accuracy=0.863)
so this is a deterministic reproduction of an already-observed result, not a
new/different one -- printed fitness/balanced_accuracy/n_features should
exactly match those rows, and the feature mask is simply read off
`result.best_info["feature_mask"]`, which was computed all along but never
persisted.

n_runs=1 per dataset by design (reproducing one specific already-identified
best run, not sampling a fresh distribution), but each run independently costs
the same ~1.7-5.8h observed for other (algorithm, dataset, run) triples under
multi_hot classifier_encoding (see hpc/README.md) -- the two datasets run in
parallel via ProcessPoolExecutor so this finishes in ~1 run's wall-clock time,
not 2, given >=2 cores.

Run with: python -m src.experiments.inspect_selected_features
"""

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.data_loader import load
from src.experiments.run_fs_experiment import _limit_threads_per_worker
from src.fitness import FitnessEvaluator
from src.optimizers import run_cso

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "tables"
OUT_PATH = RESULTS_DIR / "selected_features.csv"

POP_SIZE = 30
MAX_EVALUATIONS = 3000
CV_FOLDS = 5

# (dataset, seed) of the best (lowest best_fitness) CSO_searched_tf row
# already on record for each dataset -- see module docstring.
TARGETS = [("oxford", 2476), ("naranjo", 2776)]

RECORD_FIELDS = [
    "dataset", "seed", "best_fitness", "balanced_accuracy", "n_features",
    "n_total_features", "feature_ratio", "active_classifiers", "selected_features",
]


def _inspect_one(dataset_name, seed):
    ds = load(dataset_name)
    n_generations = max(1, round((MAX_EVALUATIONS - POP_SIZE) / (POP_SIZE / 2.0)))
    evaluator = FitnessEvaluator(ds.X, ds.y, ds.groups, cv_folds=CV_FOLDS, seed=seed)
    result = run_cso(
        evaluator, ds.X.shape[1], pop_size=POP_SIZE, n_generations=n_generations,
        seed=seed, max_evaluations=MAX_EVALUATIONS, classifier_encoding="multi_hot",
    )
    info = result.best_info
    mask = info["feature_mask"]
    selected = [name for name, on in zip(ds.feature_names, mask) if on]
    return {
        "dataset": dataset_name,
        "seed": seed,
        "best_fitness": result.best_fitness,
        "balanced_accuracy": info["balanced_accuracy"],
        "n_features": info["n_features"],
        "n_total_features": len(ds.feature_names),
        "feature_ratio": info["feature_ratio"],
        "active_classifiers": ",".join(info["active_classifiers"]),
        "selected_features": json.dumps(selected),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    with ProcessPoolExecutor(max_workers=len(TARGETS), initializer=_limit_threads_per_worker) as pool:
        futures = {pool.submit(_inspect_one, ds, seed): (ds, seed) for ds, seed in TARGETS}
        for future in as_completed(futures):
            ds, seed = futures[future]
            record = future.result()
            records.append(record)
            print(
                f"[{ds} seed={seed}] best_fitness={record['best_fitness']:.6f} "
                f"balanced_accuracy={record['balanced_accuracy']:.6f} "
                f"n_features={record['n_features']}/{record['n_total_features']} "
                f"active_classifiers={record['active_classifiers']}"
            )
            print(f"  selected_features={json.loads(record['selected_features'])}")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda r: r["dataset"]):
            writer.writerow(record)
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
