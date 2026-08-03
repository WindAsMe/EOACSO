"""Sensitivity analysis for CSO's social-term coefficient phi (Cheng & Jin
2015, Eq. 25-26), scoped to the proposed method (CSO_searched_tf) only --
phi doesn't exist in any of the 7 reproduced baselines, so this script never
touches them.

At this project's swarm size (pop_size=30 <= 100), `cso_phi(pop_size)`
(`src/optimizers/cso.py`) is exactly 0, so the main comparison's proposed-
method runs have no mean-position pull at all in the loser update, only
winner-attraction and inertia. This script instead fixes phi to each of
`--phi_values` in turn (default: 0.0, 0.05, ..., 0.50) and reruns the full
joint optimization (feature + classifier + transfer-function search) under
each, to see how sensitive the proposed pipeline is to this coefficient --
a deliberate departure from `cso_phi`'s formula, not a reproduction of it.

Mirrors `run_fs_experiment.py`'s parallelization/incremental-CSV/resume
pattern, but sweeps `run_cso`'s `fixed_phi` directly instead of iterating
over `ALGORITHMS`, since phi isn't part of that module's uniform 7-arg
runner contract.
"""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data_loader import load
from src.experiments.run_fs_experiment import _limit_threads_per_worker
from src.fitness import FitnessEvaluator
from src.optimizers import run_cso

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

PHI_VALUES_DEFAULT = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

RECORD_FIELDS = [
    "phi",
    "dataset",
    "run",
    "seed",
    "best_fitness",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "roc_auc",
    "feature_ratio",
    "n_features",
    "active_classifiers",
    "n_evaluations",
    "transfer_function",
    "elapsed_sec",
    "history",
]


def _run_single_phi(
    phi, dataset_name, run_idx, base_seed, pop_size, max_evaluations, cv_folds, classifier_encoding="multi_hot"
):
    seed = base_seed + 50 * run_idx  # e.g. base_seed=2026 -> 2026, 2076, 2126, ...
    ds = load(dataset_name)
    n_generations = max(1, round((max_evaluations - pop_size) / (pop_size / 2.0)))
    evaluator = FitnessEvaluator(ds.X, ds.y, ds.groups, cv_folds=cv_folds, seed=seed)
    t0 = time.time()
    result = run_cso(
        evaluator,
        ds.X.shape[1],
        pop_size=pop_size,
        n_generations=n_generations,
        seed=seed,
        max_evaluations=max_evaluations,
        classifier_encoding=classifier_encoding,
        fixed_phi=phi,
    )
    elapsed = time.time() - t0
    return {
        "phi": phi,
        "dataset": dataset_name,
        "run": run_idx,
        "seed": seed,
        "best_fitness": result.best_fitness,
        "balanced_accuracy": result.best_info["balanced_accuracy"],
        "accuracy": result.best_info["accuracy"],
        "precision": result.best_info["precision"],
        "recall": result.best_info["recall"],
        "specificity": result.best_info["specificity"],
        "f1": result.best_info["f1"],
        "roc_auc": result.best_info["roc_auc"],
        "feature_ratio": result.best_info["feature_ratio"],
        "n_features": result.best_info["n_features"],
        "active_classifiers": ",".join(result.best_info["active_classifiers"]),
        "n_evaluations": result.n_evaluations,
        "transfer_function": result.best_info.get("tf_name", ""),
        "elapsed_sec": elapsed,
        "history": json.dumps(result.history),
    }


def run_phi_values(
    phi_values,
    dataset_names,
    n_runs,
    pop_size,
    max_evaluations,
    base_seed=2026,
    cv_folds=5,
    n_workers=None,
    verbose=True,
    on_result=None,
    skip=None,
    classifier_encoding="multi_hot",
):
    """Runs every (phi, dataset, run) triple in `phi_values` x `dataset_names`
    x range(n_runs) in parallel across a single process pool -- same pattern
    as `run_fs_experiment.run_algorithms`, just keyed by phi instead of
    algorithm name."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    n_workers = n_workers or os.cpu_count()
    skip = skip or set()
    all_tasks = [(phi, ds, run_idx) for phi in phi_values for ds in dataset_names for run_idx in range(n_runs)]
    tasks = [t for t in all_tasks if t not in skip]
    if verbose and skip:
        print(f"resume: skipping {len(all_tasks) - len(tasks)} already-completed of {len(all_tasks)} total runs")
    records = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_limit_threads_per_worker) as pool:
        futures = {
            pool.submit(
                _run_single_phi, phi, ds, run_idx, base_seed, pop_size, max_evaluations, cv_folds, classifier_encoding
            ): (phi, ds, run_idx)
            for phi, ds, run_idx in tasks
        }
        for future in as_completed(futures):
            phi, ds, run_idx = futures[future]
            record = future.result()
            records.append(record)
            if on_result is not None:
                on_result(record)
            if verbose:
                print(
                    f"[phi={phi}|{ds}] run {run_idx + 1}/{n_runs} "
                    f"fitness={record['best_fitness']:.4f} bal_acc={record['balanced_accuracy']:.4f} "
                    f"n_features={record['n_features']} clf={record['active_classifiers']} "
                    f"n_evals={record['n_evaluations']} time={record['elapsed_sec']:.1f}s "
                    f"({len(records)}/{len(tasks)} done)"
                )
    return records


def _load_completed(path):
    """Read a possibly-partial results CSV from an interrupted run and return
    the set of (phi, dataset, run) triples it already contains."""
    completed = set()
    if not path.exists() or path.stat().st_size == 0:
        return completed
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((float(row["phi"]), row["dataset"], int(row["run"])))
    return completed


def main(
    dataset_name="oxford",
    n_runs=20,
    pop_size=30,
    max_evaluations=3000,
    base_seed=2026,
    phi_values=None,
    output_suffix="phi_sensitivity_results",
    cv_folds=5,
    n_workers=None,
    resume=False,
    classifier_encoding="multi_hot",
):
    """Shared engine used by this module's own CLI. Mirrors
    `run_fs_experiment.main`'s incremental-write-per-dataset/`resume`
    pattern: each per-dataset CSV is flushed one row at a time so a crash
    partway through a long sweep still leaves every completed run on disk,
    and `resume=True` skips (phi, dataset, run) triples already present in
    an existing output CSV instead of re-running them."""
    phi_values = phi_values if phi_values is not None else PHI_VALUES_DEFAULT
    dataset_names = [dataset_name] if isinstance(dataset_name, str) else list(dataset_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_paths = {ds: RESULTS_DIR / f"{ds}_{output_suffix}.csv" for ds in dataset_names}

    completed = set()
    if resume:
        for path in out_paths.values():
            completed |= _load_completed(path)

    files = {}
    for ds, path in out_paths.items():
        append = resume and path.exists() and path.stat().st_size > 0
        files[ds] = open(path, "a" if append else "w", newline="", encoding="utf-8")
    writers = {}
    try:
        for ds, f in files.items():
            writers[ds] = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
            if f.tell() == 0:  # fresh file (or resuming from an empty/missing one) -- write header once
                writers[ds].writeheader()
                f.flush()

        def on_result(record):
            writers[record["dataset"]].writerow(record)
            files[record["dataset"]].flush()

        records = run_phi_values(
            phi_values, dataset_names, n_runs, pop_size, max_evaluations, base_seed, cv_folds, n_workers,
            on_result=on_result,
            skip=completed,
            classifier_encoding=classifier_encoding,
        )
    finally:
        for f in files.values():
            f.close()

    df = pd.DataFrame(records)
    for ds in dataset_names:
        n_resumed = sum(1 for phi, d, r in completed if d == ds)
        n_new = int((df["dataset"] == ds).sum()) if not df.empty else 0
        print(f"{out_paths[ds]}: {n_resumed} resumed + {n_new} new = {n_resumed + n_new} total rows")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis of CSO's social-term coefficient phi for the proposed method."
    )
    parser.add_argument("--dataset", nargs="*", default=["oxford"], choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=20)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_evaluations", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--phi_values", type=float, nargs="*", default=None)
    parser.add_argument("--output_suffix", default="phi_sensitivity_results")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already present in an existing output CSV and append to it"
    )
    parser.add_argument(
        "--classifier_encoding",
        choices=["multi_hot", "top1"],
        default="multi_hot",
        help="matches CSO_searched_tf's own classifier_encoding choice in run_comparison.py",
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.n_runs,
        args.pop_size,
        args.max_evaluations,
        args.base_seed,
        args.phi_values,
        args.output_suffix,
        args.cv_folds,
        args.n_workers,
        args.resume,
        args.classifier_encoding,
    )
