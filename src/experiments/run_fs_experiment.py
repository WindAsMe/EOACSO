"""Run every optimizer available in this project -- the proposed method
(plain CSO with a per-particle searched transfer function) and the 8
reproduced literature baselines -- for `n_runs` independent seeds on a chosen
dataset, saving per-run results to CSV.

Runs are parallelized across (algorithm, run) pairs using a process pool,
since each is fully independent -- see `_run_single` / `run_algorithm`.

Every algorithm is compared under an equal fitness-evaluation budget
(`max_evaluations`), not equal generation count -- generation count isn't a
fair unit since algorithms spend very different numbers of evaluations per
generation (CSO only re-evaluates losers each generation; BPSO/BBOA/
BWOA's hybrid S/V transfer function evaluates two binarization candidates
per individual; mHGS's production/escaping operators each conditionally
add extra evaluations). See `src/optimizers/base.py::generation_schedule`
for how `max_evaluations` becomes the actual stopping condition while a
per-algorithm *nominal* generation count still normalizes each algorithm's
internal time-dependent schedules (the GWO family's a(t), MGWO-eP's e(t),
mHGS's Shrink(t)/theta(t)).

The proposed method, CSO_searched_tf: each particle argmaxes its own 5-dim
transfer-function-selector segment to pick which of transfer.TF_CANDIDATES
binarizes it, rather than every particle sharing one transfer function fixed
in advance (see `src/optimizers/cso.py`).

Reproduced baselines (see src/optimizers/*.py docstrings for exact sources
and which equations are paper-faithful vs. a documented gap-fill):
  BPSO, BBOA, BWOA   - Hashemi et al. (2026)
  BGWO, HybridGWO    - Al-Najjar et al. (2024) WOA->Hybrid-GWO cascade
  MGWO-eP            - Santhosh et al. (2025)
  mHGS               - Hashim et al. (2023)
  QMFO               - Mansour (2024)
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
from src.fitness import FitnessEvaluator
from src.optimizers import (
    run_bboa,
    run_bgwo,
    run_bpso,
    run_bwoa,
    run_cso,
    run_hybrid_gwo,
    run_mgwo_ep,
    run_mhgs,
    run_qmfo,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "tables"

RECORD_FIELDS = [
    "algorithm",
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

# Expected fitness evaluations spent per generation, used only to derive a
# *nominal* generation count from (pop_size, max_evaluations) so each
# algorithm's internal t/tmax schedules are normalized sensibly -- the real
# stopping condition is always max_evaluations (generation_schedule keeps
# going past this nominal count if the budget isn't spent yet, and stops
# early if it is, e.g. mHGS's operators fire conditionally so its per-gen
# cost is only an expectation, not exact).
_EVALS_PER_GENERATION = {
    "BPSO": lambda pop: 2 * pop,  # hybrid S/V transfer evaluates both candidates
    "BBOA": lambda pop: 2 * pop,
    "BWOA": lambda pop: 2 * pop,
    "BGWO": lambda pop: pop,
    "HybridGWO": lambda pop: pop,  # per stage; two stages share the total budget
    "MGWO-eP": lambda pop: pop,
    "mHGS": lambda pop: 2.6 * pop,  # mPO (~60%) + mLEO (100%) + coop. comm. (100%)
    "QMFO": lambda pop: 2 * pop + 2,
}


def _nominal_generations(algo_name, pop_size, max_evaluations):
    if algo_name == "CSO_searched_tf":
        per_gen = pop_size / 2.0
    else:
        per_gen = _EVALS_PER_GENERATION[algo_name](pop_size)
    return max(1, round((max_evaluations - pop_size) / per_gen))


def _run_hybrid_gwo_adapter(evaluator, n_features, pop_size, n_generations, seed, max_evaluations, classifier_encoding):
    half = max(n_generations // 2, 1)
    return run_hybrid_gwo(
        evaluator,
        n_features,
        pop_size=pop_size,
        n_gen_woa=half,
        n_gen_gwo=n_generations - half,
        seed=seed,
        max_evaluations=max_evaluations,
        classifier_encoding=classifier_encoding,
    )


ALGORITHMS = {
    "CSO_searched_tf": lambda ev, nf, p, g, s, me, ce: run_cso(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "BPSO": lambda ev, nf, p, g, s, me, ce: run_bpso(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "BBOA": lambda ev, nf, p, g, s, me, ce: run_bboa(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "BWOA": lambda ev, nf, p, g, s, me, ce: run_bwoa(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "BGWO": lambda ev, nf, p, g, s, me, ce: run_bgwo(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "HybridGWO": _run_hybrid_gwo_adapter,
    "MGWO-eP": lambda ev, nf, p, g, s, me, ce: run_mgwo_ep(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "mHGS": lambda ev, nf, p, g, s, me, ce: run_mhgs(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
    "QMFO": lambda ev, nf, p, g, s, me, ce: run_qmfo(
        ev, nf, pop_size=p, n_generations=g, seed=s, max_evaluations=me, classifier_encoding=ce
    ),
}


def _limit_threads_per_worker():
    """ProcessPoolExecutor initializer: each worker gets its own process, so
    without this, every worker's numpy/BLAS calls would ALSO spawn internal
    threads -- oversubscribing the machine's cores (N worker processes x M
    BLAS threads each) instead of cleanly using one core per worker."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = "1"


def _run_single(
    algo_name, dataset_name, run_idx, base_seed, pop_size, max_evaluations, cv_folds, classifier_encoding="multi_hot"
):
    seed = base_seed + 50 * run_idx  # e.g. base_seed=2026 -> 2026, 2076, 2126, ...
    ds = load(dataset_name)
    runner = ALGORITHMS[algo_name]
    n_generations = _nominal_generations(algo_name, pop_size, max_evaluations)
    evaluator = FitnessEvaluator(ds.X, ds.y, ds.groups, cv_folds=cv_folds, seed=seed)
    t0 = time.time()
    result = runner(evaluator, ds.X.shape[1], pop_size, n_generations, seed, max_evaluations, classifier_encoding)
    elapsed = time.time() - t0
    return {
        "algorithm": algo_name,
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
        "history": json.dumps(result.history),  # [(n_evals, best_fitness_so_far), ...] -- convergence curve
    }


def run_algorithms(
    algo_names,
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
    """Runs every (algorithm, dataset, run) triple in `algo_names` x
    `dataset_names` x range(n_runs) in parallel across a SINGLE process
    pool (each triple is fully independent) -- `dataset_names` may be a
    single dataset name or a list, so multiple datasets share the same
    worker budget instead of each spawning their own pool and
    oversubscribing the machine.

    `on_result(record)`, if given, is called as soon as each individual run
    finishes (before the next one even starts) -- `main` uses this to
    checkpoint every completed run to disk immediately, since long runs on
    this project have previously died mid-run to process-pool/OOM crashes,
    and nothing was persisted before that point without this hook.

    `skip`, if given, is a set of (algorithm, dataset, run_idx) triples to
    NOT resubmit -- `main`'s `resume=True` path populates this from rows
    already present in a prior, interrupted run's output CSV.

    `classifier_encoding` only affects the proposed method, CSO_searched_tf
    (the 8 baselines always decode top-1 regardless, see `optimizers/base.py`):
    `"multi_hot"` (default) is this project's multi-classifier soft-voting
    ensemble, `"top1"` matches the baselines' single-classifier scheme."""
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    n_workers = n_workers or os.cpu_count()
    skip = skip or set()
    all_tasks = [(algo, ds, run_idx) for algo in algo_names for ds in dataset_names for run_idx in range(n_runs)]
    tasks = [t for t in all_tasks if t not in skip]
    if verbose and skip:
        print(f"resume: skipping {len(all_tasks) - len(tasks)} already-completed of {len(all_tasks)} total runs")
    records = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_limit_threads_per_worker) as pool:
        futures = {
            pool.submit(
                _run_single, algo, ds, run_idx, base_seed, pop_size, max_evaluations, cv_folds, classifier_encoding
            ): (
                algo,
                ds,
                run_idx,
            )
            for algo, ds, run_idx in tasks
        }
        for future in as_completed(futures):
            algo, ds, run_idx = futures[future]
            record = future.result()
            records.append(record)
            if on_result is not None:
                on_result(record)
            if verbose:
                print(
                    f"[{algo}|{ds}] run {run_idx + 1}/{n_runs} "
                    f"fitness={record['best_fitness']:.4f} bal_acc={record['balanced_accuracy']:.4f} "
                    f"n_features={record['n_features']} clf={record['active_classifiers']} "
                    f"n_evals={record['n_evaluations']} time={record['elapsed_sec']:.1f}s "
                    f"({len(records)}/{len(tasks)} done)"
                )
    return records


def run_algorithm(algo_name, dataset_name, n_runs, pop_size, max_evaluations, base_seed=2026, cv_folds=5, verbose=True):
    """Single-algorithm convenience wrapper around `run_algorithms` (still
    parallelizes across its `n_runs` independent seeds)."""
    return run_algorithms(
        [algo_name], dataset_name, n_runs, pop_size, max_evaluations, base_seed, cv_folds, verbose=verbose
    )


def _load_completed(path):
    """Read a possibly-partial results CSV from an interrupted run and return
    the set of (algorithm, dataset, run) triples it already contains."""
    completed = set()
    if not path.exists() or path.stat().st_size == 0:
        return completed
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            completed.add((row["algorithm"], row["dataset"], int(row["run"])))
    return completed


def main(
    dataset_name="oxford",
    n_runs=5,
    pop_size=30,
    max_evaluations=3000,
    base_seed=2026,
    algorithms=None,
    output_suffix="comparison_results",
    cv_folds=5,
    n_workers=None,
    resume=False,
    classifier_encoding="multi_hot",
):
    """Shared engine used by the `run_comparison.py` CLI entry point (and
    directly, for ad-hoc subsets, via `--algorithms`/`--output_suffix` here).
    `dataset_name` may be a single name or a list -- multiple datasets are
    run through ONE shared process pool (see `run_algorithms`) and saved to
    separate per-dataset CSVs.

    Each per-dataset CSV is written incrementally, one row per completed run,
    flushed to disk immediately -- not assembled from an in-memory list and
    written only at the very end -- so a crash partway through a long run
    still leaves every already-completed run safely on disk.

    `resume=True` reads whatever rows already exist in each output CSV (from
    a prior, interrupted invocation with the same `output_suffix`/datasets),
    skips re-running those (algorithm, dataset, run) triples, and appends
    new rows after them instead of truncating the file."""
    algorithms = algorithms or list(ALGORITHMS)
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

        records = run_algorithms(
            algorithms, dataset_names, n_runs, pop_size, max_evaluations, base_seed, cv_folds, n_workers,
            on_result=on_result,
            skip=completed,
            classifier_encoding=classifier_encoding,
        )
    finally:
        for f in files.values():
            f.close()

    df = pd.DataFrame(records)
    for ds in dataset_names:
        n_resumed = sum(1 for a, d, r in completed if d == ds)
        n_new = int((df["dataset"] == ds).sum()) if not df.empty else 0
        print(f"{out_paths[ds]}: {n_resumed} resumed + {n_new} new = {n_resumed + n_new} total rows")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Low-level engine -- prefer run_comparison.py as the entry point."
    )
    parser.add_argument("--dataset", nargs="*", default=["oxford"], choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_evaluations", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--algorithms", nargs="*", default=None, choices=list(ALGORITHMS) + [None])
    parser.add_argument("--output_suffix", default="comparison_results")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already present in an existing output CSV and append to it"
    )
    parser.add_argument(
        "--classifier_encoding",
        choices=["multi_hot", "top1"],
        default="multi_hot",
        help="proposed-method (CSO_searched_tf) classifier encoding; baselines always decode top-1 regardless",
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.n_runs,
        args.pop_size,
        args.max_evaluations,
        args.base_seed,
        args.algorithms,
        args.output_suffix,
        args.cv_folds,
        args.n_workers,
        args.resume,
        args.classifier_encoding,
    )
