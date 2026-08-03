"""Entry point: non-EC neural network baselines (PyTorch MLP and SAINT,
full feature set, no feature selection) for `n_runs` independent seeds,
parallelized across a process pool same as run_comparison.py/run_ablation.py.

Both architectures reproduced per Oseni, Obanla & Jimoh (2026), see
src/baselines_dl.py's module docstring for exactly what is paper-faithful
vs. a documented gap-fill.

Saves to results/tables/<dataset>_dl_baseline_results.csv.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from src.data_loader import load
from src.experiments.run_fs_experiment import RESULTS_DIR, _limit_threads_per_worker

ARCHITECTURES = ["MLP", "SAINT"]


def _run_single(architecture, dataset_name, run_idx, base_seed, cv_folds, epochs):
    import torch

    from src.baselines_dl import run_mlp_baseline, run_saint_baseline

    torch.set_num_threads(1)
    seed = base_seed + 50 * run_idx  # e.g. base_seed=2026 -> 2026, 2076, 2126, ...
    ds = load(dataset_name)
    t0 = time.time()
    if architecture == "MLP":
        metrics = run_mlp_baseline(ds.X, ds.y, ds.groups, cv_folds=cv_folds, seed=seed, epochs=epochs)
    elif architecture == "SAINT":
        metrics = run_saint_baseline(ds.X, ds.y, ds.groups, cv_folds=cv_folds, seed=seed, epochs=epochs)
    else:
        raise ValueError(f"unknown architecture {architecture!r}")
    elapsed = time.time() - t0
    return {
        "algorithm": architecture,
        "dataset": dataset_name,
        "run": run_idx,
        "seed": seed,
        **metrics,
        "elapsed_sec": elapsed,
    }


def main(dataset_name="oxford", n_runs=5, cv_folds=5, epochs=200, architectures=None, n_workers=None, base_seed=2026):
    architectures = architectures or ARCHITECTURES
    n_workers = n_workers or os.cpu_count()
    tasks = [(arch, run_idx) for arch in architectures for run_idx in range(n_runs)]
    records = []
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_limit_threads_per_worker) as pool:
        futures = {
            pool.submit(_run_single, arch, dataset_name, run_idx, base_seed, cv_folds, epochs): (arch, run_idx)
            for arch, run_idx in tasks
        }
        for future in as_completed(futures):
            arch, run_idx = futures[future]
            record = future.result()
            records.append(record)
            print(
                f"[{arch}|{dataset_name}] run {run_idx + 1}/{n_runs} "
                f"bal_acc={record['balanced_accuracy']:.4f} f1={record['f1']:.4f} auc={record['auc']:.4f} "
                f"time={record['elapsed_sec']:.1f}s ({len(records)}/{len(tasks)} done)"
            )

    df = pd.DataFrame(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{dataset_name}_dl_baseline_results.csv"
    df.to_csv(out_path, index=False)
    print(f"saved {len(df)} rows to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="oxford", choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--architectures", nargs="*", default=None, choices=ARCHITECTURES + [None])
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=2026)
    args = parser.parse_args()
    main(
        dataset_name=args.dataset,
        n_runs=args.n_runs,
        cv_folds=args.cv_folds,
        epochs=args.epochs,
        architectures=args.architectures,
        n_workers=args.n_workers,
        base_seed=args.base_seed,
    )
