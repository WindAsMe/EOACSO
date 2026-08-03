"""Entry point: EOACSO_Paper ablation study -- the 5 variants that isolate
strategies 2 (elite-guided update), 3 (stagnation OBL), and 5 (diversity
archive), plus CSO_vanilla (all three off).

Saves to results/tables/<dataset>_ablation_results.csv, separate from the
baseline comparison (see `run_comparison.py`).

Defaults to `classifier_encoding="multi_hot"`, matching the main
EOACSO_full vs. baselines comparison (`run_comparison.py`, which now uses
this same default for every algorithm including the baselines) and this
project's reported method: several classifiers can be switched on and
combined by equal-weight soft voting. Pass `--classifier_encoding top1`
for a cheaper, single-classifier-per-particle run -- substantially faster
per fitness evaluation (1 classifier trained instead of up to 5), useful
for a quick preview of the variants' relative ordering but not the
configuration used for this project's reported ablation results.
"""

import argparse

from src.experiments.run_fs_experiment import EOACSO_VARIANTS, main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=["oxford"], choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_evaluations", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already present in an existing output CSV and append to it"
    )
    parser.add_argument(
        "--classifier_encoding",
        choices=["multi_hot", "top1"],
        default="multi_hot",
        help="this project's multi-classifier ensemble (default) or single-classifier top1 (matches the 8 baselines)",
    )
    args = parser.parse_args()
    main(
        dataset_name=args.dataset,
        n_runs=args.n_runs,
        pop_size=args.pop_size,
        max_evaluations=args.max_evaluations,
        base_seed=args.base_seed,
        algorithms=list(EOACSO_VARIANTS),
        output_suffix="ablation_results",
        cv_folds=args.cv_folds,
        n_workers=args.n_workers,
        resume=args.resume,
        classifier_encoding=args.classifier_encoding,
    )
