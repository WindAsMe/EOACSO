"""Entry point: EOACSO_full vs. the reproduced literature baselines
(BPSO, BBOA, BGWO, HybridGWO, MGWO-eP, mHGS, QMFO).

Defaults to `classifier_encoding="multi_hot"` for every algorithm --
EOACSO_full and all 7 baselines alike: several classifiers can be
switched on and combined by equal-weight soft voting (see README.md's
Encoding section). Pass `--classifier_encoding top1` to instead decode
top-1 (a single classifier via argmax) across the board, e.g. to
reproduce each baseline's original single-classifier scheme.

BWOA is reproduced in `src/optimizers/bwoa.py` and still registered in
`ALGORITHMS` (still runnable via `--algorithms BWOA` on the low-level
`run_fs_experiment.py` entry point, or `python -m src.experiments.run_comparison
--include_bwoa`), but is excluded from the default comparison set here: its
results were consistently the worst/most anomalous of the 8 baselines (see
oxford/naranjo comparison CSVs), so the user chose not to treat it as a
comparison baseline going forward. Not a code-fidelity issue -- see
`bwoa.py`'s own docstring for that.

Saves to results/tables/<dataset>_comparison_results.csv, separate from the
ablation study (see `run_ablation.py`).
"""

import argparse

from src.experiments.run_fs_experiment import ALGORITHMS, EOACSO_VARIANTS, main

EXCLUDED_BY_DEFAULT = {"BWOA"}
BASELINE_ALGORITHMS = [
    name for name in ALGORITHMS if name not in EOACSO_VARIANTS and name not in EXCLUDED_BY_DEFAULT
]
COMPARISON_ALGORITHMS = ["EOACSO_full"] + BASELINE_ALGORITHMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=["oxford"], choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_evaluations", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--skip_eoacso", action="store_true", help="run only the baselines, not EOACSO_full")
    parser.add_argument(
        "--include_bwoa", action="store_true", help="also run BWOA, excluded from the default comparison set"
    )
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already present in an existing output CSV and append to it"
    )
    parser.add_argument(
        "--classifier_encoding",
        choices=["multi_hot", "top1"],
        default="multi_hot",
        help="classifier encoding applied to every algorithm (EOACSO_full and all baselines)",
    )
    args = parser.parse_args()
    baseline_algos = BASELINE_ALGORITHMS + (["BWOA"] if args.include_bwoa else [])
    algos = baseline_algos if args.skip_eoacso else ["EOACSO_full"] + baseline_algos
    main(
        dataset_name=args.dataset,
        n_runs=args.n_runs,
        pop_size=args.pop_size,
        max_evaluations=args.max_evaluations,
        base_seed=args.base_seed,
        algorithms=algos,
        output_suffix="comparison_results",
        cv_folds=args.cv_folds,
        n_workers=args.n_workers,
        resume=args.resume,
        classifier_encoding=args.classifier_encoding,
    )
