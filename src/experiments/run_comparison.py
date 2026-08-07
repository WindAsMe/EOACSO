"""Entry point: CSO_searched_tf (the proposed method) vs. the reproduced
literature baselines (mHGS, BGWO, HybridGWO, QMFO, MGWO-eP, BPSO, BBOA).

Defaults to `classifier_encoding="multi_hot"` for every algorithm --
CSO_searched_tf and all 7 baselines alike: several classifiers can be
switched on and combined by equal-weight soft voting (see README.md's
Encoding section). Pass `--classifier_encoding top1` to instead decode
top-1 (a single classifier via argmax) across the board, e.g. to
reproduce each baseline's original single-classifier scheme.

Saves to results/tables/<dataset>_comparison_results.csv.
"""

import argparse

from src.experiments.run_fs_experiment import ALGORITHMS, main

PROPOSED_METHOD = "CSO_searched_tf"
BASELINE_ALGORITHMS = [name for name in ALGORITHMS if name != PROPOSED_METHOD]
COMPARISON_ALGORITHMS = [PROPOSED_METHOD] + BASELINE_ALGORITHMS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="*", default=["oxford"], choices=["oxford", "naranjo"])
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--pop_size", type=int, default=30)
    parser.add_argument("--max_evaluations", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--skip_cso", action="store_true", help="run only the baselines, not CSO_searched_tf")
    parser.add_argument(
        "--resume", action="store_true", help="skip runs already present in an existing output CSV and append to it"
    )
    parser.add_argument(
        "--classifier_encoding",
        choices=["multi_hot", "top1"],
        default="multi_hot",
        help="classifier encoding applied to every algorithm (CSO_searched_tf and all baselines)",
    )
    args = parser.parse_args()
    algos = BASELINE_ALGORITHMS if args.skip_cso else [PROPOSED_METHOD] + BASELINE_ALGORITHMS
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
