#!/bin/bash
#SBATCH --job-name=eoacso-comparison
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=CHANGE_ME

# EOACSO_full vs. 7 reproduced baselines, n_runs=20, both datasets.
# Every (algorithm, dataset, run) triple is independent -- run_algorithms()
# already parallelizes them across a ProcessPoolExecutor internally, so a
# single multi-core job (not a SLURM array) is the right shape here.
# --n_workers MUST match --cpus-per-task above (each worker gets pinned to
# 1 BLAS thread; see _limit_threads_per_worker in run_fs_experiment.py).

set -euo pipefail
mkdir -p logs results/tables

# --- environment ---
# Option A: conda (recommended, see hpc/environment.yml)
# module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
# source activate eoacso

# Option B: module-provided Python + a project-local venv, if conda isn't
# available on your cluster:
# module load python/3.11
# source .venv/bin/activate

cd "$SLURM_SUBMIT_DIR"

python -m src.experiments.run_comparison \
    --dataset oxford naranjo \
    --n_runs 20 \
    --pop_size 30 \
    --max_evaluations 3000 \
    --base_seed 2026 \
    --cv_folds 5 \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --resume
