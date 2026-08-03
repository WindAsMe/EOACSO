#!/bin/bash
#SBATCH --job-name=eoacso-ablation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=CHANGE_ME

# 5 EOACSO ablation variants, n_runs=20, both datasets. Same
# one-multi-core-job shape as run_comparison.slurm -- see that script's
# comments for why (embarrassingly parallel (algorithm, dataset, run)
# triples, already pooled internally by run_algorithms()).

set -euo pipefail
mkdir -p logs results/tables

# module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
# source activate eoacso

cd "$SLURM_SUBMIT_DIR"

python -m src.experiments.run_ablation \
    --dataset oxford naranjo \
    --n_runs 20 \
    --pop_size 30 \
    --max_evaluations 3000 \
    --base_seed 2026 \
    --cv_folds 5 \
    --n_workers "$SLURM_CPUS_PER_TASK" \
    --resume
