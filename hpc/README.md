# Running on an HPC cluster (SLURM)

## 1. Get the project onto the cluster

Either:
```bash
# from your local machine, if the cluster is reachable directly
rsync -avz --exclude 'results/tables/*.csv' --exclude '__pycache__' \
    "C:/Users/zhong/Desktop/Parkinson/" you@cluster:/path/to/EOACSO/
```
or initialize a git repo locally, push to a remote (GitHub/GitLab/institutional),
and `git clone` on the cluster -- recommended if you'll keep iterating on the
code, since it also gives you version history/rollback that this project
doesn't currently have locally.

## 2. Environment

```bash
module load anaconda3   # or miniconda3 -- whatever your cluster provides
conda env create -f hpc/environment.yml
conda activate eoacso
```
`environment.yml` pins Python 3.11 rather than this machine's 3.14 (too new
to assume broad availability across HPC conda channels); nothing in this
codebase requires anything past 3.11.

If conda isn't available, a `python -m venv .venv` + `pip install -r
requirements.txt` (plus `pip install xgboost`) works the same way -- just
swap the activation lines in the `.slurm` scripts.

## 3. Datasets

`data/raw/parkinsons.data` and `data/raw/ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
must come across with the rest of the project (small CSV/data files, `rsync`/`git`
both carry them fine). Verify with:
```bash
python -m src.data_loader
```

## 4. Submit jobs

```bash
sbatch hpc/run_comparison.slurm
sbatch hpc/run_ablation.slurm
```

Edit `--partition=CHANGE_ME` to your cluster's actual partition/queue name
first, and adjust `--cpus-per-task`/`--mem`/`--time` to what's actually
available -- 32 cores / 64G / the given walltimes are starting guesses, not
measured on your cluster's hardware.

**Why one multi-core job, not a SLURM array:** every (algorithm, dataset,
run) triple is already parallelized internally via `ProcessPoolExecutor`
(`run_algorithms()` in `run_fs_experiment.py`), so `--n_workers` matching
`--cpus-per-task` is enough to use the whole allocation -- no code changes
needed. `n_workers=1` was this project's *local-machine* default (that
machine had unrelated memory-pressure crashes under parallelism); on a
dedicated HPC node this restriction doesn't apply.

Both scripts pass `--resume`, so a killed/timed-out job (e.g. hitting
`--time`) can just be `sbatch`'d again and will skip already-completed
`(algorithm, dataset, run)` rows instead of restarting.

## 5. Retrieve results

```bash
rsync -avz you@cluster:/path/to/EOACSO/results/tables/ "C:/Users/zhong/Desktop/Parkinson/results/tables/"
```
