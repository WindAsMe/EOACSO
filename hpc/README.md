# Running on an HPC cluster (SLURM)

## 1. Get the project onto the cluster

Already pushed to GitHub: https://github.com/WindAsMe/EOACSO. On the
cluster's login node:
```bash
git clone https://github.com/WindAsMe/EOACSO.git
cd EOACSO
```

## 2. Environment

This cluster's `module avail` has no anaconda/miniconda/python module (only
compilers, CUDA/nvidia toolkits, a few simulation packages, and standalone
`pytorch`/`tensorflow-keras`/`mxnet` modules) -- so rather than depending on
a cluster-provided Python, install a self-contained miniconda in `$HOME`
once:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

cd EOACSO
conda env create -f hpc/environment.yml
conda activate eoacso
```
`environment.yml` pins Python 3.11 rather than this machine's 3.14 (too new
to assume broad availability across conda channels); nothing in this
codebase requires anything past 3.11. If the login node has no internet
access, download the Miniconda installer on your local machine first and
`scp`/upload it across instead of `wget`-ing directly on the cluster.

The two `.slurm` scripts already point at `$HOME/miniconda3/bin/activate`
directly (no `module load` needed) -- if your miniconda ends up somewhere
else, update that line in both scripts.

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
