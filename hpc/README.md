# Running on the cluster (PBS Professional)

The target cluster (server `sjms`) runs **PBS Professional** (`qstat --version`
-> `pbs_version = 2024.1.2...`), not SLURM -- use `hpc/*.pbs` +
`qsub`/`qstat`/`qdel`, not the `.slurm` files (kept only as a reference for
a *different*, SLURM-based cluster, if this project ever runs on one).

Confirmed cluster facts (re-verify if anything below stops matching; the
authoritative source is the cluster's own user guide,
`CSO_Paper/gakusai_Users_Guide_ja_1.1.pdf`, sections 5.2-5.4):
- No anaconda/miniconda/python entry in `module avail` -- only compilers,
  CUDA/nvidia toolkits, a few simulation packages, and standalone
  `pytorch`/`tensorflow-keras`/`mxnet` modules.
- Queues: `ec`/`sc`/`lc` (CPU-only compute, ascending walltime limits),
  `eg`/`sg`/`lg` (GPU), `xc` (15-minute short queue), and a personal
  `c30636g` queue (GPU-suffixed -- avoid for this CPU-only job).
- **Resource requests use `nsockets=` (CPU sockets), not `ncpus=`/`mem=`.**
  Every sample script in the user guide uses `select=<n>:nsockets=<n>[:ompthreads=<n>][:mpiprocs=<n>]`
  -- `ncpus=`/`mem=` inside `-l select=...` are simply not recognized
  attributes here and made `qsub` reject the job with the unhelpful "Job
  has not allowed option" (found by bisecting a minimal test script down
  to one `#PBS` line at a time, then confirmed against the user guide).
  1 socket = 32 cores on this hardware (`default_chunk.ncpus = 32` from
  `qstat -Qf sc`/`qstat -Qf lc`), so `select=1:nsockets=1` is what the
  `.pbs` scripts here request.
- `-W group_list=<name>` is mandatory (job submission is rejected without
  it). This user's groups are `c30636` (personal), `gaussian`, `academic`,
  `hokudai` (`groups`/`id -a`); `c30636` is what's currently set in the
  scripts.
- `-V` (inherit full login-node environment) is rejected by this site's
  policy -- the scripts source the conda environment explicitly instead
  and don't need it.

## 1. Get the project onto the cluster

Already pushed to GitHub: https://github.com/WindAsMe/EOACSO.
```bash
git clone https://github.com/WindAsMe/EOACSO.git
cd EOACSO
```
(Already done once -- from here on, `git pull` inside `EOACSO/` picks up
updates instead of re-cloning.)

## 2. Environment

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

The two `.pbs` scripts already point at `$HOME/miniconda3/bin/activate`
directly -- if your miniconda ends up somewhere else, update that line in
both scripts.

## 3. Datasets

`data/raw/parkinsons.data` and `data/raw/ReplicatedAcousticFeatures-ParkinsonDatabase.csv`
come across with `git clone`/`git pull` -- no separate transfer needed.
Verify with:
```bash
python -m src.data_loader
```
Expected:
```
oxford: X=(195, 22) y_pos=147/195 groups=32
naranjo: X=(240, 45) y_pos=120/240 groups=80
```

## 4. Submit jobs

```bash
qsub hpc/run_comparison.pbs
```

Defaults to the `sc` queue with a deliberately modest 6h walltime and
requests one full socket (`select=1:nsockets=1`, 32 cores). Estimated
actual runtime is well under 6h (160 runs, parallelized 32-way), but if
the job hits the walltime limit before finishing, just `qsub` it again --
see the `--resume` note below.

For the proposed method's phi (CSO social-term coefficient) sensitivity
sweep -- 11 fixed phi values x 20 runs x 2 datasets = 440 runs, scoped to
`CSO_searched_tf` only since phi doesn't apply to any baseline -- submit
separately:
```bash
qsub hpc/run_phi_sensitivity.pbs
```

**Why 6h, not something larger "to be safe":** this cluster's prepaid
queues (`ec`/`sc`/`lc`) pre-check token budget assuming the job runs for
its *entire requested* walltime, and reject submission outright ("Token
limit exceeded") if that worst case would push usage over 100% -- see
`gakusai_Users_Guide_ja_1.1.pdf` section 1.6.1. Run `show_token` to see
current balance/usage:
```bash
show_token
```
Raise walltime in the `.pbs` files only if 6h genuinely isn't enough and
your token balance supports it.

Check job status / cancel:
```bash
qstat -u $USER
qdel <job_id>
```

**Why one 32-core job, not a job array:** every (algorithm, dataset, run)
triple is already parallelized internally via `ProcessPoolExecutor`
(`run_algorithms()` in `run_fs_experiment.py`), so `--n_workers` matching
the job's `ncpus` is enough to use the whole allocation -- no code changes
needed. `n_workers=1` was this project's *local-machine* default (that
machine had unrelated memory-pressure crashes under parallelism); on a
dedicated cluster node this restriction doesn't apply.

Both scripts pass `--resume`, so a killed/timed-out job can just be
re-submitted with `qsub` and will skip already-completed `(algorithm,
dataset, run)` rows instead of restarting from scratch.

## 5. Retrieve results

```bash
rsync -avz c30636@grand1:~/EOACSO/results/tables/ "C:/Users/zhong/Desktop/Parkinson/results/tables/"
```
(adjust the login host/path if `grand1` is only reachable from inside the
cluster's own network, e.g. via an intermediate jump host).
