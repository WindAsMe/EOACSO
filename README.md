# CSO with a Per-Particle Searched Transfer Function for Parkinson's Disease Detection

Joint feature selection + base-classifier selection + transfer-function
selection for a heterogeneous ensemble, built on the plain Competitive Swarm
Optimizer (CSO), applied to two Parkinson's disease voice/speech datasets.

## Algorithm design

**Encoding.** A particle's continuous position has `D + 5` dimensions,
where `D` is the number of features in the dataset:

```
[ f1 f2 ... fD ]   [ SVM  RF  XGBoost  kNN  LogReg ]
  shared feature       5 classifier on/off switches
     subset
```

The first `D` dimensions binarize (via a transfer function) into a *shared*
feature mask. The last 5 dimensions binarize into independent switches
choosing which of the 5 candidate classifiers join the final ensemble; all
active classifiers train on the same selected feature subset and are
combined by equal-weight soft voting (`src/optimizers/base.py::decode_bits`).
This is now the default for **every** algorithm in the project -- the
proposed method and all 7 reproduced baselines alike -- via each runner's
`classifier_encoding="multi_hot"` default parameter. Pass
`classifier_encoding="top1"` to any runner (or `--classifier_encoding top1`
on the CLI entry points) to instead decode top-1 (argmax of the raw
continuous scores, exactly one classifier active,
`src/optimizers/base.py::decode_bits_top1`) -- e.g. to reproduce each
baseline's original single-classifier scheme. **This default has changed
direction multiple times across the project's history; re-check the
actual runner signatures in `src/optimizers/` before trusting this
paragraph.**

This lets a single optimizer answer "which features" and "which model(s)"
together, without ever optimizing ensemble weights or classifier
hyperparameters directly.

For the proposed method only (not the 7 baselines, which each keep their
own fixed transfer function), the encoding grows to `D + 10`: 5 further
dimensions, one per candidate transfer function
(`src/optimizers/transfer.py::TF_CANDIDATES`), argmaxed per particle to
pick which of the 5 binarizes that particle's own `D + 5` segment --
so different particles, and the same particle across generations, may use
different transfer functions (`src/fitness.py::FitnessEvaluator.evaluate_searched_tf`).

**Fitness** (`src/fitness.py::FitnessEvaluator`): 5-fold `StratifiedGroupKFold`
cross-validation (grouped by subject to prevent leakage across repeated
recordings), out-of-fold soft-voting balanced accuracy, combined with a
feature-parsimony penalty:

```
fitness = 0.9 * (1 - balanced_accuracy) + 0.1 * (n_selected / D)
```

Lower is better (this project treats optimization as minimization
throughout). If a particle decodes to an empty feature mask or an empty
classifier mask, one bit is force-activated at random (`decode_bits`).

**The proposed method** (`src/optimizers/cso.py::run_cso`) is the original
CSO (Cheng & Jin, 2015 -- pairwise random competition each generation,
winners pass through unchanged, losers update toward the winner),
completely unmodified -- no added search strategies. Its exact social-term
control parameter (`cso_phi`, Eq. 25-26 of Cheng & Jin 2015) is used as-is
(this is **exactly 0** for swarm sizes <=100, so there is no mean-position
term at all at our typical population sizes). The only departure from a
textbook re-implementation is the encoding it searches over (see above):
each loser's update also touches its transfer-function-selector segment,
so which of the 5 candidate transfer functions binarizes a given particle
can change from one generation to the next, rather than being fixed for
the whole run.

## Datasets (`src/data_loader.py`)

| name | samples | features | notes |
|---|---|---|---|
| `oxford` | 195 (32 subjects) | 22 | Little et al. 2008; `groups` = subject id parsed from the `name` field, required for `StratifiedGroupKFold` since each subject has 6-7 repeated recordings |
| `sakar` | 252 (188 PD / 64 control) | 753 | Sakar et al. 2018, per-subject-aggregated mirror (one row per subject, via Rdatasets) -- the official UCI package ships as a `.rar` this machine can't extract; since it's already one row per subject, no grouping is needed |

Run `python -m src.data_loader` to re-fetch/verify both.

## Reproduced baseline optimizers

Per-paper equations were extracted from the PDFs in `Papers/Compare_Papers/`
and reproduced as optimizer *cores* plugged into this project's own
encoding and `FitnessEvaluator` (not each paper's own fixed classifier /
original dataset split) -- so every algorithm below is compared to the
proposed method under identical conditions. Where a paper doesn't state its own update
equations (most application papers just cite the base algorithm) the
canonical published equations are used instead; every such gap-fill is
documented in the corresponding source file's docstring.

| algorithm | file | source paper | notable gap-fills |
|---|---|---|---|
| BPSO | `bpso.py` | Hashemi et al. 2026 | PSO update eq. not in paper -> textbook PSO |
| BBOA | `bboa.py` | Hashemi et al. 2026 | BOA fragrance/movement eq. not in paper -> canonical BOA (Arora & Singh 2019) |
| BWOA | `bwoa.py` | Hashemi et al. 2026 | WOA eq. not in paper -> canonical WOA (Mirjalili & Lewis 2016) |
| BGWO / HybridGWO | `gwo.py` | Al-Najjar et al. 2024 | "Hybrid GWO" (Al-Tashi et al. 2019) not available -> canonical GWO used as the refine stage; no transfer fn stated -> default stochastic S-shaped |
| MGWO-eP | `mgwo_ep.py` | Santhosh et al. 2025 | fully specified (Eq. 12-14); no transfer fn stated -> threshold at 0.5, positions kept in [0,1] as the paper does |
| mHGS | `mhgs.py` | Hashim et al. 2023 | fully specified (Eq. 19-24); population size / iteration count not stated -> project defaults |
| QMFO | `qmfo.py` | Mansour 2024 | Mayfly eq. fully specified; the "quantum rotation gate" has no formula linking it to position updates -> approximated as a QEA-style rotation nudge toward the global best's bits |

All 8 (BPSO/BBOA/BWOA/BGWO/HybridGWO/MGWO-eP/mHGS/QMFO), plus the 2 CSO
variants (`CSO_searched_tf`, `CSO_fixed_tf`), are registered in
`src/experiments/run_fs_experiment.py::ALGORITHMS`.

## Running experiments

```bash
python -m src.experiments.run_fs_experiment --dataset oxford --n_runs 10 --pop_size 20 --n_generations 30
```

Runs every algorithm in `ALGORITHMS` for `n_runs` independent seeds, saves
`results/tables/<dataset>_comparison_results.csv`. Pass `--algorithms X Y`
to restrict to a subset, `--dataset sakar` for the high-dimensional set.

**Compute cost warning**: this is a wrapper approach -- every fitness
evaluation trains up to 5 classifiers across 5 CV folds. At the settings
above, expect roughly 0.5s/eval on `oxford` and 2.5s/eval on `sakar`
(SVM+CalibratedClassifierCV dominates). A single run costs
`pop_size + n_generations * pop_size/2` evaluations; scale `n_runs` (30 for
publication-grade statistics) and `n_generations`/`pop_size` accordingly --
30 runs x 13 algorithms at paper-scale settings takes hours, particularly on
`sakar`.

`src/experiments/stats.py` provides `wilcoxon_signed_rank`,
`friedman_test`, and `pairwise_wilcoxon` for comparing the resulting
per-run score columns across algorithms.

## Project layout

```
src/
  data_loader.py              Oxford + Sakar dataset loaders
  fitness.py                  FitnessEvaluator (shared by every optimizer)
  optimizers/
    base.py                   encoding, transfer fn, decode/repair logic
    transfer.py                S-shaped / V-shaped / hybrid / threshold binarization,
                                 + TF_CANDIDATES/decode_and_binarize_searched_tf
    cso.py                      plain CSO (CSO_searched_tf / CSO_fixed_tf via fixed_tf_index)
    bpso.py bboa.py bwoa.py     Hashemi et al. 2026 baselines
    gwo.py                      BGWO + WOA->HybridGWO cascade (Al-Najjar et al. 2024)
    mgwo_ep.py                  Santhosh et al. 2025
    mhgs.py                     Hashim et al. 2023
    qmfo.py                     Mansour 2024
    result.py                   OptimizationResult dataclass
  experiments/
    run_fs_experiment.py       runs every algorithm, saves CSV
    stats.py                    Wilcoxon / Friedman significance tests
data/raw/                      downloaded dataset files
results/tables/, results/figures/
Papers/Review_Papers/           all candidate baseline papers (PDF)
Papers/Compare_Papers/          the 8 papers selected as reproducible baselines
```

## Known gaps / things to revisit

- The `sakar` dataset is the 252-row per-subject-aggregated mirror, not the
  official UCI 756-row (3-recordings-per-subject) package, because that
  ships as a `.rar` this machine has no extractor for.
- `HybridGWO`'s second stage uses canonical GWO as a stand-in for Al-Tashi
  et al.'s "Hybrid GWO" (that paper wasn't available for extraction).
- `QMFO`'s quantum-rotation mechanism is this project's own reasonable
  interpretation of a qualitatively-described, formula-free paragraph in
  Mansour (2024) -- see the docstring in `qmfo.py`.
- No experiment has yet been run at publication-scale settings (30
  independent runs, `pop_size`/`n_generations` large enough for the
  algorithms to meaningfully differ) -- all results so far are small-scale
  smoke tests to validate the pipeline.
