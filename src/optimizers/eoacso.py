"""EOACSO_Paper: Elite-guided Opposition-based Archive Competitive Swarm Optimizer.

Built on the original Competitive Swarm Optimizer (Cheng & Jin, 2015):
particles are randomly paired each generation, the fitter one (winner) is
left unchanged, the other (loser) updates towards the winner. EOACSO_Paper adds
three strategies on top of that skeleton:

  (2) Elite-guided update   - the loser's velocity update replaces CSO's
      uninformative "mean position" pull with a pull towards an elite
      sampled from the archive, weighted by an increasing lambda(t) so the
      search is exploration-heavy early and exploitation-heavy late.
  (3) Stagnation-triggered opposition-based learning (OBL) - if the global
      best has not improved for `stagnation_limit` generations, the worst
      fraction of the swarm is reinitialised to its opposite point to
      escape a local optimum.
  (5) Diversity elite archive - a bounded non-dominated archive (error
      rate vs. feature ratio) with Hamming-distance crowding, used both as
      the source of elites for (2) and, after the run, as a set of
      alternative accuracy/parsimony trade-off solutions.

Each strategy can be toggled off independently (`enable_elite_guided`,
`enable_obl`, `enable_archive`) for ablation studies; turning all three off
reduces this to a plain re-implementation of the original CSO.

`classifier_encoding` selects how the last 5 dimensions decode (see
`FitnessEvaluator.evaluate`): `"multi_hot"` (default) is this project's own
multi-classifier soft-voting ensemble; `"top1"` matches the 8 reproduced
baselines' single-classifier scheme, for a cheaper ablation run.
"""

import numpy as np

from .archive import EliteArchive
from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult


def cso_phi(pop_size):
    """Social-term control parameter, Cheng & Jin (2015) Eq. 25-26.
    For swarm sizes <=100 (our typical setting) this is exactly 0 -- the
    original CSO has NO mean-position pull at all at this scale, only the
    winner-pull and inertia terms.

    The paper doesn't state whether "log" is base-10 or natural log; its
    own Table II fit values (e.g. phi_R(1000)=0.3) only match base-10 --
    natural log overshoots the paper's own tested range of [0, 0.3] by an
    order of magnitude. The paper also only specifies phi as a *range*
    [phi_L(m), phi_R(m)] with no selection rule, so taking the midpoint
    below is this project's own interpretive choice, not the paper's."""
    if pop_size <= 100:
        return 0.0
    phi_l = 0.14 * np.log10(pop_size) - 0.30
    phi_r = 0.27 * np.log10(pop_size) - 0.51
    return (phi_l + phi_r) / 2.0


def run_eoacso(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    lambda_min=0.1,
    lambda_max=0.9,
    stagnation_limit=5,
    reinit_fraction=0.3,
    archive_size=30,
    enable_elite_guided=True,
    enable_obl=True,
    enable_archive=True,
    max_evaluations=None,
    classifier_encoding="multi_hot",
):
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even (CSO pairs particles up)")

    rng = np.random.default_rng(seed)
    D = dimension(n_features)

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    velocities = np.zeros((pop_size, D))
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    archive = EliteArchive(max_size=archive_size) if enable_archive else None

    for i in range(pop_size):
        fitness[i], infos[i] = evaluator.evaluate(positions[i], rng, classifier_encoding=classifier_encoding)
        if archive is not None:
            archive.try_insert(positions[i], infos[i])

    best_idx = int(np.argmin(fitness))
    best_position = positions[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    best_info = infos[best_idx]
    history = [(evaluator.n_evaluations, best_fitness)]
    stagnation_counter = 0

    for gen, t_frac in generation_schedule(n_generations, max_evaluations, evaluator):
        lam = lambda_min + (lambda_max - lambda_min) * t_frac
        order = rng.permutation(pop_size)

        for k in range(0, pop_size, 2):
            i, j = int(order[k]), int(order[k + 1])
            w_idx, l_idx = (i, j) if fitness[i] <= fitness[j] else (j, i)

            R1, R2, R3 = rng.random(D), rng.random(D), rng.random(D)

            if enable_elite_guided:
                elite_pos = archive.sample_elite(rng) if archive is not None else None
                if elite_pos is None:
                    elite_pos = best_position
                guide_term = lam * R3 * (elite_pos - positions[l_idx])
            else:
                mean_pos = positions.mean(axis=0)
                phi = cso_phi(pop_size)  # Cheng & Jin (2015), Eq. 25-26
                guide_term = phi * R3 * (mean_pos - positions[l_idx])

            velocities[l_idx] = (
                R1 * velocities[l_idx]
                + R2 * (positions[w_idx] - positions[l_idx])
                + guide_term
            )
            velocities[l_idx] = clamp(velocities[l_idx])
            positions[l_idx] = clamp(positions[l_idx] + velocities[l_idx])

            fitness[l_idx], infos[l_idx] = evaluator.evaluate(
                positions[l_idx], rng, classifier_encoding=classifier_encoding
            )
            if archive is not None:
                archive.try_insert(positions[l_idx], infos[l_idx])

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_fitness - 1e-9:
            best_fitness = float(fitness[gen_best_idx])
            best_position = positions[gen_best_idx].copy()
            best_info = infos[gen_best_idx]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if enable_obl and stagnation_counter >= stagnation_limit:
            worst_first = np.argsort(fitness)[::-1]
            n_reinit = max(1, int(reinit_fraction * pop_size))
            for idx in worst_first[:n_reinit]:
                idx = int(idx)
                positions[idx] = clamp(-positions[idx])  # opposite point: lb+ub-X, lb=-ub
                velocities[idx] = np.zeros(D)
                fitness[idx], infos[idx] = evaluator.evaluate(
                    positions[idx], rng, classifier_encoding=classifier_encoding
                )
                if archive is not None:
                    archive.try_insert(positions[idx], infos[idx])
            stagnation_counter = 0

            gen_best_idx = int(np.argmin(fitness))
            if fitness[gen_best_idx] < best_fitness - 1e-9:
                best_fitness = float(fitness[gen_best_idx])
                best_position = positions[gen_best_idx].copy()
                best_info = infos[gen_best_idx]

        history.append((evaluator.n_evaluations, best_fitness))

    return OptimizationResult(
        best_position=best_position,
        best_fitness=best_fitness,
        best_info=best_info,
        history=history,
        n_evaluations=evaluator.n_evaluations,
        archive=archive.entries if archive is not None else [],
    )
