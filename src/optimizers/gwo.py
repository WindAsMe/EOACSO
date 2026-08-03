"""Grey Wolf Optimizer, and the WOA -> Hybrid-GWO cascade of Al-Najjar et al.
(2024), "Hybrid grey wolf and whale optimization for enhanced Parkinson's
prediction based on machine learning models using biomedical sound".

That paper cascades WOA (30 agents/500 iter) into a "Hybrid GWO" it cites
from Al-Tashi et al. (2019) rather than deriving itself, and states neither
a transfer function nor a fitness formula for the search loop. We therefore:
  - seed the GWO population from WOA's best solution (Sec. 3.3/Fig. 1a) --
    only 1 of `pop_size` slots is seeded this way (the rest random), which
    is a weaker transfer of information than the paper's own wording
    ("regenerate 30 solutions based on the output of WOA") suggests; the
    paper's Fig. 1a also depicts an outer 30x repeat of the whole
    WOA->GWO cascade that this single-shot version does not implement,
  - use the canonical GWO alpha/beta/delta averaging update (Mirjalili et
    al., 2014) as a faithful stand-in for the cited "Hybrid GWO" step, since
    Al-Tashi et al.'s exact hybridization isn't available to us,
  - use the project's default stochastic S-shaped transfer function, since
    the source paper is silent on binarization,
  - default to far fewer generations (15+15) than the paper's 30-agent/
    500-iteration figure, since each generation here costs a real k-fold
    ML pipeline evaluation rather than a cheap benchmark-function call --
    both are configurable via `n_gen_woa`/`n_gen_gwo` if the paper's exact
    budget is needed.
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .bwoa import run_bwoa
from .result import OptimizationResult
from .transfer import binarize_and_eval


def _gwo_generation(positions, fitness, a, rng, D):
    order = np.argsort(fitness)
    leaders = [positions[order[0]], positions[order[1]], positions[order[2]]]
    new_positions = np.empty_like(positions)
    for i in range(len(positions)):
        candidates = []
        for leader in leaders:
            r1, r2 = rng.random(D), rng.random(D)
            A = 2 * a * r1 - a
            C = 2 * r2
            d = np.abs(C * leader - positions[i])
            candidates.append(leader - A * d)
        new_positions[i] = np.mean(candidates, axis=0)
    return new_positions


def run_bgwo(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    binarize_mode="stochastic",
    max_evaluations=None,
    classifier_encoding="multi_hot",
):
    rng = np.random.default_rng(seed)
    D = dimension(n_features)

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    bits = np.zeros((pop_size, D), dtype=bool)
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    for i in range(pop_size):
        bits[i], fitness[i], infos[i] = binarize_and_eval(
            positions[i], bits[i], rng, evaluator, n_features, mode=binarize_mode, classifier_encoding=classifier_encoding
        )

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    history = [(evaluator.n_evaluations, gbest_fit)]

    for gen, t_frac in generation_schedule(n_generations, max_evaluations, evaluator):
        a = 2.0 - 2.0 * t_frac
        positions = clamp(_gwo_generation(positions, fitness, a, rng, D))
        for i in range(pop_size):
            bits[i], fitness[i], infos[i] = binarize_and_eval(
                positions[i], bits[i], rng, evaluator, n_features, mode=binarize_mode, classifier_encoding=classifier_encoding
            )

        gen_best = int(np.argmin(fitness))
        if fitness[gen_best] < gbest_fit - 1e-9:
            gbest_fit = float(fitness[gen_best])
            gbest_pos = positions[gen_best].copy()
            gbest_info = infos[gen_best]
        history.append((evaluator.n_evaluations, gbest_fit))

    return OptimizationResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        best_info=gbest_info,
        history=history,
        n_evaluations=evaluator.n_evaluations,
        archive=[],
    )


def run_hybrid_gwo(
    evaluator,
    n_features,
    pop_size=30,
    n_gen_woa=15,
    n_gen_gwo=15,
    seed=0,
    max_evaluations=None,
    classifier_encoding="multi_hot",
):
    """Sec. 3.3 of Al-Najjar et al.: WOA's best solution seeds the GWO population.
    `max_evaluations`, if given, is split evenly between the WOA stage and
    the GWO stage (each stage gets its own half of the budget)."""
    stage_budget = None if max_evaluations is None else max_evaluations / 2.0
    woa_result = run_bwoa(
        evaluator,
        n_features,
        pop_size=pop_size,
        n_generations=n_gen_woa,
        seed=seed,
        binarize_mode="stochastic",
        max_evaluations=stage_budget,
        classifier_encoding=classifier_encoding,
    )

    rng = np.random.default_rng(seed + 1)
    D = dimension(n_features)

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    positions[0] = woa_result.best_position.copy()
    bits = np.zeros((pop_size, D), dtype=bool)
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    for i in range(pop_size):
        bits[i], fitness[i], infos[i] = binarize_and_eval(
            positions[i], bits[i], rng, evaluator, n_features, mode="stochastic"
        )

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    if woa_result.best_fitness < gbest_fit:
        gbest_fit, gbest_pos, gbest_info = woa_result.best_fitness, woa_result.best_position.copy(), woa_result.best_info
    history = list(woa_result.history) + [(evaluator.n_evaluations, gbest_fit)]

    # `evaluator.n_evaluations` is cumulative across both stages, so stage 2's
    # target is the FULL budget (not stage_budget again), letting it consume
    # whatever stage 1 left over.
    for gen, t_frac in generation_schedule(n_gen_gwo, max_evaluations, evaluator):
        a = 2.0 - 2.0 * t_frac
        positions = clamp(_gwo_generation(positions, fitness, a, rng, D))
        for i in range(pop_size):
            bits[i], fitness[i], infos[i] = binarize_and_eval(
                positions[i], bits[i], rng, evaluator, n_features, mode="stochastic"
            )

        gen_best = int(np.argmin(fitness))
        if fitness[gen_best] < gbest_fit - 1e-9:
            gbest_fit = float(fitness[gen_best])
            gbest_pos = positions[gen_best].copy()
            gbest_info = infos[gen_best]
        history.append((evaluator.n_evaluations, gbest_fit))

    return OptimizationResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        best_info=gbest_info,
        history=history,
        n_evaluations=evaluator.n_evaluations,
        archive=[],
    )
