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
    the GWO stage (each stage gets its own half of the budget). The WOA
    stage's update equations (encircling/bubble-net/search-for-prey, the
    canonical Mirjalili & Lewis 2016 formulas) are inlined below rather than
    calling out to a separately-registered WOA baseline, since Al-Najjar et
    al.'s own cascade design (Fig. 1a) treats WOA->GWO as a single algorithm,
    not a composition of two independently-reproduced ones."""
    D = dimension(n_features)
    stage_budget = None if max_evaluations is None else max_evaluations / 2.0

    # --- Stage 1: WOA (encircling / bubble-net / search-for-prey) ---
    woa_rng = np.random.default_rng(seed)
    woa_b = 1.0  # spiral shape constant

    woa_positions = woa_rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    woa_bits = np.zeros((pop_size, D), dtype=bool)
    woa_fitness = np.empty(pop_size)
    woa_infos = [None] * pop_size

    for i in range(pop_size):
        woa_bits[i], woa_fitness[i], woa_infos[i] = binarize_and_eval(
            woa_positions[i], woa_bits[i], woa_rng, evaluator, n_features,
            mode="stochastic", classifier_encoding=classifier_encoding,
        )

    woa_gbest_idx = int(np.argmin(woa_fitness))
    woa_gbest_pos = woa_positions[woa_gbest_idx].copy()
    woa_gbest_fit = float(woa_fitness[woa_gbest_idx])
    woa_gbest_info = woa_infos[woa_gbest_idx]
    woa_history = [(evaluator.n_evaluations, woa_gbest_fit)]

    for gen, t_frac in generation_schedule(n_gen_woa, stage_budget, evaluator):
        a = 2.0 - 2.0 * t_frac
        for i in range(pop_size):
            r1, r2 = woa_rng.random(D), woa_rng.random(D)
            A = 2 * a * r1 - a
            C = 2 * r2

            if woa_rng.random() < 0.5:
                encircle_mask = np.abs(A) < 1
                d_best = np.abs(C * woa_gbest_pos - woa_positions[i])
                candidate_encircle = woa_gbest_pos - A * d_best

                j = woa_rng.integers(pop_size)
                d_rand = np.abs(C * woa_positions[j] - woa_positions[i])
                candidate_search = woa_positions[j] - A * d_rand

                new_pos = np.where(encircle_mask, candidate_encircle, candidate_search)
            else:
                l = woa_rng.uniform(-1, 1, size=D)
                d_best = np.abs(woa_gbest_pos - woa_positions[i])
                new_pos = d_best * np.exp(woa_b * l) * np.cos(2 * np.pi * l) + woa_gbest_pos

            woa_positions[i] = clamp(new_pos)
            woa_bits[i], woa_fitness[i], woa_infos[i] = binarize_and_eval(
                woa_positions[i], woa_bits[i], woa_rng, evaluator, n_features,
                mode="stochastic", classifier_encoding=classifier_encoding,
            )

        gen_best = int(np.argmin(woa_fitness))
        if woa_fitness[gen_best] < woa_gbest_fit - 1e-9:
            woa_gbest_fit = float(woa_fitness[gen_best])
            woa_gbest_pos = woa_positions[gen_best].copy()
            woa_gbest_info = woa_infos[gen_best]
        woa_history.append((evaluator.n_evaluations, woa_gbest_fit))

    # --- Stage 2: GWO, seeded from the WOA stage's best solution ---
    rng = np.random.default_rng(seed + 1)

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    positions[0] = woa_gbest_pos.copy()
    bits = np.zeros((pop_size, D), dtype=bool)
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    for i in range(pop_size):
        bits[i], fitness[i], infos[i] = binarize_and_eval(
            positions[i], bits[i], rng, evaluator, n_features, mode="stochastic",
            classifier_encoding=classifier_encoding,
        )

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    if woa_gbest_fit < gbest_fit:
        gbest_fit, gbest_pos, gbest_info = woa_gbest_fit, woa_gbest_pos.copy(), woa_gbest_info
    history = list(woa_history) + [(evaluator.n_evaluations, gbest_fit)]

    # `evaluator.n_evaluations` is cumulative across both stages, so stage 2's
    # target is the FULL budget (not stage_budget again), letting it consume
    # whatever stage 1 left over.
    for gen, t_frac in generation_schedule(n_gen_gwo, max_evaluations, evaluator):
        a = 2.0 - 2.0 * t_frac
        positions = clamp(_gwo_generation(positions, fitness, a, rng, D))
        for i in range(pop_size):
            bits[i], fitness[i], infos[i] = binarize_and_eval(
                positions[i], bits[i], rng, evaluator, n_features, mode="stochastic",
                classifier_encoding=classifier_encoding,
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
    )
