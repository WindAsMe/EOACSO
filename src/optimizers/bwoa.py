"""Binary Whale Optimization Algorithm.

Standalone use reproduces Hashemi et al. (2026) Sec. 3.4.6 / Table 3 (hybrid
S/V transfer function, `binarize_mode="hybrid_sv"`, the default here).

The encircling/bubble-net/search-for-prey equations are the canonical WOA
formulas (Mirjalili & Lewis, 2016) -- neither paper that uses WOA in our
comparison set (Hashemi 2026; Al-Najjar et al. 2024) restates them, both
just cite the original algorithm.

This function is reused as stage 1 of the WOA->Hybrid-GWO cascade
(`gwo.run_hybrid_gwo`), where `binarize_mode="stochastic"` is passed instead
since Al-Najjar et al. do not specify a transfer function at all.
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult
from .transfer import binarize_and_eval


def run_bwoa(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    b=1.0,
    binarize_mode="hybrid_sv",
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
        for i in range(pop_size):
            r1, r2 = rng.random(D), rng.random(D)
            A = 2 * a * r1 - a
            C = 2 * r2

            if rng.random() < 0.5:
                encircle_mask = np.abs(A) < 1
                d_best = np.abs(C * gbest_pos - positions[i])
                candidate_encircle = gbest_pos - A * d_best

                j = rng.integers(pop_size)
                d_rand = np.abs(C * positions[j] - positions[i])
                candidate_search = positions[j] - A * d_rand

                new_pos = np.where(encircle_mask, candidate_encircle, candidate_search)
            else:
                l = rng.uniform(-1, 1, size=D)
                d_best = np.abs(gbest_pos - positions[i])
                new_pos = d_best * np.exp(b * l) * np.cos(2 * np.pi * l) + gbest_pos

            positions[i] = clamp(new_pos)
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
