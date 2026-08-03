"""MGWO-eP, reproduced per Santhosh et al. (2025), "A modified Gray Wolf
Optimization algorithm for early detection of Parkinson's Disease".

Exact equations from the paper (Eq. 12-14, replacing classical GWO's Eq. 3
and 5): exponential-decay exploration parameter `e`, and a "divergence
independent coefficient" `P` that only scales with `e` on the converging
branch. The paper initializes positions directly in [0,1] (representing
per-feature importance) and is silent on any transfer function, so we
threshold at 0.5 (Eq. 26 of Hashim et al.'s mHGS uses the same convention
for the same kind of [0,1]-native representation).
"""

import numpy as np

from .base import decode_bits, decode_bits_top1, dimension, generation_schedule
from .result import OptimizationResult


def run_mgwo_ep(
    evaluator, n_features, pop_size=30, n_generations=500, seed=0, max_evaluations=None, classifier_encoding="multi_hot"
):
    rng = np.random.default_rng(seed)
    D = dimension(n_features)

    positions = rng.uniform(0.0, 1.0, size=(pop_size, D))
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    def evaluate(pos):
        bits = pos > 0.5
        if classifier_encoding == "top1":
            fmask, cmask = decode_bits_top1(bits, pos, n_features, rng)
        else:
            fmask, cmask = decode_bits(bits, n_features, rng)
        return evaluator.evaluate_masks(fmask, cmask)

    for i in range(pop_size):
        fitness[i], infos[i] = evaluate(positions[i])

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    history = [(evaluator.n_evaluations, gbest_fit)]

    for gen, t_frac in generation_schedule(n_generations, max_evaluations, evaluator):
        e = 2.0 - 2.0 * t_frac**2  # Eq. 14
        order = np.argsort(fitness)
        leaders = [positions[order[0]], positions[order[1]], positions[order[2]]]

        new_positions = np.empty_like(positions)
        for i in range(pop_size):
            candidates = []
            for leader in leaders:
                rand1 = rng.random(D)
                rand2 = rng.random(D)
                signed = 2 * rand1 - 1
                p = np.where(signed > 0, e * signed, 2 * signed)  # Eq. 13
                q = 2 * rand2  # Eq. 4
                d = np.abs(q * leader - positions[i])
                candidates.append(leader - p * d)
            new_positions[i] = np.mean(candidates, axis=0)
        positions = np.clip(new_positions, 0.0, 1.0)

        for i in range(pop_size):
            fitness[i], infos[i] = evaluate(positions[i])

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
