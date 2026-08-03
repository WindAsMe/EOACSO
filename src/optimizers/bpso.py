"""Binary PSO, reproduced per Hashemi et al. (2026), Sec. 3.4.4 / Table 3.

The paper states hyperparameters (w=0.7, c1=2.5, c2=1.6) and the hybrid
S/V-shaped transfer function (Eq. 4-8) but not PSO's own velocity/position
update -- it only cites the method. We use the textbook PSO update equation
(Kennedy & Eberhart), since that is what any unmodified "BPSO" reduces to.
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult
from .transfer import binarize_and_eval


def run_bpso(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    w=0.7,
    c1=2.5,
    c2=1.6,
    max_evaluations=None,
    classifier_encoding="multi_hot",
):
    rng = np.random.default_rng(seed)
    D = dimension(n_features)

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    velocities = np.zeros((pop_size, D))
    bits = np.zeros((pop_size, D), dtype=bool)
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    for i in range(pop_size):
        bits[i], fitness[i], infos[i] = binarize_and_eval(
            positions[i], bits[i], rng, evaluator, n_features, mode="hybrid_sv", classifier_encoding=classifier_encoding
        )

    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()
    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    history = [(evaluator.n_evaluations, gbest_fit)]

    for gen, _ in generation_schedule(n_generations, max_evaluations, evaluator):
        for i in range(pop_size):
            r1, r2 = rng.random(D), rng.random(D)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_pos[i] - positions[i])
                + c2 * r2 * (gbest_pos - positions[i])
            )
            velocities[i] = clamp(velocities[i])
            positions[i] = clamp(positions[i] + velocities[i])
            bits[i], fitness[i], infos[i] = binarize_and_eval(
                positions[i], bits[i], rng, evaluator, n_features, mode="hybrid_sv", classifier_encoding=classifier_encoding
            )
            if fitness[i] < pbest_fit[i]:
                pbest_fit[i] = fitness[i]
                pbest_pos[i] = positions[i].copy()

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
