"""Binary Butterfly Optimization Algorithm, reproduced per Hashemi et al.
(2026), Sec. 3.4.5 / Table 3: sensory modality c=0.01, power exponent a
increasing 0.1->0.3 over the run, switch probability p=0.6, hybrid S/V
transfer function (Eq. 4-8). The fragrance/movement equations themselves
aren't restated in that paper -- we use the canonical BOA formulas (Arora &
Singh, 2019). Sensory intensity I is defined here as (1 - fitness), since
BOA is a maximization metaheuristic and our fitness is minimized in [0,1].
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult
from .transfer import binarize_and_eval


def run_bboa(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    c=0.01,
    a_min=0.1,
    a_max=0.3,
    p=0.6,
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
            positions[i], bits[i], rng, evaluator, n_features, mode="hybrid_sv", classifier_encoding=classifier_encoding
        )

    gbest_idx = int(np.argmin(fitness))
    gbest_pos = positions[gbest_idx].copy()
    gbest_fit = float(fitness[gbest_idx])
    gbest_info = infos[gbest_idx]
    history = [(evaluator.n_evaluations, gbest_fit)]

    for gen, t_frac in generation_schedule(n_generations, max_evaluations, evaluator):
        a = a_min + (a_max - a_min) * t_frac
        for i in range(pop_size):
            intensity = max(1.0 - fitness[i], 1e-6)
            f_i = c * intensity**a
            if rng.random() < p:
                r = rng.random(D)
                new_pos = positions[i] + (r**2 * gbest_pos - positions[i]) * f_i
            else:
                j, k = rng.choice(pop_size, size=2, replace=False)
                r = rng.random(D)
                new_pos = positions[i] + (r**2 * positions[j] - positions[k]) * f_i
            positions[i] = clamp(new_pos)
            bits[i], fitness[i], infos[i] = binarize_and_eval(
                positions[i], bits[i], rng, evaluator, n_features, mode="hybrid_sv", classifier_encoding=classifier_encoding
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
