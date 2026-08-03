"""The proposed method: plain Competitive Swarm Optimizer (CSO, Cheng & Jin
2015) with a per-particle searched transfer function.

Particles are randomly paired each generation; the fitter one (winner) is
left completely unchanged, the other (loser) updates its velocity/position
toward the winner using CSO's own mean-position social term (Eq. 25-26) --
no strategies are added on top of this skeleton.

The only departure from a textbook CSO re-implementation is the encoding:
each particle carries an extra 5-dim transfer-function-selector segment
(decoded via argmax, see `transfer.TF_CANDIDATES`) on top of the shared
feature-mask + classifier-mask segments, so the search picks which of 5
transfer functions binarizes its own solution -- rather than every
algorithm sharing one hard-coded transfer function, as the 7 reproduced
baselines still do. `fixed_tf_index` overrides this per-particle choice
with one fixed candidate for every particle/generation instead; unused by
the registered `CSO_searched_tf` algorithm (always `None`), available for
ad-hoc analysis.

`classifier_encoding` selects how the classifier-mask segment decodes (see
`FitnessEvaluator.evaluate_searched_tf`): `"multi_hot"` (default) is this
project's own multi-classifier soft-voting ensemble; `"top1"` matches the 7
reproduced baselines' single-classifier scheme, for a cheaper run.
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult
from .transfer import N_TF_CANDIDATES


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


def run_cso(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    max_evaluations=None,
    classifier_encoding="multi_hot",
    fixed_tf_index=None,
):
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even (CSO pairs particles up)")

    rng = np.random.default_rng(seed)
    D = dimension(n_features) + N_TF_CANDIDATES

    positions = rng.uniform(-BOUND, BOUND, size=(pop_size, D))
    velocities = np.zeros((pop_size, D))
    bits = np.zeros((pop_size, D), dtype=bool)
    fitness = np.empty(pop_size)
    infos = [None] * pop_size

    for i in range(pop_size):
        bits[i], fitness[i], infos[i] = evaluator.evaluate_searched_tf(
            positions[i], bits[i], rng, classifier_encoding=classifier_encoding, fixed_tf_index=fixed_tf_index
        )

    best_idx = int(np.argmin(fitness))
    best_position = positions[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    best_info = infos[best_idx]
    history = [(evaluator.n_evaluations, best_fitness)]

    for gen, t_frac in generation_schedule(n_generations, max_evaluations, evaluator):
        order = rng.permutation(pop_size)
        phi = cso_phi(pop_size)  # Cheng & Jin (2015), Eq. 25-26

        for k in range(0, pop_size, 2):
            i, j = int(order[k]), int(order[k + 1])
            w_idx, l_idx = (i, j) if fitness[i] <= fitness[j] else (j, i)

            R1, R2, R3 = rng.random(D), rng.random(D), rng.random(D)

            mean_pos = positions.mean(axis=0)
            guide_term = phi * R3 * (mean_pos - positions[l_idx])

            velocities[l_idx] = (
                R1 * velocities[l_idx]
                + R2 * (positions[w_idx] - positions[l_idx])
                + guide_term
            )
            velocities[l_idx] = clamp(velocities[l_idx])
            positions[l_idx] = clamp(positions[l_idx] + velocities[l_idx])

            bits[l_idx], fitness[l_idx], infos[l_idx] = evaluator.evaluate_searched_tf(
                positions[l_idx], bits[l_idx], rng,
                classifier_encoding=classifier_encoding, fixed_tf_index=fixed_tf_index,
            )

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
    )
