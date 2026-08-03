"""Quantum Mayfly Optimization (QMFO), reproduced per Mansour (2024),
"Quantum mayfly optimization based feature subset selection with hybrid CNN
for biomedical Parkinson's disease diagnosis" -- feature-selection part only
(the paper's CNN-ALSTM classifier is replaced by this project's own shared
FitnessEvaluator).

The male/female movement and offspring equations (Eqs. 1-8) are the
canonical Mayfly Algorithm that the paper itself restates, and are
reproduced faithfully below. The paper's "quantum" modification (Eqs. 9-11:
qubit representation and a rotation-gate mutation, adapted from a quantum
grasshopper algorithm) is described only qualitatively -- it gives no
formula connecting the qubit/rotation-gate mechanism to the mayfly
position-update equations, and states no transfer function or population/
iteration parameters (all explicitly flagged as gaps by the paper-reading
pass). We fill that specific gap with a standard quantum-rotation-gate step
in the spirit of Han & Kim's QEA: each generation, every individual's
continuous position is nudged toward the global best along dimensions where
its current bit disagrees with the best's bit, by a small fixed rotation
step -- the closest faithful reading of "a rotation gate mutation operator
that helps mayflies converge on the global optimum" without a worked
formula to follow exactly. Binarization otherwise uses the project's default
stochastic S-shaped transfer function, since the paper gives none.
"""

import numpy as np

from .base import BOUND, clamp, dimension, generation_schedule
from .result import OptimizationResult
from .transfer import binarize_and_eval


def run_qmfo(
    evaluator,
    n_features,
    pop_size=20,
    n_generations=30,
    seed=0,
    a1=1.0,
    a2=1.5,
    beta=2.0,
    dance=5.0,
    fl=1.0,
    rotation_step=0.15,
    max_evaluations=None,
    classifier_encoding="multi_hot",
):
    rng = np.random.default_rng(seed)
    D = dimension(n_features)
    n_half = max(pop_size // 2, 1)

    males = rng.uniform(-BOUND, BOUND, size=(n_half, D))
    females = rng.uniform(-BOUND, BOUND, size=(n_half, D))
    v_males = np.zeros((n_half, D))
    v_females = np.zeros((n_half, D))

    bits_m = np.zeros((n_half, D), dtype=bool)
    bits_f = np.zeros((n_half, D), dtype=bool)
    fit_m = np.empty(n_half)
    fit_f = np.empty(n_half)
    info_m = [None] * n_half
    info_f = [None] * n_half

    for i in range(n_half):
        bits_m[i], fit_m[i], info_m[i] = binarize_and_eval(
            males[i], bits_m[i], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
        )
        bits_f[i], fit_f[i], info_f[i] = binarize_and_eval(
            females[i], bits_f[i], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
        )

    pbest_m = males.copy()
    pbest_fit_m = fit_m.copy()

    def global_best():
        i_m, i_f = int(np.argmin(fit_m)), int(np.argmin(fit_f))
        if fit_m[i_m] <= fit_f[i_f]:
            return males[i_m].copy(), float(fit_m[i_m]), info_m[i_m], bits_m[i_m].copy()
        return females[i_f].copy(), float(fit_f[i_f]), info_f[i_f], bits_f[i_f].copy()

    gbest_pos, gbest_fit, gbest_info, _ = global_best()
    history = [(evaluator.n_evaluations, gbest_fit)]

    for gen, _ in generation_schedule(n_generations, max_evaluations, evaluator):
        # male movement, Eq. 1-2; best male instead does the nuptial dance, Eq. 5
        best_male_idx = int(np.argmin(fit_m))
        for i in range(n_half):
            if i == best_male_idx:
                v_males[i] = v_males[i] + dance * rng.uniform(-1, 1, size=D)
            else:
                r_p = np.linalg.norm(males[i] - pbest_m[i])
                r_g = np.linalg.norm(males[i] - gbest_pos)
                v_males[i] = (
                    v_males[i]
                    + a1 * np.exp(-beta * r_p**2) * (pbest_m[i] - males[i])
                    + a2 * np.exp(-beta * r_g**2) * (gbest_pos - males[i])
                )
            males[i] = clamp(males[i] + v_males[i])
            bits_m[i], fit_m[i], info_m[i] = binarize_and_eval(
                males[i], bits_m[i], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
            )
            if fit_m[i] < pbest_fit_m[i]:
                pbest_fit_m[i] = fit_m[i]
                pbest_m[i] = males[i].copy()

        # female movement, Eq. 6-7
        for i in range(n_half):
            r_mf = np.linalg.norm(males[i] - females[i])
            if fit_f[i] > fit_m[i]:
                v_females[i] = v_females[i] + a2 * np.exp(-beta * r_mf**2) * (males[i] - females[i])
            else:
                v_females[i] = v_females[i] + fl * rng.uniform(-1, 1, size=D)
            females[i] = clamp(females[i] + v_females[i])
            bits_f[i], fit_f[i], info_f[i] = binarize_and_eval(
                females[i], bits_f[i], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
            )

        # mating/offspring, Eq. 8: best male x best female, replace worst if better
        order_m, order_f = np.argsort(fit_m), np.argsort(fit_f)
        L = rng.random()
        offspring1 = clamp(L * males[order_m[0]] + (1 - L) * females[order_f[0]])
        offspring2 = clamp(L * females[order_f[0]] + (1 - L) * males[order_m[0]])
        for offspring, pool, fit_pool, bits_pool, info_pool in (
            (offspring1, males, fit_m, bits_m, info_m),
            (offspring2, females, fit_f, bits_f, info_f),
        ):
            b, f, info = binarize_and_eval(
                offspring, bits_pool[0], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
            )
            worst = int(np.argmax(fit_pool))
            if f < fit_pool[worst]:
                pool[worst], fit_pool[worst], bits_pool[worst], info_pool[worst] = offspring, f, b, info

        # quantum rotation-gate step: nudge every individual toward the
        # current global best along dimensions whose bit disagrees with it.
        # `rot_pos`/`rot_bits` are only this step's target snapshot -- kept
        # separate from `gbest_pos`/`gbest_fit`/`gbest_info` (the tracked
        # best-ever solution) so those three are only ever updated together,
        # below, and never left mismatched with each other.
        rot_pos, _, _, rot_bits = global_best()
        for pool, bits_pool, fit_pool, info_pool in (
            (males, bits_m, fit_m, info_m),
            (females, bits_f, fit_f, info_f),
        ):
            for i in range(n_half):
                disagree = bits_pool[i] != rot_bits
                pool[i] = clamp(pool[i] + rotation_step * disagree * (rot_pos - pool[i]))
                bits_pool[i], fit_pool[i], info_pool[i] = binarize_and_eval(
                    pool[i], bits_pool[i], rng, evaluator, n_features, mode="stochastic", classifier_encoding=classifier_encoding
                )

        cand_pos, cand_fit, cand_info, _ = global_best()
        if cand_fit < gbest_fit - 1e-9:
            gbest_fit, gbest_pos, gbest_info = cand_fit, cand_pos, cand_info
        history.append((evaluator.n_evaluations, gbest_fit))

    return OptimizationResult(
        best_position=gbest_pos,
        best_fitness=gbest_fit,
        best_info=gbest_info,
        history=history,
        n_evaluations=evaluator.n_evaluations,
    )
