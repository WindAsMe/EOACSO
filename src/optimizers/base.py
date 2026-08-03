"""Shared encoding utilities for binary swarm optimizers.

A particle's continuous position has length ``n_features + N_CLASSIFIERS``:
the first ``n_features`` dimensions encode a *shared* feature subset. The
last ``N_CLASSIFIERS`` dimensions can be decoded two ways:

  - ``decode_bits`` / ``decode_particle`` -- multi-hot: independent on/off
    switches, so several classifiers can join a final soft-voting ensemble.
  - ``decode_bits_top1`` -- top-1: argmax of the raw continuous scores, so
    exactly one classifier is ever active.

EOACSO_Paper (`eoacso.py`) always uses the multi-hot form. The 7 reproduced
literature baselines (`transfer.py`, `mgwo_ep.py`, `mhgs.py`) accept a
``classifier_encoding`` argument (default ``"multi_hot"``, matching
EOACSO_Paper) and can be switched to ``"top1"`` per call -- see each
module's own runner function. As of the current default, every algorithm
in this project decodes multi-hot unless explicitly overridden; RE-CHECK
this default in the actual call sites before assuming it, since this
project's history has flipped this choice several times.
"""

import numpy as np

CLASSIFIER_NAMES = ["svm", "rf", "xgb", "knn", "logreg"]
N_CLASSIFIERS = len(CLASSIFIER_NAMES)
BOUND = 4.0  # position clamp; sigmoid saturates well before this


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clamp(position):
    return np.clip(position, -BOUND, BOUND)


def stochastic_binarize(position, rng):
    """Classic S-shaped transfer function binarization (Mirjalili & Lewis, 2013)."""
    probs = sigmoid(position)
    return (probs > rng.random(position.shape[-1])).astype(bool)


def decode_bits(bits, n_features, rng):
    """Split an already-binarized bit vector into (feature_mask, classifier_mask),
    with repair so neither mask is ever empty (an empty mask is not evaluable).
    Shared by every optimizer regardless of which transfer function produced bits."""
    feature_mask = np.asarray(bits[:n_features]).copy()
    clf_mask = np.asarray(bits[n_features:]).copy()

    if not feature_mask.any():
        feature_mask[rng.integers(n_features)] = True
    if not clf_mask.any():
        clf_mask[rng.integers(N_CLASSIFIERS)] = True

    return feature_mask, clf_mask


def decode_particle(position, n_features, rng):
    """Decode a continuous position into (feature_mask, classifier_mask) using
    the default stochastic S-shaped transfer function (used by EOACSO_Paper/CSO)."""
    bits = stochastic_binarize(position, rng)
    return decode_bits(bits, n_features, rng)


def decode_bits_top1(bits, position, n_features, rng):
    """Like `decode_bits`, but the classifier portion is a top-1 categorical
    pick -- argmax of the raw continuous `position[n_features:]` scores --
    instead of independent on/off switches, so exactly one classifier is
    ever active. Used by the 8 reproduced baseline optimizers (not EOACSO_Paper,
    which uses the multi-hot `decode_bits` above via `decode_particle`)."""
    feature_mask = np.asarray(bits[:n_features]).copy()
    if not feature_mask.any():
        feature_mask[rng.integers(n_features)] = True

    clf_mask = np.zeros(N_CLASSIFIERS, dtype=bool)
    clf_mask[int(np.argmax(position[n_features:]))] = True

    return feature_mask, clf_mask


def dimension(n_features):
    return n_features + N_CLASSIFIERS


def generation_schedule(n_generations, max_evaluations, evaluator):
    """Drives every optimizer's outer loop. Yields `(gen, t_frac)` pairs,
    `gen` starting at 1 and `t_frac = gen/n_generations` (clamped to <=1)
    for time-dependent schedules (e.g. GWO's `a`, EOACSO_Paper's `lambda(t)`).

    When `max_evaluations` is given, it is the sole stopping condition --
    the loop keeps yielding past `n_generations` if the evaluation budget
    isn't spent yet (this matters for stochastic-cost algorithms like
    mHGS, whose evaluations-per-generation isn't fixed). `n_generations`
    still normalizes `t_frac` so schedules behave sensibly.

    When `max_evaluations` is None, this reduces to the legacy behavior of
    exactly `n_generations` iterations."""
    gen = 0
    while True:
        gen += 1
        t_frac = min(gen / n_generations, 1.0)
        yield gen, t_frac
        if max_evaluations is not None:
            if evaluator.n_evaluations >= max_evaluations:
                return
        elif gen >= n_generations:
            return
