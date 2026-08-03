"""Binarization / transfer functions used by the reproduced baseline
optimizers, matching what each source paper specifies (or, where a paper is
silent, the most standard convention for that algorithm family).

Each function takes a continuous position vector and returns a boolean mask
of the same length. `hybrid_sv_binarize` additionally needs a fitness
callback since it picks whichever of the S-shaped/V-shaped candidates scores
better (Hashemi et al. 2026, Eq. 8).
"""

import numpy as np
from scipy.special import erf

from .base import decode_bits, decode_bits_top1, stochastic_binarize


def _decode(bits, position, n_features, rng, classifier_encoding):
    """Dispatch to multi-hot (independent classifier switches, this
    project's ensemble design) or top-1 (argmax, single classifier)
    decoding, shared by every reproduced baseline below."""
    if classifier_encoding == "top1":
        return decode_bits_top1(bits, position, n_features, rng)
    return decode_bits(bits, n_features, rng)


def s_shaped(position, rng):
    """Eq. 4-5, Hashemi et al. Standard sigmoid transfer function."""
    prob = 1.0 / (1.0 + np.exp(-position))
    r = rng.random(position.shape[-1])
    return r >= prob  # bit=1 iff rand >= S(x), per the paper's stated rule


def v_shaped(position, rng, prev_bits):
    """Eq. 6-7, Hashemi et al. V-shaped transfer flips the previous bit with
    probability V(x); requires the current binary state to flip from."""
    prob = np.abs(erf((np.sqrt(np.pi) / 2.0) * position))
    r = rng.random(position.shape[-1])
    flip = r >= prob
    return np.where(flip, ~prev_bits, prev_bits)


def hybrid_sv_binarize(position, rng, prev_bits, eval_fn):
    """Eq. 8, Hashemi et al.: binarize via both S-shaped and V-shaped
    transfer functions, keep whichever scores better fitness.
    `eval_fn(bits) -> (fitness, info)`, lower fitness = better.
    Returns (bits, fitness, info) for whichever candidate won, so callers
    don't need a third evaluation."""
    bits_s = s_shaped(position, rng)
    bits_v = v_shaped(position, rng, prev_bits)
    fit_s, info_s = eval_fn(bits_s)
    fit_v, info_v = eval_fn(bits_v)
    return (bits_s, fit_s, info_s) if fit_s <= fit_v else (bits_v, fit_v, info_v)


def threshold_binarize(position, threshold=0.5):
    """Plain hard threshold, e.g. Eq. 26 in Hashim et al. (mHGS): positions
    live directly in [0,1] and bit=1 iff position > threshold."""
    return position > threshold


def binarize_and_eval(position, prev_bits, rng, evaluator, n_features, mode="stochastic", classifier_encoding="multi_hot"):
    """Single entry point every reproduced optimizer uses to go from a
    continuous position to (bits, fitness, info), so each can select the
    transfer function its source paper specifies (or, where the paper is
    silent, the project default `mode="stochastic"`).

    `classifier_encoding` selects how the last 5 dimensions of the decoded
    bits become a classifier mask: `"multi_hot"` (default) treats them as
    independent on/off switches feeding an equal-weight soft-voting
    ensemble, matching this project's own EOACSO design; `"top1"` instead
    picks a single classifier via argmax of the raw continuous scores."""
    if mode == "hybrid_sv":

        def eval_fn(b):
            fmask, cmask = _decode(b, position, n_features, rng, classifier_encoding)
            return evaluator.evaluate_masks(fmask, cmask)

        return hybrid_sv_binarize(position, rng, prev_bits, eval_fn)

    if mode == "stochastic":
        bits = stochastic_binarize(position, rng)
    elif mode == "threshold":
        bits = threshold_binarize(position, 0.5)
    else:
        raise ValueError(f"unknown binarize mode {mode!r}")

    fmask, cmask = _decode(bits, position, n_features, rng, classifier_encoding)
    fitness, info = evaluator.evaluate_masks(fmask, cmask)
    return bits, fitness, info
