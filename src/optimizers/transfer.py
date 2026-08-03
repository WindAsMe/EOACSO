"""Binarization / transfer functions used by the reproduced baseline
optimizers, matching what each source paper specifies (or, where a paper is
silent, the most standard convention for that algorithm family).

Each function takes a continuous position vector and returns a boolean mask
of the same length. `hybrid_sv_binarize` additionally needs a fitness
callback since it picks whichever of the S-shaped/V-shaped candidates scores
better (Hashemi et al. 2026, Eq. 8).

`TF_CANDIDATES` / `decode_tf_choice` / `decode_and_binarize_searched_tf` below
are for the proposed method only (plain CSO, `src/optimizers/cso.py`): they
let each particle pick its own transfer function as part of its own
encoding, via a 5-dim argmax segment that leads the position vector --
`[t_1..t_5 | f_1..f_D | c_1..c_5]` -- rather than trailing it, so the
choice of transfer function is decoded before, and independently of, the
feature/classifier dimensionality `D` it goes on to binarize. The 7
reproduced baselines above are untouched -- they keep calling
`binarize_and_eval` with one hard-coded `mode` per their own source paper,
over their own `[f_1..f_D | c_1..c_5]` encoding with no selector segment.
"""

from collections import namedtuple

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


def v_shaped_tanh(position, rng, prev_bits):
    """V2, Mirjalili & Lewis (2013): |tanh(x)|. Same flip-the-previous-bit
    mechanic as `v_shaped` above (V1, erf-based), just a different closed
    form -- the second of the proposed method's two V-shaped candidates."""
    prob = np.abs(np.tanh(position))
    r = rng.random(position.shape[-1])
    flip = r >= prob
    return np.where(flip, ~prev_bits, prev_bits)


def binarize_and_eval(position, prev_bits, rng, evaluator, n_features, mode="stochastic", classifier_encoding="multi_hot"):
    """Single entry point every reproduced optimizer uses to go from a
    continuous position to (bits, fitness, info), so each can select the
    transfer function its source paper specifies (or, where the paper is
    silent, the project default `mode="stochastic"`).

    `classifier_encoding` selects how the last 5 dimensions of the decoded
    bits become a classifier mask: `"multi_hot"` (default) treats them as
    independent on/off switches feeding an equal-weight soft-voting
    ensemble, matching this project's own proposed-method design; `"top1"`
    instead picks a single classifier via argmax of the raw continuous
    scores."""
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


# ================================================================
# Searched transfer function (proposed method only, see module docstring).
# ================================================================

TFCandidate = namedtuple("TFCandidate", ["name", "fn"])

N_TF_CANDIDATES = 5

# Every fn has the same (segment, rng, prev_segment) signature so callers
# never need to branch on which candidate they picked; candidates that don't
# need rng/prev_segment simply ignore those arguments.
TF_CANDIDATES = [
    TFCandidate("s_shaped_classic", lambda seg, rng, prev: stochastic_binarize(seg, rng)),
    TFCandidate("s_shaped_hashemi", lambda seg, rng, prev: s_shaped(seg, rng)),
    TFCandidate("v_shaped_erf", lambda seg, rng, prev: v_shaped(seg, rng, prev)),
    TFCandidate("v_shaped_tanh", lambda seg, rng, prev: v_shaped_tanh(seg, rng, prev)),
    # Rescaled to threshold=0.0, not the baselines' 0.5: this segment lives in
    # the symmetric [-BOUND, BOUND] range like every other candidate above
    # (all centered at x=0), not mHGS/MGWO-eP's native [0,1].
    TFCandidate("hard_threshold", lambda seg, rng, prev: threshold_binarize(seg, threshold=0.0)),
]


def decode_tf_choice(position, rng):
    """Argmax over the leading 5-dim transfer-function-selector segment --
    no binarization needed for this segment itself, exactly like the
    existing top-1 classifier argmax in `decode_bits_top1`. Returns an int
    index into `TF_CANDIDATES`."""
    return int(np.argmax(position[:N_TF_CANDIDATES]))


def decode_and_binarize_searched_tf(position, prev_bits, rng, fixed_tf_index=None):
    """Full per-particle pipeline for the proposed method: pick a transfer
    function (argmax on this particle's own leading selector segment,
    unless `fixed_tf_index` overrides it -- used only by ad-hoc analysis),
    then binarize the trailing feature+classifier dims with that
    candidate's own rule and its own slice of `prev_bits`.

    Returns `(bits, tf_index)`, where `bits` has the same length as
    `position` (the leading selector segment is left as `prev_bits`
    unchanged, since it is never itself binarized)."""
    tf_idx = fixed_tf_index if fixed_tf_index is not None else decode_tf_choice(position, rng)
    seg = position[N_TF_CANDIDATES:]
    prev_seg = prev_bits[N_TF_CANDIDATES:]
    seg_bits = TF_CANDIDATES[tf_idx].fn(seg, rng, prev_seg)
    bits = prev_bits.copy()
    bits[N_TF_CANDIDATES:] = seg_bits
    return bits, tf_idx
