"""Generates the transfer-function schematic for Methods Section 3.3.

Each curve is plotted from its exact closed form (smooth), derived
directly from the real implementation in src/optimizers/base.py and
src/optimizers/transfer.py:

  - EOACSO's stochastic S-shaped: bit=1 iff sigmoid(x) > rand(), so
    P(bit=1|x) = sigmoid(x) exactly.
  - Baselines' S-shaped (Hashemi et al. Eq. 4-5): bit=1 iff
    rand() >= sigmoid(x), so P(bit=1|x) = 1 - sigmoid(x) exactly.
  - Baselines' V-shaped (Hashemi et al. Eq. 6-7): flip iff
    rand() >= |erf(sqrt(pi)/2 * x)|, so P(flip|x) = 1 - |erf(...)| exactly.
  - MGWO-eP/mHGS threshold: bit = 1[x > 0.5], exact step function.

A Monte Carlo cross-check against the actual functions (not just the
formulas) is run first and asserted to match within sampling error, so
this script still fails loudly if the closed forms ever drift out of
sync with the code.

Run: python make_transfer_function_diagram.py  (writes
transfer_functions.pdf and .png into this same directory)."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.optimizers.base import stochastic_binarize, sigmoid
from src.optimizers.transfer import s_shaped, v_shaped, threshold_binarize

# ---- Cross-check closed forms against the real functions (Monte Carlo) ----
rng = np.random.default_rng(0)
N_TRIALS = 3000
for x in (-3.0, -1.0, 0.0, 1.0, 3.0):
    p_eoacso_mc = np.mean([stochastic_binarize(np.array([x]), rng)[0] for _ in range(N_TRIALS)])
    assert abs(p_eoacso_mc - sigmoid(x)) < 0.05, f"EOACSO S-shaped closed form mismatch at x={x}"

    p_hashemi_mc = np.mean([s_shaped(np.array([x]), rng)[0] for _ in range(N_TRIALS)])
    assert abs(p_hashemi_mc - (1 - sigmoid(x))) < 0.05, f"Baseline S-shaped closed form mismatch at x={x}"

    p_flip_mc = np.mean([
        bool(v_shaped(np.array([x]), rng, np.array([False]))[0]) for _ in range(N_TRIALS)
    ])
    assert abs(p_flip_mc - (1 - abs(erf(np.sqrt(np.pi) / 2.0 * x)))) < 0.05, f"V-shaped closed form mismatch at x={x}"
print("Closed forms verified against live code (Monte Carlo, tol=0.05). Plotting smooth curves.")

# ---- Smooth closed-form curves ---------------------------------------------
xs = np.linspace(-4, 4, 400)
p_eoacso = sigmoid(xs)
p_hashemi_s = 1 - sigmoid(xs)
p_flip = 1 - np.abs(erf(np.sqrt(np.pi) / 2.0 * xs))

xs_thresh = np.linspace(0, 1, 400)
p_thresh = threshold_binarize(xs_thresh, 0.5).astype(float)

# ==================================================================
fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2))

ax = axes[0, 0]
ax.plot(xs, p_eoacso, color="#2e7d32", lw=2.2)
ax.set_xlabel(r"position $x$")
ax.set_ylabel(r"$P(\mathrm{bit}=1\mid x)$")
ax.axhline(0.5, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.text(0.05, 0.90, r"$P(\mathrm{bit}=1\mid x)=\sigma(x)=\dfrac{1}{1+e^{-x}}$",
        transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2e7d32", alpha=0.9))
ax.set_ylim(-0.05, 1.05)

ax = axes[0, 1]
ax.plot(xs, p_hashemi_s, color="#c62828", lw=2.2)
ax.set_xlabel(r"position $x$")
ax.set_ylabel(r"$P(\mathrm{bit}=1\mid x)$")
ax.axhline(0.5, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.text(0.05, 0.90, r"$P(\mathrm{bit}=1\mid x)=1-\sigma(x)$",
        transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c62828", alpha=0.9))
ax.set_ylim(-0.05, 1.05)

ax = axes[1, 0]
ax.plot(xs, p_flip, color="#6a1b9a", lw=2.2)
ax.set_xlabel(r"velocity-like position $x$")
ax.set_ylabel(r"$P(\mathrm{flip\ previous\ bit}\mid x)$")
ax.axhline(0.5, color="gray", lw=0.6, ls="--")
ax.axvline(0, color="gray", lw=0.6, ls="--")
ax.text(0.30, 0.92, r"$P(\mathrm{flip}\mid x)=1-\left|\mathrm{erf}\!\left(\dfrac{\sqrt{\pi}}{2}x\right)\right|$",
        transform=ax.transAxes, fontsize=10.5, va="top", ha="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#6a1b9a", alpha=0.9))
ax.set_ylim(-0.05, 1.05)

ax = axes[1, 1]
ax.plot(xs_thresh, p_thresh, color="#ef6c00", lw=2.2)
ax.set_xlabel(r"position $x \in [0,1]$")
ax.set_ylabel(r"$\mathrm{bit}$")
ax.axvline(0.5, color="gray", lw=0.6, ls="--")
ax.text(0.05, 0.55, r"$\mathrm{bit} = \mathbb{1}[x > 0.5]$",
        transform=ax.transAxes, fontsize=10.5, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ef6c00", alpha=0.9))
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("transfer_functions.pdf", bbox_inches="tight")
plt.savefig("transfer_functions.png", dpi=200, bbox_inches="tight")
print("saved transfer_functions.pdf / .png")
