"""Generates the transfer-function-selector decode diagram for Methods
Section 3.6, matching the visual language of the user-authored
figures/CSO.pdf (rounded boxes, yellow labels for encoded quantities,
pale-blue labels for process steps, orange flow arrows) but showing a
single new concept: once a loser's position includes a LEADING
transfer-function-selector segment (Section 3.3) -- position layout
[selector(5) | features(D) | classifiers(5)] -- decoding it into a
feature mask and classifier mask takes one extra step -- argmax the
leading selector segment to pick one of 5 candidate transfer functions
(Section 3.4), then use that candidate to binarize the trailing
feature+classifier segment.

This replaces the old make_eoacso_diagram.py (deleted alongside EOACSO's
three strategies): no amber/violet/teal 3-strategy color coding is needed
here, since there is only one new idea to show, not three.

Run: python make_cso_tf_selector_diagram.py  (writes
cso_tf_selector_diagram.pdf/.png into this same directory)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

YELLOW = "#fdf16a"
BLUE = "#cfe6f6"
INK = "#1c1c1c"
ORANGE = "#e8944f"
SELECTED = "#1f7a6e"
SELECTED_BG = "#dcf1ec"

fig, ax = plt.subplots(figsize=(12.5, 9.8))
ax.set_xlim(0, 19)
ax.set_ylim(0, 15.5)
ax.axis("off")
ax.set_aspect("equal")


def box(cx, cy, w, h, fc, ec, lw=1.6, rounding=0.12, z=2):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.03,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
    )
    ax.add_patch(b)
    return b


def label(cx, cy, text, kind="yellow", fontsize=11, w=None, h=0.55, bold=False):
    styles = {"yellow": (YELLOW, INK), "blue": (BLUE, INK)}
    fc, ec = styles[kind]
    if w is None:
        w = 0.145 * len(text) + 0.5
    box(cx, cy, w, h, fc, ec, rounding=0.10)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            color=INK, zorder=6, fontweight="bold" if bold else "normal")
    return w, h


def arrow(p0, p1, color=ORANGE, lw=2.1, style="-", connect="arc3,rad=0.0"):
    a = FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, linestyle=style, connectionstyle=connect,
        zorder=1, shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)


# ---- Row 1: the loser's updated position, split into two segments -------
# Layout matches the actual encoding order: [selector(5) | feature+classifier(D+5)]
pos_cy = 14.2
SEL_CX, SEL_W = 3.6, 5.0
FEAT_CX, FEAT_W = 11.6, 10.6
box(SEL_CX, pos_cy, SEL_W, 1.1, SELECTED_BG, SELECTED, rounding=0.08)
ax.text(SEL_CX, pos_cy, r"selector segment" "\n" r"$t_1,\dots,t_5$",
        ha="center", va="center", fontsize=9.5, color=SELECTED, zorder=6, fontweight="bold")
box(FEAT_CX, pos_cy, FEAT_W, 1.1, "white", INK, rounding=0.08)
ax.text(FEAT_CX, pos_cy, r"feature mask $+$ classifier segment  ($D+5$ dims)",
        ha="center", va="center", fontsize=10.5, color=INK, zorder=6)
label((SEL_CX + FEAT_CX) / 2, pos_cy + 1.05, r"Loser's updated position (length $5+D+5$)", "yellow", w=9.6)

# ---- Row 2: argmax over the leading selector segment ---------------------
arrow((SEL_CX, pos_cy - 0.55), (SEL_CX, 12.35))
label(SEL_CX, 11.85, "argmax: pick 1 of 5 candidates", "blue", w=6.6)

# ---- Row 3: the 5 candidates, one highlighted as selected ----------------
cand_names = [
    "classic\nS-shaped",
    "Hashemi's\nS-shaped",
    "erf\nV-shaped",
    "tanh\nV-shaped",
    "hard\nthreshold",
]
cand_xs = [2.2, 5.9, 9.6, 13.3, 17.0]
cand_cy = 9.9
selected_idx = 3  # illustrative: this particle happened to pick tanh V-shaped
for i, (cx, name) in enumerate(zip(cand_xs, cand_names)):
    is_sel = i == selected_idx
    fc, ec = (SELECTED_BG, SELECTED) if is_sel else ("white", INK)
    box(cx, cand_cy, 2.6, 1.5, fc, ec, lw=2.0 if is_sel else 1.3, rounding=0.14)
    ax.text(cx, cand_cy, name, ha="center", va="center", fontsize=9,
            color=SELECTED if is_sel else INK, fontweight="bold" if is_sel else "normal", zorder=6)

arrow((SEL_CX, 11.35), (cand_xs[selected_idx], cand_cy + 0.75), color=SELECTED, style="--",
      connect="arc3,rad=0.55")
ax.text((SEL_CX + cand_xs[selected_idx]) / 2, 11.55, "selected (this particle)",
        fontsize=8.6, color=SELECTED, ha="center", zorder=6)

# ---- Row 4: binarize using the selected candidate ------------------------
BIN_CX, BIN_W = 9.5, 7.4
binarize_cy = 7.3
arrow((cand_xs[selected_idx], cand_cy - 0.75), (BIN_CX + 1.2, binarize_cy + 0.7), color=SELECTED,
      connect="arc3,rad=0.2")
arrow((FEAT_CX, pos_cy - 0.55), (BIN_CX + 2.4, binarize_cy + 0.7), connect="arc3,rad=-0.15")
box(BIN_CX, binarize_cy, BIN_W, 1.3, "white", INK, rounding=0.12)
ax.text(BIN_CX, binarize_cy, "binarize the trailing $D+5$ dims using\nthe selected transfer function",
        ha="center", va="center", fontsize=10, color=INK, zorder=6)

# ---- Row 5: decode into feature mask + classifier mask -------------------
decode_cy = 5.1
arrow((BIN_CX, binarize_cy - 0.65), (BIN_CX, decode_cy + 0.55))
label(BIN_CX, decode_cy, "decode: feature mask + classifier mask", "yellow", w=8.2)

# ---- Row 6: evaluate -------------------------------------------------------
eval_cy = 3.2
arrow((BIN_CX, decode_cy - 0.55), (BIN_CX, eval_cy + 0.65))
box(BIN_CX, eval_cy, 6.4, 1.3, "white", INK, rounding=0.12)
ax.text(BIN_CX, eval_cy, "evaluate: 5-fold CV, equal-weight\nsoft-voting ensemble",
        ha="center", va="center", fontsize=9.8, color=INK, zorder=6)
label(BIN_CX, eval_cy - 1.15, "fitness, info (incl. which transfer function was used)", "blue", w=9.4,
      fontsize=9.5)

plt.savefig("cso_tf_selector_diagram.pdf", bbox_inches="tight")
plt.savefig("cso_tf_selector_diagram.png", dpi=200, bbox_inches="tight")
print("saved cso_tf_selector_diagram.pdf / .png")
