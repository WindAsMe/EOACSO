"""Generates the encoding/decoding schematic for Methods Section 3.3.
Run: python make_encoding_diagram.py  (writes encoding_diagram.pdf and .png
into this same directory)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(9.5, 6.7))
ax.set_xlim(0, 10)
ax.set_ylim(3.2, 15.6)
ax.axis("off")

FEATURE_COLOR = "#dbe9f6"
CLASSIFIER_COLOR = "#fde6cf"
EOACSO_COLOR = "#d9f0d3"
BASELINE_COLOR = "#f6d9d9"
SHARED_COLOR = "#eeeeee"


def box(x, y, w, h, text, color="white", fontsize=9.5, weight="normal", ls="-"):
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.1, linestyle=ls,
        edgecolor="#333333", facecolor=color, zorder=2,
    )
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, zorder=3, linespacing=1.35)
    return (x, y, w, h)


def arrow(xy_from, xy_to, style="-|>", lw=1.3, color="#333333", connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(
        xy_from, xy_to, arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, zorder=1, connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


# 1) Continuous particle position -----------------------------------------
pos_feat = box(0.6, 13.6, 5.4, 1.3,
               r"$f_1\ \ f_2\ \ \cdots\ \ f_D$" + "\nfeature-selection scores",
               color=FEATURE_COLOR, fontsize=10)
pos_clf = box(6.2, 13.6, 3.2, 1.3,
              r"$c_1\ c_2\ c_3\ c_4\ c_5$" + "\nclassifier scores",
              color=CLASSIFIER_COLOR, fontsize=10)
ax.text(5.0, 15.15, "Continuous particle position  (length $D+5$)",
        ha="center", fontsize=11, weight="bold")

# 2) Transfer function ------------------------------------------------------
tf = box(2.6, 11.9, 4.8, 1.0,
         "Transfer function (per optimizer: stochastic S-shaped / V-shaped /\n"
         "hybrid S-V / threshold-at-0.5)", color=SHARED_COLOR, fontsize=8.7)
arrow(center_bottom(pos_feat), (tf[0] + 1.4, tf[1] + tf[3]))
arrow(center_bottom(pos_clf), (tf[0] + 3.4, tf[1] + tf[3]))

# 3) Shared feature mask -----------------------------------------------------
fmask = box(0.6, 10.3, 5.4, 1.0,
            r"Feature mask $S\subseteq\{1,\dots,D\}$  (shared by every algorithm)"
            "\nempty mask repaired: force-activate one feature at random",
            color=FEATURE_COLOR, fontsize=8.7)
arrow((tf[0] + 1.4, tf[1]), (fmask[0] + 1.4, fmask[1] + fmask[3]))

# 4) Branch label -------------------------------------------------------------
ax.text(7.8, 10.85, "classifier portion decodes\ndifferently per algorithm:",
        ha="center", fontsize=9, style="italic", color="#444444")
arrow((tf[0] + 3.4, tf[1]), (5.55, 10.85), connectionstyle="arc3,rad=-0.25")

# 5) EOACSO branch (multi-hot) ------------------------------------------------
eo_head = box(0.6, 8.75, 4.3, 0.85, "EOACSO (+ ablation variants)\nmulti-hot: independent on/off switch per classifier",
              color=EOACSO_COLOR, fontsize=8.6, weight="bold")
eo_mask = box(0.6, 7.35, 4.3, 1.0,
              r"classifier mask $C=\{c : \mathrm{bit}_c=1\}$, $|C|\geq 1$"
              "\n(empty mask repaired: activate one classifier at random)",
              color=EOACSO_COLOR, fontsize=8.3)
arrow(center_bottom(eo_head), center_top(eo_mask))

eo_ens = box(0.6, 5.55, 4.3, 1.3,
             "Soft-voting ensemble\nevery classifier in $C$ trains on feature\n"
             "subset $S$; out-of-fold probability =\nunweighted average over $C$",
             color=EOACSO_COLOR, fontsize=8.3)
arrow(center_bottom(eo_mask), center_top(eo_ens))

# 6) Baseline branch (top-1) --------------------------------------------------
bl_head = box(5.1, 8.75, 4.3, 0.85, "7 reproduced baselines\ntop-1: argmax of the 5 raw scores",
              color=BASELINE_COLOR, fontsize=8.6, weight="bold")
bl_mask = box(5.1, 7.35, 4.3, 1.0,
              r"classifier mask $C=\{\mathrm{argmax}_c\ c_i\}$, $|C|=1$"
              "\n(always exactly one classifier -- never empty)",
              color=BASELINE_COLOR, fontsize=8.3)
arrow(center_bottom(bl_head), center_top(bl_mask))

bl_single = box(5.1, 5.55, 4.3, 1.3,
                "Single classifier\nthe one classifier in $C$ trains on\n"
                "feature subset $S$; out-of-fold\nprobability = that classifier's own",
                color=BASELINE_COLOR, fontsize=8.3)
arrow(center_bottom(bl_mask), center_top(bl_single))

# connect shared feature mask down into both branch heads
arrow(center_bottom(fmask), center_top(eo_head), connectionstyle="arc3,rad=0.15")
arrow(center_bottom(fmask), center_top(bl_head), connectionstyle="arc3,rad=-0.15")

# 7) Merge into fitness evaluation --------------------------------------------
fit = box(1.6, 3.55, 6.8, 1.35,
          r"5-fold subject-grouped CV on $(S, C)$" + "\n"
          r"fitness $= \omega(1-\mathrm{BalAcc}) + (1-\omega)\ |S|/D$",
          color=SHARED_COLOR, fontsize=9.3, weight="normal")
arrow(center_bottom(eo_ens), (fit[0] + 1.8, fit[1] + fit[3]))
arrow(center_bottom(bl_single), (fit[0] + 5.0, fit[1] + fit[3]))

# Legend -----------------------------------------------------------------
legend_elems = [
    Line2D([0], [0], marker="s", color="none", markerfacecolor=FEATURE_COLOR,
           markeredgecolor="#333333", markersize=13, label="shared feature encoding"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=EOACSO_COLOR,
           markeredgecolor="#333333", markersize=13, label="EOACSO (multi-hot ensemble)"),
    Line2D([0], [0], marker="s", color="none", markerfacecolor=BASELINE_COLOR,
           markeredgecolor="#333333", markersize=13, label="baselines (top-1)"),
]
ax.legend(handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.02),
          ncol=3, frameon=False, fontsize=8.8)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("encoding_diagram.pdf", bbox_inches="tight")
plt.savefig("encoding_diagram.png", dpi=200, bbox_inches="tight")
print("saved encoding_diagram.pdf / .png")
