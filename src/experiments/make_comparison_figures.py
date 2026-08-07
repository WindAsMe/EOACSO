"""Generate comparison figures from results/tables/{dataset}_comparison_results.csv.

Covers the six algorithms the paper's comparison figures use -- BGWO, HybridGWO,
QMFO, BPSO, BBOA, CSO_searched_tf (plotted as "CSO") -- in that fixed order.
mHGS and MGWO-eP are present in the CSV but intentionally left out of these
figures.

For each dataset (oxford, naranjo) produces:
  convergence_<dataset>.pdf/png   mean best-fitness vs. evaluations
  boxplot_<dataset>.pdf/png       balanced-accuracy distribution across runs
  bar_metrics_<dataset>.pdf/png   mean +/- std of balanced accuracy / F1 / ROC-AUC
  bar_features_<dataset>.pdf/png  mean +/- std of selected-feature ratio

Run with: python -m src.experiments.make_comparison_figures
"""

import ast
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGO_ORDER = ["BGWO", "HybridGWO", "QMFO", "BPSO", "BBOA", "CSO_searched_tf"]
ALGO_LABELS = {"CSO_searched_tf": "CSO"}
COLORS = {
    "BGWO": "#2a78d6",
    "HybridGWO": "#eb6834",
    "QMFO": "#1baf7a",
    "BPSO": "#eda100",
    "BBOA": "#4a3aa7",
    "CSO_searched_tf": "#e34948",
}
PROPOSED = "CSO_searched_tf"
DATASETS = ["oxford", "naranjo"]
TABLES_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.labelcolor": INK,
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    }
)


def label(algo):
    return ALGO_LABELS.get(algo, algo)


def load(dataset):
    df = pd.read_csv(TABLES_DIR / f"{dataset}_comparison_results.csv")
    df = df[df["algorithm"].isin(ALGO_ORDER)].copy()
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=ALGO_ORDER, ordered=True)
    return df


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def plot_convergence(df, dataset):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    grid = np.arange(30, 3001, 15)
    for algo in ALGO_ORDER:
        sub = df[df["algorithm"] == algo]
        curves = []
        for hist_str in sub["history"]:
            hist = ast.literal_eval(hist_str)
            hx = np.array([p[0] for p in hist], dtype=float)
            hy = np.array([p[1] for p in hist], dtype=float)
            idx = np.clip(np.searchsorted(hx, grid, side="right") - 1, 0, len(hy) - 1)
            y = np.where(grid < hx[0], np.nan, hy[idx])
            curves.append(y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(np.vstack(curves), axis=0)
        proposed = algo == PROPOSED
        ax.plot(
            grid,
            mean,
            color=COLORS[algo],
            linewidth=2.4 if proposed else 1.4,
            label=label(algo),
            zorder=3 if proposed else 2,
            solid_capstyle="round",
        )
    ax.set_xlabel("Fitness evaluations")
    ax.set_ylabel("Mean best fitness")
    ax.set_xlim(0, 3000)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    _save(fig, f"convergence_{dataset}")


def plot_boxplot(df, dataset, metric="balanced_accuracy", metric_label="Balanced accuracy"):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    data = [df.loc[df["algorithm"] == a, metric].to_numpy() for a in ALGO_ORDER]
    positions = np.arange(1, len(ALGO_ORDER) + 1)
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color=INK, linewidth=1.4),
        whiskerprops=dict(color=INK_SECONDARY),
        capprops=dict(color=INK_SECONDARY),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=INK_MUTED, markeredgecolor="none"),
    )
    for patch, algo in zip(bp["boxes"], ALGO_ORDER):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.85)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([label(a) for a in ALGO_ORDER], rotation=20, ha="right")
    ax.set_ylabel(metric_label)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, f"boxplot_{dataset}")


METRICS = [
    ("balanced_accuracy", "Bal. Acc.", "#2a78d6"),
    ("f1", "F1", "#eb6834"),
    ("roc_auc", "ROC-AUC", "#1baf7a"),
]


def plot_bar_metrics(df, dataset):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(ALGO_ORDER))
    n_metric = len(METRICS)
    width = 0.8 / n_metric
    for i, (col, mlabel, color) in enumerate(METRICS):
        means = np.array([df.loc[df["algorithm"] == a, col].mean() for a in ALGO_ORDER])
        stds = np.array([df.loc[df["algorithm"] == a, col].std() for a in ALGO_ORDER])
        offset = (i - (n_metric - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width * 0.9,
            yerr=stds,
            capsize=2,
            color=color,
            edgecolor=INK,
            linewidth=0.5,
            label=mlabel,
            error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.8),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label(a) for a in ALGO_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, f"bar_metrics_{dataset}")


def plot_bar_features(df, dataset):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    means = np.array([df.loc[df["algorithm"] == a, "feature_ratio"].mean() for a in ALGO_ORDER])
    stds = np.array([df.loc[df["algorithm"] == a, "feature_ratio"].std() for a in ALGO_ORDER])
    colors = [COLORS[a] for a in ALGO_ORDER]
    x = np.arange(len(ALGO_ORDER))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=3,
        color=colors,
        edgecolor=INK,
        linewidth=0.6,
        error_kw=dict(ecolor=INK_SECONDARY, elinewidth=0.8),
    )
    for xi, m in zip(x, means):
        ax.text(xi, m + max(stds) + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([label(a) for a in ALGO_ORDER], rotation=20, ha="right")
    ax.set_ylabel("Selected feature ratio")
    ax.set_ylim(0, float((means + stds).max()) * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, f"bar_features_{dataset}")


def main():
    for dataset in DATASETS:
        df = load(dataset)
        plot_convergence(df, dataset)
        plot_boxplot(df, dataset)
        plot_bar_metrics(df, dataset)
        plot_bar_features(df, dataset)
        print(f"{dataset}: figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
