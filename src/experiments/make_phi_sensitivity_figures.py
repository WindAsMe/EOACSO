"""Generate phi-sensitivity figures from results/tables/{dataset}_phi_sensitivity_results.csv.

For each dataset (oxford, naranjo), plots mean +/- std of best fitness, balanced
accuracy, and selected feature ratio against CSO's social-term coefficient phi
(Cheng & Jin 2015, Eq. 25-26; see `src/optimizers/cso.py::cso_phi`).

Run with: python -m src.experiments.make_phi_sensitivity_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = ["oxford", "naranjo"]
TABLES_DIR = Path("results/tables")
FIG_DIR = Path("results/figures")

CSO_COLOR = "#e34948"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
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
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    }
)

METRICS = [
    ("best_fitness", "Mean best fitness", "fitness"),
    ("balanced_accuracy", "Mean balanced accuracy", "accuracy"),
    ("feature_ratio", "Mean selected feature ratio", "features"),
]


def load(dataset):
    df = pd.read_csv(TABLES_DIR / f"{dataset}_phi_sensitivity_results.csv")
    return df.groupby("phi")[["best_fitness", "balanced_accuracy", "feature_ratio"]].agg(["mean", "std"])


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def plot_metric(g, dataset, column, ylabel, suffix):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    phis = g.index.to_numpy()
    mean = g[(column, "mean")].to_numpy()
    std = g[(column, "std")].to_numpy()
    ax.errorbar(
        phis, mean, yerr=std, color=CSO_COLOR, linewidth=2.0, marker="o", markersize=5,
        markerfacecolor=CSO_COLOR, markeredgecolor=INK, markeredgewidth=0.5,
        capsize=3, ecolor=INK_SECONDARY, elinewidth=0.8,
    )
    best_idx = mean.argmin() if column != "balanced_accuracy" else mean.argmax()
    ax.scatter([phis[best_idx]], [mean[best_idx]], s=90, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=5)
    ax.set_xlabel(r"$\varphi$")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, f"phi_sensitivity_{suffix}_{dataset}")


def main():
    for dataset in DATASETS:
        g = load(dataset)
        for column, ylabel, suffix in METRICS:
            plot_metric(g, dataset, column, ylabel, suffix)
        print(f"{dataset}: figures written to {FIG_DIR}/")


if __name__ == "__main__":
    main()
