"""Generate comparison box-plots from multi-seed sweep metrics.

Usage (from project root):
    python evaluation/plot_comparison.py                       # reward_type=final (default)
    python evaluation/plot_comparison.py --reward final_speed  # final_speed runs

Input:  evaluation/metrics_summary_{reward_type}.csv  (produced by compute_metrics.py)
Output: evaluation/figures/{reward_type}/comparison_{metric}.pdf
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_EVAL_DIR = Path(__file__).parent

# Fixed x-axis order — consistent across all three plots
CONDITION_ORDER = [
    "no_extinction",
    "static",
    "adaptive_default",
    "adaptive_topk10",
    "adaptive_topk30",
    "adaptive_patience5",
    "adaptive_patience20",
    "adaptive_alpha01",
    "adaptive_alpha03",
]

CONDITION_LABELS: dict[str, str] = {
    "no_extinction":     "No Ext.",
    "static":            "Static",
    "adaptive_default":  "Adaptive\n(default)",
    "adaptive_topk10":   "top-k\n=10%",
    "adaptive_topk30":   "top-k\n=30%",
    "adaptive_patience5":  "patience\n=5",
    "adaptive_patience20": "patience\n=20",
    "adaptive_alpha01":  "α=0.1",
    "adaptive_alpha03":  "α=0.3",
}

METRIC_CONFIGS = [
    ("final_qd_score",       "Final QD Score"),
    ("best_fitness_reached", "Best Fitness Reached"),
    ("auc_qd_score",         "AUC of QD Score"),
    ("gen_best_fitness",     "Generation of Best Fitness"),
]

_RNG = np.random.default_rng(0)


def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, figures_dir: Path) -> None:
    present = [c for c in CONDITION_ORDER if c in df["condition_label"].values]
    n = len(present)

    data_per_condition = [
        df.loc[df["condition_label"] == cond, metric].dropna().values
        for cond in present
    ]

    _apply_style()
    fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 5))

    x_positions = list(range(n))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ax.boxplot(
            data_per_condition,
            positions=x_positions,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightsteelblue", alpha=0.7),
            medianprops=dict(color="navy", linewidth=2),
            whiskerprops=dict(color="steelblue"),
            capprops=dict(color="steelblue"),
            flierprops=dict(marker="", linestyle="none"),
        )

    for xi, vals in zip(x_positions, data_per_condition):
        if vals.size == 0:
            continue
        jitter = _RNG.uniform(-0.12, 0.12, size=vals.size)
        ax.scatter(
            xi + jitter, vals,
            color="steelblue", alpha=0.85,
            s=30, zorder=5,
            edgecolors="white", linewidths=0.5,
        )

    n_seeds = df["seed"].nunique()
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in present],
        fontsize=9,
    )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(
        f"{ylabel} by condition  (n≤{n_seeds} seeds per box)",
        fontsize=12,
    )
    ax.set_xlim(-0.6, n - 0.4)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"comparison_{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    reward_type = "final"
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--reward" and i + 1 < len(args):
            reward_type = args[i + 1]
        elif a.startswith("--reward="):
            reward_type = a.split("=", 1)[1]

    summary_csv = _EVAL_DIR / f"metrics_summary_{reward_type}.csv"
    figures_dir = _EVAL_DIR / "figures" / reward_type

    if not summary_csv.exists():
        raise FileNotFoundError(
            f"{summary_csv} not found. "
            f"Run `python evaluation/compute_metrics.py --reward {reward_type}` first."
        )
    df = pd.read_csv(summary_csv)
    print(f"Loaded {len(df)} rows from {summary_csv}")
    print(f"Conditions in data: {sorted(df['condition_label'].unique())}\n")

    for metric, ylabel in METRIC_CONFIGS:
        _plot_metric(df, metric, ylabel, figures_dir)

    print(f"\nAll plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
