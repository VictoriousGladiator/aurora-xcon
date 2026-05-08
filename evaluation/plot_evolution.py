"""Plot per-generation evolution of metrics, averaged across seeds per condition.

For each condition, loads all seeds' metrics.csv files, aligns on the common
generation axis, and plots mean ± 1 std as a shaded band.

Usage (from project root):
    python evaluation/plot_evolution.py                       # reward_type=final
    python evaluation/plot_evolution.py --reward final_speed

Output: evaluation/figures/{reward_type}/evolution_{metric}.png
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from evaluation.run_index import scan_runs, condition_label

_PROJECT_ROOT = Path(__file__).parent.parent
_EVAL_DIR = Path(__file__).parent
_KHEPERAX_OFFSET = float(np.sqrt(2) * 100)

# Fixed condition order — same as plot_comparison for consistency
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
    "encoder_post",
    "enc_random_rate",
]

CONDITION_LABELS: dict[str, str] = {
    "no_extinction":     "No Ext.",
    "static":            "Static",
    "adaptive_default":  "Adaptive (default)",
    "adaptive_topk10":   "top-k=10%",
    "adaptive_topk30":   "top-k=30%",
    "adaptive_patience5":  "patience=5",
    "adaptive_patience20": "patience=20",
    "adaptive_alpha01":  "α=0.1",
    "adaptive_alpha03":  "α=0.3",
    "encoder_post":      "Post-Encoder",
    "enc_random_rate":   "Enc. Rate (random)",
}

# Metrics to plot: (csv_column, y-label, apply_offset_for_final)
METRIC_CONFIGS = [
    ("qd_score",          "QD Score"),
    ("max_fitness",       "Max Fitness"),
    ("mean_fitness_top_k","Mean Fitness (top-k%)"),
]


def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _load_series(run_dir: Path, column: str, reward_type: str = "final") -> pd.Series | None:
    """Load one column from metrics.csv, apply offset if needed, index by generation."""
    try:
        df = pd.read_csv(run_dir / "metrics.csv")
    except Exception:
        return None
    if column not in df.columns:
        return None

    series = df.set_index("generation")[column]

    if column == "qd_score":
        # Apply offset for all reward types to align with wandb display.
        if "archive_size" not in df.columns:
            nonzero = df["mean_fitness"].abs() > 1e-10
            arc = pd.Series(0.0, index=df.index)
            arc[nonzero] = (df.loc[nonzero, "qd_score"] / df.loc[nonzero, "mean_fitness"]).round()
            df["archive_size"] = arc.astype(int)
        offset_series = df["qd_score"] + _KHEPERAX_OFFSET * df["archive_size"]
        series = pd.Series(offset_series.values, index=df["generation"].values)

    return series


def _plot_metric(
    groups: dict[str, list[pd.Series]],
    column: str,
    ylabel: str,
    figures_dir: Path,
    suffix: str = "",
) -> None:
    present = [c for c in CONDITION_ORDER if c in groups and groups[c]]
    # Also include any conditions not in CONDITION_ORDER (e.g. new modes)
    extras = [c for c in sorted(groups) if c not in CONDITION_ORDER and groups[c]]
    all_conditions = present + extras

    if not all_conditions:
        print(f"  No data for {column}, skipping.")
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = cm.tab10(np.linspace(0, 1, len(all_conditions)))

    for cond, color in zip(all_conditions, colors):
        series_list = groups[cond]
        if not series_list:
            continue

        # Align all seeds on a common generation axis using pandas concat
        combined = pd.concat(
            [s.rename(i) for i, s in enumerate(series_list)], axis=1
        ).sort_index()
        gens = combined.index.values
        matrix = combined.values  # shape: (n_gens, n_seeds)

        label = CONDITION_LABELS.get(cond, cond)

        # Thin transparent lines per seed — variance visible without ever crossing
        # physical bounds (no seed can exceed its own max, unlike mean±std bands)
        for col_i in range(matrix.shape[1]):
            ax.plot(gens, matrix[:, col_i], color=color, linewidth=0.7, alpha=0.25)

        # Bold mean line on top
        mean = np.nanmean(matrix, axis=1)
        ax.plot(gens, mean, label=label, color=color, linewidth=2.0)

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    n_seeds_max = max(len(v) for v in groups.values() if v)
    ax.set_title(
        f"{ylabel} over generations  (thin=individual seeds, bold=mean, n≤{n_seeds_max})",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best", ncol=2)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / f"evolution_{column}{suffix}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _parse_args() -> tuple[str, list[str] | None, str | None]:
    args = sys.argv[1:]
    reward_type = "final"
    conditions: list[str] | None = None
    tag: str | None = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--reward" and i + 1 < len(args):
            reward_type = args[i + 1]; i += 2
        elif a.startswith("--reward="):
            reward_type = a.split("=", 1)[1]; i += 1
        elif a == "--tag" and i + 1 < len(args):
            tag = args[i + 1]; i += 2
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]; i += 1
        elif a == "--conditions":
            conditions = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                conditions.append(args[i]); i += 1
        else:
            i += 1
    return reward_type, conditions, tag


def _file_tag(condition_filter: list[str] | None, tag: str | None) -> str:
    if tag:
        return f"_{tag}"
    if condition_filter:
        joined = "+".join(condition_filter)
        return f"_{joined}" if len(joined) <= 60 else f"_{joined[:57]}..."
    return ""


def main() -> None:
    reward_type, condition_filter, tag = _parse_args()
    suffix = _file_tag(condition_filter, tag)
    figures_dir = _EVAL_DIR / "figures" / reward_type

    runs = scan_runs(_PROJECT_ROOT)
    complete = [r for r in runs if r["is_complete"] and r.get("reward_type", "final") == reward_type]
    print(f"Found {len(complete)} complete {reward_type} run(s).")

    if not complete:
        print(f"No complete runs with reward_type='{reward_type}'.")
        return

    all_labels = sorted({condition_label(r) for r in complete})
    print(f"All conditions in data: {all_labels}")
    if condition_filter:
        print(f"Plotting subset: {condition_filter}")
    print()

    for column, ylabel in METRIC_CONFIGS:
        print(f"Building evolution plot: {column}")
        groups: dict[str, list[pd.Series]] = {}
        for run_info in complete:
            lbl = condition_label(run_info)
            if condition_filter and lbl not in condition_filter:
                continue
            series = _load_series(run_info["run_dir"], column)
            if series is not None:
                groups.setdefault(lbl, []).append(series)

        _plot_metric(groups, column, ylabel, figures_dir, suffix)

    print(f"\nAll evolution plots saved to {figures_dir}")


if __name__ == "__main__":
    main()
