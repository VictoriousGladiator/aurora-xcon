"""Coverage and occupancy entropy evolution by extinction trigger (condition).

Reads ``metrics.csv`` columns written when ``log_answer_set_geometry: true``:
  ``archive_coverage``  — fraction of archive cells occupied
  ``archive_occupancy_entropy``  — Shannon entropy of a 20×20 binned histogram
                                    of the occupied descriptors (nats)

Both metrics are encoder-rescaling invariant: coverage counts filled cells
regardless of latent scale, and entropy measures uniformity of spread within
the current latent space at each generation.

Runs missing these columns are silently skipped (old runs won't have them).

Usage (from project root)::

    python scripts/plot_coverage_entropy.py
    python scripts/plot_coverage_entropy.py --reward final_speed
    python scripts/plot_coverage_entropy.py --conditions adaptive_default static
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.plot_evolution import CONDITION_LABELS, CONDITION_ORDER
from evaluation.run_index import condition_label, scan_runs

# Theoretical maximum entropy: uniform over 20×20 = 400 bins → log(400)
_ENTROPY_UNIFORM_MAX = float(np.log(20 * 20))

PANELS: list[tuple[str, str, bool]] = [
    ("archive_coverage", "Coverage (fraction of archive cells)", True),
    ("archive_occupancy_entropy", "Occupancy Entropy (nats)", False),
]


def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _load_series(run_dir: Path, column: str) -> pd.Series | None:
    try:
        df = pd.read_csv(run_dir / "metrics.csv")
    except Exception:
        return None
    if column not in df.columns or "generation" not in df.columns:
        return None
    return df.set_index("generation")[column]


def _ordered_conditions(groups: dict) -> list[str]:
    present = [c for c in CONDITION_ORDER if c in groups and groups[c]]
    extras = [c for c in sorted(groups) if c not in CONDITION_ORDER and groups[c]]
    return present + extras


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reward", default="final",
                   help="reward_type filter (default: final)")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "analysis_output",
                   help="output directory (default: analysis_output/)")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="optional subset of condition_label keys")
    p.add_argument("--tag", default=None,
                   help="optional filename suffix")
    args = p.parse_args()

    runs = scan_runs(_PROJECT_ROOT)
    complete = [
        r for r in runs
        if r["is_complete"] and r.get("reward_type", "final") == args.reward
    ]
    print(f"Found {len(complete)} complete run(s) with reward_type={args.reward!r}.")
    if not complete:
        print("Nothing to plot.")
        return

    by_column: dict[str, dict[str, list[pd.Series]]] = {col: {} for col, _, _ in PANELS}
    for run_info in complete:
        lbl = condition_label(run_info)
        if args.conditions and lbl not in args.conditions:
            continue
        for col, _, _ in PANELS:
            s = _load_series(run_info["run_dir"], col)
            if s is not None:
                by_column[col].setdefault(lbl, []).append(s)

    if not any(by_column[col] for col, _, _ in PANELS):
        print(
            "No runs have archive_coverage / archive_occupancy_entropy columns.\n"
            "These are written by new runs with log_answer_set_geometry: true.\n"
            "Existing runs do not have them."
        )
        return

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (col, ylabel, is_coverage) in zip(axes, PANELS):
        g = by_column.get(col, {})
        all_c = _ordered_conditions(g)
        if not all_c:
            ax.set_title(ylabel)
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        colors = cm.tab10(np.linspace(0, 1, max(len(all_c), 1)))
        for cond, color in zip(all_c, colors):
            series_list = g[cond]
            if not series_list:
                continue
            combined = pd.concat(
                [s.rename(i) for i, s in enumerate(series_list)], axis=1
            ).sort_index()
            gens = combined.index.values
            matrix = combined.values
            label = CONDITION_LABELS.get(cond, cond)
            for j in range(matrix.shape[1]):
                ax.plot(gens, matrix[:, j], color=color, linewidth=0.7, alpha=0.25)
            mean = np.nanmean(matrix, axis=1)
            ax.plot(gens, mean, label=label, color=color, linewidth=2.0)

        ax.set_xlabel("Generation")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

        if is_coverage:
            ax.set_ylim(0.0, 1.05)
        else:
            ax.axhline(
                _ENTROPY_UNIFORM_MAX,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
                label=f"uniform ({_ENTROPY_UNIFORM_MAX:.2f})",
            )

        ax.legend(fontsize=7, loc="best", ncol=2)

    fig.suptitle("Archive Coverage & Entropy Evolution by Trigger Type",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    if args.conditions:
        joined = "+".join(args.conditions)
        suffix += f"_{joined}" if len(joined) <= 40 else f"_{joined[:37]}..."
    out_path = args.output / f"coverage_entropy{suffix}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
