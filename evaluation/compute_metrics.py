"""Compute per-run scalar summary metrics from completed aurora-xcon runs.

Usage (from project root):
    python evaluation/compute_metrics.py                       # reward_type=final (default)
    python evaluation/compute_metrics.py --reward final_speed  # final_speed runs

Output: evaluation/metrics_summary_{reward_type}.csv
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.run_index import scan_runs, condition_label, SUBSET_LABELS

_PROJECT_ROOT = Path(__file__).parent.parent
_EVAL_DIR = Path(__file__).parent

# Kheperax QD offset: aurora_trainer applies sqrt(2)*100 per occupied cell
# to make qd_score positive (matches wandb display). metrics.csv stores raw
# fitnesses, so we add it back here. max_fitness has no offset in either place.
_KHEPERAX_OFFSET = float(np.sqrt(2) * 100)


def compute_run_metrics(run_dir: Path, reward_type: str = "final") -> dict:
    """Read metrics.csv for one complete run and return scalar metrics.

    Metrics:
        final_qd_score           Offset QD score at the last logged iteration
        best_fitness_reached     Max fitness ever observed across the run (raw scale)
        auc_qd_score             Trapezoidal integral of offset QD score over generation axis
        extinction_count         Total extinction events triggered
        mean_extinction_interval Mean generations between consecutive events (NaN if < 2)
        gen_best_fitness         Generation at which best_fitness_reached was first achieved
    """
    df = pd.read_csv(run_dir / "metrics.csv")

    # Older runs may lack archive_size column — infer from qd_score / mean_fitness.
    # By definition qd_score = mean_fitness * n_cells, so n_cells = qd_score / mean_fitness.
    if "archive_size" not in df.columns:
        nonzero = df["mean_fitness"].abs() > 1e-10
        arc = pd.Series(0, index=df.index, dtype=float)
        arc[nonzero] = (df.loc[nonzero, "qd_score"] / df.loc[nonzero, "mean_fitness"]).round()
        df["archive_size"] = arc.astype(int)

    # Apply kheperax offset for all reward types so QD scores align with wandb display.
    # (passive_qd_metrics always adds qd_offset * n_cells regardless of reward type.)
    offset_qd = df["qd_score"] + _KHEPERAX_OFFSET * df["archive_size"]

    final_qd_score = float(offset_qd.iloc[-1])
    best_fitness_reached = float(df["max_fitness"].max())
    # Use generation as x-axis: rows are at gens 5, 10, …, 2000 (logged every 5)
    auc_qd_score = float(np.trapz(offset_qd.values, df["generation"].values))

    # extinction_event is written as Python bool -> CSV stores "True"/"False" strings
    ext_col = df["extinction_event"]
    if ext_col.dtype == object:
        ext_bool = ext_col.map({"True": True, "False": False}).fillna(False)
    else:
        ext_bool = ext_col.astype(bool)

    extinction_count = int(ext_bool.sum())

    ext_gens = df.loc[ext_bool, "generation"].values
    mean_extinction_interval = (
        float(np.mean(np.diff(ext_gens))) if len(ext_gens) >= 2 else float("nan")
    )

    gen_best_fitness = int(df.loc[df["max_fitness"].idxmax(), "generation"])

    return {
        "final_qd_score": final_qd_score,
        "best_fitness_reached": best_fitness_reached,
        "auc_qd_score": auc_qd_score,
        "extinction_count": extinction_count,
        "mean_extinction_interval": mean_extinction_interval,
        "gen_best_fitness": gen_best_fitness,
    }


def main() -> None:
    reward_type = "final"
    subset: str | None = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--reward" and i + 1 < len(args):
            reward_type = args[i + 1]; i += 2
        elif a.startswith("--reward="):
            reward_type = a.split("=", 1)[1]; i += 1
        elif a == "--subset" and i + 1 < len(args):
            subset = args[i + 1]; i += 2
        elif a.startswith("--subset="):
            subset = a.split("=", 1)[1]; i += 1
        else:
            i += 1

    if subset is not None and subset not in SUBSET_LABELS:
        print(f"ERROR: unknown --subset '{subset}'. Choose from: {sorted(SUBSET_LABELS)}")
        sys.exit(1)

    allowed_labels: set[str] | None = set(SUBSET_LABELS[subset]) if subset else None

    # Include subset in output filename so different experiment CSVs don't overwrite each other.
    csv_tag = f"_{subset}" if subset else ""
    output_csv = _EVAL_DIR / f"metrics_summary_{reward_type}{csv_tag}.csv"

    runs = scan_runs(_PROJECT_ROOT)
    complete = [r for r in runs if r["is_complete"] and r.get("reward_type", "final") == reward_type]
    total_complete = sum(1 for r in runs if r["is_complete"])
    print(f"Found {len(complete)} complete {reward_type} run(s) (out of {total_complete} total complete).")
    if subset:
        print(f"Filtering to subset '{subset}': {SUBSET_LABELS[subset]}")
    print()

    if not complete:
        print(f"No complete runs with reward_type='{reward_type}' found.")
        return

    rows = []
    skipped = 0
    for run_info in complete:
        label = condition_label(run_info)
        if allowed_labels is not None and label not in allowed_labels:
            skipped += 1
            continue
        metrics = compute_run_metrics(run_info["run_dir"], reward_type)
        rows.append({
            "condition_label": label,
            "seed": run_info["seed"],
            **metrics,
        })
        print(
            f"  [{label:25s}] seed={run_info['seed']:3d}  "
            f"qd={metrics['final_qd_score']:10.1f}  "
            f"best_fit={metrics['best_fitness_reached']:7.3f}  "
            f"ext={metrics['extinction_count']:3d}"
        )

    summary = pd.DataFrame(rows, columns=[
        "condition_label", "seed",
        "final_qd_score", "best_fitness_reached", "auc_qd_score",
        "extinction_count", "mean_extinction_interval", "gen_best_fitness",
    ])
    if skipped:
        print(f"  (skipped {skipped} run(s) outside subset '{subset}')")
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_csv, index=False)
    print(f"\nSaved {len(summary)} rows to {output_csv}")


if __name__ == "__main__":
    main()
