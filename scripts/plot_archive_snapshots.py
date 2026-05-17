"""Grid of 2D scatter plots showing latent descriptors at key generations.

Rows = conditions (extinction triggers), columns = generation snapshots.
Reads ``archive_gen_*.npz`` files from ``run_dir/checkpoints/`` produced
when training with ``save_archive_checkpoints: true``.

Each cell is a scatter of occupied archive descriptors coloured by fitness,
with shared axis limits across all panels for direct visual comparison.

Usage (from project root)::

    python scripts/plot_archive_snapshots.py
    python scripts/plot_archive_snapshots.py --generations 100 500 1000 2000
    python scripts/plot_archive_snapshots.py --reward final_speed
    python scripts/plot_archive_snapshots.py --conditions adaptive_default static
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.plot_evolution import CONDITION_LABELS, CONDITION_ORDER
from evaluation.run_index import condition_label, scan_runs


def _apply_style() -> None:
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            return
        except OSError:
            continue


def _find_checkpoint(run_dir: Path, target_gen: int, tol: int = 50) -> Path | None:
    """Return the closest checkpoint within ±tol generations of target_gen."""
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    best_path: Path | None = None
    best_dist = float("inf")
    for p in ckpt_dir.glob("archive_gen_*.npz"):
        try:
            gen = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        dist = abs(gen - target_gen)
        if dist < best_dist:
            best_dist, best_path = dist, p
    return best_path if best_dist <= tol else None


def _ordered_conditions(groups: dict) -> list[str]:
    present = [c for c in CONDITION_ORDER if c in groups and groups[c]]
    extras = [c for c in sorted(groups) if c not in CONDITION_ORDER and groups[c]]
    return present + extras


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reward", default="final",
                   help="reward_type filter (default: final)")
    p.add_argument("--generations", nargs="+", type=int,
                   default=[100, 500, 1000, 2000],
                   help="generation snapshots to show (default: 100 500 1000 2000)")
    p.add_argument("--conditions", nargs="*", default=None,
                   help="optional subset of condition_label keys")
    p.add_argument("--output", type=Path,
                   default=_PROJECT_ROOT / "analysis_output",
                   help="output directory (default: analysis_output/)")
    p.add_argument("--tag", default=None,
                   help="optional filename suffix")
    args = p.parse_args()

    runs = scan_runs(_PROJECT_ROOT)
    complete = [
        r for r in runs
        if r["is_complete"] and r.get("reward_type", "final") == args.reward
    ]
    print(f"Found {len(complete)} complete run(s) with reward_type={args.reward!r}.")

    # Group by condition; keep all seeds but display only the lowest-seed one
    groups: dict[str, list[dict]] = {}
    for run_info in complete:
        lbl = condition_label(run_info)
        if args.conditions and lbl not in args.conditions:
            continue
        groups.setdefault(lbl, []).append(run_info)

    for lbl in groups:
        groups[lbl].sort(key=lambda r: r["seed"])

    conditions = _ordered_conditions(groups)
    if not conditions:
        print("No matching conditions found.")
        return

    generations = sorted(args.generations)
    n_rows, n_cols = len(conditions), len(generations)

    # Load checkpoints for every (condition, generation) cell
    data: dict[tuple[str, int], tuple[np.ndarray, np.ndarray] | None] = {}
    for cond in conditions:
        run_info = groups[cond][0]  # lowest-seed run
        for gen in generations:
            ckpt = _find_checkpoint(run_info["run_dir"], gen)
            if ckpt is None:
                data[(cond, gen)] = None
                continue
            d = np.load(str(ckpt))
            data[(cond, gen)] = (d["descriptors"], d["fitnesses"])

    cells_with_data = [v for v in data.values() if v is not None]
    if not cells_with_data:
        print(
            "No checkpoint files found. Re-run training with:\n"
            "  save_archive_checkpoints: true\n"
            "  archive_checkpoint_interval: 100"
        )
        return

    # Shared axis limits across all panels
    all_d0 = np.concatenate([v[0][:, 0] for v in cells_with_data])
    all_d1 = np.concatenate([v[0][:, 1] for v in cells_with_data])
    all_fit = np.concatenate([v[1] for v in cells_with_data])
    x_min, x_max = float(all_d0.min()), float(all_d0.max())
    y_min, y_max = float(all_d1.min()), float(all_d1.max())
    f_min, f_max = float(all_fit.min()), float(all_fit.max())

    _apply_style()
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols + 1.5, 3.0 * n_rows + 0.8),
        squeeze=False,
    )

    for j, gen in enumerate(generations):
        axes[0, j].set_title(f"Gen {gen}", fontsize=10, fontweight="bold")

    for i, cond in enumerate(conditions):
        seed = groups[cond][0]["seed"]
        axes[i, 0].set_ylabel(
            CONDITION_LABELS.get(cond, cond),
            fontsize=9,
            rotation=90,
            labelpad=4,
        )
        for j, gen in enumerate(generations):
            ax = axes[i, j]
            cell = data.get((cond, gen))
            if cell is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
            else:
                descs, fits = cell
                ax.scatter(
                    descs[:, 0], descs[:, 1],
                    c=fits,
                    cmap="viridis",
                    s=3,
                    alpha=0.4,
                    vmin=f_min,
                    vmax=f_max,
                    rasterized=True,
                )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([])
            ax.set_yticks([])

    # Single shared colorbar on the right
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(vmin=f_min, vmax=f_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, -1], shrink=0.8, pad=0.02)
    cbar.set_label("Fitness", fontsize=9)

    fig.suptitle(
        f"Archive Descriptor Snapshots  (reward={args.reward!r},  seed shown: lowest per condition)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    if args.conditions:
        joined = "+".join(args.conditions)
        suffix += f"_{joined}" if len(joined) <= 40 else f"_{joined[:37]}..."
    out_path = args.output / f"archive_snapshots{suffix}.png"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
