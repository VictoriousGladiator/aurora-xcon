"""Run AURORA-XCon adaptive-extinction experiments sequentially.

Each entry in EXPERIMENTS is a dict of Hydra overrides.
Edit the list below to change the parameter sweep.

Usage:
    python scripts/mean_fitness_single_exp.py
    python scripts/mean_fitness_single_exp.py --dry-run   # print commands only
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Base command — shared across all experiments
# ---------------------------------------------------------------------------
BASE_CMD = [
    sys.executable, "-m", "main", "main", "aurora",
    "env=kheperax",
    "loss_type=triplet",
    "extinction_mode=adaptive",
]

# ---------------------------------------------------------------------------
# Experiment grid — edit this list
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # --- vary top_k_percent ---
    dict(seed=20, top_k_percent=10,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(seed=20, top_k_percent=20,   # <-- default
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(seed=20, top_k_percent=30,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),

    # --- vary patience ---
    dict(seed=20, top_k_percent=20,
         **{"adaptive_extinction.patience": 5,  "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(seed=20, top_k_percent=20,
         **{"adaptive_extinction.patience": 20, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),

    # --- vary alpha ---
    dict(seed=20, top_k_percent=20,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.1, "adaptive_extinction.cooldown": 10}),
    dict(seed=20, top_k_percent=20,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.3, "adaptive_extinction.cooldown": 10}),

    # --- baselines for comparison ---
    dict(seed=20, top_k_percent=20, extinction_mode="off"),    # no extinction
    dict(seed=20, top_k_percent=20, extinction_mode="static",
         extinction_freq=10, remaining_prop=0.05),              # fixed-period
]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_cmd(overrides: dict) -> list[str]:
    cmd = list(BASE_CMD)
    for k, v in overrides.items():
        # extinction_mode override replaces the base command's value
        if k == "extinction_mode":
            # replace in base or append
            try:
                idx = next(i for i, a in enumerate(cmd) if a.startswith("extinction_mode="))
                cmd[idx] = f"extinction_mode={v}"
            except StopIteration:
                cmd.append(f"extinction_mode={v}")
        else:
            cmd.append(f"{k}={v}")
    return cmd


def _latest_run_dir(project_root: Path) -> Path | None:
    """Return the most recently modified runs/* directory under output/."""
    candidates = list((project_root / "output").glob("*/*/runs/*/"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run(overrides: dict, dry_run: bool = False) -> int:
    cmd = _build_cmd(overrides)
    print("$ " + " ".join(cmd))
    if dry_run:
        return 0

    project_root = Path(__file__).parent.parent

    # wandb creates temp image files in TMPDIR then os.rename()s them to the
    # wandb run dir. Both must be on the same device or rename fails (EXDEV).
    # Since the project lives on /mnt/c/ (Windows NTFS via WSL), put TMPDIR
    # there too so the rename stays within the same mount point.
    env = os.environ.copy()
    tmp = project_root / ".tmp"
    tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp)

    result = subprocess.run(cmd, env=env, cwd=str(project_root))
    if result.returncode != 0:
        print(f"  WARNING: training exited with code {result.returncode}")

    run_dir = _latest_run_dir(project_root)
    if run_dir is not None:
        print(f"  Plotting {run_dir} ...")
        plot_cmd = [sys.executable, str(project_root / "scripts" / "plot_metrics.py"), str(run_dir)]
        subprocess.run(plot_cmd, cwd=str(project_root))
    else:
        print("  WARNING: no run dir found under output/ — skipping plot")

    return result.returncode


def main():
    dry_run = "--dry-run" in sys.argv
    n = len(EXPERIMENTS)
    print(f"Running {n} experiments {'(dry run)' if dry_run else 'sequentially'}.\n")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n=== [{i}/{n}] ===")
        _run(exp, dry_run=dry_run)
    print("\nDone.")


if __name__ == "__main__":
    main()
