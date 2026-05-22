"""Run AURORA-XCon adaptive-extinction experiments sequentially or in parallel.

wandb runs offline and logs are silenced by default for speed.
Use --verbose to restore full subprocess output (sequential only).

Each run gets an explicit, deterministic Hydra output directory
  output/<reward>_<condition>_seed<N>/
so parallel workers never collide on the same timestamp-based folder.
stdout/stderr are always captured to  <run_output>/stdout.txt  and  stderr.txt.

Usage:
    python scripts/run_sweep.py                      # sequential, quiet
    python scripts/run_sweep.py --workers 4          # 4 parallel workers
    python scripts/run_sweep.py --verbose            # show full subprocess output
    python scripts/run_sweep.py --check              # completion summary, then exit
    python scripts/run_sweep.py --dry-run            # print commands only
    python scripts/run_sweep.py --reward final_speed # use speed-bonus fitness
    python scripts/run_sweep.py --reward final_speed --speed-factor 200
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Base command
# ---------------------------------------------------------------------------
_REWARD_TYPES = ("final", "final_speed")
_DEFAULT_REWARD = "final"

BASE_CMD = [
    sys.executable, "-m", "main", "main", "aurora",
    "env=kheperax",
    "loss_type=triplet",
    "extinction_mode=fitness_trigger",
    "wandb.mode=offline",
    "hydra.verbose=false",
]

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------
CONDITIONS = [
    dict(top_k_percent=10,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=30,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20,
         **{"adaptive_extinction.patience": 5,  "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20,
         **{"adaptive_extinction.patience": 20, "adaptive_extinction.alpha": 0.2, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.1, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20,
         **{"adaptive_extinction.patience": 10, "adaptive_extinction.alpha": 0.3, "adaptive_extinction.cooldown": 10}),
    dict(top_k_percent=20, extinction_mode="off"),
    dict(top_k_percent=20, extinction_mode="static", extinction_freq=10, remaining_prop=0.05),
]

# Separate subset for encoder-coupled extinction experiments
ENCODER_CONDITIONS = [
    dict(top_k_percent=20, extinction_mode="encoder_post"),
    dict(top_k_percent=20, extinction_mode="random_encoder_rate"),
]

# New trigger experiments: d_min volatility trigger and static ramped proportion
NEW_TRIGGER_CONDITIONS = [
    # d_min volatility trigger — fixed prop=0.05
    dict(top_k_percent=20, extinction_mode="d_min_trigger",
         **{"adaptive_extinction.d_min_patience": 10,
            "adaptive_extinction.d_min_alpha": 0.2,
            "adaptive_extinction.d_min_cooldown": 10}),
    # Periodic trigger with survival fraction ramped from 5% to 40% over training
    dict(top_k_percent=20, extinction_mode="static_ramped_proportion",
         extinction_freq=10,
         **{"adaptive_extinction.ramped_prop_min": 0.05,
            "adaptive_extinction.ramped_prop_max": 0.40}),
]

COUNT_CONTROL_CONDITIONS = [
    dict(extinction_mode="random_fixed_count", target_extinction_count=12, top_k_percent=20),
    dict(extinction_mode="random_fixed_count", target_extinction_count=15, top_k_percent=20),
    dict(top_k_percent=20, extinction_mode="encoder_pre"),
    dict(top_k_percent=20, extinction_mode="static", extinction_freq=133, remaining_prop=0.05),
    dict(top_k_percent=20, extinction_mode="static", extinction_freq=167, remaining_prop=0.05),
]

SEEDS = [20, 42, 7, 13, 99]
EXPERIMENTS = [dict(seed=s, **c) for c in COUNT_CONTROL_CONDITIONS for s in SEEDS]

# ---------------------------------------------------------------------------
# Defaults matching aurora.yaml
# ---------------------------------------------------------------------------
_DEFAULTS: dict = {
    "extinction_mode": "fitness_trigger",
    "top_k_percent": 20,
    "adaptive_extinction.patience": 10,
    "adaptive_extinction.alpha": 0.2,
    "adaptive_extinction.cooldown": 10,
    "extinction_freq": 10,
    "remaining_prop": 0.05,
}

_PERF_ENV: dict[str, str] = {
    "WANDB_MODE": "offline",
    "WANDB_SILENT": "true",
    "JAX_LOG_COMPILES": "0",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}

_print_lock = Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)

# ---------------------------------------------------------------------------
# Unique output directory key — one deterministic path per (experiment, reward)
# ---------------------------------------------------------------------------

# Declares which extra parameters to embed in the output dir name for each mode.
# Each entry: extinction_mode -> [(exp_dict_key, label_prefix), ...]
#   - label_prefix + value are joined with no separator (e.g. "freq" + "133" -> "freq133")
#   - empty label_prefix means just append the value directly
#   - modes not listed here fall back to mode name only (or fitness_trigger logic)
# To add a new mode: just add an entry here — no if/elif needed.
_MODE_VARIANT_PARAMS: dict[str, list[tuple[str, str]]] = {
    "off":                       [],
    "static":                    [("extinction_freq",              "freq")],
    "static_ramped_proportion":  [("extinction_freq",              "freq")],
    "random_fixed_count":        [("target_extinction_count",      "")],
    "encoder_post":              [],
    "encoder_pre":               [],
    "random_encoder_rate":       [],
    "d_min_trigger":             [
        ("adaptive_extinction.d_min_patience", "dp"),
        ("adaptive_extinction.d_min_alpha",    "da"),
    ],
}


def _experiment_key(exp: dict, reward_type: str) -> str:
    """Return a filesystem-safe, unique key for this (experiment, reward_type) pair.

    Used as the Hydra run dir so parallel workers never share a timestamp folder.
    The condition segment is built from _MODE_VARIANT_PARAMS so that adding a new
    mode only requires a dict entry above — no if/elif chain needed.

    Examples:
        static, freq=133         -> 'final_static_freq133_seed20'
        random_fixed_count, n=12 -> 'final_random_fixed_count_12_seed20'
        encoder_pre              -> 'final_encoder_pre_seed20'
        fitness_trigger, topk=10 -> 'final_fitness_trigger_topk10_seed20'
    """
    mode = exp.get("extinction_mode", "fitness_trigger")
    seed = exp.get("seed", 0)

    if mode == "off":
        cond = "no_extinction"

    elif mode in _MODE_VARIANT_PARAMS:
        parts: list[str] = [mode]
        for exp_key, label in _MODE_VARIANT_PARAMS[mode]:
            val = exp.get(exp_key, "?")
            if isinstance(val, float):
                val = f"{val:.1f}".replace(".", "")
            parts.append(f"{label}{val}")
        cond = "_".join(parts)

    elif mode == "fitness_trigger":
        topk      = int(exp.get("top_k_percent", 20))
        pat       = int(exp.get("adaptive_extinction.patience", 10))
        alpha_str = f"{float(exp.get('adaptive_extinction.alpha', 0.2)):.1f}".replace(".", "")
        if topk != 20:
            cond = f"fitness_trigger_topk{topk}"
        elif pat != 10:
            cond = f"fitness_trigger_patience{pat}"
        elif alpha_str != "02":
            cond = f"fitness_trigger_alpha{alpha_str}"
        else:
            cond = "fitness_trigger_default"

    else:
        # Unknown / future mode — use mode name verbatim so it still gets a sane dir
        cond = mode

    return f"{reward_type}_{cond}_seed{seed}"

# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------

def _params_match(exp_dict: dict, cfg) -> bool:
    for key, exp_val in exp_dict.items():
        if "." in key:
            parts = key.split(".", 1)
            sub = cfg.get(parts[0], {}) or {}
            cfg_val = sub.get(parts[1], _DEFAULTS.get(key))
        else:
            cfg_val = cfg.get(key, _DEFAULTS.get(key))
        if isinstance(exp_val, float) or isinstance(cfg_val, float):
            try:
                if abs(float(cfg_val) - float(exp_val)) > 1e-5:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if str(cfg_val) != str(exp_val):
                return False
    return True


def _check_run_dir(run_dir: Path, exp_dict: dict, reward_type: str) -> bool:
    """Return True if run_dir contains a matching, complete run."""
    import pandas as pd
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return False
    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception:
        return False
    if str(cfg.get("reward_type", "final")) != reward_type:
        return False
    if not _params_match(exp_dict, cfg):
        return False
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return False
    try:
        df = pd.read_csv(metrics)
        return not df.empty and int(df["generation"].iloc[-1]) == 2000
    except Exception:
        return False


def _find_complete_run(project_root: Path, exp_dict: dict, reward_type: str = "final") -> Path | None:
    # Fast path: check the deterministic location first (new-style runs)
    key = _experiment_key(exp_dict, reward_type)
    known_runs = project_root / "output" / key / "runs"
    if known_runs.exists():
        for run_dir in known_runs.iterdir():
            if run_dir.is_dir() and _check_run_dir(run_dir, exp_dict, reward_type):
                return run_dir

    # Fallback: scan old-style date/time output dirs
    for cfg_path in (project_root / "output").rglob("runs/*/config.yaml"):
        run_dir = cfg_path.parent
        # Skip dirs already checked via fast path
        if run_dir.parts[-3] == key:
            continue
        if _check_run_dir(run_dir, exp_dict, reward_type):
            return run_dir

    return None

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _build_cmd(overrides: dict, extra: list[str] | None = None) -> list[str]:
    cmd = list(BASE_CMD)
    for k, v in overrides.items():
        if k == "extinction_mode":
            try:
                idx = next(i for i, a in enumerate(cmd) if a.startswith("extinction_mode="))
                cmd[idx] = f"extinction_mode={v}"
            except StopIteration:
                cmd.append(f"extinction_mode={v}")
        else:
            cmd.append(f"{k}={v}")
    if extra:
        cmd.extend(extra)
    return cmd


def _run_one(
    exp: dict,
    project_root: Path,
    verbose: bool,
    n_workers: int,
    reward_overrides: list[str] | None = None,
) -> int:
    reward_type = next(
        (r.split("=", 1)[1] for r in (reward_overrides or []) if r.startswith("env.reward_type=")),
        "final",
    )
    key = _experiment_key(exp, reward_type)
    hydra_dir = project_root / "output" / key
    hydra_dir.mkdir(parents=True, exist_ok=True)

    # Explicit Hydra run dir — prevents timestamp collisions between parallel workers
    extra = list(reward_overrides or []) + [f"hydra.run.dir={hydra_dir}"]
    cmd = _build_cmd(exp, extra)

    env = os.environ.copy()
    env.update(_PERF_ENV)
    if n_workers > 1:
        per_worker = max(1, (os.cpu_count() or 1) // n_workers)
        env["OMP_NUM_THREADS"] = str(per_worker)
        env["MKL_NUM_THREADS"] = str(per_worker)
    tmp = project_root / ".tmp"
    tmp.mkdir(exist_ok=True)
    env["TMPDIR"] = str(tmp)

    label = _short_label(exp)
    launch = time.time()

    result = subprocess.run(
        cmd, env=env, cwd=str(project_root),
        capture_output=True, text=True,
    )

    # Always save logs — they're the debugging lifeline for quiet parallel runs
    (hydra_dir / "stdout.txt").write_text(result.stdout)
    (hydra_dir / "stderr.txt").write_text(result.stderr)

    elapsed = time.time() - launch
    rc = result.returncode
    status = "DONE" if rc == 0 else f"ERROR(rc={rc})"
    _log(f"  [{status}] {label}  ({elapsed / 60:.1f} min)")

    if rc != 0:
        # Print last 20 lines of stderr so the error is visible without digging
        tail = "\n".join(result.stderr.splitlines()[-20:])
        _log(f"  --- stderr tail ({key}) ---\n{tail}\n  ---")

    if verbose and rc == 0:
        print(result.stdout)

    # Plot — run dir is inside the known hydra_dir, no guessing needed
    run_dirs = sorted((hydra_dir / "runs").glob("*/"), key=lambda p: p.stat().st_mtime) \
        if (hydra_dir / "runs").exists() else []
    if run_dirs:
        run_dir = run_dirs[-1]
        plot_result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "plot_metrics.py"), str(run_dir)],
            cwd=str(project_root), capture_output=True, text=True,
        )
        if plot_result.returncode != 0:
            _log(f"  WARNING: plot failed for {key}")
    else:
        _log(f"  WARNING: no run dir found inside {hydra_dir / 'runs'}, skipping plot")

    return rc


def _short_label(exp: dict) -> str:
    mode  = exp.get("extinction_mode", "fitness_trigger")
    topk  = exp.get("top_k_percent", 20)
    seed  = exp.get("seed", "?")
    pat   = exp.get("adaptive_extinction.patience", 10)
    alpha = exp.get("adaptive_extinction.alpha", 0.2)
    if mode == "d_min_trigger":
        dp = exp.get("adaptive_extinction.d_min_patience", 10)
        da = exp.get("adaptive_extinction.d_min_alpha", 0.2)
        return f"seed={seed} mode=d_min_trigger dp={dp} dα={da}"
    if mode == "static_ramped_proportion":
        freq = exp.get("extinction_freq", 10)
        return f"seed={seed} mode=ramped_prop freq={freq}"
    return f"seed={seed} mode={mode} topk={topk} p={pat} α={alpha}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> tuple[bool, bool, bool, int, str, float, str]:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    check   = "--check"   in args
    verbose = "--verbose" in args
    workers = 1
    reward_type = _DEFAULT_REWARD
    speed_bonus_factor = 1.0
    subset = "count_control"   # default: run only COUNT_CONTROL_CONDITIONS
    for i, a in enumerate(args):
        if a == "--workers" and i + 1 < len(args):
            try:
                workers = int(args[i + 1])
            except ValueError:
                pass
        elif a.startswith("--workers="):
            try:
                workers = int(a.split("=", 1)[1])
            except ValueError:
                pass
        elif a == "--reward" and i + 1 < len(args):
            reward_type = args[i + 1]
        elif a.startswith("--reward="):
            reward_type = a.split("=", 1)[1]
        elif a == "--speed-factor" and i + 1 < len(args):
            try:
                speed_bonus_factor = float(args[i + 1])
            except ValueError:
                pass
        elif a.startswith("--speed-factor="):
            try:
                speed_bonus_factor = float(a.split("=", 1)[1])
            except ValueError:
                pass
        elif a == "--subset" and i + 1 < len(args):
            subset = args[i + 1]
        elif a.startswith("--subset="):
            subset = a.split("=", 1)[1]
    if reward_type not in _REWARD_TYPES:
        print(f"ERROR: unknown --reward '{reward_type}'. Choose from: {_REWARD_TYPES}")
        sys.exit(1)
    if subset not in ("main", "encoder", "new_triggers", "count_control"):
        print(f"ERROR: unknown --subset '{subset}'. Choose from: main, encoder, new_triggers")
        sys.exit(1)
    return dry_run, check, verbose, workers, reward_type, speed_bonus_factor, subset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    dry_run, check_mode, verbose, n_workers, reward_type, speed_bonus_factor, subset = _parse_args()
    project_root = Path(__file__).parent.parent

    if subset == "encoder":
        conditions = ENCODER_CONDITIONS
    elif subset == "new_triggers":
        conditions = NEW_TRIGGER_CONDITIONS
    elif subset == "count_control":
        conditions = COUNT_CONTROL_CONDITIONS
    else:
        conditions = CONDITIONS
    experiments = [dict(seed=s, **c) for c in conditions for s in SEEDS]
    n = len(experiments)

    reward_overrides = [f"env.reward_type={reward_type}"]
    if reward_type == "final_speed":
        reward_overrides.append(f"env.speed_bonus_factor={speed_bonus_factor}")

    if check_mode:
        print(f"Checking {n} (condition, seed) pairs  [reward={reward_type}]...\n")
        complete = 0
        for i, exp in enumerate(experiments, 1):
            run_dir = _find_complete_run(project_root, exp, reward_type)
            label   = _short_label(exp)
            status  = f"[DONE]    {run_dir}" if run_dir else "[MISSING]"
            print(f"  [{i:3d}/{n}] {label:60s} {status}")
            if run_dir:
                complete += 1
        print(f"\n{complete}/{n} complete.")
        return

    if dry_run:
        print(f"Dry run — {n} commands  [subset={subset}, reward={reward_type}]:\n")
        # subset=new_triggers: 2 conditions × 5 seeds = 10 runs
        for exp in experiments:
            key = _experiment_key(exp, reward_type)
            extra = list(reward_overrides) + [f"hydra.run.dir=output/{key}"]
            print("$ " + " ".join(_build_cmd(exp, extra)))
        return

    print("Scanning for completed runs...", flush=True)
    pending = [exp for exp in experiments if _find_complete_run(project_root, exp, reward_type) is None]
    n_skip  = n - len(pending)
    if n_skip:
        print(f"Skipping {n_skip} already-complete run(s).")
    if not pending:
        print("All experiments complete.")
        return

    parallel = n_workers > 1
    mode_str = f"parallel ×{n_workers}" if parallel else "sequential"
    print(f"Running {len(pending)} experiment(s)  [subset={subset}, {mode_str}, reward={reward_type}, wandb=offline].\n")

    if parallel:
        completed = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_run_one, exp, project_root, verbose, n_workers, reward_overrides): exp
                for exp in pending
            }
            for fut in as_completed(futures):
                completed += 1
                exp = futures[fut]
                try:
                    fut.result()
                except Exception as exc:
                    _log(f"  [EXCEPTION] {_short_label(exp)}: {exc}")
                _log(f"  Progress: {completed}/{len(pending)}")
    else:
        for i, exp in enumerate(pending, 1):
            _log(f"\n=== [{i}/{len(pending)}] {_short_label(exp)} ===")
            _run_one(exp, project_root, verbose, n_workers=1, reward_overrides=reward_overrides)

    print("\nDone.")


if __name__ == "__main__":
    main()
