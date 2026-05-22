"""Utilities for scanning completed aurora-xcon run directories.

Pure read — no side effects when imported.

Public API:
    scan_runs(project_root)  -> list[dict]
    condition_label(cfg)     -> str
    params_match(exp, cfg)   -> bool
    SUBSET_LABELS            -> dict mapping subset name -> ordered list of labels
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

# Defaults from aurora.yaml — used when a key is absent from a run's config.yaml.
# aurora_trainer.py only writes: env, seed, extinction_mode, top_k_percent,
# adaptive_extinction.*  — so extinction_freq / remaining_prop are always absent.
_DEFAULTS: dict[str, Any] = {
    "extinction_mode": "off",
    "top_k_percent": 20,
    "patience": 10,
    "alpha": 0.2,
    "cooldown": 10,
    "extinction_freq": 10,
    "remaining_prop": 0.05,
}

# Dotted override key -> flat key used in the scan_runs dict
_KEY_MAP: dict[str, str] = {
    "adaptive_extinction.patience": "patience",
    "adaptive_extinction.alpha": "alpha",
    "adaptive_extinction.cooldown": "cooldown",
}

_DEFAULT_TOPK = 20
_DEFAULT_PATIENCE = 10
_DEFAULT_ALPHA = 0.2


def _infer_reward_type(cfg_path: Path) -> str:
    """Infer reward_type from the deterministic Hydra output dir name when absent from config.

    New-style paths:  output/{reward_type}_{condition}_seed{N}/runs/.../config.yaml
    Old-style paths:  output/{YYYY-MM-DD}/{HH-MM-SS}/runs/.../config.yaml

    cfg_path.parent          = run_dir
    cfg_path.parent.parent   = runs/
    cfg_path.parent.parent.parent = hydra_dir  (e.g. "final_speed_default_seed20")
    """
    hydra_dir_name = cfg_path.parent.parent.parent.name
    if hydra_dir_name.startswith("final_speed"):
        return "final_speed"
    return "final"


def scan_runs(project_root: Path) -> list[dict]:
    """Scan output/ for run directories and return a list of metadata dicts.

    Each dict has keys:
        run_dir, seed, extinction_mode, top_k_percent,
        patience, alpha, cooldown,
        extinction_freq, remaining_prop,
        is_complete, max_generation
    """
    results = []
    for cfg_path in sorted((project_root / "output").rglob("runs/*/config.yaml")):
        run_dir = cfg_path.parent
        try:
            cfg = OmegaConf.load(str(cfg_path))
        except Exception:
            continue

        ae = cfg.get("adaptive_extinction", {}) or {}

        record: dict[str, Any] = {
            "run_dir": run_dir,
            "seed": int(cfg.get("seed", -1)),
            "extinction_mode": str(cfg.get("extinction_mode", _DEFAULTS["extinction_mode"])),
            "top_k_percent": int(cfg.get("top_k_percent", _DEFAULTS["top_k_percent"])),
            "patience": int(ae.get("patience", _DEFAULTS["patience"])),
            "alpha": float(ae.get("alpha", _DEFAULTS["alpha"])),
            "cooldown": int(ae.get("cooldown", _DEFAULTS["cooldown"])),
            "extinction_freq": int(cfg.get("extinction_freq", _DEFAULTS["extinction_freq"])),
            "remaining_prop": float(cfg.get("remaining_prop", _DEFAULTS["remaining_prop"])),
            "target_extinction_count": cfg.get("target_extinction_count", None),
            # reward_type absent in old configs → infer from deterministic dir name if possible,
            # otherwise default to "final".
            # New-style paths: output/{reward_type}_{condition}_seed{N}/runs/...
            # e.g. output/final_speed_default_seed20/runs/...  → hydra_dir.name starts with "final_speed"
            "reward_type": str(cfg.get("reward_type", _infer_reward_type(cfg_path))),
            "speed_bonus_factor": float(cfg.get("speed_bonus_factor", 1.0)),
            "is_complete": False,
            "max_generation": 0,
        }

        metrics_path = run_dir / "metrics.csv"
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                if not df.empty:
                    max_gen = int(df["generation"].iloc[-1])
                    record["max_generation"] = max_gen
                    record["is_complete"] = (max_gen == 2000)
            except Exception:
                pass

        results.append(record)
    return results


def condition_label(cfg: dict) -> str:
    """Map a run metadata dict (from scan_runs) to a human-readable condition label.

    Labels encode the single parameter that differs from the adaptive default:
        no_extinction       extinction_mode=off
        static              extinction_mode=static
        adaptive_default    mode=adaptive, topk=20, patience=10, alpha=0.2
        adaptive_topk10     topk=10
        adaptive_topk30     topk=30
        adaptive_patience5  patience=5
        adaptive_patience20 patience=20
        adaptive_alpha01    alpha=0.1
        adaptive_alpha03    alpha=0.3
    """
    mode = cfg["extinction_mode"]
    if mode == "off":
        return "no_extinction"
    if mode == "static":
        return f"static_freq{cfg['extinction_freq']}"
    if mode == "encoder_post":
        return "encoder_post"
    if mode == "encoder_pre":
        return "encoder_pre"
    if mode == "random_encoder_rate":
        return "enc_random_rate"
    if mode == "d_min_trigger":
        return "d_min_trigger"
    if mode == "static_ramped_proportion":
        return f"ramped_prop_freq{cfg['extinction_freq']}"
    if mode == "random_fixed_count":
        count = cfg.get("target_extinction_count", "?")
        return f"random_fixed_count_{count}"

    # fitness_trigger — distinguish by topk/patience/alpha
    topk = int(cfg["top_k_percent"])
    patience = int(cfg["patience"])
    alpha = float(cfg["alpha"])

    if topk == _DEFAULT_TOPK and patience == _DEFAULT_PATIENCE and abs(alpha - _DEFAULT_ALPHA) < 1e-5:
        return "fitness_trigger_default"

    if topk != _DEFAULT_TOPK:
        return f"fitness_trigger_topk{topk}"
    if patience != _DEFAULT_PATIENCE:
        return f"fitness_trigger_patience{patience}"
    # Encode alpha: strip decimal point ("0.1" -> "01", "0.3" -> "03")
    alpha_str = f"{alpha:.1f}".replace(".", "")
    return f"fitness_trigger_alpha{alpha_str}"


def params_match(exp_dict: dict, run_cfg: dict) -> bool:
    """Return True if all exp_dict override keys match values in run_cfg (flat dict).

    Handles dotted exp_dict keys like "adaptive_extinction.patience" by mapping
    them to the flat run_cfg keys via _KEY_MAP.
    """
    for exp_key, exp_val in exp_dict.items():
        run_key = _KEY_MAP.get(exp_key, exp_key)
        if run_key not in run_cfg:
            return False
        run_val = run_cfg[run_key]
        if isinstance(exp_val, float) or isinstance(run_val, float):
            try:
                if abs(float(run_val) - float(exp_val)) > 1e-5:
                    return False
            except (TypeError, ValueError):
                return False
        else:
            if str(run_val) != str(exp_val):
                return False
    return True


# ---------------------------------------------------------------------------
# SUBSET_LABELS — loaded dynamically from scripts/run_sweep.py so it always
# reflects the current condition lists without manual syncing.
# run_sweep.py computes SUBSET_LABELS from the actual condition dicts using
# _cond_label(), which mirrors condition_label() for completed runs.
# ---------------------------------------------------------------------------

def _load_subset_labels() -> "dict[str, list[str]]":
    """Load SUBSET_LABELS from scripts/run_sweep.py via importlib (no circular import)."""
    import importlib.util
    from pathlib import Path
    _path = Path(__file__).parent.parent / "scripts" / "run_sweep.py"
    spec = importlib.util.spec_from_file_location("_run_sweep_module", _path)
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as _e:
        import warnings
        warnings.warn(f"run_index: could not load SUBSET_LABELS from {_path}: {_e}")
        return {}
    return getattr(mod, "SUBSET_LABELS", {})


SUBSET_LABELS: dict[str, list[str]] = _load_subset_labels()
