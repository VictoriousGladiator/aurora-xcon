"""Observation-only logger for adaptive-extinction signal discovery.

Pure read of algorithm state — must not alter any QD dynamics.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np


# ---------- metric primitives (pure numpy) ----------

def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= n:
        return x
    idx = rng.choice(x.shape[0], size=n, replace=False)
    return x[idx]


def effective_rank(latents: np.ndarray, rng: np.random.Generator, cap: int = 500) -> float:
    if latents.shape[0] < 2:
        return float("nan")
    x = _subsample(latents, cap, rng)
    x = x - x.mean(axis=0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    if s.sum() == 0:
        return 0.0
    return float((s.sum() ** 2) / (s ** 2).sum())


def cosine_drift(old_emb: np.ndarray, new_emb: np.ndarray) -> float:
    num = (old_emb * new_emb).sum(axis=1)
    den = np.linalg.norm(old_emb, axis=1) * np.linalg.norm(new_emb, axis=1) + 1e-12
    return float((1.0 - num / den).mean())


def latent_answer_set_metrics(
    Z: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 400,
) -> dict[str, float]:
    """Geometry of occupied archive cells in descriptor (latent) space.

    ``latent_bbox_volume`` is the product of axis-aligned spans. If any axis
    collapses, the product is **0** (degenerate archive geometry).
    """
    nan = float("nan")
    out: dict[str, float] = {
        "latent_bbox_volume": nan,
        "latent_space_diameter": nan,
        "latent_pca_volume": nan,
    }
    if Z.ndim != 2 or Z.shape[0] < 2:
        return out
    Z = np.asarray(Z, dtype=np.float64)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(Z).all():
        return out

    lo = Z.min(axis=0)
    hi = Z.max(axis=0)
    span = np.maximum(hi - lo, 1e-30)
    out["latent_bbox_volume"] = float(np.prod(span))

    n = Z.shape[0]
    P = Z if n <= max_points else Z[rng.choice(n, size=max_points, replace=False)]
    if P.shape[0] >= 2:
        try:
            from scipy.spatial.distance import pdist

            out["latent_space_diameter"] = float(np.max(pdist(P, metric="euclidean")))
        except Exception:
            out["latent_space_diameter"] = nan

    Zc = Z - Z.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(Zc, full_matrices=False)
        s = s[s > 1e-20]
        out["latent_pca_volume"] = float(np.prod(s)) if s.size > 0 else 0.0
    except np.linalg.LinAlgError:
        out["latent_pca_volume"] = 0.0
    return out


# ---------- logger ----------

@dataclass
class MetricsLogger:
    run_dir: Path
    flush_every: int = 50
    seed: int = 0
    drift_sample_size: int = 256
    _gen_rows: list = field(default_factory=list)
    _enc_rows: list = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)
    _rng: np.random.Generator = field(init=False)
    _drift_obs: Optional[np.ndarray] = field(default=None)
    _drift_old_emb: Optional[np.ndarray] = field(default=None)
    _flushed_gen_count: int = field(default=0)

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)
        self._rng = np.random.default_rng(self.seed)

    def set_drift_sample(self, repertoire: Any, aurora_extra_info: Any, encoder_fn: Callable):
        """Pin a held-out observation sample for latent-drift measurement.

        Called once after the initial aurora.train() in main().
        """
        valid = np.asarray(repertoire.fitnesses != -jnp.inf)
        obs = np.asarray(repertoire.observations)
        valid_obs = obs[valid]
        n = min(self.drift_sample_size, valid_obs.shape[0])
        idx = self._rng.choice(valid_obs.shape[0], size=n, replace=False)
        self._drift_obs = valid_obs[idx]
        self._drift_old_emb = np.asarray(encoder_fn(self._drift_obs, aurora_extra_info))

    def log_generation(
        self,
        generation: int,
        fitnesses: np.ndarray,
        is_extinction_gen: bool,
        mean_fitness_top_k: float,
        trigger_info: dict,
        archive_size: int = 0,
        dmin_trigger_info: dict | None = None,
        extinction_remaining_prop: float | None = None,
        answer_set_metrics: dict | None = None,
    ):
        """Log per-outer-iteration metrics.

        fitnesses is a pre-filtered numpy array of active archive fitnesses (empty slots removed).
        mean_fitness_top_k is pre-computed by the trainer (same sort used by the trigger).
        trigger_info keys: recent_rate, historical_rate, ratio, triggered.
        dmin_trigger_info keys: recent_volatility, historical_volatility, ratio, triggered.
        extinction_remaining_prop: prop used for this extinction event (None if no extinction).
        archive_size is the number of occupied cells (fits_np.size).
        answer_set_metrics: optional keys from :func:`latent_answer_set_metrics` plus
            ``repertoire_d_min`` when logged by the trainer.
        """
        _nan = float("nan")
        dmin = dmin_trigger_info or {}
        asm = answer_set_metrics or {}
        row = {
            "generation": int(generation),
            "wall_time_s": float(np.float32(time.time() - self._start_time)),
            "archive_size": int(archive_size),
            "qd_score": float(np.float32(fitnesses.sum())),
            "max_fitness": float(np.float32(fitnesses.max())) if fitnesses.size else _nan,
            "mean_fitness": float(np.float32(fitnesses.mean())) if fitnesses.size else _nan,
            "mean_fitness_top_k": float(np.float32(mean_fitness_top_k)),
            "extinction_event": bool(is_extinction_gen),
            "trigger_recent_rate": float(np.float32(trigger_info["recent_rate"])),
            "trigger_historical_rate": float(np.float32(trigger_info["historical_rate"])),
            "trigger_ratio": float(np.float32(trigger_info["ratio"])),
            "trigger_fired": bool(trigger_info["triggered"]),
            "d_min_recent_volatility": float(np.float32(dmin.get("recent_volatility", _nan))),
            "d_min_historical_volatility": float(np.float32(dmin.get("historical_volatility", _nan))),
            "d_min_volatility_ratio": float(np.float32(dmin.get("ratio", _nan))),
            "d_min_trigger_fired": bool(dmin.get("triggered", False)),
            "extinction_remaining_prop": (
                float(np.float32(extinction_remaining_prop))
                if extinction_remaining_prop is not None else _nan
            ),
            "latent_bbox_volume": float(np.float32(asm.get("latent_bbox_volume", _nan))),
            "latent_space_diameter": float(np.float32(asm.get("latent_space_diameter", _nan))),
            "latent_pca_volume": float(np.float32(asm.get("latent_pca_volume", _nan))),
            "repertoire_d_min": float(np.float32(asm.get("repertoire_d_min", _nan))),
        }
        self._gen_rows.append(row)
        if len(self._gen_rows) % self.flush_every == 0:
            self._flush_generation()

    def log_encoder_retrain(
        self,
        generation: int,
        model_metrics: dict,
        old_aurora_extra_info: Any,
        new_aurora_extra_info: Any,
        encoder_fn: Callable,
        archived_descriptors: Optional[np.ndarray] = None,
    ):
        """Log per-encoder-retraining metrics."""
        losses = np.asarray(model_metrics["model_loss"])
        if losses.size == 0:
            return
        s, e = float(losses[0]), float(losses[-1])
        drop = float(np.float32((s - e) / s)) if s > 0 else float("nan")

        drift = float("nan")
        if self._drift_obs is not None and self._drift_old_emb is not None:
            new_emb = np.asarray(encoder_fn(self._drift_obs, new_aurora_extra_info))
            drift = float(np.float32(cosine_drift(self._drift_old_emb, new_emb)))
            self._drift_old_emb = new_emb

        eff_rank = float("nan")
        if archived_descriptors is not None and archived_descriptors.shape[0] >= 2:
            eff_rank = float(np.float32(effective_rank(archived_descriptors, self._rng)))

        row = {
            "generation": int(generation),
            "wall_time_s": float(np.float32(time.time() - self._start_time)),
            "train_loss_start": float(np.float32(s)),
            "train_loss_end": float(np.float32(e)),
            "loss_drop_rel": drop,
            "latent_drift": drift,
            "effective_rank": eff_rank,
        }
        self._enc_rows.append(row)

    # ---------- IO ----------

    def _write_csv(self, rows: list, path: Path):
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)

    def _flush_generation(self):
        new_rows = self._gen_rows[self._flushed_gen_count:]
        if not new_rows:
            return
        path = self.run_dir / "metrics.csv"
        mode = "w" if self._flushed_gen_count == 0 else "a"
        fieldnames = list(new_rows[0].keys())
        with path.open(mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                w.writeheader()
            w.writerows(new_rows)
        self._flushed_gen_count = len(self._gen_rows)

    def flush(self):
        self._flush_generation()
        self._write_csv(self._enc_rows, self.run_dir / "encoder_log.csv")
