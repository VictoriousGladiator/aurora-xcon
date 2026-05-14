"""Render fitness panels from an instrumented AURORA-XCon run.

Usage: python scripts/plot_metrics.py runs/<run_dir>
"""
from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def _mark_extinctions(ax, gens_extinct):
    for g in gens_extinct:
        ax.axvline(g, color="red", alpha=0.3, linewidth=1.0, label="_nolegend_")


def _make_title(run_dir: Path) -> str:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        return run_dir.name
    cfg = OmegaConf.load(str(cfg_path))
    ae = cfg.get("adaptive_extinction", {})
    reward = cfg.get("reward_type", "final")
    reward_str = f"final_speed(f={cfg.get('speed_bonus_factor', 1.0)})" if reward == "final_speed" else reward
    return (
        f"{cfg.get('env', '?')} | seed={cfg.get('seed', '?')} | "
        f"fitness={reward_str} | "
        f"mode={cfg.get('extinction_mode', '?')} | "
        f"top_k={cfg.get('top_k_percent', '?')}% | "
        f"patience={ae.get('patience', '?')} | "
        f"\u03b1={ae.get('alpha', '?')} | "
        f"cooldown={ae.get('cooldown', '?')}"
    )


def plot_run(run_dir: Path):
    m = pd.read_csv(run_dir / "metrics.csv")
    out = run_dir / "plots"
    out.mkdir(exist_ok=True)

    gens_extinct = m.loc[m["extinction_event"], "generation"].tolist()
    title = _make_title(run_dir)

    # --- fitness panels ---
    cols = [
        ("archive_size", "Archive Size (# occupied cells)"),
        ("qd_score", "QD Score (active)"),
        ("max_fitness", "Max Fitness (active)"),
        ("mean_fitness", "Mean Fitness (active)"),
        ("mean_fitness_top_k", "Mean Fitness top-k% (active)"),
    ]

    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 10), sharex=True)
    fig.suptitle(title, fontsize=9, y=1.01)

    for ax, (col, label) in zip(axes, cols):
        if col not in m.columns:
            ax.set_visible(False)
            continue
        ax.plot(m["generation"], m[col], linewidth=1.2)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
        _mark_extinctions(ax, gens_extinct)
        if gens_extinct:
            ax.axvline(gens_extinct[0], color="red", alpha=0.3,
                       linewidth=1.0, label="extinction")
            ax.legend(fontsize=7)

    axes[-1].set_xlabel("generation")
    fig.tight_layout()
    fig.savefig(out / "fitness.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out / 'fitness.png'}")

    # --- d_min volatility internals (d_min_trigger mode) ---
    dmin_cols = ["d_min_recent_volatility", "d_min_historical_volatility", "d_min_volatility_ratio"]
    if all(c in m.columns for c in dmin_cols) and m[dmin_cols].notna().any().any():
        cfg_path = run_dir / "config.yaml"
        d_alpha = None
        if cfg_path.exists():
            cfg = OmegaConf.load(str(cfg_path))
            d_alpha = cfg.get("adaptive_extinction", {}).get("d_min_alpha")

        labels_d = [
            "recent volatility $v_{\\mathrm{recent}}$",
            "historical volatility $v_{\\mathrm{hist}}$",
            "ratio $v_{\\mathrm{recent}} / v_{\\mathrm{hist}}$",
        ]
        fig3, axes3 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig3.suptitle(title + " — d_min volatility", fontsize=9, y=1.01)

        for ax, col, label in zip(axes3, dmin_cols, labels_d):
            ax.plot(m["generation"], m[col], linewidth=1.2)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            _mark_extinctions(ax, gens_extinct)

        if d_alpha is not None:
            axes3[2].axhline(d_alpha, color="orange", linestyle="--",
                             linewidth=1.0, label=f"d_min_α={d_alpha}")
            axes3[2].legend(fontsize=7)

        if gens_extinct:
            axes3[0].axvline(gens_extinct[0], color="red", alpha=0.3,
                             linewidth=1.0, label="extinction")
            axes3[0].legend(fontsize=7)

        axes3[-1].set_xlabel("generation")
        fig3.tight_layout()
        fig3.savefig(out / "dmin_trigger.png", dpi=130, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved: {out / 'dmin_trigger.png'}")

    # --- fitness trigger internals (fitness_trigger mode) ---
    trigger_cols = ["trigger_recent_rate", "trigger_historical_rate", "trigger_ratio"]
    if all(c in m.columns for c in trigger_cols) and m[trigger_cols].notna().any().any():
        cfg_path = run_dir / "config.yaml"
        alpha = None
        if cfg_path.exists():
            cfg = OmegaConf.load(str(cfg_path))
            alpha = cfg.get("adaptive_extinction", {}).get("alpha")

        labels = ["recent rate $r_{\\mathrm{recent}}$",
                  "historical rate $r_{\\mathrm{hist}}$",
                  "ratio $r_{\\mathrm{recent}} / r_{\\mathrm{hist}}$"]
        fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig2.suptitle(title + " — trigger internals", fontsize=9, y=1.01)

        for ax, col, label in zip(axes2, trigger_cols, labels):
            ax.plot(m["generation"], m[col], linewidth=1.2)
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            _mark_extinctions(ax, gens_extinct)

        if alpha is not None:
            axes2[2].axhline(alpha, color="orange", linestyle="--",
                             linewidth=1.0, label=f"α={alpha}")
            axes2[2].legend(fontsize=7)

        if gens_extinct:
            axes2[0].axvline(gens_extinct[0], color="red", alpha=0.3,
                             linewidth=1.0, label="extinction")
            axes2[0].legend(fontsize=7)

        axes2[-1].set_xlabel("generation")
        fig2.tight_layout()
        fig2.savefig(out / "trigger.png", dpi=130, bbox_inches="tight")
        plt.close(fig2)
        print(f"Saved: {out / 'trigger.png'}")


if __name__ == "__main__":
    plot_run(Path(sys.argv[1]))
