# Experiments Overview

Four experiment groups are defined in `scripts/run_sweep.py`. Each uses **7 seeds** `[20, 42, 7, 13, 99, 0, 100]` and runs on the Kheperax environment with triplet loss.

Run any experiment group with:
```bash
python scripts/run_sweep.py --subset exp1   # or exp2, exp3, exp4
python scripts/run_sweep.py --subset exp1 --workers 4   # parallel
python scripts/run_sweep.py --subset exp1 --check       # completion status
python scripts/run_sweep.py --subset exp1 --dry-run     # print commands only
```

---

## Snapshot policy (all experiments)

For each condition, the **first 2 seeds** (20, 42) receive:
- Archive descriptor checkpoints (`.npz`) at generations 100, 250, 500, 750, 1000, 1500, 2000
- Latent space PNG visualisations before/after each extinction event and at each snapshot generation

The remaining 5 seeds skip all extinction PNGs to save disk space and runtime.

---

## Extinction mechanism

All experiments share the same extinction proportion: **5% of the archive is kept** after each extinction event. Experiments 1–3 use **random selection** among survivors (`safe_keep.enabled=false`). Experiment 4 uses the **fitness-coverage selection** mechanism (`safe_keep.enabled=true`, the default), which bins individuals by fitness level and favours spatially dense, representative clusters within each bin.

---

## Experiment 1 — Trigger type comparison (random extinction)

**Question:** Which trigger strategy best decides *when* to apply extinction?

**Conditions (6 × 7 seeds = 42 runs):**

| Condition | `extinction_mode` | Notes |
|-----------|-------------------|-------|
| No extinction | `off` | Baseline |
| Static periodic | `static` | Every 10 outer iterations (~every 50 gens) |
| Fitness trigger | `fitness_trigger` | Fires when recent top-k% fitness improvement drops below 20% of historical rate |
| QD score trigger | `qd_trigger` | Same logic as fitness trigger but tracks total QD score instead of top-k fitness |
| Encoder post | `encoder_post` | Extinction applied after each encoder retraining |
| Encoder pre | `encoder_pre` | Extinction applied before each encoder retraining |

Trigger parameters (fitness and QD triggers): `patience=10`, `alpha=0.2`, `cooldown=10`.

---

## Experiment 2 — Extinction placement timing (random extinction)

**Question:** Does the *timing window* of extinctions within training matter?

**Conditions (5 × 7 seeds = 35 runs):**

| Condition | `extinction_mode` | Placement | Count |
|-----------|-------------------|-----------|-------|
| No extinction | `off` | — | — |
| Static periodic | `static` | Uniform every 50 gens | ~40 |
| Early phase | `random_fixed_count_window` | Randomly within gens 0–750 | 12 |
| Uniform random | `random_fixed_count` | Randomly across all gens | 12 |
| Late phase | `random_fixed_count_window` | Randomly within gens 1000–2000 | 12 |

The three "12 extinction" conditions are matched in count to isolate the effect of placement timing.

---

## Experiment 3 — Static extinction frequency (random extinction)

**Question:** How sensitive is performance to how often extinctions occur?

**Conditions (4 × 7 seeds = 28 runs):**

| Condition | `extinction_mode` | Frequency | Approx. count over 2000 gens |
|-----------|-------------------|-----------|------------------------------|
| No extinction | `off` | — | 0 |
| High frequency | `static` freq=10 | Every 50 gens | ~40 |
| Medium frequency | `static` freq=22 | Every 110 gens | ~18 |
| Low frequency | `static` freq=33 | Every 165 gens | ~12 |

---

## Experiment 4 — Trigger type comparison with fitness-coverage extinction

**Question:** Does the fitness-coverage selection mechanism improve over random extinction across trigger types?

Identical trigger conditions to Experiment 1, but using the fitness-coverage extinction mechanism. Adds one extra fixed-schedule condition as an ablation.

**Conditions (7 × 7 seeds = 49 runs):**

| Condition | `extinction_mode` | Notes |
|-----------|-------------------|-------|
| No extinction | `off` | Baseline |
| Static periodic | `static` | Every 10 outer iterations |
| Fitness trigger | `fitness_trigger` | Same params as Exp 1 |
| QD score trigger | `qd_trigger` | Same params as Exp 1 |
| Encoder post | `encoder_post` | Extinction after encoder retrain |
| Encoder pre | `encoder_pre` | Extinction before encoder retrain |
| Fixed early bursts | `fixed_generations` | Extinctions at gens 20, 100, 250 |

The direct comparison between Exp 1 and Exp 4 isolates the contribution of the extinction mechanism (random vs fitness-coverage) from the trigger strategy.

---

## Total runs

| Experiment | Conditions | Seeds | Runs |
|------------|-----------|-------|------|
| Exp 1 | 6 | 7 | 42 |
| Exp 2 | 5 | 7 | 35 |
| Exp 3 | 4 | 7 | 28 |
| Exp 4 | 7 | 7 | 49 |
| **Total** | | | **154** |

Output lives under `output/<reward>_<condition>_seed<N>/`. Completed runs are skipped automatically on re-run.
