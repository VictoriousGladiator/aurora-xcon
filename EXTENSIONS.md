# Aurora-XCon Extensions — Fitness-Based Extinction & Evaluation Pipeline

This document describes the extensions added on the `fitness_trigger` branch on top of the base aurora-xcon codebase. All changes are backward-compatible: existing configs and runs are unaffected unless you explicitly opt into the new modes.

---

## 1. New Extinction Modes

Four extinction modes now exist (set via `extinction_mode=<value>` Hydra override or in `configs/aurora.yaml`):

| `extinction_mode` | Description |
|---|---|
| `off` | No extinction (baseline) |
| `static` | Fixed-period extinction every `extinction_freq` outer iterations |
| `adaptive` | **NEW** — trigger based on mean fitness stagnation (see below) |
| `encoder_post` | **NEW** — extinction fires immediately after each encoder retraining |
| `random_encoder_rate` | **NEW** — stochastic extinction, same expected frequency as encoder training |

### Adaptive extinction (`extinction_mode=adaptive`)

Fires when the mean fitness of the top-`k`% of the archive has stagnated: the ratio of the recent improvement rate to the historical improvement rate falls below threshold `α`.

Config block in `aurora.yaml` (all units = outer iterations):
```yaml
top_k_percent: 20          # which fraction of archive to monitor
adaptive_extinction:
  warmup: 50               # minimum outer iters before trigger is eligible
  patience: 10             # recent window size
  alpha: 0.2               # trigger threshold: recent_rate / historical_rate < alpha
  cooldown: 10             # lockout after each extinction event
```

### Encoder-coupled modes

`encoder_post` and `random_encoder_rate` both use the encoder training schedule as a reference rate. The encoder trains on an accelerating schedule derived from `default_update_base` (≈19 training events over 400 outer iterations for default kheperax settings).

- **`encoder_post`**: at each encoder training iteration, the archive is cleared *after* the encoder finishes retraining — so the new encoder sees the cleaned archive first.
- **`random_encoder_rate`**: each outer iteration fires an extinction independently with probability `n_encoder_trainings / num_outer_iterations` — same expected count, random timing.

Both use `remaining_prop` (default 0.05) to control how much of the archive survives each event.

---

## 2. New Fitness Function (`reward_type=final_speed`)

The default kheperax fitness (`reward_type=final`) is the negated final distance to the goal (0 when reached, negative otherwise). A second mode was added:

**`reward_type=final_speed`** — preserves the distance signal but rewards faster goal-reaching:
- Goal **not reached**: fitness = `-100 × final_distance` (same as `final`)
- Goal **reached at step T**: fitness = `(episode_length - T - 1) / speed_bonus_factor`

Set in `configs/env/kheperax.yaml`:
```yaml
reward_type: final          # "final" | "final_speed"
speed_bonus_factor: 1.0    # divides the steps-remaining bonus; max fitness = 199 with factor=1
```

Or override per-run:
```bash
python -m main main aurora env=kheperax env.reward_type=final_speed env.speed_bonus_factor=1.0
```

---

## 3. Per-Run Metrics Logging

Every run now writes structured CSV logs via `logging_utils/metrics_logger.py`. Files are saved inside the run's Hydra output directory:

```
output/<experiment_key>/runs/<timestamp>_kheperax_seed<N>/
    config.yaml          # run parameters (extinction mode, seed, reward_type, ...)
    metrics.csv          # per-outer-iteration metrics (see columns below)
    encoder_log.csv      # per-encoder-retrain metrics (loss, latent drift, ...)
    plots/
        fitness.png      # QD score, max/mean fitness, archive size over time
        trigger.png      # trigger internals (adaptive mode only)
```

**`metrics.csv` columns:**

| Column | Description |
|---|---|
| `generation` | Outer iteration × `metrics_log_period` |
| `qd_score` | Sum of archive fitnesses (raw, no offset) |
| `max_fitness` | Best fitness currently in archive |
| `mean_fitness` | Mean fitness of occupied cells |
| `mean_fitness_top_k` | Mean fitness of top-`k`% cells (the trigger signal) |
| `archive_size` | Number of occupied cells |
| `extinction_event` | `True` if extinction fired this iteration |
| `trigger_recent_rate` | Recent improvement rate (adaptive mode) |
| `trigger_historical_rate` | Historical improvement rate (adaptive mode) |
| `trigger_ratio` | `recent / historical` (fires when < `alpha`) |

---

## 4. Multi-Seed Sweep Runner (`scripts/run_sweep.py`)

Runs all experiments in the grid sequentially or in parallel. Each run gets a **deterministic output directory** (`output/<reward>_<condition>_seed<N>/`) so parallel workers never collide on Hydra's timestamp-based paths.

W&B runs offline and logs are silenced by default.

```bash
# Run the main sweep (9 conditions × 5 seeds)
python scripts/run_sweep.py

# Run in parallel (N workers)
python scripts/run_sweep.py --workers 4

# Run only the encoder-coupled conditions (2 conditions × 5 seeds)
python scripts/run_sweep.py --subset encoder

# Use the final_speed fitness
python scripts/run_sweep.py --reward final_speed --workers 4

# Check what is already done / what is missing
python scripts/run_sweep.py --check
python scripts/run_sweep.py --subset encoder --reward final_speed --check

# Preview commands without running
python scripts/run_sweep.py --dry-run
```

**Experiment subsets:**

| `--subset` | Conditions |
|---|---|
| `main` (default) | 9 adaptive-extinction hyperparameter conditions + static + no-extinction |
| `encoder` | `encoder_post` and `random_encoder_rate` only |

**Seeds used:** `[20, 42, 7, 13, 99]`

---

## 5. Evaluation Pipeline (`evaluation/`)

Run all scripts with `python -m evaluation.<script>` from the project root.

### Step 1 — Compute scalar metrics

```bash
python -m evaluation.compute_metrics --reward final
python -m evaluation.compute_metrics --reward final_speed
```

Scans all complete runs, computes per-run scalars, saves to `evaluation/metrics_summary_{reward}.csv`.

**Metrics computed:**

| Metric | Description |
|---|---|
| `final_qd_score` | Offset QD score at last iteration (matches W&B display) |
| `best_fitness_reached` | Peak `max_fitness` ever observed |
| `auc_qd_score` | Trapezoidal integral of QD score over all iterations |
| `extinction_count` | Total extinction events |
| `mean_extinction_interval` | Mean generations between consecutive events |
| `gen_best_fitness` | Generation at which peak fitness was first reached |

### Step 2 — Boxplots (scalar comparison across seeds)

```bash
python -m evaluation.plot_comparison --reward final
python -m evaluation.plot_comparison --reward final --conditions no_extinction static adaptive_default --tag my_subset
```

Outputs: `evaluation/figures/{reward}/comparison_{metric}_{tag}.png`

### Step 3 — Evolution plots (per-generation trajectories)

```bash
python -m evaluation.plot_evolution --reward final
python -m evaluation.plot_evolution --reward final_speed --conditions encoder_post enc_random_rate --tag encoder_exp
```

Shows individual seed traces (thin) + mean (bold) per condition. Outputs: `evaluation/figures/{reward}/evolution_{metric}_{tag}.png`

### Common flags for both plot scripts

| Flag | Description |
|---|---|
| `--reward final` or `--reward final_speed` | Which reward type to plot |
| `--conditions cond1 cond2 ...` | Plot only this subset of conditions |
| `--tag my_label` | Append `_my_label` to all output filenames (avoids overwriting) |

**Available condition labels:**

| Label | Condition |
|---|---|
| `no_extinction` | Baseline — no extinction |
| `static` | Fixed-period extinction |
| `adaptive_default` | Adaptive, default params (topk=20, patience=10, α=0.2) |
| `adaptive_topk10/30` | Vary top-k% |
| `adaptive_patience5/20` | Vary patience window |
| `adaptive_alpha01/03` | Vary trigger threshold α |
| `encoder_post` | Post-encoder extinction |
| `enc_random_rate` | Random extinction at encoder rate |

---

## 6. Typical Workflow

```bash
# 1. Run experiments
python scripts/run_sweep.py --workers 4              # main sweep, final reward
python scripts/run_sweep.py --reward final_speed --workers 4
python scripts/run_sweep.py --subset encoder --workers 5
python scripts/run_sweep.py --subset encoder --reward final_speed --workers 5

# 2. Compute metrics
python -m evaluation.compute_metrics --reward final
python -m evaluation.compute_metrics --reward final_speed

# 3. Main sweep plots
python -m evaluation.plot_comparison --reward final \
  --conditions no_extinction static adaptive_default adaptive_topk10 adaptive_topk30 \
               adaptive_patience5 adaptive_patience20 adaptive_alpha01 adaptive_alpha03 \
  --tag main_sweep
python -m evaluation.plot_evolution --reward final --tag main_sweep \
  --conditions no_extinction static adaptive_default adaptive_topk10 adaptive_topk30 \
               adaptive_patience5 adaptive_patience20 adaptive_alpha01 adaptive_alpha03

# 4. Encoder experiment plots
python -m evaluation.plot_comparison --reward final \
  --conditions encoder_post enc_random_rate adaptive_default static no_extinction \
  --tag encoder_exp
python -m evaluation.plot_evolution --reward final \
  --conditions encoder_post enc_random_rate adaptive_default static no_extinction \
  --tag encoder_exp

# (repeat steps 3-4 with --reward final_speed)
```

---

## 7. Changes to `train/aurora_trainer.py`

This is the main file that was modified. All changes are confined to the `train()` function and the `main()` entry point; the core QDax/AURORA algorithm is untouched.

**`train()` function:**

- Added `_check_adaptive_trigger()` — computes the recent vs. historical mean-fitness-top-k improvement rate and fires when their ratio falls below `α`. Called each outer iteration when `extinction_mode=adaptive`.
- Added `encoder_post` branch — sets `is_ext=True` at encoder schedule iterations, then applies the extinction *after* `aurora.train()` completes (so the encoder sees a clean archive on its next training pass).
- Added `random_encoder_rate` branch — pre-computes the encoder training probability (`≈19/400` for default kheperax settings) before the loop, then samples a Bernoulli decision each outer iteration.
- The `if is_ext` extinction block now skips application for `encoder_post` (it happens in the AE block instead), keeping the logging consistent: `extinction_event=True` is recorded at the correct generation regardless of mode.
- Added `MetricsLogger` integration — instantiated once after the initial `aurora.train()` call, called every outer iteration to write `metrics.csv`, and flushed at end of training.

**`main()` entry point:**

- Saves a `config.yaml` into the run directory containing: `env`, `seed`, `extinction_mode`, `top_k_percent`, `adaptive_extinction.*`, `reward_type`, `speed_bonus_factor`. This is what the evaluation pipeline reads to identify which condition each run belongs to.

---

## 8. Key Files Changed

| File | Change |
|---|---|
| `train/aurora_trainer.py` | See Section 7 above |
| `tasks/scoring.py` | Added `reward_type=final_speed` branch and `speed_bonus_factor` parameter |
| `configs/env/kheperax.yaml` | Added `reward_type` and `speed_bonus_factor` fields |
| `configs/aurora.yaml` | Added `extinction_mode`, `top_k_percent`, `adaptive_extinction` block |
| `logging_utils/metrics_logger.py` | New file — per-run CSV logging class |
| `scripts/run_sweep.py` | New file — parallel multi-seed sweep runner |
| `scripts/plot_metrics.py` | New file — per-run fitness and trigger plots |
| `evaluation/` | New directory — `run_index.py`, `compute_metrics.py`, `plot_comparison.py`, `plot_evolution.py` |
