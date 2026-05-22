import collections
import datetime
import functools
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

import hydra
import jax
import jax.numpy as jnp
import wandb
from omegaconf import DictConfig, OmegaConf

from ae_utils.model_train import init_autoencoder_model_training
from qdax.core.aurora_adaptive import AURORAAdaptivePassive
from qdax.core.aurora_threshold import AURORAThresholdPassive
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks import environments, environments_v1
from qdax.tasks.brax_envs import get_aurora_scoring_fn
from qdax.utils.metrics import passive_qd_metrics
from tasks import task_behavior_descriptor_extractor
from tasks.scoring import scoring_function
from tasks.step import play_step_fn
from utils import (
    get_env,
    get_observation_dims,
    init_population,
    log_final_metrics,
    log_running_metrics,
    set_env_params,
)
from viz.visualization import kheperax_viz_best_individual, viz_best_individual

from logging_utils.metrics_logger import MetricsLogger, latent_answer_set_metrics


def _check_fitness_trigger(
    history: collections.deque,
    last_ext_iter: int,
    current_iter: int,
    ae_cfg,
) -> tuple:
    """Return (fired, recent_rate, historical_rate, ratio).

    All window sizes are in outer-loop iterations (each = metrics_log_period gens).
    Requires 5*patience+1 history values before it can fire.
    Uses signed deltas because fitness should trend upward.
    """
    nan = float("nan")
    p = ae_cfg.patience
    if current_iter < ae_cfg.warmup:
        return False, nan, nan, nan
    if current_iter - last_ext_iter < ae_cfg.cooldown:
        return False, nan, nan, nan
    n = len(history)
    if n < 5 * p + 1:
        return False, nan, nan, nan
    h = list(history)
    deltas = [h[j] - h[j - 1] for j in range(1, n)]
    recent_rate = float(np.mean(deltas[-p:]))
    historical_rate = float(np.mean(deltas[-5 * p:]))
    if historical_rate < 1e-12:
        return False, recent_rate, historical_rate, nan
    ratio = recent_rate / historical_rate
    triggered = bool(ratio < ae_cfg.alpha)
    return triggered, recent_rate, historical_rate, ratio


'''def _check_dmin_trigger(
    history: collections.deque,
    last_ext_iter: int,
    current_iter: int,
    ae_cfg,
) -> tuple:
    """Return (fired, recent_volatility, historical_volatility, ratio).

    Uses absolute deltas — d_min oscillates, so we care whether it has stopped
    moving at all, not its direction. Fires when recent movement is small
    relative to historical movement (ratio < d_min_alpha).
    """
    nan = float("nan")
    p = ae_cfg.d_min_patience
    if current_iter < ae_cfg.d_min_warmup:
        return False, nan, nan, nan
    if current_iter - last_ext_iter < ae_cfg.d_min_cooldown:
        return False, nan, nan, nan
    n = len(history)
    if n < 5 * p + 1:
        return False, nan, nan, nan
    h = list(history)
    abs_deltas = [abs(h[j] - h[j - 1]) for j in range(1, n)]
    recent_vol = float(np.mean(abs_deltas[-p:]))
    historical_vol = float(np.mean(abs_deltas[-5 * p:]))
    if historical_vol < 1e-12:
        return False, recent_vol, historical_vol, nan
    ratio = recent_vol / historical_vol
    triggered = bool(ratio < ae_cfg.d_min_alpha)
    return triggered, recent_vol, historical_vol, ratio'''

def _check_hybrid_trigger(
    dmin_history: collections.deque,
    fitness_history: collections.deque,
    last_ext_iter: int,
    current_iter: int,
    state,
    ae_cfg,
) -> tuple:
    """
    Return:
        (
            triggered,
            dmin_range,
            relative_fitness_improvement,
        )

    Triggers when:
      1. d_min stagnates AND fitness stagnates (after post-extinction re-arm)
      OR
      2. d_min is inf/nan (immediate), or stays above threshold for high_d_min_patience
    """
    nan = float("nan")

    # Warmup / cooldown
    if current_iter < ae_cfg.d_min_warmup:
        return False, nan, nan

    if current_iter - last_ext_iter < ae_cfg.d_min_cooldown:
        return False, nan, nan

    # Need enough history
    if len(dmin_history) < ae_cfg.d_min_window:
        return False, nan, nan

    if len(fitness_history) < ae_cfg.fitness_window:
        return False, nan, nan

    # d_min stagnation
    d_hist = list(dmin_history)
    current_dmin = float(d_hist[-1])
    finite_d = [float(x) for x in d_hist if math.isfinite(float(x))]
    dmin_range = (
        max(finite_d) - min(finite_d) if finite_d else float("nan")
    )

    rearmed = state.get("dmin_rearmed", True)
    if not rearmed and math.isfinite(dmin_range):
        if dmin_range > ae_cfg.d_min_stagnation_threshold:
            rearmed = True
            state["dmin_rearmed"] = True

    dmin_stagnating = (
        math.isfinite(dmin_range)
        and dmin_range < ae_cfg.d_min_stagnation_threshold
    )

    # Fitness stagnation
    f_hist = list(fitness_history)

    start_fit = f_hist[0]
    end_fit = f_hist[-1]

    rel_improvement = (
        (end_fit - start_fit)
        / (abs(start_fit) + 1e-8)
    )

    fitness_stagnating = (
        rel_improvement
        < ae_cfg.fitness_improvement_threshold
    )

    # Accumulate patience counters (only after post-extinction re-arm)
    if rearmed:
        state["dmin_stagnant_count"] = (
            state["dmin_stagnant_count"] + 1
            if dmin_stagnating
            else 0
        )

        state["fitness_stagnant_count"] = (
            state["fitness_stagnant_count"] + 1
            if fitness_stagnating
            else 0
        )
    else:
        state["dmin_stagnant_count"] = 0
        state["fitness_stagnant_count"] = 0

    # Emergency: inf/nan d_min fires immediately; high d_min uses patience
    if not math.isfinite(current_dmin):
        emergency_trigger = True
    elif rearmed and current_dmin > ae_cfg.high_d_min_threshold:
        state["high_dmin_count"] = state["high_dmin_count"] + 1
        emergency_trigger = (
            state["high_dmin_count"] >= ae_cfg.high_d_min_patience
        )
    else:
        state["high_dmin_count"] = 0
        emergency_trigger = False

    # Main trigger
    hybrid_trigger = (
        rearmed
        and state["dmin_stagnant_count"] >= ae_cfg.d_min_patience
        and state["fitness_stagnant_count"] >= ae_cfg.fitness_patience
    )

    triggered = emergency_trigger or hybrid_trigger

    # Reset after trigger
    if triggered:
        state["dmin_stagnant_count"] = 0
        state["fitness_stagnant_count"] = 0
        state["high_dmin_count"] = 0
        state["dmin_rearmed"] = False

    return (
        triggered,
        dmin_range,
        rel_improvement,
    )

def sample_fixed_count(eligible, n, rng, ae_cfg, num_generations):
    pool = rng.permutation(eligible)
    chosen = []
    last = -10**9
    for it in pool:
        chosen.append(int(it))
        last = int(it)
        if len(chosen) == n:
            chosen = sorted(chosen)
            for i in range(1, len(chosen)):
                if chosen[i] - chosen[i - 1] < ae_cfg.adaptive_extinction.cooldown:
                    if chosen[i] + ae_cfg.adaptive_extinction.cooldown > num_generations - 1:
                        chosen[i] = num_generations - 1
                    else:
                        chosen[i] = chosen[i] + ae_cfg.adaptive_extinction.cooldown
            return frozenset(chosen)
    raise ValueError(f"Could only place {len(chosen)}/{n} extinctions; relax cooldown or warmup")


def train(
    cfg: DictConfig,
    aurora_scan_update,
    train_state,
    aurora_extra_info,
    repertoire,
    passive_repertoire,
    aurora,
    random_key,
    logger: MetricsLogger = None,
):

    model_params = aurora_extra_info.model_params
    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - cfg.target_size
    num_generations = cfg.num_iterations // cfg.metrics_log_period
    all_metrics = {}
    min_obs = None
    max_obs = None
    model_metrics = {}
    total_evaluations = 0
    ae_timelapse = 0
    csc_timelapse = 0

    # Design AURORA's schedule
    default_update_base = cfg.default_update_base
    update_base = int(jnp.ceil(default_update_base / cfg.metrics_log_period))
    schedules = jnp.cumsum(jnp.arange(update_base, num_generations, update_base))

    # Pre-compute expected encoder training count for random_encoder_rate mode
    _n_enc = int(jnp.sum(schedules <= num_generations))
    _p_random_ext = float(_n_enc) / max(1, num_generations)

    _top_k_history: collections.deque = collections.deque(
        maxlen=5 * cfg.adaptive_extinction.patience + 1
    )
    _last_extinction_iter: int = 0

    _dmin_history = collections.deque(
        maxlen=cfg.adaptive_extinction.d_min_window
    )

    _fitness_history = collections.deque(
        maxlen=cfg.adaptive_extinction.fitness_window
    )

    rng = np.random.default_rng(cfg.seed + 99991)
    eligible = np.arange(0, num_generations - 1)
    
    _extinction_schedule = sample_fixed_count(
        eligible, n=cfg.target_extinction_count, rng=rng, ae_cfg=cfg, num_generations=num_generations
    )

    _trigger_state = {
        "dmin_stagnant_count": 0,
        "fitness_stagnant_count": 0,
        "high_dmin_count": 0,
        "dmin_rearmed": True,
    }

    for i in range(num_generations):

        start_time = time.time()

        (
            (repertoire, passive_repertoire, random_key, aurora_extra_info),
            metrics,
        ) = jax.lax.scan(
            aurora_scan_update,
            (repertoire, passive_repertoire, random_key, aurora_extra_info),
            (),
            length=cfg.metrics_log_period,
        )

        timelapse = time.time() - start_time

        if min_obs is not None and max_obs is not None:
            min_obs = jnp.minimum(min_obs, metrics["min_obs"])
            max_obs = jnp.maximum(max_obs, metrics["max_obs"])
            del metrics["min_obs"]
            del metrics["max_obs"]
        else:
            min_obs = jnp.min(metrics["min_obs"], axis=0)
            max_obs = jnp.max(metrics["max_obs"], axis=0)
            del metrics["min_obs"]
            del metrics["max_obs"]

        total_evaluations += cfg.metrics_log_period * cfg.batch_size

        # --- per-generation metrics & trigger ---
        actual_gen = (i + 1) * cfg.metrics_log_period
        valid = repertoire.fitnesses != -jnp.inf
        fits_np = np.asarray(repertoire.fitnesses[valid])

        # top-k% mean fitness (one sort; shared by trigger and logger)
        if fits_np.size > 0:
            k_n = max(1, math.ceil(cfg.top_k_percent / 100.0 * fits_np.size))
            mean_topk = float(np.float32(np.sort(fits_np)[-k_n:].mean()))
        else:
            mean_topk = float("nan")
        _top_k_history.append(mean_topk)
        _fitness_history.append(mean_topk)

        # d_min signal (only meaningful for adaptive repertoire)
        if cfg.repertoire == "adaptive":
            _dmin_history.append(float(repertoire.d_min))

        # extinction decision
        _nan = float("nan")
        trigger_info = {
            "recent_rate": _nan,
            "historical_rate": _nan,
            "ratio": _nan,
            "triggered": False,
        }
        dmin_trigger_info = {
            "recent_volatility": _nan,
            "historical_volatility": _nan,
            "ratio": _nan,
            "triggered": False,
        }
        is_ext = False
        if i != num_generations - 1:
            if cfg.extinction_mode == "static":
                is_ext = (i + 1) % cfg.extinction_freq == 0 
            elif cfg.extinction_mode == "fitness_trigger":
                fired, rr, hr, ratio = _check_fitness_trigger(
                    _top_k_history, _last_extinction_iter, i, cfg.adaptive_extinction
                )
                trigger_info = {
                    "recent_rate": rr,
                    "historical_rate": hr,
                    "ratio": ratio,
                    "triggered": fired,
                }
                is_ext = fired
            elif cfg.extinction_mode == "d_min_trigger":
                assert cfg.repertoire == "adaptive", (
                    "extinction_mode=d_min_trigger requires repertoire=adaptive"
                )

                fired, dmin_range, fit_improvement = _check_hybrid_trigger(
                    dmin_history=_dmin_history,
                    fitness_history=_fitness_history,
                    last_ext_iter=_last_extinction_iter,
                    current_iter=i,
                    state=_trigger_state,
                    ae_cfg=cfg.adaptive_extinction,
                )

                # metrics_logger / wandb reuse volatility column names for these scalars
                dmin_trigger_info = {
                    "recent_volatility": dmin_range,
                    "historical_volatility": fit_improvement,
                    "ratio": _nan,
                    "triggered": fired,
                }

                is_ext = fired
            elif cfg.extinction_mode == "static_ramped_proportion":
                is_ext = (i + 1) % cfg.extinction_freq == 0
            elif cfg.extinction_mode == "encoder_post":
                # Logged as extinction on this iteration; applied after encoder retrains below.
                is_ext = bool((i + 1) in schedules) and not cfg.no_training
            elif cfg.extinction_mode == "encoder_pre":
                # Logged as extinction on this iteration; applied after encoder retrains below.
                is_ext = bool((i - 1) in schedules) and not cfg.no_training
            elif cfg.extinction_mode == "random_encoder_rate":
                # Stochastic trigger with same expected frequency as encoder training.
                random_key, subkey = jax.random.split(random_key)
                is_ext = bool(jax.random.bernoulli(subkey, p=_p_random_ext))
            
            elif cfg.extinction_mode == "random_fixed_count":
                is_ext = i in _extinction_schedule


        # Compute remaining_prop for this potential extinction event.
        # static_ramped_proportion ramps from prop_min to prop_max over training.
        _remaining_prop = cfg.remaining_prop
        if cfg.extinction_mode == "static_ramped_proportion" and is_ext:
            progress = min(actual_gen / cfg.num_iterations, 1.0)
            _remaining_prop = float(
                cfg.adaptive_extinction.ramped_prop_min
                + (cfg.adaptive_extinction.ramped_prop_max - cfg.adaptive_extinction.ramped_prop_min)
                * progress
            )

        answer_set_metrics = None
        if logger is not None and bool(getattr(cfg, "log_answer_set_geometry", True)):
            Z = np.asarray(repertoire.descriptors[valid])
            answer_set_metrics = latent_answer_set_metrics(Z, logger._rng)
            if cfg.repertoire == "adaptive":
                answer_set_metrics["repertoire_d_min"] = float(np.asarray(repertoire.d_min).item())
            else:
                answer_set_metrics["repertoire_d_min"] = float("nan")

        if logger is not None:
            logger.log_generation(
                actual_gen, fits_np, is_ext, mean_topk, trigger_info,
                archive_size=fits_np.size,
                dmin_trigger_info=dmin_trigger_info,
                extinction_remaining_prop=_remaining_prop if is_ext else None,
                answer_set_metrics=answer_set_metrics,
            )

        # Apply extinction before encoder (all modes except encoder_post)
        if is_ext and (cfg.extinction_mode != "encoder_post" or cfg.extinction_mode != "encoder_pre"):
            logging.info("Extinction event...")
            random_key, subkey = jax.random.split(random_key)
            repertoire = repertoire.extinction(
                remaining_prop=_remaining_prop, random_key=subkey
            )
            _last_extinction_iter = i

        # AE
        if (i + 1) in schedules and not cfg.no_training:
            logging.info("Updating AE...")
            start_time = time.time()
            # cache encoder state for drift computation
            if logger is not None:
                _old_extra_info = aurora_extra_info
            # train the autoencoder
            random_key, subkey = jax.random.split(random_key)
            if cfg.reinit_params:
                train_state = train_state.replace(params=model_params)
            repertoire, train_state, aurora_extra_info, model_metrics = aurora.train(
                repertoire, train_state, subkey
            )
            ae_timelapse = time.time() - start_time

            # encoder_post: apply extinction after encoder retraining
            if is_ext and cfg.extinction_mode == "encoder_post":
                logging.info("Extinction event (post-encoder)...")
                random_key, subkey = jax.random.split(random_key)
                repertoire = repertoire.extinction(
                    remaining_prop=_remaining_prop, random_key=subkey
                )
                _last_extinction_iter = i
            metrics_last_iter = jax.tree_util.tree_map(
                lambda metric: "{0:.4f}".format(float(metric[-1])), model_metrics
            )

            logging.info(metrics_last_iter)
            if logger is not None:
                _valid = repertoire.fitnesses != -jnp.inf
                _descs = np.asarray(repertoire.descriptors[_valid])
                logger.log_encoder_retrain(
                    generation=(i + 1) * cfg.metrics_log_period,
                    model_metrics=model_metrics,
                    old_aurora_extra_info=_old_extra_info,
                    new_aurora_extra_info=aurora_extra_info,
                    encoder_fn=aurora._encoder_fn,
                    archived_descriptors=_descs,
                )
        
        # ENCODER PRE
        if (i - 1) in schedules and not cfg.no_training:
            logging.info("Updating AE...")
            start_time = time.time()
            # cache encoder state for drift computation
            if logger is not None:
                _old_extra_info = aurora_extra_info
            # train the autoencoder
            random_key, subkey = jax.random.split(random_key)
            if cfg.reinit_params:
                train_state = train_state.replace(params=model_params)
            repertoire, train_state, aurora_extra_info, model_metrics = aurora.train(
                repertoire, train_state, subkey
            )
            ae_timelapse = time.time() - start_time

            # encoder_post: apply extinction after encoder retraining
            if is_ext and cfg.extinction_mode == "encoder_pre":
                logging.info("Extinction event (pre-encoder)...")
                random_key, subkey = jax.random.split(random_key)
                repertoire = repertoire.extinction(
                    remaining_prop=_remaining_prop, random_key=subkey
                )
                _last_extinction_iter = i
            metrics_last_iter = jax.tree_util.tree_map(
                lambda metric: "{0:.4f}".format(float(metric[-1])), model_metrics
            )

            logging.info(metrics_last_iter)
            if logger is not None:
                _valid = repertoire.fitnesses != -jnp.inf
                _descs = np.asarray(repertoire.descriptors[_valid])
                logger.log_encoder_retrain(
                    generation=(i + 1) * cfg.metrics_log_period,
                    model_metrics=model_metrics,
                    old_aurora_extra_info=_old_extra_info,
                    new_aurora_extra_info=aurora_extra_info,
                    encoder_fn=aurora._encoder_fn,
                    archived_descriptors=_descs,
                )

        # CSC
        elif i % 2 == 0 and not cfg.no_csc:
            if cfg.repertoire == "adaptive":
                pass
            else:
                start_time = time.time()
                repertoire, previous_error = aurora.container_size_control(
                    repertoire,
                    target_size=cfg.target_size,
                    prop_gain=cfg.prop_gain,
                    previous_error=previous_error,
                    csc_orig=cfg.csc_orig,
                )
                csc_timelapse = time.time() - start_time

        # Log metrics
        logged_metrics = {
            "time": timelapse + ae_timelapse + csc_timelapse,
            "time_qd": timelapse,
            "time_ae": ae_timelapse,
            "time_csc": csc_timelapse,
            "evaluations": total_evaluations,
            "iteration": 1 + i * cfg.metrics_log_period,
            "env_steps": total_evaluations * cfg.env.episode_length,
        }
        if cfg.repertoire == "adaptive":
            logged_metrics["d_min"] = repertoire.d_min
        else:
            logged_metrics["l_value"] = repertoire.l_value

        logged_metrics["mean_fitness_top_k"] = mean_topk

        if cfg.extinction_mode == "d_min_trigger":
            logged_metrics["d_min_recent_volatility"] = dmin_trigger_info["recent_volatility"]
            logged_metrics["d_min_historical_volatility"] = dmin_trigger_info["historical_volatility"]
            logged_metrics["d_min_volatility_ratio"] = dmin_trigger_info["ratio"]
            logged_metrics["d_min_trigger_fired"] = float(dmin_trigger_info["triggered"])
        if is_ext:
            logged_metrics["extinction_remaining_prop"] = _remaining_prop
            logged_metrics["extinction_iter"] = actual_gen

        logging.info(
            f"Generation {i + 1}/{num_generations} - Time: {timelapse:.2f} seconds"
        )
        metrics_last_iter = jax.tree_util.tree_map(
            lambda metric: "{0:.2f}".format(float(metric[-1])), metrics
        )
        metrics_last_iter["mean_fitness_top_k"] = f"{mean_topk:.2f}"
        logging.info(metrics_last_iter)
        if (i + 1) in schedules and not cfg.no_training:
            # Log all AE training metrics
            logged_metrics = {
                **logged_metrics,
                **{key: jnp.mean(value) for key, value in model_metrics.items()},
            }

        logged_metrics, all_metrics = log_running_metrics(
            metrics, logged_metrics, all_metrics, step=total_evaluations
        )

    return all_metrics, repertoire, passive_repertoire, (min_obs, max_obs)


# usage python -m train.aurora_trainer --config-name aurora.yaml env=brax/ant_maze
@hydra.main(config_path="../configs", config_name="aurora.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Start training run.

    Args:
        cfg (DictConfig): master config for the hydra_launcher
    """

    cfg = set_env_params(cfg)
    logging.info(cfg)
    logging.info("Training")
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    with wandb.init(config=wandb_cfg, **cfg.wandb):

        # Create local results directory for visualizations and repertoire
        results_dir = os.path.join(os.getcwd(), "results")
        _repertoire_dir = os.path.join(results_dir, "repertoire", cfg.wandb.name, "")
        _passive_archive_dir = os.path.join(
            results_dir, "passive_archive", cfg.wandb.name, ""
        )
        _best_individual_dir = Path(results_dir) / "best_individual"
        os.makedirs(_repertoire_dir, exist_ok=True)
        os.makedirs(_best_individual_dir, exist_ok=True)
        os.makedirs(_passive_archive_dir, exist_ok=True)

        # Init a random key
        random_key = jax.random.PRNGKey(cfg.seed)
        random_key, subkey = jax.random.split(random_key)

        # Init environment
        env = get_env(cfg.env)
        observations_dims = get_observation_dims(
            cfg,
            env.observation_size,
            (env.aurora_observation_size if cfg.env.name == "kheperax" else None),
        )

        # Init policy network
        policy_layer_sizes = tuple(cfg.env.policy_hidden_layer_size) + (
            env.action_size,
        )
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        keys = jnp.repeat(
            jnp.expand_dims(subkey, axis=0), repeats=cfg.batch_size, axis=0
        )
        reset_fn = jax.jit(jax.vmap(env.reset))
        init_states = reset_fn(keys)

        step_fn = functools.partial(
            play_step_fn, policy_network=policy_network, env=env
        )

        # Prepare the scoring function
        bd_extraction_fn = functools.partial(
            task_behavior_descriptor_extractor[cfg.env.name]["function"],
            **task_behavior_descriptor_extractor[cfg.env.name]["args"],
        )
        scoring_fn = functools.partial(
            scoring_function,
            cfg=cfg,
            init_states=init_states,
            play_step_fn=step_fn,
            behavior_descriptor_extractor=bd_extraction_fn,
        )

        def observation_extractor_fn(
            data,
        ):
            """Extract observation from the state."""
            data, final_state = data
            if cfg.env.observation_extraction.observation_option == "images":
                observations = final_state.info["image_obs"]

            elif cfg.env.observation_extraction.observation_option == "sensory_data":
                # state_obs: (batch_size, traj_length//sampling_freq, obs_size (max_obs_size))
                observations = data.obs[
                    :,
                    :: cfg.env.observation_extraction.sampling_freq,
                    : cfg.env.observation_extraction.max_obs_size,
                ]
            else:
                raise ValueError("Unknown observation option.")

            return observations

        aurora_scoring_fn = get_aurora_scoring_fn(
            scoring_fn=scoring_fn,
            observation_extractor_fn=observation_extractor_fn,
        )

        # Init population of controllers
        init_variables, random_key = init_population(
            cfg, env, policy_network, random_key
        )
        params_count = int(
            sum(x.size for x in jax.tree_util.tree_leaves(init_variables))
            / cfg.batch_size
        )
        logging.info(f"Policy params count (search space): {params_count}")

        # Get minimum reward value to make sure qd_score are positive
        if cfg.env.name == "kheperax":
            reward_offset = jnp.sqrt(2) * 100
        else:
            if cfg.env.version == "v1":
                reward_offset = environments_v1.reward_offset[cfg.env.qdax_name]
            else:
                reward_offset = environments.reward_offset[cfg.env.qdax_name]

        # Define a metrics function
        metrics_fn = functools.partial(
            passive_qd_metrics,
            qd_offset=(
                reward_offset * cfg.env.episode_length
                if cfg.env.name != "kheperax"
                else reward_offset
            ),
            target_size=cfg.target_size if cfg.repertoire == "threshold" else None,
        )

        # Define emitter
        variation_fn = functools.partial(
            isoline_variation, iso_sigma=cfg.iso_sigma, line_sigma=cfg.line_sigma
        )
        mixing_emitter = MixingEmitter(
            mutation_fn=lambda x, y: (x, y),
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=cfg.batch_size,
        )

        # Auto-encoder init
        encoder_fn, train_fn, train_state, aurora_extra_info = (
            init_autoencoder_model_training(cfg, observations_dims, random_key)
        )
        params_count = sum(
            x.size for x in jax.tree_util.tree_leaves(train_state.params)
        )
        logging.info(f"AE params count: {params_count}")

        @jax.jit
        def update_scan_fn(carry: Any, unused: Any) -> Any:
            """Scan the udpate function."""
            (repertoire, passive_repertoire, random_key, aurora_extra_info) = carry

            # update
            (
                repertoire,
                passive_repertoire,
                _,
                metrics,
                random_key,
            ) = aurora.update(
                repertoire,
                passive_repertoire,
                None,
                random_key,
                aurora_extra_info=aurora_extra_info,
            )

            return (
                (repertoire, passive_repertoire, random_key, aurora_extra_info),
                metrics,
            )

        # Init algorithm
        logging.info("Algorithm initialization")
        init_time = time.time()
        random_key, subkey = jax.random.split(random_key)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=env.state_descriptor_length,
            num_init_cvt_samples=cfg.env.num_init_cvt_samples,
            num_centroids=cfg.env.num_centroids,
            minval=cfg.env.min_bd,
            maxval=cfg.env.max_bd,
            random_key=subkey,
        )

        # Instantiate AURORA
        if cfg.repertoire == "adaptive":
            aurora = AURORAAdaptivePassive(
                scoring_function=aurora_scoring_fn,
                emitter=mixing_emitter,
                metrics_function=metrics_fn,
                encoder_function=encoder_fn,
                training_function=train_fn,
            )
            # init step of the aurora algorithm
            (
                repertoire,
                passive_repertoire,
                emitter_state,
                aurora_extra_info,
                random_key,
            ) = aurora.init(
                init_variables,
                centroids,
                aurora_extra_info,
                cfg.max_size,
                random_key,
            )
        elif cfg.repertoire == "threshold":
            aurora = AURORAThresholdPassive(
                scoring_function=aurora_scoring_fn,
                emitter=mixing_emitter,
                metrics_function=metrics_fn,
                encoder_function=encoder_fn,
                training_function=train_fn,
            )
            # init step of the aurora algorithm
            (
                repertoire,
                passive_repertoire,
                emitter_state,
                aurora_extra_info,
                random_key,
            ) = aurora.init(
                init_variables,
                centroids,
                aurora_extra_info,
                jnp.asarray(cfg.l_value_init),
                cfg.max_size,
                random_key,
            )

        init_repertoire_time = time.time() - init_time
        logging.info(f"Repertoire initialized in {init_repertoire_time:.2f} seconds")

        # Init means, stds and AURORA
        random_key, subkey = jax.random.split(random_key)
        repertoire, train_state, aurora_extra_info, _ = aurora.train(
            repertoire, train_state, random_key=subkey
        )

        # Set up trigger signal logger
        _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _run_dir = Path("runs") / f"{_ts}_{cfg.env.name}_seed{cfg.seed}"
        logger = MetricsLogger(run_dir=_run_dir, seed=cfg.seed)
        logger.set_drift_sample(repertoire, aurora_extra_info, aurora._encoder_fn)
        OmegaConf.save(
            OmegaConf.create({
                "env": cfg.env.name,
                "seed": int(cfg.seed),
                "extinction_mode": str(cfg.extinction_mode),
                "top_k_percent": int(cfg.top_k_percent),
                "adaptive_extinction": OmegaConf.to_container(cfg.adaptive_extinction),
                "reward_type": str(cfg.env.reward_type),
                "speed_bonus_factor": float(getattr(cfg.env, "speed_bonus_factor", 1.0)),
                "log_answer_set_geometry": bool(getattr(cfg, "log_answer_set_geometry", True)),
                "target_extinction_count": int(cfg.target_extinction_count),
            }),
            str(_run_dir / "config.yaml"),
        )
        logging.info(f"Trigger signal logger writing to {_run_dir.resolve()}")

        metrics, repertoire, passive_repertoire, obs_data = train(
            cfg=cfg,
            aurora_scan_update=update_scan_fn,
            train_state=train_state,
            aurora_extra_info=aurora_extra_info,
            repertoire=repertoire,
            passive_repertoire=passive_repertoire,
            aurora=aurora,
            random_key=random_key,
            logger=logger,
        )

        logger.flush()
        logging.info(f"Trigger signal log saved to {_run_dir}")

        total_duration = time.time() - init_time

        # Log metrics for the final repertoire
        log_final_metrics(
            cfg=cfg,
            metrics=metrics,
            total_duration=total_duration,
            repertoire=repertoire,
            passive_repertoire=passive_repertoire,
        )

        # Save visualisation of best individual
        viz_filename = (
            Path(_best_individual_dir) / (cfg.wandb.name + ".html")
            if cfg.env.name != "kheperax"
            else Path(_best_individual_dir) / (cfg.wandb.name + ".gif")
        )

        if cfg.env.name == "kheperax":
            kheperax_viz_best_individual(
                env,
                policy_network,
                repertoire,
                path=viz_filename,
                bd_extractor=bd_extraction_fn,
            )
        else:
            viz_best_individual(
                env,
                policy_network,
                repertoire,
                path=viz_filename,
                bd_extractor=bd_extraction_fn,
            )

        # Save final repertoire & AE
        repertoire.save(_repertoire_dir)
        passive_repertoire.save(_passive_archive_dir)
        # Log min/max obs data
        jnp.save(
            os.path.join(_repertoire_dir, "min_obs.npy"),
            obs_data[0],
        )
        jnp.save(
            os.path.join(_repertoire_dir, "max_obs.npy"),
            obs_data[1],
        )


if __name__ == "__main__":
    main()
