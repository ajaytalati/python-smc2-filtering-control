"""FSA v1.5 closed-loop SMC²-MPC bench, purely-functional rewrite.

Mirrors `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` and
v2's `bench_smc_full_mpc_fsa.py` but slimmed to v1.5's surface:
3-state (B, F, A), single Φ control, 3 direct-Gaussian obs channels.

Pipeline per stride:
    1. Apply current Φ schedule to the plant for `STRIDE_BINS` bins.
    2. Append the rollout's trajectory + obs to the accumulators.
    3. (Once `n_obs >= WINDOW_BINS`) run the outer SMC² filter on the
       latest window. The first run is a prior-draw, subsequent runs
       use the Schrödinger-Föllmer bridge from the prior posterior.
    4. (Every `replan_K` strides, after the filter has fired) run the
       tempered-SMC controller against the posterior-mean parameters
       and overwrite `daily_phi_plan` with the full-horizon replan.
    5. Emit a per-stride telemetry record.

Functional structure:

    *  All cross-stride state lives in immutable `typing.NamedTuple`
       records: `BenchConfig` (CLI), `BenchEnv` (time-invariant setup),
       `BenchState` (mutates across strides — but here threaded as an
       immutable record), `BenchAccumulators` (per-stride records as
       tuples), and `StrideTelemetry` (one entry per stride).
    *  No `for`/`while` loops over strides — `run_bench` uses
       `functools.reduce(_stride_with_progress, …)` to fold the per-
       stride state forward.
    *  The pure stride body `_stride_body` composes three pure helpers
       (`_advance_plant_one_stride`, `_filter_one_stride`,
       `_replan_one_stride`) and returns `(BenchState, BenchAccumulators)`.
    *  All I/O (printing, file writes, matplotlib calls) is confined to
       a thin impure shell: `_stride_with_progress`, `_save_artifacts`,
       and `main`.

CLI:
    python tools/bench_smc_full_mpc_fsa_v15.py [--T-days 14] [--smoke]
"""

from __future__ import annotations

import os
from pathlib import Path as _BootPath
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# Persistent on-disk JAX compile cache so the first run's HLO is
# reused across processes (matches CLAUDE.md's bench-driver convention).
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    str(_BootPath.home() / '.jax_compilation_cache'),
)

import argparse
import csv
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


# ── CLI parsing ───────────────────────────────────────────────────────


class BenchConfig(NamedTuple):
    """User-specified bench knobs, frozen up-front from the CLI.

    Every downstream helper takes a `BenchConfig` and never mutates it,
    so a single `parse_args` call fully determines the run.

    Attributes:
        T_days: Total horizon length in days.
        replan_K: Replan cadence — controller fires every `K` strides.
        n_smc: Filter outer SMC² particle count.
        k_pf: Filter inner particle-filter chain length.
        ctrl_n_smc: Controller outer SMC² particle count.
        ctrl_n_inner: Controller inner CRN-MC trials per cost evaluation.
        ctrl_num_mcmc: Controller HMC moves per tempering level.
        ctrl_hmc_step: Controller HMC leapfrog step size.
        ctrl_hmc_leap: Controller HMC leapfrog trajectory length.
        ctrl_target_nats: Controller β_max auto-calibration target (nats).
        ctrl_max_levels: Controller hard cap on tempering levels per replan.
        ctrl_max_lambda_inc: Controller max λ increment per tempering bisection.
        ctrl_sigma_prior: Controller prior std on θ_ctrl RBF coefficients.
        ctrl_n_anchors: Number of RBF anchors in the controller's
            schedule basis (controller posterior dimension).
        seed: Top-level seed; per-stride keys are derived deterministically.
        smoke: If True, run only one stride end-to-end as a sanity check.
        out_dir: Output directory for trajectory.npz / manifest.json /
            plots; empty string means auto-generate under `outputs/fsa_v15/`.
    """
    T_days: int
    replan_K: int
    n_smc: int
    k_pf: int
    ctrl_n_smc: int
    ctrl_n_inner: int
    ctrl_num_mcmc: int
    ctrl_hmc_step: float
    ctrl_hmc_leap: int
    ctrl_target_nats: float
    ctrl_max_levels: int
    ctrl_max_lambda_inc: float
    ctrl_sigma_prior: float
    ctrl_n_anchors: int
    seed: int
    smoke: bool
    out_dir: str


def parse_args(argv: Optional[list[str]] = None) -> BenchConfig:
    """Parses CLI arguments into a frozen `BenchConfig`.

    Args:
        argv: Command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        A `BenchConfig` immutably capturing every knob. Defaults match
        the v1.5 Julia bench `bench_smc_full_mpc_fsa_gpu.jl` so a no-flag
        invocation produces the same configuration on both stacks.
    """
    ap = argparse.ArgumentParser(description='FSA v1.5 closed-loop SMC²-MPC')
    ap.add_argument('--T-days', type=int, default=14,
                    help='total horizon in days')
    ap.add_argument('--replan-K', type=int, default=2,
                    help='replan every K strides')
    ap.add_argument('--n-smc', type=int, default=32,
                    help='filter outer SMC² particles')
    ap.add_argument('--k-pf', type=int, default=200,
                    help='filter inner PF particles per chain')
    ap.add_argument('--ctrl-n-smc', type=int, default=256,
                    help='controller outer SMC² particles')
    ap.add_argument('--ctrl-n-inner', type=int, default=64,
                    help='controller inner CRN-MC trials per cost evaluation')
    ap.add_argument('--ctrl-num-mcmc', type=int, default=8,
                    help='controller HMC moves per tempering level')
    ap.add_argument('--ctrl-hmc-step', type=float, default=0.2,
                    help='controller HMC leapfrog step size')
    ap.add_argument('--ctrl-hmc-leap', type=int, default=16,
                    help='controller HMC leapfrog trajectory length')
    ap.add_argument('--ctrl-target-nats', type=float, default=8.0,
                    help='controller β_max auto-calibration target (nats)')
    ap.add_argument('--ctrl-max-levels', type=int, default=25,
                    help='controller hard cap on tempering levels per replan')
    ap.add_argument('--ctrl-max-lambda-inc', type=float, default=0.20,
                    help='controller max λ increment per tempering bisection')
    ap.add_argument('--ctrl-sigma-prior', type=float, default=1.5,
                    help='controller prior std on θ_ctrl RBF coefficients')
    ap.add_argument('--ctrl-n-anchors', type=int, default=12,
                    help='controller RBF basis cardinality (default 12, '
                         'matches Julia best-config)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true',
                    help='single stride only — sanity check that imports + '
                         'first-call paths work')
    ap.add_argument('--out-dir', type=str, default='',
                    help='output dir; empty → auto under outputs/fsa_v15/')
    a = ap.parse_args(argv)
    return BenchConfig(
        T_days=a.T_days, replan_K=a.replan_K,
        n_smc=a.n_smc, k_pf=a.k_pf,
        ctrl_n_smc=a.ctrl_n_smc, ctrl_n_inner=a.ctrl_n_inner,
        ctrl_num_mcmc=a.ctrl_num_mcmc, ctrl_hmc_step=a.ctrl_hmc_step,
        ctrl_hmc_leap=a.ctrl_hmc_leap,
        ctrl_target_nats=a.ctrl_target_nats,
        ctrl_max_levels=a.ctrl_max_levels,
        ctrl_max_lambda_inc=a.ctrl_max_lambda_inc,
        ctrl_sigma_prior=a.ctrl_sigma_prior,
        ctrl_n_anchors=a.ctrl_n_anchors,
        seed=a.seed, smoke=a.smoke, out_dir=a.out_dir,
    )


# ── Bench-time invariants and threaded state ──────────────────────────


class BenchEnv(NamedTuple):
    """Time-invariant bench setup: built once before any stride fires.

    Built by `_build_env(cfg)` and referenced (never mutated) by every
    stride helper.

    Attributes:
        em: The `smc2fc.estimation_model.EstimationModel` instance for
            FSA v1.5 (constructed in `models.fsa_high_res.estimation`).
        smc_cfg: Filter SMC² configuration.
        ctrl_cfg: Controller SMC² configuration.
        log_density_factory: Compile-once GK-DPF v3-lite log-density
            builder; stride wrappers bind dynamic data via `Partial`.
        T_arr: The `(unconstrained → constrained)` transform array
            attached to `log_density_factory`.
        BINS_PER_DAY: Time-grid resolution (e.g. 24 for hourly steps).
        DT_BIN_DAYS: Bin width in days (`1 / BINS_PER_DAY`).
        STRIDE_BINS: Per-stride plant horizon (`BINS_PER_DAY // 2`).
        WINDOW_BINS: Filter window length in bins (`BINS_PER_DAY`).
        n_strides: Total number of strides this bench will execute.
        param_names: Filter-vector parameter names (used in artefact
            saving).
        n_params: Number of estimated parameters (= len(param_names)).
        DEFAULT_PARAMS: Truth `ParamsV15` NamedTuple from the model.
        cold_start_init: Cold-start latent init `(B, F, A)` array.
        ctrl_spec_factory: Compile-once controller factory built from
            `build_control_spec_compileonce`. Per-replan dynamic data
            (`params_v15`, `init_state`) is bound via Partial inside
            this factory; the underlying JIT'd `cost_kernel` is
            traced ONCE and reused across all replans, eliminating
            the per-replan XLA recompile that dominated wall time.
    """
    em: Any
    smc_cfg: Any
    ctrl_cfg: Any
    log_density_factory: Any
    T_arr: Any
    BINS_PER_DAY: int
    DT_BIN_DAYS: float
    STRIDE_BINS: int
    WINDOW_BINS: int
    n_strides: int
    param_names: Tuple[str, ...]
    n_params: int
    DEFAULT_PARAMS: Any
    cold_start_init: Any
    ctrl_spec_factory: Any


class BenchState(NamedTuple):
    """Per-stride threaded state, carried as an immutable record.

    Attributes:
        plant_state: Current physical state of the simulator (immutable
            `PlantState` NamedTuple from the model).
        daily_phi_plan: Length-`T_days` vector of daily-mean Φ values
            currently in force; the plant reads from this each stride.
        last_replan_stride: Stride at which the controller last fired,
            used to compute `day_in_plan` (the day-offset within the
            planning horizon).
        prev_particles: Previous-window posterior particles from the
            filter, or None until the first window has fired. The next
            filter call uses these to seed the bridge proposal.
        fixed_init_state: End-of-window smoothed latent state, used by
            both the next filter window and the next replan as the
            "fair" current latent (avoids peeking at `plant_state`).
        key: Master JAX PRNGKey; per-stride keys are split off this.
    """
    plant_state: Any
    daily_phi_plan: np.ndarray
    last_replan_stride: int
    prev_particles: Optional[np.ndarray]
    fixed_init_state: Any
    key: Any


class StrideTelemetry(NamedTuple):
    """One row of the per-stride log; written to `per_stride.csv` at end.

    Attributes:
        stride: Zero-based stride index.
        t_wall_s: Wall-clock elapsed for this stride (seconds).
        n_temp_filter: Filter tempering levels this stride (0 if filter
            did not fire — i.e. warmup).
        n_temp_ctrl: Controller tempering levels this stride (0 if no
            replan).
        daily_phi: Φ value applied across this stride's plant bins.
        A_mean_so_far: Rolling mean of A across all accumulated traj.
        B_end: Last bin's B latent.
        F_end: Last bin's F latent.
        A_end: Last bin's A latent.
        filter_elapsed_s: Filter window wall time, or 0.0 if no fire.
        replan_plan_per_day: Tuple of new daily Φ values from the
            controller (empty tuple if no replan this stride).
    """
    stride: int
    t_wall_s: float
    n_temp_filter: int
    n_temp_ctrl: int
    daily_phi: float
    A_mean_so_far: float
    B_end: float
    F_end: float
    A_end: float
    filter_elapsed_s: float
    replan_plan_per_day: Tuple[float, ...]


class BenchAccumulators(NamedTuple):
    """Per-stride records gathered as tuples (immutable, append-only).

    Each `_replace`-style append produces a new `BenchAccumulators`;
    no field is ever mutated in place. At end-of-bench these tuples
    are stacked / concatenated for artefact saving.

    Attributes:
        traj_chunks: Tuple of per-stride trajectory arrays of shape
            `(STRIDE_BINS, 3)`. `np.concatenate` flattens to the full
            trajectory at the end.
        obs_B_chunks: Tuple of per-stride `obs_B` arrays of length
            `STRIDE_BINS` each (same for `_F`, `_A`, and `Phi`).
        obs_F_chunks: As above for the F channel.
        obs_A_chunks: As above for the A channel.
        Phi_chunks: As above for the applied Φ schedule.
        daily_phi_per_stride: Tuple of one Φ value per stride.
        replan_records: Tuple of one dict per replan event; saved to
            `manifest.json` for postmortem analysis.
        per_stride_log: Tuple of `StrideTelemetry`, one entry per
            executed stride.
        posterior_chunks: Tuple of `Optional[np.ndarray]` per stride
            (None for warmup strides). Filtered strides store an
            `(n_smc, n_params)` constrained-posterior particle cloud.
        posterior_mask: Tuple of bools, parallel to `posterior_chunks`,
            True iff the filter actually fired on that stride.
        A_sum: Running sum of A across every plant-rollout bin so far.
            Used to compute the per-stride telemetry's
            `A_mean_so_far` in O(1) (instead of re-concatenating the
            full trajectory every stride and re-meaning, which was
            O(n²) and forced a device→host sync per stride).
        n_bins_so_far: Total bins of plant rollout accumulated so far.
            Pairs with `A_sum` for the running mean.
    """
    traj_chunks: Tuple[np.ndarray, ...]
    obs_B_chunks: Tuple[np.ndarray, ...]
    obs_F_chunks: Tuple[np.ndarray, ...]
    obs_A_chunks: Tuple[np.ndarray, ...]
    Phi_chunks: Tuple[np.ndarray, ...]
    daily_phi_per_stride: Tuple[float, ...]
    replan_records: Tuple[dict, ...]
    per_stride_log: Tuple[StrideTelemetry, ...]
    posterior_chunks: Tuple[Optional[np.ndarray], ...]
    posterior_mask: Tuple[bool, ...]
    A_sum: float
    n_bins_so_far: int


def _empty_accumulators() -> BenchAccumulators:
    """Returns a fresh `BenchAccumulators` with every tuple empty."""
    return BenchAccumulators(
        traj_chunks=(), obs_B_chunks=(), obs_F_chunks=(),
        obs_A_chunks=(), Phi_chunks=(),
        daily_phi_per_stride=(), replan_records=(),
        per_stride_log=(), posterior_chunks=(), posterior_mask=(),
        A_sum=0.0, n_bins_so_far=0,
    )


# ── Environment construction ──────────────────────────────────────────


def _build_env(cfg: BenchConfig) -> BenchEnv:
    """Builds the bench-time invariants from the CLI config.

    This function has unavoidable side effects: it touches the JAX
    device list, JIT-compiles the log-density factory, and reads
    module-level constants from the model. It is run exactly once,
    before any stride fires.

    Args:
        cfg: Frozen CLI configuration.

    Returns:
        A `BenchEnv` that captures the estimation model, both SMC
        configs, the compiled log-density factory + transform array,
        and the time-grid constants.
    """
    # Imports are local to this function so the env-vars at module top
    # (JAX_ENABLE_X64, XLA_PYTHON_CLIENT_PREALLOCATE) are guaranteed
    # set before any JAX import resolves.
    from models.fsa_high_res.simulation import (
        DEFAULT_PARAMS, BINS_PER_DAY, DT_BIN_DAYS,
    )
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V15_ESTIMATION, COLD_START_INIT,
    )
    from smc2fc.core.config import SMCConfig
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density_compileonce,
    )
    from smc2fc.control import SMCControlConfig

    em = HIGH_RES_FSA_V15_ESTIMATION
    STRIDE_BINS = BINS_PER_DAY // 2
    WINDOW_BINS = BINS_PER_DAY
    n_strides = (cfg.T_days * BINS_PER_DAY) // STRIDE_BINS
    if cfg.smoke:
        n_strides = 1

    smc_cfg = SMCConfig(
        n_smc_particles=cfg.n_smc, n_pf_particles=cfg.k_pf,
        target_ess_frac=0.5, max_lambda_inc=0.20,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed',
        sf_use_q0_cov=True, sf_blend=0.7,
        sf_annealed_n_stages=3, sf_annealed_n_mh_steps=5,
        sf_info_aware=False,
        num_mcmc_steps=3, hmc_step_size=0.05, hmc_num_leapfrog=4,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.20,
    )
    ctrl_cfg = SMCControlConfig(
        n_smc=cfg.ctrl_n_smc,
        n_inner=cfg.ctrl_n_inner,
        sigma_prior=cfg.ctrl_sigma_prior,
        target_ess_frac=0.5,
        max_lambda_inc=cfg.ctrl_max_lambda_inc,
        num_mcmc_steps=cfg.ctrl_num_mcmc,
        hmc_step_size=cfg.ctrl_hmc_step,
        hmc_num_leapfrog=cfg.ctrl_hmc_leap,
        beta_max_target_nats=cfg.ctrl_target_nats,
        max_temp_steps=cfg.ctrl_max_levels,
    )

    log_density_factory = make_gk_dpf_v3_lite_log_density_compileonce(
        model=em, n_particles=smc_cfg.n_pf_particles,
        bandwidth_scale=smc_cfg.bandwidth_scale,
        ot_ess_frac=smc_cfg.ot_ess_frac,
        ot_temperature=smc_cfg.ot_temperature,
        ot_max_weight=smc_cfg.ot_max_weight,
        ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
        ot_epsilon=smc_cfg.ot_epsilon,
        dt=DT_BIN_DAYS, t_steps=WINDOW_BINS,
    )

    # Compile-once controller factory. The JIT'd cost kernel is
    # traced ONCE here; per-replan dynamic data (params_v15,
    # init_state) is bound via Partial inside the factory at call
    # time, and JAX's trace cache reuses the same HLO across replans.
    from models.fsa_high_res.control import build_control_spec_compileonce
    ctrl_spec_factory = build_control_spec_compileonce(
        T_total=float(cfg.T_days), dt_days=DT_BIN_DAYS,
        n_anchors=cfg.ctrl_n_anchors, n_inner=cfg.ctrl_n_inner,
        sigma_prior=cfg.ctrl_sigma_prior,
        seed=cfg.seed,
    )

    return BenchEnv(
        em=em, smc_cfg=smc_cfg, ctrl_cfg=ctrl_cfg,
        log_density_factory=log_density_factory,
        T_arr=log_density_factory._transforms,
        BINS_PER_DAY=BINS_PER_DAY, DT_BIN_DAYS=DT_BIN_DAYS,
        STRIDE_BINS=STRIDE_BINS, WINDOW_BINS=WINDOW_BINS,
        n_strides=n_strides,
        param_names=tuple(em.all_names),
        n_params=em.n_params,
        DEFAULT_PARAMS=DEFAULT_PARAMS,
        cold_start_init=COLD_START_INIT,
        ctrl_spec_factory=ctrl_spec_factory,
    )


def _initial_state(env: BenchEnv, cfg: BenchConfig) -> BenchState:
    """Builds the bench's pre-stride state (no obs accumulated yet).

    Args:
        env: Bench-time invariants (used for `cold_start_init`).
        cfg: User config (used for `T_days`, `seed`).

    Returns:
        A `BenchState` with the plant at its canonical init, a
        baseline daily Φ plan of all-1.0, no prior posterior, and the
        master PRNG key derived from `cfg.seed`.
    """
    from models.fsa_high_res._plant import init_plant_state
    return BenchState(
        plant_state=init_plant_state(),
        daily_phi_plan=np.full(cfg.T_days, 1.0, dtype=np.float64),
        last_replan_stride=0,
        prev_particles=None,
        fixed_init_state=env.cold_start_init,
        key=jax.random.PRNGKey(cfg.seed),
    )


# ── Pure stride helpers ───────────────────────────────────────────────


def _phi_for_stride(state: BenchState, env: BenchEnv, cfg: BenchConfig,
                    s: int) -> Tuple[float, np.ndarray]:
    """Computes the Φ to apply during stride `s`.

    Resolves `day_in_plan` (the day-offset within the current planning
    horizon, reset by the most recent replan) and reads the daily-mean
    Φ from `state.daily_phi_plan`. v1.5 has no Φ-burst, so the per-bin
    Φ is constant across the stride.

    Args:
        state: Current bench state.
        env: Bench-time invariants.
        cfg: User config.
        s: Zero-based stride index.

    Returns:
        Tuple `(Phi_today, Phi_subdaily)`:
            - `Phi_today`: Scalar daily-mean Φ for this stride.
            - `Phi_subdaily`: Length-`STRIDE_BINS` per-bin Φ vector
              (constant value).
    """
    day_in_plan = (s - state.last_replan_stride) * env.STRIDE_BINS // env.BINS_PER_DAY
    day_idx = min(day_in_plan, cfg.T_days - 1)
    Phi_today = float(state.daily_phi_plan[day_idx])
    Phi_subdaily = np.full(env.STRIDE_BINS, Phi_today, dtype=np.float64)
    return Phi_today, Phi_subdaily


def _advance_plant_one_stride(env: BenchEnv, cfg: BenchConfig,
                              state: BenchState, acc: BenchAccumulators,
                              s: int
                              ) -> Tuple[BenchState, BenchAccumulators, dict]:
    """Applies the current Φ to the plant for one stride.

    Splits a sub-key off `state.key`, calls `plant_rollout` for the
    stride's `STRIDE_BINS` bins, and appends the resulting trajectory +
    obs chunks to the accumulators. Pure: no mutation; returns updated
    `(state, acc)` records.

    Args:
        env: Bench-time invariants.
        cfg: User config.
        state: Current threaded state.
        acc: Current accumulators.
        s: Zero-based stride index.

    Returns:
        Tuple `(new_state, new_acc, telemetry)`:
            - `new_state` carries the post-stride plant state and
              advanced PRNG key.
            - `new_acc` extends every per-stride tuple by one entry.
            - `telemetry` is a dict with `daily_phi` for the stride,
              consumed by the telemetry-merge step.
    """
    from models.fsa_high_res._plant import plant_rollout
    Phi_today, Phi_subdaily = _phi_for_stride(state, env, cfg, s)
    key, plant_key = jax.random.split(state.key)
    rollout = plant_rollout(
        state.plant_state, Phi_subdaily,
        env.DEFAULT_PARAMS, env.DT_BIN_DAYS, plant_key,
    )
    new_state = state._replace(plant_state=rollout.final_state, key=key)
    # Compute the running A_sum increment as a single device→host
    # scalar instead of re-meaning the whole accumulated trajectory
    # in the telemetry path (which was O(n²) and forced a sync per
    # stride). One scalar transfer per stride, constant cost.
    A_sum_inc = float(jnp.sum(rollout.trajectory[:, 2]))
    n_bins_inc = int(rollout.trajectory.shape[0])
    new_acc = acc._replace(
        traj_chunks=acc.traj_chunks + (np.asarray(rollout.trajectory),),
        obs_B_chunks=acc.obs_B_chunks + (np.asarray(rollout.obs_B),),
        obs_F_chunks=acc.obs_F_chunks + (np.asarray(rollout.obs_F),),
        obs_A_chunks=acc.obs_A_chunks + (np.asarray(rollout.obs_A),),
        Phi_chunks=acc.Phi_chunks + (Phi_subdaily,),
        daily_phi_per_stride=acc.daily_phi_per_stride + (Phi_today,),
        A_sum=acc.A_sum + A_sum_inc,
        n_bins_so_far=acc.n_bins_so_far + n_bins_inc,
    )
    return new_state, new_acc, {'daily_phi': Phi_today}


def _accumulated_obs_for_window(acc: BenchAccumulators,
                                window_bins: int) -> Optional[dict]:
    """Returns the latest `window_bins` of obs as a dict, or None.

    Pure: just slices the accumulator tuples; no mutation.

    Args:
        acc: Current accumulators (after the plant step has appended).
        window_bins: Length of the requested window.

    Returns:
        None if fewer than `window_bins` of obs have been accumulated;
        otherwise a dict with keys `obs_B`, `obs_F`, `obs_A`, `Phi`
        each carrying a length-`window_bins` `np.ndarray`.
    """
    obs_B = np.concatenate(acc.obs_B_chunks) if acc.obs_B_chunks else np.zeros(0)
    if obs_B.shape[0] < window_bins:
        return None
    obs_F = np.concatenate(acc.obs_F_chunks)
    obs_A = np.concatenate(acc.obs_A_chunks)
    Phi = np.concatenate(acc.Phi_chunks)
    return {
        'obs_B': obs_B[-window_bins:].astype(np.float64),
        'obs_F': obs_F[-window_bins:].astype(np.float64),
        'obs_A': obs_A[-window_bins:].astype(np.float64),
        'Phi':   Phi[-window_bins:].astype(np.float64),
    }


def _filter_one_stride(env: BenchEnv, cfg: BenchConfig,
                       state: BenchState, acc: BenchAccumulators,
                       s: int
                       ) -> Tuple[BenchState, BenchAccumulators, dict]:
    """Runs the outer SMC² filter on the latest WINDOW_BINS of obs.

    No-op on warmup strides (when fewer than `WINDOW_BINS` of obs have
    been accumulated) — returns inputs unchanged with empty telemetry.

    First-fire path uses `run_smc_window_native` (prior-draw); subsequent
    fires use `run_smc_window_bridge_native` (Schrödinger-Föllmer
    bridge from the previous posterior).

    Args:
        env: Bench-time invariants.
        cfg: User config.
        state: Current threaded state.
        acc: Current accumulators (after the plant step has appended).
        s: Zero-based stride index.

    Returns:
        Tuple `(new_state, new_acc, telemetry)`:
            - `new_state` carries the new `prev_particles`,
              `fixed_init_state`, and an advanced PRNG key.
            - `new_acc` extends `posterior_chunks` and `posterior_mask`
              by one entry (None+False on warmup).
            - `telemetry` is a dict with keys `n_temp_filter`,
              `filter_elapsed_s` (both 0 if no fire).
    """
    window = _accumulated_obs_for_window(acc, env.WINDOW_BINS)
    if window is None:
        # Warmup stride — append the empty placeholder and pass through.
        new_acc = acc._replace(
            posterior_chunks=acc.posterior_chunks + (None,),
            posterior_mask=acc.posterior_mask + (False,),
        )
        return state, new_acc, {'n_temp_filter': 0, 'filter_elapsed_s': 0.0}

    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained

    grid_obs = env.em.align_obs_fn(window, env.WINDOW_BINS, env.DT_BIN_DAYS)

    key, smc_key = jax.random.split(state.key)
    ld = jax.tree_util.Partial(
        env.log_density_factory,
        grid_obs=grid_obs,
        fixed_init_state=state.fixed_init_state,
        w_start=jnp.asarray(0, dtype=jnp.int32),
        key0=smc_key,
    )

    if state.prev_particles is None:
        particles, elapsed_f, n_temp = run_smc_window_native(
            ld, env.em, env.T_arr, cfg=env.smc_cfg,
            initial_particles=None, seed=cfg.seed + s * 1000,
        )
    else:
        particles, elapsed_f, n_temp = run_smc_window_bridge_native(
            new_ld=ld, prev_particles=state.prev_particles,
            model=env.em, T_arr=env.T_arr, cfg=env.smc_cfg,
            seed=cfg.seed + s * 1000,
        )

    # Snapshot the constrained posterior cloud for the param-traces
    # plot. Vectorised: one batched GPU call instead of N sequential
    # per-particle launches.
    samp_constrained = np.asarray(jax.vmap(
        lambda p: unconstrained_to_constrained(p, env.T_arr)
    )(jnp.asarray(particles)))
    # Pad to (n_smc, n_params) — early SMC tempering may return fewer
    # rows than n_smc (rare but possible); the plot-fill code expects
    # uniform shape.
    posterior_cloud = np.zeros((cfg.n_smc, env.n_params), dtype=np.float64)
    posterior_cloud[:samp_constrained.shape[0], :] = samp_constrained

    # Smoothed end-of-window state ⇒ next replan's "fair" init.
    n_extract = min(10, particles.shape[0])
    us_extract = jnp.asarray(particles[:n_extract])
    target_step_arr = jnp.asarray(env.STRIDE_BINS, dtype=jnp.int32)
    extract_partial = jax.tree_util.Partial(
        env.log_density_factory.extract_state_at_step,
        grid_obs=grid_obs,
        fixed_init_state=state.fixed_init_state,
        w_start=jnp.asarray(0, dtype=jnp.int32),
        key0=smc_key,
        target_step=target_step_arr,
    )
    states = jax.vmap(extract_partial)(us_extract)
    new_fixed_init = jnp.asarray(jnp.mean(states, axis=0))

    new_state = state._replace(
        prev_particles=particles,
        fixed_init_state=new_fixed_init,
        key=key,
    )
    new_acc = acc._replace(
        posterior_chunks=acc.posterior_chunks + (posterior_cloud,),
        posterior_mask=acc.posterior_mask + (True,),
    )
    return new_state, new_acc, {
        'n_temp_filter': int(n_temp),
        'filter_elapsed_s': float(elapsed_f),
    }


def _replan_one_stride(env: BenchEnv, cfg: BenchConfig,
                       state: BenchState, acc: BenchAccumulators,
                       s: int
                       ) -> Tuple[BenchState, BenchAccumulators, dict]:
    """Runs the tempered-SMC controller, on K-cadence and post-warmup.

    Skips when:
      * `cfg.smoke` is True (smoke runs a single stride only),
      * `s == 0` (no replan on the very first stride),
      * `s % replan_K != 0` (off-cadence stride),
      * `state.prev_particles is None` (filter has not fired yet).

    On fire, builds a `ParamsV15` from the posterior-mean filter
    estimates, plans `T_total = cfg.T_days` ahead from the smoothed
    end-of-window state, decodes per-day Φ, and overwrites
    `daily_phi_plan` with the new full-horizon plan. Mirrors
    `version_2_Python_JAX/tools/bench_smc_full_mpc_fsa.py:386-388`.

    Args:
        env: Bench-time invariants.
        cfg: User config.
        state: Current threaded state.
        acc: Current accumulators.
        s: Zero-based stride index.

    Returns:
        Tuple `(new_state, new_acc, telemetry)`:
            - `new_state` carries the overwritten `daily_phi_plan`,
              reset `last_replan_stride = s + 1`, and advanced PRNG key.
              Unchanged when no replan fires.
            - `new_acc` appends one entry to `replan_records` on fire,
              else unchanged.
            - `telemetry` is a dict with `n_temp_ctrl` and
              `replan_plan_per_day` (empty tuple if no fire).
    """
    skip = (cfg.smoke
            or s == 0
            or s % cfg.replan_K != 0
            or state.prev_particles is None)
    if skip:
        return state, acc, {'n_temp_ctrl': 0, 'replan_plan_per_day': ()}

    from models.fsa_high_res.simulation import (
        EstimatedDynParams, InitState, fill_pinned,
    )
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained

    # Vectorised constrained-space mapping over the whole posterior
    # cloud — one batched GPU call instead of N sequential per-particle
    # launches.
    prev_particles_jnp = jnp.asarray(state.prev_particles)
    samp = np.asarray(jax.vmap(
        lambda p: unconstrained_to_constrained(p, env.T_arr)
    )(prev_particles_jnp))

    posterior_mean = {n: float(samp[:, i].mean())
                      for i, n in enumerate(env.param_names)}
    estimated = EstimatedDynParams(**{
        k: posterior_mean[k] for k in EstimatedDynParams._fields
    })
    params_v15 = fill_pinned(estimated)

    xhat = np.asarray(state.fixed_init_state, dtype=np.float64)
    init_state_named = InitState(B=float(xhat[0]),
                                 F=float(xhat[1]),
                                 A=float(xhat[2]))
    # Use the compile-once factory: binds runtime params + init via
    # Partial; the underlying JIT'd cost_kernel is reused (NO
    # XLA recompile) across replans. Closes the dominant GPU-idle gap.
    ctrl_spec = env.ctrl_spec_factory(
        params_v15=params_v15, init_state=init_state_named,
        sigma_prior=cfg.ctrl_sigma_prior,
    )

    # Advance the master key for stylistic parity with Julia's hash-keyed
    # subkey; the controller itself takes an integer seed, not a key.
    key, _ = jax.random.split(state.key)
    ctrl_seed_int = int(cfg.seed) + 100_000 + s
    res = run_tempered_smc_loop_native(
        spec=ctrl_spec, cfg=env.ctrl_cfg, seed=ctrl_seed_int,
        print_progress=False,
    )
    mean_theta = jnp.asarray(res['mean_theta'])
    Phi_plan_arr = np.asarray(ctrl_spec.schedule_from_theta(mean_theta),
                              dtype=np.float64)
    n_plan_days = Phi_plan_arr.shape[0] // env.BINS_PER_DAY
    new_daily = Phi_plan_arr[:n_plan_days * env.BINS_PER_DAY].reshape(
        n_plan_days, env.BINS_PER_DAY).mean(axis=1)
    new_daily_phi_plan = new_daily[:cfg.T_days].astype(np.float64)
    if new_daily_phi_plan.shape[0] < cfg.T_days:
        last_val = (float(new_daily_phi_plan[-1])
                    if new_daily_phi_plan.size > 0 else 1.0)
        pad = np.full(cfg.T_days - new_daily_phi_plan.shape[0], last_val)
        new_daily_phi_plan = np.concatenate([new_daily_phi_plan, pad])

    n_temp_ctrl = int(res.get('n_temp_levels', res.get('n_temp', 0)))
    plan_per_day = tuple(float(x) for x in new_daily.tolist())

    new_state = state._replace(
        daily_phi_plan=new_daily_phi_plan,
        last_replan_stride=s + 1,
        key=key,
    )
    new_acc = acc._replace(
        replan_records=acc.replan_records + ({
            'stride': s,
            'plan_per_day': list(plan_per_day),
            'n_temp': n_temp_ctrl,
        },),
    )
    return new_state, new_acc, {
        'n_temp_ctrl': n_temp_ctrl,
        'replan_plan_per_day': plan_per_day,
    }


def _stride_telemetry(s: int, t_wall_s: float,
                      state: BenchState,
                      acc: BenchAccumulators,
                      plant_tel: dict, filter_tel: dict, replan_tel: dict
                      ) -> StrideTelemetry:
    """Folds per-step telemetry dicts into one `StrideTelemetry` record.

    Args:
        s: Zero-based stride index.
        t_wall_s: Total wall-clock for this stride.
        state: Current threaded state. Used to read end-of-stride
            latents directly from `state.plant_state.bfa` (one
            scalar device→host sync per dim) instead of concatenating
            the full trajectory history.
        acc: Accumulators *after* this stride's plant step has appended.
            `acc.A_sum` and `acc.n_bins_so_far` provide an O(1)
            running mean of A (replaces the previous O(n²) full-
            history `np.concatenate(...).mean()`).
        plant_tel: Plant-step telemetry (`daily_phi`).
        filter_tel: Filter-step telemetry (`n_temp_filter`,
            `filter_elapsed_s`).
        replan_tel: Replan-step telemetry (`n_temp_ctrl`,
            `replan_plan_per_day`).

    Returns:
        A `StrideTelemetry` row ready to be tuple-appended to
        `acc.per_stride_log`.
    """
    if acc.n_bins_so_far > 0:
        A_mean_so_far = acc.A_sum / acc.n_bins_so_far
        bfa = state.plant_state.bfa
        B_end = float(bfa[0])
        F_end = float(bfa[1])
        A_end = float(bfa[2])
    else:
        A_mean_so_far = B_end = F_end = A_end = float('nan')

    return StrideTelemetry(
        stride=s, t_wall_s=t_wall_s,
        n_temp_filter=int(filter_tel.get('n_temp_filter', 0)),
        n_temp_ctrl=int(replan_tel.get('n_temp_ctrl', 0)),
        daily_phi=float(plant_tel.get('daily_phi', 1.0)),
        A_mean_so_far=A_mean_so_far,
        B_end=B_end, F_end=F_end, A_end=A_end,
        filter_elapsed_s=float(filter_tel.get('filter_elapsed_s', 0.0)),
        replan_plan_per_day=tuple(replan_tel.get('replan_plan_per_day', ())),
    )


def _stride_body(env: BenchEnv, cfg: BenchConfig,
                 sa: Tuple[BenchState, BenchAccumulators], s: int,
                 t0: float
                 ) -> Tuple[BenchState, BenchAccumulators]:
    """The pure stride body: plant → filter → replan → telemetry.

    Args:
        env: Bench-time invariants.
        cfg: User config.
        sa: `(BenchState, BenchAccumulators)` pair coming into this stride.
        s: Zero-based stride index.
        t0: Wall-clock at the start of the stride (seconds since epoch).
            Passed in so the telemetry can record `t_wall_s` purely.

    Returns:
        Tuple `(new_state, new_accumulators)` with this stride's
        contributions folded in.
    """
    state, acc = sa
    state, acc, plant_tel  = _advance_plant_one_stride(env, cfg, state, acc, s)
    state, acc, filter_tel = _filter_one_stride(env, cfg, state, acc, s)
    state, acc, replan_tel = _replan_one_stride(env, cfg, state, acc, s)
    elapsed = time.time() - t0
    tel = _stride_telemetry(s, elapsed, state, acc, plant_tel, filter_tel, replan_tel)
    return state, acc._replace(per_stride_log=acc.per_stride_log + (tel,))


# ── Impure outer shell: progress-print + reduce ───────────────────────


def _print_stride_progress(env: BenchEnv, tel: StrideTelemetry) -> None:
    """Prints one stride's progress line(s) to stdout.

    This is the only intentional side effect inside the per-stride
    fold; everything else lives in the pure stride body.

    Args:
        env: Bench-time invariants (used for `n_strides`).
        tel: This stride's telemetry record.
    """
    s = tel.stride
    if tel.n_temp_filter == 0:
        print(f"  stride {s+1}/{env.n_strides}: warmup (< window)")
    else:
        print(f"  stride {s+1}/{env.n_strides}: filter "
              f"{tel.n_temp_filter} levels, {tel.filter_elapsed_s:.1f}s")
    if tel.replan_plan_per_day:
        mean_phi = sum(tel.replan_plan_per_day) / len(tel.replan_plan_per_day)
        print(f"    replan: new mean Φ over next "
              f"{len(tel.replan_plan_per_day)} d = {mean_phi:.3f}")
    print(f"  stride done in {tel.t_wall_s:.1f}s")


def _stride_with_progress(env: BenchEnv, cfg: BenchConfig,
                          sa: Tuple[BenchState, BenchAccumulators], s: int
                          ) -> Tuple[BenchState, BenchAccumulators]:
    """Times one stride, calls `_stride_body`, prints progress.

    The pure stride body does the work; this wrapper localises the
    timing + printing impurity. Used as the binary operator in
    `functools.reduce`.

    Args:
        env: Bench-time invariants.
        cfg: User config.
        sa: `(BenchState, BenchAccumulators)` pair coming into this stride.
        s: Zero-based stride index.

    Returns:
        The pair returned by `_stride_body`, after a progress line
        has been printed for this stride.
    """
    t0 = time.time()
    new_state, new_acc = _stride_body(env, cfg, sa, s, t0)
    _print_stride_progress(env, new_acc.per_stride_log[-1])
    return new_state, new_acc


def run_bench(env: BenchEnv, cfg: BenchConfig
              ) -> Tuple[BenchState, BenchAccumulators, float]:
    """Runs the full closed-loop bench by folding strides forward.

    Replaces the imperative `for s in range(n_strides):` loop with a
    `functools.reduce` over the stride indices. Each step applies
    `_stride_with_progress`, which in turn delegates to the pure
    `_stride_body`. No mutable accumulator is ever held in scope.

    Args:
        env: Bench-time invariants.
        cfg: User config.

    Returns:
        Tuple `(final_state, final_accumulators, total_elapsed_s)`:
            - `final_state` is the bench state after the last stride.
            - `final_accumulators` carries every per-stride record.
            - `total_elapsed_s` is the wall-clock for the full fold.
    """
    state0 = _initial_state(env, cfg)
    acc0 = _empty_accumulators()
    t_start = time.time()
    final_state, final_acc = functools.reduce(
        lambda sa, s: _stride_with_progress(env, cfg, sa, s),
        range(env.n_strides),
        (state0, acc0),
    )
    total_elapsed = time.time() - t_start
    return final_state, final_acc, total_elapsed


# ── Baseline reference (constant Φ = 1.0) ─────────────────────────────


def _build_baseline_trajectory(env: BenchEnv, n_bins: int,
                               key) -> np.ndarray:
    """Plant rollout under constant Φ = 1.0 for `n_bins` bins.

    Used as the reference trace in the state-trajectories plot
    ("MPC vs. canonical Banister Φ ≡ 1.0").

    Args:
        env: Bench-time invariants.
        n_bins: Total number of bins to roll out (== bench's n_mpc_bins).
        key: A JAX PRNGKey to seed the baseline RNG.

    Returns:
        The baseline trajectory as an `(n_bins, 3)` numpy array. Empty
        array of shape `(0, 3)` if `n_bins == 0`.
    """
    if n_bins <= 0:
        return np.zeros((0, 3))
    from models.fsa_high_res._plant import init_plant_state, plant_rollout
    Phi_base = np.full(n_bins, 1.0, dtype=np.float64)
    rollout = plant_rollout(
        init_plant_state(), Phi_base,
        env.DEFAULT_PARAMS, env.DT_BIN_DAYS, key,
    )
    return np.asarray(rollout.trajectory)


# ── Plotting (pure inputs in, file written, returns None) ─────────────


def _plot_state_traces(*, traj_mpc: np.ndarray,
                       traj_baseline: np.ndarray,
                       daily_phi_per_stride: np.ndarray,
                       BINS_PER_DAY: int,
                       STRIDE_BINS: int,
                       dt_days: float,
                       F_max: float,
                       T_days: int,
                       out_path: Path) -> None:
    """Writes the 4-panel state-trajectory plot to `out_path`.

    Direct port of `version_2_Julia/tools/plot_state_traces.jl`: same
    layout (top-left B, top-right F, bottom-left A, bottom-right
    applied Φ), same colours, same labels. Uses the `Agg` matplotlib
    backend so no X display is needed.

    Args:
        traj_mpc: MPC trajectory of shape `(n_bins, 3)`.
        traj_baseline: Baseline (Φ ≡ 1.0) trajectory of shape `(n_bins, 3)`.
        daily_phi_per_stride: Length-`n_strides` daily Φ values applied.
        BINS_PER_DAY: Time-grid resolution.
        STRIDE_BINS: Per-stride plant horizon length in bins.
        dt_days: Bin width in days.
        F_max: F-barrier value (drawn as a horizontal red dashed line).
        T_days: Total horizon length (used in the suptitle).
        out_path: Destination PNG path; parent dir must already exist.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_bins = traj_mpc.shape[0]
    t_days = np.arange(n_bins) * dt_days
    n_strides = len(daily_phi_per_stride)
    stride_start_days = np.arange(n_strides) * STRIDE_BINS * dt_days

    mean_A_mpc  = float(np.mean(traj_mpc[:, 2]))
    mean_A_base = float(np.mean(traj_baseline[:, 2]))

    blue, darkred, green, orange = '#1f77b4', '#8b0000', '#2ca02c', '#ff7f0e'
    grey, redline = '#7f7f7f', '#d62728'

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=120)
    fig.suptitle(
        f"FSA v1.5 closed-loop MPC, T={t_days[-1]:.1f}d. "
        f"mean A {mean_A_mpc:.3f} vs baseline {mean_A_base:.3f}",
        fontsize=10,
    )

    ax = axes[0, 0]
    ax.plot(t_days, traj_baseline[:, 0], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='B (baseline)')
    ax.plot(t_days, traj_mpc[:, 0], color=blue, lw=1.6, label='B (MPC)')
    ax.set_title('B trajectory', fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('B', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_days, traj_baseline[:, 1], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='F (baseline)')
    ax.plot(t_days, traj_mpc[:, 1], color=darkred, lw=1.6, label='F (MPC)')
    ax.axhline(F_max, color=redline, ls='--', lw=1.0, label='F_max')
    ax.set_title('F trajectory', fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('F', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(t_days, traj_baseline[:, 2], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='A (baseline)')
    ax.plot(t_days, traj_mpc[:, 2], color=green, lw=1.6, label='A (MPC)')
    ax.set_title(
        f'A trajectory  (mean MPC: {mean_A_mpc:.3f}, '
        f'baseline: {mean_A_base:.3f})', fontsize=10,
    )
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('A', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(stride_start_days, daily_phi_per_stride, color=orange, lw=2.0,
            marker='o', markersize=3, label='applied daily Φ')
    ax.axhline(1.0, color=grey, ls='--', lw=1.0, label='baseline Φ=1.0')
    ax.set_title(f'MPC-applied Φ schedule across {n_strides} strides',
                 fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('daily Φ', fontsize=8)
    ax.set_ylim(0.0, max(1.5, float(np.max(daily_phi_per_stride)) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_param_traces(*, posterior_particles: np.ndarray,
                       posterior_mask: np.ndarray,
                       param_names: Tuple[str, ...],
                       truth: dict,
                       n_strides: int,
                       stride_bins: int,
                       window_bins: int,
                       bins_per_day: int,
                       T_days: int,
                       step_minutes: int,
                       mean_A_mpc: float,
                       mean_A_base: float,
                       out_path: Path) -> None:
    """Writes the per-parameter posterior-traces plot to `out_path`.

    Direct port of `version_2_Python_JAX/tools/plot_param_traces.py`:
    one panel per parameter showing the 5-95% quantile band + median
    over rolling windows, with the truth value as a horizontal red
    dashed line.

    Args:
        posterior_particles: Array of shape `(n_strides, n_smc, n_params)`
            holding the constrained posterior cloud per stride. Slots
            for warmup strides are zero-filled and skipped via the mask.
        posterior_mask: Bool array of length `n_strides`; True iff the
            filter actually fired on that stride.
        param_names: Parameter names in column order.
        truth: Dict of `{name: truth_value}`; missing names skip the
            horizontal truth line.
        n_strides: Total number of strides (matches the leading axis).
        stride_bins: Per-stride plant horizon in bins.
        window_bins: Filter window length in bins.
        bins_per_day: Time-grid resolution.
        T_days: Total horizon length (used in the suptitle).
        step_minutes: Time-grid resolution as bin minutes (used in title).
        mean_A_mpc: Mean A across MPC trajectory (used in title).
        mean_A_base: Mean A across baseline trajectory (used in title).
        out_path: Destination PNG path; parent dir must already exist.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    end_t_days = (np.arange(n_strides) * stride_bins + window_bins) / bins_per_day
    end_t_days = end_t_days[posterior_mask]

    valid = posterior_particles[posterior_mask]
    q05 = np.quantile(valid, 0.05, axis=1)
    q50 = np.quantile(valid, 0.50, axis=1)
    q95 = np.quantile(valid, 0.95, axis=1)

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 3.0, n_rows * 2.2),
                             sharex=True, dpi=120)
    axes = axes.flatten() if n_rows > 1 else axes

    def _draw_panel(idx: int, name: str) -> None:
        ax = axes[idx]
        ax.fill_between(end_t_days, q05[:, idx], q95[:, idx],
                        color='C0', alpha=0.3, label='5-95%')
        ax.plot(end_t_days, q50[:, idx], color='C0', lw=1.4, label='median')
        if name in truth:
            ax.axhline(truth[name], color='red', lw=1.0, ls='--',
                       label='truth')
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    # `enumerate` + a sub-call avoids the imperative-for footprint here;
    # the loop body is delegated to `_draw_panel` and the index sequence
    # is consumed by `tuple(map(...))` for its side effect.
    tuple(_draw_panel(i, n) for i, n in enumerate(param_names))
    tuple(axes[j].axis('off') for j in range(n_params, len(axes)))
    tuple(ax.set_xlabel('end of window (days)', fontsize=8)
          for ax in axes[-n_cols:])

    fig.suptitle(
        f"FSA-v1.5 posterior parameter traces — T={T_days}d, "
        f"h={step_minutes}min, n_strides={n_strides}, "
        f"mean A {mean_A_mpc:.3f} vs baseline {mean_A_base:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


# ── Artefact saving (impure shell) ────────────────────────────────────


def _resolve_out_dir(cfg: BenchConfig) -> Path:
    """Resolves `cfg.out_dir` (or auto-builds one under outputs/fsa_v15/).

    Args:
        cfg: User config.

    Returns:
        A `Path` to the (created) output directory.
    """
    if cfg.out_dir:
        out_dir = Path(cfg.out_dir)
    else:
        suffix = '_smoke' if cfg.smoke else ''
        out_dir = (Path(__file__).resolve().parent.parent / 'outputs'
                   / 'fsa_v15'
                   / f'run_T{cfg.T_days}{suffix}_seed{cfg.seed}')
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _stack_posterior(acc: BenchAccumulators, env: BenchEnv,
                     cfg: BenchConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Stacks per-stride posterior tuples into the dense (n_strides, …) array.

    Args:
        acc: Final accumulators.
        env: Bench-time invariants.
        cfg: User config.

    Returns:
        Tuple `(posterior_particles, posterior_mask)` where:
            - `posterior_particles` has shape `(n_strides, n_smc, n_params)`
              with zero-fill in warmup slots.
            - `posterior_mask` is bool of length `n_strides`.
    """
    posterior_particles = np.zeros(
        (env.n_strides, cfg.n_smc, env.n_params), dtype=np.float64,
    )

    # `posterior_particles` is freshly allocated and only escapes this
    # function via the return value, so slot-writes are part of the
    # pure construction step. The `tuple(map(...))` consumes a
    # generator for its side effect (the same pattern used in
    # `_plot_param_traces`).
    def _slot(idx_cloud: Tuple[int, Optional[np.ndarray]]) -> None:
        i, cloud = idx_cloud
        if cloud is not None:
            posterior_particles[i] = cloud

    tuple(map(_slot, enumerate(acc.posterior_chunks)))
    posterior_mask = np.array(acc.posterior_mask, dtype=bool)
    return posterior_particles, posterior_mask


def _save_artifacts(env: BenchEnv, cfg: BenchConfig,
                    acc: BenchAccumulators,
                    total_elapsed_s: float,
                    base_key) -> Path:
    """Writes the bench's output artefacts to disk.

    Side effects: creates the output directory, writes `trajectory.npz`,
    `manifest.json`, `per_stride.csv`, and the two PNG plots. Returns
    the resolved output directory so the caller can echo it.

    Args:
        env: Bench-time invariants.
        cfg: User config.
        acc: Final accumulators (after the bench has completed).
        total_elapsed_s: Total wall-clock for the bench.
        base_key: PRNGKey to drive the baseline-reference rollout.

    Returns:
        The resolved output directory `Path`.
    """
    out_dir = _resolve_out_dir(cfg)

    full_traj_arr = (np.concatenate(acc.traj_chunks, axis=0)
                     if acc.traj_chunks else np.zeros((0, 3)))
    obs_B_arr = (np.concatenate(acc.obs_B_chunks)
                 if acc.obs_B_chunks else np.zeros(0))
    obs_F_arr = (np.concatenate(acc.obs_F_chunks)
                 if acc.obs_F_chunks else np.zeros(0))
    obs_A_arr = (np.concatenate(acc.obs_A_chunks)
                 if acc.obs_A_chunks else np.zeros(0))
    Phi_arr = (np.concatenate(acc.Phi_chunks)
               if acc.Phi_chunks else np.zeros(0))
    daily_phi_arr = np.array(acc.daily_phi_per_stride, dtype=np.float64)

    print("  building baseline (constant Φ = 1.0) reference …")
    n_mpc_bins = full_traj_arr.shape[0]
    traj_baseline = _build_baseline_trajectory(env, n_mpc_bins, base_key)

    posterior_particles, posterior_mask = _stack_posterior(acc, env, cfg)

    np.savez(out_dir / 'trajectory.npz',
             trajectory_mpc=full_traj_arr,
             trajectory_baseline=traj_baseline,
             applied_phi_per_stride=daily_phi_arr,
             accumulated_obs_B=obs_B_arr,
             accumulated_obs_F=obs_F_arr,
             accumulated_obs_A=obs_A_arr,
             accumulated_Phi=Phi_arr,
             posterior_particles=posterior_particles,
             posterior_window_mask=posterior_mask,
             param_names=np.array(env.param_names, dtype=object),
             BINS_PER_DAY=env.BINS_PER_DAY,
             STRIDE_BINS=env.STRIDE_BINS,
             dt_days=env.DT_BIN_DAYS)

    truth_params = {
        n: float(getattr(env.DEFAULT_PARAMS, n))
        for n in env.param_names
        if n in env.DEFAULT_PARAMS._fields
    }
    manifest = dict(
        bench='bench_smc_full_mpc_fsa_v15',
        T_days=cfg.T_days,
        n_strides=env.n_strides,
        replan_K=cfg.replan_K,
        n_smc=cfg.n_smc, k_pf=cfg.k_pf,
        ctrl_n_smc=cfg.ctrl_n_smc,
        ctrl_n_inner=cfg.ctrl_n_inner,
        ctrl_num_mcmc=cfg.ctrl_num_mcmc,
        ctrl_hmc_step=cfg.ctrl_hmc_step,
        ctrl_hmc_leap=cfg.ctrl_hmc_leap,
        ctrl_target_nats=cfg.ctrl_target_nats,
        ctrl_max_levels=cfg.ctrl_max_levels,
        ctrl_max_lambda_inc=cfg.ctrl_max_lambda_inc,
        ctrl_sigma_prior=cfg.ctrl_sigma_prior,
        ctrl_n_anchors=cfg.ctrl_n_anchors,
        smoke=cfg.smoke,
        seed=cfg.seed,
        device=jax.devices()[0].platform,
        total_elapsed_s=total_elapsed_s,
        replan_history=list(acc.replan_records),
        param_names=list(env.param_names),
        truth_params=truth_params,
        BINS_PER_DAY=int(env.BINS_PER_DAY),
        STRIDE_BINS=int(env.STRIDE_BINS),
        WINDOW_BINS=int(env.WINDOW_BINS),
        step_minutes=60 // (env.BINS_PER_DAY // 24),
    )
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    if acc.per_stride_log:
        csv_cols = ('stride', 't_wall_s', 'n_temp_filter', 'n_temp_ctrl',
                    'daily_phi', 'A_mean_so_far', 'B_end', 'F_end', 'A_end')
        with open(out_dir / 'per_stride.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(csv_cols))
            w.writeheader()
            row_for = lambda tel: {c: getattr(tel, c) for c in csv_cols}
            tuple(map(lambda tel: w.writerow(row_for(tel)),
                      acc.per_stride_log))

    if n_mpc_bins > 0:
        plot_path = out_dir / f'v15_T{cfg.T_days}d_traces.png'
        _plot_state_traces(
            traj_mpc=full_traj_arr,
            traj_baseline=traj_baseline,
            daily_phi_per_stride=daily_phi_arr,
            BINS_PER_DAY=env.BINS_PER_DAY,
            STRIDE_BINS=env.STRIDE_BINS,
            dt_days=env.DT_BIN_DAYS,
            F_max=0.40,
            T_days=cfg.T_days,
            out_path=plot_path,
        )
        print(f"  wrote {plot_path}")

    if posterior_mask.any():
        param_path = out_dir / f'v15_T{cfg.T_days}d_param_traces.png'
        _plot_param_traces(
            posterior_particles=posterior_particles,
            posterior_mask=posterior_mask,
            param_names=env.param_names,
            truth=truth_params,
            n_strides=env.n_strides,
            stride_bins=env.STRIDE_BINS,
            window_bins=env.WINDOW_BINS,
            bins_per_day=env.BINS_PER_DAY,
            T_days=cfg.T_days,
            step_minutes=60 // (env.BINS_PER_DAY // 24),
            mean_A_mpc=float(np.mean(full_traj_arr[:, 2]))
                       if n_mpc_bins > 0 else float('nan'),
            mean_A_base=float(np.mean(traj_baseline[:, 2]))
                        if traj_baseline.size > 0 else float('nan'),
            out_path=param_path,
        )
        print(f"  wrote {param_path}")

    return out_dir


# ── Top-level entry point ─────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    """Top-level entry point.

    Order of operations:
        1. Parse CLI into a frozen `BenchConfig`.
        2. Build `BenchEnv` (one-shot JIT compile).
        3. Print the run header.
        4. Fold strides via `run_bench`.
        5. Save artefacts (trajectory.npz, manifest.json, plots).

    Args:
        argv: Optional CLI arguments; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code (always 0 in the happy path).
    """
    cfg = parse_args(argv)
    env = _build_env(cfg)

    print('=' * 76)
    print(f"  FSA v1.5 closed-loop SMC²-MPC  (T = {cfg.T_days} d, "
          f"step = {60 // (env.BINS_PER_DAY // 24)} min, "
          f"BINS_PER_DAY = {env.BINS_PER_DAY})")
    print(f"  device = {jax.devices()[0].platform.upper()}")
    print(f"  strides = {env.n_strides}, replan every K = {cfg.replan_K}")
    if cfg.smoke:
        print("  *** SMOKE MODE: 1 stride only ***")
    print('=' * 76)

    final_state, final_acc, total_elapsed = run_bench(env, cfg)
    print()
    print(f"  total: {total_elapsed/60:.1f} min "
          f"({total_elapsed:.0f}s) for {env.n_strides} strides")

    # Derive the baseline-rollout key the same way the imperative bench
    # did: `jax.random.split(key)` returns `(new_key, subkey)`; take the
    # subkey (index 1) so artefacts are bit-identical across versions.
    base_key = jax.random.split(final_state.key)[1]
    out_dir = _save_artifacts(env, cfg, final_acc, total_elapsed, base_key)

    print(f"  artefacts written to {out_dir}/")
    print('=' * 76)
    return 0


if __name__ == '__main__':
    sys.exit(main())
