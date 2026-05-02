"""SWAT closed-loop SMC²-MPC bench.

Mirrors ``bench_smc_full_mpc_fsa.py`` but for the 4-state SWAT model
with three exogenous control variates (V_h, V_n, V_c). Reuses the
framework's filter (jax_native_smc) and controller
(run_tempered_smc_loop_native) without changes — those are
model-agnostic.

Loop structure per stride:
  1. plant.advance(STRIDE_BINS, V_h_daily, V_n_daily, V_c_daily) → new obs
  2. Filter window with bridge handoff from previous posterior
  3. Every K strides: plan next-stride controls from new posterior
     Otherwise: reuse previously-planned controls

Acceptance gates (working set; will tune in Phase 3):
  1. mean ∫T under closed-loop MPC ≥ 0.95 × constant baseline
     (V_h=1, V_n=0.3, V_c=0)
  2. T-floor (T_floor_violation_frac ≤ 5%) — analog of FSA's F-max
  3. id-cov ≥ subset / windows-with-obs
  4. Compute ≤ 4 hours

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_smc_full_mpc_swat.py 14
"""
from __future__ import annotations

import os
import sys


# ── Parse --step-minutes BEFORE model imports ─────────────────────────
def _pop_step_minutes_from_argv() -> int:
    if '--step-minutes' in sys.argv:
        i = sys.argv.index('--step-minutes')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--step-minutes requires a value")
        val = int(sys.argv[i + 1])
        if val > 15:
            print(f"WARNING: --step-minutes {val} > 15. SWAT identifiability "
                   f"depends on sub-hour resolution of sleep/wake transitions. "
                   f"Recommend --step-minutes 15 or finer.", file=sys.stderr)
        del sys.argv[i:i + 2]
        return val
    # SWAT default: 15-minute bins. The sleep/wake switching transitions
    # happen on a 30-60-minute timescale; coarser bins (e.g. h=1h that
    # works for FSA-v2) bin-average those transitions and lose the
    # information that identifies the fast-subsystem + obs-channel
    # parameters (kappa, lmbda, alpha_HR, c_tilde, W_thresh, ...). The
    # source spec uses h=5min; h=15min is a 4x compute saving while
    # preserving identifiability per user 2026-04-30.
    return 15

_STEP_MINUTES = _pop_step_minutes_from_argv()
os.environ['FSA_STEP_MINUTES'] = str(_STEP_MINUTES)
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import pathlib as _pathlib
_CACHE_DIR = _pathlib.Path.home() / ".jax_compilation_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(_CACHE_DIR))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '1')

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from models.swat.simulation import BINS_PER_DAY


# ── SWAT-specific replan cadence (per user 2026-04-30) ──────────────
# SWAT replans every 6 hours wall-clock, regardless of step_minutes.
# Rationale: SWAT's fast subsystem (W, Z, a) has τ ~ 2-3 hours, and
# the sleep/wake transitions that drive identifiability happen on
# 30-60-minute timescales. Daily replan (FSA-v2's choice) is too
# coarse — by the time the controller decides anything, the patient
# has been through a full sleep/wake cycle. 6-hour replan lets the
# controller adapt V_h, V_n, V_c within-day.
#
# Computed from wall-clock constants so the cadence holds across
# step-minutes choices (h=5min, 15min, 1h all give 6h replan).

STRIDE_HOURS  = 3.0     # 3-hour stride
WINDOW_HOURS  = 24.0    # 1-day filter window
REPLAN_HOURS  = 6.0     # replan every 6 hours wall-clock

# Convert wall-clock to bin counts using the actual step_minutes.
BINS_PER_HOUR = 60 // _STEP_MINUTES        # 4 at h=15min, 1 at h=1h, 12 at h=5min
STRIDE_BINS   = int(round(STRIDE_HOURS * BINS_PER_HOUR))
WINDOW_BINS   = int(round(WINDOW_HOURS * BINS_PER_HOUR))    # = BINS_PER_DAY
DT            = 1.0 / BINS_PER_DAY


def _replan_K_for_horizon(T_total_days: int) -> int:
    """Replan every K strides where K = REPLAN_HOURS / STRIDE_HOURS.

    With STRIDE_HOURS=3, REPLAN_HOURS=6, K=2 strides → 6-hour replan.
    """
    return int(round(REPLAN_HOURS / STRIDE_HOURS))


def _hmc_step_for_horizon(T_total_days: int) -> float:
    if T_total_days >= 70.0:
        return 0.05
    if T_total_days >= 50.0:
        return 0.12
    return 0.20


def extract_window(obs_data, start: int, end: int):
    """Slice a window of obs from the global accumulators."""
    window = {}
    for ch_name, ch_data in obs_data.items():
        t_idx = np.asarray(ch_data['t_idx'])
        mask = (t_idx >= start) & (t_idx < end)
        new_ch = {'t_idx': (t_idx[mask] - start).astype(np.int32)}
        n_t = len(t_idx)
        for key in ch_data:
            if key == 't_idx':
                continue
            val = ch_data[key]
            arr = np.asarray(val)
            if arr.ndim >= 1 and len(arr) == n_t:
                new_ch[key] = arr[mask]
            else:
                new_ch[key] = val
        window[ch_name] = new_ch
    return window


def _build_swat_control_spec(*, dyn_params: dict, init_state: np.ndarray,
                                plan_horizon_days: int):
    """Build a SWAT ControlSpec from posterior-mean dynamics params.

    This is the SWAT analog of FSA-v2's ``_build_phase2_control_spec``.
    Uses the controller-side 4-state form (which is mathematically
    identical to the filter's 7-state estimation form per the cross-
    repo consistency check).
    """
    from models.swat.control import build_control_spec
    from models.swat.simulation import DEFAULT_PARAMS

    # Merge: posterior-mean dynamics overrides DEFAULT_PARAMS;
    # obs-side params keep their truth values.
    truth_params = dict(DEFAULT_PARAMS)
    for k, v in dyn_params.items():
        if k in truth_params:
            truth_params[k] = float(v)

    n_steps = plan_horizon_days * BINS_PER_DAY
    spec = build_control_spec(
        n_steps=n_steps,
        dt=DT,
        n_anchors=8,
        n_inner=64,
        n_substeps=4,
        sigma_prior=1.5,
        seed=42,
    )
    # Override the spec's truth_params + init_state with posterior-derived values
    object.__setattr__(spec, 'truth_params', truth_params)
    object.__setattr__(spec, 'initial_state',
                        jnp.asarray(init_state, dtype=jnp.float64))
    return spec


def _pop_scenario_from_argv() -> str:
    """Parse --scenario {pathological, set_A}; default 'pathological'.

    pathological:  the most challenging cold-start. Patient walks in
                    with V_h=0 (no vitality), V_n=4 (max chronic load),
                    V_c=12 (max phase shift), T_0=0 (testosterone
                    collapsed). Controller must bring all four states
                    back. Worse than psim Set C (which only collapsed T).
    set_A:         healthy baseline. V_h=1, V_n=0.3, V_c=0, T_0=0.5.
                    Used for sanity checks; the controller has little
                    work to do here.
    """
    if '--scenario' in sys.argv:
        i = sys.argv.index('--scenario')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--scenario requires a value")
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if val not in ('pathological', 'set_A'):
            raise SystemExit(f"--scenario must be 'pathological' or 'set_A'")
        return val
    return 'pathological'


# Scenario presets — patient state at trial start + the pre-controller
# (status quo) control levels. The controller takes over after the
# warm-up window and picks new schedules.
SCENARIO_CONFIGS = {
    'pathological': {
        # Z rescaled from 3.5 (in [0,6]) to 0.583 (in [0,1]).
        'init_state':   np.array([0.5, 0.583, 0.5, 0.0], dtype=np.float64),
        'baseline_v_h': 0.0,
        'baseline_v_n': 4.0,
        'baseline_v_c': 12.0,
    },
    'set_A': {
        # Healthy baseline matches dev repo INIT_STATE_A: V_n=0.2 (was 0.3).
        'init_state':   np.array([0.5, 0.583, 0.5, 0.5], dtype=np.float64),
        'baseline_v_h': 1.0,
        'baseline_v_n': 0.2,
        'baseline_v_c': 0.0,
    },
}


def main():
    # CLI: argv[1] = T_total_days (default 14); argv[2] = run_name override
    SCENARIO = _pop_scenario_from_argv()
    T_total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    run_name_override = sys.argv[2] if len(sys.argv) > 2 else None
    REPLAN_EVERY_K = _replan_K_for_horizon(T_total_days)

    sc = SCENARIO_CONFIGS[SCENARIO]
    print("=" * 76)
    print(f"  SWAT closed-loop SMC²-MPC (T = {T_total_days} d, "
          f"scenario = {SCENARIO})")
    if SCENARIO == 'pathological':
        print(f"  Pathological cold-start: T_0=0, V_h=0, V_n=4, V_c=12 (max).")
        print(f"  Controller must drive recovery across the bifurcation.")
    print("=" * 76)

    from models.swat._plant import StepwisePlant
    from models.swat.simulation import DEFAULT_PARAMS
    from models.swat.estimation import SWAT_ESTIMATION, COLD_START_INIT

    from smc2fc.core.config import SMCConfig
    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density_compileonce,
    )
    from smc2fc.control import SMCControlConfig
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native

    truth = dict(DEFAULT_PARAMS)
    em = SWAT_ESTIMATION
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}

    # SWAT identifiable subset — params we expect to recover under the
    # 4-channel obs model. lambda_step / W_thresh dropped after Poisson
    # → log-Gaussian wake-gated steps switch (Phase 3.5); replaced by
    # mu_step0 / beta_W_steps which are linear-in-W so well-identified.
    identifiable_subset = {
        'alpha_HR', 'sigma_HR', 'tau_W', 'tau_Z',
        'beta_Z', 'tau_a', 'mu_step0', 'beta_W_steps',
        'mu_0', 'mu_E',  # the F_max-from-data analogs
    }

    n_windows = (T_total_days * BINS_PER_DAY - WINDOW_BINS) // STRIDE_BINS + 1
    n_strides = n_windows
    print(f"  device:    {jax.devices()[0].platform.upper()}")
    print(f"  T_total:   {T_total_days} days")
    print(f"  step:      {_STEP_MINUTES} min ({BINS_PER_DAY} bins/day, "
          f"{BINS_PER_HOUR} bins/hour)")
    print(f"  window:    {WINDOW_BINS} bins ({WINDOW_HOURS:.0f}h)")
    print(f"  stride:    {STRIDE_BINS} bins ({STRIDE_HOURS:.0f}h)")
    print(f"  windows:   {n_windows}")
    print(f"  replan:    every K={REPLAN_EVERY_K} stride(s) "
          f"= {REPLAN_EVERY_K * STRIDE_HOURS:.0f}h wall-clock")
    print()

    # ── Initialize plant ──
    plant = StepwisePlant(seed_offset=42, state=sc['init_state'].copy())

    daily_v_h_baseline = sc['baseline_v_h']
    daily_v_n_baseline = sc['baseline_v_n']
    daily_v_c_baseline = sc['baseline_v_c']

    daily_v_h_plan = np.full(T_total_days, daily_v_h_baseline, dtype=np.float64)
    daily_v_n_plan = np.full(T_total_days, daily_v_n_baseline, dtype=np.float64)
    daily_v_c_plan = np.full(T_total_days, daily_v_c_baseline, dtype=np.float64)
    last_replan_stride = 0

    print(f"  init state (W,Z,a,T) = {tuple(sc['init_state'])}")
    print(f"  pre-controller / counterfactual baseline: "
          f"V_h={daily_v_h_baseline}, V_n={daily_v_n_baseline}, "
          f"V_c={daily_v_c_baseline}h")

    # Accumulators (SWAT-specific obs structure)
    accumulated_obs = {
        'obs_HR':     {'t_idx': [], 'obs_value': []},
        'obs_sleep':  {'t_idx': [], 'obs_label': []},
        'obs_steps':  {'t_idx': [], 'log_value': [], 'present_mask': []},
        'obs_stress': {'t_idx': [], 'obs_value': []},
        'V_h':        {'t_idx': [], 'value': []},
        'V_n':        {'t_idx': [], 'value': []},
        'V_c':        {'t_idx': [], 'value': []},
    }
    full_traj = []
    daily_v_h_per_stride = []
    daily_v_n_per_stride = []
    daily_v_c_per_stride = []
    replan_history = []

    # ── SMC config ──
    # Doubled back to 512/400 for Phase 3.5 — at T=7 (vs the OOM-
    # triggering T=14) the controller's MPC rollout fits comfortably
    # in 32 GB. If revisiting longer horizons later, halve again.
    smc_cfg = SMCConfig(
        n_smc_particles=512,
        n_pf_particles=400,
        target_ess_frac=0.5,
        max_lambda_inc=0.5,
        num_mcmc_steps=5,
        hmc_step_size=0.2,
        hmc_num_leapfrog=10,
        sf_annealed_n_stages=3,
        sf_annealed_n_mh_steps=5,
        sf_blend=0.7,
        sf_entropy_reg=0.0,
        sf_info_aware=False,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed',
    )

    ctrl_cfg = SMCControlConfig(
        n_smc=512, n_inner=64,
        target_ess_frac=0.5, max_lambda_inc=0.5,
        num_mcmc_steps=5,
        hmc_step_size=_hmc_step_for_horizon(T_total_days),
        hmc_num_leapfrog=10,
    )

    # ── Compile-once log_density factory ──
    log_density_factory = make_gk_dpf_v3_lite_log_density_compileonce(
        model=em,
        n_particles=smc_cfg.n_pf_particles,
        bandwidth_scale=smc_cfg.bandwidth_scale,
        ot_ess_frac=smc_cfg.ot_ess_frac,
        ot_temperature=smc_cfg.ot_temperature,
        ot_max_weight=smc_cfg.ot_max_weight,
        ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
        ot_epsilon=smc_cfg.ot_epsilon,
        dt=DT, t_steps=WINDOW_BINS,
    )
    T_arr = log_density_factory._transforms

    # Filter state across strides
    fixed_init_state = COLD_START_INIT
    prev_particles = None      # (n_smc, theta_dim) — for bridge
    all_results = []

    total_t0 = time.time()

    for s in range(n_strides):
        print(f"  Stride {s+1:>2}/{n_strides}: ", end='', flush=True)

        # ── 1. Plant advances ──
        # Apply the day-of-plan controls (relative to last replan)
        day_in_plan = (s - last_replan_stride) // 2
        v_h_today = float(daily_v_h_plan[day_in_plan])
        v_n_today = float(daily_v_n_plan[day_in_plan])
        v_c_today = float(daily_v_c_plan[day_in_plan])

        # Two-day window for advance (covers the stride + safety margin)
        n_days_advance = 2
        v_h_arr = np.full(n_days_advance, v_h_today)
        v_n_arr = np.full(n_days_advance, v_n_today)
        v_c_arr = np.full(n_days_advance, v_c_today)

        if s == 0:
            # First stride: advance a full WINDOW (so the first window has data)
            obs_stride = plant.advance(WINDOW_BINS, v_h_arr, v_n_arr, v_c_arr)
        else:
            obs_stride = plant.advance(STRIDE_BINS, v_h_arr, v_n_arr, v_c_arr)

        full_traj.append(obs_stride['trajectory'].copy())
        daily_v_h_per_stride.append(v_h_today)
        daily_v_n_per_stride.append(v_n_today)
        daily_v_c_per_stride.append(v_c_today)

        # Append per-stride obs to the global accumulators
        for ch in ('obs_HR', 'obs_stress'):
            accumulated_obs[ch]['t_idx'].append(obs_stride[ch]['t_idx'])
            accumulated_obs[ch]['obs_value'].append(obs_stride[ch]['obs_value'])
        accumulated_obs['obs_sleep']['t_idx'].append(obs_stride['obs_sleep']['t_idx'])
        accumulated_obs['obs_sleep']['obs_label'].append(obs_stride['obs_sleep']['obs_label'])
        accumulated_obs['obs_steps']['t_idx'].append(obs_stride['obs_steps']['t_idx'])
        accumulated_obs['obs_steps']['log_value'].append(obs_stride['obs_steps']['log_value'])
        accumulated_obs['obs_steps']['present_mask'].append(obs_stride['obs_steps']['present_mask'])
        for ch in ('V_h', 'V_n', 'V_c'):
            accumulated_obs[ch]['t_idx'].append(obs_stride[ch]['t_idx'])
            accumulated_obs[ch]['value'].append(obs_stride[ch]['value'])

        # Define this window's bin range (LAST WINDOW_BINS up to current t_bin)
        end_bin = plant.t_bin
        start_bin = max(0, end_bin - WINDOW_BINS)

        if end_bin - start_bin < WINDOW_BINS:
            print(f"warm-up (window not full yet)", flush=True)
            continue

        # ── 2. Filter the window ──
        # Concatenate per-stride accumulators into a single dict per channel
        obs_data_full = {}
        for ch in accumulated_obs:
            obs_data_full[ch] = {'t_idx': np.concatenate(
                [np.asarray(a) for a in accumulated_obs[ch]['t_idx']])}
            for key in accumulated_obs[ch]:
                if key == 't_idx':
                    continue
                val = accumulated_obs[ch][key]
                if isinstance(val, list):
                    obs_data_full[ch][key] = np.concatenate(
                        [np.asarray(a) for a in val])
                else:
                    obs_data_full[ch][key] = val      # bin_hours scalar
        window_obs = extract_window(obs_data_full, start_bin, end_bin)
        # Pre-align to grid (numpy)
        grid_obs = em.align_obs_fn(window_obs, t_steps=WINDOW_BINS, dt=DT)
        # Convert numpy arrays to jnp for the factory
        grid_obs = {k: jnp.asarray(v) if isinstance(v, np.ndarray)
                          else jnp.float64(v) if isinstance(v, float)
                          else v
                     for k, v in grid_obs.items()}

        # w_start is a SCALAR start_bin index (per FSA bench convention)
        w_start_arr = jnp.asarray(start_bin, dtype=jnp.int32)
        key0_stride = jax.random.PRNGKey(42 + s * 1000)

        ld = jax.tree_util.Partial(
            log_density_factory,
            grid_obs=grid_obs,
            fixed_init_state=fixed_init_state,
            w_start=w_start_arr,
            key0=key0_stride,
        )

        if prev_particles is None:
            print(f"filter (cold/native)... ", end='', flush=True)
            particles_unc, elapsed_f, n_temp_f = run_smc_window_native(
                ld, em, T_arr, cfg=smc_cfg,
                initial_particles=None, seed=42 + s * 1000,
            )
        else:
            print(f"filter (bridge/native)... ", end='', flush=True)
            particles_unc, elapsed_f, n_temp_f = run_smc_window_bridge_native(
                new_ld=ld, prev_particles=prev_particles,
                model=em, T_arr=T_arr, cfg=smc_cfg,
                seed=42 + s * 1000,
            )

        # Convert to constrained for posterior summaries
        particles = np.array([
            np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
            for p in np.asarray(particles_unc)
        ])

        # id-coverage check (does posterior 5/95% interval cover truth?)
        n_id_covered = sum(
            1 for name in identifiable_subset
            if np.quantile(particles[:, name_to_idx[name]], 0.05)
                <= truth[name]
                <= np.quantile(particles[:, name_to_idx[name]], 0.95)
        )
        print(f"{n_temp_f}lvl/{elapsed_f:.0f}s, "
              f"id={n_id_covered}/{len(identifiable_subset)} ", end='',
              flush=True)

        # Smoothed end-of-window state for next bridge prior
        n_extract = min(10, particles_unc.shape[0])
        us_extract = jnp.asarray(particles_unc[:n_extract])
        target_step_arr = jnp.asarray(STRIDE_BINS, dtype=jnp.int32)
        extract_partial = jax.tree_util.Partial(
            log_density_factory.extract_state_at_step,
            grid_obs=grid_obs,
            fixed_init_state=fixed_init_state,
            w_start=w_start_arr,
            key0=key0_stride,
            target_step=target_step_arr,
        )
        states = jax.vmap(extract_partial)(us_extract)
        smoothed_state = np.asarray(jnp.mean(states, axis=0))
        fixed_init_state = jnp.array(smoothed_state)
        prev_particles = particles_unc

        # ── 3. Replan if scheduled ──
        if (s + 1) % REPLAN_EVERY_K == 0:
            post_means = {n: float(particles[:, name_to_idx[n]].mean())
                           for n in em.all_names}
            for name, val in em.frozen_params.items():
                if name not in post_means:
                    post_means[name] = float(val)

            dyn_params = {k: post_means[k] for k in (
                'kappa', 'lmbda', 'gamma_3', 'beta_Z',
                'tau_W', 'tau_Z', 'tau_a', 'tau_T',
                'mu_0', 'mu_E', 'eta', 'alpha_T',
                'T_W', 'T_Z', 'T_a', 'T_T',
                'lambda_amp_W', 'lambda_amp_Z', 'V_n_scale',
            ) if k in post_means}

            spec = _build_swat_control_spec(
                dyn_params=dyn_params, init_state=smoothed_state,
                plan_horizon_days=T_total_days,
            )
            res_ctrl = run_tempered_smc_loop_native(
                spec=spec, cfg=ctrl_cfg, seed=42 + s,
                print_progress=False,
            )
            schedule = np.asarray(res_ctrl['mean_schedule'])  # (n_steps, 3)
            n_days_in_plan = T_total_days
            if schedule.shape[0] >= n_days_in_plan * BINS_PER_DAY:
                sched_per_day = (schedule[:n_days_in_plan * BINS_PER_DAY]
                                  .reshape(n_days_in_plan, BINS_PER_DAY, 3)
                                  .mean(axis=1))    # (n_days, 3)
            else:
                sched_per_day = np.tile(schedule.mean(axis=0),
                                         (n_days_in_plan, 1))
            daily_v_h_plan = sched_per_day[:, 0].astype(np.float64)
            daily_v_n_plan = sched_per_day[:, 1].astype(np.float64)
            daily_v_c_plan = sched_per_day[:, 2].astype(np.float64)
            last_replan_stride = s + 1

            print(f"plan: {res_ctrl['n_temp_levels']}lvl, "
                   f"V_h̄={daily_v_h_plan.mean():.2f}  "
                   f"V_n̄={daily_v_n_plan.mean():.2f}  "
                   f"V_c̄={daily_v_c_plan.mean():.2f}", flush=True)
            replan_history.append({
                'stride': int(s),
                'plan_v_h_per_day': daily_v_h_plan.copy(),
                'plan_v_n_per_day': daily_v_n_plan.copy(),
                'plan_v_c_per_day': daily_v_c_plan.copy(),
                'n_temp': int(res_ctrl['n_temp_levels']),
            })
        else:
            print(f"reuse plan day={day_in_plan}, "
                   f"V_h={v_h_today:.2f}/V_n={v_n_today:.2f}/V_c={v_c_today:.2f}",
                   flush=True)

        all_results.append({
            'stride': s, 'start_bin': start_bin, 'end_bin': end_bin,
            'n_temp_filter': n_temp_f, 'elapsed_filter_s': elapsed_f,
            'id_covered': n_id_covered,
            'particles_constrained': particles,
        })

    total_elapsed = time.time() - total_t0

    # ── Counterfactual baseline (status-quo controls held constant) ──
    print()
    print(f"  Running counterfactual baseline ("
          f"constant V_h={daily_v_h_baseline}, "
          f"V_n={daily_v_n_baseline}, V_c={daily_v_c_baseline}) ...")
    traj_full = np.concatenate(full_traj)
    n_total_bins = traj_full.shape[0]
    n_days_baseline = (n_total_bins + BINS_PER_DAY - 1) // BINS_PER_DAY
    plant_b = StepwisePlant(seed_offset=42, state=sc['init_state'].copy())
    plant_b.advance(n_days_baseline * BINS_PER_DAY,
                     np.full(n_days_baseline, daily_v_h_baseline),
                     np.full(n_days_baseline, daily_v_n_baseline),
                     np.full(n_days_baseline, daily_v_c_baseline))
    traj_baseline = np.concatenate(plant_b.history['trajectory'])
    traj_baseline = traj_baseline[:n_total_bins]

    mean_T_mpc = float(traj_full[:, 3].mean())
    mean_T_baseline = float(traj_baseline[:, 3].mean())
    T_floor_violation = float((traj_full[:, 3] < 0.05).mean())
    n_id_pass = sum(1 for r in all_results
                     if r['id_covered'] >= len(identifiable_subset) - 1)

    print(f"  Total compute: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")
    print(f"  mean T (closed-loop MPC):     {mean_T_mpc:.4f}")
    print(f"  mean T (baseline V_h=1,V_n=0.3,V_c=0): {mean_T_baseline:.4f}")
    print(f"  T-floor violation (MPC):      {T_floor_violation:.2%}")

    print(f"\n  Acceptance gates:")
    gates = {
        'mean_T_geq_0.95x_baseline':
            mean_T_mpc >= 0.95 * mean_T_baseline,
        'T_floor_violation_leq_5pct':
            T_floor_violation <= 0.05,
        'n_pass_id_geq_threshold':
            n_id_pass >= max(1, n_strides // 2),
        'compute_leq_4h':
            total_elapsed <= 4 * 3600,
    }
    for gname, passed in gates.items():
        mark = "✓" if passed else "⛔"
        print(f"    {mark}  {gname}: {passed}")
    print(f"  {'✓ all gates pass' if all(gates.values()) else '⛔ gate fail'}")

    # ── Save outputs ──
    h_suffix = f"_h{int(_STEP_MINUTES)}min"
    auto_run_name = (f"swat_T{T_total_days}d_replanK{REPLAN_EVERY_K}"
                      f"{h_suffix}_{SCENARIO}")
    run_name = run_name_override if run_name_override else auto_run_name
    run_dir = f"outputs/swat/swat_runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Manifest
    manifest = {
        'schema_version': 1,
        'T_total_days': T_total_days,
        'step_minutes': _STEP_MINUTES,
        'BINS_PER_DAY': BINS_PER_DAY,
        'STRIDE_BINS': STRIDE_BINS,
        'WINDOW_BINS': WINDOW_BINS,
        'DT': DT,
        'n_strides': n_strides,
        'replan_K': REPLAN_EVERY_K,
        'n_replans': len(replan_history),
        'param_names': list(em.all_names),
        'truth_params': {k: float(v) for k, v in truth.items()},
        'init_state': COLD_START_INIT.tolist(),
        'identifiable_subset': sorted(identifiable_subset),
        'seeds': {'plant': 42, 'filter': 42, 'controller': 42},
        'smc_cfg': smc_cfg.__dict__ if hasattr(smc_cfg, '__dict__')
                    else {},
        'ctrl_cfg': ctrl_cfg.__dict__ if hasattr(ctrl_cfg, '__dict__')
                    else {},
        'summary': {
            'mean_T_mpc': mean_T_mpc,
            'mean_T_baseline': mean_T_baseline,
            'T_floor_violation_frac_mpc': T_floor_violation,
            'total_compute_s': total_elapsed,
            'n_windows_pass_id_cov': n_id_pass,
            'gates': gates,
        },
    }
    with open(f"{run_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Data npz
    # Build posterior_particles with shape (n_strides, n_smc, n_params).
    # Strides where the filter ran (s >= 1 in our pattern) get real
    # posteriors; warm-up stride (s=0) gets zeros + mask=False.
    if all_results:
        n_smc, n_params = all_results[0]['particles_constrained'].shape
        posterior_particles = np.zeros(
            (n_strides, n_smc, n_params), dtype=np.float32)
        posterior_window_mask = np.zeros(n_strides, dtype=bool)
        for r in all_results:
            posterior_particles[r['stride']] = r['particles_constrained']
            posterior_window_mask[r['stride']] = True
    else:
        posterior_particles = np.zeros((n_strides, 1, len(em.all_names)),
                                        dtype=np.float32)
        posterior_window_mask = np.zeros(n_strides, dtype=bool)

    np.savez(f"{run_dir}/data.npz",
              trajectory_mpc=traj_full,
              trajectory_baseline=traj_baseline,
              daily_v_h_per_stride=np.asarray(daily_v_h_per_stride),
              daily_v_n_per_stride=np.asarray(daily_v_n_per_stride),
              daily_v_c_per_stride=np.asarray(daily_v_c_per_stride),
              posterior_particles=posterior_particles,
              posterior_window_mask=posterior_window_mask)

    # Auto-plot: latents + 3-control schedule
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t_days = np.arange(n_total_bins) * DT

    ax = axes[0, 0]
    ax.plot(t_days, traj_full[:, 3], 'C2', label='T (MPC)', lw=1.5)
    ax.plot(t_days, traj_baseline[:, 3], '--', color='grey',
             label='T (baseline)', lw=1.0)
    ax.axhline(0.05, color='red', ls=':', label='T_floor=0.05')
    ax.set_xlabel('time (days)'); ax.set_ylabel('T')
    ax.set_title(f'T trajectory (mean MPC: {mean_T_mpc:.3f}, '
                  f'baseline: {mean_T_baseline:.3f})')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_days, traj_full[:, 0], 'C0', label='W (MPC)', lw=1.0)
    ax.plot(t_days, traj_full[:, 1], 'C1', label='Z (MPC)', lw=1.0)
    ax.plot(t_days, traj_full[:, 2], 'C3', label='a (MPC)', lw=1.0)
    ax.set_xlabel('time (days)')
    ax.set_title('Other latent states')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    stride_t = np.arange(len(daily_v_h_per_stride)) * 0.5  # half-day stride
    ax.plot(stride_t, daily_v_h_per_stride, 'o-', color='C0',
             label='V_h applied')
    ax.plot(stride_t, daily_v_n_per_stride, 'o-', color='C3',
             label='V_n applied')
    ax.set_xlabel('time (days)'); ax.set_ylabel('V_h / V_n')
    ax.set_title('Applied V_h / V_n schedule')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(stride_t, daily_v_c_per_stride, 'o-', color='C2',
             label='V_c applied')
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.axhline(3, color='red', ls=':', label='V_c_max=3h')
    ax.axhline(-3, color='red', ls=':')
    ax.set_xlabel('time (days)'); ax.set_ylabel('V_c (hours)')
    ax.set_title('Applied V_c phase-shift schedule')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'SWAT closed-loop SMC²-MPC, T={T_total_days}d  '
                  f'mean T={mean_T_mpc:.3f} vs baseline {mean_T_baseline:.3f}, '
                  f'T-floor viol={T_floor_violation:.1%}, '
                  f'{int(total_elapsed//60)} min')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{run_dir}/E5_full_mpc_swat_T{T_total_days}d_traces.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    # ── Diagnostic plot 1: latents + circadian overlay ──────────────────
    # C(t) is the OBJECTIVE EXTERNAL light cycle — sin(2π t + φ₀), the
    # ground-truth circadian reference (the sun). It is independent of
    # V_c. The subject's internal shifted drive C_eff = sin(2π(t -
    # V_c/24) + φ₀) is what enters u_W inside the SDE drift, but on a
    # diagnostic plot the user wants to see the objective time-of-day
    # reference, not the V_c-shifted subjective drive.
    PHI_0 = -math.pi / 3.0
    v_c_per_bin = np.concatenate(
        [np.asarray(a) for a in accumulated_obs['V_c']['value']])
    v_c_per_bin = v_c_per_bin[:n_total_bins]
    C_t = np.sin(2.0 * np.pi * t_days + PHI_0)

    # Subject's internal/subjective drive (V_c-shifted) for the second
    # panel comparing wall-clock vs subjective circadian.
    C_t_internal = np.sin(2.0 * np.pi * (t_days - v_c_per_bin / 24.0)
                           + PHI_0)

    fig2, (ax_lat, ax_c2) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]})
    ax_lat.plot(t_days, traj_full[:, 0], color='C0', lw=1.0, label='W')
    ax_lat.plot(t_days, traj_full[:, 1], color='C1', lw=1.0, label='Z')
    ax_lat.plot(t_days, traj_full[:, 2], color='C3', lw=1.0, label='a')
    ax_lat.set_ylabel('W / Z / a  (in [0, 1])')
    ax_lat.set_ylim(-0.05, 1.05)
    ax_lat.grid(alpha=0.3)
    ax_circ = ax_lat.twinx()
    ax_circ.plot(t_days, C_t, color='grey', lw=0.8, alpha=0.5,
                 label='C(t) external')
    ax_circ.set_ylabel('C(t) external (ground-truth)')
    ax_circ.set_ylim(-1.1, 1.1)
    lines1, labels1 = ax_lat.get_legend_handles_labels()
    lines2, labels2 = ax_circ.get_legend_handles_labels()
    ax_lat.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=9, loc='upper right')
    ax_lat.set_title(f'SWAT latents + circadian, T={T_total_days}d, '
                     f'scenario={SCENARIO}')

    # Bottom panel: external vs subjective circadian time
    ax_c2.plot(t_days, C_t, color='black', lw=1.0,
               label='C(t) external (wall-clock)')
    ax_c2.plot(t_days, C_t_internal, color='royalblue', lw=1.0,
               label='C_eff(t) subjective (V_c-shifted)')
    ax_c2.axhline(0, color='grey', lw=0.5, alpha=0.5)
    ax_c2.set_ylim(-1.15, 1.15)
    ax_c2.set_ylabel('C(t)')
    ax_c2.set_xlabel('time (days)')
    ax_c2.grid(alpha=0.3)
    ax_c2.legend(fontsize=9, loc='upper right')

    fig2.tight_layout()
    out_path2 = f"{run_dir}/E5_latents_circadian_T{T_total_days}d.png"
    fig2.savefig(out_path2, dpi=120)
    plt.close(fig2)

    # ── Diagnostic plot 2: obs channels + circadian overlay ─────────────
    # Reconstruct per-bin obs from the accumulators. For wake-gated steps
    # we only scatter the "present" bins (wake bins where steps were
    # actually observed).
    def _flat_obs(ch_dict, val_key):
        if not ch_dict['t_idx']:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.concatenate([np.asarray(a) for a in ch_dict['t_idx']])
        val = np.concatenate([np.asarray(a) for a in ch_dict[val_key]])
        return idx, val

    hr_idx, hr_val = _flat_obs(accumulated_obs['obs_HR'], 'obs_value')
    sl_idx, sl_val = _flat_obs(accumulated_obs['obs_sleep'], 'obs_label')
    st_idx, st_logval = _flat_obs(accumulated_obs['obs_steps'], 'log_value')
    _, st_present = _flat_obs(accumulated_obs['obs_steps'], 'present_mask')
    sr_idx, sr_val = _flat_obs(accumulated_obs['obs_stress'], 'obs_value')

    # Wake-gate the steps scatter
    if st_present.size > 0:
        wake_mask = st_present > 0.5
        st_idx_p = st_idx[wake_mask]
        st_logval_p = st_logval[wake_mask]
    else:
        st_idx_p = st_idx
        st_logval_p = st_logval

    fig3, ax3 = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    channels = [
        ('HR (bpm)', hr_idx, hr_val, 'C0', 'line'),
        ('Sleep label (0=wake,1=light,2=deep)', sl_idx, sl_val, 'C2', 'step'),
        ('log(steps+1) — wake bins only', st_idx_p, st_logval_p, 'C3', 'scatter'),
        ('Stress (0-100)', sr_idx, sr_val, 'C4', 'line'),
    ]
    for axi, (lbl, ix, vv, col, kind) in zip(ax3, channels):
        if ix.size > 0:
            tt = ix * DT
            if kind == 'line':
                axi.plot(tt, vv, color=col, lw=1.0)
            elif kind == 'step':
                axi.step(tt, vv, color=col, lw=1.0, where='post')
            else:  # scatter
                axi.scatter(tt, vv, color=col, s=6, alpha=0.6)
        axi.set_ylabel(lbl, fontsize=9)
        axi.grid(alpha=0.3)
        # Circadian overlay on right axis
        ax_c = axi.twinx()
        ax_c.plot(t_days, C_t, color='grey', lw=0.6, alpha=0.5)
        ax_c.set_ylabel('C(t)', color='grey', fontsize=8)
        ax_c.set_ylim(-1.1, 1.1)
        ax_c.tick_params(axis='y', labelcolor='grey', labelsize=7)
    ax3[-1].set_xlabel('time (days)')
    fig3.suptitle(f'SWAT obs channels + circadian, T={T_total_days}d, '
                  f'scenario={SCENARIO}')
    fig3.tight_layout(rect=[0, 0, 1, 0.97])
    out_path3 = f"{run_dir}/E5_obs_circadian_T{T_total_days}d.png"
    fig3.savefig(out_path3, dpi=120)
    plt.close(fig3)

    # ── Dev-repo-style 3-panel set: latent_states / observations /     ─
    #    entrainment, mirroring SWAT_model_dev/outputs/swat/<scen>/     ─
    # The bench has 4-state + per-bin controls; the adapter in            ─
    # models.swat.sim_plots reuses the dev plotting math against           ─
    # those shapes so the PNGs are visually equivalent.                    ─
    from models.swat.sim_plots import plot_swat_panels
    v_h_pb = np.concatenate(
        [np.asarray(a) for a in accumulated_obs['V_h']['value']])[:n_total_bins]
    v_n_pb = np.concatenate(
        [np.asarray(a) for a in accumulated_obs['V_n']['value']])[:n_total_bins]
    v_c_pb = np.concatenate(
        [np.asarray(a) for a in accumulated_obs['V_c']['value']])[:n_total_bins]
    obs_HR_flat = {
        't_idx':     hr_idx,
        'obs_value': hr_val,
    }
    obs_sleep_flat = {
        't_idx':     sl_idx,
        'obs_label': sl_val,
    }
    obs_steps_flat = {
        't_idx':        st_idx,
        'log_value':    st_logval,
        'present_mask': st_present,
    }
    obs_stress_flat = {
        't_idx':     sr_idx,
        'obs_value': sr_val,
    }
    p_lat, p_obs, p_ent = plot_swat_panels(
        trajectory=traj_full,
        t_grid_days=t_days,
        V_h_per_bin=v_h_pb,
        V_n_per_bin=v_n_pb,
        V_c_per_bin=v_c_pb,
        obs_HR=obs_HR_flat,
        obs_sleep=obs_sleep_flat,
        obs_steps=obs_steps_flat,
        obs_stress=obs_stress_flat,
        params=DEFAULT_PARAMS,
        save_dir=run_dir,
        suffix='',
    )

    print()
    print(f"  Checkpoint: {run_dir}/manifest.json + data.npz")
    print(f"  Plot: {out_path}")
    print(f"  Plot: {out_path2}")
    print(f"  Plot: {out_path3}")
    print(f"  Plot: {p_lat}")
    print(f"  Plot: {p_obs}")
    print(f"  Plot: {p_ent}")
    print("=" * 76)


if __name__ == '__main__':
    main()
