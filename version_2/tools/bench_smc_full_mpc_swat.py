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
        del sys.argv[i:i + 2]
        return val
    return 60  # default 1-hour bins for SWAT (matches FSA-v2 post Stage M)

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


WINDOW_BINS  = BINS_PER_DAY        # 1 day
STRIDE_BINS  = BINS_PER_DAY // 2   # 12 hours
DT           = 1.0 / BINS_PER_DAY


def _replan_K_for_horizon(T_total_days: int) -> int:
    """Daily replan (K=2 strides = 1 day) at all horizons."""
    return 2


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
        T_floor=0.05,
        lam_h=0.01, lam_n=0.01, lam_c=0.001, lam_T_floor=1.0,
        seed=42,
    )
    # Override the spec's truth_params + init_state with posterior-derived values
    object.__setattr__(spec, 'truth_params', truth_params)
    object.__setattr__(spec, 'initial_state',
                        jnp.asarray(init_state, dtype=jnp.float64))
    return spec


def main():
    # CLI: argv[1] = T_total_days (default 14); argv[2] = run_name override
    T_total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    run_name_override = sys.argv[2] if len(sys.argv) > 2 else None
    REPLAN_EVERY_K = _replan_K_for_horizon(T_total_days)

    print("=" * 76)
    print(f"  SWAT closed-loop SMC²-MPC (T = {T_total_days} d)")
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
    # 4-channel obs model. From the SWAT identifiability extension's
    # rank analysis (see safekeeping repo's
    # SWAT_Identifiability_Extension.md). Will tune in Phase 3.
    identifiable_subset = {
        'alpha_HR', 'sigma_HR', 'tau_W', 'tau_Z',
        'beta_Z', 'tau_a', 'lambda_step', 'W_thresh',
        'mu_0', 'mu_E',  # the F_max-from-data analogs
    }

    n_windows = (T_total_days * BINS_PER_DAY - WINDOW_BINS) // STRIDE_BINS + 1
    n_strides = n_windows
    print(f"  device:   {jax.devices()[0].platform.upper()}")
    print(f"  T_total:  {T_total_days} days")
    print(f"  step:     {_STEP_MINUTES} min ({BINS_PER_DAY} bins/day)")
    print(f"  windows:  {n_windows} (1-day, 12h stride)")
    print(f"  replan:   every K={REPLAN_EVERY_K} stride(s) "
          f"(≈ {REPLAN_EVERY_K * 0.5:.1f} day cadence)")
    print()

    # ── Initialize plant ──
    plant = StepwisePlant(seed_offset=42)

    # Default operating-point schedules (V_h=1, V_n=0.3, V_c=0)
    daily_v_h_baseline = 1.0
    daily_v_n_baseline = 0.3
    daily_v_c_baseline = 0.0

    daily_v_h_plan = np.full(T_total_days, daily_v_h_baseline, dtype=np.float64)
    daily_v_n_plan = np.full(T_total_days, daily_v_n_baseline, dtype=np.float64)
    daily_v_c_plan = np.full(T_total_days, daily_v_c_baseline, dtype=np.float64)
    last_replan_stride = 0

    # Accumulators (SWAT-specific obs structure)
    accumulated_obs = {
        'obs_HR':     {'t_idx': [], 'obs_value': []},
        'obs_sleep':  {'t_idx': [], 'obs_label': []},
        'obs_steps':  {'t_idx': [], 'obs_count': [], 'bin_hours': 1.0},
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
    smc_cfg = SMCConfig(
        n_smc_particles=1024,
        n_pf_particles=800,
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
        n_smc=1024, n_inner=128,
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
        accumulated_obs['obs_steps']['obs_count'].append(obs_stride['obs_steps']['obs_count'])
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

    # ── Counterfactual baseline (constant V_h=1, V_n=0.3, V_c=0) ──
    print()
    print(f"  Running counterfactual baseline (constant V_h=1, V_n=0.3, V_c=0) ...")
    traj_full = np.concatenate(full_traj)
    n_total_bins = traj_full.shape[0]
    n_days_baseline = (n_total_bins + BINS_PER_DAY - 1) // BINS_PER_DAY
    plant_b = StepwisePlant(seed_offset=42)
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
    auto_run_name = f"swat_T{T_total_days}d_replanK{REPLAN_EVERY_K}{h_suffix}"
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

    print()
    print(f"  Checkpoint: {run_dir}/manifest.json + data.npz")
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
