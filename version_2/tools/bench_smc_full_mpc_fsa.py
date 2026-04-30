"""Stage E5: full 27-window rolling MPC on FSA-v2.

Loops the E4 closed-loop cycle 27 times, with replanning every K windows
(default K=2 = once per day at the wake boundary). Builds on:

  - E2's StepwisePlant (the simulator-as-plant)
  - E3's rolling-window SMC² filter machinery
  - E4's Phase-2 ControlSpec construction from posterior-mean params
  - Stage D's `run_tempered_smc_loop` controller

Loop structure per window:
  1. plant.advance(STRIDE_BINS, Φ_for_this_stride) → new obs
  2. filter window with bridge handoff from previous posterior
  3. Every K windows: plan next-stride Φ from new posterior
     Otherwise: reuse previously-planned Φ
  4. Repeat

Acceptance gates (E5):
  1. Mean ∫A under closed-loop MPC ≥ 0.95 × const Φ=1 baseline
     (no degradation from posterior-mean compression).
  2. ≥ 24 of 27 windows pass per-window 90% CI gate on identifiable subset.
  3. F-violation fraction over 14 days ≤ 5%.
  4. Compute ≤ 4 hours on RTX 5090.

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa.py
"""

from __future__ import annotations

import os
import sys

# Parse --step-minutes before importing model/bench code so the env var
# governs the FSA grid (BINS_PER_DAY etc.) at module-import time.
def _pop_step_minutes_from_argv() -> int:
    """Extract `--step-minutes N` from sys.argv (default 15). Mutates argv."""
    if '--step-minutes' in sys.argv:
        i = sys.argv.index('--step-minutes')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--step-minutes requires a value")
        val = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return val
    return 15

_STEP_MINUTES = _pop_step_minutes_from_argv()
os.environ['FSA_STEP_MINUTES'] = str(_STEP_MINUTES)
os.environ.setdefault('JAX_ENABLE_X64', 'True')

# Stage L (cheap variant): enable JAX's persistent compilation cache.
# Per-stride re-tracing of log_density / smc_kernel / cost_fn produces
# many identical HLO modules. With the disk cache populated, XLA hits
# it and skips full optimisation/code-gen on cache-hit.
import pathlib as _pathlib
_CACHE_DIR = _pathlib.Path.home() / ".jax_compilation_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(_CACHE_DIR))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '1')

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Re-use E4's helper for building a posterior-mean control spec
from tools.bench_smc_closed_loop_fsa import _build_phase2_control_spec
from models.fsa_high_res.simulation import BINS_PER_DAY


WINDOW_BINS  = BINS_PER_DAY        # 1 day
STRIDE_BINS  = BINS_PER_DAY // 2   # 12 hours
DT           = 1.0 / BINS_PER_DAY


def _replan_K_for_horizon(T_total_days: int) -> int:
    """Daily replan for short horizons; weekly for longer horizons.

    Compute scales linearly with `n_replans × controller_compute(T_total)`,
    so keeping replans bounded prevents the T=84 run blowing past the
    GPU budget. Daily (K=2) at T=14 = 13 replans; weekly (K=14) at
    T=84 = 12 replans (still meaningfully closed-loop).
    """
    if T_total_days <= 14:
        return 2     # daily (every 12h stride doubled)
    return 14        # weekly


def _hmc_step_for_horizon(T_total_days: int) -> float:
    """Controller-side HMC step scaling — port from v1
    bench_smc_control_fsa.py. At long horizons the cost-surface
    curvature scales as cost_std² so a fixed step (0.20-0.30) makes
    leapfrog diverge (acc → 0). The ladder below was empirically
    validated on v1 Stage D at T = 28, 42, 56, 84.
    """
    if T_total_days >= 70.0:
        return 0.05
    if T_total_days >= 50.0:
        return 0.12
    return 0.20


def extract_window(obs_data, start: int, end: int):
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


def main():
    # Parse CLI:
    #   sys.argv[1] : T_total_days (default 14)
    #   sys.argv[2] : run_name override (default auto-derived from settings)
    T_total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    run_name_override = sys.argv[2] if len(sys.argv) > 2 else None
    REPLAN_EVERY_K = _replan_K_for_horizon(T_total_days)

    print("=" * 76)
    print(f"  Stage E5/F/G4 — closed-loop MPC on FSA-v2 (T = {T_total_days} d)")
    print("=" * 76)

    from models.fsa_high_res._plant import StepwisePlant
    from models.fsa_high_res.simulation import DEFAULT_PARAMS
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V2_ESTIMATION, COLD_START_INIT,
    )
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.tempered_smc import run_smc_window, run_smc_window_bridge
    # Stage M: JAX-native tempered SMC (replaces BlackJAX layer at the
    # call site to avoid per-stride smc_kernel recompilation).
    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density,
        make_gk_dpf_v3_lite_log_density_compileonce,
    )
    from smc2fc.control import SMCControlConfig, run_tempered_smc_loop

    truth = dict(DEFAULT_PARAMS)
    em = HIGH_RES_FSA_V2_ESTIMATION
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}
    identifiable_subset = {'HR_base', 'S_base', 'mu_step0',
                            'kappa_B_HR', 'k_F', 'beta_C_HR'}

    n_windows = (T_total_days * BINS_PER_DAY - WINDOW_BINS) // STRIDE_BINS + 1
    n_strides = n_windows
    print(f"  device:   {jax.devices()[0].platform.upper()}")
    print(f"  T_total:  {T_total_days} days")
    print(f"  step:     {_STEP_MINUTES} min ({BINS_PER_DAY} bins/day)")
    print(f"  windows:  {n_windows} (1-day, 12h stride)")
    print(f"  replan:   every K={REPLAN_EVERY_K} stride(s) "
          f"(≈ {REPLAN_EVERY_K * 0.5:.1f} day cadence)")
    print(f"  plan horizon: T_total = {T_total_days} d at each replan")
    print()

    # ── Initialize plant ──
    plant = StepwisePlant(seed_offset=42)
    daily_phi_baseline = 1.0
    # Receding-horizon plan: full per-day Φ for the entire macrocycle.
    # `daily_phi_plan[i]` is day-i's planned Φ. Updated at every replan;
    # at each stride we apply `daily_phi_plan[day_in_plan]` where
    # day_in_plan = (s - last_replan_stride) // 2.
    daily_phi_plan = np.full(T_total_days, daily_phi_baseline, dtype=np.float64)
    last_replan_stride = 0    # track day_in_plan offset since last replan

    # Accumulators
    accumulated_obs = {
        'obs_HR':     {'t_idx': [], 'obs_value': []},
        'obs_sleep':  {'t_idx': [], 'sleep_label': []},
        'obs_stress': {'t_idx': [], 'obs_value': []},
        'obs_steps':  {'t_idx': [], 'obs_value': []},
        'Phi':        {'t_idx': [], 'Phi_value': []},
        'C':          {'t_idx': [], 'C_value':   []},
    }
    full_traj = []
    daily_phi_per_stride = []
    # G4 checkpointing: track per-replan plans + per-replan tempering levels.
    # `replan_history` is appended every time a fresh plan is computed.
    replan_history = []    # list of {'stride': int, 'plan_per_day': np.ndarray, 'n_temp': int}

    # F1+F2: SF Path B-fixed bridge — fixes Gaussian-bridge mid-period
    # drift that caused E3 18/27 and E5 3/26 coverage. Reference:
    # smc2-blackjax-rolling 27/27 PASS at 98.5% on FSA-v2.
    smc_cfg = SMCConfig(
        n_smc_particles=128, n_pf_particles=200,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed',
        sf_use_q0_cov=True,
        sf_blend=0.7,
        sf_annealed_n_stages=3,
        sf_annealed_n_mh_steps=5,
        sf_info_aware=False,                  # G2 tested True on E3, regressed 23/27→17/27;
                                              # the G1 reparametrization is enough on its own.
                                              # See PROGRESS_F_G.md for the empirical rationale.
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.15,
    )
    # Controller-side HMC step scales with the planning horizon
    # (long-horizon cost surfaces have steeper curvature; see v1 Stage D
    # at T=84 where step had to drop from 0.30 to 0.05).
    hmc_step_ctrl = _hmc_step_for_horizon(T_total_days)
    ctrl_cfg = SMCControlConfig(
        n_smc=128, n_inner=32, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=10, hmc_step_size=hmc_step_ctrl, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0, max_temp_steps=30,
    )
    print(f"  controller hmc_step = {hmc_step_ctrl} (scaled for T={T_total_days} d)")
    print()

    prev_particles = None
    fixed_init_state = COLD_START_INIT
    all_results = []

    # Stage L1-deep: build the GK-DPF v3-lite log_density factory ONCE
    # before the per-stride loop. The returned function takes per-stride
    # data (grid_obs, fixed_init_state, w_start, key0) as ARGUMENTS, so
    # JAX's JIT cache hits across strides instead of recompiling each time.
    log_density_factory = make_gk_dpf_v3_lite_log_density_compileonce(
        model=em, n_particles=smc_cfg.n_pf_particles,
        bandwidth_scale=smc_cfg.bandwidth_scale,
        ot_ess_frac=smc_cfg.ot_ess_frac,
        ot_temperature=smc_cfg.ot_temperature,
        ot_max_weight=smc_cfg.ot_max_weight,
        ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
        ot_epsilon=smc_cfg.ot_epsilon,
        dt=DT, t_steps=WINDOW_BINS,
    )
    T_arr = log_density_factory._transforms

    total_t0 = time.time()

    for s in range(n_strides):
        print(f"  Stride {s+1:>2}/{n_strides}: ", end='', flush=True)

        # 1. Apply the day's planned Phi to plant. day_in_plan = days since
        # last replan (each day = 2 strides @ 12h). For K=2 (daily replan)
        # this is always 0; for K=14 (weekly) it iterates 0..6.
        day_in_plan = (s - last_replan_stride) // 2
        phi_idx = min(day_in_plan, len(daily_phi_plan) - 1)
        daily_phi_planned = float(daily_phi_plan[phi_idx])
        obs_stride = plant.advance(STRIDE_BINS, np.array([daily_phi_planned]))
        full_traj.append(obs_stride['trajectory'])
        daily_phi_per_stride.append(daily_phi_planned)

        # Append obs to global accumulators
        for ch in ('obs_HR', 'obs_stress', 'obs_steps'):
            accumulated_obs[ch]['t_idx'].append(obs_stride[ch]['t_idx'])
            accumulated_obs[ch]['obs_value'].append(obs_stride[ch]['obs_value'])
        accumulated_obs['obs_sleep']['t_idx'].append(obs_stride['obs_sleep']['t_idx'])
        accumulated_obs['obs_sleep']['sleep_label'].append(obs_stride['obs_sleep']['sleep_label'])
        accumulated_obs['Phi']['t_idx'].append(obs_stride['Phi']['t_idx'])
        accumulated_obs['Phi']['Phi_value'].append(obs_stride['Phi']['Phi_value'])
        accumulated_obs['C']['t_idx'].append(obs_stride['C']['t_idx'])
        accumulated_obs['C']['C_value'].append(obs_stride['C']['C_value'])

        # Define this window's bin range — the LAST full WINDOW_BINS up to t_bin
        end_bin = plant.t_bin
        start_bin = max(0, end_bin - WINDOW_BINS)

        # 2. Filter the latest window
        # Build per-channel obs slice at global bin indices [start_bin, end_bin)
        obs_data_full = {
            ch: {
                't_idx': np.concatenate(accumulated_obs[ch]['t_idx']),
                **{k: np.concatenate(accumulated_obs[ch][k])
                   for k in accumulated_obs[ch] if k != 't_idx'},
            }
            for ch in accumulated_obs
        }
        if end_bin - start_bin < WINDOW_BINS:
            # First stride: window not yet full. Skip filter+plan; warm-up.
            print(f"warm-up (window not full yet)")
            continue

        window_obs = extract_window(obs_data_full, start_bin, end_bin)

        grid_obs = em.align_obs_fn(window_obs, WINDOW_BINS, DT)

        # Stage L1-deep: bind per-stride dynamic data to the compile-once
        # factory via jax.tree_util.Partial. The Partial is a stable JAX
        # pytree -> the underlying jitted function's cache hits across
        # strides (no per-stride recompile of the SDE scan).
        key0_stride = jax.random.PRNGKey(42 + s * 1000)
        w_start_arr = jnp.asarray(start_bin, dtype=jnp.int32)
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

        particles = np.array([
            np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
            for p in np.asarray(particles_unc)
        ])
        n_id_covered = sum(
            1 for name in identifiable_subset
            if np.quantile(particles[:, name_to_idx[name]], 0.05) <= truth[name]
                <= np.quantile(particles[:, name_to_idx[name]], 0.95)
        )
        print(f"{n_temp_f}lvl/{elapsed_f:.0f}s, id={n_id_covered}/6 ", end='', flush=True)

        # Smoothed end-of-window state (for next window's PF init AND control plan).
        # J1b + L1-deep: vmap-batched extract via the compile-once factory.
        # The Partial wrapper carries the same dynamic data as the bridge
        # log_density above, so the cache hits.
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

        # 3. Replan if scheduled
        if (s + 1) % REPLAN_EVERY_K == 0:
            post_means = {n: float(particles[:, name_to_idx[n]].mean())
                           for n in em.all_names}
            for name, val in em.frozen_params.items():
                if name not in post_means:
                    post_means[name] = float(val)
            dyn_params = {
                k: post_means[k] for k in (
                    'tau_B', 'tau_F', 'kappa_B', 'kappa_F',
                    'epsilon_A', 'lambda_A',
                    'mu_0', 'mu_B', 'mu_F', 'mu_FF', 'eta',
                    'sigma_B', 'sigma_F', 'sigma_A',
                )
            }
            # F2/F3: plan T_total days ahead at every replan
            # (matches v1 open-loop's full-horizon planning).
            spec = _build_phase2_control_spec(
                dyn_params=dyn_params, init_state=smoothed_state,
                plan_horizon_days=T_total_days,
            )
            res_ctrl = run_tempered_smc_loop(
                spec=spec, cfg=ctrl_cfg, seed=42 + s,
                print_progress=False,
            )
            # Controller plans T_total*96 bins. Aggregate to per-day means
            # so the plant's Gamma-burst expander gets a daily Φ for each
            # day in the plan. The next K/2 days will draw from these.
            schedule = np.asarray(res_ctrl['mean_schedule'])
            n_days_in_plan = T_total_days
            if schedule.shape[0] >= n_days_in_plan * BINS_PER_DAY:
                sched_per_day = (schedule[:n_days_in_plan * BINS_PER_DAY]
                                  .reshape(n_days_in_plan, BINS_PER_DAY)
                                  .mean(axis=1))
            else:
                sched_per_day = np.full(n_days_in_plan, schedule.mean())
            daily_phi_plan = sched_per_day.astype(np.float64)
            last_replan_stride = s + 1
            phi_summary = (f"day0={sched_per_day[0]:.3f}, "
                            f"day_mid={sched_per_day[n_days_in_plan//2]:.3f}, "
                            f"day_last={sched_per_day[-1]:.3f}")
            print(f"plan: {res_ctrl['n_temp_levels']}lvl, "
                   f"Φ̄={sched_per_day.mean():.3f}  ({phi_summary})", flush=True)
            # G4 checkpoint: record the full plan from this replan
            replan_history.append({
                'stride': int(s),
                'plan_per_day': sched_per_day.astype(np.float64).copy(),
                'n_temp': int(res_ctrl['n_temp_levels']),
            })
        else:
            print(f"reuse plan day={day_in_plan}, Φ={daily_phi_planned:.3f}", flush=True)

        all_results.append({
            'stride': s,
            'start_bin': start_bin, 'end_bin': end_bin,
            'n_temp_filter': n_temp_f, 'elapsed_filter_s': elapsed_f,
            'id_covered': n_id_covered,
            'daily_phi_applied': daily_phi_per_stride[s] if s < len(daily_phi_per_stride) else daily_phi_planned,
            'particles_constrained': particles,
        })

    total_elapsed = time.time() - total_t0

    # ── Run a counterfactual baseline at constant Φ=1.0 (replay plant) ──
    print()
    print(f"  Running counterfactual baseline (constant Φ=1.0) ...")
    traj_full = np.concatenate(full_traj)
    n_total_bins = traj_full.shape[0]
    n_days_baseline = (n_total_bins + BINS_PER_DAY - 1) // BINS_PER_DAY
    plant_b = StepwisePlant(seed_offset=42)
    plant_b.advance(n_days_baseline * BINS_PER_DAY,
                    np.full(n_days_baseline, 1.0))
    traj_baseline = np.concatenate(plant_b.history['trajectory'])
    # Truncate baseline to match MPC trajectory length for fair comparison
    traj_baseline = traj_baseline[:n_total_bins]

    mean_A_mpc = float(traj_full[:, 2].mean())
    mean_A_baseline = float(traj_baseline[:, 2].mean())
    F_viol_mpc = float((traj_full[:, 1] > 0.40).mean())

    print(f"  Total compute: {total_elapsed/60:.1f} min "
          f"({total_elapsed:.0f}s)")
    print(f"  mean A (closed-loop MPC):  {mean_A_mpc:.4f}")
    print(f"  mean A (constant Φ=1.0):   {mean_A_baseline:.4f}")
    print(f"  F-violation (MPC):         {F_viol_mpc:.2%}")

    n_pass_id = sum(1 for r in all_results if r['id_covered'] >= 5)
    print()
    print(f"  Acceptance gates:")
    gate1 = mean_A_mpc >= 0.95 * mean_A_baseline
    print(f"    {'✓' if gate1 else '✗'}  mean A (MPC) ≥ 0.95 × baseline "
          f"({mean_A_mpc:.4f} vs {0.95*mean_A_baseline:.4f})")
    gate2 = n_pass_id >= 24
    print(f"    {'✓' if gate2 else '✗'}  ≥24 windows have ≥5/6 identifiable "
          f"covered (actual: {n_pass_id}/{len(all_results)})")
    gate3 = F_viol_mpc <= 0.05
    print(f"    {'✓' if gate3 else '✗'}  F-violation ≤ 5%  ({F_viol_mpc:.2%})")
    gate4 = total_elapsed <= 4 * 3600
    print(f"    {'✓' if gate4 else '✗'}  Total time ≤ 4 hours  "
          f"(actual: {total_elapsed/3600:.2f} h)")
    all_pass = gate1 and gate2 and gate3 and gate4
    print(f"  {'✓ all gates pass' if all_pass else '✗ one or more gates fail'}")

    # ── G4 checkpointing: per-run folder with manifest.json + data.npz ──
    bridge_tag = ('infoaware' if smc_cfg.sf_info_aware else 'no_infoaware')
    # h-suffix only when non-default to keep existing 15-min run dirs intact.
    h_suffix = '' if _STEP_MINUTES == 15 else f"_h{_STEP_MINUTES}min"
    auto_run_name = f"T{T_total_days}d_replanK{REPLAN_EVERY_K}{h_suffix}_{bridge_tag}"
    run_name = run_name_override if run_name_override else auto_run_name
    run_dir = f"outputs/fsa_high_res/g4_runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    # Build manifest
    manifest = {
        'schema_version': '1.1',
        'T_total_days': int(T_total_days),
        'replan_K': int(REPLAN_EVERY_K),
        'n_strides': int(n_strides),
        'n_replans': int(len(replan_history)),
        'step_minutes': int(_STEP_MINUTES),
        'BINS_PER_DAY': int(BINS_PER_DAY),
        'WINDOW_BINS': int(WINDOW_BINS),
        'STRIDE_BINS': int(STRIDE_BINS),
        'DT': float(DT),
        'smc_cfg': {
            'n_smc_particles': smc_cfg.n_smc_particles,
            'n_pf_particles': smc_cfg.n_pf_particles,
            'target_ess_frac': smc_cfg.target_ess_frac,
            'max_lambda_inc': smc_cfg.max_lambda_inc,
            'bridge_type': smc_cfg.bridge_type,
            'sf_q1_mode': smc_cfg.sf_q1_mode,
            'sf_use_q0_cov': smc_cfg.sf_use_q0_cov,
            'sf_blend': smc_cfg.sf_blend,
            'sf_annealed_n_stages': smc_cfg.sf_annealed_n_stages,
            'sf_annealed_n_mh_steps': smc_cfg.sf_annealed_n_mh_steps,
            'sf_info_aware': smc_cfg.sf_info_aware,
            'num_mcmc_steps': smc_cfg.num_mcmc_steps,
            'hmc_step_size': smc_cfg.hmc_step_size,
            'hmc_num_leapfrog': smc_cfg.hmc_num_leapfrog,
        },
        'ctrl_cfg': {
            'n_smc': ctrl_cfg.n_smc,
            'n_inner': ctrl_cfg.n_inner,
            'sigma_prior': ctrl_cfg.sigma_prior,
            'num_mcmc_steps': ctrl_cfg.num_mcmc_steps,
            'hmc_step_size': ctrl_cfg.hmc_step_size,
            'hmc_num_leapfrog': ctrl_cfg.hmc_num_leapfrog,
            'beta_max_target_nats': ctrl_cfg.beta_max_target_nats,
            'max_temp_steps': ctrl_cfg.max_temp_steps,
        },
        'truth_params': {k: float(v) for k, v in truth.items()},
        'reparam_constants': {
            'A_typ': 0.10,
            'F_typ': 0.20,
            'note': "Stage G1 reparametrization — see _dynamics.py docstring",
        },
        'init_state': {'B': 0.05, 'F': 0.30, 'A': 0.10},
        'seeds': {'plant': 42, 'filter_base': 42, 'ctrl_base': 42},
        'param_names': list(em.all_names),
        'identifiable_subset': sorted(identifiable_subset),
        'summary': {
            'mean_A_mpc': mean_A_mpc,
            'mean_A_baseline': mean_A_baseline,
            'F_violation_frac_mpc': F_viol_mpc,
            'total_compute_s': float(total_elapsed),
            'n_windows_pass_id_cov_5_of_6': int(n_pass_id),
            'gates': {
                'mean_A_geq_0.95x_baseline': bool(gate1),
                'n_pass_id_geq_24': bool(gate2),
                'F_violation_leq_5pct': bool(gate3),
                'compute_leq_4h': bool(gate4),
            },
        },
    }
    import json
    with open(f"{run_dir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Build data.npz (one bundled archive)
    n_params = len(em.all_names)
    n_smc = smc_cfg.n_smc_particles
    posterior_particles_all = np.zeros(
        (n_strides, n_smc, n_params), dtype=np.float32,
    )
    posterior_window_mask = np.zeros(n_strides, dtype=np.bool_)
    n_temp_per_window = np.zeros(n_strides, dtype=np.int32)
    elapsed_per_window_s = np.zeros(n_strides, dtype=np.float32)
    id_cov_per_window = np.zeros(n_strides, dtype=np.int32)
    for r in all_results:
        s_idx = r['stride']
        posterior_particles_all[s_idx] = r['particles_constrained'].astype(np.float32)
        posterior_window_mask[s_idx] = True
        n_temp_per_window[s_idx] = r['n_temp_filter']
        elapsed_per_window_s[s_idx] = r['elapsed_filter_s']
        id_cov_per_window[s_idx] = r['id_covered']

    # Per-replan plan as a (n_replans, T_total_days) array
    daily_phi_plan_per_replan = np.array(
        [r['plan_per_day'] for r in replan_history], dtype=np.float32,
    )
    replan_strides = np.array([r['stride'] for r in replan_history], dtype=np.int32)
    replan_n_temp = np.array([r['n_temp'] for r in replan_history], dtype=np.int32)

    # Concatenate accumulated obs
    obs_HR_t_idx     = np.concatenate(accumulated_obs['obs_HR']['t_idx']).astype(np.int32)
    obs_HR_value     = np.concatenate(accumulated_obs['obs_HR']['obs_value']).astype(np.float32)
    obs_sleep_label  = np.concatenate(accumulated_obs['obs_sleep']['sleep_label']).astype(np.int32)
    obs_stress_t_idx = np.concatenate(accumulated_obs['obs_stress']['t_idx']).astype(np.int32)
    obs_stress_value = np.concatenate(accumulated_obs['obs_stress']['obs_value']).astype(np.float32)
    obs_steps_t_idx  = np.concatenate(accumulated_obs['obs_steps']['t_idx']).astype(np.int32)
    obs_steps_value  = np.concatenate(accumulated_obs['obs_steps']['obs_value']).astype(np.float32)
    Phi_per_bin      = np.concatenate(accumulated_obs['Phi']['Phi_value']).astype(np.float32)
    C_per_bin        = np.concatenate(accumulated_obs['C']['C_value']).astype(np.float32)

    np.savez_compressed(
        f"{run_dir}/data.npz",
        # plant trajectories
        trajectory_mpc=traj_full.astype(np.float32),
        trajectory_baseline=traj_baseline.astype(np.float32),
        Phi_per_bin=Phi_per_bin,
        C_per_bin=C_per_bin,
        # 4 obs channels (sparse t_idx + value) for the MPC plant
        obs_HR_t_idx=obs_HR_t_idx, obs_HR_value=obs_HR_value,
        obs_sleep_label=obs_sleep_label,
        obs_stress_t_idx=obs_stress_t_idx, obs_stress_value=obs_stress_value,
        obs_steps_t_idx=obs_steps_t_idx, obs_steps_value=obs_steps_value,
        # Per-stride applied Φ + per-replan plans
        daily_phi_per_stride=np.array(daily_phi_per_stride, dtype=np.float32),
        daily_phi_plan_per_replan=daily_phi_plan_per_replan,
        replan_strides=replan_strides,
        replan_n_temp=replan_n_temp,
        # Per-window posterior particle clouds (constrained)
        posterior_particles=posterior_particles_all,
        posterior_window_mask=posterior_window_mask,
        n_temp_per_window=n_temp_per_window,
        elapsed_per_window_s=elapsed_per_window_s,
        id_cov_per_window=id_cov_per_window,
    )
    print(f"  Checkpoint: {run_dir}/manifest.json + data.npz", flush=True)
    print()

    # ── Diagnostic plot (also dumped into the run folder) ──
    out_dir = run_dir
    out_path = f"{out_dir}/E5_full_mpc_T{T_total_days}d_traces.png"
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t_full = np.arange(traj_full.shape[0]) * DT

    ax = axes[0, 0]
    ax.plot(t_full, traj_full[:, 0], '-', color='steelblue', lw=1.5, label='B (MPC)')
    ax.plot(t_full, traj_baseline[:, 0], '--', color='gray', lw=1, alpha=0.7, label='B (baseline)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('B'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('B trajectory')

    ax = axes[0, 1]
    ax.plot(t_full, traj_full[:, 1], '-', color='darkred', lw=1.5, label='F (MPC)')
    ax.plot(t_full, traj_baseline[:, 1], '--', color='gray', lw=1, alpha=0.7, label='F (baseline)')
    ax.axhline(0.40, color='red', linestyle='--', alpha=0.5, label='F_max')
    ax.set_xlabel('time (days)'); ax.set_ylabel('F'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('F trajectory')

    ax = axes[1, 0]
    ax.plot(t_full, traj_full[:, 2], '-', color='green', lw=1.5, label='A (MPC)')
    ax.plot(t_full, traj_baseline[:, 2], '--', color='gray', lw=1, alpha=0.7, label='A (baseline)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('A'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_title('A trajectory  (mean MPC: {:.3f}, baseline: {:.3f})'.format(
        mean_A_mpc, mean_A_baseline))

    ax = axes[1, 1]
    t_strides = np.arange(len(daily_phi_per_stride)) * STRIDE_BINS * DT
    ax.plot(t_strides, daily_phi_per_stride, 'o-', color='darkorange', lw=2,
              markersize=5, label='applied daily Φ')
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='baseline Φ=1.0')
    ax.set_xlabel('time (days)'); ax.set_ylabel('daily Φ')
    ax.set_title('MPC-applied Φ schedule across {} strides'.format(n_strides))
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stage F — FSA-v2 closed-loop MPC, T={T_total_days}d: '
                  f'mean A {mean_A_mpc:.3f} vs baseline {mean_A_baseline:.3f}, '
                  f'F-viol {F_viol_mpc:.1%}, {total_elapsed/60:.0f} min',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
