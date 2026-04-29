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
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Re-use E4's helper for building a posterior-mean control spec
from tools.bench_smc_closed_loop_fsa import _build_phase2_control_spec


N_DAYS_TOTAL = 14
WINDOW_BINS  = 96    # 1 day
STRIDE_BINS  = 48    # 12 hours
DT = 1.0 / 96.0
REPLAN_EVERY_K = 2   # replan once per day (every 2 strides)


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
    print("=" * 76)
    print("  Stage E5 — full 27-window rolling MPC on FSA-v2")
    print("=" * 76)

    from models.fsa_high_res._plant import StepwisePlant
    from models.fsa_high_res.simulation import DEFAULT_PARAMS
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V2_ESTIMATION, COLD_START_INIT,
    )
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.tempered_smc import run_smc_window, run_smc_window_bridge
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained
    from smc2fc.filtering.gk_dpf_v3_lite import make_gk_dpf_v3_lite_log_density
    from smc2fc.control import SMCControlConfig, run_tempered_smc_loop

    truth = dict(DEFAULT_PARAMS)
    em = HIGH_RES_FSA_V2_ESTIMATION
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}
    identifiable_subset = {'HR_base', 'S_base', 'mu_step0',
                            'kappa_B_HR', 'k_F', 'beta_C_HR'}

    n_windows = (N_DAYS_TOTAL * 96 - WINDOW_BINS) // STRIDE_BINS + 1
    n_strides = n_windows
    print(f"  device:   {jax.devices()[0].platform.upper()}")
    print(f"  windows:  {n_windows} (1-day, 12h stride)")
    print(f"  replan:   every K={REPLAN_EVERY_K} stride(s)")
    print()

    # ── Initialize plant ──
    plant = StepwisePlant(seed_offset=42)
    daily_phi_baseline = 1.0
    daily_phi_planned = daily_phi_baseline    # warm-start identical to baseline

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

    smc_cfg = SMCConfig(
        n_smc_particles=128, n_pf_particles=200,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='gaussian',
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.15,
    )
    ctrl_cfg = SMCControlConfig(
        n_smc=128, n_inner=32, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=10, hmc_step_size=0.20, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0, max_temp_steps=30,
    )

    prev_particles = None
    fixed_init_state = COLD_START_INIT
    all_results = []

    total_t0 = time.time()

    for s in range(n_strides):
        print(f"  Stride {s+1:>2}/{n_strides}: ", end='', flush=True)

        # 1. Apply current planned Phi to plant
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
        ld = make_gk_dpf_v3_lite_log_density(
            model=em, grid_obs=grid_obs, n_particles=smc_cfg.n_pf_particles,
            bandwidth_scale=smc_cfg.bandwidth_scale,
            ot_ess_frac=smc_cfg.ot_ess_frac, ot_temperature=smc_cfg.ot_temperature,
            ot_max_weight=smc_cfg.ot_max_weight,
            ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
            ot_epsilon=smc_cfg.ot_epsilon,
            dt=DT, seed=42 + s,
            fixed_init_state=fixed_init_state, window_start_bin=int(start_bin),
        )
        T_arr = ld._transforms

        if prev_particles is None:
            print(f"filter (cold)... ", end='', flush=True)
            particles_unc, elapsed_f, n_temp_f = run_smc_window(
                ld, em, T_arr, cfg=smc_cfg,
                initial_particles=None, seed=42 + s * 1000,
            )
        else:
            print(f"filter (bridge)... ", end='', flush=True)
            particles_unc, elapsed_f, n_temp_f = run_smc_window_bridge(
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

        # Smoothed end-of-window state (for next window's PF init AND control plan)
        n_extract = min(10, particles_unc.shape[0])
        extracted = []
        for ei in range(n_extract):
            u_draw = jnp.array(particles_unc[ei])
            st = ld.extract_state_at_step(u_draw, STRIDE_BINS)
            extracted.append(np.array(st))
        smoothed_state = np.mean(extracted, axis=0)
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
            spec = _build_phase2_control_spec(
                dyn_params=dyn_params, init_state=smoothed_state,
            )
            res_ctrl = run_tempered_smc_loop(
                spec=spec, cfg=ctrl_cfg, seed=42 + s,
                print_progress=False,
            )
            schedule = np.asarray(res_ctrl['mean_schedule'])
            new_daily_phi = float(schedule.mean())
            print(f"plan: {res_ctrl['n_temp_levels']}lvl, Φ̄={new_daily_phi:.3f}",
                   flush=True)
            daily_phi_planned = new_daily_phi
        else:
            print(f"reuse Φ={daily_phi_planned:.3f}", flush=True)

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
    n_total_bins = traj_full.shape[0]   # 27 strides × 48 = 1296 bins
    n_days_baseline = (n_total_bins + 95) // 96   # round up to whole days
    plant_b = StepwisePlant(seed_offset=42)
    plant_b.advance(n_days_baseline * 96, np.full(n_days_baseline, 1.0))
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

    # ── Diagnostic plot ──
    out_dir = "outputs/fsa_high_res"
    out_path = f"{out_dir}/E5_full_mpc_traces.png"
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

    plt.suptitle(f'Stage E5 — FSA-v2 27-window rolling MPC: '
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
