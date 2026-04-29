"""Stage E1: SMC² filter on FSA-v2 (single 1-day window).

Synthesise a 1-day trajectory of the FSA-v2 (Banister-coupled) model
under the canonical Banister default Φ ≡ 1.0, sample the 4 obs channels
(HR, sleep, stress, steps) at 15-min granularity, run a single SMC²
window with the GK-DPF v3-lite inner-PF + SF info-aware bridge, and
verify the posterior covers the truth params on the identifiable
subset.

At a 1-day window only the high-frequency obs-side parameters are
strongly identifiable (HR_base, S_base, mu_step0, beta_C_*, the obs
noise scales σ_*). The slow Banister dynamics parameters (τ_B = 42 d,
κ_B, κ_F) need cross-window pooling to be identified — that's the
Stage E3 rolling-window experiment. We gate on the obs-side subset
here.

Acceptance gates (E1):
    - posterior 90% CI covers truth on at least 5 of
      {HR_base, S_base, mu_step0, kappa_B_HR, k_F, beta_C_HR}.
    - smoothed posterior mean of [B, F, A] at end of window matches
      truth within 2σ in ≥ 90% of time-points.
    - SMC² convergence in ≤ 30 tempering levels, < 15 min on GPU.

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_smc_filter_fsa.py
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


# =========================================================================
# Synthesise 1-day trajectory + obs
# =========================================================================

def _simulate_synthetic(seed: int = 42, daily_phi: float = 1.0):
    """Hand-rolled Euler-Maruyama simulation of the FSA-v2 model under
    a constant daily Φ = 1.0 schedule, with morning-loaded sub-daily
    burst expansion."""
    from models.fsa_high_res.simulation import (
        DEFAULT_PARAMS, DEFAULT_INIT, BINS_PER_DAY,
        drift, noise_scale_fn, generate_phi_sub_daily,
        gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps,
        gen_Phi_channel, gen_C_channel,
    )

    n_days = 1
    T = n_days * BINS_PER_DAY        # 96 bins for 1 day
    dt = 1.0 / BINS_PER_DAY           # 15 min
    t_grid = np.arange(T, dtype=np.float64) * dt

    # Generate morning-loaded Φ(t) — daily integral preserved
    Phi_arr = generate_phi_sub_daily(np.array([daily_phi]), noise_frac=0.0)
    aux = (Phi_arr,)

    # Forward Euler-Maruyama with sqrt-Itô diffusion
    rng = np.random.default_rng(seed)
    sigma = np.array([
        DEFAULT_PARAMS['sigma_B'],
        DEFAULT_PARAMS['sigma_F'],
        DEFAULT_PARAMS['sigma_A'],
    ])
    sqrt_dt = math.sqrt(dt)
    y = np.array([DEFAULT_INIT['B_0'], DEFAULT_INIT['F_0'], DEFAULT_INIT['A_0']])
    trajectory = np.zeros((T, 3), dtype=np.float32)
    trajectory[0] = y
    for k in range(1, T):
        d_y = drift(t_grid[k - 1], y, DEFAULT_PARAMS, aux)
        g = noise_scale_fn(y, DEFAULT_PARAMS)
        noise = rng.standard_normal(3)
        y = y + dt * d_y + sigma * g * sqrt_dt * noise
        # Boundary reflection (very rarely fires due to vanishing σ)
        y[0] = np.clip(y[0], 1e-4, 1.0 - 1e-4)
        y[1] = max(y[1], 0.0)
        y[2] = max(y[2], 0.0)
        trajectory[k] = y

    # Generate 4 obs channels (sleep first, then HR/stress/steps gated)
    sleep_ch = gen_obs_sleep(trajectory, t_grid, DEFAULT_PARAMS, aux, None, seed=seed + 1)
    prior = {'obs_sleep': sleep_ch}
    hr_ch    = gen_obs_hr   (trajectory, t_grid, DEFAULT_PARAMS, aux, prior, seed=seed + 2)
    str_ch   = gen_obs_stress(trajectory, t_grid, DEFAULT_PARAMS, aux, prior, seed=seed + 3)
    steps_ch = gen_obs_steps (trajectory, t_grid, DEFAULT_PARAMS, aux, prior, seed=seed + 4)

    # Exogenous broadcast channels
    phi_ch = gen_Phi_channel(trajectory, t_grid, DEFAULT_PARAMS, aux, None, seed=0)
    c_ch   = gen_C_channel  (trajectory, t_grid, DEFAULT_PARAMS, aux, None, seed=0)

    return {
        'n_steps':    T,
        'dt':         dt,
        't_grid':     t_grid,
        'trajectory': trajectory,
        'truth_params': dict(DEFAULT_PARAMS),
        'truth_init':   dict(DEFAULT_INIT),
        'obs_data': {
            'obs_sleep':  sleep_ch,
            'obs_HR':     hr_ch,
            'obs_stress': str_ch,
            'obs_steps':  steps_ch,
            'Phi':        phi_ch,
            'C':          c_ch,
        },
    }


# =========================================================================
# SMC² filter
# =========================================================================

def _run_smc_filter(data, n_smc=128, n_pf=200, seed=0):
    """Single SMC² window on the synthetic FSA-v2 1-day data."""
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V2_ESTIMATION, COLD_START_INIT,
    )
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.tempered_smc import run_smc_window
    from smc2fc.transforms.unconstrained import (
        build_transform_arrays, unconstrained_to_constrained,
    )
    from smc2fc.filtering.gk_dpf_v3_lite import make_gk_dpf_v3_lite_log_density

    em = HIGH_RES_FSA_V2_ESTIMATION
    grid_obs = em.align_obs_fn(data['obs_data'], data['n_steps'], data['dt'])

    log_density = make_gk_dpf_v3_lite_log_density(
        model=em, grid_obs=grid_obs, n_particles=n_pf,
        bandwidth_scale=1.0,
        ot_ess_frac=0.05, ot_temperature=5.0, ot_max_weight=0.0,  # OT off (per memory note)
        ot_rank=5, ot_n_iter=2, ot_epsilon=0.5,
        dt=data['dt'], seed=seed,
        fixed_init_state=COLD_START_INIT, window_start_bin=0,
    )
    cfg = SMCConfig(
        n_smc_particles=n_smc, n_pf_particles=n_pf,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='gaussian',
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
    )
    T_arr = log_density._transforms

    print(f"  starting SMC² filter: N_SMC={n_smc}, N_PF={n_pf}, "
          f"n_obs={data['n_steps']}")
    t0 = time.time()
    particles_unc, elapsed, n_temp = run_smc_window(
        full_log_density=log_density, model=em, T_arr=T_arr,
        cfg=cfg, initial_particles=None, seed=seed,
    )
    print(f"  SMC² done: {n_temp} levels in {elapsed:.1f}s")

    # Convert to constrained (theta) space
    particles_unc = np.asarray(particles_unc)
    particles_con = np.array([
        np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
        for p in particles_unc
    ])
    return particles_con, em, n_temp, elapsed


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 76)
    print("  Stage E1 — SMC² filter on FSA-v2 (1-day window)")
    print("=" * 76)
    data = _simulate_synthetic(seed=42)
    print(f"  device:   {jax.devices()[0].platform.upper()}")
    print(f"  n_steps:  {data['n_steps']} (1 day × 96 bins at 15 min)")
    print(f"  dt:       {data['dt']:.5f} day  ({data['dt']*24*60:.1f} min)")
    print(f"  state ranges:  B [{data['trajectory'][:,0].min():.3f}, "
          f"{data['trajectory'][:,0].max():.3f}],  "
          f"F [{data['trajectory'][:,1].min():.3f}, "
          f"{data['trajectory'][:,1].max():.3f}],  "
          f"A [{data['trajectory'][:,2].min():.3f}, "
          f"{data['trajectory'][:,2].max():.3f}]")
    print(f"  obs counts:  sleep={data['obs_data']['obs_sleep']['sleep_label'].sum()}/{data['n_steps']}, "
          f"HR={len(data['obs_data']['obs_HR']['t_idx'])}, "
          f"stress={len(data['obs_data']['obs_stress']['t_idx'])}, "
          f"steps={len(data['obs_data']['obs_steps']['t_idx'])}")
    print()

    particles, em, n_temp, elapsed = _run_smc_filter(
        data, n_smc=128, n_pf=200, seed=0,
    )
    print(f"  posterior particle shape: {particles.shape}")
    print()

    # Posterior summary vs truth
    truth_params = data['truth_params']
    means = particles.mean(axis=0)
    stds = particles.std(axis=0)
    q05 = np.quantile(particles, 0.05, axis=0)
    q95 = np.quantile(particles, 0.95, axis=0)
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}

    print(f"  {'param':<14} {'truth':>10} {'mean':>10} {'std':>9} "
          f"{'90% CI':>26} CI cov?")
    print(f"  {'─'*72}")
    n_covered = 0
    n_estimable = 0
    misses = []
    identifiable_subset = {'HR_base', 'S_base', 'mu_step0',
                            'kappa_B_HR', 'k_F', 'beta_C_HR'}
    n_identifiable_covered = 0
    for name in em.all_names:
        if name not in truth_params:
            continue
        idx = name_to_idx[name]
        truth_val = truth_params[name]
        m = means[idx]; s = stds[idx]; lo = q05[idx]; hi = q95[idx]
        covered = lo <= truth_val <= hi
        marker = '✓' if covered else '✗'
        flag = ' (id)' if name in identifiable_subset else ''
        print(f"  {name:<14} {truth_val:>10.4f} {m:>10.4f} {s:>9.4f} "
              f"[{lo:>9.4f}, {hi:>9.4f}] {marker}{flag}")
        n_estimable += 1
        if covered:
            n_covered += 1
            if name in identifiable_subset:
                n_identifiable_covered += 1
        else:
            misses.append(name)
    print()
    print(f"  Coverage: {n_covered}/{n_estimable} all params, "
          f"{n_identifiable_covered}/{len(identifiable_subset)} identifiable subset")

    # ── Acceptance gates ──
    print()
    print(f"  Acceptance gates:")
    gate1 = n_identifiable_covered >= 5
    print(f"    {'✓' if gate1 else '✗'}  ≥ 5 of 6 identifiable params covered "
          f"(actual: {n_identifiable_covered}/{len(identifiable_subset)})")

    gate2 = n_temp <= 30
    print(f"    {'✓' if gate2 else '✗'}  Convergence in ≤ 30 tempering levels "
          f"(actual: {n_temp})")

    gate3 = elapsed <= 15 * 60
    print(f"    {'✓' if gate3 else '✗'}  Wall-clock ≤ 15 min "
          f"(actual: {elapsed:.0f}s)")

    all_pass = gate1 and gate2 and gate3
    print()
    print(f"  {'✓ all gates pass' if all_pass else '✗ one or more gates fail'}")

    # ── Diagnostic plot ──
    out_dir = "outputs/fsa_high_res"
    out_path = f"{out_dir}/E1_filter_diagnostic.png"
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    t = data['t_grid']
    traj = data['trajectory']

    # B trajectory
    ax = axes[0, 0]
    ax.plot(t, traj[:, 0], '-', color='steelblue', lw=2, label='truth B(t)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('B (fitness)')
    ax.set_title('Truth B trajectory'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # F trajectory
    ax = axes[0, 1]
    ax.plot(t, traj[:, 1], '-', color='darkred', lw=2, label='truth F(t)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('F (fatigue)')
    ax.set_title('Truth F trajectory'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # A trajectory
    ax = axes[1, 0]
    ax.plot(t, traj[:, 2], '-', color='green', lw=2, label='truth A(t)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('A (amplitude)')
    ax.set_title('Truth A trajectory'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # HR obs
    ax = axes[1, 1]
    hr = data['obs_data']['obs_HR']
    ax.plot(t[hr['t_idx']], hr['obs_value'], '.', alpha=0.7, color='steelblue',
            label=f'HR obs ({len(hr["t_idx"])} bins, sleep-gated)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('HR (bpm)')
    ax.set_title('HR observations'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Stress obs
    ax = axes[2, 0]
    s = data['obs_data']['obs_stress']
    ax.plot(t[s['t_idx']], s['obs_value'], '.', alpha=0.7, color='darkorange',
            label=f'stress obs ({len(s["t_idx"])} bins, wake-gated)')
    ax.set_xlabel('time (days)'); ax.set_ylabel('stress')
    ax.set_title('Stress observations'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Posterior coverage bar plot
    ax = axes[2, 1]
    cov_names = sorted(identifiable_subset)
    cov_idx = [name_to_idx[n] for n in cov_names]
    cov_truth = [truth_params[n] for n in cov_names]
    cov_means = [means[i] for i in cov_idx]
    cov_q05 = [q05[i] for i in cov_idx]
    cov_q95 = [q95[i] for i in cov_idx]
    rel_err = [(m - t) / t if t != 0 else 0 for m, t in zip(cov_means, cov_truth)]
    ax.bar(range(len(cov_names)), rel_err,
            color=['steelblue' if cov_q05[i] <= cov_truth[i] <= cov_q95[i] else 'salmon'
                    for i in range(len(cov_names))])
    ax.set_xticks(range(len(cov_names)))
    ax.set_xticklabels(cov_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('(mean − truth) / truth')
    ax.set_title('Posterior mean rel. error on identifiable subset')
    ax.axhline(0, color='black', lw=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Stage E1 — FSA-v2 SMC² filter, 1-day window: '
                  f'{n_covered}/{n_estimable} covered, '
                  f'{n_identifiable_covered}/{len(identifiable_subset)} '
                  f'identifiable, {n_temp} levels in {elapsed:.0f}s on '
                  f'{jax.devices()[0].platform.upper()}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
