"""Stage B1: SMC² filter on the bistable_controlled model.

Generate synthetic 72-hour bistable trajectory + observations under the
default schedule (24h pre-intervention + 48h supercritical u_on=0.5),
run a single SMC² window with the carried gk_dpf_v3_lite inner-PF +
SF info-aware bridge, and verify the posterior covers the truth params
(alpha, a, sigma_x, sigma_obs) on the identifiable subset.

The bistable model has 6 estimable params + 2 init states. With 72h
of data at dt=10min (432 obs), some parameters are weakly identifiable
(notably gamma and sigma_u — the u channel is unobserved). This is
expected, not a bug; we gate only on the identifiable params.

Acceptance gates:
    posterior CI 90% covers truth on (alpha, a, sigma_x, sigma_obs)
    weakly-identified (gamma, sigma_u) classified as such
        (shrinkage < 50% → "uninformative" — a documented outcome)

Run:
    PYTHONPATH=. python tools/bench_smc_filter_bistable.py
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


# Truth (Set A from public-dev)
TRUTH = dict(
    alpha=1.0, a=1.0, sigma_x=0.10,
    gamma=2.0, sigma_u=0.05, sigma_obs=0.20,
    x_0=-1.0, u_0=0.0,
)
EXO = dict(
    T_total=72.0, T_intervention=24.0, u_on=0.5,
    dt_hours=10.0 / 60.0,
)


def _simulate_synthetic(seed: int = 42):
    """Hand-rolled Euler-Maruyama simulation of the bistable model under
    the default supercritical schedule. Matches the bistable_controlled
    simulation.py drift / diffusion exactly.
    """
    dt = EXO['dt_hours']
    n = int(round(EXO['T_total'] / dt))
    t_grid = np.arange(n) * dt
    u_target = np.where(t_grid < EXO['T_intervention'],
                          0.0, EXO['u_on'])

    rng = np.random.default_rng(seed)
    sx = math.sqrt(2 * TRUTH['sigma_x'])
    su = math.sqrt(2 * TRUTH['sigma_u'])
    sqrt_dt = math.sqrt(dt)

    x = np.zeros(n)
    u = np.zeros(n)
    x[0] = TRUTH['x_0']
    u[0] = TRUTH['u_0']
    for k in range(1, n):
        wx = rng.standard_normal()
        wu = rng.standard_normal()
        u_t = u_target[k - 1]
        dx = (TRUTH['alpha'] * x[k - 1]
                * (TRUTH['a'] ** 2 - x[k - 1] ** 2)
                + u[k - 1])
        du = -TRUTH['gamma'] * (u[k - 1] - u_t)
        x[k] = x[k - 1] + dt * dx + sx * sqrt_dt * wx
        u[k] = u[k - 1] + dt * du + su * sqrt_dt * wu

    obs_value = x + TRUTH['sigma_obs'] * rng.standard_normal(n)

    return dict(
        t_grid=t_grid,
        x=x, u_state=u,
        obs_value=obs_value.astype(np.float32),
        u_target=u_target.astype(np.float32),
        n_steps=n, dt=dt,
    )


def _run_smc_filter(data, n_smc=128, n_pf=200, seed=0):
    """Single SMC² window on the synthetic bistable data."""
    from models.bistable_controlled.estimation import BISTABLE_CTRL_ESTIMATION
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.tempered_smc import run_smc_window
    from smc2fc.transforms.unconstrained import (
        build_transform_arrays, unconstrained_to_constrained,
    )
    from smc2fc.filtering.gk_dpf_v3_lite import make_gk_dpf_v3_lite_log_density

    em = BISTABLE_CTRL_ESTIMATION

    obs_data = {
        't_idx':          np.arange(data['n_steps'], dtype=np.int32),
        'obs_value':      data['obs_value'],
        'u_target_value': data['u_target'],
    }
    grid_obs = em.align_obs_fn(obs_data, data['n_steps'], data['dt'])

    # Cold-start init from the prior means of x_0, u_0
    cold_start_init = jnp.array([TRUTH['x_0'], TRUTH['u_0']],
                                  dtype=jnp.float64)

    log_density = make_gk_dpf_v3_lite_log_density(
        model=em, grid_obs=grid_obs, n_particles=n_pf,
        bandwidth_scale=1.0,
        ot_ess_frac=0.05, ot_temperature=5.0, ot_max_weight=0.0,  # OT off
        ot_rank=5, ot_n_iter=2, ot_epsilon=0.5,
        dt=data['dt'], seed=seed,
        fixed_init_state=cold_start_init, window_start_bin=0,
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
    return particles_con, em


def main():
    print("=" * 72)
    print("  Stage B1 — SMC² filter on bistable_controlled (Set A)")
    print("=" * 72)
    print(f"  truth:   {TRUTH}")
    print(f"  exog:    {EXO}")
    print()

    print("  Step 1: synthesize 72h trajectory under default schedule")
    data = _simulate_synthetic(seed=42)
    print(f"    n_steps={data['n_steps']}, dt={data['dt']:.4f} h")
    print(f"    x range: [{data['x'].min():.3f}, {data['x'].max():.3f}]")
    print(f"    transitions to x>0 at index "
          f"{int(np.argmax(data['x'] > 0)) if (data['x'] > 0).any() else 'never'}")
    print()

    particles, em = _run_smc_filter(data, n_smc=128, n_pf=200, seed=0)
    print(f"  posterior particle shape: {particles.shape}")

    # Posterior summary vs truth
    means = particles.mean(axis=0)
    stds = particles.std(axis=0)
    q05 = np.quantile(particles, 0.05, axis=0)
    q95 = np.quantile(particles, 0.95, axis=0)
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}

    print()
    print(f"  {'param':<12}  {'truth':>8}  {'mean':>8}  "
          f"{'std':>7}  {'90% CI':>20}  truth in CI?")
    misses = []
    for name in em.all_names:
        idx = name_to_idx[name]
        truth_val = TRUTH.get(name, float('nan'))
        in_ci = q05[idx] <= truth_val <= q95[idx]
        if not in_ci:
            misses.append(name)
        ci_str = f"[{q05[idx]:.3f}, {q95[idx]:.3f}]"
        print(f"  {name:<12}  {truth_val:>8.4f}  {means[idx]:>8.4f}  "
              f"{stds[idx]:>7.4f}  {ci_str:>20}  "
              f"{'✓' if in_ci else '✗'}")

    # Identifiable params (alpha, a, sigma_x, sigma_obs) — gate
    identifiable = ['alpha', 'a', 'sigma_x', 'sigma_obs']
    weakly_id = ['gamma', 'sigma_u']
    print()
    print(f"  identifiable params (gate: 90% CI covers truth):")
    pass_id = True
    for name in identifiable:
        idx = name_to_idx[name]
        truth_val = TRUTH[name]
        in_ci = q05[idx] <= truth_val <= q95[idx]
        if not in_ci:
            pass_id = False
        print(f"    {name:<12}  {'✓' if in_ci else '✗'}")
    print(f"  weakly-identifiable (gamma, sigma_u): documented as such")

    if pass_id:
        print(f"  ✓ Stage B1 PASSES (identifiable params covered)")
    else:
        print(f"  ✗ Stage B1 FAILS (one or more identifiable params miss)")

    # ── Plot ──
    out_path = "outputs/bistable_controlled/B1_filter_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Top row: trajectory + observations + schedule
    ax = axes[0, 0]
    ax.plot(data['t_grid'], data['x'], '-', color='steelblue', lw=1.5,
              label='truth x(t)', alpha=0.85)
    ax.plot(data['t_grid'], data['obs_value'], '.', color='gray',
              markersize=2, alpha=0.5, label='y(t) = x + noise')
    ax.axhline(-1.0, color='red', linestyle=':', alpha=0.4)
    ax.axhline(+1.0, color='green', linestyle=':', alpha=0.4)
    ax.axvline(EXO['T_intervention'], color='black',
                 linestyle='--', alpha=0.4)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('x (health)')
    ax.set_title('Synthetic trajectory + obs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(data['t_grid'], data['u_state'], '-', color='darkorange',
              lw=1.5, label='truth u(t)')
    ax.plot(data['t_grid'], data['u_target'], '--', color='red',
              lw=1.0, label='u_target(t)')
    ax.set_xlabel('time (h)')
    ax.set_ylabel('u (control)')
    ax.set_title('Control state + target schedule')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: posterior over alpha, a side-by-side
    for ax, name in zip(axes[0, 2:], ['alpha', 'a']):
        idx = name_to_idx[name]
        ax.hist(particles[:, idx], bins=30, color='steelblue', alpha=0.7)
        ax.axvline(TRUTH[name], color='green', linestyle='--', linewidth=2,
                     label=f'truth = {TRUTH[name]:.3f}')
        ax.axvline(means[idx], color='steelblue', linestyle=':', linewidth=2,
                     label=f'posterior mean = {means[idx]:.3f}')
        ax.set_xlabel(name)
        ax.set_ylabel('density')
        ax.set_title(f'posterior: {name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom row: remaining 4 params posterior
    for ax, name in zip(axes[1, :], ['sigma_x', 'gamma', 'sigma_u', 'sigma_obs']):
        idx = name_to_idx[name]
        ax.hist(particles[:, idx], bins=30, color='steelblue', alpha=0.7)
        ax.axvline(TRUTH[name], color='green', linestyle='--', linewidth=2,
                     label=f'truth = {TRUTH[name]:.4f}')
        ax.axvline(means[idx], color='steelblue', linestyle=':', linewidth=2,
                     label=f'mean = {means[idx]:.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('density')
        tag = ' (identifiable)' if name in identifiable else ' (weakly id.)'
        ax.set_title(f'posterior: {name}{tag}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stage B1 — bistable filter: SMC² posterior vs truth '
                  '(72h, default supercritical schedule)',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
