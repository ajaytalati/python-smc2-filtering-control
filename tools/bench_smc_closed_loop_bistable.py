"""Stage B3: closed-loop SMC² on bistable_controlled.

Single-iteration MPC: filter Phase 1 (0-24h, no control) → posterior
mean params → SMC² controller plans Phase 2 → apply to TRUE plant.

The Phase-2 SMC² controller uses a ControlSpec built from the
posterior-mean params (oracle uses truth params for comparison).
Both controllers reuse smc2fc.control + the same RBF basis +
cost functional from models/bistable_controlled/control.py — only
the truth_params dict differs.
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

from smc2fc.control import (
    SMCControlConfig, run_tempered_smc_loop, evaluate_gates, RBFSchedule,
)
from smc2fc.control.calibration import build_crn_noise_grids


# Truth + filter machinery — keep at this script level since B3
# combines filter + control phases (model-specific orchestration).

TRUTH = dict(
    alpha=1.0, a=1.0, sigma_x=0.10,
    gamma=2.0, sigma_u=0.05, sigma_obs=0.20,
    x_0=-1.0, u_0=0.0,
)
EXO = dict(
    T_total=72.0, T_intervention=24.0, u_on=0.5,
    dt_hours=10.0 / 60.0,
)


def _simulate_phase1(t_phase1_h=24.0, seed=42):
    """Synthesize 0..t_phase1_h with u_target=0 (pre-intervention)."""
    dt = EXO['dt_hours']
    n = int(round(t_phase1_h / dt))
    rng = np.random.default_rng(seed)
    sx = math.sqrt(2 * TRUTH['sigma_x'])
    su = math.sqrt(2 * TRUTH['sigma_u'])
    sqrt_dt = math.sqrt(dt)

    x = np.zeros(n)
    u = np.zeros(n)
    x[0] = TRUTH['x_0']
    u[0] = TRUTH['u_0']
    for k in range(1, n):
        wx, wu = rng.standard_normal(), rng.standard_normal()
        dx = TRUTH['alpha'] * x[k-1] * (TRUTH['a']**2 - x[k-1]**2) + u[k-1]
        du_t = -TRUTH['gamma'] * u[k-1]
        x[k] = x[k-1] + dt * dx + sx * sqrt_dt * wx
        u[k] = u[k-1] + dt * du_t + su * sqrt_dt * wu

    obs = x + TRUTH['sigma_obs'] * rng.standard_normal(n)
    return dict(t_grid=np.arange(n) * dt, x=x, u_state=u,
                obs_value=obs.astype(np.float32),
                u_target=np.zeros(n, dtype=np.float32),
                n_steps=n, dt=dt,
                final_state=np.array([x[-1], u[-1]]))


def _filter_phase1(data, n_smc=64, n_pf=128, seed=0):
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
    cold = jnp.array([TRUTH['x_0'], TRUTH['u_0']], dtype=jnp.float64)
    log_density = make_gk_dpf_v3_lite_log_density(
        model=em, grid_obs=grid_obs, n_particles=n_pf,
        bandwidth_scale=1.0,
        ot_ess_frac=0.05, ot_temperature=5.0, ot_max_weight=0.0,
        ot_rank=5, ot_n_iter=2, ot_epsilon=0.5,
        dt=data['dt'], seed=seed,
        fixed_init_state=cold, window_start_bin=0,
    )
    cfg = SMCConfig(
        n_smc_particles=n_smc, n_pf_particles=n_pf,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='gaussian',
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
    )
    T_arr = log_density._transforms

    print(f"  filtering Phase 1 ({data['n_steps']} obs)...")
    t0 = time.time()
    particles_unc, elapsed, n_temp = run_smc_window(
        full_log_density=log_density, model=em, T_arr=T_arr,
        cfg=cfg, initial_particles=None, seed=seed,
    )
    print(f"  filter done: {n_temp} levels in {elapsed:.1f}s")

    particles_unc = np.asarray(particles_unc)
    particles_con = np.array([
        np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
        for p in particles_unc
    ])
    return {n: float(particles_con[:, i].mean())
            for i, n in enumerate(em.all_names)}


# ── Build a Phase-2 ControlSpec parameterised by truth/posterior params ──

def _build_phase2_spec(params, x_init, u_init,
                          t_phase2_h=48.0, n_anchors=6,
                          n_inner=32, seed=11):
    """Build a ControlSpec that uses the given params dict (truth or
    posterior-mean) for Phase-2 dynamics. Cost: tracking + control
    effort + barrier."""
    from smc2fc.control import ControlSpec

    dt = EXO['dt_hours']
    n_steps = int(round(t_phase2_h / dt))

    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors,
                        output='softplus')
    Phi = rbf.design_matrix()

    sx_proc = math.sqrt(2 * params['sigma_x'])
    su_proc = math.sqrt(2 * params['sigma_u'])
    sqrt_dt = math.sqrt(dt)

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=2, seed=seed,
    )
    fixed_wx = grids['wiener'][:, :, 0]
    fixed_wu = grids['wiener'][:, :, 1]

    @jax.jit
    def schedule_from_theta(theta):
        return rbf.from_theta(theta, Phi=Phi)

    @jax.jit
    def cost_fn(theta):
        u_target = schedule_from_theta(theta)

        def trial(wx_seq, wu_seq):
            def step(carry, k):
                x, u, c = carry
                u_tgt = u_target[k]
                dx = (params['alpha'] * x
                       * (params['a'] ** 2 - x ** 2) + u) * dt
                du = -params['gamma'] * (u - u_tgt) * dt
                x_next = x + dx + sx_proc * sqrt_dt * wx_seq[k]
                u_next = u + du + su_proc * sqrt_dt * wu_seq[k]
                stage = ((x - 1.0) ** 2 + 0.5 * u_tgt ** 2
                         + 50.0 * jax.nn.softplus(-(x + 0.5) - 1.0)) * dt
                return (x_next, u_next, c + stage), None
            (x_T, _, total), _ = jax.lax.scan(
                step,
                (jnp.float64(x_init), jnp.float64(u_init), jnp.float64(0.0)),
                jnp.arange(n_steps),
            )
            return total + 5.0 * (x_T - 1.0) ** 2
        return jnp.mean(jax.vmap(trial)(fixed_wx, fixed_wu))

    spec = ControlSpec(
        name='bistable_phase2',
        version='1.0',
        dt=dt, n_steps=n_steps, n_substeps=1,
        initial_state=jnp.array([x_init, u_init]),
        truth_params=params,
        theta_dim=n_anchors,
        sigma_prior=1.5,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
        acceptance_gates={},
    )
    return spec


def _apply_schedule(x_init, u_init, schedule, n_steps, dt,
                      n_trials=100, seed=23):
    rng = np.random.default_rng(seed)
    sx = math.sqrt(2 * TRUTH['sigma_x'])
    su = math.sqrt(2 * TRUTH['sigma_u'])
    sqrt_dt = math.sqrt(dt)
    trajs_x = np.zeros((n_trials, n_steps))
    trajs_u = np.zeros((n_trials, n_steps))
    successes = 0
    for trial in range(n_trials):
        x = x_init
        u = u_init
        success = False
        for k in range(n_steps):
            wx, wu = rng.standard_normal(), rng.standard_normal()
            u_tgt = float(schedule[k])
            dx = (TRUTH['alpha'] * x
                   * (TRUTH['a'] ** 2 - x ** 2) + u)
            du = -TRUTH['gamma'] * (u - u_tgt)
            x = x + dt * dx + sx * sqrt_dt * wx
            u = u + dt * du + su * sqrt_dt * wu
            trajs_x[trial, k] = x
            trajs_u[trial, k] = u
            if x > 0.5 and not success:
                success = True
        if success:
            successes += 1
    return successes / n_trials, trajs_x, trajs_u


def main():
    print("=" * 72)
    print("  Stage B3 — closed-loop SMC² on bistable_controlled (refactored)")
    print("=" * 72)
    print()

    print("  Step 1: simulate Phase 1 (0-24h, no control)")
    p1 = _simulate_phase1(t_phase1_h=24.0, seed=42)
    x_at_24h, u_at_24h = p1['final_state']
    print(f"    final state at 24h: x={x_at_24h:.3f}, u={u_at_24h:.3f}")
    print()

    print("  Step 2: SMC² filter on Phase-1 obs")
    posterior = _filter_phase1(p1, n_smc=64, n_pf=128, seed=0)
    for k in ('alpha', 'a', 'sigma_x', 'gamma', 'sigma_u', 'sigma_obs'):
        rel_err = abs(posterior[k] - TRUTH[k]) / TRUTH[k]
        print(f"    {k:<10}  posterior {posterior[k]:>8.4f}  "
              f"truth {TRUTH[k]:>8.4f}  rel_err {rel_err:.2%}")
    print()

    cfg = SMCControlConfig(
        n_smc=128, n_inner=32, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5, hmc_step_size=0.05, hmc_num_leapfrog=8,
        beta_max_target_nats=8.0,
    )

    print("  Step 3: plan Phase 2 with SMC² + posterior-mean params")
    spec_post = _build_phase2_spec(posterior, x_at_24h, u_at_24h, seed=11)
    res_post = run_tempered_smc_loop(spec=spec_post, cfg=cfg, seed=42)
    schedule_post = np.asarray(res_post['mean_schedule'])
    print()

    print("  Step 4: oracle (controller using TRUTH params)")
    spec_oracle = _build_phase2_spec(
        {k: TRUTH[k] for k in ('alpha', 'a', 'sigma_x', 'gamma', 'sigma_u')},
        x_at_24h, u_at_24h, seed=11,
    )
    res_oracle = run_tempered_smc_loop(spec=spec_oracle, cfg=cfg, seed=42)
    schedule_oracle = np.asarray(res_oracle['mean_schedule'])
    print()

    n2 = spec_post.n_steps
    dt = spec_post.dt
    schedule_default = np.full(n2, EXO['u_on'])

    print("  Step 5: apply each schedule to TRUE plant (100 trials each)")
    rate_post, trajs_x_post, _ = _apply_schedule(
        x_at_24h, u_at_24h, schedule_post, n2, dt, n_trials=100, seed=23)
    rate_oracle, trajs_x_oracle, _ = _apply_schedule(
        x_at_24h, u_at_24h, schedule_oracle, n2, dt, n_trials=100, seed=23)
    rate_default, trajs_x_default, _ = _apply_schedule(
        x_at_24h, u_at_24h, schedule_default, n2, dt, n_trials=100, seed=23)
    print(f"    SMC²-with-posterior rate: {rate_post:.0%}")
    print(f"    SMC²-with-truth rate:     {rate_oracle:.0%}")
    print(f"    default rate:             {rate_default:.0%}")

    def _cum_cost(traj_x, schedule, dt):
        return float(((traj_x - 1.0) ** 2 * dt).sum(axis=1).mean()
                     + 0.5 * (schedule ** 2 * dt).sum()
                     + 5.0 * ((traj_x[:, -1] - 1.0) ** 2).mean())

    cost_post = _cum_cost(trajs_x_post, schedule_post, dt)
    cost_oracle = _cum_cost(trajs_x_oracle, schedule_oracle, dt)
    cost_default = _cum_cost(trajs_x_default, schedule_default, dt)

    print()
    print(f"  Cumulative costs (mean over 100 trials):")
    print(f"    SMC²-with-posterior:  {cost_post:.3f}")
    print(f"    SMC²-with-truth:      {cost_oracle:.3f}")
    print(f"    default:              {cost_default:.3f}")
    print()
    pass_rate = rate_post >= 0.80
    pass_cost = cost_post <= 1.20 * cost_oracle
    print(f"  Acceptance gates:")
    print(f"    transition rate ≥ 80%:    {rate_post:.0%}  "
          f"{'✓' if pass_rate else '✗'}")
    print(f"    cost ≤ 1.20 × oracle:     {cost_post / cost_oracle:.3f}×  "
          f"{'✓' if pass_cost else '✗'}")
    if pass_rate and pass_cost:
        print(f"  ✓ Stage B3 PASSES")
    else:
        print(f"  ✗ Stage B3 fails one or more gates")

    # Plot
    out_path = "outputs/bistable_controlled/B3_closed_loop_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.plot(p1['t_grid'], p1['x'], '-', color='steelblue', lw=1.5,
              label='truth x(t) [Phase 1]')
    ax.plot(p1['t_grid'], p1['obs_value'], '.', color='gray', markersize=2,
              alpha=0.5, label='observations')
    ax.axhline(-1.0, color='red', linestyle=':', alpha=0.4)
    ax.axvline(EXO['T_intervention'], color='black', linestyle='--', alpha=0.4)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('x (health)')
    ax.set_title('Phase 1: 0-24h, no control')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    t_phase2 = EXO['T_intervention'] + np.arange(n2) * dt
    ax.plot(t_phase2, schedule_post, '-', color='steelblue', lw=2,
              label='SMC² (posterior-params)')
    ax.plot(t_phase2, schedule_oracle, '--', color='purple', lw=1.5,
              label='oracle (truth-params)', alpha=0.85)
    ax.plot(t_phase2, schedule_default, '-', color='darkred', lw=1.5,
              label='default u_on=0.5')
    u_c = 2.0 * TRUTH['alpha'] * TRUTH['a'] ** 3 / (3 * math.sqrt(3))
    ax.axhline(u_c, color='red', linestyle=':', alpha=0.5,
                 label=f'u_c={u_c:.3f}')
    ax.set_xlabel('time (h)')
    ax.set_ylabel('u_target')
    ax.set_title('Phase 2 schedules: SMC² closed-loop vs oracle vs default')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    n_show = 20
    for i in range(n_show):
        ax.plot(t_phase2, trajs_x_post[i], alpha=0.4, lw=0.7,
                  color='steelblue')
    ax.plot(t_phase2, trajs_x_post[:n_show].mean(axis=0), '-',
              color='steelblue', lw=2, label='mean x(t)')
    ax.axhline(-1.0, color='red', linestyle=':', alpha=0.4)
    ax.axhline(+1.0, color='green', linestyle=':', alpha=0.4)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('x (health)')
    ax.set_title(f'Phase 2 closed-loop: SMC² (transition {rate_post:.0%})')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_phase2, trajs_x_post.mean(axis=0), '-', color='steelblue',
              lw=2, label=f'SMC² posterior ({rate_post:.0%})')
    ax.plot(t_phase2, trajs_x_oracle.mean(axis=0), '--', color='purple',
              lw=1.5, label=f'oracle truth ({rate_oracle:.0%})', alpha=0.85)
    ax.plot(t_phase2, trajs_x_default.mean(axis=0), '-', color='darkred',
              lw=1.5, label=f'default ({rate_default:.0%})')
    ax.axhline(-1.0, color='red', linestyle=':', alpha=0.4)
    ax.axhline(+1.0, color='green', linestyle=':', alpha=0.4)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('mean x(t)')
    ax.set_title('Phase 2: mean x(t) under each schedule')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stage B3 — closed-loop. Costs: SMC² {cost_post:.0f}, '
                  f'oracle {cost_oracle:.0f}, default {cost_default:.0f}',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
