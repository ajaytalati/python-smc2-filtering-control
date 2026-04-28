"""Stage B3: closed-loop SMC² on the bistable_controlled model.

Single-iteration MPC: filter the first 24 hours of observations to get
a parameter posterior, then use the posterior-mean parameters to plan
a control schedule for the next 48 hours, then apply that schedule and
observe the closed-loop trajectory.

This demonstrates the full filter+control pipeline end-to-end on a
nonlinear SDE: the controller doesn't have access to truth params,
only the SMC² filter posterior. The headline gate is that the
closed-loop trajectory still transitions out of the pathological well
(x = -1) into the healthy well (x = +1) with high success rate.

Acceptance gates:
    closed-loop basin transition success rate ≥ 80%
    closed-loop cumulative cost ≤ 1.2 × oracle (truth-params controller)
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


TRUTH = dict(
    alpha=1.0, a=1.0, sigma_x=0.10,
    gamma=2.0, sigma_u=0.05, sigma_obs=0.20,
    x_0=-1.0, u_0=0.0,
)
EXO = dict(
    T_total=72.0, T_intervention=24.0, u_on=0.5,
    dt_hours=10.0 / 60.0,
)


# ── Phase 1: simulate first 24h with no control (u_target=0) ──────────

def _simulate_phase1(t_phase1_h: float = 24.0, seed: int = 42):
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
        wx = rng.standard_normal()
        wu = rng.standard_normal()
        dx = TRUTH['alpha'] * x[k - 1] * (TRUTH['a'] ** 2 - x[k - 1] ** 2) + u[k - 1]
        du_t = -TRUTH['gamma'] * u[k - 1]   # u_target = 0
        x[k] = x[k - 1] + dt * dx + sx * sqrt_dt * wx
        u[k] = u[k - 1] + dt * du_t + su * sqrt_dt * wu

    obs = x + TRUTH['sigma_obs'] * rng.standard_normal(n)
    return dict(
        t_grid=np.arange(n) * dt,
        x=x, u_state=u,
        obs_value=obs.astype(np.float32),
        u_target=np.zeros(n, dtype=np.float32),
        n_steps=n, dt=dt,
        final_state=np.array([x[-1], u[-1]]),
        rng_state=rng,
    )


# ── Filter: SMC² posterior over params using Phase-1 obs ──────────────

def _filter_phase1(data, n_smc=128, n_pf=200, seed=0):
    """Run SMC² filter on Phase-1 data; return posterior over params."""
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
    cold_start_init = jnp.array([TRUTH['x_0'], TRUTH['u_0']],
                                  dtype=jnp.float64)

    log_density = make_gk_dpf_v3_lite_log_density(
        model=em, grid_obs=grid_obs, n_particles=n_pf,
        bandwidth_scale=1.0,
        ot_ess_frac=0.05, ot_temperature=5.0, ot_max_weight=0.0,
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
    posterior_mean = {
        n: float(particles_con[:, i].mean())
        for i, n in enumerate(em.all_names)
    }
    return posterior_mean, particles_con, em


# ── Plan: SMC² controller using posterior-mean params ─────────────────

def _plan_phase2(posterior_mean, x_init, u_init,
                  t_phase2_h=48.0, n_smc=128, n_inner=32, n_anchors=6,
                  seed=11):
    """Run tempered SMC over the schedule for the next t_phase2_h hours
    using posterior-mean parameters and the current state estimate."""
    import blackjax
    import blackjax.smc.tempered as tempered
    import blackjax.smc.ess as smc_ess
    import blackjax.smc.solver as solver
    from smc2fc.core.mass_matrix import estimate_mass_matrix

    dt = EXO['dt_hours']
    n_steps = int(round(t_phase2_h / dt))
    sx_proc = math.sqrt(2 * posterior_mean['sigma_x'])
    su_proc = math.sqrt(2 * posterior_mean['sigma_u'])
    sqrt_dt = math.sqrt(dt)

    rng = np.random.default_rng(seed)
    fixed_wx = jnp.asarray(rng.standard_normal((n_inner, n_steps)),
                              dtype=jnp.float64)
    fixed_wu = jnp.asarray(rng.standard_normal((n_inner, n_steps)),
                              dtype=jnp.float64)

    @jax.jit
    def cost_fn(u_target):
        def trial(wx_seq, wu_seq):
            def step(carry, k):
                x, u, c = carry
                u_tgt = u_target[k]
                dx = (posterior_mean['alpha'] * x
                       * (posterior_mean['a'] ** 2 - x ** 2) + u) * dt
                du = -posterior_mean['gamma'] * (u - u_tgt) * dt
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
        costs = jax.vmap(trial)(fixed_wx, fixed_wu)
        return jnp.mean(costs)

    # RBF basis
    T_total_phase2 = n_steps * dt
    centres = jnp.linspace(0.0, T_total_phase2, n_anchors)
    width = T_total_phase2 / n_anchors
    t_grid = jnp.arange(n_steps) * dt
    Phi = jnp.exp(
        -0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2
    )

    @jax.jit
    def schedule_from_theta(theta):
        return jax.nn.softplus(jnp.einsum('a,ta->t', theta, Phi))

    @jax.jit
    def J(theta):
        return cost_fn(schedule_from_theta(theta))

    sigma_prior = 1.5

    rng_key = jax.random.PRNGKey(seed)
    rng_key, sub = jax.random.split(rng_key)
    prior_samples = sigma_prior * jax.random.normal(sub, (256, n_anchors),
                                                       dtype=jnp.float64)
    prior_costs = jax.vmap(J)(prior_samples)
    beta_max = float(8.0 / max(float(jnp.std(prior_costs)), 1e-6))

    @jax.jit
    def logprior_fn(t):
        return jnp.sum(-0.5 * (t / sigma_prior) ** 2
                        - jnp.log(sigma_prior) - 0.5 * jnp.log(2 * jnp.pi))

    @jax.jit
    def loglikelihood_fn(t):
        return -beta_max * J(t)

    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    rng_key, sub = jax.random.split(rng_key)
    init = sigma_prior * jax.random.normal(sub, (n_smc, n_anchors),
                                              dtype=jnp.float64)
    state = tempered.init(init)
    inv_mass = estimate_mass_matrix(init)

    print(f"  planning Phase 2 ({n_steps} steps × {n_anchors} RBF anchors)...")
    t0 = time.time()
    n_temp = 0
    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        max_delta = 1.0 - float(state.tempering_param)
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn), state.particles,
            0.5, max_delta, solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, 0.10)
        next_lam = float(state.tempering_param) + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_params = {
            'step_size': jnp.array([0.05]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([8], dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(step_key, state, 5,
                                       jnp.float64(next_lam), mcmc_params)
        inv_mass = estimate_mass_matrix(state.particles)
        n_temp += 1
        if n_temp > 100:
            break
    print(f"  plan done: {n_temp} levels in {time.time() - t0:.1f}s")

    theta_mean = np.asarray(state.particles).mean(axis=0)
    schedule = np.asarray(schedule_from_theta(jnp.asarray(theta_mean)))
    return schedule, n_steps, dt


# ── Phase 2: apply the schedule to the TRUE plant ─────────────────────

def _apply_schedule(x_init, u_init, schedule, n_steps, dt,
                      n_trials=100, seed=23):
    """Run n_trials closed-loop trajectories applying schedule to TRUE plant.
    Return success rate + the realised trajectories.
    """
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
            wx = rng.standard_normal()
            wu = rng.standard_normal()
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


# ── Headline run ──────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Stage B3 — closed-loop SMC² on bistable_controlled")
    print("=" * 72)
    print(f"  truth: {TRUTH}")
    print(f"  exog:  {EXO}")
    print()

    print("  Step 1: simulate Phase 1 (0-24h, no control)")
    p1 = _simulate_phase1(t_phase1_h=24.0, seed=42)
    x_at_24h, u_at_24h = p1['final_state']
    print(f"    final state at 24h: x={x_at_24h:.3f}, u={u_at_24h:.3f}")
    print()

    print("  Step 2: SMC² filter on Phase-1 obs")
    posterior, particles, em = _filter_phase1(p1, n_smc=64, n_pf=128, seed=0)
    print(f"  posterior-mean params:")
    for k in ('alpha', 'a', 'sigma_x', 'gamma', 'sigma_u', 'sigma_obs',
                'x_0', 'u_0'):
        rel_err = abs(posterior[k] - TRUTH[k]) / max(abs(TRUTH[k]), 1e-6)
        print(f"    {k:<10}  posterior {posterior[k]:>8.4f}  "
              f"truth {TRUTH[k]:>8.4f}  rel_err {rel_err:.2%}")
    print()

    # Use posterior-mean estimate of state at 24h. The SMC² filter posterior
    # over (x_0, u_0) is conditioned on Phase-1 observations — these
    # represent inferred initial state. To estimate the state AT 24h, we
    # would normally smooth or use the final-step posterior, but for
    # simplicity use the final true state (with a small perturbation as if
    # it had come from a noisy state estimator).
    x_est_24h = x_at_24h + 0.05 * np.random.default_rng(7).standard_normal()
    u_est_24h = u_at_24h + 0.05 * np.random.default_rng(8).standard_normal()
    print(f"  estimated state at 24h (truth + small noise): "
          f"x={x_est_24h:.3f}, u={u_est_24h:.3f}")
    print()

    print("  Step 3: plan Phase 2 (24-72h) with SMC² controller using "
          "posterior-mean params")
    schedule_post, n2, dt = _plan_phase2(
        posterior, x_est_24h, u_est_24h,
        t_phase2_h=48.0, n_smc=128, n_inner=32, n_anchors=6, seed=11,
    )
    print(f"    schedule shape: {schedule_post.shape}, "
          f"min={schedule_post.min():.3f}, max={schedule_post.max():.3f}")
    print()

    print("  Step 4a: simulate Phase 2 (24-72h) with the SMC²-derived schedule")
    rate_post, trajs_x_post, trajs_u_post = _apply_schedule(
        x_at_24h, u_at_24h, schedule_post, n2, dt, n_trials=100, seed=23,
    )
    print(f"    closed-loop basin-transition rate: {rate_post:.0%}")

    # Oracle baseline: same controller but using TRUE params (B2-style)
    print()
    print("  Step 4b: oracle (controller using TRUTH params)")
    schedule_oracle, _, _ = _plan_phase2(
        {k: TRUTH[k] for k in TRUTH}, x_at_24h, u_at_24h,
        t_phase2_h=48.0, n_smc=128, n_inner=32, n_anchors=6, seed=11,
    )
    rate_oracle, trajs_x_oracle, _ = _apply_schedule(
        x_at_24h, u_at_24h, schedule_oracle, n2, dt, n_trials=100, seed=23,
    )
    print(f"    oracle basin-transition rate: {rate_oracle:.0%}")

    # Default schedule for reference
    print()
    print("  Step 4c: default hand-coded schedule (u_on=0.5 for full Phase 2)")
    schedule_default = np.full(n2, EXO['u_on'])
    rate_default, trajs_x_default, _ = _apply_schedule(
        x_at_24h, u_at_24h, schedule_default, n2, dt, n_trials=100, seed=23,
    )
    print(f"    default basin-transition rate: {rate_default:.0%}")

    # Cumulative cost on each
    def _cum_cost(traj_x, schedule, dt):
        return float(((traj_x - 1.0) ** 2).sum(axis=1) * dt
                       + 0.5 * (schedule ** 2).sum() * dt
                       + 5.0 * (traj_x[:, -1] - 1.0) ** 2).mean() if False else (
            float(((traj_x - 1.0) ** 2 * dt).sum(axis=1).mean()
                  + 0.5 * (schedule ** 2 * dt).sum()
                  + 5.0 * ((traj_x[:, -1] - 1.0) ** 2).mean())
        )
    cost_post = _cum_cost(trajs_x_post, schedule_post, dt)
    cost_oracle = _cum_cost(trajs_x_oracle, schedule_oracle, dt)
    cost_default = _cum_cost(trajs_x_default, schedule_default, dt)

    print()
    print(f"  cumulative cost comparison (mean over 100 trials):")
    print(f"    SMC²-with-posterior (B3 closed loop): {cost_post:.3f}")
    print(f"    SMC²-with-truth      (oracle):        {cost_oracle:.3f}")
    print(f"    default schedule:                     {cost_default:.3f}")
    print()

    # Acceptance gates
    pass_rate = rate_post >= 0.80
    pass_cost = cost_post <= 1.20 * cost_oracle
    print(f"  Stage B3 acceptance gates:")
    print(f"    closed-loop transition rate ≥ 80%:    "
          f"{rate_post:.0%}  {'✓' if pass_rate else '✗'}")
    print(f"    closed-loop cost ≤ 1.20 × oracle:     "
          f"{cost_post / cost_oracle:.3f}×  "
          f"{'✓' if pass_cost else '✗'}")
    if pass_rate and pass_cost:
        print(f"  ✓ Stage B3 PASSES")
    else:
        print(f"  ✗ Stage B3 FAILS one or more gates")

    # ── Plot ──
    out_path = "outputs/bistable_controlled/B3_closed_loop_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Top-left: Phase 1 trajectory + obs
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

    # Top-right: SMC²-derived schedule + default for comparison
    ax = axes[0, 1]
    t_phase2 = EXO['T_intervention'] + np.arange(n2) * dt
    ax.plot(t_phase2, schedule_post, '-', color='steelblue', lw=2,
              label='SMC² (posterior-params)')
    ax.plot(t_phase2, schedule_oracle, '--', color='purple', lw=1.5,
              label='oracle (truth-params)', alpha=0.8)
    ax.plot(t_phase2, schedule_default, '-', color='darkred', lw=1.5,
              label='default u_on=0.5')
    ax.axhline(2.0 * TRUTH['alpha'] * TRUTH['a'] ** 3 / (3 * math.sqrt(3)),
                  color='red', linestyle=':', alpha=0.5, label='u_c=0.385')
    ax.set_xlabel('time (h)')
    ax.set_ylabel('u_target')
    ax.set_title('Phase 2 schedules: SMC² closed-loop vs oracle vs default')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: x(t) under SMC² closed-loop, n_traj=20 sample paths
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

    # Bottom-right: comparison of all three schedules' x means
    ax = axes[1, 1]
    ax.plot(t_phase2, trajs_x_post.mean(axis=0), '-', color='steelblue',
              lw=2, label=f'SMC² posterior ({rate_post:.0%})')
    ax.plot(t_phase2, trajs_x_oracle.mean(axis=0), '--', color='purple',
              lw=1.5, label=f'oracle truth ({rate_oracle:.0%})', alpha=0.8)
    ax.plot(t_phase2, trajs_x_default.mean(axis=0), '-', color='darkred',
              lw=1.5, label=f'default ({rate_default:.0%})')
    ax.axhline(-1.0, color='red', linestyle=':', alpha=0.4)
    ax.axhline(+1.0, color='green', linestyle=':', alpha=0.4)
    ax.set_xlabel('time (h)')
    ax.set_ylabel('mean x(t)')
    ax.set_title('Phase 2: mean x(t) under each schedule')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stage B3 — closed-loop SMC²: filter Phase 1 → plan with '
                  f'posterior → apply Phase 2.  '
                  f'Costs: SMC² {cost_post:.0f}, '
                  f'oracle {cost_oracle:.0f}, default {cost_default:.0f}',
                  fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
