"""Stage A2: SMC²-as-controller on the scalar OU LQG model.

The headline duality demonstration: the same outer tempered-SMC engine
that does filtering is applied as an optimiser over the control
schedule, with loglikelihood = -beta * J(u) where J is the quadratic
cost functional. As beta is annealed from 0 to beta_max, the SMC
particle cloud concentrates near argmin J — for an open-loop schedule
parameterisation, that's the **best deterministic schedule**, which on
a linear-Gaussian system with x_0 ~ N(0, 1) is u(t) = 0 (the
expectation of the LQR feedback under the prior). Closed-loop LQR
feedback (u = -K · x_t) requires Stage A3, where the schedule is
parameterised as a function of the state estimate.

This script is hard-coded against scalar OU on purpose (per the
bottom-up plan): no abstractions, no `smc2fc/control/` package. After
Stage B2 has bistable control hard-coded the same way, we'll have two
concrete examples and can extract whatever is genuinely common into
the framework package.

What's hard-coded here:
  - Schedule parameterisation: u = (u_0, ..., u_{T-1}) ∈ ℝ^T
    (raw 5-min pulses, no RBF basis)
  - Cost functional: J(u) = E[sum_k (q*x_k^2 + r*u_k^2) + s*x_T^2]
    evaluated by Monte-Carlo forward simulation of the SDE
  - Prior over schedules: u_k ~ N(0, sigma_prior^2) i.i.d.
  - Tempering target: log target(u) = log_prior(u) + (-beta * J(u))

Acceptance gate (for an open-loop schedule problem):
    SMC²-mean-schedule cost / open-loop u=0 cost ∈ [0.95, 1.10]

Recovering the tighter LQR-vs-LQG gap (cost = 5.41 vs 9.29) requires
a state-feedback schedule parameterisation — that's Stage A3.

Run:
    PYTHONPATH=. python tools/bench_smc_control_ou.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import blackjax
import blackjax.smc.tempered as tempered
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as solver

from models.scalar_ou_lqg.bench_lqr import (
    lqr_riccati, lqr_optimal_cost,
    lqg_optimal_cost_monte_carlo,
    open_loop_zero_control_cost_monte_carlo,
)


# ── Truth parameters ──────────────────────────────────────────────────

TRUTH = dict(
    a=1.0, b=1.0, sigma_w=0.3, sigma_v=0.2,
    q=1.0, r=0.1, s=1.0,
    dt=0.05, T=20,
    x0_mean=0.0, x0_var=1.0,
)


# ── Cost evaluator ────────────────────────────────────────────────────

def _build_cost_fn(*, n_inner: int, seed: int):
    """Build a JIT-compiled cost evaluator J(u) for a fixed Monte Carlo
    grid of forward-simulation noise. Reusing the same noise across all
    SMC particles is the standard variance-reduction trick (CRN —
    common random numbers).
    """
    A = 1.0 - TRUTH['a'] * TRUTH['dt']
    B = TRUTH['b'] * TRUTH['dt']
    sw = TRUTH['sigma_w'] * jnp.sqrt(TRUTH['dt'])
    T_steps = TRUTH['T']

    rng = np.random.default_rng(seed)
    # Fixed Wiener increments and initial states across cost evaluations.
    fixed_w = jnp.asarray(
        rng.standard_normal((n_inner, T_steps)), dtype=jnp.float64)
    fixed_x0 = jnp.asarray(
        TRUTH['x0_mean'] + jnp.sqrt(TRUTH['x0_var']) *
        rng.standard_normal((n_inner,)), dtype=jnp.float64)

    @jax.jit
    def J(u):
        """Mean-cost evaluator J(u) under fixed noise grid."""
        def trial(x0, w_seq):
            def step(carry, k):
                x, cost = carry
                u_k = u[k]
                cost_k = TRUTH['q'] * x ** 2 + TRUTH['r'] * u_k ** 2
                x_next = A * x + B * u_k + sw * w_seq[k]
                return (x_next, cost + cost_k), None
            (x_T, total_stage), _ = jax.lax.scan(
                step, (x0, jnp.float64(0.0)),
                jnp.arange(T_steps))
            terminal_cost = TRUTH['s'] * x_T ** 2
            return total_stage + terminal_cost
        costs = jax.vmap(trial)(fixed_x0, fixed_w)
        return jnp.mean(costs)

    return J


# ── Outer tempered SMC over the control schedule ──────────────────────

def run_tempered_smc_over_schedules(
    *,
    n_smc: int = 256,
    n_inner: int = 64,
    sigma_prior: float = 2.0,
    beta_max: float | None = None,
    target_ess_frac: float = 0.5,
    max_lambda_inc: float = 0.10,
    num_mcmc_steps: int = 5,
    hmc_step_size: float = 0.05,
    hmc_num_leapfrog: int = 8,
    seed: int = 42,
):
    """Run tempered SMC over u ∈ ℝ^T with target = prior * exp(-beta * J(u)).

    Auto-calibrates beta_max so that beta_max * std(J under prior) ≈ 8 nats
    (yields ~16 effective tempering levels for the standard ESS schedule).

    Returns dict with keys:
        particles_final  (n_smc, T)  — final schedule samples
        n_temp_levels    int          — number of tempering steps
        elapsed_s        float
        beta_max         float
        prior_cost_std   float
    """
    T_steps = TRUTH['T']
    dim = T_steps
    cost_fn = _build_cost_fn(n_inner=n_inner, seed=seed)

    # 1. Auto-calibrate beta_max from prior-cloud cost spread.
    rng_key = jax.random.PRNGKey(seed)
    rng_key, sub = jax.random.split(rng_key)
    prior_samples = sigma_prior * jax.random.normal(sub, (1024, dim),
                                                       dtype=jnp.float64)
    prior_costs = jax.vmap(cost_fn)(prior_samples)
    prior_cost_std = float(jnp.std(prior_costs))
    if beta_max is None:
        beta_max = float(8.0 / max(prior_cost_std, 1e-6))
    print(f"  prior cost mean = {float(jnp.mean(prior_costs)):.3f}, "
          f"std = {prior_cost_std:.3f}")
    print(f"  beta_max (auto) = {beta_max:.4f}")

    @jax.jit
    def logprior_fn(u):
        return jnp.sum(-0.5 * (u / sigma_prior) ** 2 - jnp.log(sigma_prior)
                        - 0.5 * jnp.log(2 * jnp.pi))

    # The "loglikelihood" for the tempered SMC is the increment from
    # prior to the full target. Tempered SMC interpolates as
    # log target_lambda(u) = logprior(u) + lambda * loglikelihood(u).
    # We want at lambda=1: log target = logprior - beta_max * J(u).
    @jax.jit
    def loglikelihood_fn(u):
        return -beta_max * cost_fn(u)

    # 2. Build outer SMC kernel — pattern from smc2fc/core/tempered_smc.py.
    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    # 3. Initial particles from the prior.
    rng_key, sub = jax.random.split(rng_key)
    init_particles = sigma_prior * jax.random.normal(sub, (n_smc, dim),
                                                       dtype=jnp.float64)
    state = tempered.init(init_particles)

    # 4. Adaptive ESS-based tempering loop (pattern from tempered_smc.py).
    from smc2fc.core.mass_matrix import estimate_mass_matrix
    inv_mass = estimate_mass_matrix(init_particles)
    print(f"  starting tempered SMC over T={dim}-D schedule, "
          f"N_SMC={n_smc}, N_inner={n_inner}")
    t0 = time.time()
    n_temp = 0
    prev_lam = 0.0
    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            target_ess_frac,
            max_delta,
            solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, max_lambda_inc)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_params = {
            'step_size': jnp.array([hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([hmc_num_leapfrog], dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(step_key, state, num_mcmc_steps,
                                       jnp.float64(next_lam), mcmc_params)
        # Re-estimate mass matrix from updated particles
        inv_mass = estimate_mass_matrix(state.particles)
        n_temp += 1
        lam = float(state.tempering_param)
        try:
            acc = float(jnp.mean(info.update_info.acceptance_rate))
        except Exception:
            acc = float('nan')
        if n_temp % 5 == 0 or lam >= 1.0:
            mean_cost = float(jnp.mean(jax.vmap(cost_fn)(state.particles)))
            print(f"    step {n_temp:3d}: λ={lam:.3f}  acc={acc:.3f}  "
                  f"mean cost = {mean_cost:.3f}")
        prev_lam = lam
        if n_temp > 200:
            print("  hit max tempering steps; breaking")
            break

    elapsed = time.time() - t0

    return {
        'particles_final': np.asarray(state.particles),
        'n_temp_levels':   n_temp,
        'elapsed_s':       elapsed,
        'beta_max':        beta_max,
        'prior_cost_std':  prior_cost_std,
    }


# ── Headline run ──────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Stage A2 — SMC²-as-controller on scalar OU LQG")
    print("=" * 72)

    # Analytical references
    riccati = lqr_riccati(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    lqr_perfect = lqr_optimal_cost(
        riccati=riccati,
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    lqg = lqg_optimal_cost_monte_carlo(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], sigma_v=TRUTH['sigma_v'],
        dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        n_trials=5000, seed=0,
    )
    open_loop = open_loop_zero_control_cost_monte_carlo(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'],
        dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        n_trials=5000, seed=0,
    )

    print(f"  analytical LQR (perfect state):   {lqr_perfect:.4f}")
    print(f"  MC LQG (Kalman + LQR):            {lqg['mean_cost']:.4f} ± {lqg['stderr']:.4f}")
    print(f"  MC open-loop (u=0):               {open_loop['mean_cost']:.4f} ± {open_loop['stderr']:.4f}")
    print()

    # SMC² over schedules
    res = run_tempered_smc_over_schedules(
        n_smc=128, n_inner=64,
        sigma_prior=2.0,
        target_ess_frac=0.5,
        max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        seed=42,
    )
    particles = res['particles_final']      # (n_smc, T)
    schedule_mean = particles.mean(axis=0)  # (T,)
    schedule_std = particles.std(axis=0)    # (T,)

    print()
    print(f"  tempered SMC done: {res['n_temp_levels']} levels in "
          f"{res['elapsed_s']:.1f}s")
    print(f"  posterior schedule mean: {schedule_mean}")
    print()

    # Evaluate the SMC posterior-mean schedule's true cost (open-loop apply)
    # and compare to the LQR perfect-state cost.
    cost_fn = _build_cost_fn(n_inner=2000, seed=99)
    smc_mean_cost = float(cost_fn(jnp.asarray(schedule_mean)))
    smc_per_particle_cost = jax.vmap(cost_fn)(jnp.asarray(particles))
    smc_posterior_mean_cost = float(jnp.mean(smc_per_particle_cost))

    print(f"  SMC²-mean-schedule cost (perfect state, MC=2000): "
          f"{smc_mean_cost:.4f}")
    print(f"  SMC² posterior mean cost (per-particle, then mean): "
          f"{smc_posterior_mean_cost:.4f}")
    print()

    # Stage A2 gate: SMC²-mean cost vs open-loop u=0 cost.
    # Open-loop u=0 IS the optimal deterministic schedule under x_0 ~ N(0,1)
    # (since E[x_t] = 0 and the LQR feedback at zero state is 0). SMC²
    # over an open-loop parameterisation should recover this.
    open_loop_ratio = smc_mean_cost / open_loop['mean_cost']
    lqr_ratio = smc_mean_cost / lqr_perfect
    print(f"  ratio  SMC² / open-loop (u=0):       {open_loop_ratio:.4f}")
    print(f"  ratio  SMC² / LQR perfect feedback:  {lqr_ratio:.4f}  "
          f"(needs state-feedback param. — Stage A3)")
    if 0.95 <= open_loop_ratio <= 1.10:
        print(f"  ✓ Stage A2 acceptance gate "
              f"(SMC² / open-loop ∈ [0.95, 1.10]) PASSES")
    else:
        print(f"  ✗ Stage A2 acceptance gate "
              f"(SMC² / open-loop ∈ [0.95, 1.10]) FAILS")

    # Plot
    out_path = "outputs/scalar_ou_lqg/A2_control_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: SMC posterior schedule vs LQR feedback at deterministic state
    # The LQR open-loop "schedule" from x_0 = 0 (no noise) is u_k* = 0
    # for all k since x stays at 0. So we can't compare schedules
    # directly when starting at x_0=0. Instead, plot SMC schedule + a
    # "single-realisation LQR-applied" schedule for visualization.
    rng = np.random.default_rng(7)
    A = 1.0 - TRUTH['a'] * TRUTH['dt']
    B = TRUTH['b'] * TRUTH['dt']
    sw = TRUTH['sigma_w'] * np.sqrt(TRUTH['dt'])
    x_lqr = np.zeros(TRUTH['T'])
    u_lqr = np.zeros(TRUTH['T'])
    x_lqr[0] = TRUTH['x0_mean'] + np.sqrt(TRUTH['x0_var']) * rng.standard_normal()
    for k in range(TRUTH['T']):
        u_lqr[k] = -riccati.gains[k] * x_lqr[k]
        x_next = A * x_lqr[k] + B * u_lqr[k] + (
            sw * rng.standard_normal() if k < TRUTH['T'] - 1 else 0.0)
        if k < TRUTH['T'] - 1:
            x_lqr[k + 1] = x_next

    t = np.arange(TRUTH['T']) * TRUTH['dt']
    axes[0].fill_between(t,
                            schedule_mean - schedule_std,
                            schedule_mean + schedule_std,
                            alpha=0.3, color='steelblue', label='SMC ±1σ')
    axes[0].plot(t, schedule_mean, 'o-', color='steelblue',
                   label='SMC mean schedule', alpha=0.85)
    axes[0].plot(t, u_lqr, 's--', color='darkred',
                   label='LQR u*(t) on one realisation', alpha=0.7)
    axes[0].axhline(0, color='black', alpha=0.3, linewidth=0.5)
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('u (control)')
    axes[0].set_title('Control schedule: SMC² posterior vs LQR sample')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: cost histogram across the SMC² particle cloud
    axes[1].hist(np.asarray(smc_per_particle_cost), bins=30,
                   color='steelblue', alpha=0.7, label='SMC² per-particle cost')
    axes[1].axvline(lqr_perfect, color='green', linestyle='--', linewidth=2,
                       label=f'LQR perfect state = {lqr_perfect:.2f}')
    axes[1].axvline(lqg['mean_cost'], color='orange', linestyle='--', linewidth=2,
                       label=f'MC LQG = {lqg["mean_cost"]:.2f}')
    axes[1].axvline(open_loop['mean_cost'], color='red', linestyle='--', linewidth=2,
                       label=f'open-loop = {open_loop["mean_cost"]:.2f}')
    axes[1].axvline(smc_mean_cost, color='steelblue', linestyle=':', linewidth=2,
                       label=f'SMC² mean schedule = {smc_mean_cost:.2f}')
    axes[1].set_xlabel('cost')
    axes[1].set_ylabel('density')
    axes[1].set_title('SMC² cost distribution vs analytical references')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)

    return {
        'lqr_perfect': lqr_perfect,
        'lqg_mc': lqg['mean_cost'],
        'open_loop_mc': open_loop['mean_cost'],
        'smc_mean_schedule_cost': smc_mean_cost,
        'open_loop_ratio': open_loop_ratio,
        'lqr_ratio': lqr_ratio,
        'pass': 0.95 <= open_loop_ratio <= 1.10,
        'schedule_mean': schedule_mean,
        'schedule_std': schedule_std,
        'n_temp_levels': res['n_temp_levels'],
        'elapsed_s': res['elapsed_s'],
    }


if __name__ == '__main__':
    main()
