"""Stage B2: SMC²-as-controller on the bistable model.

The headline pedagogical demonstration: tempered SMC over a control
schedule u_target(t) drives the bistable system out of the
pathological well (x = -1) into the healthy well (x = +1). The
discovered schedule should produce a supercritical tilt
(u_target > u_c ≈ 0.385) for at least part of the trial — without
having been told that explicitly.

Hard-coded against bistable per the bottom-up plan; will be refactored
into smc2fc/control/ once Stage A2/A3 + Stage B2 have settled the
right abstractions.

The control schedule is parameterised as 6 Gaussian RBF anchors over
the 72-hour horizon, with non-negative coefficients via softplus.
This is a 6-D search space — small enough that tempered SMC converges
in a few minutes on CPU.

Cost functional:
    J(theta) = E_traj[ ∫ (x_t - x_target)^2 dt + lambda · ∫ u_target_t^2 dt ]

with x_target = +1 (healthy well), lambda = 0.5 (control penalty).

Acceptance gates:
    SMC²-derived schedule cost ≤ default-schedule cost
    SMC²-derived basin transition success rate ≥ 80%

Run:
    PYTHONPATH=. python tools/bench_smc_control_bistable.py
"""

from __future__ import annotations

import math
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

from smc2fc.core.mass_matrix import estimate_mass_matrix


# ── Truth parameters (Set A from public-dev) ──────────────────────────

PARAMS_A = dict(
    alpha=1.0, a=1.0, sigma_x=0.10,
    gamma=2.0, sigma_u=0.05, sigma_obs=0.20,
)

INIT_A = dict(x_0=-1.0, u_0=0.0)

# Schedule for a 3-day (72 h) window. Original default: 24h pre-intervention
# at u=0, then 48h supercritical at u=0.5. We compare SMC²-derived
# against this default.
EXOGENOUS_A = dict(
    T_intervention=24.0,
    T_total=72.0,
    u_on=0.5,    # supercritical (>u_c=0.385)
    dt_hours=10.0 / 60.0,    # 10 min, matches public-dev v1.1 default
)

# Critical tilt: u_c = 2 * alpha * a^3 / (3 sqrt(3))
U_CRIT = 2.0 * PARAMS_A['alpha'] * PARAMS_A['a'] ** 3 / (3.0 * math.sqrt(3.0))


# ── Forward simulator (JAX, for use inside cost evaluator) ────────────

def _make_simulator(*, n_steps, dt, seed_base):
    """Build a JIT-compiled SDE simulator that maps:
        u_target(t) [array of shape (n_steps,)]
        ⟶  expected cost under fixed CRN noise.
    """
    alpha = PARAMS_A['alpha']
    a = PARAMS_A['a']
    gamma = PARAMS_A['gamma']
    sigma_x = jnp.sqrt(2 * PARAMS_A['sigma_x'])
    sigma_u = jnp.sqrt(2 * PARAMS_A['sigma_u'])
    sqrt_dt = jnp.sqrt(dt)

    def make_cost_fn(n_inner: int, x_target: float = 1.0,
                       lam: float = 0.5, x_barrier: float = -0.5):
        """Construct the cost evaluator for a given Monte Carlo size."""
        rng = np.random.default_rng(seed_base)
        # Fixed Wiener increments and initial states across cost evaluations
        fixed_wx = jnp.asarray(
            rng.standard_normal((n_inner, n_steps)), dtype=jnp.float64)
        fixed_wu = jnp.asarray(
            rng.standard_normal((n_inner, n_steps)), dtype=jnp.float64)
        fixed_x0 = jnp.full((n_inner,), INIT_A['x_0'], dtype=jnp.float64)
        fixed_u0 = jnp.full((n_inner,), INIT_A['u_0'], dtype=jnp.float64)

        @jax.jit
        def cost_fn(u_target):
            """Mean cost over the fixed CRN ensemble."""
            def trial(x0, u0, wx_seq, wu_seq):
                def step(carry, k):
                    x, u, cost = carry
                    u_tgt_k = u_target[k]
                    dx = (alpha * x * (a ** 2 - x ** 2) + u) * dt
                    du = -gamma * (u - u_tgt_k) * dt
                    x_next = x + dx + sigma_x * sqrt_dt * wx_seq[k]
                    u_next = u + du + sigma_u * sqrt_dt * wu_seq[k]
                    # Stage cost: target-tracking + control-effort + barrier
                    stage = ((x - x_target) ** 2
                             + lam * u_tgt_k ** 2
                             + 50.0 * jax.nn.softplus(-(x - x_barrier) - 1.0))
                    return (x_next, u_next, cost + stage * dt), None
                (x_T, u_T, total), _ = jax.lax.scan(
                    step, (x0, u0, jnp.float64(0.0)), jnp.arange(n_steps)
                )
                # Terminal cost: how far from target are we
                terminal = 5.0 * (x_T - x_target) ** 2
                return total + terminal
            costs = jax.vmap(trial)(fixed_x0, fixed_u0, fixed_wx, fixed_wu)
            return jnp.mean(costs)

        @jax.jit
        def trajectory_sample(u_target, key):
            """Return a single SAMPLE trajectory (x, u) under u_target."""
            def step(carry, args):
                x, u = carry
                k, wx, wu = args
                u_tgt_k = u_target[k]
                dx = (alpha * x * (a ** 2 - x ** 2) + u) * dt
                du = -gamma * (u - u_tgt_k) * dt
                x_next = x + dx + sigma_x * sqrt_dt * wx
                u_next = u + du + sigma_u * sqrt_dt * wu
                return (x_next, u_next), (x_next, u_next)
            wx_seq = jax.random.normal(key, (n_steps,))
            key2, _ = jax.random.split(key)
            wu_seq = jax.random.normal(key2, (n_steps,))
            (_, _), traj = jax.lax.scan(
                step,
                (jnp.float64(INIT_A['x_0']), jnp.float64(INIT_A['u_0'])),
                (jnp.arange(n_steps), wx_seq, wu_seq),
            )
            return traj

        return cost_fn, trajectory_sample

    return make_cost_fn


# ── RBF basis for the schedule parameterisation ───────────────────────

def _rbf_design_matrix(n_steps: int, dt: float, n_anchors: int):
    """Return Phi of shape (n_steps, n_anchors), Gaussian RBF basis.

    Anchors evenly spaced over [0, T_total]; widths set to anchor spacing.
    """
    T_total = n_steps * dt
    centres = jnp.linspace(0.0, T_total, n_anchors)
    width = T_total / n_anchors
    t_grid = jnp.arange(n_steps) * dt
    Phi = jnp.exp(-0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2)
    return Phi


def _schedule_from_theta(theta: jnp.ndarray, Phi: jnp.ndarray) -> jnp.ndarray:
    """Build a non-negative schedule u_target(t) from RBF coefficients.

    softplus(theta @ Phi^T) ensures u_target ≥ 0; theta unconstrained.
    """
    raw = jnp.einsum('a,ta->t', theta, Phi)
    return jax.nn.softplus(raw)


# ── Tempered SMC outer loop over RBF coefficients ─────────────────────

def run_tempered_smc_over_schedule(
    *,
    n_smc: int = 128,
    n_inner: int = 32,
    n_anchors: int = 6,
    sigma_prior: float = 1.5,
    target_ess_frac: float = 0.5,
    max_lambda_inc: float = 0.10,
    num_mcmc_steps: int = 5,
    hmc_step_size: float = 0.05,
    hmc_num_leapfrog: int = 8,
    seed: int = 42,
):
    dt = EXOGENOUS_A['dt_hours']
    n_steps = int(round(EXOGENOUS_A['T_total'] / dt))
    Phi = _rbf_design_matrix(n_steps, dt, n_anchors)

    make_cost_fn = _make_simulator(n_steps=n_steps, dt=dt, seed_base=seed)
    raw_cost_fn, _ = make_cost_fn(n_inner=n_inner)

    @jax.jit
    def cost_fn(theta):
        u_target = _schedule_from_theta(theta, Phi)
        return raw_cost_fn(u_target)

    rng_key = jax.random.PRNGKey(seed)

    # Calibrate beta_max from prior cloud cost spread
    rng_key, sub = jax.random.split(rng_key)
    prior_samples = sigma_prior * jax.random.normal(
        sub, (256, n_anchors), dtype=jnp.float64)
    prior_costs = jax.vmap(cost_fn)(prior_samples)
    prior_cost_std = float(jnp.std(prior_costs))
    beta_max = float(8.0 / max(prior_cost_std, 1e-6))
    print(f"  prior cost mean = {float(jnp.mean(prior_costs)):.3f}, "
          f"std = {prior_cost_std:.3f}")
    print(f"  beta_max (auto) = {beta_max:.4f}")

    @jax.jit
    def logprior_fn(theta):
        return jnp.sum(
            -0.5 * (theta / sigma_prior) ** 2
            - jnp.log(sigma_prior) - 0.5 * jnp.log(2 * jnp.pi)
        )

    @jax.jit
    def loglikelihood_fn(theta):
        return -beta_max * cost_fn(theta)

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
    init_particles = sigma_prior * jax.random.normal(
        sub, (n_smc, n_anchors), dtype=jnp.float64)
    state = tempered.init(init_particles)
    inv_mass = estimate_mass_matrix(init_particles)

    print(f"  starting tempered SMC over θ ∈ ℝ^{n_anchors} (RBF coefs), "
          f"N_SMC={n_smc}, N_inner={n_inner}, n_steps={n_steps}")
    t0 = time.time()
    n_temp = 0
    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            target_ess_frac, max_delta, solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, max_lambda_inc)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_params = {
            'step_size': jnp.array([hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array(
                [hmc_num_leapfrog], dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, num_mcmc_steps,
            jnp.float64(next_lam), mcmc_params,
        )
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
        if n_temp > 200:
            print("  hit max tempering steps; breaking")
            break

    elapsed = time.time() - t0

    return {
        'particles_final': np.asarray(state.particles),
        'Phi':             np.asarray(Phi),
        'n_temp_levels':   n_temp,
        'elapsed_s':       elapsed,
        'beta_max':        beta_max,
        'cost_fn':         cost_fn,
        'raw_cost_fn':     raw_cost_fn,
        'n_steps':         n_steps,
        'dt':              dt,
    }


# ── Headline run ──────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Stage B2 — SMC²-as-controller on bistable_controlled")
    print("=" * 72)
    print(f"  truth params (Set A): {PARAMS_A}")
    print(f"  init state:           {INIT_A}")
    print(f"  exogenous schedule:   T_total={EXOGENOUS_A['T_total']} h, "
          f"u_on={EXOGENOUS_A['u_on']} (default supercritical)")
    print(f"  u_critical = 2 alpha a^3 / (3√3) = {U_CRIT:.4f}")
    print()

    res = run_tempered_smc_over_schedule(
        n_smc=128, n_inner=32, n_anchors=6,
        sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        seed=42,
    )
    particles = res['particles_final']         # (n_smc, n_anchors)
    Phi = res['Phi']                            # (n_steps, n_anchors)
    n_steps = res['n_steps']
    dt = res['dt']

    # Build SMC² posterior-mean schedule
    theta_mean = particles.mean(axis=0)
    theta_std = particles.std(axis=0)
    smc_schedule = jax.nn.softplus(
        jnp.einsum('a,ta->t', jnp.asarray(theta_mean), jnp.asarray(Phi))
    )
    smc_schedule = np.asarray(smc_schedule)

    # Default schedule (24h pre at u=0, then u_on=0.5)
    t_grid = np.arange(n_steps) * dt
    default_schedule = np.where(
        t_grid < EXOGENOUS_A['T_intervention'],
        0.0, EXOGENOUS_A['u_on'])

    # Evaluate cost on a fresh CRN grid (n_inner=500, fresh seed)
    print()
    cost_fn_eval, traj_sample_fn = _make_simulator(
        n_steps=n_steps, dt=dt, seed_base=99,
    )(n_inner=500)
    smc_cost = float(cost_fn_eval(jnp.asarray(smc_schedule)))
    default_cost = float(cost_fn_eval(jnp.asarray(default_schedule)))
    zero_cost = float(cost_fn_eval(jnp.zeros(n_steps)))

    print(f"  SMC²-mean-schedule cost:      {smc_cost:.4f}")
    print(f"  Default schedule cost:        {default_cost:.4f}")
    print(f"  Zero schedule (no control):   {zero_cost:.4f}")
    print()

    # Basin-transition success rate: in 100 fresh trajectories under each
    # schedule, fraction that reaches x > 0.5 by t = 60 h.
    rng = np.random.default_rng(123)

    def _succ(schedule):
        n_trials = 100
        wins = 0
        for trial in range(n_trials):
            wx = rng.standard_normal(n_steps)
            wu = rng.standard_normal(n_steps)
            x = INIT_A['x_0']
            u = INIT_A['u_0']
            success = False
            for k in range(n_steps):
                t = k * dt
                u_tgt = float(schedule[k])
                dx = (PARAMS_A['alpha'] * x
                       * (PARAMS_A['a'] ** 2 - x ** 2) + u) * dt
                du = -PARAMS_A['gamma'] * (u - u_tgt) * dt
                x = x + dx + math.sqrt(2 * PARAMS_A['sigma_x']
                                          * dt) * wx[k]
                u = u + du + math.sqrt(2 * PARAMS_A['sigma_u']
                                          * dt) * wu[k]
                if t >= 60.0 and x > 0.5:
                    success = True
                    break
            wins += int(success)
        return wins / n_trials

    smc_success = _succ(smc_schedule)
    default_success = _succ(default_schedule)
    zero_success = _succ(np.zeros(n_steps))

    print(f"  basin-transition success rates (100 trials each):")
    print(f"    SMC² schedule:      {smc_success:.0%}")
    print(f"    default schedule:   {default_success:.0%}")
    print(f"    zero schedule:      {zero_success:.0%}")
    print()

    # Acceptance gates
    pass_cost = smc_cost <= default_cost
    pass_success = smc_success >= 0.80
    print(f"  Stage B2 acceptance gates:")
    print(f"    SMC² cost ≤ default cost:                "
          f"{smc_cost:.2f} ≤ {default_cost:.2f}  "
          f"{'✓' if pass_cost else '✗'}")
    print(f"    SMC² basin-transition rate ≥ 80%:        "
          f"{smc_success:.0%}  "
          f"{'✓' if pass_success else '✗'}")
    if pass_cost and pass_success:
        print(f"  ✓ Stage B2 PASSES")
    else:
        print(f"  ✗ Stage B2 has gate misses (see above)")

    # ── Plot ──
    out_path = "outputs/bistable_controlled/B2_control_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Sample trajectories under each schedule
    rng_key = jax.random.PRNGKey(7)
    n_traj = 5

    # Top-left: SMC² schedule + sample trajectories
    axes[0, 0].plot(t_grid, smc_schedule, '-', color='steelblue', lw=2,
                       label='SMC² posterior-mean u_target(t)')
    axes[0, 0].axhline(U_CRIT, color='red', linestyle=':', alpha=0.7,
                          label=f'u_c = {U_CRIT:.3f}')
    axes[0, 0].axhline(EXOGENOUS_A['u_on'], color='gray', linestyle=':',
                          alpha=0.7, label=f'default u_on = {EXOGENOUS_A["u_on"]}')
    axes[0, 0].set_xlabel('time (h)')
    axes[0, 0].set_ylabel('u_target')
    axes[0, 0].set_title('SMC²-derived control schedule')
    axes[0, 0].legend(loc='best', fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: default schedule for reference
    axes[0, 1].plot(t_grid, default_schedule, '-', color='darkred', lw=2,
                       label='default u_target(t)')
    axes[0, 1].axhline(U_CRIT, color='red', linestyle=':', alpha=0.7,
                          label=f'u_c = {U_CRIT:.3f}')
    axes[0, 1].axvline(EXOGENOUS_A['T_intervention'], color='gray',
                          linestyle='--', alpha=0.5,
                          label=f'T_intervention = {EXOGENOUS_A["T_intervention"]} h')
    axes[0, 1].set_xlabel('time (h)')
    axes[0, 1].set_ylabel('u_target')
    axes[0, 1].set_title('Default (hand-coded) schedule for comparison')
    axes[0, 1].legend(loc='best', fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    for traj_i in range(n_traj):
        rng_key, sub = jax.random.split(rng_key)
        traj_smc = traj_sample_fn(jnp.asarray(smc_schedule), sub)
        rng_key, sub = jax.random.split(rng_key)
        traj_default = traj_sample_fn(jnp.asarray(default_schedule), sub)
        x_smc = np.asarray(traj_smc[0])
        x_default = np.asarray(traj_default[0])
        axes[1, 0].plot(t_grid, x_smc, alpha=0.6, lw=1)
        axes[1, 1].plot(t_grid, x_default, alpha=0.6, lw=1)

    for ax, title in zip(axes[1, :],
                           ['x(t) under SMC² schedule',
                            'x(t) under default schedule']):
        ax.axhline(-1.0, color='red', linestyle=':', alpha=0.5, label='x=-1 (sick)')
        ax.axhline(+1.0, color='green', linestyle=':', alpha=0.5, label='x=+1 (well)')
        ax.axhline(0.0, color='black', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('time (h)')
        ax.set_ylabel('x (health)')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Stage B2 — Bistable controlled: '
                  f'SMC² (cost {smc_cost:.2f}, {smc_success:.0%} success)  vs  '
                  f'default (cost {default_cost:.2f}, {default_success:.0%} success)',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
