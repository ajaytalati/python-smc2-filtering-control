"""Stage A3: SMC²-as-controller with state-feedback parameterisation.

Stage A2 demonstrated that tempered SMC over an open-loop schedule
recovers the optimal *deterministic* schedule (u(t) ≡ 0 in this case).
The 4-unit cost gap to the LQR/LQG optimum is the value of state
feedback, unreachable by any open-loop schedule.

Stage A3 closes that gap: reparameterise the control as

    u_k = -K_k * x̂_k             (state-feedback law)

where x̂_k is the inline Kalman posterior mean of the latent state at
step k, and run tempered SMC over the GAIN VECTOR K = (K_0, ..., K_{T-1}).

The optimal K under expected-cost minimisation is the analytical LQR
Riccati gain sequence; the joint LQG cost (with Kalman filter for x̂)
should match the LQR perfect-state cost via the separation principle.

Acceptance gate:
    SMC²-derived gain RMS error vs Riccati gains < 25%
    SMC²-derived cost / MC LQG cost ∈ [0.95, 1.10]
    SMC²-derived cost / open-loop cost ≤ 0.7  (state feedback genuinely helps)

Run:
    PYTHONPATH=. python tools/bench_smc_control_ou_state_feedback.py
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

from smc2fc.core.mass_matrix import estimate_mass_matrix

from models.scalar_ou_lqg.bench_lqr import (
    lqr_riccati, lqr_optimal_cost,
    lqg_optimal_cost_monte_carlo,
    open_loop_zero_control_cost_monte_carlo,
)


# ── Truth parameters (same as Stage A2) ───────────────────────────────

TRUTH = dict(
    a=1.0, b=1.0, sigma_w=0.3, sigma_v=0.2,
    q=1.0, r=0.1, s=1.0,
    dt=0.05, T=20,
    x0_mean=0.0, x0_var=1.0,
)


# ── Cost evaluator under STATE-FEEDBACK control u_k = -K_k * x̂_k ─────

def _build_cost_fn_state_feedback(*, n_inner: int, seed: int):
    """Build a JIT-compiled J(K) under the state-feedback law u = -K · x̂.

    The Kalman filter for x̂_k is rolled inline inside the cost
    evaluator so the cost function is a clean K → ℝ map for tempered
    SMC.

    Common-random-numbers (CRN): Wiener increments and observation
    noise are FIXED across cost evaluations so per-particle cost
    differences reflect K differences, not noise differences.
    """
    A = 1.0 - TRUTH['a'] * TRUTH['dt']
    B = TRUTH['b'] * TRUTH['dt']
    sw = TRUTH['sigma_w'] * jnp.sqrt(TRUTH['dt'])
    Q = TRUTH['sigma_w'] ** 2 * TRUTH['dt']
    R = TRUTH['sigma_v'] ** 2
    T_steps = TRUTH['T']

    rng = np.random.default_rng(seed)
    fixed_w = jnp.asarray(
        rng.standard_normal((n_inner, T_steps)), dtype=jnp.float64)
    fixed_v = jnp.asarray(
        rng.standard_normal((n_inner, T_steps)), dtype=jnp.float64)
    fixed_x0 = jnp.asarray(
        TRUTH['x0_mean']
        + jnp.sqrt(TRUTH['x0_var']) * rng.standard_normal((n_inner,)),
        dtype=jnp.float64,
    )

    @jax.jit
    def J(K):
        """Mean cost over the fixed CRN ensemble under feedback law u=-K·x̂."""
        def trial(x0, w_seq, v_seq):
            def step(carry, k):
                x, x_hat_mean, x_hat_var, cost = carry
                # Observe + Kalman update at step k
                y = x + TRUTH['sigma_v'] * v_seq[k]
                S = x_hat_var + R
                G = x_hat_var / S
                x_hat_mean_post = x_hat_mean + G * (y - x_hat_mean)
                x_hat_var_post = (1.0 - G) * x_hat_var
                # State-feedback action
                u_k = -K[k] * x_hat_mean_post
                stage_cost = TRUTH['q'] * x ** 2 + TRUTH['r'] * u_k ** 2
                # Advance true state and Kalman predictive
                x_next = A * x + B * u_k + sw * w_seq[k]
                x_hat_mean_pred = A * x_hat_mean_post + B * u_k
                x_hat_var_pred = A * A * x_hat_var_post + Q
                return (
                    x_next, x_hat_mean_pred, x_hat_var_pred,
                    cost + stage_cost,
                ), None
            init_carry = (
                x0,
                jnp.float64(TRUTH['x0_mean']),
                jnp.float64(TRUTH['x0_var']),
                jnp.float64(0.0),
            )
            (x_T, _, _, total_stage), _ = jax.lax.scan(
                step, init_carry, jnp.arange(T_steps)
            )
            return total_stage + TRUTH['s'] * x_T ** 2
        costs = jax.vmap(trial)(fixed_x0, fixed_w, fixed_v)
        return jnp.mean(costs)

    return J


# ── Tempered SMC outer loop over gain vector K ────────────────────────

def run_tempered_smc_over_gains(
    *,
    n_smc: int = 256,
    n_inner: int = 64,
    sigma_prior: float = 3.0,
    K_prior_mean: float = 1.5,
    beta_max: float | None = None,
    target_ess_frac: float = 0.5,
    max_lambda_inc: float = 0.10,
    num_mcmc_steps: int = 5,
    hmc_step_size: float = 0.05,
    hmc_num_leapfrog: int = 8,
    seed: int = 42,
):
    """Tempered SMC over K ∈ ℝ^T with target ∝ prior(K) * exp(-β · J(K)).

    The prior is N(K_prior_mean, sigma_prior²) per component — broad,
    centered at a positive value (LQR Riccati gains for this setup are
    in roughly [0.5, 2.5]).
    """
    T_steps = TRUTH['T']
    dim = T_steps
    cost_fn = _build_cost_fn_state_feedback(n_inner=n_inner, seed=seed)

    rng_key = jax.random.PRNGKey(seed)
    # Sample prior cloud for β_max calibration
    rng_key, sub = jax.random.split(rng_key)
    prior_samples = (K_prior_mean
                     + sigma_prior
                     * jax.random.normal(sub, (1024, dim), dtype=jnp.float64))
    prior_costs = jax.vmap(cost_fn)(prior_samples)
    prior_cost_std = float(jnp.std(prior_costs))
    if beta_max is None:
        beta_max = float(8.0 / max(prior_cost_std, 1e-6))
    print(f"  prior cost mean = {float(jnp.mean(prior_costs)):.3f}, "
          f"std = {prior_cost_std:.3f}")
    print(f"  beta_max (auto) = {beta_max:.4f}")

    @jax.jit
    def logprior_fn(K):
        return jnp.sum(
            -0.5 * ((K - K_prior_mean) / sigma_prior) ** 2
            - jnp.log(sigma_prior) - 0.5 * jnp.log(2 * jnp.pi)
        )

    @jax.jit
    def loglikelihood_fn(K):
        return -beta_max * cost_fn(K)

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
    init_particles = (K_prior_mean
                      + sigma_prior
                      * jax.random.normal(sub, (n_smc, dim), dtype=jnp.float64))
    state = tempered.init(init_particles)
    inv_mass = estimate_mass_matrix(init_particles)

    print(f"  starting tempered SMC over K ∈ ℝ^{dim}, "
          f"N_SMC={n_smc}, N_inner={n_inner}")
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
            'num_integration_steps': jnp.array([hmc_num_leapfrog],
                                                dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(step_key, state, num_mcmc_steps,
                                       jnp.float64(next_lam), mcmc_params)
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
        'n_temp_levels':   n_temp,
        'elapsed_s':       elapsed,
        'beta_max':        beta_max,
    }


# ── Headline run ──────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Stage A3 — SMC²-as-controller with state feedback (LQR/LQG demo)")
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
    print(f"  MC LQG (Kalman + LQR Riccati):    {lqg['mean_cost']:.4f} ± {lqg['stderr']:.4f}")
    print(f"  MC open-loop (u=0):               {open_loop['mean_cost']:.4f} ± {open_loop['stderr']:.4f}")
    print(f"  Riccati gains K_0..K_{TRUTH['T']-1}:")
    print(f"    {[f'{x:.3f}' for x in riccati.gains]}")
    print()

    res = run_tempered_smc_over_gains(
        n_smc=128, n_inner=64,
        sigma_prior=3.0, K_prior_mean=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        seed=42,
    )
    particles = res['particles_final']
    K_mean = particles.mean(axis=0)
    K_std = particles.std(axis=0)

    print()
    print(f"  tempered SMC done: {res['n_temp_levels']} levels in "
          f"{res['elapsed_s']:.1f}s")
    print(f"  posterior mean gain K:")
    print(f"    {[f'{x:.3f}' for x in K_mean]}")
    print()

    # Cost of the SMC²-mean gain schedule under fresh CRN
    cost_fn_eval = _build_cost_fn_state_feedback(n_inner=2000, seed=99)
    smc_mean_cost = float(cost_fn_eval(jnp.asarray(K_mean)))
    smc_per_particle_cost = jax.vmap(cost_fn_eval)(jnp.asarray(particles))
    smc_posterior_per_particle_cost = float(jnp.mean(smc_per_particle_cost))

    # Cost of using the analytical Riccati gains in the same evaluator
    riccati_cost = float(cost_fn_eval(jnp.asarray(riccati.gains)))

    print(f"  SMC²-mean-gain cost:                 {smc_mean_cost:.4f}")
    print(f"  Riccati-gain cost (same evaluator):  {riccati_cost:.4f}")
    print(f"  SMC² posterior per-particle cost:    "
          f"{smc_posterior_per_particle_cost:.4f}")
    print()
    print(f"  ratio  SMC²-mean / Riccati-evaluator: "
          f"{smc_mean_cost / riccati_cost:.4f}")
    print(f"  ratio  SMC²-mean / MC LQG:           "
          f"{smc_mean_cost / lqg['mean_cost']:.4f}")
    print(f"  ratio  SMC²-mean / open-loop:        "
          f"{smc_mean_cost / open_loop['mean_cost']:.4f}")

    # Acceptance gates
    K_rms_err = float(np.sqrt(
        np.mean((K_mean - np.asarray(riccati.gains)) ** 2)
        / np.mean(np.asarray(riccati.gains) ** 2)
    ))
    pass_rms = K_rms_err < 0.25
    pass_lqg = 0.95 <= smc_mean_cost / lqg['mean_cost'] <= 1.10
    pass_better = smc_mean_cost / open_loop['mean_cost'] <= 0.7
    print()
    print(f"  Stage A3 acceptance gates:")
    print(f"    K RMS error vs Riccati < 25%:           {K_rms_err:.3f}  "
          f"{'✓' if pass_rms else '✗'}")
    print(f"    cost / MC LQG ∈ [0.95, 1.10]:           "
          f"{smc_mean_cost / lqg['mean_cost']:.3f}  "
          f"{'✓' if pass_lqg else '✗'}")
    print(f"    cost / open-loop ≤ 0.7:                 "
          f"{smc_mean_cost / open_loop['mean_cost']:.3f}  "
          f"{'✓' if pass_better else '✗'}")
    if pass_rms and pass_lqg and pass_better:
        print(f"  ✓ Stage A3 PASSES all acceptance gates")
    else:
        print(f"  ✗ Stage A3 FAILS one or more gates")

    # ── Plot ──
    out_path = "outputs/scalar_ou_lqg/A3_state_feedback_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    t = np.arange(TRUTH['T']) * TRUTH['dt']
    axes[0].fill_between(t, K_mean - K_std, K_mean + K_std,
                            alpha=0.3, color='steelblue', label='SMC ±1σ')
    axes[0].plot(t, K_mean, 'o-', color='steelblue',
                   label='SMC² posterior mean K_k', alpha=0.85)
    axes[0].plot(t, np.asarray(riccati.gains), 's--', color='darkred',
                   label='LQR Riccati K_k*', alpha=0.85)
    axes[0].axhline(0, color='black', alpha=0.3, linewidth=0.5)
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('feedback gain K_k')
    axes[0].set_title('State-feedback gain: SMC² posterior vs Riccati')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.asarray(smc_per_particle_cost), bins=30,
                   color='steelblue', alpha=0.7,
                   label='SMC² per-particle cost')
    axes[1].axvline(lqr_perfect, color='green', linestyle='--', linewidth=2,
                       label=f'LQR perfect state = {lqr_perfect:.2f}')
    axes[1].axvline(lqg['mean_cost'], color='orange', linestyle='--',
                       linewidth=2,
                       label=f'MC LQG = {lqg["mean_cost"]:.2f}')
    axes[1].axvline(open_loop['mean_cost'], color='red', linestyle='--',
                       linewidth=2,
                       label=f'open-loop = {open_loop["mean_cost"]:.2f}')
    axes[1].axvline(smc_mean_cost, color='steelblue', linestyle=':',
                       linewidth=2,
                       label=f'SMC² mean K = {smc_mean_cost:.2f}')
    axes[1].axvline(riccati_cost, color='purple', linestyle=':',
                       linewidth=2,
                       label=f'Riccati K = {riccati_cost:.2f}')
    axes[1].set_xlabel('cost')
    axes[1].set_ylabel('density')
    axes[1].set_title('SMC² state-feedback cost vs analytical references')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
