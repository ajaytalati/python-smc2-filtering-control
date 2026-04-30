"""Generic tempered-SMC outer loop for SMC²-as-controller.

Mirrors smc2fc.core.tempered_smc.run_smc_window but tuned for the
control side:

  - The loglikelihood_fn is `-β · J(θ)` where J is the model's cost
    functional.
  - The logprior_fn is a Gaussian prior over the schedule
    parameter vector θ (broad, sigma_prior).
  - β_max is auto-calibrated from the prior-cloud cost spread.
  - The mass matrix is re-estimated each tempering level from the
    current particle cloud (smc2fc.core.mass_matrix.estimate_mass_matrix).

Returns a dict with the final particle cloud, per-particle costs at
the final temperature, the SMC²-mean schedule, the tempering-level
count, and elapsed time. The model-specific acceptance gates and
plotting consume this result dict.
"""

from __future__ import annotations

import time
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
import blackjax.smc.tempered as tempered
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as solver

from smc2fc.core.mass_matrix import estimate_mass_matrix

from smc2fc.control.config import SMCControlConfig
from smc2fc.control.control_spec import ControlSpec
from smc2fc.control.calibration import calibrate_beta_max


def run_tempered_smc_loop_native(
    *,
    spec: ControlSpec,
    cfg: SMCControlConfig,
    seed: int = 42,
    print_progress: bool = True,
) -> dict:
    """Stage M mirror for the controller side.

    Mirrors `run_tempered_smc_loop` but routes through the JAX-native
    tempered SMC kernel (smc2fc.core.jax_native_smc), which compiles
    once at module load and reuses the trace across replans. Closes
    the per-replan ~15s recompile that was the controller-side
    counterpart to the filter-side BlackJAX issue.

    Same return-dict contract as the BlackJAX version.
    """
    if spec.cost_fn is None:
        raise ValueError(f"ControlSpec {spec.name!r} has no cost_fn")
    if spec.theta_dim <= 0:
        raise ValueError(f"ControlSpec {spec.name!r} has theta_dim={spec.theta_dim}")

    from smc2fc.core.jax_native_smc import _run_tempered_chain_jit

    cost_fn = spec.cost_fn

    # 1. β_max calibration (unchanged from BlackJAX path)
    beta_max, prior_cost_mean, prior_cost_std = calibrate_beta_max(
        cost_fn,
        theta_dim=spec.theta_dim,
        sigma_prior=spec.sigma_prior,
        prior_mean=spec.prior_mean,
        n_samples=cfg.n_calibration_samples,
        target_nats=cfg.beta_max_target_nats,
        seed=seed,
    )
    if print_progress:
        print(f"  prior cost mean = {prior_cost_mean:.3f}, "
              f"std = {prior_cost_std:.3f}")
        print(f"  beta_max (auto) = {beta_max:.4f}")

    # 2. Pytree-stable Partial-wrapped logprior + loglikelihood.
    sigma_prior = float(spec.sigma_prior)
    # Ensure prior_mean is a (theta_dim,) array even if spec stores a scalar.
    pm_arr = jnp.asarray(spec.prior_mean, dtype=jnp.float64)
    if pm_arr.ndim == 0:
        pm_arr = jnp.broadcast_to(pm_arr, (spec.theta_dim,))
    prior_mean = pm_arr

    def _logprior_inner(sigma, mean, theta):
        return jnp.sum(
            -0.5 * ((theta - mean) / sigma) ** 2
            - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)
        )

    logprior_fn = jax.tree_util.Partial(
        _logprior_inner, jnp.float64(sigma_prior), prior_mean,
    )

    def _loglikelihood_inner(beta, cost_callable, theta):
        return -beta * cost_callable(theta)

    loglikelihood_fn = jax.tree_util.Partial(
        _loglikelihood_inner, jnp.float64(beta_max), cost_fn,
    )

    # 3. Initial particle cloud — drawn from prior.
    rng_key = jax.random.PRNGKey(seed + 17)
    rng_key, sub = jax.random.split(rng_key)
    init_particles = (
        prior_mean[None, :]
        + sigma_prior * jax.random.normal(
            sub, (cfg.n_smc, spec.theta_dim), dtype=jnp.float64,
        )
    )

    if print_progress:
        print(f"  starting tempered SMC (native): θ ∈ ℝ^{spec.theta_dim}, "
              f"N_SMC={cfg.n_smc}, N_inner={cfg.n_inner}, "
              f"n_steps={spec.n_steps}")

    # 4. Run native chain — single jitted call.
    t0 = time.time()
    final_particles, n_temp = _run_tempered_chain_jit(
        init_particles, logprior_fn, loglikelihood_fn,
        rng_key,
        jnp.float64(cfg.target_ess_frac),
        jnp.float64(cfg.max_lambda_inc),
        int(cfg.num_mcmc_steps),
        jnp.float64(cfg.hmc_step_size),
        int(cfg.hmc_num_leapfrog),
    )
    final_particles.block_until_ready()
    elapsed = time.time() - t0

    particles = np.asarray(final_particles)
    mean_theta = particles.mean(axis=0)

    particle_costs = np.asarray(jax.vmap(cost_fn)(jnp.asarray(particles)))

    if spec.schedule_from_theta is not None:
        mean_schedule = np.asarray(
            spec.schedule_from_theta(jnp.asarray(mean_theta))
        )
    else:
        mean_schedule = None

    if print_progress:
        print(f"  done: n_temp={int(n_temp)}, elapsed={elapsed:.1f}s, "
              f"mean cost ≈ {float(particle_costs.mean()):.3f}")

    return {
        'particles':       particles,
        'particle_costs':  particle_costs,
        'mean_theta':      mean_theta,
        'mean_schedule':   mean_schedule,
        'beta_max':        beta_max,
        'prior_cost_mean': prior_cost_mean,
        'prior_cost_std':  prior_cost_std,
        'n_temp_levels':   int(n_temp),
        'elapsed_s':       float(elapsed),
        'spec':            spec,
        'cfg':             cfg,
    }


def run_tempered_smc_loop(
    *,
    spec: ControlSpec,
    cfg: SMCControlConfig,
    seed: int = 42,
    print_progress: bool = True,
) -> dict:
    """Run tempered SMC over a ControlSpec's schedule parameter vector.

    Args:
        spec: model-specific ControlSpec (cost_fn, theta_dim, etc.).
        cfg: tempered-SMC + HMC knobs.
        seed: integer RNG seed for reproducibility.
        print_progress: log per-tempering-level status.

    Returns:
        dict with keys:
            'particles'              (n_smc, theta_dim) final cloud
            'particle_costs'         (n_smc,) per-particle final cost
            'mean_theta'             (theta_dim,) posterior-mean θ
            'mean_schedule'          (n_steps,) or model-specific shape
            'beta_max'               float
            'prior_cost_mean'        float
            'prior_cost_std'         float
            'n_temp_levels'          int
            'elapsed_s'              float
            'spec'                   the ControlSpec (passed through)
            'cfg'                    the SMCControlConfig (passed through)
    """
    if spec.cost_fn is None:
        raise ValueError(f"ControlSpec {spec.name!r} has no cost_fn")
    if spec.theta_dim <= 0:
        raise ValueError(f"ControlSpec {spec.name!r} has theta_dim={spec.theta_dim}")

    cost_fn = spec.cost_fn

    # 1. β_max calibration
    beta_max, prior_cost_mean, prior_cost_std = calibrate_beta_max(
        cost_fn,
        theta_dim=spec.theta_dim,
        sigma_prior=spec.sigma_prior,
        prior_mean=spec.prior_mean,
        n_samples=cfg.n_calibration_samples,
        target_nats=cfg.beta_max_target_nats,
        seed=seed,
    )
    if print_progress:
        print(f"  prior cost mean = {prior_cost_mean:.3f}, "
              f"std = {prior_cost_std:.3f}")
        print(f"  beta_max (auto) = {beta_max:.4f}")

    # 2. logprior + loglikelihood
    sigma_prior = spec.sigma_prior
    prior_mean = spec.prior_mean

    @jax.jit
    def logprior_fn(theta):
        return jnp.sum(
            -0.5 * ((theta - prior_mean) / sigma_prior) ** 2
            - jnp.log(sigma_prior) - 0.5 * jnp.log(2 * jnp.pi)
        )

    @jax.jit
    def loglikelihood_fn(theta):
        return -beta_max * cost_fn(theta)

    # 3. Build tempered-SMC kernel
    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    # 4. Initial particle cloud — drawn from prior `N(prior_mean, sigma_prior²)`
    rng_key = jax.random.PRNGKey(seed + 17)
    rng_key, sub = jax.random.split(rng_key)
    init_particles = (prior_mean
                      + sigma_prior * jax.random.normal(
                          sub, (cfg.n_smc, spec.theta_dim), dtype=jnp.float64,
                      ))
    state = tempered.init(init_particles)
    inv_mass = estimate_mass_matrix(init_particles)

    if print_progress:
        print(f"  starting tempered SMC: θ ∈ ℝ^{spec.theta_dim}, "
              f"N_SMC={cfg.n_smc}, N_inner={cfg.n_inner}, "
              f"n_steps={spec.n_steps}")

    t0 = time.time()
    n_temp = 0
    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)
        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            cfg.target_ess_frac, max_delta, solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, cfg.max_lambda_inc)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_params = {
            'step_size': jnp.array([cfg.hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array(
                [cfg.hmc_num_leapfrog], dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, cfg.num_mcmc_steps,
            jnp.float64(next_lam), mcmc_params,
        )
        inv_mass = estimate_mass_matrix(state.particles)
        n_temp += 1

        lam = float(state.tempering_param)
        try:
            acc = float(jnp.mean(info.update_info.acceptance_rate))
        except Exception:
            acc = float('nan')

        if print_progress and (
            n_temp % cfg.log_every_n_steps == 0 or lam >= 1.0
        ):
            mean_cost = float(jnp.mean(jax.vmap(cost_fn)(state.particles)))
            print(f"    step {n_temp:3d}: λ={lam:.3f}  acc={acc:.3f}  "
                  f"mean cost = {mean_cost:.3f}")

        if n_temp > cfg.max_temp_steps:
            if print_progress:
                print(f"  hit max_temp_steps={cfg.max_temp_steps}; breaking")
            break

    elapsed = time.time() - t0
    particles = np.asarray(state.particles)
    mean_theta = particles.mean(axis=0)

    # Per-particle final cost
    particle_costs = np.asarray(jax.vmap(cost_fn)(jnp.asarray(particles)))

    # Mean schedule (if the spec exposes one)
    if spec.schedule_from_theta is not None:
        mean_schedule = np.asarray(
            spec.schedule_from_theta(jnp.asarray(mean_theta))
        )
    else:
        mean_schedule = None

    return {
        'particles':       particles,
        'particle_costs':  particle_costs,
        'mean_theta':      mean_theta,
        'mean_schedule':   mean_schedule,
        'beta_max':        beta_max,
        'prior_cost_mean': prior_cost_mean,
        'prior_cost_std':  prior_cost_std,
        'n_temp_levels':   int(n_temp),
        'elapsed_s':       float(elapsed),
        'spec':            spec,
        'cfg':             cfg,
    }
