"""Smoke test for run_tempered_smc_loop on a 1-D toy problem.

Target: minimise J(theta) = (theta - 1)^2. Tempered SMC should
concentrate the posterior near theta = 1.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import (
    ControlSpec, SMCControlConfig, run_tempered_smc_loop,
)


def test_tempered_smc_finds_quadratic_optimum():
    """SMC² posterior over θ ∈ ℝ should concentrate near argmin J = 1."""

    @jax.jit
    def J(theta):
        # 1-D quadratic with optimum at theta = 1
        return jnp.sum((theta - 1.0) ** 2)

    spec = ControlSpec(
        name='quadratic_toy', version='1.0',
        dt=1.0, n_steps=1, n_substeps=1,
        initial_state=None,
        truth_params={},
        theta_dim=1,
        sigma_prior=2.0,
        cost_fn=J,
        schedule_from_theta=None,
        acceptance_gates={},
    )
    cfg = SMCControlConfig(
        n_smc=64, n_inner=1, sigma_prior=2.0,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        beta_max_target_nats=8.0,
        n_calibration_samples=256,
    )
    result = run_tempered_smc_loop(spec=spec, cfg=cfg, seed=42,
                                    print_progress=False)

    mean_theta = float(result['mean_theta'][0])
    std_theta = float(result['particles'].std(axis=0)[0])

    # Posterior should be concentrated within ±0.3 of optimum θ=1
    assert abs(mean_theta - 1.0) < 0.3, (
        f"posterior mean {mean_theta:.3f} too far from argmin θ=1"
    )
    # And sufficiently concentrated (well below the prior σ=2)
    assert std_theta < 1.0, (
        f"posterior std {std_theta:.3f} should be < prior σ=2 / 2"
    )
    assert result['n_temp_levels'] >= 5
    assert result['n_temp_levels'] <= 50


def test_tempered_smc_with_prior_mean_shift():
    """When the optimum is far from 0, a prior_mean shift speeds convergence."""

    @jax.jit
    def J(theta):
        return jnp.sum((theta - 5.0) ** 2)    # optimum at θ=5

    spec = ControlSpec(
        name='shifted_toy', version='1.0',
        dt=1.0, n_steps=1, n_substeps=1,
        initial_state=None,
        truth_params={},
        theta_dim=1,
        sigma_prior=2.0,
        prior_mean=5.0,    # set near the known optimum
        cost_fn=J,
        schedule_from_theta=None,
        acceptance_gates={},
    )
    cfg = SMCControlConfig(
        n_smc=64, n_inner=1, sigma_prior=2.0,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        beta_max_target_nats=8.0,
    )
    result = run_tempered_smc_loop(spec=spec, cfg=cfg, seed=42,
                                    print_progress=False)
    mean_theta = float(result['mean_theta'][0])
    assert abs(mean_theta - 5.0) < 0.3, (
        f"posterior mean {mean_theta:.3f} should be near θ=5"
    )
