"""Tests for smc2fc.control.calibration."""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control.calibration import calibrate_beta_max, build_crn_noise_grids


def test_calibrate_beta_max_target_nats_match():
    """β_max = target_nats / std(J under prior) — to floating point."""
    @jax.jit
    def J(theta):
        return jnp.sum(theta ** 2)

    theta_dim = 5
    sigma_prior = 1.0
    target_nats = 8.0

    beta, mean_cost, std_cost = calibrate_beta_max(
        J, theta_dim=theta_dim, sigma_prior=sigma_prior,
        n_samples=1024, target_nats=target_nats, seed=42,
    )
    expected_beta = target_nats / std_cost
    assert abs(beta - expected_beta) < 1e-9
    # Sanity: prior J ~ chi-square with 5 dof × sigma² → mean ≈ 5
    assert abs(mean_cost - 5.0) < 0.5
    assert std_cost > 0.0


def test_calibrate_beta_max_with_prior_mean_shift():
    """A non-zero prior_mean shifts the prior cost distribution."""
    @jax.jit
    def J(theta):
        return jnp.sum((theta - 1.5) ** 2)

    # If prior centered at 0, J is large (offset by 1.5²×n)
    _, mean_at_zero, _ = calibrate_beta_max(
        J, theta_dim=4, sigma_prior=0.5, prior_mean=0.0,
        n_samples=512, seed=0,
    )
    # If prior centered at 1.5, J is small (centered on the optimum)
    _, mean_at_1_5, _ = calibrate_beta_max(
        J, theta_dim=4, sigma_prior=0.5, prior_mean=1.5,
        n_samples=512, seed=0,
    )
    assert mean_at_zero > mean_at_1_5 + 1.0, (
        f"prior_mean shift had no effect: {mean_at_zero=}, {mean_at_1_5=}"
    )


def test_build_crn_noise_grids_shapes_and_determinism():
    """Returned arrays have the right shape and are seed-deterministic."""
    grids = build_crn_noise_grids(
        n_inner=4, n_steps=10, n_channels=2, seed=7,
    )
    assert grids['wiener'].shape == (4, 10, 2)
    assert grids['initial'].shape == (4,)

    # Re-create with same seed → identical arrays
    grids2 = build_crn_noise_grids(
        n_inner=4, n_steps=10, n_channels=2, seed=7,
    )
    np.testing.assert_array_equal(np.asarray(grids['wiener']),
                                    np.asarray(grids2['wiener']))
    np.testing.assert_array_equal(np.asarray(grids['initial']),
                                    np.asarray(grids2['initial']))

    # Different seed → different arrays
    grids3 = build_crn_noise_grids(
        n_inner=4, n_steps=10, n_channels=2, seed=8,
    )
    assert not np.allclose(np.asarray(grids['wiener']),
                              np.asarray(grids3['wiener']))
