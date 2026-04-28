"""Generic calibration helpers for the SMC² control engine:

  - calibrate_beta_max: auto-set β_max so the prior-cloud cost spread
    maps to a target nat-budget (default 8 nats ≈ 16 effective
    tempering levels).

  - build_crn_noise_grids: construct fixed Wiener-increment + initial-
    condition arrays once, reused across all cost evaluations
    (common-random-numbers variance reduction).
"""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np


def calibrate_beta_max(
    cost_fn,
    *,
    theta_dim: int,
    sigma_prior: float,
    n_samples: int = 256,
    target_nats: float = 8.0,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Auto-calibrate β_max from the prior-cloud cost spread.

    Sample N_calib θ values from the broad Gaussian prior, evaluate
    cost_fn on each, set β_max = target_nats / std(costs).

    Returns (β_max, prior_cost_mean, prior_cost_std).
    """
    rng_key = jax.random.PRNGKey(seed)
    prior_samples = sigma_prior * jax.random.normal(
        rng_key, (n_samples, theta_dim), dtype=jnp.float64,
    )
    prior_costs = jax.vmap(cost_fn)(prior_samples)
    prior_cost_mean = float(jnp.mean(prior_costs))
    prior_cost_std = float(jnp.std(prior_costs))
    beta_max = float(target_nats / max(prior_cost_std, 1e-6))
    return beta_max, prior_cost_mean, prior_cost_std


def build_crn_noise_grids(
    *,
    n_inner: int,
    n_steps: int,
    n_channels: int = 1,
    seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    """Construct fixed Gaussian noise arrays for common-random-numbers
    cost evaluation.

    Each call returns a deterministic (seeded) dict of JAX arrays that
    the model's cost_fn closure captures and reuses across all SMC²
    particles. Same noise → cost differences reflect θ differences,
    not noise differences (variance reduction).

    Args:
        n_inner: number of MC trajectories per cost evaluation.
        n_steps: number of outer time steps.
        n_channels: number of independent Wiener processes (e.g. 2
            for bistable's [B_x, B_u], 3 for FSA's [B_B, B_F, B_A]).
        seed: integer RNG seed.

    Returns:
        dict with keys 'wiener' (shape (n_inner, n_steps, n_channels))
        and 'initial' (shape (n_inner,)) — the latter for sampling
        the initial-state distribution. Models that fix the initial
        state to a specific value can ignore 'initial'.
    """
    rng = np.random.default_rng(seed)
    wiener = jnp.asarray(
        rng.standard_normal((n_inner, n_steps, n_channels)),
        dtype=jnp.float64,
    )
    initial = jnp.asarray(
        rng.standard_normal((n_inner,)), dtype=jnp.float64,
    )
    return {'wiener': wiener, 'initial': initial}
