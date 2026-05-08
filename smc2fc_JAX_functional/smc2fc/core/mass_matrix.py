"""Diagonal mass-matrix estimation for the per-tempering-level HMC kernel.

Full mass matrices fail on this problem: the PF likelihood landscape
punishes correlated HMC proposals and acceptance collapses to zero by
λ ~ 0.3. The diagonal approximation is stable and adapts per-level
from the current particle cloud's per-dimension variance.
"""

import jax.numpy as jnp
from jax import Array


def estimate_mass_matrix(particles: Array,
                          regularisation: float = 1e-4) -> Array:
    """Returns the diagonal inverse mass matrix from per-dim variance.

    Pure JAX: variance is computed across the particle axis, then
    floored at ``regularisation`` to keep HMC stable when a dim
    happens to be (near-)deterministic.

    Args:
        particles: Particle cloud of shape ``(n_smc, n_dim)``.
        regularisation: Lower bound applied to each variance entry.
            Prevents the mass matrix from blowing up when a dim has
            collapsed to a near-point estimate.

    Returns:
        Diagonal inverse mass matrix of shape ``(1, n_dim)``. The
        leading ``1`` axis matches BlackJAX HMC's expected
        ``inverse_mass_matrix`` shape (broadcast over chains).
    """
    var = jnp.var(particles, axis=0)
    var = jnp.maximum(var, regularisation)
    return var[None, :]
