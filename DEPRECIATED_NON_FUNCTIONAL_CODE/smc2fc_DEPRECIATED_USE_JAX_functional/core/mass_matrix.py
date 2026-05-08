"""Diagonal mass-matrix estimation for the per-tempering-level HMC kernel.

Full mass matrices fail on this problem: the PF likelihood landscape
punishes correlated HMC proposals and acceptance collapses to zero by
lambda ~0.3. The diagonal approximation is stable and adapts per-level
from the current particle cloud's per-dimension variance.
"""

import jax.numpy as jnp


def estimate_mass_matrix(particles, regularisation: float = 1e-4):
    """Diagonal inverse mass matrix from per-dim particle variance.

    Returns a ``(1, n_dim)`` array — the leading axis matches BlackJAX HMC's
    expected shape for ``inverse_mass_matrix``.
    """
    var = jnp.var(particles, axis=0)
    var = jnp.maximum(var, regularisation)
    return var[None, :]
