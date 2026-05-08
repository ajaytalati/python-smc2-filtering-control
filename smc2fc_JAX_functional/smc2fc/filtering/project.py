"""Barycentric projection for optimal transport resampling.

Computes transported particle positions as a weighted average
(convex combination) using the Sinkhorn coupling.  The convex
combination property guarantees [0,1]^d preservation automatically.

Date:    15 April 2026
Version: 4.0 (functional refactor)
"""

import jax.numpy as jnp
from jax import Array

from smc2fc.filtering.transport_kernel import factor_matvec, factor_matvec_batch


def barycentric_projection(u: Array, v: Array, x: Array,
                           K_NR: Array) -> Array:
    """Transport particles via factored Sinkhorn coupling.

    Computes new positions as the normalised weighted average:
        new_x_i = (u_i * [K_approx @ (v ⊙ x)]_i) / (u_i * [K_approx @ v]_i)

    This is a convex combination of the input positions x, so if all
    x_j ∈ [0,1]^d, then new_x_i ∈ [0,1]^d automatically.  No logit
    transform or sigmoid back-map needed.

    Complexity: O(N × r × d), no N×N matrix materialised.

    Args:
        u: Row scaling from Sinkhorn, shape (N,).
        v: Column scaling from Sinkhorn, shape (N,).
        x: Current particle positions, shape (N, d).
        K_NR: Kernel factor, shape (N, r).

    Returns:
        Transported particle positions, shape (N, d).
        All entries in [0,1] if inputs are in [0,1] (convex combination).
    """
    denom = u * factor_matvec(v, K_NR)
    denom = jnp.maximum(denom, 1e-30)

    vx = v[:, None] * x
    numer = u[:, None] * factor_matvec_batch(vx, K_NR)

    return numer / denom[:, None]
