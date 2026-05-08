"""Main API for low-rank OT resampling.

Combines kernel factorisation, Sinkhorn iterations, and barycentric
projection into a single differentiable resampling function.

This is the drop-in replacement for systematic resampling inside
the particle filter scan.  It is fully differentiable via jax.grad.

Date:    16 April 2026
Version: 4.1 — model-agnostic stochastic_indices

Change history:
  v4.1 (16 Apr 2026) — accept stochastic_indices as an argument so the
        same resampler runs for the OU test model, sleep-wake, and
        any other future EstimationModel.  Previously the indices were
        hardcoded to sleep-wake's (0, 1, 2, 4, 5).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Sequence

from smc2fc.filtering.transport_kernel import compute_kernel_factor
from smc2fc.filtering.sinkhorn import sinkhorn_scalings
from smc2fc.filtering.project import barycentric_projection

# Default hyperparameters (can be overridden per call)
DEFAULT_EPSILON = 0.5     # Sinkhorn regularisation
DEFAULT_N_ITER = 10       # Sinkhorn iterations
DEFAULT_RANK = 50         # Nyström anchor count


def ot_resample_lr(particles: Array, log_weights: Array,
                   rng_key: Array,
                   stochastic_indices: Sequence[int],
                   epsilon: float = DEFAULT_EPSILON,
                   n_iter: int = DEFAULT_N_ITER,
                   rank: int = DEFAULT_RANK) -> Array:
    """Differentiable OT resampling with O(N×r) Nyström Sinkhorn.

    Replaces systematic resampling (zero gradient) with a smooth
    transport map (finite gradient).  Works directly in [0,1]^d_s —
    the barycentric projection is a convex combination that preserves
    bounds automatically.

    The transport is applied ONLY to the stochastic states identified
    by ``stochastic_indices``.  Deterministic state components (e.g.
    analytical circadian terms) pass through unchanged.

    Algorithm:
        1. Extract the d_s stochastic states from each particle.
        2. Select r random anchor points among the N particles.
        3. Compute the Nyström kernel factor K_NR ∈ R^{N×r}.
        4. Run Sinkhorn iterations → scaling vectors (u, v).
        5. Barycentric projection → new positions.
        6. Reassemble the full state vector.

    Args:
        particles: Current particle states, shape (N, n_states).
            Each stochastic component must lie in [0, 1].
        log_weights: Unnormalised log-weights, shape (N,).
        rng_key: JAX PRNG key for random anchor selection.
        stochastic_indices: Sequence of integer indices identifying
            stochastic state dimensions.  Pass
            ``model.stochastic_indices`` from the EstimationModel.
        epsilon: Sinkhorn entropic regularisation.  Default 0.5.
        n_iter: Number of Sinkhorn iterations.  Default 10.
        rank: Number of Nyström anchor points.  Default 50.

    Returns:
        New particle states, shape (N, n_states).  Stochastic states
        are in [0, 1] (convex-combination guarantee).  Deterministic
        states are byte-identical to the input.
    """
    N = particles.shape[0]
    sto_idx = jnp.asarray(tuple(stochastic_indices), dtype=jnp.int32)

    # Marginals
    a = jnp.ones(N) / N
    b = jnp.exp(log_weights - jax.nn.logsumexp(log_weights))

    # Extract stochastic states only.
    x = particles[:, sto_idx]                                # (N, d_s)

    # Select random anchors among the N particles.
    anchor_idx = jax.random.choice(rng_key, N, shape=(rank,),
                                    replace=False)

    # Nyström kernel factors.
    K_NR = compute_kernel_factor(x, anchor_idx, epsilon)

    # Low-rank Sinkhorn.
    u, v = sinkhorn_scalings(a, b, K_NR, n_iter)

    # Transport.
    new_x = barycentric_projection(u, v, x, K_NR)

    # Reassemble: write transported stochastic states back into the
    # full state, leaving deterministic components unchanged.
    out = particles.at[:, sto_idx].set(new_x)
    return out
