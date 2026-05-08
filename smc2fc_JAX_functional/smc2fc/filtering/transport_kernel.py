"""Nyström kernel factorisation for low-rank OT.

Computes the factor K_NR ∈ R^{N×r} between all particles and r anchor
points.  The approximate kernel is K ≈ K_NR @ K_NR^T (column-normalised).

No matrix inverse (avoids cuSolver GPU handle exhaustion inside
jax.lax.scan + jax.checkpoint).  All operations are matmuls only.

Date:    15 April 2026
Version: 4.0 (functional refactor)
"""

import jax.numpy as jnp
from jax import Array


def compute_kernel_factor(x: Array, anchor_idx: Array,
                          epsilon: float) -> Array:
    """Compute Nyström kernel factor between particles and anchors.

    The Gaussian kernel K[i,j] = exp(-||x_i - z_j||² / ε) is computed
    between all N particles and r anchors, then column-normalised for
    numerical stability.  No matrix inverse is needed.

    Args:
        x: Particle positions in [0,1]^d, shape (N, d).
        anchor_idx: Indices of anchor particles, shape (r,).
            Must satisfy 0 <= anchor_idx[j] < N for all j.
        epsilon: Kernel bandwidth (Sinkhorn regularisation).
            Larger ε → smoother kernel → faster convergence.
            For [0,1]^5 with max distance √5 ≈ 2.24, ε=0.5 is typical.

    Returns:
        K_NR: Column-normalised kernel factor, shape (N, r).
            Satisfies: K_approx = K_NR @ K_NR^T ≈ K (the full kernel).
            All entries are non-negative.
    """
    anchors = x[anchor_idx]                                  # (r, d)
    diff = x[:, None, :] - anchors[None, :, :]               # (N, r, d)
    K_NR = jnp.exp(-jnp.sum(diff ** 2, axis=-1) / epsilon)   # (N, r)

    col_norms = jnp.maximum(jnp.sum(K_NR, axis=0), 1e-30)    # (r,)
    K_NR = K_NR / col_norms[None, :]

    return K_NR


def factor_matvec(v: Array, K_NR: Array) -> Array:
    """Approximate kernel-vector product: K_approx @ v.

    Computes K_NR @ (K_NR^T @ v) in O(N×r) instead of O(N²).

    Args:
        v: Vector to multiply, shape (N,).
        K_NR: Kernel factor from compute_kernel_factor, shape (N, r).

    Returns:
        Result of K_approx @ v, shape (N,).
    """
    return K_NR @ (K_NR.T @ v)


def factor_matvec_batch(V: Array, K_NR: Array) -> Array:
    """Batched kernel-matrix product: K_approx @ V.

    Computes K_NR @ (K_NR^T @ V) in O(N×r×d).

    Args:
        V: Matrix to multiply, shape (N, d).
        K_NR: Kernel factor from compute_kernel_factor, shape (N, r).

    Returns:
        Result of K_approx @ V, shape (N, d).
    """
    return K_NR @ (K_NR.T @ V)
