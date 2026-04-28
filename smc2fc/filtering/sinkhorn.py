"""Low-rank Sinkhorn iterations in scaling form.

Finds scaling vectors (u, v) such that diag(u) K_approx diag(v)
has prescribed row sums a and column sums b, where K_approx is
the Nyström kernel factorisation from kernel.py.

All operations are O(N×r) per iteration (no N×N matrices).
Fixed iteration count for JIT/fori_loop compatibility.

Date:    15 April 2026
Version: 4.1 — fori_loop replaces scan (XLA compile-time fix)

Change history:
  v4.1 (18 Apr 2026) — replace ``jax.lax.scan`` with
        ``jax.lax.fori_loop`` for the Sinkhorn iterations.

        Root cause of XLA compile hang (gk_dpf_hybrid):
        The outer particle-filter scan (T=500) calls sinkhorn_scalings
        inside a jax.lax.cond branch.  XLA compiles BOTH branches of
        every cond.  With scan(n_iter=10) inside, XLA unrolls 10
        Sinkhorn steps at each of the 500 particle-filter positions,
        producing a flat computation graph of 5000 Sinkhorn bodies that
        the XLA kernel-fusion pass must jointly optimise.  Vmapped over
        16 chains this becomes intractable (30+ minute compile times).

        jax.lax.fori_loop compiles to a real loop instruction in XLA
        bytecode, NOT an unrolled sequence.  XLA only traces the loop
        body ONCE regardless of n_iter.  The outer scan sees "one opaque
        loop object" instead of "10 unrolled Sinkhorn steps", reducing
        the compilation graph by ~10× for the Sinkhorn part.

        Note: fori_loop is fully differentiable in JAX via gradient
        checkpointing, but the hybrid filter wraps ot_resample_lr in
        jax.lax.stop_gradient so Sinkhorn AD is never needed in that
        context anyway.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Tuple

from smc2fc.filtering.transport_kernel import factor_matvec


def sinkhorn_scalings(a: Array, b: Array, K_NR: Array,
                      n_iter: int = 10) -> Tuple[Array, Array]:
    """Compute Sinkhorn scaling vectors via fixed-point iteration.

    Alternates between row and column normalisation of the approximate
    kernel, converging to scalings (u, v) such that:
        diag(u) @ K_approx @ diag(v) has marginals (a, b).

    Uses the factored kernel K_approx = K_NR @ K_NR^T throughout,
    so each iteration costs O(N×r) instead of O(N²).

    Args:
        a: Source marginal (typically uniform = 1/N), shape (N,).
            Must be positive and sum to 1.
        b: Target marginal (normalised particle weights), shape (N,).
            Must be positive and sum to 1.
        K_NR: Kernel factor from kernel.compute_kernel_factor, shape (N, r).
        n_iter: Number of Sinkhorn iterations.  Fixed for JIT compatibility.
            10 iterations is sufficient for MCMC gradient quality in [0,1]^5.

    Returns:
        Tuple (u, v) of scaling vectors, each shape (N,).
        The coupling is π_ij ≈ u_i * K_approx_ij * v_j.
    """
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)

    # Use fori_loop (not scan) to compile to a real XLA loop instruction.
    # K_NR, a, b are captured as constants in the closure; only (u, v)
    # are updated per iteration.  This reduces the outer scan's computation
    # graph from O(T × n_iter) unrolled bodies to O(T × 1) loop objects.
    def _iter_body(_, carry):
        u, v = carry
        Kv = factor_matvec(v, K_NR)
        u_new = a / jnp.maximum(Kv, 1e-30)
        Ku = factor_matvec(u_new, K_NR)
        v_new = b / jnp.maximum(Ku, 1e-30)
        return (u_new, v_new)

    u, v = jax.lax.fori_loop(0, n_iter, _iter_body, (u, v))
    return u, v
