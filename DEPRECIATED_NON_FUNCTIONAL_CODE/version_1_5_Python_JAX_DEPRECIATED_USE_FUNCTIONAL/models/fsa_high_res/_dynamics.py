"""Pure-JAX dynamics for FSA v1.5 — Banister-coupled, v1 formulas.

Direct port of `version_1_5_Julia/models/fsa_high_res/_dynamics.jl`,
which is itself the v1 SDE (verbatim copy). The math:

  μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F²
  dB/dt   = κ_B · (1 + ε_A·A) · Φ(t)  −  B / τ_B
  dF/dt   = κ_F · Φ(t)                −  (1 + λ_A·A) / τ_F · F
  dA/dt   = μ · A − η · A³

  σ_B(B) = σ_B · √(B(1−B))      Jacobi (boundary 0 and 1)
  σ_F(F) = σ_F · √F             CIR    (boundary 0)
  σ_A(A) = σ_A · √A             CIR    (boundary 0)

State y = [B, F, A] (length-3 array). Single exogenous control Φ(t) ≥ 0.

NOT the v2 G1-reparametrized form — v1.5's parameter basis is
re-rotated separately (see `simulation.py:params_v15_to_v1`), but the
drift formulas at the v1 basis are identical to v1's.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Truth parameters in the v1 basis (same numbers as v1.5 Julia
#    `Dynamics.TRUTH_PARAMS` at `_dynamics.jl:26-48`) ──────────────
TRUTH_PARAMS = dict(
    # Banister timescales + gains
    tau_B=42.0,
    tau_F=7.0,
    kappa_B=0.012,
    kappa_F=0.030,

    # A-coupling
    epsilon_A=0.40,
    lambda_A=1.00,

    # Stuart-Landau bifurcation parameter
    mu_0=0.02,
    mu_B=0.30,
    mu_F=0.10,
    mu_FF=0.40,
    eta=0.20,

    # State-dependent diffusion
    sigma_B=0.010,
    sigma_F=0.012,
    sigma_A=0.020,
)


def drift_jax(y, params, Phi_t):
    """Per-day drift d[B, F, A]/dt at state `y` under control `Phi_t`.

    Args:
        y: array_like of shape (3,) — [B, F, A].
        params: dict with v1-form keys (`kappa_B`, `kappa_F`, …).
        Phi_t: scalar — training-strain at current bin.

    Returns:
        jnp.ndarray of shape (3,) in /day units.
    """
    B = y[0]
    F = y[1]
    A = y[2]

    mu = (params['mu_0']
          + params['mu_B'] * B
          - params['mu_F'] * F
          - params['mu_FF'] * F * F)

    dB = (params['kappa_B'] * (1.0 + params['epsilon_A'] * A) * Phi_t
          - B / params['tau_B'])
    dF = (params['kappa_F'] * Phi_t
          - (1.0 + params['lambda_A'] * A) / params['tau_F'] * F)
    dA = mu * A - params['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector. Each component vanishes
    at its respective domain boundary so the SDE keeps each state in its
    physiological range without clipping.

    Returns
        jnp.ndarray of shape (3,) — [σ_B √(B(1−B)), σ_F √F, σ_A √A].
    """
    B = y[0]
    F = y[1]
    A = y[2]
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
    ])


def em_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion + boundary
    reflection. Direct port of Julia's `Dynamics.em_step_substepped` at
    `_dynamics.jl:100-119`.

    Performs `n_substeps` deterministic drift sub-steps of size
    `dt/n_substeps` (handles cubic SL stiffness + Banister timescales),
    then a single Wiener increment of variance `σ(y)²·dt` at the outer
    boundary. Boundaries are enforced by reflection (B → −B if < 0;
    B → 2−B if > 1; F, A → |·|).

    `noise` is a pre-drawn standard-normal triple (shape (3,)). The
    diff-test driver supplies it; both impls bit-identical for the same
    noise.

    Returns y_next of shape (3,).
    """
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
    B_next = jnp.where(B_pred < 0.0, -B_pred,
                       jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
    F_next = jnp.abs(F_pred)
    A_next = jnp.abs(A_pred)

    return jnp.array([B_next, F_next, A_next])
