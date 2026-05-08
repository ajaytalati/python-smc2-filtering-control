"""Pure-JAX dynamics for FSA v1.5 — Banister-coupled, v1 formulas.

Functional rewrite of `version_1_5_Python_JAX/models/fsa_high_res/_dynamics.py`.
Same math, same numerical results; the only change is `params` is now an
immutable `ParamsV1` NamedTuple rather than a Python dict, so field
access is `params.kappa_B` instead of `params['kappa_B']`.

The math (v1 basis):

  μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F²
  dB/dt   = κ_B · (1 + ε_A·A) · Φ(t)  −  B / τ_B
  dF/dt   = κ_F · Φ(t)                −  (1 + λ_A·A) / τ_F · F
  dA/dt   = μ · A − η · A³

  σ_B(B) = σ_B · √(B(1−B))      Jacobi (boundary 0 and 1)
  σ_F(F) = σ_F · √F             CIR    (boundary 0)
  σ_A(A) = σ_A · √A             CIR    (boundary 0)

State y = [B, F, A] (length-3 array). Single exogenous control Φ(t) ≥ 0.

`ParamsV1` is defined here (and re-exported from `simulation.py`) to
break the simulation/dynamics import cycle: `simulation.py` builds
`ParamsV15` (the user-facing basis) which it converts to `ParamsV1` via
`params_v15_to_v1` for `drift_jax` / `diffusion_state_dep`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Typed parameter record (v1 basis) ─────────────────────────────────
class ParamsV1(NamedTuple):
    """14 dynamics params in the v1 basis (kappa_B, kappa_F) + 3 obs sigmas.

    Field types are `float` for documentation, but JAX scalars
    (`jnp.ndarray` of shape `()`) are also accepted at every call site —
    `typing.NamedTuple` does not enforce field types at runtime, and
    being a registered JAX pytree means the record flows naturally
    through `jit` / `scan` / `vmap`.
    """
    tau_B:       float
    tau_F:       float
    kappa_B:     float
    kappa_F:     float
    epsilon_A:   float
    lambda_A:    float
    mu_0:        float
    mu_B:        float
    mu_F:        float
    mu_FF:       float
    eta:         float
    sigma_B:     float
    sigma_F:     float
    sigma_A:     float
    sigma_B_obs: float
    sigma_F_obs: float
    sigma_A_obs: float


# ── Truth parameters in the v1 basis (same numbers as v1.5 Julia
#    `Dynamics.TRUTH_PARAMS` at `_dynamics.jl:26-48`) ──────────────
TRUTH_PARAMS_V1 = ParamsV1(
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

    # Observation noise (same defaults as the dict-based source).
    sigma_B_obs=0.005,
    sigma_F_obs=0.005,
    sigma_A_obs=0.005,
)


def drift_jax(y, params: ParamsV1, Phi_t):
    """Per-day drift d[B, F, A]/dt at state `y` under control `Phi_t`.

    Args:
        y: array_like of shape (3,) — [B, F, A].
        params: `ParamsV1` — v1-basis parameter record.
        Phi_t: scalar — training-strain at current bin.

    Returns:
        jnp.ndarray of shape (3,) in /day units.
    """
    B = y[0]
    F = y[1]
    A = y[2]

    mu = (params.mu_0
          + params.mu_B * B
          - params.mu_F * F
          - params.mu_FF * F * F)

    dB = (params.kappa_B * (1.0 + params.epsilon_A * A) * Phi_t
          - B / params.tau_B)
    dF = (params.kappa_F * Phi_t
          - (1.0 + params.lambda_A * A) / params.tau_F * F)
    dA = mu * A - params.eta * A * A * A

    return jnp.array([dB, dF, dA])


def diffusion_state_dep(y, params: ParamsV1):
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
        params.sigma_B * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params.sigma_F * jnp.sqrt(jnp.maximum(F, 0.0)),
        params.sigma_A * jnp.sqrt(jnp.maximum(A, 0.0)),
    ])


def em_step_substepped(y, params: ParamsV1, noise, Phi_t, dt,
                        n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion + boundary
    reflection. Direct port of Julia's `Dynamics.em_step_substepped` at
    `_dynamics.jl:100-119`.

    Performs `n_substeps` deterministic drift sub-steps of size
    `dt/n_substeps` (handles cubic SL stiffness + Banister timescales),
    then a single Wiener increment of variance `σ(y)²·dt` at the outer
    boundary. Boundaries are enforced by reflection (B → −B if < 0;
    B → 2−B if > 1; F, A → |·|).

    `noise` is a pre-drawn standard-normal triple (shape (3,)).
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
