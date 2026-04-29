"""Pure-JAX dynamics for the FSA high-res model — v2 (Banister-coupled).

State [B, F, A]:
  B  fitness     (Banister chronic, Jacobi diffusion in [0, 1])
  F  fatigue     (Banister acute,   CIR diffusion in [0, ∞))
  A  amplitude   (Stuart-Landau,    CIR diffusion in [0, ∞))

Drift (frequency-dependent, /day units):

  μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F²       (Stuart-Landau bifurcation parameter)
  dB/dt = κ_B·(1 + ε_A·A)·Φ(t)  −  B / τ_B          (Banister chronic, A boosts gain)
  dF/dt = κ_F·Φ(t)             −  (1 + λ_A·A)/τ_F · F   (Banister acute, A speeds clearance)
  dA/dt = μ·A − η·A³                                (Stuart-Landau cubic)

Diffusion (state-dependent Itô):

  σ_B · √(B (1 − B)) · dW_B   (Jacobi — vanishes at both 0 and 1)
  σ_F · √F           · dW_F   (CIR     — vanishes at 0)
  σ_A · √A           · dW_A   (CIR     — vanishes at 0)

The single exogenous control input is Φ(t) (training-strain rate, ≥ 0).
Fitness B and fatigue F are BOTH driven by the same training stimulus Φ
(canonical Banister) — fitness accrues slowly with τ_B = 42 days,
fatigue accrues fast and decays fast with τ_F = 7 days. Autonomic /
circadian amplitude A modulates both adaptation efficiency (B-gain
multiplier 1+ε_A·A) and recovery rate (F-clearance multiplier 1+λ_A·A).

This replaces the v1 model where T_B(t) was an exogenous "fitness
target" the body magically converged toward independent of training —
which made the optimum trivially "rest with high target" and so was
non-physiological. v2 couples B to Φ explicitly so the control problem
becomes the real Banister periodisation trade-off.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Truth parameters (Set A v2) ───────────────────────────────────────

TRUTH_PARAMS = dict(
    # Banister timescales + gains
    tau_B=42.0,        # days — canonical chronic time constant
    tau_F=7.0,         # days — canonical acute time constant
    kappa_B=0.012,     # B-gain per unit Φ → B_ss ≈ 0.5 at Φ=1.0
    kappa_F=0.030,     # F-gain per unit Φ → F_ss ≈ 0.2 at Φ=1.0

    # A-coupling: autonomic state modulates both adaptation + recovery
    epsilon_A=0.40,    # A boosts B-gain (≤40% at A=1)
    lambda_A=1.00,     # A doubles F-clearance rate at A=1

    # Stuart-Landau bifurcation parameter
    mu_0=0.02,         # baseline (weakly subcritical)
    mu_B=0.30,         # fitness raises μ
    mu_F=0.10,         # fatigue suppresses μ (linear)
    mu_FF=0.40,        # fatigue suppresses μ (quadratic — overtraining collapse)
    eta=0.20,          # cubic damping; A* ≈ 1.0 at μ ≈ 0.20

    # State-dependent diffusion (sqrt-Itô).
    # σ values scaled so the noise level at the typical operating point
    # matches the v1 constant-σ values.
    sigma_B=0.010,     # √(B(1-B)) Jacobi
    sigma_F=0.012,     # √F CIR
    sigma_A=0.020,     # √A CIR
)


def drift_jax(y, params, Phi_t):
    """JAX drift for the FSA-v2 SDE.

    Args:
        y: state vector [B, F, A] of shape (3,).
        params: dict with the keys listed in TRUTH_PARAMS (or a JAX-pytree
            equivalent; only the dynamics keys are read here).
        Phi_t: scalar — training-strain schedule value at the current step.

    Returns:
        d[B, F, A]/dt as a (3,) array (in /day units).
    """
    B = y[0]
    F = y[1]
    A = y[2]

    mu = (params['mu_0']
          + params['mu_B'] * B
          - params['mu_F'] * F
          - params['mu_FF'] * F * F)

    dB = params['kappa_B'] * (1.0 + params['epsilon_A'] * A) * Phi_t \
         - B / params['tau_B']
    dF = params['kappa_F'] * Phi_t \
         - (1.0 + params['lambda_A'] * A) / params['tau_F'] * F
    dA = mu * A - params['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector.

    Returns
        [σ_B · √(B(1−B)),
         σ_F · √F,
         σ_A · √A]   shape (3,)

    Each component vanishes at its respective domain boundary, so the
    SDE keeps each state in its physiological range without clipping.
    """
    B = y[0]
    F = y[1]
    A = y[2]
    # jnp.maximum(_, 0.0) is a numerical safety rail — analytically the
    # arguments to sqrt are non-negative whenever B ∈ [0,1], F ≥ 0, A ≥ 0.
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
    ])


def imex_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion + boundary
    reflection.

    Performs `n_substeps` deterministic drift steps of size `dt/n_substeps`
    (handles the cubic Stuart-Landau term and Banister stiffness), then a
    single Wiener increment of variance `σ(y)²·dt` at the outer-step
    boundary. Boundaries are enforced by reflection rather than clipping;
    because σ(y) → 0 at every boundary, the reflection rarely fires and
    the SDE stays in [0,1] × [0,∞) × [0,∞) analytically.

    Returns y_next of shape (3,).
    """
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    # Reflect at boundaries (B ∈ [0, 1], F ≥ 0, A ≥ 0)
    B_pred = y_pred[0]
    F_pred = y_pred[1]
    A_pred = y_pred[2]

    B_next = jnp.where(B_pred < 0.0, -B_pred,
                        jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
    F_next = jnp.abs(F_pred)
    A_next = jnp.abs(A_pred)

    return jnp.array([B_next, F_next, A_next])
