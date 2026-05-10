"""Pure-JAX dynamics for FSA-v2 (Banister-coupled) — **G1-reparametrized**.

State [B, F, A]:
  B  fitness     (Banister chronic, Jacobi diffusion in [0, 1])
  F  fatigue     (Banister acute,   CIR diffusion in [0, ∞))
  A  amplitude   (Stuart-Landau,    CIR diffusion in [0, ∞))

## Reparametrization motivation (Stage G1)

The original v2 spec had three parameter pairs that are FIM rank-
deficient at 1-day windows under near-constant Φ and near-constant
A ≈ 0.1:

  (κ_B, ε_A)        — only κ_B(1 + ε_A·A_typ) identifiable
  (μ_F, μ_{FF})     — only μ_F + 2·F_typ·μ_{FF} identifiable
  (τ_F, λ_A)        — only (1 + λ_A·A_typ)/τ_F identifiable

The bridge handoff propagated phantom information on these directions,
causing posterior drift across windows (E3 18/27, E5 closed-loop 3/26).

This module rotates each pair into a (strongly-identified, weakly-
identified-residual) decomposition, keeping the parameter NAMES
unchanged (so downstream control.py and bench drivers don't require
edits) but redefining the meanings + adjusting truth values:

| param name   | NEW meaning                            | NEW truth value |
|--------------|----------------------------------------|-----------------|
| `kappa_B`    | κ_B^eff = κ_B·(1 + ε_A·A_typ)          | 0.012·1.04 = 0.01248 |
| `epsilon_A`  | residual A-boost beyond A_typ          | 0.40 (unchanged) |
| `mu_F`       | μ_F^eff = μ_F + 2·F_typ·μ_{FF} (slope at F_typ) | 0.10 + 0.16 = 0.26 |
| `mu_FF`      | curvature; centered (F − F_typ)²       | 0.40 (unchanged) |
| `mu_0`       | μ_0 + μ_{FF}·F_typ²  (absorbs constant) | 0.02 + 0.016 = 0.036 |
| `tau_F`      | τ_F^eff = τ_F/(1 + λ_A·A_typ)          | 7/1.1 = 6.3636… |
| `lambda_A`   | residual A-coupling beyond A_typ       | 1.00 (unchanged) |

The **drift formulas are mathematically equivalent** to the v2 spec
when the new truth values are used — the reparametrization is a pure
coordinate change. A drift-parity unit test in `tests/test_g1_reparam.py`
verifies this.

The estimation-side priors tighten on `epsilon_A`, `mu_FF`, `lambda_A`
(the residuals) because they are weakly informed at 1-day windows
(see Stage G plan for rationale).

## Drift (frequency-dependent, /day units, reparametrized)

  μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_{FF}·(F − F_typ)²

  dB/dt = κ_B · (1 + ε_A·A) / (1 + ε_A·A_typ) · Φ(t) − B/τ_B
  dF/dt = κ_F·Φ(t) − (1 + λ_A·A) / (1 + λ_A·A_typ) / τ_F · F
  dA/dt = μ·A − η·A³

At A = A_typ and F = F_typ, the residual factors (1+ε_A·A)/(1+ε_A·A_typ)
and (1+λ_A·A)/(1+λ_A·A_typ) both equal 1, so the drift collapses to:

  dB/dt = κ_B·Φ − B/τ_B           ← κ_B is the effective B-gain
  dF/dt = κ_F·Φ − F/τ_F           ← τ_F is the effective acute timescale
  μ = μ_0 + μ_B·B − μ_F·F         ← μ_F is the local slope at F_typ

i.e. at the typical operating point only the strongly-identified
parameters appear in the dynamics.

## Diffusion (state-dependent Itô — unchanged)

  σ_B · √(B (1 − B)) · dW_B   (Jacobi)
  σ_F · √F           · dW_F   (CIR)
  σ_A · √A           · dW_A   (CIR)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Operating-point reference constants ───────────────────────────────
# These define the "linearization point" around which the
# reparametrization is centered. Priors on the residual parameters
# (epsilon_A, mu_FF, lambda_A) encode "we believe the system is
# typically near this point". Real-data extension may relax these.

A_TYP   = 0.10     # initial-state A; representative of de-trained subject
F_TYP   = 0.20     # mid-window F under typical Φ at typical A
PHI_TYP = 1.0      # canonical Banister default (1 unit of TRIMP/day)


# ── Truth parameters (Set A v2, G1-reparametrized) ────────────────────

TRUTH_PARAMS = dict(
    # Banister timescales + (effective) gains
    tau_B=42.0,
    tau_F=7.0 / (1.0 + 1.00 * A_TYP),     # = 6.3636…  ← REPARAMETRIZED
    kappa_B=0.012 * (1.0 + 0.40 * A_TYP),  # = 0.01248 ← REPARAMETRIZED
    kappa_F=0.030,
    epsilon_A=0.40,                         # residual (same number; tighter prior in estimation.py)
    lambda_A=1.00,                          # residual (same number; tighter prior in estimation.py)

    # Stuart-Landau bifurcation parameter
    mu_0=0.02 + 0.40 * (F_TYP ** 2),       # = 0.036 ← absorbs μ_FF·F_typ² constant
    mu_B=0.30,
    mu_F=0.10 + 2.0 * F_TYP * 0.40,        # = 0.26  ← REPARAMETRIZED (slope at F_typ)
    mu_FF=0.40,                             # residual curvature
    eta=0.20,

    # State-dependent diffusion (unchanged from v2)
    sigma_B=0.010,
    sigma_F=0.012,
    sigma_A=0.020,
)


def drift_jax(y, params, Phi_t):
    """JAX drift for the FSA-v2 SDE — G1-reparametrized.

    Args:
        y: state vector [B, F, A] of shape (3,).
        params: dict with the keys listed in TRUTH_PARAMS.
        Phi_t: scalar — training-strain rate at current step.

    Returns:
        d[B, F, A]/dt of shape (3,), in /day units.
    """
    B = y[0]
    F = y[1]
    A = y[2]

    # μ_FF curvature is centered at F_typ so μ_F is the local linear slope
    F_dev = F - F_TYP
    mu = (params['mu_0']
          + params['mu_B'] * B
          - params['mu_F'] * F
          - params['mu_FF'] * F_dev * F_dev)

    # B equation: kappa_B is now κ_B^eff (= effective B-gain at A_typ).
    # The residual factor (1+ε_A·A)/(1+ε_A·A_typ) equals 1 at A=A_typ.
    a_factor_B = ((1.0 + params['epsilon_A'] * A)
                   / (1.0 + params['epsilon_A'] * A_TYP))
    dB = params['kappa_B'] * a_factor_B * Phi_t - B / params['tau_B']

    # F equation: tau_F is now τ_F^eff (= effective acute timescale at A_typ).
    a_factor_F = ((1.0 + params['lambda_A'] * A)
                   / (1.0 + params['lambda_A'] * A_TYP))
    dF = (params['kappa_F'] * Phi_t
          - a_factor_F / params['tau_F'] * F)

    dA = mu * A - params['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector (unchanged from v2)."""
    B = y[0]
    F = y[1]
    A = y[2]
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
    ])


def imex_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent diffusion + boundary
    reflection (unchanged from v2).
    """
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    B_pred = y_pred[0]
    F_pred = y_pred[1]
    A_pred = y_pred[2]

    B_next = jnp.where(B_pred < 0.0, -B_pred,
                        jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
    F_next = jnp.abs(F_pred)
    A_next = jnp.abs(A_pred)

    return jnp.array([B_next, F_next, A_next])
