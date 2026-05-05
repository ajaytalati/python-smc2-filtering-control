"""Pure-JAX dynamics for FSA-v5 — Variable Dose + Hill Deconditioning.

This module is the single source of truth for the FSA-v5 SDE drift, the
state-dependent diffusion, and the substepped Euler--Maruyama integrator.
Every other file in ``models/fsa_high_res/`` (and any consumer in the
``smc2fc`` repository) defers to ``drift_jax`` here. That is deliberate:
the drift equations are the model, and duplicating them across files is the
single most reliable way to introduce v4/v5 inconsistencies.

Map of symbols to LaTeX documentation
-------------------------------------
The complete v5 model specification is laid out in §11.1 of
``LaTex_docs/main.tex``. The following table cross-references each symbol:

  ====================  ===============================================
  Symbol                Where it first appears in the doc
  ====================  ===============================================
  B, S, F, A            §1 (physiology), §2 (math) — 4D FSA
  K_FB, K_FS            §4   (FSA-v4 variable-dose extension)
  drift_jax (this file) §11.1 (full equations gathered)
  Hill deconditioning   §10.2, equation (eq:v5-mubar)
  TRUTH_PARAMS_V5       §10.4 (Numerical confirmation table)
  diffusion_state_dep   §11.1 (Diffusion paragraph)
  ====================  ===============================================

State vector layout (6D)
-------------------------
::

    y = [B, S, F, A, K_FB, K_FS]^T

  B    Aerobic fitness                 (Banister chronic, Jacobi diffusion in [0,1])
  S    Strength capacity               (Banister chronic, Jacobi diffusion in [0,1])
  F    Unified fatigue pool            (CIR diffusion in [0, ∞))
  A    Autonomic amplitude             (Stuart-Landau, CIR diffusion in [0, ∞))
  K_FB Aerobic fatigue gain            (Busso variable, CIR diffusion in [0, ∞))
  K_FS Strength fatigue gain           (Busso variable, CIR diffusion in [0, ∞))

Busso Variable-Dose Principle (Busso 2003, FSA-v4)
---------------------------------------------------
The fatigue gains K_FB and K_FS are dynamic latent states, not static
parameters. Training stimulus concurrently builds fitness/strength AND
increases the fatigue sensitivity of the athlete:

    dK_{Fi}/dt = (K_{Fi}^0 - K_{Fi}) / tau_K  +  mu_K * Phi_i(t)

with K_{Fi}^0 the baseline sensitivity and mu_K the 'damage' rate.

FSA-v5 Hill-deconditioning extension
-------------------------------------
v5 adds a one-sided Hill saturating term to the autonomic drive mu(B,S,F)
that fires when chronic capacities fall below a deconditioning threshold:

    mu_v5 = mu_v4(B,S,F) - mu_{B-} * B_dec^n / (B^n + B_dec^n)
                         - mu_{S-} * S_dec^n / (S^n + S_dec^n)

Setting mu_dec_B = mu_dec_S = 0 reduces v5 exactly to v4. ``TRUTH_PARAMS``
defaults to that v4-recovering choice; ``TRUTH_PARAMS_V5`` overrides with
the scanned values that produce the closed-island basin topology
described in §10.

Numerical conventions
---------------------
- All time-scales are in days (tau_B, tau_S, tau_F, tau_K).
- All gains and rates are dimensionless (mu_*, kappa_*, eta).
- Phi_t is a 2-vector [Phi_B, Phi_S] of stimulus *rates* (not amounts).
- Operating point reference values A_TYP, F_TYP set the G1 reparametrisation
  scale (so that the model is centred on a physiologically typical state).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ── Operating-point reference constants ────────────────────────────────────
# These are the "typical" values around which the G1 reparametrisation is
# centred. They appear inside the a_factor_* functions (autonomic feedback
# on B, S and F dynamics) and in the mu_0 / mu_F derivations below.
A_TYP   = 0.10   # typical autonomic amplitude (dimensionless)
F_TYP   = 0.20   # typical unified fatigue (dimensionless)
PHI_TYP = 1.0    # typical stimulus rate (dimensionless)


# ── Truth parameters (FSA-v5-capable, with v4-recovering defaults) ─────────
#
# All scalar parameters of the v5 SDE drift live in this dict. Note that
# the diffusion scales (sigma_*) are listed here for completeness, but in
# the production estimation pipeline they are FROZEN — see
# estimation.py:HIGH_RES_FSA_V5_ESTIMATION.frozen_params.
#
# By default (mu_dec_B = mu_dec_S = 0) this dict reduces v5 to v4
# numerically. This makes back-compat with v4 tests / examples trivial.

TRUTH_PARAMS = dict(
    # ── Aerobic Fitness B (linear in Phi_B, decays with τ_B) ──
    tau_B=42.0,
    kappa_B=0.012 * (1.0 + 0.40 * A_TYP),  # κ_B^eff in G1 form
    epsilon_AB=0.40,                       # autonomic boost coefficient

    # ── Strength Adaptation S ──
    tau_S=60.0,
    kappa_S=0.008 * (1.0 + 0.20 * A_TYP),  # κ_S^eff in G1 form
    epsilon_AS=0.20,

    # ── Unified Fatigue F (driven by K_FB Phi_B + K_FS Phi_S) ──
    tau_F=7.0 / (1.0 + 1.00 * A_TYP),      # τ_F^eff in G1 form
    lambda_A=1.00,                         # autonomic-fatigue coupling

    # ── Busso Variable-Dose Sensitivity K_FB, K_FS (FSA-v4) ──
    KFB_0=0.030,                           # baseline aerobic fatigue gain
    KFS_0=0.050,                           # baseline strength fatigue gain
    tau_K=21.0,                            # ~3 weeks recovery timescale
    mu_K=0.005,                            # 'damage' rate (Busso 2003)

    # ── Stuart-Landau bifurcation parameter mu(B,S,F) ──
    mu_0=0.02 + 0.40 * (F_TYP ** 2),       # baseline autonomic drive
    mu_B=0.30,                             # B → A coupling (positive)
    mu_S=0.15,                             # S → A coupling (positive)
    mu_F=0.10 + 2.0 * F_TYP * 0.40,        # F → A coupling (negative)
    mu_FF=0.40,                            # quadratic F penalty around F_TYP
    eta=0.20,                              # cubic damping in A

    # ── State-dependent diffusion scales (frozen in production) ──
    sigma_B=0.010,
    sigma_S=0.008,
    sigma_F=0.012,
    sigma_A=0.020,
    sigma_K=0.005,                         # shared by K_FB and K_FS

    # ── FSA-v5 Hill deconditioning (silent when mu_dec_* = 0) ──
    # See §10.2, equation (eq:v5-mubar). The penalty saturates at
    # mu_dec_i when B (resp. S) is far below B_dec (resp. S_dec) and
    # vanishes as B (resp. S) >> B_dec. C^∞ smooth — no kinks.
    B_dec=0.10,                            # aerobic-fitness threshold
    S_dec=0.10,                            # strength threshold
    mu_dec_B=0.0,                          # aerobic decond penalty (0 = v4)
    mu_dec_S=0.0,                          # strength decond penalty (0 = v4)
    n_dec=4.0,                             # Hill exponent (steepness)
)


# ── FSA-v5 truth parameters (closed-island basin topology) ─────────────────
#
# Derived from TRUTH_PARAMS, overriding only the v5 deconditioning entries.
# These values were chosen by parameter scan (see §10.3, Figure 7) to
# satisfy four qualitative-realism constraints simultaneously:
#
#   1. Sedentary  Phi=(0,0)        →  mu_bar(0) << 0   (clear collapse)
#   2. Aerobic-only along Phi_S=0  →  mu_bar(0) < 0    (no aerobic-only health)
#   3. Strength-only along Phi_B=0 →  mu_bar(0) < 0    (no strength-only health)
#   4. Balanced moderate (Phi_B≈Phi_S≈0.30)  → mu_bar(0) > 0  (healthy island)
#
# The healthy island is a topological disc strictly interior to the
# (Phi_B, Phi_S) > 0 quadrant — the model encodes that *both* aerobic and
# strength stimuli are required for sustained autonomic health.
#
# These are starting values for an N=1 fit; tune against the subject's
# actual detraining/overload episode timestamps if a richer dataset is
# available.

TRUTH_PARAMS_V5 = dict(TRUTH_PARAMS)
TRUTH_PARAMS_V5.update(
    B_dec=0.07,
    S_dec=0.07,
    mu_dec_B=0.10,
    mu_dec_S=0.10,
    n_dec=4.0,
)


# ===========================================================================
# DRIFT — the full FSA-v5 deterministic part of the SDE
# ===========================================================================

def drift_jax(y, params, Phi_t):
    """JAX drift for the FSA-v5 SDE. Single source of truth.

    Computes the deterministic time-derivatives of all 6 state variables.
    The diffusion (stochastic) part is in ``diffusion_state_dep`` below.

    Args:
        y: state vector ``[B, S, F, A, K_FB, K_FS]`` of shape (6,).
            Expected in physically-bounded ranges; the caller is responsible
            for clipping/reflecting before each call (see _plant.py and
            simulation.py for examples).
        params: dict of scalar parameters. Must contain every key in
            ``TRUTH_PARAMS`` (including the v5 Hill-deconditioning keys
            ``B_dec``, ``S_dec``, ``mu_dec_B``, ``mu_dec_S``, ``n_dec``).
            Setting ``mu_dec_B = mu_dec_S = 0`` recovers exact v4 numerics.
        Phi_t: shape-(2,) vector ``[Phi_B, Phi_S]`` — stimulus rates at the
            current time. Units: dimensionless.

    Returns:
        ``d[B, S, F, A, K_FB, K_FS]/dt`` of shape (6,). Same units as y/day.

    Maps to LaTeX §11.1 equations (B, S, F, K, A blocks) and the v5
    bifurcation parameter (eq:v5-mubar).
    """
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    Phi_B, Phi_S = Phi_t[0], Phi_t[1]

    # ── Bifurcation parameter mu(B, S, F) — FSA-v5 ──────────────────────
    # See §10.2, eq. (eq:v5-mubar). Two contributions:
    #   (a) The classical FSA-v4 form:
    #         mu_v4 = mu_0 + mu_B B + mu_S S - mu_F F - mu_FF (F-F_TYP)^2
    #   (b) The new v5 Hill-deconditioning subtractions:
    #         dec_i = mu_dec_i * x_dec_i^n / (x_i^n + x_dec_i^n)   for i in {B, S}
    # Each Hill term is in [0, mu_dec_i]: at x = 0 it saturates to
    # mu_dec_i, at x = x_dec_i it equals mu_dec_i / 2, at x >> x_dec_i it
    # vanishes. C^∞ smooth — friendly to gradient and SMC inference.
    F_dev = F - F_TYP

    n   = params['n_dec']
    Bn  = jnp.power(jnp.maximum(B, 0.0), n)
    Sn  = jnp.power(jnp.maximum(S, 0.0), n)
    Bdn = jnp.power(params['B_dec'], n)
    Sdn = jnp.power(params['S_dec'], n)
    dec_B = params['mu_dec_B'] * Bdn / (Bn + Bdn)
    dec_S = params['mu_dec_S'] * Sdn / (Sn + Sdn)

    mu = (params['mu_0']
          + params['mu_B'] * B
          + params['mu_S'] * S
          - params['mu_F'] * F
          - params['mu_FF'] * F_dev * F_dev
          - dec_B
          - dec_S)

    # ── Aerobic capacity B ──────────────────────────────────────────────
    # First-order linear; gain modulated by autonomic state via a_factor_B.
    a_factor_B = (1.0 + params['epsilon_AB'] * A) / (1.0 + params['epsilon_AB'] * A_TYP)
    dB = params['kappa_B'] * a_factor_B * Phi_B - B / params['tau_B']

    # ── Strength capacity S ─────────────────────────────────────────────
    # Same structural form as B, with its own time constant and coupling.
    a_factor_S = (1.0 + params['epsilon_AS'] * A) / (1.0 + params['epsilon_AS'] * A_TYP)
    dS = params['kappa_S'] * a_factor_S * Phi_S - S / params['tau_S']

    # ── Unified fatigue pool F (FSA-v4: dynamic gains) ──────────────────
    # F is driven by stimuli weighted by the (slowly-varying) gains
    # K_FB and K_FS. Decay rate is also autonomic-modulated.
    a_factor_F = (1.0 + params['lambda_A'] * A) / (1.0 + params['lambda_A'] * A_TYP)
    dF = (KFB * Phi_B + KFS * Phi_S - a_factor_F / params['tau_F'] * F)

    # ── Autonomic amplitude A (Stuart-Landau) ───────────────────────────
    # Linear growth from mu, cubic damping. Sign of mu = sign of stability
    # of A=0 boundary equilibrium. See §7 stability analysis.
    dA = mu * A - params['eta'] * A * A * A

    # ── Busso variable-dose K dynamics (FSA-v4) ─────────────────────────
    # Linear relaxation toward baseline K_{Fi}^0 plus stimulus damage at
    # rate mu_K. K's eq is K_{Fi}^* = K_{Fi}^0 + tau_K * mu_K * Phi_i.
    dKFB = (params['KFB_0'] - KFB) / params['tau_K'] + params['mu_K'] * Phi_B
    dKFS = (params['KFS_0'] - KFS) / params['tau_K'] + params['mu_K'] * Phi_S

    return jnp.array([dB, dS, dF, dA, dKFB, dKFS])


# ===========================================================================
# DIFFUSION — state-dependent scales for each component
# ===========================================================================

def diffusion_state_dep(y, params):
    """State-dependent diagonal diffusion vector (6D).

    Returns the per-component diffusion magnitudes ``sigma_i(y)`` such that
    the SDE is ``dy = drift dt + sigma(y) * dW``. Each component has a
    physically-motivated state-dependence:

      B, S    Jacobi-style:  sigma_i * sqrt(x_i (1 - x_i))   keeps x_i in [0,1]
      F, A, K CIR-style:     sigma_i * sqrt(x_i)             keeps x_i ≥ 0

    The K-block uses a single shared diffusion scale ``sigma_K`` for both
    K_FB and K_FS (justified empirically — K dynamics are slow and the
    SDE-noise contribution is small relative to the drift).
    """
    B, S, F, A, KFB, KFS = y[0], y[1], y[2], y[3], y[4], y[5]
    return jnp.array([
        params['sigma_B'] * jnp.sqrt(jnp.maximum(B * (1.0 - B), 0.0)),
        params['sigma_S'] * jnp.sqrt(jnp.maximum(S * (1.0 - S), 0.0)),
        params['sigma_F'] * jnp.sqrt(jnp.maximum(F, 0.0)),
        params['sigma_A'] * jnp.sqrt(jnp.maximum(A, 0.0)),
        params['sigma_K'] * jnp.sqrt(jnp.maximum(KFB, 0.0)),
        params['sigma_K'] * jnp.sqrt(jnp.maximum(KFS, 0.0)),
    ])


# ===========================================================================
# SUB-STEPPED EULER–MARUYAMA — for tools that need it (not the production path)
# ===========================================================================

def imex_step_substepped(y, params, noise, Phi_t, dt, n_substeps: int = 4):
    """Sub-stepped Euler--Maruyama with state-dependent diffusion (6D).

    NOTE: the production estimation/plant pipeline uses ONE Euler step per
    bin (no sub-stepping; see _plant.py:_plant_em_step), matching
    FSA_STEP_MINUTES = 15 → dt = 1/96 day. This sub-stepped helper exists
    for tools that drive the SDE with coarser dt (e.g., basin-sweep tools
    in tools/stability_basins_v4.py).

    Performs ``n_substeps`` deterministic Euler steps for the drift, then
    applies the diffusion contribution once at the post-substepping state.
    Reflection boundary applied at the end so all components remain in
    their physical ranges.
    """
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    # Reflection-into-physical-domain. B, S in [0,1]; F, A, K_* >= 0.
    B_p, S_p, F_p, A_p, KFB_p, KFS_p = (
        y_pred[0], y_pred[1], y_pred[2], y_pred[3], y_pred[4], y_pred[5])

    B_next = jnp.where(B_p < 0.0, -B_p, jnp.where(B_p > 1.0, 2.0 - B_p, B_p))
    S_next = jnp.where(S_p < 0.0, -S_p, jnp.where(S_p > 1.0, 2.0 - S_p, S_p))
    F_next = jnp.abs(F_p)
    A_next = jnp.abs(A_p)
    KFB_next = jnp.abs(KFB_p)
    KFS_next = jnp.abs(KFS_p)

    return jnp.array([B_next, S_next, F_next, A_next, KFB_next, KFS_next])
