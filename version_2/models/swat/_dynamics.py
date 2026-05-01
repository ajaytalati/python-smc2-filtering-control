"""Pure-JAX dynamics for SWAT (Sleep-Wake-Adenosine-Testosterone).

This is the **4-state control form** of SWAT, ported from the
authoritative Stuart-Landau bifurcation model with the V_h-anabolic
structural fix (upstream Python-Model-Development-Simulation PR #11,
mirrored byte-equivalent in Python-Model-Validation's vendored copy).

Cross-repo consistency to floating-point precision was verified
2026-04-30 — see
``project_upgrade_plans/swat_consistency_check/`` at the repo root.

State vector (4 latents)
------------------------
    y[0] = W       wakefulness                in [0, 1]
    y[1] = Z       sleep depth (rescaled)     in [0, A_scale = 6]
    y[2] = a       adenosine                  >= 0
    y[3] = T       testosterone amplitude     >= 0

Control vector (3 exogenous inputs the MPC manipulates)
--------------------------------------------------------
    u[0] = V_h     vitality reserve           dimensionless
    u[1] = V_n     chronic load               dimensionless, >= 0
    u[2] = V_c     phase shift                hours, in [-12, 12]

Time convention
---------------
``t`` is in DAYS throughout. The circadian formula is
``sin(2*pi*t + phi_0)`` with period 1 day. V_c (in hours) is
converted to a fractional-day phase shift inside the drift.

This matches the FSA-v2 model's day-based time unit so the same
SMC²-MPC bench infrastructure can drive both models without unit
conversions.

Drift summary
-------------
    dW/dt = (sigmoid(u_W)        - W) / tau_W
    dZ/dt = (A_scale*sigmoid(u_Z) - Z) / tau_Z
    da/dt = (W                    - a) / tau_a
    dT/dt = (mu(E)*T - eta*T^3)       / tau_T

with ``u_W = lambda*C_eff(t, V_c) + V_n - a - kappa*Z + alpha_T*T``
and ``u_Z = -gamma_3*W - V_n + beta_Z*a``.

The Stuart-Landau bifurcation parameter is

    mu(E) = mu_0 + mu_E * E

where E is the entrainment quality (V_h-anabolic, V_n-catabolic,
phase-clamped) — see ``entrainment_quality`` below.

Diffusion is state-independent and diagonal (constant per
component). The framework's existing FSA-v2 plant integrator
(version_2/models/fsa_high_res/_plant.py) accepts a state-dependent
diffusion vector via ``sigma(y, params)``; SWAT just returns
constants.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# ── Operating-point reference constants ───────────────────────────────
# These define the canonical morning chronotype and the rescaled
# sleep-depth domain. Frozen across all SWAT scenarios (per upstream).

A_SCALE_FROZEN = 6.0                  # Z domain upper bound
PHI_0_FROZEN   = -math.pi / 3.0       # circadian baseline phase
V_C_MAX_HOURS  = 3.0                  # clinical pathology threshold (hours)


# ── Truth parameters (Set A: healthy baseline) ────────────────────────
#
# Mirrors Python-Model-Development-Simulation's PARAM_SET_A as of
# 2026-04-26 (post V_h-inversion fix), with all timescales in days
# (the framework's native unit). 23 estimable + 4 frozen-at-truth
# values used for synthesis. The estimable subset sits in
# estimation.py:PARAM_PRIOR_CONFIG.

_HOURS_PER_DAY = 24.0

TRUTH_PARAMS = dict(
    # Sigmoid couplings (block F — fast subsystem)
    kappa=6.67,
    lmbda=32.0,
    gamma_3=8.0,
    beta_Z=4.0,

    # Timescales (days)
    tau_W=2.0  / _HOURS_PER_DAY,        # 0.0833 d
    tau_Z=2.0  / _HOURS_PER_DAY,        # 0.0833 d
    tau_a=3.0  / _HOURS_PER_DAY,        # 0.125  d
    tau_T=48.0 / _HOURS_PER_DAY,        # 2.0    d

    # Stuart-Landau testosterone (block T)
    mu_0=-0.5,
    mu_E=1.0,
    eta=0.5,
    alpha_T=0.3,

    # Diffusion temperatures (per day)
    T_W=0.01   * _HOURS_PER_DAY,        # 0.24    /d
    T_Z=0.05   * _HOURS_PER_DAY,        # 1.20    /d
    T_a=0.01   * _HOURS_PER_DAY,        # 0.24    /d
    T_T=0.0001 * _HOURS_PER_DAY,        # 0.0024  /d

    # V_h-anabolic structural fix (upstream PR #11)
    lambda_amp_W=5.0,
    lambda_amp_Z=8.0,
    V_n_scale=2.0,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))


def _circadian(t_days, V_c_hours, phi_0=PHI_0_FROZEN):
    """C_eff(t) = sin(2*pi*(t - V_c/24) + phi_0).

    With t in days and V_c in hours, V_c/24 turns the phase shift
    into the day fraction the circadian formula expects.
    """
    return jnp.sin(2.0 * jnp.pi * (t_days - V_c_hours / _HOURS_PER_DAY)
                    + phi_0)


def entrainment_quality(W, Z, a, T, V_h, V_n, V_c, params):
    """Entrainment quality E in [0, 1] — V_h-anabolic, V_n-catabolic.

    E = damp(V_n) * amp_W * amp_Z * phase(V_c)

    where:
        A_W   = lambda_amp_W * V_h          (V_h-driven amplitude)
        A_Z   = lambda_amp_Z * V_h
        B_W   = V_n - a + alpha_T * T       (slow backdrop, no V_h)
        B_Z   = -V_n + beta_Z * a
        amp_W = sigma(B_W + A_W) - sigma(B_W - A_W)
        amp_Z = sigma(B_Z + A_Z) - sigma(B_Z - A_Z)
        damp  = exp(-V_n / V_n_scale)       (chronic-load damper)
        phase = cos(pi * min(|V_c|, V_c_max) / (2 V_c_max))   (=0 past V_c_max)

    Replaces the pre-fix ``4*sigma(mu)(1-sigma(mu))`` form.

    Args:
        W, Z, a, T: latent state components (scalar or matching shape).
        V_h, V_n, V_c: control values (V_c in hours).
        params: dict with entries ``alpha_T``, ``beta_Z``,
            ``lambda_amp_W``, ``lambda_amp_Z``, ``V_n_scale``.

    Returns:
        Scalar in [0, 1].
    """
    alpha_T = params['alpha_T']
    beta_Z = params['beta_Z']
    lam_amp_W = params['lambda_amp_W']
    lam_amp_Z = params['lambda_amp_Z']
    V_n_scale = params['V_n_scale']

    A_W = lam_amp_W * V_h
    A_Z = lam_amp_Z * V_h
    B_W = V_n - a + alpha_T * T
    B_Z = -V_n + beta_Z * a
    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)

    damp = jnp.exp(-V_n / V_n_scale)

    V_c_eff = jnp.minimum(jnp.abs(V_c), V_C_MAX_HOURS)
    phase = jnp.cos(jnp.pi * V_c_eff / (2.0 * V_C_MAX_HOURS))

    return damp * amp_W * amp_Z * phase


# ── Drift + diffusion ─────────────────────────────────────────────────

def drift_jax(y, params, t, u):
    """SWAT drift (4-state, JAX).

    Args:
        y:      state vector (W, Z, a, T) of shape (4,).
        params: dict matching keys in TRUTH_PARAMS.
        t:      scalar time in days.
        u:      control vector (V_h, V_n, V_c) of shape (3,).

    Returns:
        dy/dt of shape (4,) in /day units.
    """
    W = y[0]
    Z = y[1]
    a = y[2]
    T = y[3]
    V_h = u[0]
    V_n = u[1]
    V_c = u[2]

    kappa = params['kappa']
    lmbda = params['lmbda']
    gamma_3 = params['gamma_3']
    beta_Z = params['beta_Z']
    A_scale = A_SCALE_FROZEN
    tau_W = params['tau_W']
    tau_Z = params['tau_Z']
    tau_a = params['tau_a']
    tau_T = params['tau_T']
    mu_0 = params['mu_0']
    mu_E = params['mu_E']
    eta = params['eta']
    alpha_T = params['alpha_T']

    C_eff = _circadian(t, V_c, PHI_0_FROZEN)
    u_W = lmbda * C_eff + V_n - a - kappa * Z + alpha_T * T
    u_Z = -gamma_3 * W - V_n + beta_Z * a

    # Jacobi-style: Z and a now bounded in [0, 1] (was [0, A_scale=6]
    # and [0, ∞) respectively). Z saturates at sigmoid max = 1; a is
    # a low-pass of W ∈ [0,1] so naturally bounded by [0,1].
    # Diffusion (in diffusion_state_dep) is now Jacobi sqrt(x*(1-x))
    # to vanish at the boundaries, mirroring W and FSA-v2's B.
    del A_scale  # no longer used in drift
    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (_sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    E_dyn = entrainment_quality(W, Z, a, T, V_h, V_n, V_c, params)
    mu = mu_0 + mu_E * E_dyn
    dT = (mu * T - eta * T * T * T) / tau_T

    return jnp.array([dW, dZ, da, dT])


def diffusion_state_dep(y, params):
    """SWAT diagonal diffusion — Jacobi for W, Z, a + sqrt-CIR for T.

    The W, Z, a states are now bounded in [0,1] with Jacobi diffusion
    (vanishes at boundaries), matching FSA-v2's pattern for B. T uses
    sqrt-CIR (vanishes only at lower boundary).

    Args:
        y:      state (W, Z, a, T). Bounded states use Jacobi.
        params: dict with T_W, T_Z, T_a, T_T (noise temperatures).

    Returns:
        Diagonal noise vector of shape (4,):
            σ_W √(W(1-W)), σ_Z √(Z(1-Z)), σ_a √(a(1-a)), σ_T √T
        where σ_i = √(2·T_i).
    """
    W = y[0]
    Z = y[1]
    a = y[2]
    T = y[3]
    sigma_W = jnp.sqrt(2.0 * params['T_W'])
    sigma_Z = jnp.sqrt(2.0 * params['T_Z'])
    sigma_a = jnp.sqrt(2.0 * params['T_a'])
    sigma_T = jnp.sqrt(2.0 * params['T_T'])
    return jnp.array([
        sigma_W * jnp.sqrt(jnp.maximum(W * (1.0 - W), 0.0)),
        sigma_Z * jnp.sqrt(jnp.maximum(Z * (1.0 - Z), 0.0)),
        sigma_a * jnp.sqrt(jnp.maximum(a * (1.0 - a), 0.0)),
        sigma_T * jnp.sqrt(jnp.maximum(T, 0.0)),
    ])


# ── Boundary projection (after each Euler-Maruyama step) ──────────────

def state_clip(y):
    """Project the SWAT latent state back into the physical domain.

    The Euler-Maruyama discretisation can push states slightly
    outside their valid ranges. After each step:

        W in [0, 1]
        Z in [0, A_scale]
        a >= 0
        T >= 0

    Args:
        y: state of shape (4,).

    Returns:
        Clipped state of shape (4,).
    """
    return jnp.array([
        jnp.clip(y[0], 0.0, 1.0),       # W ∈ [0, 1]
        jnp.clip(y[1], 0.0, 1.0),       # Z ∈ [0, 1] (was [0, A_scale])
        jnp.clip(y[2], 0.0, 1.0),       # a ∈ [0, 1] (was [0, ∞))
        jnp.maximum(y[3], 0.0),         # T ∈ [0, ∞) (Stuart-Landau)
    ])


def imex_step_substepped(y, params, noise, t, u, dt, n_substeps: int = 4):
    """Substepped Euler-Maruyama with state-dependent boundary clip.

    Mirrors FSA-v2's `imex_step_substepped` so the bench can use a
    single integrator interface across both models.

    Args:
        y:          state (4,).
        params:     dict.
        noise:      shape (4,) standard normal.
        t:          scalar time (days).
        u:          control (3,).
        dt:         outer step (days).
        n_substeps: deterministic substeps for stiffness control.

    Returns:
        Next state (4,).
    """
    sub_dt = dt / float(n_substeps)

    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, params, t, u), None

    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))
    sigma_y = diffusion_state_dep(y_det, params)
    y_pred = y_det + sigma_y * jnp.sqrt(dt) * noise

    return state_clip(y_pred)
