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

    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (A_scale * _sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    E_dyn = entrainment_quality(W, Z, a, T, V_h, V_n, V_c, params)
    mu = mu_0 + mu_E * E_dyn
    dT = (mu * T - eta * T * T * T) / tau_T

    return jnp.array([dW, dZ, da, dT])


def diffusion_state_dep(y, params):
    """SWAT diagonal diffusion (state-INDEPENDENT, despite the name).

    The name keeps the FSA-v2 convention so the same plant integrator
    can call this function. Returns sqrt(2 * temperature) per
    component, giving Euler-Maruyama steps of the form

        y_{t+dt}[i] = y_t[i] + drift_i * dt + sigma_i * sqrt(dt) * xi_i

    Args:
        y:      state (unused — present for FSA-v2 API parity).
        params: dict with T_W, T_Z, T_a, T_T.

    Returns:
        Diagonal noise vector of shape (4,).
    """
    del y  # diffusion is state-independent in SWAT
    return jnp.array([
        jnp.sqrt(2.0 * params['T_W']),
        jnp.sqrt(2.0 * params['T_Z']),
        jnp.sqrt(2.0 * params['T_a']),
        jnp.sqrt(2.0 * params['T_T']),
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
        jnp.clip(y[0], 0.0, 1.0),
        jnp.clip(y[1], 0.0, A_SCALE_FROZEN),
        jnp.maximum(y[2], 0.0),
        jnp.maximum(y[3], 0.0),
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
