"""Stage G1 verification: drift-parity tests for the reparametrized v2 model.

The reparametrization is a coordinate change — drift_jax(y, truth_repram, Φ)
must equal v1's drift formulation `drift_jax(y, truth_v2_orig, Φ)` for any
state y and Φ. We assert this to ≤ 1e-10 fp precision.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np
import pytest


# Original v2 truth (BEFORE reparametrization) for parity check
ORIGINAL_V2_TRUTH = dict(
    tau_B=42.0,
    tau_F=7.0,
    kappa_B=0.012,
    kappa_F=0.030,
    epsilon_A=0.40,
    lambda_A=1.00,
    mu_0=0.02,
    mu_B=0.30,
    mu_F=0.10,
    mu_FF=0.40,
    eta=0.20,
    sigma_B=0.010,
    sigma_F=0.012,
    sigma_A=0.020,
)


def drift_v2_original(y, params, Phi_t):
    """The ORIGINAL v2 drift formulation (BEFORE G1 reparametrization).

    Reproduced inline for the parity test. Once parity is verified, this
    function isn't used elsewhere — it's just the reference.
    """
    B, F, A = y[0], y[1], y[2]
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


def test_drift_parity_at_typical_state():
    """At (B, F, A) = (0.05, 0.30, 0.10), Φ=1.0: reparam drift = v2 drift."""
    from models.fsa_high_res._dynamics import drift_jax, TRUTH_PARAMS
    y = jnp.array([0.05, 0.30, 0.10])
    Phi_t = 1.0
    drift_repram = drift_jax(y, TRUTH_PARAMS, Phi_t)
    drift_orig = drift_v2_original(y, ORIGINAL_V2_TRUTH, Phi_t)
    np.testing.assert_allclose(np.asarray(drift_repram),
                                  np.asarray(drift_orig),
                                  atol=1e-10, rtol=1e-10)


def test_drift_parity_grid():
    """Drift parity across a grid of (B, F, A, Φ) values.

    The reparametrization is a coordinate change so should hold globally,
    not just at the typical point.
    """
    from models.fsa_high_res._dynamics import drift_jax, TRUTH_PARAMS
    rng = np.random.default_rng(0)
    for _ in range(50):
        B = rng.uniform(0.05, 0.95)
        F = rng.uniform(0.05, 0.50)
        A = rng.uniform(0.05, 1.20)
        Phi = rng.uniform(0.0, 3.0)
        y = jnp.array([B, F, A])
        d_rep  = drift_jax(y, TRUTH_PARAMS, Phi)
        d_orig = drift_v2_original(y, ORIGINAL_V2_TRUTH, Phi)
        np.testing.assert_allclose(np.asarray(d_rep), np.asarray(d_orig),
                                      atol=1e-10, rtol=1e-10,
                                      err_msg=f"diverged at y={y}, Phi={Phi}")


def test_truth_param_values_match_derivation():
    """The reparametrized truth values match the closed-form derivation."""
    from models.fsa_high_res._dynamics import TRUTH_PARAMS, A_TYP, F_TYP

    # κ_B^eff = κ_B · (1 + ε_A · A_typ) = 0.012 · 1.04 = 0.01248
    assert abs(TRUTH_PARAMS['kappa_B'] - 0.01248) < 1e-10

    # τ_F^eff = τ_F / (1 + λ_A · A_typ) = 7 / 1.1 ≈ 6.3636…
    assert abs(TRUTH_PARAMS['tau_F'] - 7.0 / 1.1) < 1e-10

    # μ_F^eff = μ_F + 2·F_typ·μ_{FF} = 0.10 + 2·0.20·0.40 = 0.26
    assert abs(TRUTH_PARAMS['mu_F'] - 0.26) < 1e-10

    # μ_0^eff = μ_0 + μ_{FF} · F_typ² = 0.02 + 0.40·0.04 = 0.036
    assert abs(TRUTH_PARAMS['mu_0'] - 0.036) < 1e-10

    # Unchanged
    assert abs(TRUTH_PARAMS['epsilon_A'] - 0.40) < 1e-10
    assert abs(TRUTH_PARAMS['lambda_A'] - 1.00) < 1e-10
    assert abs(TRUTH_PARAMS['mu_FF'] - 0.40) < 1e-10
    assert abs(TRUTH_PARAMS['tau_B'] - 42.0) < 1e-10


def test_reparametrization_isolates_residual_at_typical_point():
    """At (A=A_typ, F=F_typ), residual params should drop out of drift —
    only the strongly-identified params (kappa_B, tau_F, mu_F, etc.) appear.
    """
    from models.fsa_high_res._dynamics import drift_jax, TRUTH_PARAMS, A_TYP, F_TYP

    y_typ = jnp.array([0.5, F_TYP, A_TYP])
    Phi_t = 1.0

    # Perturb residuals: epsilon_A, mu_FF, lambda_A
    truth_perturbed = dict(TRUTH_PARAMS)
    truth_perturbed['epsilon_A'] = 0.0     # was 0.40
    truth_perturbed['mu_FF']     = 0.0     # was 0.40
    truth_perturbed['lambda_A']  = 0.0     # was 1.00

    drift_typ = drift_jax(y_typ, TRUTH_PARAMS, Phi_t)
    drift_pert = drift_jax(y_typ, truth_perturbed, Phi_t)

    # B equation: a_factor_B(A=A_typ) = (1+ε_A·A_typ)/(1+ε_A·A_typ) = 1 regardless of ε_A
    # F equation: a_factor_F(A=A_typ) = (1+λ_A·A_typ)/(1+λ_A·A_typ) = 1 regardless of λ_A
    # μ: μ_FF·(F-F_typ)² = 0 at F=F_typ regardless of μ_FF
    # → drift should be unchanged when residuals are perturbed at the typical point
    np.testing.assert_allclose(np.asarray(drift_typ),
                                  np.asarray(drift_pert),
                                  atol=1e-10, rtol=1e-10,
                                  err_msg="residual params should drop out at typical point")
