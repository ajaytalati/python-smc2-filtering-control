"""Regression test for Bug 2 (particle-0 separator template).

History: until 2026-05-06, ``_compute_cost_internals`` in
``control_v5.py`` collapsed the SMC² particle ensemble to particle-0
when computing ``A_sep_per_bin`` — the analytical separatrix between
the healthy and collapsed basins of attraction. The separatrix
depends on the bifurcation parameters mu_0, mu_B, mu_S, mu_F, mu_FF,
eta, B_dec, S_dec, mu_dec_B, mu_dec_S, n_dec — all of which differ
across the SMC² particle population. Using only particle-0's
parameters meant the cost function evaluated chance-constraint
violations under one fixed bifurcation, biasing the controller's
decisions during HMC.

The fix vmaps the separator computation over the particle axis so
each particle gets its own A_sep at every bin.

These tests construct a particle ensemble where particle-0 sits in
the mono-stable HEALTHY regime (A_sep = -inf at the test Phi) and
particle-1 sits in the mono-stable COLLAPSED regime (A_sep = +inf).
Pre-fix: both particles would see the same A_sep (particle-0's, -inf)
→ indicator(A_t < -inf) = 0 for both. Post-fix: each sees its own
A_sep, so particle-1 reports indicator = 1 (always violates).
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math
import numpy as np
import pytest

import jax
jax.config.update('jax_enable_x64', True)


def _make_two_particle_dict():
    """Build a 2-particle theta_stacked where:
      particle 0: TRUTH_PARAMS_V5 (closed-island healthy basin at Phi=(0.30,0.30))
      particle 1: same EXCEPT mu_0 dragged way negative so the system
                  is mono-stable collapsed at the same Phi.

    Picks mu_0 = -10.0 for particle 1 — this dominates every other
    drift contribution at small A and forces ``g(A_min) < 0`` (the
    'is_healthy' check fails) and no sign change inside the bisection
    bracket → A_sep = +inf.
    """
    import jax.numpy as jnp
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5

    base = dict(TRUTH_PARAMS_V5)
    keys = list(base.keys())
    stacked = {k: jnp.array([base[k], base[k]], dtype=jnp.float64) for k in keys}
    # Override mu_0 only on particle-1.
    stacked['mu_0'] = jnp.array([base['mu_0'], -10.0], dtype=jnp.float64)
    return stacked


def test_jax_find_A_sep_differs_per_particle():
    """Sanity check: with diverged mu_0 across particles, the separator
    function returns different values per particle (-inf vs +inf)."""
    import jax
    import jax.numpy as jnp
    from version_3.models.fsa_v5.control_v5 import _jax_find_A_sep

    theta = _make_two_particle_dict()
    Phi_B, Phi_S = 0.30, 0.30   # the v5 healthy island centre

    # Per-particle A_sep at the test Phi.
    def asep_at(params_single):
        return _jax_find_A_sep(jnp.float64(Phi_B), jnp.float64(Phi_S),
                                params_single)
    A_sep_pp = jax.vmap(asep_at)(theta)
    A_sep_pp = np.asarray(A_sep_pp)

    assert np.isneginf(A_sep_pp[0]), (
        f"particle-0 (truth) at Phi=(0.30,0.30) should be mono-stable "
        f"healthy (A_sep=-inf); got {A_sep_pp[0]}"
    )
    assert np.isposinf(A_sep_pp[1]), (
        f"particle-1 (mu_0=-10) at any Phi should be mono-stable "
        f"collapsed (A_sep=+inf); got {A_sep_pp[1]}"
    )


def test_compute_cost_internals_returns_per_particle_A_sep():
    """``_compute_cost_internals`` must return A_sep with shape
    (n_particles, n_steps) — one separator value per particle per bin
    — not (n_steps,) with the same values for every particle."""
    import jax.numpy as jnp
    from version_3.models.fsa_v5.control_v5 import _compute_cost_internals

    theta = _make_two_particle_dict()
    n_steps = 8
    Phi_schedule = jnp.full((n_steps, 2), 0.30, dtype=jnp.float64)
    init_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                            dtype=jnp.float64)
    weights = jnp.array([0.5, 0.5], dtype=jnp.float64)

    effort, A_traj_pp, A_sep_pp = _compute_cost_internals(
        theta, weights, Phi_schedule, init_state, dt=1.0/96)

    A_sep_pp = np.asarray(A_sep_pp)
    assert A_sep_pp.shape == (2, n_steps), (
        f"A_sep_pp must be per-particle per-bin (shape (n_particles, "
        f"n_steps)). Got {A_sep_pp.shape}. If you see (n_steps,) "
        f"only, the particle-0 template bug is back."
    )
    # Particle-0 healthy → all -inf; particle-1 collapsed → all +inf.
    assert np.all(np.isneginf(A_sep_pp[0])), (
        f"particle-0 row should be all -inf, got {A_sep_pp[0]}"
    )
    assert np.all(np.isposinf(A_sep_pp[1])), (
        f"particle-1 row should be all +inf, got {A_sep_pp[1]}"
    )


def test_hard_cost_indicator_uses_per_particle_separator():
    """The hard-variant indicator must respect each particle's own
    separator, not particle-0's.

    Particle-0 (healthy, A_sep = -inf) should never violate.
    Particle-1 (collapsed, A_sep = +inf) should always violate.

    Pre-fix: both particles would share particle-0's -inf separator,
    so both per-particle violation rates would be 0.0.
    Post-fix: vrpp = [0.0, 1.0]; weighted_violation_rate = 0.5.
    """
    import jax.numpy as jnp
    from version_3.models.fsa_v5.control_v5 import _cost_hard_jit

    theta = _make_two_particle_dict()
    n_steps = 16
    Phi_schedule = jnp.full((n_steps, 2), 0.30, dtype=jnp.float64)
    init_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                            dtype=jnp.float64)
    weights = jnp.array([0.5, 0.5], dtype=jnp.float64)

    out = _cost_hard_jit(theta, weights, Phi_schedule, init_state,
                          dt=1.0/96, alpha=0.05, A_target=0.0)
    vrpp = np.asarray(out['violation_rate_per_particle'])
    weighted_vr = float(out['weighted_violation_rate'])

    assert math.isclose(vrpp[0], 0.0, abs_tol=1e-12), (
        f"particle-0 (healthy) should have zero violation rate; got {vrpp[0]}. "
        f"If non-zero, the per-particle indicator isn't using particle-0's "
        f"separator (or the separator computation is wrong)."
    )
    assert math.isclose(vrpp[1], 1.0, abs_tol=1e-12), (
        f"particle-1 (collapsed, A_sep=+inf) should have full violation "
        f"rate; got {vrpp[1]}. If 0.0, particle-1 is being scored against "
        f"particle-0's separator (Bug 2 has regressed)."
    )
    assert math.isclose(weighted_vr, 0.5, abs_tol=1e-12), (
        f"weighted_violation_rate should be 0.5*0.0 + 0.5*1.0 = 0.5; "
        f"got {weighted_vr}."
    )


def test_soft_cost_indicator_uses_per_particle_separator():
    """Same shape check for the soft-variant cost. The sigmoid
    indicator should be ≈0 for particle-0 (healthy) and ≈1 for
    particle-1 (collapsed) at any reasonable beta."""
    import jax.numpy as jnp
    from version_3.models.fsa_v5.control_v5 import _cost_soft_jit

    theta = _make_two_particle_dict()
    n_steps = 16
    Phi_schedule = jnp.full((n_steps, 2), 0.30, dtype=jnp.float64)
    init_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                            dtype=jnp.float64)
    weights = jnp.array([0.5, 0.5], dtype=jnp.float64)

    out = _cost_soft_jit(theta, weights, Phi_schedule, init_state,
                          dt=1.0/96, alpha=0.05, A_target=0.0,
                          beta=50.0, scale=0.1)
    vrpp = np.asarray(out['violation_rate_per_particle'])

    # sigmoid(beta*(-inf - A_t)/scale) = sigmoid(-inf) = 0
    # sigmoid(beta*(+inf - A_t)/scale) = sigmoid(+inf) = 1
    assert vrpp[0] < 1e-9, (
        f"particle-0 soft violation rate should be ~0; got {vrpp[0]}"
    )
    assert vrpp[1] > 1.0 - 1e-9, (
        f"particle-1 soft violation rate should be ~1; got {vrpp[1]}"
    )
