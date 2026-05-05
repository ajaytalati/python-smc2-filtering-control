"""End-to-end smoke test for the FSA-v5 codebase as exposed to ``smc2fc``.

Runs the full v5 stack:
    1. ``StepwisePlant`` forward simulation under v5 truth params.
    2. ``estimation.propagate_fn_v5`` Kalman-fused step over the synthetic data.
    3. ``control_v5.evaluate_chance_constrained_cost`` on a 10-particle cloud.

Pass criterion = "no NaN, all states in physical bounds, log-likelihood
finite, chance-constraint metrics in expected ranges". This is **not** an
inference test — it's a structural check that the v5 promotion of the
folder is internally consistent and that nothing imports broken.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import numpy as np
import jax
import jax.numpy as jnp
from jax import config
config.update('jax_enable_x64', True)


def test_v5_imports_clean():
    """Single-line import of the v5 boundary should not raise."""
    import version_3.models.fsa_v5 as fh
    assert hasattr(fh, 'HIGH_RES_FSA_V5_MODEL')
    assert hasattr(fh, 'HIGH_RES_FSA_V5_ESTIMATION')
    assert hasattr(fh, 'StepwisePlant')
    assert hasattr(fh, 'evaluate_chance_constrained_cost')
    assert hasattr(fh, 'find_A_sep_v5')


def test_v5_plant_forward_pipeline():
    """StepwisePlant 6D forward simulation under v5 truth — sane state ranges."""
    from version_3.models.fsa_v5 import StepwisePlant, DEFAULT_PARAMS_V5

    plant = StepwisePlant(truth_params=dict(DEFAULT_PARAMS_V5))

    # 14 days at moderate Phi=(0.30, 0.30) — inside the v5 healthy island.
    Phi_daily = np.tile([[0.30, 0.30]], (14, 1))
    out = plant.advance(stride_bins=14 * 96, Phi_daily=Phi_daily)
    traj = out['trajectory']

    # 6D, 14*96 bins
    assert traj.shape == (14 * 96, 6), f"unexpected traj shape {traj.shape}"

    # No NaN, no Inf
    assert np.all(np.isfinite(traj)), "NaN/Inf in plant trajectory"

    # Physical bounds
    assert np.all(traj[:, 0] >= 0.0) and np.all(traj[:, 0] <= 1.0), "B out of [0,1]"
    assert np.all(traj[:, 1] >= 0.0) and np.all(traj[:, 1] <= 1.0), "S out of [0,1]"
    assert np.all(traj[:, 2] >= 0.0), "F < 0"
    assert np.all(traj[:, 3] >= 0.0), "A < 0"
    assert np.all(traj[:, 4] >= 0.0), "K_FB < 0"
    assert np.all(traj[:, 5] >= 0.0), "K_FS < 0"

    # Inside the v5 island under moderate Phi, A should stay > 0 — not collapse.
    assert traj[-1, 3] > 0.01, (
        f"A collapsed to {traj[-1,3]} under moderate Phi (expected to stay healthy)")

    # All five obs channels present
    for ch in ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps', 'obs_volumeload'):
        assert ch in out, f"missing obs channel {ch}"
        assert 't_idx' in out[ch], f"obs channel {ch} missing t_idx"


def test_v5_estimation_propagate_fn_runs():
    """estimation.propagate_fn_v5 evaluates without NaN on a single bin."""
    from version_3.models.fsa_v5 import HIGH_RES_FSA_V5_ESTIMATION
    from version_3.models.fsa_v5.estimation import (
        propagate_fn_v5, get_init_theta, INIT_STATE_PRIOR_CONFIG,
    )

    # Initial parameter vector at the prior mode
    theta = jnp.asarray(get_init_theta())
    n_steps = 96
    # Build a minimal grid_obs dict that propagate_fn_v5 can read
    grid_obs = {
        'Phi':              jnp.tile(jnp.array([[0.30, 0.30]]), (n_steps, 1)),
        'C':                jnp.cos(2 * jnp.pi * jnp.arange(n_steps) / 96),
        'hr_value':         jnp.zeros(n_steps),
        'hr_present':       jnp.zeros(n_steps),
        'stress_value':     jnp.zeros(n_steps),
        'stress_present':   jnp.zeros(n_steps),
        'log_steps_value':  jnp.zeros(n_steps),
        'steps_present':    jnp.zeros(n_steps),
        'sleep_label':      jnp.zeros(n_steps, dtype=jnp.int32),
        'sleep_present':    jnp.zeros(n_steps),
        'vl_value':         jnp.zeros(n_steps),
        'vl_present':       jnp.zeros(n_steps),
    }

    y0 = jnp.array([0.05, 0.10, 0.30, 0.10, 0.030, 0.050])
    dt = jnp.float64(1.0 / 96)
    sigma_diag = jnp.zeros(6)   # not used in v5 propagate_fn body
    noise = jnp.zeros(6)
    rng_key = jax.random.PRNGKey(0)

    y_new, log_w = propagate_fn_v5(y0, jnp.float64(0.0), dt, theta,
                                    grid_obs, 0, sigma_diag, noise, rng_key)

    assert y_new.shape == (6,), f"unexpected state shape {y_new.shape}"
    assert jnp.all(jnp.isfinite(y_new)), f"NaN in y_new: {y_new}"
    assert jnp.isfinite(log_w), f"non-finite log-weight: {log_w}"


def test_v5_chance_constrained_cost_smoke():
    """evaluate_chance_constrained_cost runs on a 10-particle cloud."""
    from version_3.models.fsa_v5 import (
        evaluate_chance_constrained_cost, TRUTH_PARAMS_V5,
    )

    # 10-particle cloud, all at the v5 truth
    n_particles = 10
    particles = [dict(TRUTH_PARAMS_V5) for _ in range(n_particles)]
    weights = np.ones(n_particles) / n_particles

    # 14-day moderate Phi schedule
    n_steps = 14 * 96
    Phi = np.tile([0.30, 0.30], (n_steps, 1))

    out = evaluate_chance_constrained_cost(
        particles, weights, Phi,
        dt=1.0/96, alpha=0.05, A_target=2.0,
        truth_params_template=TRUTH_PARAMS_V5,
    )

    expected_keys = {
        'mean_effort', 'mean_A_integral',
        'violation_rate_per_particle', 'weighted_violation_rate',
        'satisfies_chance_constraint', 'satisfies_target',
        'A_sep_per_bin',
    }
    assert expected_keys <= set(out.keys()), (
        f"chance-constraint cost missing keys: {expected_keys - set(out.keys())}")

    # Sanity ranges
    assert 0.0 <= out['weighted_violation_rate'] <= 1.0, \
        f"weighted_violation_rate out of [0,1]: {out['weighted_violation_rate']}"
    assert out['mean_effort'] > 0.0, f"effort should be positive: {out['mean_effort']}"
    assert out['mean_A_integral'] > 0.0, \
        f"mean_A_integral should be positive: {out['mean_A_integral']}"
    assert out['violation_rate_per_particle'].shape == (n_particles,), \
        f"per-particle rate shape: {out['violation_rate_per_particle'].shape}"

    # At v5 truth + moderate Phi (inside the island) the chance constraint
    # should be trivially satisfied.
    assert out['satisfies_chance_constraint'], \
        f"v5 truth + moderate Phi failed chance constraint: " \
        f"violation_rate={out['weighted_violation_rate']:.4f}"


# ─── JIT / vmap variants of the chance-constrained cost ────────────────────
# Added in the post-port rewrite of control_v5.py. The legacy NumPy
# evaluator above (``evaluate_chance_constrained_cost``) is now an alias
# for ``evaluate_chance_constrained_cost_hard``; the slow legacy
# implementation is preserved as the private
# ``_evaluate_chance_constrained_cost_legacy`` for debug.

def _build_smoke_cloud(n_particles=5, n_days=14):
    """Helper: 10-particle cloud at v5 truth + 14-day moderate Phi."""
    from version_3.models.fsa_v5 import TRUTH_PARAMS_V5
    particles = [dict(TRUTH_PARAMS_V5) for _ in range(n_particles)]
    weights = np.ones(n_particles) / n_particles
    Phi = np.tile([0.30, 0.30], (n_days * 96, 1))
    return particles, weights, Phi


def test_v5_cost_hard_jits():
    """Hard variant returns finite metrics inside a tight 5-particle, 1-day run."""
    from version_3.models.fsa_v5.control_v5 import evaluate_chance_constrained_cost_hard

    particles, weights, Phi = _build_smoke_cloud(n_particles=5, n_days=1)
    out = evaluate_chance_constrained_cost_hard(
        particles, weights, Phi, dt=1.0/96, alpha=0.05, A_target=2.0)
    assert jnp.isfinite(out['mean_effort'])
    assert jnp.isfinite(out['mean_A_integral'])
    assert jnp.isfinite(out['weighted_violation_rate'])
    # Trivially-satisfied chance constraint at v5 truth + moderate Phi.
    assert bool(out['satisfies_chance_constraint'])


def test_v5_cost_soft_jits():
    """Soft variant runs and returns finite metrics."""
    from version_3.models.fsa_v5.control_v5 import evaluate_chance_constrained_cost_soft

    particles, weights, Phi = _build_smoke_cloud(n_particles=5, n_days=1)
    out = evaluate_chance_constrained_cost_soft(
        particles, weights, Phi, dt=1.0/96, alpha=0.05, A_target=2.0,
        beta=50.0, scale=0.1)
    assert jnp.isfinite(out['mean_effort'])
    assert jnp.isfinite(out['mean_A_integral'])
    assert jnp.isfinite(out['weighted_violation_rate'])
    # Soft variant returns a value in [0, 1] but doesn't have to be exactly
    # the same as hard — sigmoid is smooth, so weighted_vr can be slightly
    # nonzero even at v5 truth.
    assert 0.0 <= float(out['weighted_violation_rate']) <= 1.0


def test_v5_cost_soft_grad_finite():
    """jax.grad of soft variant wrt a single parameter returns finite, non-zero gradient."""
    from version_3.models.fsa_v5.control_v5 import (
        _cost_soft_jit, _stack_particle_dicts, _ensure_v5_keys,
    )
    from version_3.models.fsa_v5 import TRUTH_PARAMS_V5

    n_particles = 5
    particles = [dict(TRUTH_PARAMS_V5) for _ in range(n_particles)]
    weights = jnp.ones(n_particles) / n_particles
    Phi = jnp.tile(jnp.array([0.30, 0.30]), (96, 1))   # 1 day
    initial_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07])

    theta_stacked = _ensure_v5_keys(_stack_particle_dicts(particles), TRUTH_PARAMS_V5)

    def scalar_cost(theta_dict):
        out = _cost_soft_jit(theta_dict, weights, Phi, initial_state,
                              1.0/96, 0.05, 2.0, 50.0, 0.1)
        # Differentiable cost with a real dependency on theta:
        # effort is Phi-only (no grad), but mean_A_integral depends on
        # the per-particle trajectory which depends on theta.
        return out['mean_effort'] - out['mean_A_integral']

    grad_fn = jax.grad(scalar_cost)
    grads = grad_fn(theta_stacked)
    # Check at least one parameter has a finite, non-trivial gradient.
    # Parameters that drive the autonomic state (mu_0, mu_B, eta, etc.)
    # should all have non-zero gradients via the A trajectory.
    found_finite = False
    for k, g in grads.items():
        assert jnp.all(jnp.isfinite(g)), f"non-finite gradient for {k}: {g}"
        if jnp.any(jnp.abs(g) > 1e-12):
            found_finite = True
    assert found_finite, (
        f"all gradients near zero — autodiff not threading through trajectory? "
        f"max |grad| over all keys = {max(float(jnp.max(jnp.abs(g))) for g in grads.values()):.3e}")


def test_v5_cost_soft_to_hard_limit():
    """Soft variant with large beta approaches hard variant."""
    from version_3.models.fsa_v5.control_v5 import (
        evaluate_chance_constrained_cost_hard,
        evaluate_chance_constrained_cost_soft,
    )

    particles, weights, Phi = _build_smoke_cloud(n_particles=5, n_days=1)

    # At v5 truth + moderate Phi, all particles are well above A_sep
    # (or A_sep is -inf). Both variants should return ~0 violation rate.
    out_hard = evaluate_chance_constrained_cost_hard(
        particles, weights, Phi, dt=1.0/96, alpha=0.05, A_target=2.0)
    out_soft = evaluate_chance_constrained_cost_soft(
        particles, weights, Phi, dt=1.0/96, alpha=0.05, A_target=2.0,
        beta=10000.0, scale=0.1)

    # Effort + ∫A integral are deterministic in Phi/state, identical between variants.
    assert jnp.isclose(out_hard['mean_effort'], out_soft['mean_effort'])
    assert jnp.isclose(out_hard['mean_A_integral'], out_soft['mean_A_integral'])
    # Violation rates: at large beta the soft sigmoid → hard indicator.
    diff = abs(float(out_hard['weighted_violation_rate'])
               - float(out_soft['weighted_violation_rate']))
    assert diff < 1e-3, f"hard vs soft (β=1e4) violation rate differs by {diff:.4e}"


if __name__ == '__main__':
    # Allow running as a script too — avoids needing pytest for a quick check
    test_v5_imports_clean();                        print("PASS: imports")
    test_v5_plant_forward_pipeline();               print("PASS: plant forward pipeline")
    test_v5_estimation_propagate_fn_runs();         print("PASS: estimation propagate_fn")
    test_v5_chance_constrained_cost_smoke();        print("PASS: chance-constrained cost (legacy alias)")
    test_v5_cost_hard_jits();                       print("PASS: cost hard variant")
    test_v5_cost_soft_jits();                       print("PASS: cost soft variant")
    test_v5_cost_soft_grad_finite();                print("PASS: soft variant gradient")
    test_v5_cost_soft_to_hard_limit();              print("PASS: soft → hard limit")
    print("All v5 smoke tests passed.")
