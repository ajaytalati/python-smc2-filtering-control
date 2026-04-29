"""Smoke tests for the FSA-v2 (Banister-coupled) dynamics + control spec.

Verifies:
  - Φ ≡ 0 sends the system toward sedentary equilibrium (mean_A close to
    A_0 decay) — confirming v1's 'rest cures all' pathology is gone.
  - Φ ≫ Φ_optimum collapses A via μ_FF·F² (overtraining bound is endogenous).
  - State-dependent sqrt-diffusion keeps B ∈ [0, 1] and F, A ≥ 0 without
    clipping — boundary reflection rarely fires.
  - The control spec builds and the cost function returns a finite scalar.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np


def _rollout_constant_phi(Phi_const, T_total=21.0, dt=1.0/96.0,
                            n_substeps=4, n_trials=10, seed=0):
    """Roll out the v2 SDE under a constant Φ; return trajectory tensor."""
    from models.fsa_high_res._dynamics import (
        TRUTH_PARAMS, imex_step_substepped,
    )
    p = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
    init = jnp.array([0.05, 0.30, 0.10])
    n_steps = int(round(T_total / dt))

    def trial(noise_seq):
        def step(y, k):
            y_next = imex_step_substepped(y, p, noise_seq[k],
                                            Phi_const, dt, n_substeps)
            return y_next, y_next
        _, traj = jax.lax.scan(step, init, jnp.arange(n_steps))
        return traj

    key = jax.random.PRNGKey(seed)
    noise = jax.random.normal(key, (n_trials, n_steps, 3), dtype=jnp.float64)
    return jax.vmap(trial)(noise)


def test_v2_sedentary_does_not_cure_all():
    """Φ=0 should *not* drive A up — it should leave amplitude near
    its sedentary slow-decay equilibrium. v1's 'rest cures all' pathology
    is rejected by the v2 dynamics (B accrues from Φ, not from a free
    target T_B)."""
    traj = _rollout_constant_phi(0.0, T_total=21.0)
    mean_A = float(jnp.mean(traj[:, :, 2]))
    # Sedentary mean ∫A/T must NOT be high — under the broken v1 model
    # with the same parameters and a 'rest' command, mean_A reached ~0.4.
    # In v2 it stays ~0.10-0.15 (dominated by initial-A slow decay).
    assert mean_A < 0.20, (
        f"Sedentary mean_A = {mean_A:.3f} too high — model still admits"
        f" the rest-cures-all pathology"
    )


def test_v2_overtraining_collapses_amplitude():
    """Φ much larger than the Banister homeostatic point should drive F
    high enough that μ(B,F) goes negative via μ_FF·F², collapsing A."""
    traj = _rollout_constant_phi(5.0, T_total=42.0)
    mean_A_overtrained = float(jnp.mean(traj[:, :, 2]))
    # Compare to a moderate Φ (which should be much better)
    traj_mod = _rollout_constant_phi(1.0, T_total=42.0)
    mean_A_mod = float(jnp.mean(traj_mod[:, :, 2]))
    assert mean_A_overtrained < mean_A_mod, (
        f"Overtrained Φ=5 mean_A = {mean_A_overtrained:.3f} should be"
        f" worse than moderate Φ=1 mean_A = {mean_A_mod:.3f}"
    )


def test_v2_states_stay_in_physiological_range():
    """Sqrt-Itô diffusion + boundary reflection should keep B ∈ [0, 1],
    F ≥ 0, A ≥ 0 without state clipping artifacts."""
    traj = _rollout_constant_phi(2.0, T_total=42.0, n_trials=20)
    B_min = float(jnp.min(traj[:, :, 0]))
    B_max = float(jnp.max(traj[:, :, 0]))
    F_min = float(jnp.min(traj[:, :, 1]))
    A_min = float(jnp.min(traj[:, :, 2]))
    assert B_min >= 0.0, f"B underflow: min = {B_min}"
    assert B_max <= 1.0, f"B overflow: max = {B_max}"
    assert F_min >= 0.0, f"F underflow: min = {F_min}"
    assert A_min >= 0.0, f"A underflow: min = {A_min}"


def test_v2_control_spec_builds_and_costs_are_finite():
    """The ControlSpec instantiates and cost_fn returns a finite scalar."""
    from models.fsa_high_res.control import build_control_spec

    # Use the smallest reasonable horizon for a quick test
    spec = build_control_spec(T_total_days=7.0, n_inner=4)
    assert spec.theta_dim == 8
    cost_zero = float(spec.cost_fn(jnp.zeros(spec.theta_dim)))
    assert np.isfinite(cost_zero), f"cost at θ=0 is not finite: {cost_zero}"

    # θ = 0 ⇒ schedule should be approximately constant Φ_default = 1.0
    Phi_arr = np.asarray(spec.schedule_from_theta(jnp.zeros(spec.theta_dim)))
    assert abs(float(Phi_arr.mean()) - 1.0) < 0.05, (
        f"θ=0 schedule mean Φ = {Phi_arr.mean()} should be ≈ 1.0"
    )
