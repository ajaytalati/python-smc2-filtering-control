"""FSA high-res control task spec — v2 (Banister-coupled).

Builds and exports a `ControlSpec` for the v2 dynamics. The single
control input is now Φ(t) (training-strain rate). T_B is gone — in v2
fitness B accrues from training Φ explicitly (Banister chronic), so
the trivial "rest with high target" optimum that v1 admitted is no
longer reachable.

Cost functional (mean over MC noise grid, common random numbers):

    J(θ) = E_τ [ −∫ A(t) dt
                + λ_Φ · ∫ Φ(t)² dt
                + λ_F · ∫ max(F(t) − F_max, 0)² dt ]

Schedule parameterisation: 8 Gaussian RBF anchors over the horizon,
sigmoid output transform with a logit-bias offset so θ at the prior
mean produces the canonical-Banister Φ ≈ 1.0 default. Total search
dimension: θ ∈ ℝ^8 (half v1's dimension, since T_B was dropped).

Horizon is parameterised so the same model can be run at T = 28, 42,
56, or 84 days (Banister τ_B = 42 d is canonical chronic; we sweep).
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import ControlSpec, RBFSchedule
from smc2fc.control.calibration import build_crn_noise_grids

from models.fsa_high_res._dynamics import (
    TRUTH_PARAMS, drift_jax, diffusion_state_dep,
)


# ── Initial conditions + horizon defaults ─────────────────────────────

INIT_STATE = dict(
    B=0.05,    # very low fitness (de-trained)
    F=0.30,    # high residual fatigue
    A=0.10,    # low autonomic amplitude
)

EXOGENOUS = dict(
    T_total=42.0,           # days — canonical (one chronic time constant)
    dt_days=1.0 / 96.0,     # 15 min outer step
    n_substeps=4,
    F_max=0.40,             # overtraining-fatigue limit
    Phi_max=3.0,            # max training-strain rate the schedule can request
    Phi_default=1.0,        # canonical Banister default — 1 unit of TRIMP/day
)


# ── Schedule decoder: θ ∈ ℝ^n_anchors → Φ(t) ─────────────────────────

def _make_schedule(*, n_steps: int, dt: float, n_anchors: int,
                    Phi_default: float = EXOGENOUS['Phi_default'],
                    Phi_max: float = EXOGENOUS['Phi_max']):
    """Return the RBF basis + a packed schedule_from_theta closure.

    Parameterisation:
        Φ(t) = Phi_max · sigmoid( c_Phi + Φ(t) · θ )

    With c_Phi = logit(Phi_default / Phi_max), θ = 0 ⇒ Φ ≡ Phi_default.
    The Gaussian prior N(0, σ²·I) over θ explores schedules near
    this baseline rather than at extreme rest or overtraining.
    """
    rbf = RBFSchedule(
        n_steps=n_steps, dt=dt, n_anchors=n_anchors, output='identity',
    )
    Phi_design = rbf.design_matrix()    # (n_steps, n_anchors)

    # Logit bias: at θ=0, sigmoid(c_Phi) · Phi_max = Phi_default
    p_ratio = Phi_default / Phi_max
    c_Phi = float(np.log(p_ratio / (1.0 - p_ratio)))

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        """θ shape (n_anchors,) → Φ(t) of shape (n_steps,)."""
        raw = c_Phi + jnp.einsum('a,ta->t', theta, Phi_design)
        return Phi_max * jax.nn.sigmoid(raw)

    return rbf, schedule_from_theta


# ── Substepped Euler-Maruyama with sqrt-diffusion + reflection ────────

def _make_em_step_fn(params, dt, n_substeps):
    sub_dt = dt / float(n_substeps)
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def em_step(y, Phi_t, noise_3d):
        def sub_body(y_inner, _):
            return y_inner + sub_dt * drift_jax(y_inner, params, Phi_t), None
        y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))

        sigma_y = diffusion_state_dep(y_det, params)
        y_pred = y_det + sigma_y * sqrt_dt * noise_3d

        # Boundary reflection (B ∈ [0,1], F ≥ 0, A ≥ 0). σ vanishes at
        # each boundary, so reflection rarely fires.
        B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
        B_next = jnp.where(B_pred < 0.0, -B_pred,
                            jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
        F_next = jnp.abs(F_pred)
        A_next = jnp.abs(A_pred)
        return jnp.array([B_next, F_next, A_next])

    return em_step


# ── Cost-functional builder ───────────────────────────────────────────

def _build_cost_and_traj_fns(
    *,
    n_inner: int,
    n_steps: int,
    dt: float,
    n_substeps: int,
    schedule_from_theta,
    F_max: float,
    lam_phi: float = 0.0,
    lam_barrier: float = 1.0,
    seed: int = 42,
):
    # lam_phi=0 by default: the F-barrier (lam_barrier · max(F-F_max, 0)²)
    # already penalises overtraining via fatigue overshoot, and the
    # Stuart-Landau μ_FF·F² term makes high-Phi schedules collapse μ → 0
    # endogenously. An additional Phi² penalty is double-counting and
    # was empirically observed to pull SMC toward sub-optimal Phi ≈ 0.5
    # (T=42 d, λ_Phi=0.05) with mean ∫A/T = 0.182 < baseline 0.216.
    """Build (cost_fn, traj_sample_fn) closures for the v2 FSA control task."""

    p_jax = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=3, seed=seed,
    )
    fixed_w = grids['wiener']    # (n_inner, n_steps, 3)

    init_arr = jnp.array([INIT_STATE['B'], INIT_STATE['F'], INIT_STATE['A']])

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)    # (n_steps,)

        def trial(w_seq):
            def step(carry, k):
                y, A_acc, Phi_acc, barrier_acc = carry
                Phi_t = Phi_arr[k]
                y_next = em_step(y, Phi_t, w_seq[k])
                A_acc = A_acc + y[2] * dt                                # ∫A dt
                Phi_acc = Phi_acc + Phi_t * Phi_t * dt                    # ∫Φ² dt
                barrier_acc = barrier_acc + (
                    jnp.maximum(y[1] - F_max, 0.0) ** 2 * dt
                )                                                          # ∫max(F-F_max,0)² dt
                return (y_next, A_acc, Phi_acc, barrier_acc), None

            init_carry = (init_arr, jnp.float64(0.0),
                           jnp.float64(0.0), jnp.float64(0.0))
            (_, A_acc, Phi_acc, barrier_acc), _ = jax.lax.scan(
                step, init_carry, jnp.arange(n_steps),
            )
            return -A_acc + lam_phi * Phi_acc + lam_barrier * barrier_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)
        w_seq = jax.random.normal(key, (n_steps, 3), dtype=jnp.float64)

        def step(y, k):
            y_next = em_step(y, Phi_arr[k], w_seq[k])
            return y_next, y_next

        _, traj = jax.lax.scan(step, init_arr, jnp.arange(n_steps))
        return traj    # (n_steps, 3)

    return cost_fn, traj_sample_fn


# ── Acceptance gates ──────────────────────────────────────────────────

def _build_gates(*, schedule_from_theta,
                  n_steps: int, dt: float, n_substeps: int,
                  n_baseline_trials: int = 50,
                  n_eval_trials: int = 50,
                  Phi_baseline: float = EXOGENOUS['Phi_default'],
                  F_max: float = 0.40,
                  seed: int = 123):
    """Build acceptance gates that operate on the result dict.

    References:
      - constant-Φ baseline at Φ_baseline (canonical Banister default).
      - sedentary (Φ ≡ 0) — model-integrity reference; SMC² must beat
        this clearly, since v2 dynamics no longer admit the v1 "rest
        cures all" pathology.

    Both baselines + the per-result SMC schedule eval are fully
    JIT/vmap-compiled.
    """
    p_jax = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
    em_step = _make_em_step_fn(p_jax, dt, n_substeps)

    init_arr = jnp.array([INIT_STATE['B'], INIT_STATE['F'], INIT_STATE['A']])

    def _make_eval_fn(n_trials_static: int):
        @jax.jit
        def _evaluate_schedule_jit(Phi_arr, key):
            def trial(w_seq):
                def step(carry, k):
                    y, A_acc, F_viol = carry
                    y_next = em_step(y, Phi_arr[k], w_seq[k])
                    A_acc = A_acc + y[2] * dt
                    F_viol = F_viol + jnp.where(y[1] > F_max, 1.0, 0.0)
                    return (y_next, A_acc, F_viol), None

                init_carry = (init_arr, jnp.float64(0.0), jnp.float64(0.0))
                (_, A_acc, F_viol), _ = jax.lax.scan(
                    step, init_carry, jnp.arange(n_steps),
                )
                return A_acc / (n_steps * dt), F_viol / n_steps

            w = jax.random.normal(key, (n_trials_static, n_steps, 3),
                                    dtype=jnp.float64)
            A_means, F_viols = jax.vmap(trial)(w)
            return jnp.mean(A_means), jnp.mean(F_viols)
        return _evaluate_schedule_jit

    _eval_baseline_jit = _make_eval_fn(n_baseline_trials)
    _eval_smc_jit = _make_eval_fn(n_eval_trials)

    # Constant-Φ_baseline reference
    Phi_const = jnp.full(n_steps, Phi_baseline, dtype=jnp.float64)
    base_key = jax.random.PRNGKey(seed)
    baseline_mean_A_j, baseline_F_violation_j = _eval_baseline_jit(
        Phi_const, base_key,
    )
    baseline_mean_A = float(baseline_mean_A_j)
    baseline_F_violation = float(baseline_F_violation_j)

    # Sedentary (Φ ≡ 0) reference
    Phi_zero = jnp.zeros(n_steps, dtype=jnp.float64)
    sedentary_mean_A_j, _ = _eval_baseline_jit(
        Phi_zero, jax.random.PRNGKey(seed + 100),
    )
    sedentary_mean_A = float(sedentary_mean_A_j)

    print(f"  baseline (constant Φ={Phi_baseline}):")
    print(f"    mean ∫A/T = {baseline_mean_A:.3f}, "
          f"F-violation fraction = {baseline_F_violation:.2%}")
    print(f"  sedentary (constant Φ=0):")
    print(f"    mean ∫A/T = {sedentary_mean_A:.3f}")

    eval_key = jax.random.PRNGKey(seed + 1)
    cache: dict = {}

    def _evaluate_smc_schedule(result):
        key_id = id(result)
        if key_id in cache:
            return cache[key_id]
        theta_mean = jnp.asarray(result['mean_theta'])
        Phi_arr = schedule_from_theta(theta_mean)
        A_mean_j, F_viol_j = _eval_smc_jit(Phi_arr, eval_key)
        out = (float(A_mean_j), float(F_viol_j), float(jnp.mean(Phi_arr)))
        cache[key_id] = out
        return out

    def gate_mean_A_matches_baseline(result):
        smc_A, _, _ = _evaluate_smc_schedule(result)
        target = baseline_mean_A * 0.97
        passed = smc_A >= target
        return passed, smc_A, (
            f"SMC² mean ∫A/T = {smc_A:.3f}  vs baseline*0.97 = {target:.3f}  "
            f"({'passes' if passed else 'fails'}) — SMC matches the "
            f"best constant baseline within 3%"
        )

    def gate_mean_A_beats_sedentary(result):
        smc_A, _, _ = _evaluate_smc_schedule(result)
        target = sedentary_mean_A * 1.40
        passed = smc_A >= target
        return passed, smc_A, (
            f"SMC² mean ∫A/T = {smc_A:.3f}  vs sedentary*1.40 = {target:.3f}  "
            f"({'passes' if passed else 'fails'}) — model-integrity "
            f"(rejects rest-cures-all)"
        )

    def gate_mean_phi_in_range(result):
        _, _, mean_phi = _evaluate_smc_schedule(result)
        passed = (mean_phi >= 0.5) and (mean_phi <= 2.5)
        return passed, mean_phi, (
            f"SMC² mean Φ = {mean_phi:.3f}  ∈ [0.5, 2.5]  "
            f"({'passes' if passed else 'fails'}) — physiologically "
            f"reasonable training range"
        )

    def gate_fatigue_within_bound(result):
        _, viol, _ = _evaluate_smc_schedule(result)
        passed = viol <= 0.05
        return passed, viol, (
            f"F-violation fraction = {viol:.2%} ≤ 5%  "
            f"({'passes' if passed else 'fails'})"
        )

    return {
        'mean_A_matches_baseline_(within_3%)':  gate_mean_A_matches_baseline,
        'mean_A_>=_1.40_x_sedentary_(Phi=0)':   gate_mean_A_beats_sedentary,
        'mean_Phi_in_[0.5, 2.5]':                gate_mean_phi_in_range,
        'F_violation_fraction_<=_5%':            gate_fatigue_within_bound,
    }, dict(
        baseline_mean_A=baseline_mean_A,
        baseline_F_violation=baseline_F_violation,
        baseline_Phi=Phi_baseline,
        sedentary_mean_A=sedentary_mean_A,
    )


# ── Build the spec ────────────────────────────────────────────────────

def build_control_spec(
    *,
    T_total_days: float = EXOGENOUS['T_total'],
    dt_days: float = EXOGENOUS['dt_days'],
    n_substeps: int = EXOGENOUS['n_substeps'],
    n_anchors: int = 8,
    n_inner: int = 32,
    seed: int = 42,
    F_max: float = EXOGENOUS['F_max'],
) -> ControlSpec:
    """Construct an FSA-v2 ControlSpec for the given horizon.

    Pass T_total_days=42 for the canonical Banister chronic time
    constant; other values supported for the horizon-sweep experiments.
    """
    n_steps = int(round(T_total_days / dt_days))

    rbf, schedule_from_theta = _make_schedule(
        n_steps=n_steps, dt=dt_days, n_anchors=n_anchors,
    )

    cost_fn, traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt_days, n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        F_max=F_max, seed=seed,
    )

    gates, refs = _build_gates(
        schedule_from_theta=schedule_from_theta,
        n_steps=n_steps, dt=dt_days, n_substeps=n_substeps, F_max=F_max,
    )

    spec = ControlSpec(
        name=f'fsa_high_res_v2_T{int(T_total_days)}d',
        version='2.0',
        dt=dt_days,
        n_steps=n_steps,
        n_substeps=n_substeps,
        initial_state=jnp.array([INIT_STATE['B'], INIT_STATE['F'],
                                    INIT_STATE['A']]),
        truth_params=dict(TRUTH_PARAMS),
        theta_dim=n_anchors,
        sigma_prior=1.5,
        prior_mean=0.0,    # θ=0 already gives Φ=Phi_default via the logit bias
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates,
    )
    object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)
    object.__setattr__(spec, '_refs', refs)
    object.__setattr__(spec, '_F_max', F_max)
    object.__setattr__(spec, '_n_anchors', n_anchors)
    object.__setattr__(spec, '_T_total_days', T_total_days)
    object.__setattr__(spec, '_Phi_max', EXOGENOUS['Phi_max'])
    return spec


def get_control_spec(**kwargs) -> ControlSpec:
    return build_control_spec(**kwargs)


FSA_CONTROL_SPEC = None
