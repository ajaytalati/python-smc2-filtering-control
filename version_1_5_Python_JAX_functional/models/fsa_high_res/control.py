"""FSA v1.5 control task spec — purely-functional rewrite.

Functional rewrite of `version_1_5_Python_JAX/models/fsa_high_res/control.py`.
Two refactors vs the imperative source:

  (1) `params_v15` is a `ParamsV15` NamedTuple at every entry point;
      the v1-basis closure inside `_make_em_step_fn` is built via
      `jax.tree_map(jnp.asarray, params_v15_to_v1(params_v15))`.
  (2) `build_control_spec` no longer attaches `_traj_sample_fn` to the
      frozen `ControlSpec` via `object.__setattr__`. The diagnostic
      sampler is dropped (it was unused in v1.5 Python). Callers that
      want it can construct it via `_build_cost_and_traj_fns` directly.

Single Φ control input; 8 RBF anchors over the planning horizon; sigmoid
output transform with logit-bias offset so θ at the prior mean produces
the canonical Banister Φ ≈ 1.0. Total search dim: θ ∈ ℝ^8.

Cost functional (mean over CRN MC noise grid):

    J(θ) = E_τ [ −∫ A(t) dt
                + λ_F · ∫ max(F(t) − F_max, 0)² dt ]

NO Φ² penalty (matching `version_1_5_Julia/models/fsa_high_res/gpu_control.jl`
which is "v1 verbatim" minus the λ_Φ term). The Stuart-Landau μ_FF·F²
term and the F-barrier already discourage overtraining endogenously.
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

from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS, INIT_STATE, ParamsV15, InitState, params_v15_to_v1,
)


# ── Defaults ──────────────────────────────────────────────────────────
EXOGENOUS = dict(
    T_total=14.0,           # days — matches v1.5 Julia bench's 14-day open-loop
    dt_days=1.0 / 24.0,     # 1-hour outer step (matches default BINS_PER_DAY=24)
    n_substeps=4,
    F_max=0.40,
    Phi_max=3.0,
    Phi_default=1.0,
)


# ── RBF schedule decoder: θ ∈ ℝ^n_anchors → Φ(t) ──────────────────────
def _make_schedule(*, n_steps: int, dt: float, n_anchors: int = 8,
                    Phi_default: float = EXOGENOUS['Phi_default'],
                    Phi_max: float = EXOGENOUS['Phi_max']):
    """RBF basis + a JIT'd schedule_from_theta(θ) closure.

    Φ(t) = Phi_max · sigmoid(c_Φ + θ · design_matrix(t))
    With c_Φ = logit(Phi_default / Phi_max), θ = 0 ⇒ Φ ≡ Phi_default.
    """
    rbf = RBFSchedule(
        n_steps=n_steps, dt=dt, n_anchors=n_anchors, output='identity',
    )
    Phi_design = rbf.design_matrix()    # (n_steps, n_anchors)

    p_ratio = Phi_default / Phi_max
    c_Phi = float(np.log(p_ratio / (1.0 - p_ratio)))

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        raw = c_Phi + jnp.einsum('a,ta->t', theta, Phi_design)
        return Phi_max * jax.nn.sigmoid(raw)

    return rbf, schedule_from_theta


# ── EM step factory (v1.5 → v1 basis rotation done once per build) ────
def _make_em_step_fn(params_v15: ParamsV15, dt: float, n_substeps: int):
    p_v1 = params_v15_to_v1(params_v15)
    # Wrap each scalar field in jnp.asarray for consistent JAX dtype
    # under JIT. NamedTuple is a registered pytree so tree_map works.
    p_jax = jax.tree_util.tree_map(lambda v: jnp.asarray(float(v)), p_v1)
    sub_dt = dt / float(n_substeps)
    sqrt_dt = jnp.sqrt(dt)

    @jax.jit
    def em_step(y, Phi_t, noise_3d):
        def sub_body(y_inner, _):
            return y_inner + sub_dt * drift_jax(y_inner, p_jax, Phi_t), None
        y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))
        sigma_y = diffusion_state_dep(y_det, p_jax)
        y_pred = y_det + sigma_y * sqrt_dt * noise_3d
        B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
        B_next = jnp.where(B_pred < 0.0, -B_pred,
                            jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
        F_next = jnp.abs(F_pred)
        A_next = jnp.abs(A_pred)
        return jnp.array([B_next, F_next, A_next])

    return em_step


# ── Cost + diagnostic-trajectory builders ──────────────────────────────
def _build_cost_and_traj_fns(*, n_inner: int, n_steps: int,
                              dt: float, n_substeps: int,
                              schedule_from_theta,
                              params_v15: ParamsV15,
                              init_state: InitState,
                              F_max: float = EXOGENOUS['F_max'],
                              lam_barrier: float = 1.0,
                              seed: int = 42):
    em_step = _make_em_step_fn(params_v15, dt, n_substeps)
    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=3, seed=seed,
    )
    fixed_w = grids['wiener']    # (n_inner, n_steps, 3)
    init_arr = jnp.array([float(init_state.B),
                          float(init_state.F),
                          float(init_state.A)])

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)

        def trial(w_seq):
            def step(carry, k):
                y, A_acc, barrier_acc = carry
                Phi_t = Phi_arr[k]
                y_next = em_step(y, Phi_t, w_seq[k])
                A_acc = A_acc + y[2] * dt
                barrier_acc = barrier_acc + (
                    jnp.maximum(y[1] - F_max, 0.0) ** 2 * dt
                )
                return (y_next, A_acc, barrier_acc), None

            init_carry = (init_arr, jnp.float64(0.0), jnp.float64(0.0))
            (_, A_acc, barrier_acc), _ = jax.lax.scan(
                step, init_carry, jnp.arange(n_steps),
            )
            return -A_acc + lam_barrier * barrier_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> jnp.ndarray:
        Phi_arr = schedule_from_theta(theta)
        w_seq = jax.random.normal(key, (n_steps, 3), dtype=jnp.float64)

        def step(y, k):
            y_next = em_step(y, Phi_arr[k], w_seq[k])
            return y_next, y_next

        _, traj = jax.lax.scan(step, init_arr, jnp.arange(n_steps))
        return traj

    return cost_fn, traj_sample_fn


# ── ControlSpec factory ───────────────────────────────────────────────
def build_control_spec(*, T_total: float = EXOGENOUS['T_total'],
                         dt_days: float = EXOGENOUS['dt_days'],
                         n_substeps: int = EXOGENOUS['n_substeps'],
                         n_anchors: int = 8,
                         n_inner: int = 32,
                         Phi_max: float = EXOGENOUS['Phi_max'],
                         Phi_default: float = EXOGENOUS['Phi_default'],
                         F_max: float = EXOGENOUS['F_max'],
                         lam_barrier: float = 1.0,
                         params_v15: ParamsV15 = None,
                         init_state: InitState = None,
                         sigma_prior: float = 1.5,
                         seed: int = 42) -> ControlSpec:
    """Build a `smc2fc.control.ControlSpec` for v1.5.

    Defaults match `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`.
    Supplying `params_v15` (e.g. a posterior-mean `ParamsV15` from the
    filter) swaps the truth params for closed-loop replans.

    Returns the `ControlSpec` only — the diagnostic `traj_sample_fn`
    is no longer attached via `object.__setattr__` (the imperative
    source did this and no consumer in v1.5 read it).
    """
    p_v15 = DEFAULT_PARAMS if params_v15 is None else params_v15
    init = INIT_STATE if init_state is None else init_state

    n_steps = int(round(T_total / dt_days))
    rbf, schedule_from_theta = _make_schedule(
        n_steps=n_steps, dt=dt_days, n_anchors=n_anchors,
        Phi_default=Phi_default, Phi_max=Phi_max,
    )
    cost_fn, _traj_sample_fn = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt_days,
        n_substeps=n_substeps,
        schedule_from_theta=schedule_from_theta,
        params_v15=p_v15, init_state=init,
        F_max=F_max, lam_barrier=lam_barrier, seed=seed,
    )

    initial_state_jnp = jnp.array(
        [float(init.B), float(init.F), float(init.A)],
        dtype=jnp.float64,
    )
    return ControlSpec(
        name='fsa_high_res_v15',
        version='1.5',
        dt=dt_days,
        n_steps=n_steps,
        n_substeps=n_substeps,
        initial_state=initial_state_jnp,
        truth_params={k: float(v) for k, v in p_v15._asdict().items()},
        theta_dim=n_anchors,
        sigma_prior=sigma_prior,
        prior_mean=0.0,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
    )


# ── Compile-once factory (the GPU-saturation fix) ────────────────────
def build_control_spec_compileonce(
        *,
        T_total: float = EXOGENOUS['T_total'],
        dt_days: float = EXOGENOUS['dt_days'],
        n_substeps: int = EXOGENOUS['n_substeps'],
        n_anchors: int = 8,
        n_inner: int = 32,
        Phi_max: float = EXOGENOUS['Phi_max'],
        Phi_default: float = EXOGENOUS['Phi_default'],
        F_max: float = EXOGENOUS['F_max'],
        lam_barrier: float = 1.0,
        sigma_prior: float = 1.5,
        seed: int = 42,
):
    """Builds the JIT'd controller cost kernel ONCE; returns a per-replan factory.

    Closes the controller-side analogue of the filter's BlackJAX
    per-stride-recompile issue (cf. CLAUDE.md "MANDATORY: use the
    JAX-native compile-once SMC kernel path").

    Mechanism:

        * `schedule_from_theta` and `cost_kernel` are both JIT'd ONCE
          at build time. Static config (n_steps, n_substeps, n_anchors,
          n_inner, dt, F_max, lam_barrier, Phi_max, Phi_default, seed)
          is captured in their closures.
        * Dynamic per-replan data (`params_v15`, `init_state`) is passed
          as RUNTIME ARGUMENTS to `cost_kernel`, not closure-captured.
        * The returned `make_spec(params_v15, init_state)` factory binds
          those two runtime args via `jax.tree_util.Partial`, producing
          a `ControlSpec` whose `cost_fn` is `Partial(cost_kernel,
          params_jax, init_arr)`. Because the underlying `cost_kernel`
          PjitFunction identity is stable across all replans, JAX's
          trace cache hits and XLA does not recompile.

    Saving on a 28-day, 27-replan run: ≈10-30 s of XLA compilation per
    replan eliminated → 5-15 min of host-side stalling removed → the
    GPU stays warm between replans.

    Args:
        T_total: Planning horizon in days (matches the filter horizon).
        dt_days: Outer step size in days.
        n_substeps: EM sub-steps per outer step.
        n_anchors: RBF basis cardinality (controller posterior dim).
        n_inner: Common-random-numbers MC trials per cost evaluation.
        Phi_max: Sigmoid cap on the applied Φ.
        Phi_default: Φ value reached at θ=0 (sets the logit bias).
        F_max: Soft-barrier threshold on F.
        lam_barrier: Soft-barrier weight in the cost.
        sigma_prior: Default σ for the controller prior on θ.
        seed: Seed for the CRN noise grid.

    Returns:
        A factory function ``make_spec(params_v15: ParamsV15,
        init_state: InitState, *, sigma_prior: float = …) ->
        ControlSpec``. The factory binds the runtime args and returns
        a `ControlSpec` whose `cost_fn` is a `Partial`-wrapped
        compile-once kernel; calling that `cost_fn(theta)` reuses the
        cached HLO across replans.
    """
    n_steps = int(round(T_total / dt_days))
    sub_dt = dt_days / float(n_substeps)
    sqrt_dt = jnp.sqrt(jnp.float64(dt_days))

    rbf = RBFSchedule(
        n_steps=n_steps, dt=dt_days, n_anchors=n_anchors, output='identity',
    )
    Phi_design = jnp.asarray(rbf.design_matrix(), dtype=jnp.float64)

    p_ratio = Phi_default / Phi_max
    c_Phi = float(np.log(p_ratio / (1.0 - p_ratio)))

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=3, seed=seed,
    )
    fixed_w = jnp.asarray(grids['wiener'], dtype=jnp.float64)

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        raw = c_Phi + jnp.einsum('a,ta->t', theta, Phi_design)
        return Phi_max * jax.nn.sigmoid(raw)

    @jax.jit
    def cost_kernel(params_v15: ParamsV15,
                    init_arr: jnp.ndarray,
                    theta: jnp.ndarray) -> jnp.ndarray:
        """The single JIT'd cost kernel. Compiled once, reused per replan.

        Args:
            params_v15: `ParamsV15` NamedTuple (JAX pytree of float
                scalars). Posterior mean filter parameters at this
                replan.
            init_arr: ``(3,)`` jnp.float64 — smoothed end-of-window
                latent state at this replan.
            theta: ``(n_anchors,)`` jnp.float64 — controller decision
                variable.

        Returns:
            Scalar mean cost over the CRN-MC noise grid.
        """
        p_v1 = params_v15_to_v1(params_v15)
        Phi_arr = schedule_from_theta(theta)

        def em_step(y, Phi_t, noise_3d):
            def sub_body(y_inner, _):
                return y_inner + sub_dt * drift_jax(y_inner, p_v1, Phi_t), None
            y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(n_substeps))
            sigma_y = diffusion_state_dep(y_det, p_v1)
            y_pred = y_det + sigma_y * sqrt_dt * noise_3d
            B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
            B_next = jnp.where(
                B_pred < 0.0, -B_pred,
                jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
            F_next = jnp.abs(F_pred)
            A_next = jnp.abs(A_pred)
            return jnp.array([B_next, F_next, A_next])

        def trial(w_seq):
            def step(carry, k):
                y, A_acc, barrier_acc = carry
                Phi_t = Phi_arr[k]
                y_next = em_step(y, Phi_t, w_seq[k])
                A_acc = A_acc + y[2] * dt_days
                barrier_acc = barrier_acc + (
                    jnp.maximum(y[1] - F_max, 0.0) ** 2 * dt_days
                )
                return (y_next, A_acc, barrier_acc), None

            init_carry = (init_arr, jnp.float64(0.0), jnp.float64(0.0))
            (_, A_acc, barrier_acc), _ = jax.lax.scan(
                step, init_carry, jnp.arange(n_steps),
            )
            return -A_acc + lam_barrier * barrier_acc

        return jnp.mean(jax.vmap(trial)(fixed_w))

    def make_spec(params_v15: ParamsV15,
                  init_state: InitState,
                  *,
                  sigma_prior: float = sigma_prior) -> ControlSpec:
        """Binds runtime args and returns a `ControlSpec`.

        The returned spec's `cost_fn` is a `jax.tree_util.Partial`
        over the shared JIT'd `cost_kernel`. The outer SMC kernel
        (`run_tempered_smc_loop_native`) wraps it in another Partial
        and feeds it to the module-level-JIT'd
        `_run_tempered_chain_jit`; trace-cache hits across all
        replans because the underlying PjitFunction identity is
        stable.

        Args:
            params_v15: This replan's `ParamsV15` (posterior mean).
            init_state: This replan's `InitState` (smoothed
                end-of-window latent).
            sigma_prior: Optional override of the prior std on θ.

        Returns:
            A `ControlSpec` ready to feed to
            `run_tempered_smc_loop_native`.
        """
        init_arr = jnp.array(
            [init_state.B, init_state.F, init_state.A],
            dtype=jnp.float64,
        )
        params_jax = jax.tree_util.tree_map(
            lambda v: jnp.asarray(float(v), dtype=jnp.float64), params_v15,
        )
        cost_fn_bound = jax.tree_util.Partial(
            cost_kernel, params_jax, init_arr,
        )
        return ControlSpec(
            name='fsa_high_res_v15',
            version='1.5',
            dt=dt_days,
            n_steps=n_steps,
            n_substeps=n_substeps,
            initial_state=init_arr,
            truth_params={k: float(v) for k, v in params_v15._asdict().items()},
            theta_dim=n_anchors,
            sigma_prior=sigma_prior,
            prior_mean=0.0,
            cost_fn=cost_fn_bound,
            schedule_from_theta=schedule_from_theta,
        )

    # Expose the static state for diagnostics + sanity tests.
    make_spec.cost_kernel = cost_kernel
    make_spec.schedule_from_theta = schedule_from_theta
    make_spec.n_steps = n_steps
    make_spec.n_anchors = n_anchors
    make_spec.n_substeps = n_substeps
    make_spec.dt = dt_days
    return make_spec
