"""Optimised soft variant of the v5 chance-constrained cost (`soft_fast`).

Same mathematical structure as `evaluate_chance_constrained_cost_soft`
in `control_v5.py`, but with four wall-clock optimisations bundled per
[`Geminis_plan_speed_to_speed_up_soft_controller.md`](
../../../claude_plans/Geminis_plan_speed_to_speed_up_soft_controller.md):

  1. **fp32 dtype throughout** the cost path (state, params, schedule,
     weights, separatrix). The 5090 has ~64x more fp32 cores than fp64
     cores and the soft sigmoid surrogate is robust to the lost
     precision. SMC log-weights and the cost scalar itself are produced
     by `_aggregate` (re-used from `control_v5.py`); they stay fp32 here
     because the inputs are fp32, but the SMC outer loop in `smc2fc` will
     accumulate in its own dtype.

  2. **Relaxed separatrix bisection.** `_jax_find_A_sep_fast` uses a
     32-point grid + 20 bisection iters (the strict version uses 64+40).
     The soft sigmoid already smooths the basin boundary so microscopic
     precision on $A_{\\rm sep}$ is wasted work.

  3. **Sub-sampled chance-constraint bins.** The SDE rolls every bin
     (so the A-trajectory is at full 15-min resolution — the controller
     needs that signal), but the analytical separatrix
     $A_{\\rm sep}(\\Phi_t)$ is only re-computed every `bin_stride`'th
     bin (default 4 = hourly) and broadcast back. A moves slowly so the
     sub-sampling is essentially lossless.

  4. **Trimmed HMC** (`num_mcmc_steps=5`, `hmc_num_leapfrog=8`,
     `n_smc=128`) — applied at the BENCH-config level, not in this file.
     The `--cost soft_fast` branch in `bench_controller_only_fsa_v5.py`
     and `bench_smc_full_mpc_fsa_v5.py` overrides those three knobs.

The two helpers (`_aggregate`, `_stack_particle_dicts`, `_ensure_v5_keys`)
are imported from `control_v5.py` so this file is small and the shared
logic stays canonical. `_jax_mu_bar` is also re-used (pure-JAX,
already-fp32-clean).

Public entry point: `evaluate_chance_constrained_cost_soft_fast(...)`,
mirroring `evaluate_chance_constrained_cost_soft` plus a `bin_stride`
kwarg.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from version_3.models.fsa_v5._dynamics import drift_jax, TRUTH_PARAMS_V5
from version_3.models.fsa_v5.control_v5 import (
    _jax_mu_bar,
    _stack_particle_dicts,
    _ensure_v5_keys,
    _aggregate,
)


# ── Optimisation #2: relaxed-tolerance JAX bisection ─────────────────

def _jax_find_A_sep_fast(Phi_B, Phi_S, params,
                          A_min: float = 1e-4,
                          A_max: float = 2.0,
                          n_grid: int = 32,
                          n_bisect: int = 20):
    """Lower-precision counterpart to `_jax_find_A_sep`.

    Same -inf / finite / +inf three-way return semantics. 32-point grid
    + 20-iter bisection (strict version uses 64 + 40). At fp32 the
    last few bits of the bisection are noise anyway, and the soft
    sigmoid surrogate masks any small bias in the located root.
    """
    eta = params['eta']

    def g_at(A):
        return _jax_mu_bar(A, Phi_B, Phi_S, params) - eta * A * A

    A_grid = jnp.linspace(A_min, A_max, n_grid)
    g_vals = jax.vmap(g_at)(A_grid)

    is_healthy = g_vals[0] > 0.0
    sign_chg   = (g_vals[:-1] < 0.0) & (g_vals[1:] > 0.0)
    has_sep    = jnp.any(sign_chg)
    idx        = jnp.argmax(sign_chg.astype(jnp.int32))
    a0         = A_grid[idx]
    b0         = A_grid[idx + 1]

    def bisect_body(state, _):
        lo, hi = state
        mid    = 0.5 * (lo + hi)
        g_mid  = g_at(mid)
        new_lo = jnp.where(g_mid < 0.0, mid, lo)
        new_hi = jnp.where(g_mid < 0.0, hi, mid)
        return (new_lo, new_hi), None

    (lo, hi), _ = jax.lax.scan(bisect_body, (a0, b0), None, length=n_bisect)
    root = 0.5 * (lo + hi)
    return jnp.where(is_healthy, -jnp.inf,
                      jnp.where(has_sep, root, jnp.inf))


# ── Optimisation #1: fp32 RK4 forward rollout ────────────────────────

def _make_forward_rollout_fn_fast(dt):
    """fp32 drift-only RK4. dt + clipping bounds cast to fp32 inside.

    Mirrors `_make_forward_rollout_fn` from `control_v5.py` line-for-line
    except for the dtype-cast scalar locals (so JAX doesn't promote
    fp32 state through 1.0/0.0/dt Python literals).
    """
    dt_f = jnp.float32(dt)
    half = jnp.float32(0.5)
    two  = jnp.float32(2.0)
    six  = jnp.float32(6.0)
    one  = jnp.float32(1.0)
    zero = jnp.float32(0.0)

    def rk4_step(y, params, Phi_t):
        k1 = drift_jax(y, params, Phi_t)
        k2 = drift_jax(y + half * dt_f * k1, params, Phi_t)
        k3 = drift_jax(y + half * dt_f * k2, params, Phi_t)
        k4 = drift_jax(y + dt_f * k3, params, Phi_t)
        y_new = y + (dt_f / six) * (k1 + two * k2 + two * k3 + k4)
        y_new = y_new.at[0].set(jnp.clip(y_new[0], zero, one))
        y_new = y_new.at[1].set(jnp.clip(y_new[1], zero, one))
        y_new = y_new.at[2].set(jnp.maximum(y_new[2], zero))
        y_new = y_new.at[3].set(jnp.maximum(y_new[3], zero))
        y_new = y_new.at[4].set(jnp.maximum(y_new[4], zero))
        y_new = y_new.at[5].set(jnp.maximum(y_new[5], zero))
        return y_new

    @jax.jit
    def rollout(y0, params, Phi_schedule):
        def step(y, k):
            y_next = rk4_step(y, params, Phi_schedule[k])
            return y_next, y_next
        n_steps = Phi_schedule.shape[0]
        _, traj = jax.lax.scan(step, y0, jnp.arange(n_steps))
        return traj

    return rollout


# ── Optimisations #1 + #2 + #3 fused: cost-internals fast variant ────

def _compute_cost_internals_fast(theta_stacked, weights, Phi_schedule,
                                   initial_state, dt, bin_stride):
    """fp32 + sub-sampled-separatrix counterpart to `_compute_cost_internals`.

    Returns the same `(effort, A_traj_per_particle, A_sep_per_bin)` tuple
    so `_aggregate` works unchanged. Two differences vs the strict version:

      1. The separatrix `A_sep` is evaluated only at every `bin_stride`'th
         bin (template = particle-0 params, same simplification as the
         strict cost), then broadcast back to full bin resolution by
         `jnp.repeat`. A moves slowly so this is essentially lossless.
      2. The forward rollout uses `_make_forward_rollout_fn_fast` (fp32
         RK4). The A-trajectory is at full bin resolution because the
         controller needs that signal.
    """
    template = jax.tree_util.tree_map(lambda x: x[0], theta_stacked)

    n_steps = Phi_schedule.shape[0]
    Phi_sub = Phi_schedule[::bin_stride]

    def find_sep_at(Phi_t):
        return _jax_find_A_sep_fast(Phi_t[0], Phi_t[1], template)

    A_sep_sub = jax.vmap(find_sep_at)(Phi_sub)
    # Repeat each sub-sampled value bin_stride times, then crop to n_steps
    # so the indicator broadcast in _aggregate works unchanged.
    A_sep_per_bin = jnp.repeat(A_sep_sub, bin_stride)[:n_steps]

    rollout_fn = _make_forward_rollout_fn_fast(dt)

    def single_particle_A_traj(params_single):
        traj = rollout_fn(initial_state, params_single, Phi_schedule)
        return traj[:, 3]

    A_traj_per_particle = jax.vmap(single_particle_A_traj)(theta_stacked)
    effort = jnp.sum(Phi_schedule ** 2) * dt

    return effort, A_traj_per_particle, A_sep_per_bin


# ── JIT'd inner core ─────────────────────────────────────────────────

@functools.partial(jax.jit, static_argnames=('dt', 'alpha', 'A_target',
                                              'beta', 'scale', 'bin_stride'))
def _cost_soft_fast_jit(theta_stacked, weights, Phi_schedule, initial_state,
                         dt, alpha, A_target, beta, scale, bin_stride):
    effort, A_traj_pp, A_sep = _compute_cost_internals_fast(
        theta_stacked, weights, Phi_schedule, initial_state, dt, bin_stride)
    # Sigmoid surrogate, identical to control_v5._cost_soft_jit (but fp32).
    indicator = jax.nn.sigmoid(beta * (A_sep[None, :] - A_traj_pp) / scale)
    return _aggregate(A_traj_pp, A_sep, weights, dt, indicator,
                       alpha, A_target, effort)


# ── Public API ───────────────────────────────────────────────────────

def evaluate_chance_constrained_cost_soft_fast(
    theta_particles,
    weights,
    Phi_schedule,
    *,
    dt: float = 1.0 / 96,
    alpha: float = 0.05,
    A_target: float = 2.0,
    beta: float = 50.0,
    scale: float = 0.1,
    bin_stride: int = 4,
    truth_params_template: dict | None = None,
    initial_state=None,
) -> dict:
    """fp32 + relaxed-bisection + sub-sampled soft chance-constrained cost.

    Mirrors ``evaluate_chance_constrained_cost_soft`` (in
    ``control_v5.py``) but with the four optimisations from Gemini's
    plan applied (see file docstring). Same dict return shape so
    callers don't need to know which variant produced the result.

    Args:
        theta_particles: list of param dicts OR dict-of-arrays.
        weights: shape (n_particles,) particle weights.
        Phi_schedule: shape (n_steps, 2) per-bin (Phi_B, Phi_S).
        dt: bin width in days. Default 1/96 (15 min).
        alpha: chance-constraint budget.
        A_target: minimum required time-averaged A.
        beta: sigmoid surrogate temperature. Same default (50.0) as the
            strict soft variant; production may want to anneal upward.
        scale: A-units the sigmoid spans. Default 0.1.
        bin_stride: separatrix sub-sample stride. Default 4 (hourly).
        truth_params_template: dict for filling missing v5 frozen keys.
        initial_state: shape (6,) start state.

    Returns:
        Same dict shape as ``evaluate_chance_constrained_cost_soft``:
        ``mean_effort``, ``mean_A_integral``, ``violation_rate_per_particle``,
        ``weighted_violation_rate``, ``satisfies_chance_constraint``,
        ``satisfies_target``, ``A_sep_per_bin``.
    """
    if truth_params_template is None:
        truth_params_template = TRUTH_PARAMS_V5
    if initial_state is None:
        initial_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                                    dtype=jnp.float32)
    else:
        initial_state = jnp.asarray(initial_state, dtype=jnp.float32)

    # 1. Coerce theta_particles -> dict of fp32 arrays
    if isinstance(theta_particles, list):
        theta_stacked = _stack_particle_dicts(theta_particles)
    elif isinstance(theta_particles, dict):
        theta_stacked = {k: jnp.asarray(v, dtype=jnp.float32)
                          for k, v in theta_particles.items()}
    else:
        raise ValueError(
            "theta_particles must be a list of dicts or a dict of arrays; "
            f"got {type(theta_particles)}.")

    # 2. Fill in v5 frozen Hill / cascade keys, then cast EVERYTHING to fp32.
    #    `_ensure_v5_keys` (in control_v5.py) uses fp64 for the fill values
    #    so we cast after the merge.
    theta_stacked = _ensure_v5_keys(theta_stacked, truth_params_template)
    theta_stacked = {k: v.astype(jnp.float32) for k, v in theta_stacked.items()}

    weights = jnp.asarray(weights, dtype=jnp.float32)
    weights = weights / jnp.sum(weights)
    Phi_schedule = jnp.asarray(Phi_schedule, dtype=jnp.float32)

    return _cost_soft_fast_jit(theta_stacked, weights, Phi_schedule,
                                 initial_state,
                                 float(dt), float(alpha), float(A_target),
                                 float(beta), float(scale), int(bin_stride))
