"""GK-DPF Version 3-lite — Systematic resampling + Liu-West + OT rescue.

Date:    20 April 2026
Version: 1.0

Identical to gk_dpf_v3 (v3.8) except the O(K^2) Gaussian-kernel blend is
replaced with O(K) systematic resampling.  Everything else is preserved:
Liu-West shrinkage correction, OT rescue with sigmoid interpolation,
split checkpoint structure (core PF checkpointed, OT stored).

Cost comparison (per scan step, K particles, n_st stochastic dims):
    v3 (v3.8):  O(K^2 * n_st) + O(K * r)  — GK kernel + OT
    v3-lite:    O(K) + O(K * r)            — systematic + OT

Algorithm (v3-lite)
-------------------
For each time step k = 1 ... T:
    Core (checkpointed):
      1. Propagate + weight update + marginal-LL increment.
      2. Systematic resampling (cumsum + searchsorted, O(K)).
      3. Liu-West correction: corrected = a * resampled + (1-a) * mu_w
         where a = sqrt(1 - h_norm^2), h_norm from ESS-scaled Silverman factor.
    OT rescue (stored, not recomputed):
      4. OT transport via ot_resample_lr with stop_gradient.
      5. Sigmoid interpolation:
             ot_weight = ot_max * sigmoid((ot_threshold - ESS) / ot_temp)
             output = (1 - ot_weight) * sys_lw + ot_weight * ot_out
      6. Reset log_w to uniform on observed steps.

Public API
----------
    make_gk_dpf_v3_lite_log_density(
        model, grid_obs, n_particles, bandwidth_scale,
        ot_ess_frac, ot_temperature, ot_max_weight,
        ot_rank, ot_n_iter, ot_epsilon,
        dt, seed
    ) -> Callable
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from smc2fc.estimation_model import EstimationModel
from smc2fc.filtering._gk_kernel import compute_ess
from smc2fc.filtering.resample import ot_resample_lr
from smc2fc.transforms.unconstrained import (
    build_transform_arrays,
    unconstrained_to_constrained,
    log_prior_unconstrained,
    split_theta,
)


def make_gk_dpf_v3_lite_log_density_compileonce(
        model: EstimationModel,
        n_particles: int = 500,
        bandwidth_scale: float = 1.0,
        ot_ess_frac: float = 0.05,
        ot_temperature: float = 5.0,
        ot_max_weight: float = 0.01,
        ot_rank: int = 5,
        ot_n_iter: int = 2,
        ot_epsilon: float = 0.5,
        dt: float = 5.0 / 60.0,
        t_steps: int = 24,
) -> Callable:
    """Stage L1-deep: compile-once factory for the GK-DPF v3-lite log density.

    Returns ``log_density(u, grid_obs, fixed_init_state, w_start, key0)``
    as a STABLE jitted function. Build it ONCE at bench setup; per-stride,
    wrap it via ``jax.tree_util.Partial(log_density, grid_obs, fixed_init,
    w_start, key0)`` to give BlackJAX a callable that has ``f(u)`` semantics
    but whose JAX trace cache hits across strides (because the bound args
    are pytree leaves with stable shapes/dtypes).

    Static config (n_particles, bandwidth_scale, OT params, dt, t_steps,
    model) is captured in closure. Dynamic per-stride data (grid_obs,
    fixed_init_state, w_start, key0) is passed as arguments.
    """
    if model.propagate_fn is None or model.obs_log_weight_fn is None:
        raise ValueError(
            f"Model '{model.name}' must provide propagate_fn and "
            f"obs_log_weight_fn for GK-DPF v3-lite.")

    K       = int(n_particles)
    sqrt_dt = jnp.sqrt(jnp.float64(dt))
    n_s     = model.n_states

    stochastic_idx_list = list(model.stochastic_indices)
    n_st                = len(stochastic_idx_list)
    bw_scale            = jnp.float64(bandwidth_scale)
    log_K               = jnp.log(jnp.float64(K))
    K_float             = jnp.float64(K)

    ot_threshold    = jnp.float64(K * ot_ess_frac)
    ot_temp         = jnp.float64(ot_temperature)
    ot_max          = jnp.float64(ot_max_weight)

    _ot_rank    = int(ot_rank)
    _ot_n_iter  = int(ot_n_iter)
    _ot_epsilon = float(ot_epsilon)

    T_arr = build_transform_arrays(model)

    sys_offsets = jnp.arange(K, dtype=jnp.float64) / K_float
    silverman_factor = (4.0 / (n_st + 2.0)) ** (1.0 / (n_st + 4.0))
    k_factor         = K ** (-1.0 / (n_st + 4.0))

    @jax.jit
    def log_density(u, grid_obs, fixed_init_state, w_start, key0):
        theta = unconstrained_to_constrained(u, T_arr)
        params = theta[:model.n_params]
        init = fixed_init_state
        sigma_diag = model.diffusion_fn(params)

        exogenous = {k: grid_obs[k] for k in model.exogenous_keys}
        base = model.shard_init_fn(w_start, params, exogenous, init)
        key0_l, ik = jax.random.split(key0)
        noise_init = jax.random.normal(ik, (K, n_s))
        particles = base[None, :] + sigma_diag[None, :] * sqrt_dt * noise_init
        for i, (lo, hi) in enumerate(model.state_bounds):
            particles = particles.at[:, i].set(
                jnp.clip(particles[:, i], lo, hi))

        log_w_init = jnp.zeros(K, dtype=u.dtype)

        # Stage K f32 inputs (cast once, used inside scan)
        params_f32 = params.astype(jnp.float32)
        sigma_diag_f32 = sigma_diag.astype(jnp.float32)
        grid_obs_f32 = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.float32)
                       if (hasattr(v, 'dtype')
                            and jnp.issubdtype(v.dtype, jnp.floating))
                       else v, grid_obs)

        @jax.checkpoint
        def _core_step(particles, log_w, key, k):
            key, kp, kn, kr = jax.random.split(key, 4)
            rk = jax.random.fold_in(key, jnp.int32(7))
            t = jnp.asarray((w_start + k) * dt, dtype=jnp.float32)

            noise = jax.random.normal(kn, (K, n_s), dtype=jnp.float32)
            particles_f32 = particles.astype(jnp.float32)

            def _propagate_one(y, xi):
                x_new, pred_lw = model.propagate_fn(
                    y, t, dt, params_f32, grid_obs_f32, k,
                    sigma_diag_f32, xi, kp)
                obs_lw = model.obs_log_weight_fn(
                    x_new, grid_obs_f32, k, params_f32)
                return x_new, pred_lw + obs_lw

            new_particles_f32, step_lw_f32 = jax.vmap(
                _propagate_one)(particles_f32, noise)
            new_particles = new_particles_f32.astype(u.dtype)
            step_lw = step_lw_f32.astype(u.dtype)
            log_w_pre = log_w + step_lw

            lik_inc = (jax.nn.logsumexp(log_w_pre)
                       - jax.nn.logsumexp(log_w))

            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            log_w_norm = log_w_pre - jax.nn.logsumexp(log_w_pre)
            weights = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights)
            u_shift = jax.random.uniform(kr, (), dtype=u.dtype) / K_float
            u_points = sys_offsets + u_shift
            indices = jnp.searchsorted(cumsum, u_points)
            indices = jnp.clip(indices, 0, K - 1)
            resampled = new_particles[indices]

            ess = compute_ess(log_w_pre)
            ess_frac = jnp.clip(ess / K_float, 0.0, 1.0)
            ess_factor = (1.0 - ess_frac) ** 2
            effective_scale = bw_scale * ess_factor

            h_norm = silverman_factor * k_factor * effective_scale
            a = jnp.sqrt(jnp.clip(1.0 - h_norm ** 2, 0.0, 1.0))

            mu_w = jnp.sum(weights[:, None] * new_particles, axis=0)
            sys_lw = a * resampled + (1.0 - a) * mu_w[None, :]

            return new_particles, log_w_pre, lik_inc, sys_lw, has_obs, key, rk

        def scan_step(carry, k):
            particles, log_w, ll_acc, key = carry
            (new_particles, log_w_pre, lik_inc, sys_lw,
             has_obs, key, rk) = _core_step(particles, log_w, key, k)

            if ot_max_weight >= 1e-6:
                ot_raw = ot_resample_lr(
                    new_particles, log_w_pre, rk,
                    stochastic_indices=stochastic_idx_list,
                    epsilon=_ot_epsilon, n_iter=_ot_n_iter, rank=_ot_rank)
                ot_valid = jnp.isfinite(ot_raw) & (jnp.abs(ot_raw) <= 1e10)
                ot_safe = jnp.where(ot_valid, ot_raw, sys_lw)
                for i, (lo, hi) in enumerate(model.state_bounds):
                    ot_safe = ot_safe.at[:, i].set(
                        jnp.clip(ot_safe[:, i], lo, hi))
                ot_out = jax.lax.stop_gradient(ot_safe)
                ess = compute_ess(log_w_pre)
                ot_weight = ot_max * jax.nn.sigmoid(
                    (ot_threshold - ess) / ot_temp)
                resampled = (1.0 - ot_weight) * sys_lw + ot_weight * ot_out
            else:
                resampled = sys_lw

            particles_next = jnp.where(has_obs > 0.5, resampled, new_particles)
            log_w_next = jnp.where(
                has_obs > 0.5, jnp.zeros(K, dtype=u.dtype), log_w_pre)

            for i, (lo, hi) in enumerate(model.state_bounds):
                particles_next = particles_next.at[:, i].set(
                    jnp.clip(particles_next[:, i], lo, hi))

            return (particles_next, log_w_next, ll_acc + lik_inc, key), None

        init_carry = (particles, log_w_init,
                      jnp.zeros((), dtype=u.dtype), key0_l)
        (_, lw_final, total_ll, _), _ = jax.lax.scan(
            scan_step, init_carry, jnp.arange(t_steps))

        total_ll = total_ll + jax.nn.logsumexp(lw_final) - log_K
        lp = log_prior_unconstrained(u, T_arr)
        return total_ll + lp

    @jax.jit
    def extract_state_at_step(u, grid_obs, fixed_init_state, w_start,
                               key0, target_step):
        theta = unconstrained_to_constrained(u, T_arr)
        params = theta[:model.n_params]
        init = fixed_init_state
        sigma_diag = model.diffusion_fn(params)

        exogenous = {k: grid_obs[k] for k in model.exogenous_keys}
        base = model.shard_init_fn(w_start, params, exogenous, init)
        key0_l, ik = jax.random.split(key0)
        noise_init = jax.random.normal(ik, (K, n_s))
        particles = base[None, :] + sigma_diag[None, :] * sqrt_dt * noise_init
        for i, (lo, hi) in enumerate(model.state_bounds):
            particles = particles.at[:, i].set(
                jnp.clip(particles[:, i], lo, hi))
        log_w_init = jnp.zeros(K, dtype=u.dtype)

        params_f32 = params.astype(jnp.float32)
        sigma_diag_f32 = sigma_diag.astype(jnp.float32)
        grid_obs_f32 = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.float32)
                       if (hasattr(v, 'dtype')
                            and jnp.issubdtype(v.dtype, jnp.floating))
                       else v, grid_obs)

        def scan_step_extract(carry, k):
            parts, log_w, saved_parts, saved_lw, key = carry
            key, sk, kr = jax.random.split(key, 3)
            noise = jax.random.normal(sk, (K, n_s), dtype=jnp.float32)
            parts_f32 = parts.astype(jnp.float32)

            def _prop_one(y, xi):
                t_step = jnp.asarray((w_start + k) * dt, dtype=jnp.float32)
                x_new, pred_lw = model.propagate_fn(
                    y, t_step, dt, params_f32, grid_obs_f32, k,
                    sigma_diag_f32, xi, None)
                obs_lw = model.obs_log_weight_fn(
                    x_new, grid_obs_f32, k, params_f32)
                return x_new, pred_lw + obs_lw

            new_parts_f32, step_lw_f32 = jax.vmap(_prop_one)(parts_f32, noise)
            new_parts = new_parts_f32.astype(u.dtype)
            step_lw = step_lw_f32.astype(u.dtype)
            log_w_pre = log_w + step_lw

            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            log_w_norm = log_w_pre - jax.nn.logsumexp(log_w_pre)
            weights = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights)
            u_shift = jax.random.uniform(kr, (), dtype=u.dtype) / K_float
            u_points = sys_offsets + u_shift
            indices = jnp.searchsorted(cumsum, u_points)
            indices = jnp.clip(indices, 0, K - 1)
            resampled = new_parts[indices]

            particles_next = jnp.where(has_obs > 0.5, resampled, new_parts)
            log_w_next = jnp.where(
                has_obs > 0.5, jnp.zeros(K, dtype=u.dtype), log_w_pre)

            for i, (lo, hi) in enumerate(model.state_bounds):
                particles_next = particles_next.at[:, i].set(
                    jnp.clip(particles_next[:, i], lo, hi))

            at_target = (k == target_step)
            saved_parts = jnp.where(at_target, particles_next, saved_parts)
            saved_lw = jnp.where(at_target, log_w_next, saved_lw)

            return (particles_next, log_w_next,
                    saved_parts, saved_lw, key), None

        init_carry = (particles, log_w_init,
                      jnp.zeros_like(particles), jnp.zeros_like(log_w_init),
                      key0_l)
        (_, _, saved_particles, saved_log_w, _), _ = jax.lax.scan(
            scan_step_extract, init_carry, jnp.arange(t_steps))

        w = jax.nn.softmax(saved_log_w)
        state_est = jnp.sum(w[:, None] * saved_particles, axis=0)
        return state_est

    log_density.extract_state_at_step = extract_state_at_step
    log_density._transforms = T_arr
    log_density._model = model
    log_density._method = 'gk_dpf_v3_lite_compileonce'
    return log_density


def make_gk_dpf_v3_lite_log_density(
        model: EstimationModel,
        grid_obs: Dict[str, Array],
        n_particles: int = 500,
        bandwidth_scale: float = 1.0,
        ot_ess_frac: float = 0.05,
        ot_temperature: float = 5.0,
        ot_max_weight: float = 0.01,
        ot_rank: int = 5,
        ot_n_iter: int = 2,
        ot_epsilon: float = 0.5,
        dt: float = 5.0 / 60.0,
        seed: int = 42,
        fixed_init_state: 'jnp.ndarray | None' = None,
        window_start_bin: int = 0,
) -> Callable:
    """Build the v3-lite (systematic + Liu-West + OT rescue) log-density.

    Same structure as v3.8 but with O(K) systematic resampling replacing
    the O(K^2) Gaussian-kernel blend.  Liu-West correction and OT rescue
    are identical to v3.8.

    Args:
        model: EstimationModel with propagate_fn, obs_log_weight_fn,
               diffusion_fn, shard_init_fn, state_bounds, stochastic_indices.
        grid_obs: dict of grid-aligned JAX arrays.  Optionally contains
            'has_any_obs': shape (T,) float -- 1.0 at observed steps.
        n_particles: K -- particle count.
        bandwidth_scale: Liu-West shrinkage scale (default 1.0).
            Controls the ESS-scaled shrinkage factor a.
        ot_ess_frac: ESS/K value at which ot_weight = 0.5.  Default 0.05.
        ot_temperature: sigmoid sharpness.  Default 5.0.
        ot_max_weight: maximum OT interpolation weight.  Default 0.01.
        ot_rank: Nystrom anchor count for low-rank Sinkhorn.  Default 5.
        ot_n_iter: Sinkhorn iterations.  Default 2.
        ot_epsilon: Sinkhorn entropic regularisation.  Default 0.5.
        dt: grid step size.
        seed: RNG seed.
        window_start_bin: rolling-window's GLOBAL start (in bin units).
            Used so that ``t = (window_start_bin + k) * dt`` reflects the
            ABSOLUTE time-of-day for models whose ``propagate_fn`` uses
            ``t`` to compute time-of-day-dependent quantities (e.g.
            SWAT's ``C_eff = sin(2π(t - V_c)/24 + φ)``). Defaults to 0
            for the cold-start window. Models that ignore ``t`` (like
            fsa_high_res, which does ``del t`` in ``propagate_fn`` and
            reads C from ``grid_obs['C']``) are unaffected by this
            value.

            ALSO passed as ``time_offset`` to ``model.shard_init_fn``
            so that any analytical state initialisation (e.g. SWAT's
            C(0) = sin(2π·t_start/24 + φ)) reflects the window's true
            global start time, not the within-window t=0.

    Returns:
        JIT-compiled log_density(u) -> scalar.
        Attributes: ._transforms, ._model, ._method = 'gk_dpf_v3_lite'.
    """
    if model.propagate_fn is None or model.obs_log_weight_fn is None:
        raise ValueError(
            f"Model '{model.name}' must provide propagate_fn and "
            f"obs_log_weight_fn for GK-DPF v3-lite.")

    K       = int(n_particles)
    sqrt_dt = jnp.sqrt(jnp.float64(dt))
    n_s     = model.n_states

    stochastic_idx_list = list(model.stochastic_indices)
    n_st                = len(stochastic_idx_list)
    bw_scale            = jnp.float64(bandwidth_scale)
    log_K               = jnp.log(jnp.float64(K))
    K_float             = jnp.float64(K)

    ot_threshold    = jnp.float64(K * ot_ess_frac)
    ot_temp         = jnp.float64(ot_temperature)
    ot_max          = jnp.float64(ot_max_weight)

    _ot_rank    = int(ot_rank)
    _ot_n_iter  = int(ot_n_iter)
    _ot_epsilon = float(ot_epsilon)

    first_key = next(k for k in grid_obs if k not in model.exogenous_keys)
    t_steps   = int(jnp.asarray(grid_obs[first_key]).shape[0])

    T_arr     = build_transform_arrays(model)
    exogenous = {k: jnp.asarray(grid_obs[k]) for k in model.exogenous_keys}

    # Pre-compute uniform offsets for systematic resampling (fixed)
    sys_offsets = jnp.arange(K, dtype=jnp.float64) / K_float

    # Liu-West: Silverman factors (constant across steps)
    silverman_factor = (4.0 / (n_st + 2.0)) ** (1.0 / (n_st + 4.0))
    k_factor         = K ** (-1.0 / (n_st + 4.0))

    _ot_active = ot_max_weight >= 1e-6
    print(f"    GK-DPF v3-lite (systematic + Liu-West + OT rescue, "
          f"split checkpoint):")
    print(f"      {t_steps} steps, K={K} particles, "
          f"bandwidth_scale={bandwidth_scale}")
    print(f"      Resampling: systematic at observed steps (O(K) per step)")
    print(f"      Liu-West: ESS-scaled shrinkage correction")
    if _ot_active:
        print(f"      OT:    rank={_ot_rank}, n_iter={_ot_n_iter}, "
              f"epsilon={_ot_epsilon}, max_weight={ot_max_weight}")
        print(f"      Interpolation: sigmoid((K*{ot_ess_frac} - ESS) "
              f"/ {ot_temperature})")
        print(f"      Split checkpoint: core PF checkpointed, "
              f"OT stored (no recompute)")
    else:
        print(f"      OT:    DISABLED (ot_max_weight={ot_max_weight} < 1e-6)")
        print(f"      Scan body = systematic + Liu-West only")

    # Fixed init state: [B_0, F_0, A_0] passed externally (not estimated)
    _fixed_init = fixed_init_state
    # Window-start global offset (in bin units). 0 for cold-start;
    # rolling driver passes the actual window's start_bin for w >= 1
    # so that t-of-day-dependent dynamics see global time, not within-
    # window time. fsa_high_res ignores both via `del t` and `del time_offset`.
    # Stage L1: ensure these are JAX arrays (not Python ints) so JIT
    # doesn't specialise on concrete value -> XLA cache hits across strides.
    _w_start = jnp.asarray(window_start_bin, dtype=jnp.int32)
    _seed_key = (seed if isinstance(seed, jax.Array)
                  else jax.random.PRNGKey(int(seed)))

    @jax.jit
    def log_density(u):
        theta  = unconstrained_to_constrained(u, T_arr)
        params = theta[:model.n_params]
        init   = _fixed_init  # externally fixed [B_0, F_0, A_0]
        sigma_diag   = model.diffusion_fn(params)

        base      = model.shard_init_fn(_w_start, params, exogenous, init)
        # Stage L1: use the JAX-array seed key from closure (not a fresh
        # PRNGKey(int) which would specialise on the Python int).
        key0, ik  = jax.random.split(_seed_key)
        noise_init = jax.random.normal(ik, (K, n_s))
        particles  = base[None, :] + sigma_diag[None, :] * sqrt_dt * noise_init
        for i, (lo, hi) in enumerate(model.state_bounds):
            particles = particles.at[:, i].set(
                jnp.clip(particles[:, i], lo, hi))

        log_w_init = jnp.zeros(K, dtype=u.dtype)

        # ── Split checkpoint (same as v3.8) ──────────────────────────
        # Core PF (propagation, weights, systematic+LW) is checkpointed.
        # OT computation lives OUTSIDE the checkpoint, so its forward
        # runs ONCE and intermediates are stored for backward.

        # Stage K: dual precision — cast SDE-loop inputs to f32 once.
        # Keeps every FP op inside propagate_fn (drift, diffusion, Kalman
        # fusion, Cholesky, sampling) at FP32, where consumer Blackwell
        # is ~64x faster than FP64. Log-weights stay in u.dtype (FP64)
        # for log-domain reductions.
        params_f32 = params.astype(jnp.float32)
        sigma_diag_f32 = sigma_diag.astype(jnp.float32)
        grid_obs_f32 = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.float32)
                       if (hasattr(v, 'dtype')
                            and jnp.issubdtype(v.dtype, jnp.floating))
                       else v,
            grid_obs)

        @jax.checkpoint
        def _core_step(particles, log_w, key, k):
            """Core PF: propagate, weight update, systematic+LW."""
            key, kp, kn, kr = jax.random.split(key, 4)
            rk = jax.random.fold_in(key, jnp.int32(7))
            # GLOBAL time = (window_start_bin + k) * dt — accurate
            # for time-of-day-dependent dynamics across rolling windows.
            t = jnp.asarray((_w_start + k) * dt, dtype=jnp.float32)

            noise = jax.random.normal(kn, (K, n_s), dtype=jnp.float32)
            particles_f32 = particles.astype(jnp.float32)

            def _propagate_one(y, xi):
                x_new, pred_lw = model.propagate_fn(
                    y, t, dt, params_f32, grid_obs_f32, k,
                    sigma_diag_f32, xi, kp)
                obs_lw = model.obs_log_weight_fn(
                    x_new, grid_obs_f32, k, params_f32)
                return x_new, pred_lw + obs_lw

            new_particles_f32, step_lw_f32 = jax.vmap(
                _propagate_one)(particles_f32, noise)
            new_particles = new_particles_f32.astype(u.dtype)
            step_lw = step_lw_f32.astype(u.dtype)
            log_w_pre = log_w + step_lw

            lik_inc = (jax.nn.logsumexp(log_w_pre)
                       - jax.nn.logsumexp(log_w))

            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            # ── Systematic resampling — O(K) ─────────────────────────
            log_w_norm = log_w_pre - jax.nn.logsumexp(log_w_pre)
            weights = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights)
            u_shift = jax.random.uniform(kr, (), dtype=u.dtype) / K_float
            u_points = sys_offsets + u_shift
            indices = jnp.searchsorted(cumsum, u_points)
            indices = jnp.clip(indices, 0, K - 1)
            resampled = new_particles[indices]

            # ── Liu-West correction (same as v3.8) ───────────────────
            # ESS-scaled shrinkage: a -> 1 at healthy ESS (near-identity),
            # a < 1 at degenerate ESS (shrink toward weighted mean).
            ess = compute_ess(log_w_pre)
            ess_frac = jnp.clip(ess / K_float, 0.0, 1.0)
            ess_factor = (1.0 - ess_frac) ** 2
            effective_scale = bw_scale * ess_factor

            h_norm = silverman_factor * k_factor * effective_scale
            a = jnp.sqrt(jnp.clip(1.0 - h_norm ** 2, 0.0, 1.0))

            mu_w = jnp.sum(weights[:, None] * new_particles, axis=0)
            sys_lw = a * resampled + (1.0 - a) * mu_w[None, :]

            return new_particles, log_w_pre, lik_inc, sys_lw, has_obs, key, rk

        def scan_step(carry, k):
            particles, log_w, ll_acc, key = carry

            # Core PF (checkpointed -> recomputed during backward)
            (new_particles, log_w_pre, lik_inc, sys_lw,
             has_obs, key, rk) = _core_step(particles, log_w, key, k)

            # ── OT rescue or pure systematic+LW ──
            # Python-level conditional evaluated at trace time (not runtime).
            # When ot_max_weight < 1e-6, OT is disabled and the scan body
            # compiles to systematic + Liu-West only.
            if ot_max_weight >= 1e-6:
                # OT rescue (NOT checkpointed -> forward runs once, stored)
                ot_raw = ot_resample_lr(
                    new_particles, log_w_pre, rk,
                    stochastic_indices=stochastic_idx_list,
                    epsilon=_ot_epsilon,
                    n_iter=_ot_n_iter,
                    rank=_ot_rank)
                ot_valid = jnp.isfinite(ot_raw) & (jnp.abs(ot_raw) <= 1e10)
                ot_safe  = jnp.where(ot_valid, ot_raw, sys_lw)
                for i, (lo, hi) in enumerate(model.state_bounds):
                    ot_safe = ot_safe.at[:, i].set(
                        jnp.clip(ot_safe[:, i], lo, hi))
                ot_out = jax.lax.stop_gradient(ot_safe)

                ess       = compute_ess(log_w_pre)
                ot_weight = ot_max * jax.nn.sigmoid(
                    (ot_threshold - ess) / ot_temp)

                resampled = (1.0 - ot_weight) * sys_lw + ot_weight * ot_out
            else:
                # OT disabled — systematic + Liu-West only
                resampled = sys_lw

            particles_next = jnp.where(has_obs > 0.5, resampled, new_particles)
            log_w_next     = jnp.where(
                has_obs > 0.5,
                jnp.zeros(K, dtype=u.dtype),
                log_w_pre)

            for i, (lo, hi) in enumerate(model.state_bounds):
                particles_next = particles_next.at[:, i].set(
                    jnp.clip(particles_next[:, i], lo, hi))

            return (particles_next, log_w_next, ll_acc + lik_inc, key), None

        init_carry = (particles, log_w_init,
                       jnp.zeros((), dtype=u.dtype), key0)
        (_, lw_final, total_ll, _), _ = jax.lax.scan(
            scan_step, init_carry, jnp.arange(t_steps))

        total_ll = total_ll + jax.nn.logsumexp(lw_final) - log_K
        lp       = log_prior_unconstrained(u, T_arr)
        return total_ll + lp

    log_density._transforms = T_arr
    log_density._model      = model
    log_density._method     = 'gk_dpf_v3_lite'

    @jax.jit
    def extract_state_at_step(u, target_step):
        """Run PF and return weighted particle mean at target_step.

        Used to extract smoothed [B, F, A] at the overlap point for the
        next window's fixed initial state. Uses the same core PF as
        log_density but saves particles at target_step.

        Uses the same window_start_bin offset as log_density so the
        extracted state's time-of-day-dependent dynamics are correct.
        """
        theta  = unconstrained_to_constrained(u, T_arr)
        params = theta[:model.n_params]
        init   = _fixed_init
        sigma_diag = model.diffusion_fn(params)

        base   = model.shard_init_fn(_w_start, params, exogenous, init)
        # Stage L1: same seed-key reuse as log_density above.
        key0, ik = jax.random.split(_seed_key)
        noise_init = jax.random.normal(ik, (K, n_s))
        particles  = base[None, :] + sigma_diag[None, :] * sqrt_dt * noise_init
        for i, (lo, hi) in enumerate(model.state_bounds):
            particles = particles.at[:, i].set(
                jnp.clip(particles[:, i], lo, hi))

        log_w_init = jnp.zeros(K, dtype=u.dtype)

        # Stage K: same FP32-SDE casting as in log_density.
        params_f32 = params.astype(jnp.float32)
        sigma_diag_f32 = sigma_diag.astype(jnp.float32)
        grid_obs_f32 = jax.tree_util.tree_map(
            lambda v: v.astype(jnp.float32)
                       if (hasattr(v, 'dtype')
                            and jnp.issubdtype(v.dtype, jnp.floating))
                       else v,
            grid_obs)

        def scan_step_extract(carry, k):
            parts, log_w, saved_parts, saved_lw, key = carry
            key, sk, kr = jax.random.split(key, 3)
            noise = jax.random.normal(sk, (K, n_s), dtype=jnp.float32)
            parts_f32 = parts.astype(jnp.float32)

            def _prop_one(y, xi):
                t_step = jnp.asarray((_w_start + k) * dt, dtype=jnp.float32)
                x_new, pred_lw = model.propagate_fn(
                    y, t_step, dt, params_f32, grid_obs_f32, k,
                    sigma_diag_f32, xi, None)
                obs_lw = model.obs_log_weight_fn(
                    x_new, grid_obs_f32, k, params_f32)
                return x_new, pred_lw + obs_lw

            new_parts_f32, step_lw_f32 = jax.vmap(_prop_one)(parts_f32, noise)
            new_parts = new_parts_f32.astype(u.dtype)
            step_lw = step_lw_f32.astype(u.dtype)
            log_w_pre = log_w + step_lw

            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            # Systematic resampling (no OT for speed)
            log_w_norm = log_w_pre - jax.nn.logsumexp(log_w_pre)
            weights = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights)
            u_shift = jax.random.uniform(kr, (), dtype=u.dtype) / K_float
            u_points = sys_offsets + u_shift
            indices = jnp.searchsorted(cumsum, u_points)
            indices = jnp.clip(indices, 0, K - 1)
            resampled = new_parts[indices]

            particles_next = jnp.where(has_obs > 0.5, resampled, new_parts)
            log_w_next = jnp.where(
                has_obs > 0.5,
                jnp.zeros(K, dtype=u.dtype), log_w_pre)

            for i, (lo, hi) in enumerate(model.state_bounds):
                particles_next = particles_next.at[:, i].set(
                    jnp.clip(particles_next[:, i], lo, hi))

            # Save state at target step
            at_target = (k == target_step)
            saved_parts = jnp.where(at_target,
                                     particles_next, saved_parts)
            saved_lw = jnp.where(at_target, log_w_next, saved_lw)

            return (particles_next, log_w_next,
                    saved_parts, saved_lw, key), None

        init_carry = (particles, log_w_init,
                       jnp.zeros_like(particles), jnp.zeros_like(log_w_init),
                       key0)
        (_, _, saved_particles, saved_log_w, _), _ = jax.lax.scan(
            scan_step_extract, init_carry, jnp.arange(t_steps))

        # Weighted mean of saved particles
        w = jax.nn.softmax(saved_log_w)
        state_est = jnp.sum(w[:, None] * saved_particles, axis=0)
        return state_est

    log_density.extract_state_at_step = extract_state_at_step
    return log_density
