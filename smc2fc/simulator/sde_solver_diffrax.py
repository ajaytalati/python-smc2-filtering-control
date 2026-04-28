"""
sde_solver_diffrax.py — Generic SDE Solvers (JAX + Diffrax)
=============================================================
Date:    19 April 2026
Version: 1.1

Model-agnostic JAX/Diffrax solvers.  Require:
  - The model provides drift_fn_jax and make_aux_fn_jax
  - JAX_ENABLE_X64=True is set before import

Deterministic: Kvaerno5 + PIDController(rtol=1e-8, atol=1e-10)
Stochastic:    jax.lax.scan Euler-Maruyama (JIT-compiled GPU kernel)

Changelog from 1.0:
  - solve_sde_jax now supports DIFFUSION_DIAGONAL_STATE via
    model.noise_scale_fn_jax(y, params_jax).  The DIFFUSION_DIAGONAL_
    CONSTANT code path is unchanged.
"""

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax

from smc2fc.simulator.sde_model import (SDEModel, DIFFUSION_DIAGONAL_CONSTANT,
                        DIFFUSION_DIAGONAL_STATE)


def solve_deterministic_jax(model, params, init_state, t_grid, exogenous=None):
    """Solve the ODE with Diffrax Kvaerno5 (5th-order L-stable implicit)."""
    if model.drift_fn_jax is None:
        raise ValueError(f"Model '{model.name}' does not provide drift_fn_jax")
    if model.make_aux_fn_jax is None:
        raise ValueError(f"Model '{model.name}' does not provide make_aux_fn_jax")

    exogenous = exogenous or {}
    args_jax = model.make_aux_fn_jax(params, init_state, t_grid, exogenous)
    y0 = jnp.array(model.make_y0_fn(init_state, params), dtype=jnp.float64)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model.drift_fn_jax),
        diffrax.Kvaerno5(),
        t0=float(t_grid[0]), t1=float(t_grid[-1]),
        dt0=0.001,
        y0=y0, args=args_jax,
        saveat=diffrax.SaveAt(ts=jnp.array(t_grid, dtype=jnp.float64)),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        max_steps=1_000_000,
    )

    traj = np.array(sol.ys, dtype=np.float64, copy=True)

    # Overwrite deterministic states
    for idx in model.deterministic_indices:
        fn = model.states[idx].analytical_fn
        if fn is not None:
            traj[:, idx] = np.array([fn(float(t), params) for t in t_grid])

    # Apply constraints
    for i, (lo, hi) in enumerate(model.bounds):
        traj[:, i] = np.clip(traj[:, i], lo, hi)

    return traj


def solve_sde_jax(model, params, init_state, t_grid, exogenous=None,
                   seed=42, n_substeps=10):
    """Solve the SDE via jax.lax.scan Euler-Maruyama (JIT-compiled)."""
    if model.drift_fn_jax is None:
        raise ValueError(f"Model '{model.name}' does not provide drift_fn_jax")

    exogenous = exogenous or {}
    args_jax = model.make_aux_fn_jax(params, init_state, t_grid, exogenous)
    y0 = jnp.array(model.make_y0_fn(init_state, params), dtype=jnp.float64)
    n_states = model.n_states

    # Validate diffusion type and resolve state-dependent scaling fn
    if model.diffusion_fn is None:
        raise ValueError(f"Model '{model.name}' does not provide diffusion_fn")
    if model.diffusion_type not in (DIFFUSION_DIAGONAL_CONSTANT,
                                     DIFFUSION_DIAGONAL_STATE):
        raise NotImplementedError(
            f"Diffusion type '{model.diffusion_type}' not yet supported "
            f"in diffrax solver.  Supported: DIFFUSION_DIAGONAL_CONSTANT, "
            f"DIFFUSION_DIAGONAL_STATE.")

    sigma = jnp.array(model.diffusion_fn(params), dtype=jnp.float64)

    state_dependent = (model.diffusion_type == DIFFUSION_DIAGONAL_STATE)
    if state_dependent and model.noise_scale_fn_jax is None:
        raise ValueError(
            f"Model '{model.name}' declares DIFFUSION_DIAGONAL_STATE "
            f"but does not provide noise_scale_fn_jax.  Pass "
            f"noise_scale_fn_jax=<fn> to SDEModel(...).")

    # Bounds
    lo_bounds = jnp.array([b[0] for b in model.bounds], dtype=jnp.float64)
    hi_bounds = jnp.array([b[1] for b in model.bounds], dtype=jnp.float64)

    dt_grid = float(t_grid[1] - t_grid[0])
    dt_sub = jnp.float64(dt_grid / n_substeps)
    sqrt_dt = jnp.sqrt(dt_sub)

    # Pre-generate all noise.
    # For DIFFUSION_DIAGONAL_CONSTANT we can pre-scale by sigma * sqrt_dt
    # (the original optimisation).  For DIFFUSION_DIAGONAL_STATE we pre-
    # scale by sqrt_dt only; sigma(y) is applied inside the scan where
    # the current state is available.
    key = jax.random.PRNGKey(seed)
    n_grid = len(t_grid) - 1
    total_substeps = n_grid * n_substeps
    all_noise = jax.random.normal(key, (total_substeps, n_states),
                                   dtype=jnp.float64)
    if state_dependent:
        all_noise = all_noise * sqrt_dt
    else:
        all_noise = all_noise * sigma[None, :] * sqrt_dt
    noise_reshaped = all_noise.reshape(n_grid, n_substeps, n_states)

    # Deterministic state masks
    det_mask = jnp.zeros(n_states, dtype=jnp.float64)
    for idx in model.deterministic_indices:
        det_mask = det_mask.at[idx].set(1.0)
    sto_mask = 1.0 - det_mask

    # Extract JAX params dict from args_jax (must be the first element by convention)
    # make_aux_fn_jax returns either (p_jax, ...) tuple or a bare aux object.
    # For deterministic-state overwrites we need the JAX params dict.
    if isinstance(args_jax, (tuple, list)) and len(args_jax) > 0 and isinstance(args_jax[0], dict):
        params_jax = args_jax[0]
    else:
        # Fallback: build directly
        params_jax = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}

    def inner_step(state, noise_vec):
        y, t_now = state[:n_states], state[n_states]
        dy = model.drift_fn_jax(t_now, y, args_jax)
        if state_dependent:
            # noise_vec is N(0,1) * sqrt_dt; multiply by sigma * g(y)
            scale = model.noise_scale_fn_jax(y, params_jax)
            diff = sigma * scale * noise_vec
        else:
            # noise_vec already = sigma * sqrt_dt * N(0,1)
            diff = noise_vec
        y_new = y + dt_sub * dy + diff
        t_next = t_now + dt_sub
        # Overwrite deterministic states using JAX-compatible analytical function
        for idx in model.deterministic_indices:
            fn_jax = model.states[idx].analytical_fn_jax
            if fn_jax is not None:
                y_new = y_new.at[idx].set(fn_jax(t_next, params_jax))
        # Constraints
        y_new = jnp.clip(y_new, lo_bounds, hi_bounds)
        return jnp.concatenate([y_new, t_next[None]]), None

    def outer_step(state, grid_noise):
        state_final, _ = jax.lax.scan(inner_step, state, grid_noise)
        return state_final, state_final[:n_states]

    import time
    print("    JIT compiling scan...", end="", flush=True)
    t0 = time.time()

    y0_with_t = jnp.concatenate([y0, jnp.array([t_grid[0]], dtype=jnp.float64)])

    @jax.jit
    def run_scan(y0t, noise):
        _, trajectory = jax.lax.scan(outer_step, y0t, noise)
        return trajectory

    traj_body = run_scan(y0_with_t, noise_reshaped)
    traj_body.block_until_ready()
    print(f" {time.time()-t0:.1f}s")

    traj_np = np.array(traj_body, dtype=np.float64, copy=True)
    y0_np = np.array(y0, dtype=np.float64).reshape(1, n_states)
    trajectory = np.concatenate([y0_np, traj_np], axis=0)

    # Final overwrite of deterministic states
    for idx in model.deterministic_indices:
        fn = model.states[idx].analytical_fn
        if fn is not None:
            trajectory[:, idx] = np.array([fn(float(t), params) for t in t_grid])

    for i, (lo, hi) in enumerate(model.bounds):
        trajectory[:, i] = np.clip(trajectory[:, i], lo, hi)

    return trajectory
