"""Generic SDE solvers (JAX + Diffrax), purely-functional rewrite.

Model-agnostic JAX/Diffrax solvers. Require:

    * The model provides ``drift_fn_jax`` and ``make_aux_fn_jax``.
    * ``JAX_ENABLE_X64=True`` is set before import (the module also
      sets it defensively at module load).

Solvers:

    * Deterministic: Kvaerno5 + ``PIDController(rtol=1e-8, atol=1e-10)``.
    * Stochastic:    ``jax.lax.scan`` Euler-Maruyama (JIT-compiled GPU
      kernel).

Refactor vs the imperative source:

    * Numpy post-compute cleanup loops over ``deterministic_indices``
      and ``bounds`` are replaced by vectorised :func:`numpy.clip` and
      stacked-comprehension assignments.
    * Inside the JIT'd inner step, the ``for idx in
      model.deterministic_indices: y = y.at[idx].set(...)`` pattern
      becomes :func:`functools.reduce` over the indices. Because
      ``deterministic_indices`` is a static Python list, this unrolls
      at trace time exactly as the original loop did.

The diffusion-type contract is unchanged:

    * ``DIFFUSION_DIAGONAL_CONSTANT``: pre-multiply noise by
      ``sigma * sqrt(dt)`` once.
    * ``DIFFUSION_DIAGONAL_STATE``: keep noise as ``sqrt(dt) * N(0,1)``
      and multiply by ``sigma * g(y)`` inside the scan, where ``g(y)``
      is the model's ``noise_scale_fn_jax``.
"""

import functools
import os
import time

os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import diffrax

from smc2fc.simulator.sde_model import (
    DIFFUSION_DIAGONAL_CONSTANT, DIFFUSION_DIAGONAL_STATE,
)


def _overwrite_deterministic_numpy(traj: np.ndarray, model, params,
                                   t_grid: np.ndarray) -> np.ndarray:
    """Returns a copy of ``traj`` with deterministic states overwritten.

    Pure: builds a new array via ``np.where``-style replacement. For
    each declared deterministic state index whose ``analytical_fn`` is
    non-None, replaces that column with the analytical evaluation of
    the function across ``t_grid``.

    Args:
        traj: Full trajectory of shape ``(T, n_states)``.
        model: The ``SDEModel`` whose ``deterministic_indices`` and
            ``states[idx].analytical_fn`` are read.
        params: Dynamics parameter dict passed to each
            ``analytical_fn``.
        t_grid: Time-grid of shape ``(T,)``.

    Returns:
        New trajectory array, same shape and dtype as ``traj``.
    """
    out = traj.copy()

    def apply_one(arr: np.ndarray, idx: int) -> np.ndarray:
        fn = model.states[idx].analytical_fn
        if fn is None:
            return arr
        new_col = np.array([fn(float(t), params) for t in t_grid])
        arr2 = arr.copy()
        arr2[:, idx] = new_col
        return arr2

    return functools.reduce(apply_one, model.deterministic_indices, out)


def _clip_to_bounds_numpy(traj: np.ndarray, model) -> np.ndarray:
    """Returns ``traj`` clipped to per-state bounds via vectorised clip.

    Pure: no in-place mutation. Builds the bound vectors once, then
    calls :func:`numpy.clip` along the state axis.

    Args:
        traj: Full trajectory of shape ``(T, n_states)``.
        model: The ``SDEModel`` whose ``bounds`` (a list of
            ``(lo, hi)`` per state) is read.

    Returns:
        New trajectory array, same shape and dtype as ``traj``.
    """
    lo = np.array([b[0] for b in model.bounds], dtype=traj.dtype)
    hi = np.array([b[1] for b in model.bounds], dtype=traj.dtype)
    return np.clip(traj, lo[None, :], hi[None, :])


def solve_deterministic_jax(model, params, init_state, t_grid,
                            exogenous=None) -> np.ndarray:
    """Solves the ODE with Diffrax Kvaerno5 (5th-order L-stable implicit).

    Args:
        model: ``SDEModel`` with ``drift_fn_jax`` and ``make_aux_fn_jax``.
        params: Dict of dynamics parameters.
        init_state: Dict of initial-state values (the model's
            ``make_y0_fn`` builds the y0 array from this).
        t_grid: Numpy array of length ``T`` giving the time-grid (in
            whatever units the drift function uses).
        exogenous: Optional dict of exogenous inputs passed to the
            model's ``make_aux_fn_jax``.

    Returns:
        Trajectory array of shape ``(T, n_states)``. Deterministic
        states (per ``model.states[i].is_deterministic``) are
        overwritten with their analytical values; every state is
        clipped to its declared bounds.

    Raises:
        ValueError: If the model lacks ``drift_fn_jax`` or
            ``make_aux_fn_jax``.
    """
    if model.drift_fn_jax is None:
        raise ValueError(
            f"Model '{model.name}' does not provide drift_fn_jax")
    if model.make_aux_fn_jax is None:
        raise ValueError(
            f"Model '{model.name}' does not provide make_aux_fn_jax")

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
    traj = _overwrite_deterministic_numpy(traj, model, params, t_grid)
    return _clip_to_bounds_numpy(traj, model)


def solve_sde_jax(model, params, init_state, t_grid, exogenous=None,
                  seed: int = 42, n_substeps: int = 10) -> np.ndarray:
    """Solves the SDE via ``jax.lax.scan`` Euler-Maruyama (JIT-compiled).

    Pre-generates the full noise tensor once, then runs an outer scan
    over the time-grid with an inner scan of ``n_substeps``
    Euler-Maruyama micro-steps per grid bin. Deterministic-state
    overwrites + bounds clipping are applied both inside the scan (at
    every micro-step) and on the assembled trajectory at exit.

    The JIT-compile time of the scan kernel is logged to stdout (a
    one-line ``"JIT compiling scan... <s>s"`` message).

    Args:
        model: ``SDEModel`` with ``drift_fn_jax``, ``diffusion_fn``,
            and (if ``DIFFUSION_DIAGONAL_STATE``) ``noise_scale_fn_jax``.
        params: Dict of dynamics parameters.
        init_state: Dict of initial-state values.
        t_grid: Numpy array of length ``T`` giving the time-grid.
        exogenous: Optional exogenous inputs passed to the aux builder.
        seed: Top-level integer seed for the noise tensor.
        n_substeps: Number of Euler-Maruyama micro-steps per grid bin.

    Returns:
        Trajectory array of shape ``(T, n_states)``. The first row is
        ``y0`` exactly; subsequent rows are the post-scan output with
        deterministic-state overwrites + bounds clipping applied.

    Raises:
        ValueError: If a required model field is missing.
        NotImplementedError: If ``model.diffusion_type`` is not one
            of ``DIFFUSION_DIAGONAL_CONSTANT`` or
            ``DIFFUSION_DIAGONAL_STATE``.
    """
    if model.drift_fn_jax is None:
        raise ValueError(
            f"Model '{model.name}' does not provide drift_fn_jax")
    if model.diffusion_fn is None:
        raise ValueError(
            f"Model '{model.name}' does not provide diffusion_fn")
    if model.diffusion_type not in (DIFFUSION_DIAGONAL_CONSTANT,
                                    DIFFUSION_DIAGONAL_STATE):
        raise NotImplementedError(
            f"Diffusion type '{model.diffusion_type}' not yet supported "
            f"in diffrax solver. Supported: DIFFUSION_DIAGONAL_CONSTANT, "
            f"DIFFUSION_DIAGONAL_STATE.")

    exogenous = exogenous or {}
    args_jax = model.make_aux_fn_jax(params, init_state, t_grid, exogenous)
    y0 = jnp.array(model.make_y0_fn(init_state, params), dtype=jnp.float64)
    n_states = model.n_states

    sigma = jnp.array(model.diffusion_fn(params), dtype=jnp.float64)

    state_dependent = (model.diffusion_type == DIFFUSION_DIAGONAL_STATE)
    if state_dependent and model.noise_scale_fn_jax is None:
        raise ValueError(
            f"Model '{model.name}' declares DIFFUSION_DIAGONAL_STATE "
            f"but does not provide noise_scale_fn_jax. Pass "
            f"noise_scale_fn_jax=<fn> to SDEModel(...).")

    lo_bounds = jnp.array([b[0] for b in model.bounds], dtype=jnp.float64)
    hi_bounds = jnp.array([b[1] for b in model.bounds], dtype=jnp.float64)

    dt_grid = float(t_grid[1] - t_grid[0])
    dt_sub = jnp.float64(dt_grid / n_substeps)
    sqrt_dt = jnp.sqrt(dt_sub)

    # Pre-generate the noise tensor once. For DIFFUSION_DIAGONAL_CONSTANT
    # we can pre-multiply by sigma * sqrt_dt; for DIFFUSION_DIAGONAL_STATE
    # we keep it at sqrt_dt scaling and apply sigma * g(y) inside the
    # scan where the current state is available.
    key = jax.random.PRNGKey(seed)
    n_grid = len(t_grid) - 1
    total_substeps = n_grid * n_substeps
    raw_noise = jax.random.normal(key, (total_substeps, n_states),
                                  dtype=jnp.float64)
    all_noise = (raw_noise * sqrt_dt
                 if state_dependent
                 else raw_noise * sigma[None, :] * sqrt_dt)
    noise_reshaped = all_noise.reshape(n_grid, n_substeps, n_states)

    # Build the deterministic-state mask declaratively.
    det_mask = jnp.array(
        [1.0 if i in model.deterministic_indices else 0.0
         for i in range(n_states)],
        dtype=jnp.float64,
    )
    # `sto_mask` is unused but retained to mirror the original module's
    # contract; left as a one-line declarative expression.
    _sto_mask = 1.0 - det_mask

    # Extract JAX params dict from args_jax (must be the first element
    # by convention). make_aux_fn_jax returns either (p_jax, ...) tuple
    # or a bare aux object. For deterministic-state overwrites we need
    # the JAX params dict.
    if (isinstance(args_jax, (tuple, list))
            and len(args_jax) > 0
            and isinstance(args_jax[0], dict)):
        params_jax = args_jax[0]
    else:
        params_jax = {k: jnp.asarray(v, dtype=jnp.float64)
                      for k, v in params.items()}

    def _apply_det_overwrite_jax(y_new: jnp.ndarray,
                                 t_next: jnp.ndarray) -> jnp.ndarray:
        """Folds deterministic-state overwrites declaratively."""
        def fold(arr: jnp.ndarray, idx: int) -> jnp.ndarray:
            fn_jax = model.states[idx].analytical_fn_jax
            if fn_jax is None:
                return arr
            return arr.at[idx].set(fn_jax(t_next, params_jax))
        return functools.reduce(fold, model.deterministic_indices, y_new)

    def inner_step(state, noise_vec):
        y, t_now = state[:n_states], state[n_states]
        dy = model.drift_fn_jax(t_now, y, args_jax)
        if state_dependent:
            scale = model.noise_scale_fn_jax(y, params_jax)
            diff = sigma * scale * noise_vec
        else:
            diff = noise_vec
        y_new = y + dt_sub * dy + diff
        t_next = t_now + dt_sub
        y_new = _apply_det_overwrite_jax(y_new, t_next)
        y_new = jnp.clip(y_new, lo_bounds, hi_bounds)
        return jnp.concatenate([y_new, t_next[None]]), None

    def outer_step(state, grid_noise):
        state_final, _ = jax.lax.scan(inner_step, state, grid_noise)
        return state_final, state_final[:n_states]

    print("    JIT compiling scan...", end="", flush=True)
    t0 = time.time()
    y0_with_t = jnp.concatenate(
        [y0, jnp.array([t_grid[0]], dtype=jnp.float64)])

    @jax.jit
    def run_scan(y0t, noise):
        _, trajectory = jax.lax.scan(outer_step, y0t, noise)
        return trajectory

    traj_body = run_scan(y0_with_t, noise_reshaped)
    traj_body.block_until_ready()
    print(f" {time.time() - t0:.1f}s")

    traj_np = np.array(traj_body, dtype=np.float64, copy=True)
    y0_np = np.array(y0, dtype=np.float64).reshape(1, n_states)
    trajectory = np.concatenate([y0_np, traj_np], axis=0)
    trajectory = _overwrite_deterministic_numpy(
        trajectory, model, params, t_grid)
    return _clip_to_bounds_numpy(trajectory, model)
