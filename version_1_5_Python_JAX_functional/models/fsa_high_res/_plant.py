"""FSA v1.5 plant — purely-functional, stateless.

Functional rewrite of `version_1_5_Python_JAX/models/fsa_high_res/_plant.py`.
Two refactors vs the imperative source:

  (1) `plant_rollout` replaces its `for k in range(n)` loop and
      `np.empty` index-assignments with a single `jax.lax.scan` whose
      carry is `(state, key)` and whose emit is a length-6 vector
      (latents + obs). The result is bit-identical to the imperative
      loop because the per-step `jax.random.split(key, 3)` sequence is
      preserved exactly.
  (2) `plant_step` returns an `Obs` NamedTuple (not a dict) and accepts
      a `ParamsV15` NamedTuple (not a dict). `plant_step` is now fully
      JAX-traceable, with no `float()` calls in its body.

`PlantState` and `RolloutResult` are immutable `typing.NamedTuple`s.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
from models.fsa_high_res.simulation import (
    INIT_STATE, ParamsV15, InitState, Obs, params_v15_to_v1, sample_obs_bfa,
)


class PlantState(NamedTuple):
    """Immutable plant state. (B, F, A) carried as a length-3 jnp.ndarray;
    `t_bin` is the global bin counter (advances by 1 per `plant_step`).
    Matches the Julia `struct PlantState` at `_plant.jl:33-36`.
    """
    bfa: jnp.ndarray   # shape (3,) float64
    t_bin: jnp.ndarray  # scalar int — kept array-shaped for scan-compat


class RolloutResult(NamedTuple):
    """Output of `plant_rollout`. Replaces the previous dict-return."""
    final_state: PlantState
    trajectory: jnp.ndarray   # (n, 3)
    obs_B: jnp.ndarray        # (n,)
    obs_F: jnp.ndarray        # (n,)
    obs_A: jnp.ndarray        # (n,)
    Phi: jnp.ndarray          # (n,)


def init_plant_state(init_state: InitState = None) -> PlantState:
    """Build a fresh `PlantState` at the canonical init. Matches
    `_plant.jl:43-50` (`init_plant_state`).
    """
    s = INIT_STATE if init_state is None else init_state
    return PlantState(
        bfa=jnp.array([float(s.B), float(s.F), float(s.A)],
                      dtype=jnp.float64),
        t_bin=jnp.asarray(0, dtype=jnp.int32),
    )


def _reflect_unit(x):
    """★ @match site #2 from writeup §7.1 (Julia `_plant.jl:57-61`).
    The Julia `@match x begin x, if x < 0 end => -x; … end` is
    `if/else if/else` here — same evaluation semantics on Float.
    """
    return jnp.where(x < 0.0, -x, jnp.where(x > 1.0, 2.0 - x, x))


def plant_step(state: PlantState, Phi_t, params: ParamsV15,
                dt: float, sde_noise, obs_noise=None,
                obs_noise_key=None):
    """One Euler-Maruyama step under control `Phi_t` from `state`.

    Args:
        state: `PlantState` (immutable).
        Phi_t: scalar — control input at this bin.
        params: `ParamsV15` — full 14 dynamics + 3 obs noise.
        dt: bin width in days.
        sde_noise: shape (3,) array — pre-drawn standard-normal triple
            for the SDE step.
        obs_noise: shape (3,) array — pre-drawn standard-normal triple
            for the obs sample. If None, `obs_noise_key` (a JAX PRNGKey)
            is used to draw it via `sample_obs_bfa`.
        obs_noise_key: optional JAX PRNGKey used to sample obs noise
            when `obs_noise` is None.

    Returns:
        Tuple `(next_state, obs)` where `next_state` is a `PlantState`
        and `obs` is an `Obs` NamedTuple.
    """
    params_v1 = params_v15_to_v1(params)
    bfa = state.bfa
    d = drift_jax(bfa, params_v1, Phi_t)
    sigma = diffusion_state_dep(bfa, params_v1)
    noise = jnp.asarray(sde_noise, dtype=jnp.float64)
    y_pred = bfa + d * dt + sigma * jnp.sqrt(dt) * noise

    # Boundary reflection (B ∈ [0, 1], F & A ≥ 0)
    next_bfa = jnp.array([
        _reflect_unit(y_pred[0]),
        jnp.abs(y_pred[1]),
        jnp.abs(y_pred[2]),
    ])

    next_state = PlantState(
        bfa=next_bfa,
        t_bin=state.t_bin + jnp.asarray(1, dtype=state.t_bin.dtype),
    )

    # Obs sample. Two paths kept for symmetry with the Julia API:
    # (a) caller supplies obs_noise — bit-identical to a Julia driver
    #     that pre-draws the noise. Used by the diff test.
    # (b) caller supplies obs_noise_key — JAX-native sampling for the
    #     bench / closed-loop path.
    if obs_noise is not None:
        on = jnp.asarray(obs_noise, dtype=jnp.float64)
        obs = Obs(
            obs_B=next_bfa[0] + params.sigma_B_obs * on[0],
            obs_F=next_bfa[1] + params.sigma_F_obs * on[1],
            obs_A=next_bfa[2] + params.sigma_A_obs * on[2],
        )
    else:
        if obs_noise_key is None:
            raise ValueError("plant_step needs obs_noise or obs_noise_key")
        obs = sample_obs_bfa(next_bfa, params, obs_noise_key)

    return next_state, obs


def plant_rollout(s0: PlantState, Phi_subdaily, params: ParamsV15,
                   dt: float, key0) -> RolloutResult:
    """Apply `len(Phi_subdaily)` consecutive `plant_step`s from `s0`.
    Matches `_plant.jl:124-156` (`plant_rollout`).

    Pure: replaces the imperative `for k in range(n)` + `np.empty`
    index-assignment of the dict-based source with a single
    `jax.lax.scan` whose carry is `(state, key)`. The per-step
    `jax.random.split(key, 3)` sequence is identical to the imperative
    version, so trajectories are bit-identical for the same `key0`.

    Args:
        s0: initial `PlantState`.
        Phi_subdaily: shape (n,) — per-bin Φ values.
        params: `ParamsV15` (14 dynamics + 3 obs noise).
        dt: bin width in days.
        key0: JAX PRNGKey; sub-keys for each bin's SDE + obs sampling
            are split deterministically from this.

    Returns:
        `RolloutResult` NamedTuple.
    """
    Phi_arr = jnp.asarray(Phi_subdaily, dtype=jnp.float64)
    n = int(Phi_arr.shape[0])

    def step(carry, k):
        state, key = carry
        key, sde_key, obs_key = jax.random.split(key, 3)
        sde_noise = jax.random.normal(sde_key, (3,), dtype=jnp.float64)
        next_state, obs = plant_step(
            state, Phi_arr[k], params, dt,
            sde_noise=sde_noise, obs_noise_key=obs_key,
        )
        emitted = jnp.array([
            next_state.bfa[0], next_state.bfa[1], next_state.bfa[2],
            obs.obs_B, obs.obs_F, obs.obs_A,
        ])
        return (next_state, key), emitted

    (final_state, _), stacked = jax.lax.scan(
        step, (s0, key0), jnp.arange(n)
    )

    return RolloutResult(
        final_state=final_state,
        trajectory=stacked[:, 0:3],
        obs_B=stacked[:, 3],
        obs_F=stacked[:, 4],
        obs_A=stacked[:, 5],
        Phi=Phi_arr,
    )
