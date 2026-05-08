"""FSA v1.5 plant — purely-functional, stateless. Direct port of
`version_1_5_Julia/models/fsa_high_res/_plant.jl`.

Two pure functions over an immutable (B, F, A, t_bin) state record:

  plant_step(state, Phi_t, params, dt, sde_noise, obs_noise)
      → (next_state, obs)

  plant_rollout(s0, Phi_subdaily, params, dt, sde_noise_seq, obs_noise_seq)
      → (final_state, traj, obs_B, obs_F, obs_A, Phi)

No mutable class, no `!`-suffix, no RNG threaded as state. The bench
builds up a history by collecting returned values via accumulator.

RNG handling: the Julia source draws 3 SDE noise + 3 obs noise per
bin from `StableRNG(key)`. The Python+JAX equivalent takes pre-drawn
noise tuples directly so:
  (a) the diff test against Julia is straightforward (same noise → same
      output, no need to reimplement Julia's hash-based key derivation),
  (b) JAX's pure-RNG style works naturally — caller splits the key.

`PlantState` is a tiny dataclass-like NamedTuple for clarity; mirrors
the Julia `struct PlantState { bfa::SVector{3,Float64}, t_bin::Int }`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
from models.fsa_high_res.simulation import (
    INIT_STATE, params_v15_to_v1, sample_obs_bfa,
)


class PlantState(NamedTuple):
    """Immutable plant state. (B, F, A) carried as a length-3 jnp.ndarray;
    t_bin is the global bin counter (advances by 1 per `plant_step`).
    Matches the Julia `struct PlantState` at `_plant.jl:33-36`.
    """
    bfa: jnp.ndarray   # shape (3,) float64
    t_bin: int


def init_plant_state(init_state=None) -> PlantState:
    """Build a fresh PlantState at the canonical init. Matches
    `_plant.jl:43-50` (`init_plant_state`).
    """
    s = init_state if init_state is not None else INIT_STATE
    return PlantState(
        bfa=jnp.array([float(s['B']), float(s['F']), float(s['A'])],
                      dtype=jnp.float64),
        t_bin=0,
    )


def _reflect_unit(x):
    """★ @match site #2 from writeup §7.1 (Julia `_plant.jl:57-61`).
    The Julia `@match x begin x, if x < 0 end => -x; … end` is
    `if/else if/else` here — same evaluation semantics on Float.
    """
    return jnp.where(x < 0.0, -x, jnp.where(x > 1.0, 2.0 - x, x))


def plant_step(state: PlantState, Phi_t: float, params: dict,
                dt: float, sde_noise, obs_noise=None,
                obs_noise_key=None) -> tuple:
    """One Euler-Maruyama step under control `Phi_t` from `state`.

    Args:
        state: PlantState (immutable).
        Phi_t: scalar — control input at this bin.
        params: v1.5-form dict with full 14 dynamics + 3 obs noise keys.
        dt: bin width in days.
        sde_noise: shape (3,) array — pre-drawn standard-normal triple
            for the SDE step.
        obs_noise: shape (3,) array — pre-drawn standard-normal triple
            for the obs sample. If None, `obs_noise_key` (a JAX PRNGKey)
            is used to draw it via `sample_obs_bfa`.
        obs_noise_key: optional JAX PRNGKey used to sample obs noise
            when `obs_noise` is None.

    Returns:
        (PlantState, dict) — next state and `(obs_B, obs_F, obs_A)`.
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

    next_state = PlantState(bfa=next_bfa, t_bin=state.t_bin + 1)

    # Obs sample. Two paths kept for symmetry with the Julia API:
    # (a) caller supplies obs_noise — bit-identical to a Julia driver
    #     that pre-draws the noise. Used by the diff test.
    # (b) caller supplies obs_noise_key — JAX-native sampling for the
    #     bench / closed-loop path.
    if obs_noise is not None:
        obs = dict(
            obs_B=float(next_bfa[0]) + params['sigma_B_obs'] * float(obs_noise[0]),
            obs_F=float(next_bfa[1]) + params['sigma_F_obs'] * float(obs_noise[1]),
            obs_A=float(next_bfa[2]) + params['sigma_A_obs'] * float(obs_noise[2]),
        )
    else:
        if obs_noise_key is None:
            raise ValueError("plant_step needs obs_noise or obs_noise_key")
        obs = sample_obs_bfa(next_bfa, params, obs_noise_key)

    return next_state, obs


def plant_rollout(s0: PlantState, Phi_subdaily, params: dict,
                   dt: float, key0):
    """Apply `len(Phi_subdaily)` consecutive `plant_step`s from `s0`.
    Matches `_plant.jl:124-156` (`plant_rollout`).

    Args:
        s0: initial PlantState.
        Phi_subdaily: shape (n,) — per-bin Φ values.
        params: v1.5-form dict (14 dynamics + 3 obs noise).
        dt: bin width in days.
        key0: JAX PRNGKey; sub-keys for each bin's SDE + obs sampling
            are split deterministically from this.

    Returns:
        dict with keys:
            final_state: PlantState
            trajectory:  ndarray (n, 3)
            obs_B, obs_F, obs_A: ndarray (n,)
            Phi:         ndarray (n,)
    """
    n = int(len(Phi_subdaily))
    Phi_arr = np.asarray(Phi_subdaily, dtype=np.float64)
    traj = np.zeros((n, 3), dtype=np.float64)
    obs_B = np.zeros(n, dtype=np.float64)
    obs_F = np.zeros(n, dtype=np.float64)
    obs_A = np.zeros(n, dtype=np.float64)

    s = s0
    key = key0
    for k in range(n):
        key, sde_key, obs_key = jax.random.split(key, 3)
        sde_noise = jax.random.normal(sde_key, (3,), dtype=jnp.float64)
        s_next, obs = plant_step(
            s, float(Phi_arr[k]), params, dt,
            sde_noise=sde_noise, obs_noise_key=obs_key,
        )
        s = s_next
        traj[k, 0] = float(s.bfa[0])
        traj[k, 1] = float(s.bfa[1])
        traj[k, 2] = float(s.bfa[2])
        obs_B[k] = obs['obs_B']
        obs_F[k] = obs['obs_F']
        obs_A[k] = obs['obs_A']

    return dict(
        final_state=s,
        trajectory=traj,
        obs_B=obs_B,
        obs_F=obs_F,
        obs_A=obs_A,
        Phi=Phi_arr,
    )
