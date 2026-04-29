"""Sub-daily Φ-burst expansion — JAX-vectorised, deterministic.

Converts a per-day Φ schedule (`(n_days,)` array of daily training-strain
rates) to a per-15-min Φ(t) schedule (`(n_days·96,)` array) with a
**morning-loaded** Gamma-shape activity profile per wake-window:

    t = hour_of_day − wake_hour      (zero at wake)
    shape(t) = t · exp(−t / τ)        (Gamma(k=2) shape)
                                       peaks at t = τ (~3h post-wake)

Normalised so each day's integrated Φ equals 24 · Φ_daily — i.e. the
slow Banister dynamics see the same daily load whether Φ is constant
or burst-shaped.

Sleep hours [sleep_hour, wake_hour+24]: Φ = 0.

Differences from public-dev `simulation.py:generate_phi_sub_daily`:
- JAX `vmap` over days × precomputed per-bin shape (no Python loop).
- **No multiplicative noise** (the public-dev default adds 15%); here Φ
  is a deterministic control output, no realism noise needed.

Used by:
- `simulation.py:generate_phi_sub_daily` (numpy reference, kept for
  parity with public-dev callers)
- `_plant.py:StepwisePlant.advance` (closed-loop control hook)
- `tools/bench_smc_*_fsa.py` drivers (open-loop and closed-loop)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# Time-grid constants (must match simulation.py)
DT_BIN_DAYS  = 1.0 / 96.0
DT_BIN_HOURS = 24.0 / 96.0
BINS_PER_DAY = 96


def _build_per_day_envelope(wake_hour: float = 7.0,
                              sleep_hour: float = 23.0,
                              tau_hours: float = 3.0) -> jnp.ndarray:
    """Construct the per-day Φ envelope (shape (96,) in JAX, sums to 24).

    Returns an array `e[k]` for k ∈ [0, 96) such that for any daily-Φ
    value `Φ_d`, the per-bin Φ(t_k) = `Φ_d · e[k]`.
    Specifically the envelope is normalised so `sum(e[k] * dt_hours) = 24`
    regardless of the wake/sleep window — preserving the daily integral.
    """
    h = jnp.arange(BINS_PER_DAY, dtype=jnp.float64) * DT_BIN_HOURS
    in_wake = (h >= wake_hour) & (h < sleep_hour)
    t_post = jnp.where(in_wake, h - wake_hour, 0.0)
    raw_shape = jnp.where(in_wake, t_post * jnp.exp(-t_post / tau_hours), 0.0)
    # Normalise so daily integral = 24 · 1 = 24
    daily_integral = (raw_shape * DT_BIN_HOURS).sum()
    envelope = raw_shape * (24.0 / jnp.maximum(daily_integral, 1e-12))
    return envelope


@jax.jit
def expand_daily_phi_to_subdaily_jax(daily_phi: jnp.ndarray,
                                       wake_hour: float = 7.0,
                                       sleep_hour: float = 23.0,
                                       tau_hours: float = 3.0) -> jnp.ndarray:
    """JAX-vectorised expansion: `(n_days,) → (n_days·96,)`.

    Each daily Φ value is multiplied by the precomputed per-day envelope.
    Result is a flat per-15-min array compatible with the v2 simulator's
    `aux = (Phi_arr,)` contract.
    """
    envelope = _build_per_day_envelope(wake_hour, sleep_hour, tau_hours)
    # Broadcast: (n_days, 1) × (1, 96) → (n_days, 96) → flatten
    out = jnp.outer(daily_phi, envelope)
    return out.reshape(-1)


def expand_daily_phi_to_subdaily(daily_phi: np.ndarray,
                                   wake_hour: float = 7.0,
                                   sleep_hour: float = 23.0,
                                   tau_hours: float = 3.0) -> np.ndarray:
    """Numpy convenience wrapper around the JAX implementation."""
    out_jax = expand_daily_phi_to_subdaily_jax(
        jnp.asarray(daily_phi, dtype=jnp.float64),
        wake_hour=wake_hour, sleep_hour=sleep_hour, tau_hours=tau_hours,
    )
    return np.asarray(out_jax, dtype=np.float32)
