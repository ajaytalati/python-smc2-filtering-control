"""Sub-daily expansion of SWAT control schedules — JAX-vectorised.

The SWAT MPC controller picks a daily piecewise-constant schedule for
each of the three intervention variates V_h, V_n, V_c. This module
expands those daily schedules onto the sub-daily simulation grid.

Unlike FSA-v2's `_phi_burst.py` (which applies a morning-loaded
Gamma-shape envelope to convert daily Φ into a circadian burst),
SWAT's interventions are **clinical knobs** — the clinician sets a
target for the day (e.g. "V_c offset = -2 hours via light therapy
this evening"), and the value holds across the day until the next
day's setting takes over. Hence piecewise-constant.

Time-grid constants must match `simulation.py` (both read
`FSA_STEP_MINUTES` for inter-model parity with FSA-v2's bench).

Used by:
- `simulation.py` for forward simulation under a given schedule.
- `_plant.py:StepwisePlant.advance` for closed-loop control hook.
- `tools/bench_smc_full_mpc_swat.py` for the closed-loop driver.
"""
from __future__ import annotations

import os as _os

import jax
import jax.numpy as jnp
import numpy as np


# Time-grid constants — match the FSA-v2 convention so SWAT can run on
# the same bench step-minutes setting (``--step-minutes 60`` for h=1h).
_STEP_MIN = int(_os.environ.get('FSA_STEP_MINUTES', '15'))
if (60 * 24) % _STEP_MIN != 0:
    raise ValueError(f"FSA_STEP_MINUTES={_STEP_MIN} must divide 1440")
BINS_PER_DAY = (60 * 24) // _STEP_MIN
DT_BIN_DAYS  = 1.0 / BINS_PER_DAY
DT_BIN_HOURS = 24.0 / BINS_PER_DAY


# ── Control bounds (per OT-Control adapter, see SWAT port plan §Q3) ──

V_H_BOUNDS = (0.0,  4.0)    # vitality reserve
V_N_BOUNDS = (0.0,  5.0)    # chronic load
V_C_BOUNDS = (-12.0, 12.0)  # phase shift in hours


@jax.jit
def expand_daily_to_subdaily_jax(daily_values: jnp.ndarray) -> jnp.ndarray:
    """Piecewise-constant expansion: ``(n_days,) → (n_days * BINS_PER_DAY,)``.

    Each daily value holds across all bins of that day.

    Args:
        daily_values: array of shape (n_days,).

    Returns:
        Array of shape (n_days * BINS_PER_DAY,) where bin ``k`` of day
        ``d`` carries ``daily_values[d]``.
    """
    return jnp.repeat(daily_values, BINS_PER_DAY)


def expand_daily_to_subdaily(daily_values: np.ndarray) -> np.ndarray:
    """Numpy wrapper around the JAX expansion."""
    out_jax = expand_daily_to_subdaily_jax(
        jnp.asarray(daily_values, dtype=jnp.float64))
    return np.asarray(out_jax, dtype=np.float64)


@jax.jit
def expand_three_schedules_jax(v_h_daily: jnp.ndarray,
                                 v_n_daily: jnp.ndarray,
                                 v_c_daily: jnp.ndarray
                                 ) -> jnp.ndarray:
    """Expand all three SWAT control schedules in one call.

    Args:
        v_h_daily, v_n_daily, v_c_daily: each shape (n_days,).

    Returns:
        Array of shape (n_days * BINS_PER_DAY, 3) with columns
        (V_h, V_n, V_c). Suitable as the per-bin control input to
        ``_dynamics.drift_jax``.
    """
    v_h = expand_daily_to_subdaily_jax(v_h_daily)
    v_n = expand_daily_to_subdaily_jax(v_n_daily)
    v_c = expand_daily_to_subdaily_jax(v_c_daily)
    return jnp.stack([v_h, v_n, v_c], axis=-1)


def expand_three_schedules(v_h_daily: np.ndarray,
                            v_n_daily: np.ndarray,
                            v_c_daily: np.ndarray) -> np.ndarray:
    """Numpy wrapper around the three-schedule expansion."""
    out_jax = expand_three_schedules_jax(
        jnp.asarray(v_h_daily, dtype=jnp.float64),
        jnp.asarray(v_n_daily, dtype=jnp.float64),
        jnp.asarray(v_c_daily, dtype=jnp.float64),
    )
    return np.asarray(out_jax, dtype=np.float64)


def clip_schedules(v_h_daily: np.ndarray,
                    v_n_daily: np.ndarray,
                    v_c_daily: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clip each daily schedule to its physical bounds.

    Used to enforce controller-output safety before the schedule
    reaches the plant.
    """
    return (
        np.clip(v_h_daily, *V_H_BOUNDS),
        np.clip(v_n_daily, *V_N_BOUNDS),
        np.clip(v_c_daily, *V_C_BOUNDS),
    )
