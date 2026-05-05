"""Plant ↔ Estimator reconciliation (Mirror Test) for FSA-v5.

The v5 implementation already factored the drift into a single source of
truth: `version_3.models.fsa_v5._dynamics.drift_jax`. Both the plant
(`_plant.py:_plant_em_step`) and the estimator's `propagate_fn_v5`
import + call this same function — so they cannot structurally drift
apart at the drift level, by construction.

This test pins that contract anyway, because:

  * If a future "optimisation" pass inlines the drift back into either
    side (a common refactor temptation), the test will catch the
    divergence the moment the inline copy ships.
  * It exercises the full Plant API end-to-end (1-bin advance) to
    confirm the SDE integrator runs cleanly under the v5 6D state with
    bimodal Phi.
"""
import os
import sys
from pathlib import Path

# Force CPU + X64 BEFORE importing JAX.
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from version_3.models.fsa_v5._dynamics import drift_jax
from version_3.models.fsa_v5._plant import StepwisePlant
from version_3.models.fsa_v5.simulation import (
    DEFAULT_INIT,
    DEFAULT_PARAMS_V5,
    BINS_PER_DAY,
    DT_BIN_DAYS,
    circadian,
)
from version_3.models.fsa_v5.estimation import (
    HIGH_RES_FSA_V5_ESTIMATION,
    _PI,
    propagate_fn_v5,
    _FROZEN_V5_DYNAMICS,
)


def test_plant_and_estimator_share_drift_v5():
    """One-step Euler prediction must be bit-equivalent on both sides
    (plant calls `drift_jax`, estimator calls `_drift_jax_canonical`
    which IS `drift_jax`). With `noise = 0` and all `*_present` masks
    = 0, the estimator's `propagate_fn_v5` returns its `mu_prior` =
    `y0 + dt · drift`, which must match the plant's first Euler step.

    Catches: any future inline-drift copy in either side.
    """
    # 6D initial state from DEFAULT_INIT
    y0 = jnp.array([DEFAULT_INIT['B_0'], DEFAULT_INIT['S_0'],
                     DEFAULT_INIT['F_0'], DEFAULT_INIT['A_0'],
                     DEFAULT_INIT['KFB_0'], DEFAULT_INIT['KFS_0']],
                    dtype=jnp.float64)

    Phi_t = jnp.array([0.30, 0.30], dtype=jnp.float64)   # bimodal moderate
    t_days = 0.20    # mid-morning bin
    dt = DT_BIN_DAYS

    # --- Plant's drift evaluation (the ground-truth side). ---
    # Plant builds full params dict by merging estimated + frozen v5
    # Hill keys; replicate that here for an apples-to-apples comparison.
    p_full = {**DEFAULT_PARAMS_V5}
    drift_plant = drift_jax(y0, p_full, Phi_t)
    y_plant_euler = y0 + dt * drift_plant

    # --- Estimator side via propagate_fn_v5 with no fusion + no noise. ---
    p_vec = jnp.array(
        [DEFAULT_PARAMS_V5[name] for name in _PI], dtype=jnp.float64)
    C_t = float(circadian(jnp.array([t_days]),
                            phi=DEFAULT_PARAMS_V5.get('phi', 0.0))[0])
    grid_obs = {
        'C':                jnp.array([C_t]),
        'Phi':              jnp.array([Phi_t]),                # (1, 2)
        'hr_value':         jnp.array([0.0]),  'hr_present':     jnp.array([0.0]),
        'stress_value':     jnp.array([0.0]),  'stress_present': jnp.array([0.0]),
        'log_steps_value':  jnp.array([0.0]),  'steps_present':  jnp.array([0.0]),
        'vl_value':         jnp.array([0.0]),  'vl_present':     jnp.array([0.0]),
        'sleep_label':      jnp.array([0]),    'sleep_present':  jnp.array([0.0]),
    }
    noise_zero = jnp.zeros(6, dtype=jnp.float64)
    y_est, pred_lw = propagate_fn_v5(
        y0, t_days, dt, p_vec, grid_obs, k=0,
        sigma_diag=None, noise=noise_zero, rng_key=jax.random.PRNGKey(0))

    diff = np.abs(np.asarray(y_est) - np.asarray(y_plant_euler))
    assert float(diff.max()) < 1e-10, (
        f"Plant and estimator drift formulae disagree:\n"
        f"  plant Euler step:    {np.asarray(y_plant_euler)}\n"
        f"  estimator mu_prior:  {np.asarray(y_est)}\n"
        f"  abs diff:            {diff}\n"
        f"This means models/fsa_high_res/_dynamics.py:drift_jax has\n"
        f"drifted from the version called inside estimation.py:propagate_fn_v5.\n"
        f"Closed-loop MPC will silently produce wrong results.")


def test_plant_advance_smoke_v5():
    """Drive `StepwisePlant.advance(1 bin)` with bimodal Phi and verify
    the plant returns sensible state shapes + ranges under the v5 6D
    model.
    """
    plant = StepwisePlant(
        truth_params=dict(DEFAULT_PARAMS_V5),
        seed_offset=42,
        dt=DT_BIN_DAYS,
    )

    # advance one 15-min bin under bimodal moderate Phi.
    Phi_daily = np.array([[0.30, 0.30]])    # (n_days=1, 2)
    out = plant.advance(stride_bins=1, Phi_daily=Phi_daily)

    assert 'trajectory' in out
    assert out['trajectory'].shape == (1, 6), (
        f"plant.advance(1) trajectory shape {out['trajectory'].shape} != (1, 6)")

    # State must remain in physical bounds: B,S in [0,1]; F,A,KFB,KFS >= 0.
    B, S, F, A, KFB, KFS = out['trajectory'][0]
    assert 0.0 <= B <= 1.0, f"B={B} out of [0,1] after one bin"
    assert 0.0 <= S <= 1.0, f"S={S} out of [0,1] after one bin"
    assert F >= 0.0,        f"F={F} negative after one bin"
    assert A >= 0.0,        f"A={A} negative after one bin"
    assert KFB >= 0.0,      f"KFB={KFB} negative after one bin"
    assert KFS >= 0.0,      f"KFS={KFS} negative after one bin"

    # All 5 obs channels (HR, sleep, stress, steps, VolumeLoad) + Phi + C
    # must be in the output dict — confirm the v5 plant API contract.
    # Note: plant uses the long key 'obs_volumeload' (not 'obs_VL').
    expected_keys = ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps',
                     'obs_volumeload', 'Phi', 'C')
    for key in expected_keys:
        assert key in out, (
            f"plant output missing {key!r}; v5 plant must emit all 5 obs "
            f"channels + Phi + C per the v5 technical guide §3")


if __name__ == "__main__":
    test_plant_and_estimator_share_drift_v5()
    test_plant_advance_smoke_v5()
    print("FSA-v5 reconciliation tests PASSED.")
