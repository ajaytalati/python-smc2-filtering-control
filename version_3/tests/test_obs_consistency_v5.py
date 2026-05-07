"""Sim ↔ Estimator observation-channel consistency for FSA-v5.

Mirrors the same-named test on the `claude/dev-sandbox-main` branch
(which covered v2's 4 channels). For v5, this version adapts to:

  - 6D latent state `[B, S, F, A, K_FB, K_FS]` (vs v2's 3D `[B, F, A]`)
  - 5 observation channels — adds VolumeLoad on top of v2's 4
    (HR / Sleep / Stress / Steps)
  - The historical `sigma_S` name collision (v5 guide §9.1) was
    resolved on 2026-05-06 by renaming the stress-channel obs noise
    to `sigma_S_obs`. So `params['sigma_S']` is now unambiguously the
    latent-S Jacobi diffusion scale (~0.008), and `params['sigma_S_obs']`
    is the stress-channel obs noise (~4.0). This test exercises the
    obs side, so it reads `sigma_S_obs`.

For each channel:
  1. Compute the LaTeX-prescribed channel mean (or Bernoulli prob for
     Sleep) directly in the test from `DEFAULT_PARAMS_V5`, using the
     formulas in the v5 technical guide §3.
  2. Run the simulator's `gen_obs_*` with the noise scale set to ~0 and
     confirm the returned observation equals the LaTeX mean (within
     float tolerance).
  3. Run the estimator's `obs_log_weight_fn` with `obs = LaTeX_mean`
     and only that channel's `*_present` mask = 1 — the resulting
     log-weight must equal the Gaussian-peak value `−log(σ·√(2π))`
     (or, for the Bernoulli sleep, `log p` where `p` is the LaTeX
     probability).

If either side drifts away from the LaTeX, exactly one of (2) or (3)
fails — pinpointing which side broke.
"""
import math
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

from version_3.models.fsa_v5.simulation import (
    DEFAULT_PARAMS_V5,
    circadian,
    gen_obs_hr,
    gen_obs_sleep,
    gen_obs_steps,
    gen_obs_stress,
    gen_obs_volumeload,
    BINS_PER_DAY,
)
from version_3.models.fsa_v5.estimation import (
    _PI,
    obs_log_weight_fn,
)


# ─────────────────────────────────────────────────────────────────────
# Test fixtures — a single non-trivial 6D state at a known C(t)
# ─────────────────────────────────────────────────────────────────────
B_state   = 0.55
S_state   = 0.40
F_state   = 0.20
A_state   = 0.45
KFB_state = 0.06
KFS_state = 0.07
T_DAYS    = 0.20   # mid-morning — circadian C ≈ cos(2π·0.20) ≈ 0.31


def _build_params() -> dict:
    """Truth params dict (DEFAULT_PARAMS_V5 unchanged).

    Post-rename (2026-05-06): ``params['sigma_S']`` is the latent-S
    Jacobi diffusion scale (~0.008); the stress-channel obs noise is
    now under ``params['sigma_S_obs']`` (~4.0). No more collision.
    """
    return dict(DEFAULT_PARAMS_V5)


def _build_estimator_params_vec(p_dict: dict) -> jnp.ndarray:
    """Pack a flat params vector indexed by `_PI` for `obs_log_weight_fn`."""
    return jnp.array([p_dict[name] for name in _PI], dtype=jnp.float64)


def _trajectory() -> np.ndarray:
    """Single-bin (6,) state trajectory at the test fixture state."""
    return np.array([[B_state, S_state, F_state, A_state, KFB_state, KFS_state]],
                     dtype=np.float64)


def _t_grid() -> np.ndarray:
    return np.array([T_DAYS], dtype=np.float64)


def _C_at_t() -> float:
    """Circadian value at the test bin (matches `grid_obs['C'][k]`)."""
    return float(circadian(_t_grid(), phi=DEFAULT_PARAMS_V5.get('phi', 0.0))[0])


def _est_log_lik(channel: str, obs_value, x_state, p_dict, sleep_label=0) -> float:
    """Single-bin estimator log-weight, with only the named channel present.

    Builds a one-bin grid_obs that flips on exactly one channel's
    presence mask so the returned log-weight is the channel's likelihood
    in isolation. Lets us isolate one channel at a time.
    """
    C_k = _C_at_t()
    grid_obs = {
        'C':                jnp.array([C_k]),
        'Phi':              jnp.array([[0.0, 0.0]]),
        'hr_value':         jnp.array([0.0]),  'hr_present':     jnp.array([0.0]),
        'stress_value':     jnp.array([0.0]),  'stress_present': jnp.array([0.0]),
        'log_steps_value':  jnp.array([0.0]),  'steps_present':  jnp.array([0.0]),
        'vl_value':         jnp.array([0.0]),  'vl_present':     jnp.array([0.0]),
        'sleep_label':      jnp.array([int(sleep_label)]),
        'sleep_present':    jnp.array([0.0]),
    }
    if channel == 'hr':
        grid_obs['hr_value'] = jnp.array([float(obs_value)])
        grid_obs['hr_present'] = jnp.array([1.0])
    elif channel == 'stress':
        grid_obs['stress_value'] = jnp.array([float(obs_value)])
        grid_obs['stress_present'] = jnp.array([1.0])
    elif channel == 'steps':
        grid_obs['log_steps_value'] = jnp.array([float(obs_value)])
        grid_obs['steps_present'] = jnp.array([1.0])
    elif channel == 'vl':
        grid_obs['vl_value'] = jnp.array([float(obs_value)])
        grid_obs['vl_present'] = jnp.array([1.0])
    elif channel == 'sleep':
        grid_obs['sleep_label'] = jnp.array([int(obs_value)])
        grid_obs['sleep_present'] = jnp.array([1.0])
    else:
        raise ValueError(f"unknown channel {channel!r}")

    p_vec = _build_estimator_params_vec(p_dict)
    x_jax = jnp.asarray(x_state, dtype=jnp.float64)
    log_w = obs_log_weight_fn(x_jax, grid_obs, k=0, params=p_vec)
    return float(log_w)


def _max_gauss_log_density(sigma: float) -> float:
    """log N(x | x, sigma^2) — the value of the log-density at its peak."""
    return -math.log(sigma * math.sqrt(2.0 * math.pi))


# ─────────────────────────────────────────────────────────────────────
# Channel tests
# ─────────────────────────────────────────────────────────────────────
# Pattern for each Gaussian channel:
#   1. Compute the LaTeX mean directly in the test.
#   2. Assert sim's noise-free output (sigma → 0) equals the LaTeX mean.
#   3. Assert estimator's log-likelihood at obs = LaTeX mean equals
#      the Gaussian-peak value −log(σ·√(2π)).


def test_hr_channel_consistency_v5():
    """HR (sleep-gated): HR_base − κ_B^HR·B + α_A^HR·A + β_C^HR·C."""
    p = _build_params()
    state = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])
    C_k = _C_at_t()

    expected_mean = (p['HR_base']
                     - p['kappa_B_HR'] * B_state
                     + p['alpha_A_HR'] * A_state
                     + p['beta_C_HR'] * C_k)

    # Sim side: sigma_HR=0, force "asleep" so sleep-gated channel emits.
    p_no_noise = dict(p, sigma_HR=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([1], dtype=np.int32)}}
    out = gen_obs_hr(_trajectory(), _t_grid(), p_no_noise,
                      aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1, "sleep-gated HR should emit one sample"
    assert math.isclose(float(out['obs_value'][0]), expected_mean, abs_tol=1e-5), \
        f"sim HR mean disagrees with LaTeX (v5 guide §3 eq:obs-HR)"

    # Estimator side
    est_lp = _est_log_lik('hr', expected_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_HR'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), \
        f"estimator HR mean disagrees: log-lik {est_lp:.6f} != peak {expected_peak:.6f}"


def test_stress_channel_consistency_v5():
    """Stress (wake-gated): S_base + k_F·F − k_A,S·A + β_C^S·C."""
    p = _build_params()
    state = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])
    C_k = _C_at_t()

    expected_mean = (p['S_base']
                     + p['k_F']    * F_state
                     - p['k_A_S']  * A_state
                     + p['beta_C_S'] * C_k)

    # Sim side: zero out the stress obs noise (sigma_S_obs); the latent-S
    # state-noise (sigma_S=0.008) is unrelated to this channel.
    p_no_noise = dict(p, sigma_S_obs=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([0], dtype=np.int32)}}
    out = gen_obs_stress(_trajectory(), _t_grid(), p_no_noise,
                          aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1, "wake-gated stress should emit one sample"
    assert math.isclose(float(out['obs_value'][0]), expected_mean, abs_tol=1e-5), \
        f"sim stress mean disagrees with LaTeX (v5 guide §3 eq:obs-S)"

    est_lp = _est_log_lik('stress', expected_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_S_obs'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), \
        f"estimator stress mean disagrees: log-lik {est_lp:.6f} != peak {expected_peak:.6f}"


def test_steps_channel_consistency_v5():
    """Steps (wake-gated, log-Gaussian):
       log_mean = μ_step0 + β_B_st·B − β_F_st·F + β_A_st·A + β_C_st·C.

    Sim returns RAW step counts (`exp(log_obs) − 1`); estimator's
    align_obs_fn log-transforms them back. This test works in log-space.
    """
    p = _build_params()
    state = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])
    C_k = _C_at_t()

    expected_log_mean = (p['mu_step0']
                         + p['beta_B_st'] * B_state
                         - p['beta_F_st'] * F_state
                         + p['beta_A_st'] * A_state
                         + p['beta_C_st'] * C_k)

    # Sim side: sigma_st=0, awake.
    p_no_noise = dict(p, sigma_st=0.0)
    prior = {'obs_sleep': {'sleep_label': np.array([0], dtype=np.int32)}}
    out = gen_obs_steps(_trajectory(), _t_grid(), p_no_noise,
                         aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) == 1
    raw_steps = float(out['obs_value'][0])
    sim_log_mean = math.log(raw_steps + 1.0)
    assert math.isclose(sim_log_mean, expected_log_mean, abs_tol=1e-5), \
        f"sim steps log-mean disagrees with LaTeX (v5 guide §3 eq:obs-st)"

    est_lp = _est_log_lik('steps', expected_log_mean, state, p)
    expected_peak = _max_gauss_log_density(p['sigma_st'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), \
        f"estimator steps log-mean disagrees: log-lik {est_lp:.6f} != peak {expected_peak:.6f}"


def test_volumeload_channel_consistency_v5():
    """VolumeLoad (training-session-gated, NEW v5 channel):
       VL = β_S^VL·S − β_F^VL·F  (no intercept, no circadian).

    The sim's gating is "mid-wake bin every 2 days" — to test the
    formula in isolation, we construct a multi-day trajectory long
    enough that `gen_obs_volumeload` actually emits a sample, then
    grab the first one. The state is constant across the trajectory
    so the per-bin VL value equals the LaTeX formula.
    """
    p = _build_params()
    state_6d = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])

    expected_mean = p['beta_S_VL'] * S_state - p['beta_F_VL'] * F_state

    # Sim side. VL emits once per 2 days at mid-wake. Build a 4-day
    # trajectory so we get at least one sample.
    n_days = 4
    n_bins = n_days * BINS_PER_DAY
    trajectory = np.tile(state_6d[None, :], (n_bins, 1))
    t_grid = np.arange(n_bins, dtype=np.float64) / BINS_PER_DAY
    p_no_noise = dict(p, sigma_VL=0.0)
    # Force entire trajectory to be "awake" so mid-wake exists in every day.
    prior = {'obs_sleep': {'sleep_label': np.zeros(n_bins, dtype=np.int32)}}
    out = gen_obs_volumeload(trajectory, t_grid, p_no_noise,
                               aux=None, prior_channels=prior, seed=0)
    assert len(out['t_idx']) >= 1, \
        f"gen_obs_volumeload emitted nothing in {n_days}d wake trajectory"
    sim_mean = float(out['obs_value'][0])
    assert math.isclose(sim_mean, expected_mean, abs_tol=1e-5), \
        f"sim VL mean disagrees with LaTeX (v5 guide §3 eq:obs-VL)"

    # Estimator side
    est_lp = _est_log_lik('vl', expected_mean, state_6d, p)
    expected_peak = _max_gauss_log_density(p['sigma_VL'])
    assert math.isclose(est_lp, expected_peak, abs_tol=1e-8), \
        f"estimator VL mean disagrees: log-lik {est_lp:.6f} != peak {expected_peak:.6f}"


def test_sleep_channel_consistency_v5():
    """Sleep (Bernoulli): p = sigmoid(k_C·C + k_A·A − c_tilde).

    Estimator: log P(label=1 | A, C) must equal log p; symmetric for label=0.
    Sim: empirical frequency of label=1 over many bins must converge to p.
    """
    p = _build_params()
    state_6d = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])
    C_k = _C_at_t()

    # 1. LaTeX probability
    z = p['k_C'] * C_k + p['k_A'] * A_state - p['c_tilde']
    p_sleep_LaTeX = 1.0 / (1.0 + math.exp(-z))

    # 2. Estimator
    lp_label1 = _est_log_lik('sleep', 1, state_6d, p)
    lp_label0 = _est_log_lik('sleep', 0, state_6d, p)
    assert math.isclose(lp_label1, math.log(p_sleep_LaTeX), abs_tol=1e-10), \
        f"estimator P(label=1) disagrees: {lp_label1} vs {math.log(p_sleep_LaTeX)}"
    assert math.isclose(lp_label0, math.log(1.0 - p_sleep_LaTeX), abs_tol=1e-10), \
        f"estimator P(label=0) disagrees: {lp_label0} vs {math.log(1.0 - p_sleep_LaTeX)}"

    # 3. Sim — empirical frequency of label=1 over many independent bins
    n_bins = 30000
    trajectory = np.tile(state_6d[None, :], (n_bins, 1))
    t_grid = np.full(n_bins, T_DAYS, dtype=np.float64)
    out = gen_obs_sleep(trajectory, t_grid, p, aux=None,
                          prior_channels=None, seed=12345)
    p_emp = float(out['sleep_label'].mean())
    assert abs(p_emp - p_sleep_LaTeX) < 0.012, \
        f"sim sleep marginal disagrees: empirical {p_emp:.4f} vs expected {p_sleep_LaTeX:.4f}"


# ─────────────────────────────────────────────────────────────────────
# Belt-and-braces — sweeps over a known offset to catch
# D1/D2-class regressions (where one side silently drops a term).
# ─────────────────────────────────────────────────────────────────────


def test_hr_offset_actually_shifts_the_signal_v5():
    """Sweep `HR_base` and check sim HR mean shifts 1:1.

    This is the FSA-v5 analogue of the SWAT D1 regression catcher.
    """
    state_6d = np.array([B_state, S_state, F_state, A_state, KFB_state, KFS_state])
    p_zero = dict(DEFAULT_PARAMS_V5)
    p_zero['HR_base'] = 0.0
    prior = {'obs_sleep': {'sleep_label': np.array([1], dtype=np.int32)}}

    for shift in (-10.0, -3.0, 5.0, 17.0):
        p_shifted = dict(p_zero, HR_base=shift, sigma_HR=0.0)
        out_shifted = gen_obs_hr(_trajectory(), _t_grid(), p_shifted,
                                   aux=None, prior_channels=prior, seed=0)
        out_baseline = gen_obs_hr(_trajectory(), _t_grid(),
                                    dict(p_zero, sigma_HR=0.0),
                                    aux=None, prior_channels=prior, seed=0)
        delta = float(out_shifted['obs_value'][0]) - float(out_baseline['obs_value'][0])
        assert math.isclose(delta, shift, abs_tol=1e-5), \
            f"HR_base={shift} should shift mean by {shift}; got {delta} (D1-class regression)"


if __name__ == "__main__":
    test_hr_channel_consistency_v5()
    test_stress_channel_consistency_v5()
    test_steps_channel_consistency_v5()
    test_volumeload_channel_consistency_v5()
    test_sleep_channel_consistency_v5()
    test_hr_offset_actually_shifts_the_signal_v5()
    print("All FSA-v5 sim/est obs-consistency tests PASSED.")
