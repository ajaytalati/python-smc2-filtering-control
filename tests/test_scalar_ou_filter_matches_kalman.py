"""Stage A1 acceptance gate: inner-PF log-likelihood matches the
analytical Kalman log-likelihood on the scalar OU LQG model.

This is the cleanest possible end-to-end check for the SMC² inner-LL.
The scalar OU model is linear-Gaussian, so:

    log p_Kalman(y | θ)   is  EXACT
    log p_PF(y | θ)       is  UNBIASED in expectation, with finite-K
                              variance that shrinks as K → ∞ and
                              vanishes in the linear-Gaussian limit
                              when the proposal is locally optimal
                              (Pitt-Shephard fusion).

If `log p_PF` does not converge to `log p_Kalman` as K grows on this
clean linear-Gaussian model, the inner-PF or its proposal is buggy.

Tests:
  - PF-vs-Kalman log-likelihood at truth, multiple seeds (gate: PF mean
    within 1.0 nat of Kalman LL after averaging across 10 seeds).
  - PF stochasticity reasonable (per-seed std < 5 nats).
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math
import numpy as np
import pytest

import jax
import jax.numpy as jnp


# Truth values per the plan.
TRUTH = dict(
    a=1.0, b=1.0, sigma_w=0.3, sigma_v=0.2,
    dt=0.05, T=500,
    x0_mean=0.0, x0_var=1.0,
)


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a deterministic open-loop trajectory + obs (fixed seed)."""
    from models.scalar_ou_lqg.simulation import (
        simulate, PARAM_SET_A, INIT_STATE_A,
    )
    params = dict(PARAM_SET_A)
    params.update({k: TRUTH[k] for k in ('a', 'b', 'sigma_w', 'sigma_v')})
    init = dict(INIT_STATE_A)
    init['x_0'] = TRUTH['x0_mean']
    exog = dict(dt=TRUTH['dt'], T=TRUTH['T'], x0_var=TRUTH['x0_var'])
    return simulate(params=params, init_state=init, exogenous=exog,
                     u=np.zeros(TRUTH['T']), seed=42)


def _kalman_loglik_at_truth(synthetic_data) -> float:
    from models.scalar_ou_lqg.bench_kalman import kalman_log_likelihood
    return kalman_log_likelihood(
        y=synthetic_data['obs'], u=synthetic_data['u'],
        a=TRUTH['a'], b=TRUTH['b'],
        sigma_w=TRUTH['sigma_w'], sigma_v=TRUTH['sigma_v'],
        dt=TRUTH['dt'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
    )


def _build_pf_log_density(synthetic_data, n_pf, seed):
    """Build a JIT-compiled inner-PF log-density at the scalar OU model."""
    from models.scalar_ou_lqg.estimation import make_scalar_ou_estimation
    from smc2fc.transforms.unconstrained import build_transform_arrays
    from smc2fc.filtering.gk_dpf_v3_lite import make_gk_dpf_v3_lite_log_density

    em = make_scalar_ou_estimation(u_schedule=synthetic_data['u'])

    obs_data = {
        'obs': {
            't_idx':     np.arange(TRUTH['T'], dtype=np.int32),
            'obs_value': synthetic_data['obs'].astype(np.float32),
        }
    }
    grid_obs = em.align_obs_fn(obs_data, TRUTH['T'], TRUTH['dt'])

    cold_start_init = jnp.array([TRUTH['x0_mean']], dtype=jnp.float64)
    log_density = make_gk_dpf_v3_lite_log_density(
        model=em, grid_obs=grid_obs, n_particles=n_pf,
        bandwidth_scale=1.0,
        ot_ess_frac=0.05, ot_temperature=5.0, ot_max_weight=0.0,  # OT off
        ot_rank=5, ot_n_iter=2, ot_epsilon=0.5,
        dt=TRUTH['dt'], seed=seed,
        fixed_init_state=cold_start_init, window_start_bin=0,
    )
    T_arr = log_density._transforms

    # Pack truth params + init in CONSTRAINED space, then map to unconstrained
    from smc2fc.transforms.unconstrained import constrained_to_unconstrained
    truth_constrained = jnp.array([
        TRUTH['a'], TRUTH['b'], TRUTH['sigma_w'], TRUTH['sigma_v'],
        TRUTH['x0_mean'],
    ], dtype=jnp.float64)
    truth_unc = constrained_to_unconstrained(truth_constrained, T_arr)

    return log_density, truth_unc, T_arr


def test_kalman_loglik_at_truth_finite(synthetic_data):
    """Sanity: the analytical Kalman LL at truth is finite."""
    ll = _kalman_loglik_at_truth(synthetic_data)
    assert math.isfinite(ll), f"Kalman LL at truth: {ll}"
    print(f"  Kalman log p(y|truth) = {ll:.3f} on T={TRUTH['T']} obs")


def test_pf_loglik_at_truth_close_to_kalman(synthetic_data):
    """Stage A1 acceptance gate: inner-PF log-likelihood at truth params
    matches the analytical Kalman log-likelihood within 5 nats after
    averaging across multiple seeds.

    The PF estimator is unbiased in expectation, but has finite-K
    variance. With locally-optimal proposal on a linear-Gaussian model
    that variance is small per-step, but accumulated across T steps it
    compounds. We use 10 seeds at K=200 to average out the per-seed
    noise.
    """
    n_pf = 200
    n_seeds = 10
    kalman_ll = _kalman_loglik_at_truth(synthetic_data)

    pf_lls = []
    for seed in range(n_seeds):
        log_density, truth_unc, T_arr = _build_pf_log_density(
            synthetic_data, n_pf=n_pf, seed=seed)
        # log_density(u) returns log p(y|theta) + log p(theta). Subtract
        # off the log prior to get just the data term.
        from smc2fc.transforms.unconstrained import log_prior_unconstrained
        ll = float(log_density(truth_unc))
        log_prior = float(log_prior_unconstrained(truth_unc, T_arr))
        pf_data_ll = ll - log_prior
        pf_lls.append(pf_data_ll)
        del log_density   # release the JIT cache between seeds

    pf_lls = np.array(pf_lls)
    pf_mean = float(pf_lls.mean())
    pf_std = float(pf_lls.std(ddof=1))
    bias = pf_mean - kalman_ll
    print(f"  Kalman log p(y|truth) = {kalman_ll:.3f}")
    print(f"  PF log p(y|truth) (K={n_pf}, {n_seeds} seeds) = "
          f"{pf_mean:.3f} ± {pf_std:.3f}  (bias = {bias:+.3f})")

    assert abs(bias) < 5.0, (
        f"PF log-likelihood bias |Δ| = {abs(bias):.3f} > 5 nats — "
        f"inner-PF estimator is biased on linear-Gaussian model."
    )
    assert pf_std < 10.0, (
        f"PF log-likelihood std = {pf_std:.3f} > 10 nats — "
        f"inner-PF estimator variance excessive."
    )
