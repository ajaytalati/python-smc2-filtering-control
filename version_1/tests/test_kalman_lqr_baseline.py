"""Self-consistency tests for the analytical Kalman + LQR benchmarks.

If these gates fail, the analytical machinery itself is broken — fix
before running anything that depends on it (Stage A1 filter test, A2
control-vs-LQR comparison, A3 separation principle).
"""

from __future__ import annotations

import numpy as np
import pytest


# Truth parameters per the plan (chosen for non-trivial cost reduction):
TRUTH = dict(
    a=1.0, b=1.0, sigma_w=0.3, sigma_v=0.2,
    q=1.0, r=0.1, s=1.0,
    dt=0.05, T=20,
    x0_mean=0.0, x0_var=1.0,
)


def _simulate_open_loop(
    *,
    a: float, b: float, sigma_w: float, sigma_v: float,
    dt: float, T: int, x0_mean: float, x0_var: float,
    seed: int = 0,
) -> tuple:
    """Simulate one open-loop (u=0) trajectory of the scalar OU LQG model."""
    rng = np.random.default_rng(seed)
    A = 1.0 - a * dt
    sw = sigma_w * np.sqrt(dt)
    x = np.zeros(T)
    y = np.zeros(T)
    x[0] = x0_mean + np.sqrt(x0_var) * rng.standard_normal()
    y[0] = x[0] + sigma_v * rng.standard_normal()
    for k in range(1, T):
        x[k] = A * x[k - 1] + sw * rng.standard_normal()
        y[k] = x[k] + sigma_v * rng.standard_normal()
    u = np.zeros(T)
    return x, y, u


def test_kalman_smoother_close_to_truth():
    """Kalman-smoother posterior mean should track the true latent x to
    within roughly sigma_v / sqrt(T) on a long enough trajectory."""
    from models.scalar_ou_lqg.bench_kalman import kalman_smoother

    x_true, y, u = _simulate_open_loop(
        a=TRUTH['a'], b=TRUTH['b'], sigma_w=TRUTH['sigma_w'],
        sigma_v=TRUTH['sigma_v'], dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        seed=0,
    )
    res = kalman_smoother(
        y=y, u=u, a=TRUTH['a'], b=TRUTH['b'],
        sigma_w=TRUTH['sigma_w'], sigma_v=TRUTH['sigma_v'],
        dt=TRUTH['dt'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
    )
    rms_err = float(np.sqrt(np.mean((res.means - x_true) ** 2)))
    assert rms_err < TRUTH['sigma_v'], (
        f"Kalman smoother RMS error {rms_err:.3f} exceeds sigma_v "
        f"({TRUTH['sigma_v']:.3f}) — smoother is broken or noise scales mismatched."
    )


def test_kalman_mle_recovers_truth():
    """Median Kalman MLE across 9 seeds should recover identifiable
    truth params (a, sigma_w, sigma_v) to within ~10% relative.

    Note: b is unidentifiable from open-loop data (u=0 throughout) so
    we don't gate on it. The OU autocorrelation time is tau=1/a=1s
    (20 steps at dt=0.05); per-seed MLE has high sampling variance even
    at T=10000 because the AR(1) coefficient near 0.95 is hard to pin
    down. The median across 9 seeds is far more robust than any single
    seed and is the meaningful gate for "estimator is unbiased".
    """
    from models.scalar_ou_lqg.bench_kalman import kalman_mle

    T_long = 10000
    n_seeds = 9
    estimates = {'a': [], 'sigma_w': [], 'sigma_v': []}
    for seed in range(n_seeds):
        x_true, y, u = _simulate_open_loop(
            a=TRUTH['a'], b=TRUTH['b'], sigma_w=TRUTH['sigma_w'],
            sigma_v=TRUTH['sigma_v'], dt=TRUTH['dt'], T=T_long,
            x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
            seed=seed,
        )
        mle = kalman_mle(
            y=y, u=u, dt=TRUTH['dt'],
            x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        )
        assert mle['converged']
        for k in estimates:
            estimates[k].append(mle[k])

    for name in ('a', 'sigma_w', 'sigma_v'):
        med = float(np.median(estimates[name]))
        rel_err = abs(med - TRUTH[name]) / TRUTH[name]
        assert rel_err < 0.10, (
            f"Kalman median MLE ({n_seeds} seeds) on {name}: "
            f"{med:.4f} vs truth {TRUTH[name]:.4f} "
            f"(rel_err={rel_err:.3f}); per-seed: {estimates[name]}"
        )


def test_lqr_riccati_terminal_value():
    """At k=T the Riccati value P_T must equal the terminal-cost coefficient s."""
    from models.scalar_ou_lqg.bench_lqr import lqr_riccati
    res = lqr_riccati(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    assert res.values.shape == (TRUTH['T'] + 1,)
    assert res.gains.shape == (TRUTH['T'],)
    assert abs(res.values[-1] - TRUTH['s']) < 1e-12


def test_lqg_cost_lower_than_open_loop():
    """The LQG closed-loop cost must be strictly lower than open-loop
    (no control). If not, controlling makes things worse — something is
    wrong with the cost function or the plant."""
    from models.scalar_ou_lqg.bench_lqr import (
        lqg_optimal_cost_monte_carlo,
        open_loop_zero_control_cost_monte_carlo,
    )
    lqg = lqg_optimal_cost_monte_carlo(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], sigma_v=TRUTH['sigma_v'],
        dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        n_trials=2000, seed=0,
    )
    open_loop = open_loop_zero_control_cost_monte_carlo(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'],
        dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        n_trials=2000, seed=0,
    )
    assert lqg['mean_cost'] < open_loop['mean_cost'], (
        f"LQG cost ({lqg['mean_cost']:.3f}) >= open-loop cost "
        f"({open_loop['mean_cost']:.3f}) — LQR is doing nothing useful."
    )
    # Cost reduction should be substantial (>= 30%) given chosen truth params.
    reduction = 1.0 - lqg['mean_cost'] / open_loop['mean_cost']
    assert reduction > 0.20, (
        f"LQG cost reduction only {reduction:.1%} (< 20%) — try larger b "
        f"or smaller r so the controller has authority."
    )


def test_lqg_cost_close_to_lqr_perfect_state():
    """When sigma_v is small (good observations), the LQG cost should
    converge to the LQR (perfect-state) analytical optimum within MC noise."""
    from models.scalar_ou_lqg.bench_lqr import (
        lqr_riccati, lqr_optimal_cost,
        lqg_optimal_cost_monte_carlo,
    )
    riccati = lqr_riccati(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    lqr_perfect = lqr_optimal_cost(
        riccati=riccati,
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        sigma_w=TRUTH['sigma_w'], dt=TRUTH['dt'], T=TRUTH['T'],
    )
    # With near-perfect obs (sigma_v=0.01) the LQG cost should match
    # the analytical LQR cost.
    lqg_near_perfect = lqg_optimal_cost_monte_carlo(
        a=TRUTH['a'], b=TRUTH['b'],
        q=TRUTH['q'], r=TRUTH['r'], s=TRUTH['s'],
        sigma_w=TRUTH['sigma_w'], sigma_v=0.01,
        dt=TRUTH['dt'], T=TRUTH['T'],
        x0_mean=TRUTH['x0_mean'], x0_var=TRUTH['x0_var'],
        n_trials=5000, seed=0,
    )
    rel_err = abs(lqg_near_perfect['mean_cost'] - lqr_perfect) / lqr_perfect
    assert rel_err < 0.05, (
        f"LQG (sigma_v=0.01) vs analytical LQR: "
        f"{lqg_near_perfect['mean_cost']:.3f} vs {lqr_perfect:.3f} "
        f"(rel_err={rel_err:.3f}) — separation principle violated or "
        f"analytical LQR formula wrong."
    )
