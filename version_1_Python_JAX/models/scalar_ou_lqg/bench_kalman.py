"""Analytical scalar Kalman filter for the scalar OU LQG model.

Discrete-time linear-Gaussian state-space model (1-D):

    x_{k+1} = A x_k + B u_k + w_k,   w_k ~ N(0, Q)
    y_k     = C x_k + v_k,           v_k ~ N(0, R)

with A = 1 - a*dt,  B = b*dt,  Q = sigma_w**2 * dt,  C = 1,  R = sigma_v**2.

Provides:
  - kalman_filter(...)       : forward pass -> (means, covars, log_lik)
  - kalman_log_likelihood(...): just the marginal log-likelihood
  - kalman_smoother(...)     : backward smoothing (used for ground-truth
                                state estimates)

Closed-form, deterministic, no Monte Carlo. Used as ground truth for
the SMC² filter side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class KalmanResult:
    """Result of a forward Kalman pass on the scalar OU model."""
    means: np.ndarray         # (T,) posterior means x_{k|k}
    covars: np.ndarray        # (T,) posterior variances P_{k|k}
    pred_means: np.ndarray    # (T,) predictive means x_{k|k-1}
    pred_covars: np.ndarray   # (T,) predictive variances P_{k|k-1}
    log_likelihood: float     # sum of log innovation densities


def kalman_filter(
    *,
    y: np.ndarray,            # (T,) observations
    u: np.ndarray,             # (T,) control inputs (use zeros for open loop)
    a: float,
    b: float,
    sigma_w: float,
    sigma_v: float,
    dt: float,
    x0_mean: float,
    x0_var: float,
) -> KalmanResult:
    """One forward Kalman pass on the scalar OU LQG model.

    Conventions: x_{k+1} is computed from x_k and u_k (the control taken
    at step k). Observation y_k is generated from x_k + v_k.

    Returns a KalmanResult with posterior moments at every step plus the
    cumulative marginal log-likelihood log p(y_{0:T-1} | u_{0:T-1}, theta).
    """
    T = int(np.asarray(y).shape[0])
    A = 1.0 - a * dt
    B = b * dt
    Q = sigma_w ** 2 * dt
    R = sigma_v ** 2

    means = np.zeros(T)
    covars = np.zeros(T)
    pred_means = np.zeros(T)
    pred_covars = np.zeros(T)

    # Step 0: predict from prior, then update with y_0
    pred_means[0] = x0_mean
    pred_covars[0] = x0_var
    S0 = pred_covars[0] + R
    K0 = pred_covars[0] / S0
    means[0] = pred_means[0] + K0 * (y[0] - pred_means[0])
    covars[0] = (1.0 - K0) * pred_covars[0]
    log_lik = -0.5 * (np.log(2 * np.pi * S0)
                       + (y[0] - pred_means[0]) ** 2 / S0)

    for k in range(1, T):
        # Predict step k from posterior at k-1 + control u_{k-1}
        pred_means[k] = A * means[k - 1] + B * u[k - 1]
        pred_covars[k] = A * A * covars[k - 1] + Q

        # Update with y_k
        S = pred_covars[k] + R
        K = pred_covars[k] / S
        means[k] = pred_means[k] + K * (y[k] - pred_means[k])
        covars[k] = (1.0 - K) * pred_covars[k]
        log_lik += -0.5 * (np.log(2 * np.pi * S)
                            + (y[k] - pred_means[k]) ** 2 / S)

    return KalmanResult(
        means=means, covars=covars,
        pred_means=pred_means, pred_covars=pred_covars,
        log_likelihood=float(log_lik),
    )


def kalman_log_likelihood(
    *,
    y: np.ndarray,
    u: np.ndarray,
    a: float, b: float, sigma_w: float, sigma_v: float,
    dt: float, x0_mean: float, x0_var: float,
) -> float:
    """Just the marginal log-likelihood log p(y | theta, u). For MLE fits."""
    res = kalman_filter(
        y=y, u=u, a=a, b=b, sigma_w=sigma_w, sigma_v=sigma_v,
        dt=dt, x0_mean=x0_mean, x0_var=x0_var,
    )
    return res.log_likelihood


def kalman_smoother(
    *,
    y: np.ndarray, u: np.ndarray,
    a: float, b: float, sigma_w: float, sigma_v: float,
    dt: float, x0_mean: float, x0_var: float,
) -> KalmanResult:
    """Forward filter + backward Rauch-Tung-Striebel smoother.

    Returns a KalmanResult whose ``means`` and ``covars`` are the
    smoothed (full-data) posterior moments at every step.
    """
    forward = kalman_filter(
        y=y, u=u, a=a, b=b, sigma_w=sigma_w, sigma_v=sigma_v,
        dt=dt, x0_mean=x0_mean, x0_var=x0_var,
    )
    T = int(forward.means.shape[0])
    A = 1.0 - a * dt

    smoothed_means = forward.means.copy()
    smoothed_covars = forward.covars.copy()
    for k in range(T - 2, -1, -1):
        # Smoother gain: J_k = P_{k|k} A / P_{k+1|k}
        denom = forward.pred_covars[k + 1]
        if denom < 1e-30:
            continue
        J = forward.covars[k] * A / denom
        smoothed_means[k] = (
            forward.means[k]
            + J * (smoothed_means[k + 1] - forward.pred_means[k + 1])
        )
        smoothed_covars[k] = (
            forward.covars[k]
            + J * J * (smoothed_covars[k + 1] - forward.pred_covars[k + 1])
        )

    return KalmanResult(
        means=smoothed_means, covars=smoothed_covars,
        pred_means=forward.pred_means, pred_covars=forward.pred_covars,
        log_likelihood=forward.log_likelihood,
    )


def kalman_mle(
    *,
    y: np.ndarray, u: np.ndarray, dt: float,
    x0_mean: float, x0_var: float,
    a_init: float = 1.0, b_init: float = 1.0,
    sigma_w_init: float = 0.3, sigma_v_init: float = 0.2,
) -> dict:
    """Maximum-likelihood estimate of (a, b, sigma_w, sigma_v) given y, u.

    Uses scipy.optimize.minimize with a numerical L-BFGS-B search in
    log-positive parameterisation to enforce sigma_w, sigma_v > 0.
    """
    from scipy.optimize import minimize

    def neg_ll(theta_log):
        a, b, log_sw, log_sv = theta_log
        sigma_w = float(np.exp(log_sw))
        sigma_v = float(np.exp(log_sv))
        return -kalman_log_likelihood(
            y=y, u=u, a=float(a), b=float(b),
            sigma_w=sigma_w, sigma_v=sigma_v,
            dt=dt, x0_mean=x0_mean, x0_var=x0_var,
        )

    x_init = np.array([a_init, b_init,
                        np.log(sigma_w_init), np.log(sigma_v_init)])
    res = minimize(neg_ll, x_init, method='L-BFGS-B')
    a_hat, b_hat, log_sw_hat, log_sv_hat = res.x
    return {
        'a': float(a_hat),
        'b': float(b_hat),
        'sigma_w': float(np.exp(log_sw_hat)),
        'sigma_v': float(np.exp(log_sv_hat)),
        'log_likelihood': float(-res.fun),
        'converged': bool(res.success),
    }
