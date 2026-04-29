"""Analytical scalar LQR + LQG joint cost for the scalar OU LQG model.

Discrete-time LQR setup:

    x_{k+1} = A x_k + B u_k + w_k     w_k ~ N(0, Q)
    cost   = sum_{k=0}^{T-1} (q x_k^2 + r u_k^2) + s x_T^2

The discrete-time Riccati recursion runs backwards from k=T to k=0:

    P_T = s
    K_k = (B^T P_{k+1} B + r)^{-1} B^T P_{k+1} A          (1 x 1 in scalar case)
    P_k = q + A^T P_{k+1} A - A^T P_{k+1} B K_k

The optimal control law is u_k = -K_k * x_k. In the LQG setting (state
not directly observed), the optimal control becomes u_k = -K_k * x_hat_k
where x_hat_k is the Kalman posterior mean — this is the separation
principle, and the closed-loop expected cost decomposes additively:

    J_LQG = J_LQR(perfect_state) + estimation_cost

with estimation_cost = sum_k P_{k+1} * K_k * B * (something_in_filter_var)
— in the scalar case this reduces to a closed-form expression we
compute directly via Monte Carlo (cheap; just simulate many trajectories).

This module provides:
  - lqr_riccati(...)        : backward pass -> Riccati gains K_k and value P_k
  - lqr_optimal_cost(...)   : analytical expected cost under perfect state
  - lqg_optimal_cost(...)   : Monte-Carlo expected cost under Kalman + LQR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class LQRResult:
    gains: np.ndarray   # (T,) feedback gains K_k. u*_k = -K_k * x_k
    values: np.ndarray  # (T+1,) value-function coefficients P_k


def lqr_riccati(
    *,
    a: float, b: float,
    q: float, r: float, s: float,
    sigma_w: float, dt: float, T: int,
) -> LQRResult:
    """Backward Riccati recursion for the discrete-time scalar LQR problem.

    Returns gains K_0..K_{T-1} (T entries) and values P_0..P_T (T+1 entries).
    The optimal control at step k is u*_k = -K_k * x_k.
    """
    A = 1.0 - a * dt
    B = b * dt

    P = np.zeros(T + 1)
    K = np.zeros(T)
    P[T] = s
    for k in range(T - 1, -1, -1):
        # K_k = (B^T P_{k+1} B + r)^{-1} B^T P_{k+1} A
        K[k] = (B * P[k + 1] * A) / (B * B * P[k + 1] + r)
        # P_k = q + A^T P_{k+1} A - A^T P_{k+1} B K_k
        P[k] = q + A * A * P[k + 1] - A * P[k + 1] * B * K[k]

    return LQRResult(gains=K, values=P)


def lqr_optimal_cost(
    *,
    riccati: LQRResult,
    x0_mean: float, x0_var: float,
    sigma_w: float, dt: float, T: int,
) -> float:
    """Expected optimal LQR cost under perfect state observation.

    For the LQR problem with Gaussian process noise and Gaussian initial
    state, the expected cost has a closed form:

        E[J] = E[x_0^2] * P_0 + sum_{k=0}^{T-1} P_{k+1} * Q

    where Q = sigma_w^2 * dt is the per-step process-noise variance.
    The first term is the contribution from initial-state uncertainty
    propagated through the value function; the second is the "noise
    cost" — even with the optimal feedback law, every kick of process
    noise costs `P_{k+1} * Q` of expected future cost.
    """
    Q = sigma_w ** 2 * dt
    initial_cost = (x0_mean ** 2 + x0_var) * riccati.values[0]
    noise_cost = sum(riccati.values[k + 1] * Q for k in range(T))
    return float(initial_cost + noise_cost)


def lqg_optimal_cost_monte_carlo(
    *,
    a: float, b: float,
    q: float, r: float, s: float,
    sigma_w: float, sigma_v: float,
    dt: float, T: int,
    x0_mean: float, x0_var: float,
    n_trials: int = 5000,
    seed: int = 42,
) -> dict:
    """Closed-loop LQG expected cost via Monte Carlo simulation.

    At each step k:
      - simulate true state x_{k+1} = A x_k + B u_k + sigma_w*sqrt(dt)*xi
      - simulate observation y_k = x_k + sigma_v*nu
      - run scalar Kalman filter forward to get x_hat_k
      - apply LQR feedback u_k = -K_k * x_hat_k
      - accumulate stage cost q*x_k^2 + r*u_k^2
      - terminal cost s*x_T^2

    Returns mean and stderr of the empirical cost over n_trials.
    """
    from .bench_kalman import kalman_filter
    riccati = lqr_riccati(
        a=a, b=b, q=q, r=r, s=s,
        sigma_w=sigma_w, dt=dt, T=T,
    )

    rng = np.random.default_rng(seed)
    A = 1.0 - a * dt
    B = b * dt
    sw = sigma_w * np.sqrt(dt)

    costs = np.zeros(n_trials)
    for trial in range(n_trials):
        x = x0_mean + np.sqrt(x0_var) * rng.standard_normal()
        x_hat_mean = x0_mean
        x_hat_var = x0_var
        u_history = np.zeros(T)
        cost = 0.0
        for k in range(T):
            # Observe + Kalman update
            y = x + sigma_v * rng.standard_normal()
            S = x_hat_var + sigma_v ** 2
            K = x_hat_var / S
            x_hat_mean = x_hat_mean + K * (y - x_hat_mean)
            x_hat_var = (1.0 - K) * x_hat_var
            # LQR feedback
            u = -riccati.gains[k] * x_hat_mean
            u_history[k] = u
            # Accumulate stage cost
            cost += q * x ** 2 + r * u ** 2
            # Advance true state and Kalman predictive
            x = A * x + B * u + sw * rng.standard_normal()
            x_hat_mean = A * x_hat_mean + B * u
            x_hat_var = A * A * x_hat_var + sigma_w ** 2 * dt
        cost += s * x ** 2
        costs[trial] = cost

    return {
        'mean_cost': float(costs.mean()),
        'stderr':    float(costs.std(ddof=1) / np.sqrt(n_trials)),
        'cost_std':  float(costs.std(ddof=1)),
        'n_trials':  int(n_trials),
    }


def open_loop_zero_control_cost_monte_carlo(
    *,
    a: float, b: float,
    q: float, r: float, s: float,
    sigma_w: float,
    dt: float, T: int,
    x0_mean: float, x0_var: float,
    n_trials: int = 5000,
    seed: int = 42,
) -> dict:
    """Baseline: expected cost when u_k = 0 always (no control).

    Used to confirm the LQR optimum is non-trivially better than doing
    nothing — i.e. the "headline number" that the SMC²-as-controller
    has to recover.
    """
    rng = np.random.default_rng(seed)
    A = 1.0 - a * dt
    sw = sigma_w * np.sqrt(dt)

    costs = np.zeros(n_trials)
    for trial in range(n_trials):
        x = x0_mean + np.sqrt(x0_var) * rng.standard_normal()
        cost = 0.0
        for k in range(T):
            cost += q * x ** 2     # u_k = 0 so r*u^2 contributes nothing
            x = A * x + sw * rng.standard_normal()
        cost += s * x ** 2
        costs[trial] = cost

    return {
        'mean_cost': float(costs.mean()),
        'stderr':    float(costs.std(ddof=1) / np.sqrt(n_trials)),
        'n_trials':  int(n_trials),
    }
