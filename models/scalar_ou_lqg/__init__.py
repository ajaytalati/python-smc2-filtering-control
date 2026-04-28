"""Scalar OU LQG — the simplest principled filter+control test model.

State-space model (continuous-time):
    dx_t = -a x_t dt + b u_t dt + sigma_w dW_t
    y_t  = x_t + sigma_v nu_t                 (Gaussian obs each step)
    J(u) = sum_k (q x_k^2 + r u_k^2) dt + s x_T^2

Discrete-time forward (Euler):
    x_{k+1} = (1 - a*dt) x_k + b*dt*u_k + sqrt(dt)*sigma_w*xi_k
    y_k     = x_k + sigma_v*nu_k

Truth values (chosen so closed-loop LQG cost reduction is detectable):
    a = 1.0, b = 1.0, sigma_w = 0.3, sigma_v = 0.2,
    q = 1.0, r = 0.1, s = 1.0, T = 1.0 s, dt = 0.05 -> 20 steps.

Closed-form benchmarks live in:
  - bench_kalman.py : analytical scalar Kalman filter
  - bench_lqr.py    : analytical scalar LQR + LQG joint cost
"""
