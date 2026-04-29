"""JAX-autodiff linearisation of an arbitrary FSA-style drift around a
deterministic operating point.

Given a drift function `drift_jax(y, params, Phi)` that returns dy/dt,
returns the Jacobians

    A_lin = d(drift)/dy   evaluated at (x*, Phi*, params)
    B_lin = d(drift)/dPhi  evaluated at (x*, Phi*, params)

Both are constant matrices for time-invariant dynamics; for time-
varying drifts, call this at each time-grid point.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp


def linearize_drift_at(
    drift_jax: Callable,
    x_star: jnp.ndarray,
    phi_star: float,
    params: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (A_lin, B_lin) Jacobians of `drift_jax` at the operating point.

    Parameters
    ----------
    drift_jax : Callable
        Signature `(y, params, Phi) -> drift`. y is shape (n_x,);
        Phi is a scalar; drift returns shape (n_x,).
    x_star : jnp.ndarray, shape (n_x,)
        Operating-point state (e.g. (B_typ, F_typ, A_typ)).
    phi_star : float
        Operating-point control (e.g. Phi_default = 1.0).
    params : dict
        Truth params at which to evaluate (e.g. simulation.DEFAULT_PARAMS).

    Returns
    -------
    A_lin : jnp.ndarray, shape (n_x, n_x)
        Jacobian d(drift)/dy at the operating point.
    B_lin : jnp.ndarray, shape (n_x, 1)
        Jacobian d(drift)/dPhi at the operating point (column vector).
    """
    x_star = jnp.asarray(x_star, dtype=jnp.float64)
    phi_star_arr = jnp.asarray(phi_star, dtype=jnp.float64)
    p_jax = {k: jnp.asarray(float(v), dtype=jnp.float64)
             for k, v in params.items()}

    def f_y(y):
        return drift_jax(y, p_jax, phi_star_arr)

    def f_phi(phi):
        return drift_jax(x_star, p_jax, phi)

    A_lin = jax.jacfwd(f_y)(x_star)
    B_lin = jax.jacfwd(f_phi)(phi_star_arr).reshape(-1, 1)
    return A_lin, B_lin
