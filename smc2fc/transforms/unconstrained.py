"""Generic constrained/unconstrained bijections.

Builds transform arrays from any EstimationModel's prior specification.
No model-specific content — works for any combination of lognormal,
normal, vonmises, and beta priors.

Date:    15 April 2026
Version: 5.0 (model-agnostic)
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import Array
from typing import Dict, Tuple
from collections import OrderedDict

from smc2fc.estimation_model import EstimationModel


def build_transform_arrays(model: EstimationModel) -> Dict[str, Array]:
    """Build indicator and parameter arrays for vectorised transforms.

    Reads the prior specification from the model and constructs
    JAX arrays for each transform type. Called once at setup.

    Args:
        model: EstimationModel with param_prior_config and
            init_state_prior_config.

    Returns:
        Dictionary of JAX arrays, each shape (n_dim,).
    """
    all_config = OrderedDict()
    all_config.update(model.param_prior_config)
    all_config.update(model.init_state_prior_config)
    n_dim = model.n_dim

    arrays = {k: np.zeros(n_dim, dtype=np.float32) for k in [
        'is_log', 'is_logit', 'is_ident', 'is_ln', 'is_norm', 'is_vm', 'is_bt'
    ]}
    arrays.update({k: np.zeros(n_dim, dtype=np.float32) for k in [
        'ln_mu', 'n_mu', 'vm_mu'
    ]})
    arrays.update({k: np.ones(n_dim, dtype=np.float32) for k in [
        'ln_sigma', 'n_sigma', 'vm_kappa', 'beta_a', 'beta_b'
    ]})

    for i, (name, (ptype, pargs)) in enumerate(all_config.items()):
        if ptype == 'lognormal':
            arrays['is_log'][i] = 1; arrays['is_ln'][i] = 1
            arrays['ln_mu'][i], arrays['ln_sigma'][i] = pargs
        elif ptype == 'normal':
            arrays['is_ident'][i] = 1; arrays['is_norm'][i] = 1
            arrays['n_mu'][i], arrays['n_sigma'][i] = pargs
        elif ptype == 'vonmises':
            arrays['is_ident'][i] = 1; arrays['is_vm'][i] = 1
            arrays['vm_mu'][i], arrays['vm_kappa'][i] = pargs
        elif ptype == 'beta':
            arrays['is_logit'][i] = 1; arrays['is_bt'][i] = 1
            arrays['beta_a'][i], arrays['beta_b'][i] = pargs

    return {k: jnp.array(v) for k, v in arrays.items()}


def constrained_to_unconstrained(theta: Array, T: Dict[str, Array]) -> Array:
    """Map constrained parameters to unconstrained space.

    Args:
        theta: Constrained vector, shape (n_dim,).
        T: Transform arrays from build_transform_arrays.

    Returns:
        Unconstrained vector u, shape (n_dim,).
    """
    log_v = jnp.log(jnp.maximum(theta, 1e-30))
    clip_v = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
    logit_v = jnp.log(clip_v / (1.0 - clip_v))
    return (T['is_log'] * log_v
            + T['is_logit'] * logit_v
            + T['is_ident'] * theta)


def unconstrained_to_constrained(u: Array, T: Dict[str, Array]) -> Array:
    """Map unconstrained u back to constrained parameters.

    Args:
        u: Unconstrained vector, shape (n_dim,).
        T: Transform arrays from build_transform_arrays.

    Returns:
        Constrained parameter vector, shape (n_dim,).
    """
    return (T['is_log'] * jnp.exp(jnp.clip(u, -20, 20))
            + T['is_logit'] * jax.nn.sigmoid(u)
            + T['is_ident'] * u)


def log_prior_unconstrained(u: Array, T: Dict[str, Array]) -> Array:
    """Log prior density in unconstrained space.

    Args:
        u: Unconstrained vector, shape (n_dim,).
        T: Transform arrays from build_transform_arrays.

    Returns:
        Scalar log prior density.
    """
    lp = (T['is_ln'] * (-0.5 * ((u - T['ln_mu']) / T['ln_sigma']) ** 2
                          - jnp.log(T['ln_sigma']))
          + T['is_norm'] * (-0.5 * ((u - T['n_mu']) / T['n_sigma']) ** 2
                              - jnp.log(T['n_sigma']))
          + T['is_vm'] * T['vm_kappa'] * jnp.cos(u - T['vm_mu'])
          + T['is_bt'] * (T['beta_a'] * jax.nn.log_sigmoid(u)
                           + T['beta_b'] * jax.nn.log_sigmoid(-u)))
    return jnp.sum(lp)


def split_theta(theta: Array, n_params: int) -> Tuple[Array, Array]:
    """Split combined theta into params and init states.

    Args:
        theta: Combined vector, shape (n_dim,).
        n_params: Number of parameters (rest are init states).

    Returns:
        Tuple (params, init_states).
    """
    return theta[:n_params], theta[n_params:]
