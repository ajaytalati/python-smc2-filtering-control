"""Generic constrained ↔ unconstrained bijections (purely-functional).

Builds transform arrays from any `EstimationModel`'s prior
specification. No model-specific content — works for any combination
of lognormal, normal, vonmises, and beta priors.

Contract:

    * Each parameter name is mapped to one of four prior families
      (lognormal, normal, vonmises, beta) at setup time.
    * `build_transform_arrays` produces a dict of `(n_dim,)` JAX
      arrays carrying per-dim indicator flags + family-specific
      hyperparameters; this dict is the pytree threaded through the
      hot path.
    * `constrained_to_unconstrained` and `unconstrained_to_constrained`
      are pure JAX (no Python control flow) — they sum masked
      contributions from each family using the indicator flags.
"""

from collections import OrderedDict
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from smc2fc.estimation_model import EstimationModel


# Per-prior-type indicator flag tuples in the order
# (is_log, is_logit, is_ident, is_ln, is_norm, is_vm, is_bt). Captures
# both how the unconstrained → constrained map is computed AND which
# prior family contributes to the log-density.
_FLAG_DEFAULTS: Tuple[float, ...] = (0.0,) * 7
_FLAGS_BY_PTYPE: Dict[str, Tuple[float, ...]] = {
    'lognormal': (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    'normal':    (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
    'vonmises':  (0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0),
    'beta':      (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}


def _hparam_or_default(items, ptype: str, idx: int, default: float
                       ) -> np.ndarray:
    """Extracts the ``idx``-th hyperparameter from items matching ``ptype``.

    For a row whose prior type is ``ptype``, returns the requested
    hyperparameter value; for any other row, returns ``default``.

    Args:
        items: List of ``(name, (ptype, pargs))`` tuples in dim order.
        ptype: The prior family to extract for (e.g. ``'lognormal'``).
        idx: Hyperparameter index — 0 for mu, 1 for sigma (or kappa,
            beta_b, etc.) — depends on the family.
        default: Value used for rows whose prior type does not match.

    Returns:
        Length-`len(items)` numpy array of dtype float32.
    """
    return np.array(
        [item[1][1][idx] if item[1][0] == ptype else default
         for item in items],
        dtype=np.float32,
    )


def build_transform_arrays(model: EstimationModel) -> Dict[str, Array]:
    """Builds indicator + hyperparameter arrays for vectorised transforms.

    Reads the prior specification from the model and constructs JAX
    arrays for each transform type. Called once at setup; the returned
    dict is then passed verbatim into every JIT'd transform call.

    Args:
        model: An `EstimationModel` carrying both `param_prior_config`
            and `init_state_prior_config`.

    Returns:
        Dictionary mapping each transform-array name to a JAX array of
        shape `(n_dim,)`:

        * Indicator flags (1 iff the dim's prior is of the named family):
          ``is_log``, ``is_logit``, ``is_ident``, ``is_ln``, ``is_norm``,
          ``is_vm``, ``is_bt``.
        * Family-specific hyperparameters (defaults of 0 for *_mu and
          1 for *_sigma / kappa / beta_a / beta_b on rows that don't
          match the family):
          ``ln_mu``, ``ln_sigma``, ``n_mu``, ``n_sigma``,
          ``vm_mu``, ``vm_kappa``, ``beta_a``, ``beta_b``.
    """
    all_config: OrderedDict = OrderedDict()
    all_config.update(model.param_prior_config)
    all_config.update(model.init_state_prior_config)
    items = list(all_config.items())  # dim-ordered (name, (ptype, pargs))

    # Stack per-dim flag rows into a single (n_dim, 7) array, then peel
    # off each indicator column.
    flag_rows = np.array(
        [_FLAGS_BY_PTYPE.get(item[1][0], _FLAG_DEFAULTS) for item in items],
        dtype=np.float32,
    )
    is_log, is_logit, is_ident, is_ln, is_norm, is_vm, is_bt = (
        flag_rows[:, k] for k in range(7)
    )

    return {
        'is_log':   jnp.array(is_log),
        'is_logit': jnp.array(is_logit),
        'is_ident': jnp.array(is_ident),
        'is_ln':    jnp.array(is_ln),
        'is_norm':  jnp.array(is_norm),
        'is_vm':    jnp.array(is_vm),
        'is_bt':    jnp.array(is_bt),
        'ln_mu':    jnp.array(_hparam_or_default(items, 'lognormal', 0, 0.0)),
        'ln_sigma': jnp.array(_hparam_or_default(items, 'lognormal', 1, 1.0)),
        'n_mu':     jnp.array(_hparam_or_default(items, 'normal',    0, 0.0)),
        'n_sigma':  jnp.array(_hparam_or_default(items, 'normal',    1, 1.0)),
        'vm_mu':    jnp.array(_hparam_or_default(items, 'vonmises',  0, 0.0)),
        'vm_kappa': jnp.array(_hparam_or_default(items, 'vonmises',  1, 1.0)),
        'beta_a':   jnp.array(_hparam_or_default(items, 'beta',      0, 1.0)),
        'beta_b':   jnp.array(_hparam_or_default(items, 'beta',      1, 1.0)),
    }


def constrained_to_unconstrained(theta: Array, T: Dict[str, Array]) -> Array:
    """Maps a constrained-space parameter vector to unconstrained space.

    Pure JAX: applies ``log`` for lognormal dims, ``logit`` for beta
    dims, and identity for normal / vonmises dims, then sums the
    contributions weighted by the indicator flags from ``T``.

    Args:
        theta: Constrained parameter vector of shape ``(n_dim,)``.
        T: Transform arrays from :func:`build_transform_arrays`.

    Returns:
        Unconstrained vector ``u`` of shape ``(n_dim,)``.
    """
    log_v = jnp.log(jnp.maximum(theta, 1e-30))
    clip_v = jnp.clip(theta, 1e-6, 1.0 - 1e-6)
    logit_v = jnp.log(clip_v / (1.0 - clip_v))
    return (T['is_log'] * log_v
            + T['is_logit'] * logit_v
            + T['is_ident'] * theta)


def unconstrained_to_constrained(u: Array, T: Dict[str, Array]) -> Array:
    """Maps an unconstrained vector back to constrained space.

    Inverse of :func:`constrained_to_unconstrained`. Pure JAX.

    Args:
        u: Unconstrained vector of shape ``(n_dim,)``.
        T: Transform arrays from :func:`build_transform_arrays`.

    Returns:
        Constrained parameter vector of shape ``(n_dim,)``.
    """
    return (T['is_log'] * jnp.exp(jnp.clip(u, -20, 20))
            + T['is_logit'] * jax.nn.sigmoid(u)
            + T['is_ident'] * u)


def log_prior_unconstrained(u: Array, T: Dict[str, Array]) -> Array:
    """Computes the log prior density evaluated in unconstrained space.

    Each prior family contributes via its indicator flag — only one
    flag is 1 per dim, so the sum picks out the family-specific log-
    pdf. Constants (incl. ``HALF_LOG_2PI``) are not included; this is
    fine for posterior shape but should NOT be used for absolute
    marginal-likelihood comparisons.

    Args:
        u: Unconstrained vector of shape ``(n_dim,)``.
        T: Transform arrays from :func:`build_transform_arrays`.

    Returns:
        Scalar log-prior density (sum across dimensions).
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
    """Splits a combined theta vector into params and init-state slices.

    Args:
        theta: Combined vector of shape ``(n_dim,)`` where the first
            ``n_params`` entries are dynamics parameters and the rest
            are estimated initial states.
        n_params: Number of dynamics parameters (the rest are init
            states).

    Returns:
        Tuple ``(params, init_states)``: the two slices of ``theta``.
    """
    return theta[:n_params], theta[n_params:]
