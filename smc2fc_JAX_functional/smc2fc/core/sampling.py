"""Initial particle sampling from the prior in unconstrained space.

Pure JAX. The original imperative source built the particle array via
``for i in range(n_dim): particles = particles.at[:, i].set(...)`` and
branched on the prior family per-dim. The functional rewrite computes
the log-normal and normal contributions in fully-vectorised form
(``(n_particles, n_dim)``), then uses the model's per-dim indicator
flags from ``T_arr`` to combine them — no Python loop over dims.
"""

import jax
import jax.numpy as jnp
from jax import Array


def sample_from_prior(n_particles: int,
                      T_arr: dict,
                      n_dim: int,
                      rng_key: Array) -> Array:
    """Draws ``n_particles`` from the prior, in unconstrained space.

    For each parameter dimension, the prior is one of:

        * ``lognormal(ln_mu, ln_sigma)``: drawn as ``ln_mu + ln_sigma * Z``
          with ``Z ~ N(0, 1)``.
        * ``normal(n_mu, n_sigma)``: drawn as ``n_mu + n_sigma * Z`` with
          ``Z ~ N(0, 1)``.

    Other prior families (vonmises, beta) do not contribute samples
    here — those dims default to zero, matching the original code's
    behaviour (the original ``for`` loop simply did not enter the
    ``elif`` branches for those families).

    Args:
        n_particles: Number of samples to draw.
        T_arr: Transform-array dict from
            :func:`smc2fc.transforms.unconstrained.build_transform_arrays`.
            Must include ``is_ln``, ``is_norm``, ``ln_mu``, ``ln_sigma``,
            ``n_mu``, ``n_sigma`` (per-dim arrays of length ``n_dim``).
        n_dim: Problem dimensionality (``model.n_dim``).
        rng_key: JAX PRNGKey driving the draw. Split into ``n_dim``
            sub-keys so that each dimension uses its own independent
            randomness (preserves per-dim reproducibility).

    Returns:
        Particle matrix of shape ``(n_particles, n_dim)``, dtype
        float64.
    """
    keys = jax.random.split(rng_key, n_dim)

    # Per-dim independent N(0, 1) samples — vmap over dims so the
    # randomness is identical to the imperative source's per-dim split.
    z = jax.vmap(
        lambda k: jax.random.normal(k, (n_particles,), dtype=jnp.float64)
    )(keys).T                                    # shape (n_particles, n_dim)

    is_ln    = T_arr['is_ln'].astype(jnp.float64)[None, :]
    is_norm  = T_arr['is_norm'].astype(jnp.float64)[None, :]
    ln_mu    = T_arr['ln_mu'].astype(jnp.float64)[None, :]
    ln_sigma = T_arr['ln_sigma'].astype(jnp.float64)[None, :]
    n_mu     = T_arr['n_mu'].astype(jnp.float64)[None, :]
    n_sigma  = T_arr['n_sigma'].astype(jnp.float64)[None, :]

    # Each dim contributes via exactly one of (is_ln, is_norm); both
    # masks are 0/1 and mutually exclusive on any given dim.
    return (is_ln   * (ln_mu + ln_sigma * z)
            + is_norm * (n_mu + n_sigma * z))
