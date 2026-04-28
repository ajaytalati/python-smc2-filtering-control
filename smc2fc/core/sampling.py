"""Initial particle sampling from the (unconstrained-space) prior."""

import jax
import jax.numpy as jnp


def sample_from_prior(n_particles, T_arr, n_dim, rng_key):
    """Draw ``n_particles`` from the prior in unconstrained space.

    Parameters
    ----------
    n_particles : int
    T_arr : dict
        Transform arrays produced by ``smc2fc.transforms.unconstrained``.
        Must provide 'is_ln', 'is_norm', 'ln_mu', 'ln_sigma', 'n_mu',
        'n_sigma' (per dimension).
    n_dim : int
        Problem dimension (= model.n_dim).
    rng_key : jax.random.PRNGKey
    """
    keys = jax.random.split(rng_key, n_dim)
    particles = jnp.zeros((n_particles, n_dim), dtype=jnp.float64)
    for i in range(n_dim):
        z = jax.random.normal(keys[i], (n_particles,), dtype=jnp.float64)
        if float(T_arr['is_ln'][i]) > 0.5:
            particles = particles.at[:, i].set(
                float(T_arr['ln_mu'][i]) + float(T_arr['ln_sigma'][i]) * z)
        elif float(T_arr['is_norm'][i]) > 0.5:
            particles = particles.at[:, i].set(
                float(T_arr['n_mu'][i]) + float(T_arr['n_sigma'][i]) * z)
    return particles
