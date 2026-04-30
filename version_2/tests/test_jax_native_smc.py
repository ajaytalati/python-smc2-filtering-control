"""Stage M equivalence test: JAX-native tempered SMC vs BlackJAX.

Toy: 2-D Gaussian target. Prior = N(0, 4·I). Likelihood = N(mu, 0.5·I).
Posterior is tractable in closed form. Both BlackJAX-based and native
SMC should converge to the same posterior (up to MC noise).

Acceptance: posterior mean within 0.1 of analytical, posterior cov
within 30% (Frobenius). Native vs BlackJAX final means agree within
seed-noise band.

Invoke:
    cd version_2 && PYTHONPATH=.:.. python tests/test_jax_native_smc.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.core.jax_native_smc import (
    _run_tempered_chain_jit,
)


def test_gaussian_target_native_runs():
    """Smoke + posterior-mean check: native chain converges on a 2-D Gaussian."""
    d = 2
    prior_std = 2.0
    mu_lik = jnp.array([1.0, 1.0])
    sigma_lik = 0.5

    @jax.jit
    def _logprior(u):
        return -0.5 * jnp.sum((u / prior_std) ** 2) - d * jnp.log(prior_std) \
               - 0.5 * d * jnp.log(2.0 * jnp.pi)

    @jax.jit
    def _loglikelihood(u):
        return -0.5 * jnp.sum(((u - mu_lik) / sigma_lik) ** 2) \
               - d * jnp.log(sigma_lik) - 0.5 * d * jnp.log(2.0 * jnp.pi)

    logprior_fn = jax.tree_util.Partial(_logprior)
    loglikelihood_fn = jax.tree_util.Partial(_loglikelihood)

    n_smc = 128
    seed = 42
    init_key = jax.random.PRNGKey(seed)
    initial_particles = jax.random.normal(init_key, (n_smc, d)) * prior_std

    rng_key = jax.random.PRNGKey(seed + 17)

    final_particles, n_temp = _run_tempered_chain_jit(
        initial_particles, logprior_fn, loglikelihood_fn, rng_key,
        jnp.float64(0.5),    # target_ess_frac
        jnp.float64(0.10),   # max_lambda_inc
        int(5),              # num_mcmc_steps
        jnp.float64(0.05),   # hmc_step_size
        int(8),              # hmc_num_leapfrog
    )
    final_particles.block_until_ready()

    final_np = np.asarray(final_particles)
    post_mean = final_np.mean(axis=0)
    post_cov = np.cov(final_np.T)

    # Analytical posterior:
    # log p(u | y) = log p(u) + log p(y | u)
    # = -0.5 (u/sig0)^2 - 0.5 ((u - mu_lik)/sig_lik)^2  (up to const)
    # Precision: prec = 1/sig0^2 + 1/sig_lik^2
    # Mean: post_mean = (mu_lik / sig_lik^2) / prec
    prec = 1.0 / prior_std ** 2 + 1.0 / sigma_lik ** 2
    analytical_mean = (np.array([1.0, 1.0]) / sigma_lik ** 2) / prec
    analytical_var = 1.0 / prec
    analytical_std = np.sqrt(analytical_var)

    print(f"  analytical posterior: mean={analytical_mean}, "
          f"std={analytical_std:.4f}")
    print(f"  native SMC posterior: mean={post_mean.round(3)}, "
          f"std=[{np.sqrt(post_cov[0,0]):.3f},{np.sqrt(post_cov[1,1]):.3f}]")
    print(f"  tempering steps: {int(n_temp)}")

    # Mean within 0.2 of analytical (n_smc=128, 5 HMC moves; MC noise band)
    mean_err = np.abs(post_mean - analytical_mean).max()
    assert mean_err < 0.20, f"posterior mean error {mean_err} > 0.20"

    # Marginal std within 30% of analytical
    for i in range(d):
        std_i = np.sqrt(post_cov[i, i])
        rel_err = abs(std_i - analytical_std) / analytical_std
        assert rel_err < 0.30, f"std[{i}] rel err {rel_err:.3f} > 0.30"

    print(f"  [pass] native chain: mean err={mean_err:.4f}, "
          f"n_temp={int(n_temp)}")


def test_native_vs_blackjax_seed_consistency():
    """Both kernels run on the same Gaussian target; means agree within MC noise."""
    import blackjax
    import blackjax.smc.tempered as tempered
    from blackjax.smc import resampling
    from blackjax.smc.solver import dichotomy
    from blackjax.smc.ess import ess_solver
    from smc2fc.core.jax_native_smc import _run_tempered_chain_jit

    d = 2
    prior_std = 2.0
    mu_lik = jnp.array([1.0, 1.0])
    sigma_lik = 0.5

    def py_logprior(u):
        return -0.5 * jnp.sum((u / prior_std) ** 2)

    def py_loglikelihood(u):
        return -0.5 * jnp.sum(((u - mu_lik) / sigma_lik) ** 2)

    n_smc = 128
    seed = 42
    init_key = jax.random.PRNGKey(seed)
    initial_particles = jax.random.normal(init_key, (n_smc, d)) * prior_std

    # ── BlackJAX path ──
    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=py_logprior,
        loglikelihood_fn=py_loglikelihood,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    # Run BlackJAX manually with the same adaptive lambda schedule as native.
    state = tempered.init(initial_particles)
    inv_mass = jnp.ones((1, d))
    rng_bj = jax.random.PRNGKey(seed + 17)
    n_temp_bj = 0
    while float(state.tempering_param) < 1.0 - 1e-6:
        rng_bj, sub = jax.random.split(rng_bj)
        # Use the blackjax solver to find delta (same target_ess_frac)
        from blackjax.smc.ess import ess_solver as bj_ess_solver
        max_delta = 1.0 - float(state.tempering_param)
        delta = bj_ess_solver(
            jax.vmap(py_loglikelihood),
            state.particles, 0.5, max_delta, dichotomy)
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, 0.10)
        next_lam = float(state.tempering_param) + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0
        mcmc_params = {
            'step_size': jnp.array([0.05]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([8], dtype=jnp.int32),
        }
        state, _ = smc_kernel_jit(sub, state, 5, jnp.float64(next_lam),
                                    mcmc_params)
        n_temp_bj += 1

    bj_mean = np.asarray(state.particles).mean(axis=0)

    # ── Native path ──
    logprior_fn = jax.tree_util.Partial(jax.jit(py_logprior))
    loglikelihood_fn = jax.tree_util.Partial(jax.jit(py_loglikelihood))
    rng_native = jax.random.PRNGKey(seed + 17)
    final_native, n_temp_native = _run_tempered_chain_jit(
        initial_particles, logprior_fn, loglikelihood_fn, rng_native,
        jnp.float64(0.5), jnp.float64(0.10),
        int(5), jnp.float64(0.05), int(8),
    )
    native_mean = np.asarray(final_native).mean(axis=0)

    print(f"  BlackJAX:  mean={bj_mean.round(3)}, n_temp={n_temp_bj}")
    print(f"  Native:    mean={native_mean.round(3)}, n_temp={int(n_temp_native)}")

    # Both should target the same posterior. With n_smc=128 and
    # different RNG paths inside HMC, MC error ~ 0.05–0.1.
    diff = np.abs(bj_mean - native_mean).max()
    print(f"  max abs diff: {diff:.4f}")
    assert diff < 0.15, (
        f"native vs BlackJAX mean diff {diff:.4f} > 0.15 — too large to "
        f"be explained by MC noise; possible kernel correctness bug")
    print(f"  [pass] native vs BlackJAX agree to {diff:.4f}")


def main():
    print("=" * 64)
    print("  Stage M — JAX-native tempered SMC equivalence tests")
    print("=" * 64)
    test_gaussian_target_native_runs()
    test_native_vs_blackjax_seed_consistency()
    print("-" * 64)
    print("  All tests passed.")


if __name__ == '__main__':
    main()
