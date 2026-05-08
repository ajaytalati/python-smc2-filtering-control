"""JAX-native tempered SMC kernel — replaces BlackJAX's tempered.build_kernel
to eliminate per-stride JIT recompilation.

WHY THIS EXISTS
---------------
BlackJAX's `tempered.build_kernel(loglikelihood_fn=..., ...)` bakes
`loglikelihood_fn` into the closure of the returned kernel. The bench
builds a fresh `kernel` per stride (because `loglikelihood_fn` closes
over per-stride data), so every `jax.jit(kernel, static_argnums=(2,))`
call creates a new JAX cache entry → ~15 s of HLO recompile per
stride.

This module's `tempered_step` is **module-level @jax.jit**'d and takes
`loglikelihood_fn` and `logprior_fn` as RUNTIME ARGUMENTS (passed as
`jax.tree_util.Partial`-wrapped callables). The Partial is a JAX
pytree → bound-arg shapes/dtypes are part of the trace cache key, but
not bound-arg VALUES. Same shape/dtype across strides → one compile
per process.

The HMC kernel itself is built ONCE at module load via
`blackjax.mcmc.hmc.build_kernel()`, which is fine because that's a
module-level constant.

USAGE
-----
    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    # Drop-in replacement for BlackJAX-based versions in tempered_smc.py.
"""

from __future__ import annotations

import time
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import blackjax.mcmc.hmc as hmc
import blackjax.smc.resampling as resampling

from smc2fc.core.config import SMCConfig
from smc2fc.core.mass_matrix import estimate_mass_matrix
from smc2fc.core.sampling import sample_from_prior
from smc2fc.core.sf_bridge import fit_sf_base
from smc2fc.transforms.unconstrained import log_prior_unconstrained


# Module-level HMC kernel — compiled once when this module imports.
HMC_KERNEL = hmc.build_kernel()
HMC_INIT = hmc.init


class TemperedState(NamedTuple):
    particles: jnp.ndarray   # (n_smc, d)
    log_w: jnp.ndarray       # (n_smc,)  unused for normalised weights
    tempering_param: jnp.ndarray  # scalar


# =========================================================================
# JAX-native ESS solver — fixed-iteration bisection, fully on-device.
# =========================================================================

def _ess_at_delta(loglikelihood_fn, particles, delta):
    """ESS as a function of the temperature lift delta."""
    lp = jax.vmap(loglikelihood_fn)(particles)
    log_w = delta * lp
    log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
    log_ess = -jax.scipy.special.logsumexp(2.0 * log_w_norm)
    return jnp.exp(log_ess)


def _solve_delta_for_ess(loglikelihood_fn, particles,
                          target_ess_frac, max_delta,
                          n_bisect_steps: int = 30):
    """Bisection: find delta in [0, max_delta] such that
    ESS(delta) ≈ target_ess_frac * N. Returns delta, or max_delta
    if even ESS(max_delta) >= target.

    Pure JAX, runs inside jit / lax.while_loop.
    """
    n = particles.shape[0]
    target_ess = target_ess_frac * n

    # Pre-compute lp once outside the bisection
    lp = jax.vmap(loglikelihood_fn)(particles)

    def ess_at(delta):
        log_w = delta * lp
        log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
        log_ess = -jax.scipy.special.logsumexp(2.0 * log_w_norm)
        return jnp.exp(log_ess)

    # If ess(max_delta) > target → can take the full step
    ess_max = ess_at(max_delta)

    def step(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        ess_mid = ess_at(mid)
        # If ESS still above target at mid, we can go higher → lo = mid
        new_lo = jnp.where(ess_mid > target_ess, mid, lo)
        new_hi = jnp.where(ess_mid > target_ess, hi, mid)
        return (new_lo, new_hi), None

    (lo_final, _), _ = lax.scan(step, (0.0, max_delta), jnp.arange(n_bisect_steps))

    # If ESS(max_delta) > target, we'd have monotonically pushed lo to
    # max_delta. Snap to max_delta when applicable.
    delta_solved = jnp.where(ess_max > target_ess, max_delta, lo_final)
    return delta_solved


# =========================================================================
# One tempered SMC step: reweight + resample + HMC moves.
# =========================================================================

def _make_tempered_logposterior(logprior_fn, loglikelihood_fn, lam):
    """Return a CALLABLE that's a pytree-stable Partial.

    Partial PREPENDS its bound args, so f's signature must put the
    runtime arg LAST: f(*bound_args, u). Calling Partial(f, *bound)(u)
    invokes f(*bound, u).
    """
    def f(lp_fn, ll_fn, lam_val, u):
        return lp_fn(u) + lam_val * ll_fn(u)
    return jax.tree_util.Partial(f, logprior_fn, loglikelihood_fn, lam)


def _hmc_step_chain(initial_position, logpost_fn, num_steps,
                     step_size, inv_mass, num_leapfrog, rng_key):
    """Apply num_steps HMC moves to a single particle's position.
    Pure JAX, jit-friendly.
    """
    # Initialise HMC state from position.
    init_state = HMC_INIT(initial_position, logpost_fn)

    def body(carry, _):
        state, key = carry
        key, sub = jax.random.split(key)
        new_state, _info = HMC_KERNEL(
            sub, state, logpost_fn,
            step_size, inv_mass, num_leapfrog)
        return (new_state, key), None

    (final_state, _), _ = lax.scan(body, (init_state, rng_key), jnp.arange(num_steps))
    return final_state.position


# =========================================================================
# The on-device adaptive tempering loop — module-level JIT'd.
# =========================================================================

def run_tempered_chain(
    initial_particles,
    logprior_fn,            # Partial: u -> scalar
    loglikelihood_fn,       # Partial: u -> scalar
    rng_key,
    target_ess_frac,
    max_lambda_inc,
    num_mcmc_steps,
    hmc_step_size,
    hmc_num_leapfrog,
):
    """Run adaptive tempering chain from lambda=0 to lambda=1.

    All adaptive-lambda decisions stay on device via lax.while_loop.

    Returns (final_particles, n_temp_steps).
    """
    # We jit at module level via the wrapper below
    return _run_tempered_chain_impl(
        initial_particles, logprior_fn, loglikelihood_fn, rng_key,
        jnp.float64(target_ess_frac), jnp.float64(max_lambda_inc),
        num_mcmc_steps, hmc_step_size, hmc_num_leapfrog,
    )


def _run_tempered_chain_impl(
    initial_particles, logprior_fn, loglikelihood_fn, rng_key,
    target_ess_frac, max_lambda_inc,
    num_mcmc_steps, hmc_step_size, hmc_num_leapfrog,
):
    n_smc, d = initial_particles.shape

    init_state = TemperedState(
        particles=initial_particles,
        log_w=jnp.zeros(n_smc),
        tempering_param=jnp.float64(0.0),
    )

    def cond_fn(carry):
        state, _key, _step_idx = carry
        return state.tempering_param < jnp.float64(1.0) - 1e-6

    def body_fn(carry):
        state, key, step_idx = carry
        key, k_resample, k_hmc = jax.random.split(key, 3)

        # Adaptive lambda: max possible delta = 1 - current
        lam_curr = state.tempering_param
        max_delta_step = jnp.minimum(1.0 - lam_curr, max_lambda_inc)
        delta = _solve_delta_for_ess(
            loglikelihood_fn, state.particles,
            target_ess_frac, max_delta_step)
        # snap to 1 if very close
        next_lam = jnp.where(
            (lam_curr + delta) > 1.0 - 1e-6,
            jnp.float64(1.0),
            lam_curr + delta)

        # Reweight by (next_lam - lam_curr) * loglikelihood
        delta_lam = next_lam - lam_curr
        lp = jax.vmap(loglikelihood_fn)(state.particles)
        log_w = delta_lam * lp
        log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
        weights = jnp.exp(log_w_norm)

        # Systematic resample
        indices = resampling.systematic(k_resample, weights, n_smc)
        resampled = state.particles[indices]

        # Mass matrix from resampled particles (already JAX)
        inv_mass = estimate_mass_matrix(resampled)
        # estimate_mass_matrix returns shape (1, d); HMC expects (d,) for diag
        inv_mass_diag = inv_mass[0]

        # HMC moves at temperature next_lam
        logpost_fn = _make_tempered_logposterior(
            logprior_fn, loglikelihood_fn, next_lam)

        keys_per_particle = jax.random.split(k_hmc, n_smc)
        new_particles = jax.vmap(
            _hmc_step_chain,
            in_axes=(0, None, None, None, None, None, 0),
        )(
            resampled, logpost_fn, num_mcmc_steps,
            hmc_step_size, inv_mass_diag, hmc_num_leapfrog,
            keys_per_particle,
        )

        new_state = TemperedState(
            particles=new_particles,
            log_w=jnp.zeros(n_smc),
            tempering_param=next_lam,
        )
        return (new_state, key, step_idx + 1)

    final_state, _final_key, n_temp = lax.while_loop(
        cond_fn, body_fn,
        (init_state, rng_key, jnp.int32(0)),
    )
    return final_state.particles, n_temp


# Stable jitted entry point — module level, compiles ONCE per process.
_run_tempered_chain_jit = jax.jit(
    _run_tempered_chain_impl,
    static_argnames=("num_mcmc_steps", "hmc_num_leapfrog"),
)


# =========================================================================
# Drop-in replacements for BlackJAX-based run_smc_window* in tempered_smc.py.
# =========================================================================

def run_smc_window_native(
    full_log_density,        # Callable u -> scalar (pytree-friendly)
    model,
    T_arr,
    cfg: SMCConfig,
    initial_particles=None,
    seed: int = 42,
):
    """Cold-start native SMC: prior -> posterior via adaptive tempering.

    `full_log_density` should be a `jax.tree_util.Partial`-wrapped callable
    so JAX's jit cache hits across calls.
    """
    n_dim = model.n_dim
    n_smc = cfg.n_smc_particles

    # Build pytree-stable callables for prior + likelihood. NO @jax.jit
    # on the inner — they're called inside an outer jit anyway, and
    # an inner jit interferes with Partial-of-callable tracing.
    def _logprior_inner(u):
        return log_prior_unconstrained(u, T_arr)

    logprior_fn = jax.tree_util.Partial(_logprior_inner)

    def _loglikelihood_inner(ld_fn, u):
        # Bound-args first (for Partial), runtime arg `u` last.
        return ld_fn(u) - log_prior_unconstrained(u, T_arr)

    loglikelihood_fn = jax.tree_util.Partial(
        _loglikelihood_inner, full_log_density)

    if initial_particles is None:
        init_key = jax.random.PRNGKey(seed)
        initial_particles = sample_from_prior(n_smc, T_arr, n_dim, init_key)

    rng_key = jax.random.PRNGKey(seed + 123)

    t0 = time.time()
    final_particles, n_temp = _run_tempered_chain_jit(
        initial_particles, logprior_fn, loglikelihood_fn, rng_key,
        jnp.float64(cfg.target_ess_frac),
        jnp.float64(cfg.max_lambda_inc),
        int(cfg.num_mcmc_steps),
        jnp.float64(cfg.hmc_step_size),
        int(cfg.hmc_num_leapfrog),
    )
    final_particles.block_until_ready()
    elapsed = time.time() - t0

    particles_np = np.asarray(final_particles)
    return particles_np, elapsed, int(n_temp)


def run_smc_window_bridge_native(
    new_ld,                   # Partial-wrapped log_density(u)
    prev_particles,           # (n_smc, d) numpy array from previous window
    model,
    T_arr,
    cfg: SMCConfig,
    seed: int = 42,
):
    """Bridge native SMC: q_0 (prev posterior fit) -> new posterior."""
    n_smc = cfg.n_smc_particles
    prev_jax = jnp.asarray(prev_particles, dtype=jnp.float64)
    N, d = prev_jax.shape

    # ── Fit Schrödinger-Föllmer base measure (numpy/JAX, once per stride) ──
    if cfg.bridge_type != 'schrodinger_follmer':
        raise NotImplementedError(
            "run_smc_window_bridge_native only implements the SF bridge "
            "(matches Stage J+K+L production code path).")

    sf_rng = jax.random.PRNGKey(seed + 17)
    sf = fit_sf_base(
        prev_jax, new_ld,
        blend=cfg.sf_blend, entropy_reg=cfg.sf_entropy_reg,
        q1_mode=cfg.sf_q1_mode,
        annealed_n_stages=cfg.sf_annealed_n_stages,
        annealed_n_mh_steps=cfg.sf_annealed_n_mh_steps,
        annealed_proposal_scale=cfg.sf_annealed_proposal_scale,
        use_q0_cov=cfg.sf_use_q0_cov,
        info_aware=cfg.sf_info_aware,
        info_lambda_thresh_quantile=cfg.sf_info_lambda_thresh_quantile,
        info_blend_temperature=cfg.sf_info_blend_temperature,
        rng_key=sf_rng,
    )
    m, L_chol, L_inv, log_det = sf['m'], sf['L_chol'], sf['L_inv'], sf['log_det']

    # Diagnostic print (matches the BlackJAX-version output line-for-line)
    q0_to_q1 = float(jnp.linalg.norm(sf['q1_mean'] - sf['q0_mean']))
    if sf['q1_mode'] == 'annealed':
        mode_diag = (f"n_stages={cfg.sf_annealed_n_stages}, "
                     f"n_mh={cfg.sf_annealed_n_mh_steps}, "
                     f"min ESS={sf['n_eff']:.1f}/{N}, "
                     f"MH acc={sf['accept_mean']:.2f}")
    else:
        mode_diag = f"IS n_eff={sf['n_eff']:.1f}/{N}"
    cov_tag = 'q0_cov' if sf['use_q0_cov'] else 'BW_cov'
    print(f"      SF base ({sf['q1_mode']}/{cov_tag}): blend={sf['blend']:.2f}, "
          f"entropy_reg={sf['entropy_reg']:.3g}, "
          f"log_det={float(log_det):.1f}, "
          f"||m1-m0||={q0_to_q1:.3f}, "
          f"{mode_diag}",
          flush=True)

    # logprior under the SF Gaussian base measure. No @jax.jit on inners
    # — bound-args first (Partial prepends), runtime `u` last.
    def _logprior_inner(m_, L_inv_, log_det_, u):
        diff = u - m_
        maha = jnp.sum((L_inv_ @ diff) ** 2)
        const = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_)
        return const - 0.5 * maha

    logprior_fn = jax.tree_util.Partial(
        _logprior_inner, m, L_inv, log_det)

    def _loglikelihood_inner(ld_fn, m_, L_inv_, log_det_, u):
        diff = u - m_
        maha = jnp.sum((L_inv_ @ diff) ** 2)
        const = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_)
        log_q0 = const - 0.5 * maha
        return ld_fn(u) - log_q0

    loglikelihood_fn = jax.tree_util.Partial(
        _loglikelihood_inner, new_ld, m, L_inv, log_det)

    # Initial particles drawn from q_0.
    init_key = jax.random.PRNGKey(seed)
    z = jax.random.normal(init_key, (n_smc, d), dtype=jnp.float64)
    initial_particles = m[None, :] + z @ L_chol.T

    rng_key = jax.random.PRNGKey(seed + 123)

    t0 = time.time()
    final_particles, n_temp = _run_tempered_chain_jit(
        initial_particles, logprior_fn, loglikelihood_fn, rng_key,
        jnp.float64(cfg.target_ess_frac),
        jnp.float64(cfg.max_lambda_inc_bridge),
        int(cfg.num_mcmc_steps_bridge),
        jnp.float64(cfg.hmc_step_size),
        int(cfg.hmc_num_leapfrog),
    )
    final_particles.block_until_ready()
    elapsed = time.time() - t0

    particles_np = np.asarray(final_particles)
    return particles_np, elapsed, int(n_temp)


__all__ = [
    "TemperedState",
    "run_tempered_chain",
    "run_smc_window_native",
    "run_smc_window_bridge_native",
]
