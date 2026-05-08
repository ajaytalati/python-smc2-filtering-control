"""Per-window adaptive tempered SMC² with two entry points:

  - ``run_smc_window``        — cold start (prior → posterior).
  - ``run_smc_window_bridge`` — bridge (Gaussian fit of previous posterior
                                → new posterior). Used for warm-start
                                between rolling windows.

Both use BlackJAX's tempered SMC kernel with a clamped adaptive schedule:
the ESS solver finds the optimal lambda increment, then we clamp it to
``max_lambda_inc`` (or ``max_lambda_inc_bridge``) to guarantee a minimum
number of tempering levels.

The per-level MCMC kernel is HMC with a diagonal mass matrix re-estimated
from the current particle cloud after each step.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
import blackjax.smc.tempered as tempered
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as solver

from smc2fc.transforms.unconstrained import log_prior_unconstrained
from smc2fc.core.config import SMCConfig
from smc2fc.core.mass_matrix import estimate_mass_matrix
from smc2fc.core.sampling import sample_from_prior
from smc2fc.core.sf_bridge import fit_sf_base


def _kmeans_labels(X: np.ndarray, K: int, n_iter: int = 20,
                   seed: int = 0) -> np.ndarray:
    """Plain-numpy K-means (no sklearn dep). Returns cluster labels (N,)."""
    rng = np.random.default_rng(seed)
    N, d = X.shape
    # k-means++ seeding: first centre random, others far from previous
    centres = np.zeros((K, d))
    centres[0] = X[rng.integers(N)]
    for k in range(1, K):
        dists = np.min(
            np.linalg.norm(X[:, None, :] - centres[None, :k, :], axis=2) ** 2,
            axis=1,
        )
        probs = dists / (dists.sum() + 1e-12)
        centres[k] = X[rng.choice(N, p=probs)]
    for _ in range(n_iter):
        d2 = np.linalg.norm(
            X[:, None, :] - centres[None, :, :], axis=2
        ) ** 2
        labels = np.argmin(d2, axis=1)
        new_centres = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else centres[k]
            for k in range(K)
        ])
        if np.allclose(new_centres, centres):
            break
        centres = new_centres
    return labels


def _fit_mog_bridge(prev_particles: np.ndarray, K: int = 2, seed: int = 0):
    """Fit a K-component mixture of Gaussians to `prev_particles`.

    Each component is a full-covariance Gaussian with Ledoit-Wolf shrinkage
    applied to its covariance. Weights are proportional to cluster size.

    Returns:
        weights : (K,) log-weights of components
        mus     : (K, d)
        L_chols : (K, d, d) Cholesky of each component's regularised cov
        L_invs  : (K, d, d) inverse
        log_dets: (K,) 2 * sum(log(diag(L_chol)))
    """
    N, d = prev_particles.shape
    labels = _kmeans_labels(prev_particles, K=K, seed=seed)

    mus = np.zeros((K, d))
    L_chols = np.zeros((K, d, d))
    L_invs = np.zeros((K, d, d))
    log_dets = np.zeros(K)
    log_weights = np.zeros(K)

    for k in range(K):
        members = prev_particles[labels == k]
        n_k = len(members)
        if n_k < 2:
            # Fallback: fit component to full cloud with a wide eigenvalue floor
            members = prev_particles
            n_k = N
        mu_k = members.mean(axis=0)
        S_k = np.cov(members.T)
        # LW shrinkage (per-component)
        X_c = members - mu_k[None, :]
        mu_target = float(np.trace(S_k) / d)
        delta_mat = S_k - mu_target * np.eye(d)
        delta_sq_sum = float((delta_mat ** 2).sum())
        X2 = (X_c[:, :, None] * X_c[:, None, :])
        b_bar = float(((X2 - S_k[None, :, :]) ** 2).sum() / (n_k * n_k))
        alpha = min(b_bar / max(delta_sq_sum, 1e-10), 1.0)
        cov_lw = (1.0 - alpha) * S_k + alpha * mu_target * np.eye(d)
        cov_reg = cov_lw + 1e-4 * np.eye(d)
        L = np.linalg.cholesky(cov_reg)
        mus[k] = mu_k
        L_chols[k] = L
        # L_inv via triangular solve
        L_invs[k] = np.linalg.solve(L, np.eye(d))
        log_dets[k] = 2.0 * np.sum(np.log(np.diag(L)))
        log_weights[k] = np.log(max(n_k / N, 1e-6))

    return log_weights, mus, L_chols, L_invs, log_dets


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start
# ─────────────────────────────────────────────────────────────────────────────

def run_smc_window(full_log_density, model, T_arr, cfg: SMCConfig,
                   initial_particles=None, seed: int = 42):
    """Adaptive tempered SMC from the prior to the posterior.

    If ``initial_particles`` is None, draws from the prior (cold start).
    Otherwise warm-starts from the provided particles (rare; use
    ``run_smc_window_bridge`` for the standard warm-start).

    Returns ``(particles, elapsed_s, n_tempering_steps)``.
    """
    n_dim = model.n_dim
    n_smc = cfg.n_smc_particles

    @jax.jit
    def logprior_fn(u):
        return log_prior_unconstrained(u, T_arr)

    @jax.jit
    def loglikelihood_fn(u):
        return full_log_density(u) - log_prior_unconstrained(u, T_arr)

    if initial_particles is None:
        init_key = jax.random.PRNGKey(seed)
        initial_particles = sample_from_prior(n_smc, T_arr, n_dim, init_key)

    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    state = tempered.init(initial_particles)
    inv_mass = estimate_mass_matrix(initial_particles)

    rng_key = jax.random.PRNGKey(seed + 123)
    step_idx = 0
    t0 = time.time()

    # Stage J1a: defer per-step printing-only syncs. We keep the
    # control-flow-required floats (while-condition, delta clip, snap-to-1)
    # but stash diagnostics as device arrays and drain them once at the end.
    diag_buf = []   # list of {'lam_dev', 'delta', 'acc_dev', 'elapsed'}

    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)

        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            cfg.target_ess_frac,
            max_delta,
            solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, cfg.max_lambda_inc)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_parameters = {
            'step_size': jnp.array([cfg.hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([cfg.hmc_num_leapfrog],
                                                dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, cfg.num_mcmc_steps,
            jnp.float64(next_lam), mcmc_parameters)
        inv_mass = estimate_mass_matrix(state.particles)

        step_idx += 1
        try:
            acc_dev = jnp.mean(info.update_info.acceptance_rate)
        except Exception:
            acc_dev = jnp.float64('nan')
        diag_buf.append({
            'lam_dev':   state.tempering_param,
            'delta':     delta,
            'acc_dev':   acc_dev,
            'elapsed':   time.time() - t0,
        })

    elapsed = time.time() - t0
    particles = np.array(jax.device_get(state.particles))

    # Drain the diagnostics buffer in one pass — single batched sync.
    prev_lam = 0.0
    for i, d in enumerate(diag_buf):
        lam = float(d['lam_dev'])
        acc = float(d['acc_dev'])
        actual_delta = lam - prev_lam
        prev_lam = lam
        compile_note = " (JIT)" if i == 0 else ""
        print(f"      step {i + 1:3d}  lam={lam:.6f}  "
              f"d={actual_delta:.4f}  acc={acc:.3f}  "
              f"[{d['elapsed']:.0f}s{compile_note}]",
              flush=True)
    return particles, elapsed, step_idx


# ─────────────────────────────────────────────────────────────────────────────
# Bridge (warm-start between rolling windows)
# ─────────────────────────────────────────────────────────────────────────────

def run_smc_window_bridge(new_ld, prev_particles, model, T_arr,
                          cfg: SMCConfig, seed: int = 42):
    """Bridge tempered SMC: Gaussian / MoG base measure → new posterior.

    Data-annealing bridge (tempered SMC²):
      logprior_fn(u)      = log q_0(u)                       [base fit of prev post]
      loglikelihood_fn(u) = new_ld(u) - logprior_fn(u)       [1 PF eval]

    At lambda=0 the target is q_0 ≈ old posterior (where particles start).
    At lambda=1 the target is new_ld(u) = correct new-window posterior.

    Three base-measure options (cfg.bridge_type):
      - 'gaussian': single Gaussian fit with Ledoit-Wolf shrinkage (default)
      - 'mog':      K-component mixture of Gaussians, K = cfg.bridge_mog_components.
                    Each component full-covariance with per-component LW shrinkage.
                    Weights proportional to K-means cluster sizes.
      - 'schrodinger_follmer': Bures-Wasserstein geodesic at t=cfg.sf_blend
                    between the prev-posterior Gaussian fit and an
                    importance-matched new-posterior Gaussian estimate.
                    Tightens the bridge by starting closer to π_new.
                    See smc2fc.core.sf_bridge for details.
    """
    prev_np = np.asarray(prev_particles, dtype=np.float64)
    N, d = prev_np.shape

    if cfg.bridge_type == 'schrodinger_follmer':
        prev = jnp.array(prev_particles, dtype=jnp.float64)
        sf_rng = jax.random.PRNGKey(seed + 17)   # offset to decorrelate from outer init key
        sf = fit_sf_base(
            prev, new_ld,
            blend=cfg.sf_blend,
            entropy_reg=cfg.sf_entropy_reg,
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
        const = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det)

        # SF diagnostics: print BW endpoint distance + per-mode ESS / acceptance
        q0_to_q1 = float(jnp.linalg.norm(sf['q1_mean'] - sf['q0_mean']))
        if sf['q1_mode'] == 'annealed':
            mode_diag = (f"n_stages={cfg.sf_annealed_n_stages}, "
                         f"n_mh={cfg.sf_annealed_n_mh_steps}, "
                         f"min ESS={sf['n_eff']:.1f}/{N}, "
                         f"MH acc={sf['accept_mean']:.2f}")
        else:
            mode_diag = f"IS n_eff={sf['n_eff']:.1f}/{N}"
        cov_tag = 'q0_cov' if sf['use_q0_cov'] else 'BW_cov'
        info_tag = ''
        if sf.get('info_aware'):
            diag = sf['info_diagnostics']
            eigvals = np.asarray(diag['fim_eigvals'])
            blends = np.asarray(diag['blend_per_eig'])
            n_held = int(np.sum(blends < 0.5 * cfg.sf_blend))
            info_tag = (f", info_aware: λ∈[{eigvals.min():.1e}, {eigvals.max():.1e}] "
                         f"({n_held}/{d} held)")
        print(f"      SF base ({sf['q1_mode']}/{cov_tag}): blend={sf['blend']:.2f}, "
              f"entropy_reg={sf['entropy_reg']:.3g}, "
              f"log_det={float(log_det):.1f}, "
              f"||m1-m0||={q0_to_q1:.3f}, "
              f"{mode_diag}{info_tag}",
              flush=True)

        @jax.jit
        def logprior_fn(u):
            diff = u - m
            maha = jnp.sum((L_inv @ diff) ** 2)
            return const - 0.5 * maha

        init_key = jax.random.PRNGKey(seed)
        z = jax.random.normal(init_key, (cfg.n_smc_particles, d),
                              dtype=jnp.float64)
        initial_particles = m[None, :] + z @ L_chol.T

    elif cfg.bridge_type == 'mog':
        K = int(cfg.bridge_mog_components)
        log_w_np, mus_np, L_chols_np, L_invs_np, log_dets_np = _fit_mog_bridge(
            prev_np, K=K, seed=seed,
        )
        log_w = jnp.asarray(log_w_np, dtype=jnp.float64)
        mus = jnp.asarray(mus_np, dtype=jnp.float64)
        L_chols = jnp.asarray(L_chols_np, dtype=jnp.float64)
        L_invs = jnp.asarray(L_invs_np, dtype=jnp.float64)
        log_dets = jnp.asarray(log_dets_np, dtype=jnp.float64)

        const_vec = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_dets)  # (K,)

        print(f"      MoG base: K={K}  cluster sizes≈{np.exp(log_w_np) * N}  "
              f"log_dets={[f'{x:.1f}' for x in log_dets_np]}", flush=True)

        @jax.jit
        def logprior_fn(u):
            # log-sum-exp over K components, each log N(u; mu_k, L_k L_k^T)
            diffs = u[None, :] - mus         # (K, d)
            # (L_inv diff) per component
            whit = jnp.einsum('kij,kj->ki', L_invs, diffs)  # (K, d)
            maha = jnp.sum(whit ** 2, axis=-1)              # (K,)
            comp_logps = const_vec - 0.5 * maha             # (K,)
            return jax.scipy.special.logsumexp(log_w + comp_logps)

        # Sample initial particles from the mixture: pick component per
        # particle by the component weights, then Gaussian conditional.
        init_key = jax.random.PRNGKey(seed)
        k_key, z_key = jax.random.split(init_key)
        n_smc = cfg.n_smc_particles
        comp_ids = jax.random.categorical(
            k_key, log_w, shape=(n_smc,),
        )  # (n_smc,) integers in [0, K)
        z = jax.random.normal(z_key, (n_smc, d), dtype=jnp.float64)
        # For each particle i, sample from N(mus[k_i], L_chols[k_i])
        L_picked = L_chols[comp_ids]           # (n_smc, d, d)
        mu_picked = mus[comp_ids]              # (n_smc, d)
        initial_particles = mu_picked + jnp.einsum(
            'nij,nj->ni', L_picked, z)
    else:
        # Single Gaussian with Ledoit-Wolf shrinkage
        prev = jnp.array(prev_particles, dtype=jnp.float64)
        mu = jnp.mean(prev, axis=0)
        S = jnp.cov(prev.T)

        # Ledoit-Wolf optimal shrinkage (Ledoit & Wolf 2004, Eq. 2)
        X_c = prev - mu[None, :]
        mu_target = jnp.trace(S) / d
        delta_mat = S - mu_target * jnp.eye(d)
        delta_sq_sum = jnp.sum(delta_mat ** 2)
        X2 = (X_c[:, :, None] * X_c[:, None, :])
        b_bar = jnp.sum((X2 - S[None, :, :]) ** 2) / (N * N)
        alpha = min(float(b_bar / jnp.maximum(delta_sq_sum, 1e-10)), 1.0)

        cov_lw = (1.0 - alpha) * S + alpha * mu_target * jnp.eye(d)
        cov_reg = cov_lw + 1e-4 * jnp.eye(d)
        L_chol = jnp.linalg.cholesky(cov_reg)
        L_inv = jax.scipy.linalg.solve_triangular(
            L_chol, jnp.eye(d), lower=True)
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_chol)))
        const = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det)

        print(f"      Gaussian base: LW shrinkage={alpha:.3f}, "
              f"log_det={float(log_det):.1f}", flush=True)

        @jax.jit
        def logprior_fn(u):
            diff = u - mu
            maha = jnp.sum((L_inv @ diff) ** 2)
            return const - 0.5 * maha

        init_key = jax.random.PRNGKey(seed)
        z = jax.random.normal(init_key, (cfg.n_smc_particles, d),
                              dtype=jnp.float64)
        initial_particles = mu[None, :] + z @ L_chol.T

    @jax.jit
    def loglikelihood_fn(u):
        return new_ld(u) - logprior_fn(u)

    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    state = tempered.init(initial_particles)
    inv_mass = estimate_mass_matrix(initial_particles)

    rng_key = jax.random.PRNGKey(seed + 123)
    step_idx = 0
    t0 = time.time()

    # Stage J1a: same per-step-sync deferral as run_smc_window. The
    # initial incr_ll diagnostic is queued (not synced) so the first
    # tempering kernel can dispatch in parallel with it being formatted.
    incr_ll_init = jax.vmap(loglikelihood_fn)(initial_particles)
    incr_var_init_dev = jnp.var(incr_ll_init)
    incr_range_init_dev = jnp.max(incr_ll_init) - jnp.min(incr_ll_init)

    diag_buf = []   # list of {'lam_dev', 'delta', 'acc_dev', 'incr_var_dev', 'elapsed'}

    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)

        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            cfg.target_ess_frac,
            max_delta,
            solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, cfg.max_lambda_inc_bridge)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_parameters = {
            'step_size': jnp.array([cfg.hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([cfg.hmc_num_leapfrog],
                                                dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, cfg.num_mcmc_steps_bridge,
            jnp.float64(next_lam), mcmc_parameters)
        inv_mass = estimate_mass_matrix(state.particles)

        step_idx += 1
        try:
            acc_dev = jnp.mean(info.update_info.acceptance_rate)
        except Exception:
            acc_dev = jnp.float64('nan')
        incr_var_dev = jnp.var(jax.vmap(loglikelihood_fn)(state.particles))
        diag_buf.append({
            'lam_dev':       state.tempering_param,
            'delta':         delta,
            'acc_dev':       acc_dev,
            'incr_var_dev':  incr_var_dev,
            'elapsed':       time.time() - t0,
        })

    elapsed = time.time() - t0
    particles = np.array(jax.device_get(state.particles))

    # Drain init-diagnostic + per-step buffers in one batched sync.
    print(f"      Bridge init: incr_ll var={float(incr_var_init_dev):.1f} "
          f"range={float(incr_range_init_dev):.1f}", flush=True)
    prev_lam = 0.0
    for i, d in enumerate(diag_buf):
        lam = float(d['lam_dev'])
        acc = float(d['acc_dev'])
        incr_var = float(d['incr_var_dev'])
        actual_delta = lam - prev_lam
        prev_lam = lam
        compile_note = " (JIT)" if i == 0 else ""
        print(f"      step {i + 1:3d}  lam={lam:.6f}  "
              f"d={actual_delta:.4f}  acc={acc:.3f}  "
              f"incr_var={incr_var:.1f}  "
              f"[{d['elapsed']:.0f}s{compile_note}]",
              flush=True)
    return particles, elapsed, step_idx
