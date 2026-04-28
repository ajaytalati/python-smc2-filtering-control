"""Schrödinger–Föllmer bridge between Gaussian endpoints.

Closed-form Path A: the Schrödinger bridge between Gaussian source
:math:`\\mu_0 = \\mathcal{N}(m_0, S_0)` and Gaussian target
:math:`\\mu_1 = \\mathcal{N}(m_1, S_1)` is the **Bures–Wasserstein
geodesic** (when entropic regularisation :math:`\\varepsilon \\to 0`):

.. math::

    m_t &= (1-t) m_0 + t m_1, \\\\
    S_t &= ((1-t) I + t T) S_0 ((1-t) I + t T)

where :math:`T = S_0^{-1/2} (S_0^{1/2} S_1 S_0^{1/2})^{1/2} S_0^{-1/2}`
is the Bures-Wasserstein optimal transport map from
:math:`\\mu_0` to :math:`\\mu_1`. With finite entropic regularisation,
the covariance gains an additional :math:`\\varepsilon t (1-t) I` term
(Schrödinger bridge between Gaussians; see e.g. Janati, Muzellec,
Peyré, Cuturi 2020).

This module provides:

  - ``bw_geodesic(m0, S0, m1, S1, t, entropy_reg)``: closed-form
    interpolation. JAX-jittable.
  - ``estimate_target_gaussian(prev_particles, log_w_to_target)``:
    importance-weighted moment-match of the new posterior, given prev
    particles weighted by ``new_ld(u) - log q_0(u)`` (one new_ld eval
    per prev particle — same cost as the existing bridge).
  - ``fit_sf_base(prev_particles, new_ld_fn, blend, entropy_reg, ...)``:
    full pipeline returning the SF base measure parameters at
    ``t = blend`` along the BW geodesic from prev-posterior Gaussian
    to importance-matched new-posterior Gaussian.

The SMC² rolling-window bridge dispatches on
``cfg.bridge_type == 'schrodinger_follmer'`` and uses ``fit_sf_base``
to produce the base measure for tempered SMC. ``blend=0`` recovers
the existing Gaussian bridge; ``blend=1`` puts the base at the
moment-matched new-posterior estimate; ``blend=0.5`` (default for
SF) sits at the BW midpoint — closer to π_new than the prev
posterior, so the tempering bridge has less distance to cover and
less prev-posterior bias to compound across windows.

Why this should help SWAT-class problems:
  The Gaussian bridge concentrates particles tightly (per-dim SD
  ~1e-2 for SWAT's 35-dim posterior) — it can't shed accumulated
  bias from previous windows. The SF base at the BW midpoint has
  higher variance than either endpoint when the endpoints differ,
  so the tempering bridge starts wider and is less biased.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp


# ─────────────────────────────────────────────────────────────────────
# Bures-Wasserstein geodesic between Gaussians
# ─────────────────────────────────────────────────────────────────────

def _matrix_sqrt_psd(A: jnp.ndarray) -> jnp.ndarray:
    """Symmetric matrix square root of a PSD matrix via eigendecomposition.

    A is assumed symmetric PSD up to numerical noise; eigenvalues are
    clamped to ``>= 0`` before sqrt.
    """
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)
    eigvals_clipped = jnp.maximum(eigvals, 0.0)
    sqrt_eigvals = jnp.sqrt(eigvals_clipped)
    return (eigvecs * sqrt_eigvals[None, :]) @ eigvecs.T


def _matrix_invsqrt_psd(A: jnp.ndarray, eps: float = 1e-10) -> jnp.ndarray:
    """Symmetric inverse square root of a PSD matrix via eigendecomposition.

    Eigenvalues clamped to ``>= eps`` for numerical stability.
    """
    A_sym = 0.5 * (A + A.T)
    eigvals, eigvecs = jnp.linalg.eigh(A_sym)
    eigvals_clipped = jnp.maximum(eigvals, eps)
    invsqrt = 1.0 / jnp.sqrt(eigvals_clipped)
    return (eigvecs * invsqrt[None, :]) @ eigvecs.T


def bures_wasserstein_map(S0: jnp.ndarray, S1: jnp.ndarray) -> jnp.ndarray:
    """Bures-Wasserstein optimal transport map ``T`` from N(0, S0) to N(0, S1).

    Defined via :math:`T = S_0^{-1/2} (S_0^{1/2} S_1 S_0^{1/2})^{1/2} S_0^{-1/2}`.
    The push-forward of ``N(0, S0)`` by ``y → T y`` is ``N(0, S1)``.

    Pure JAX, jittable.
    """
    S0_half = _matrix_sqrt_psd(S0)
    inner = S0_half @ S1 @ S0_half
    inner_half = _matrix_sqrt_psd(inner)
    S0_invhalf = _matrix_invsqrt_psd(S0)
    return S0_invhalf @ inner_half @ S0_invhalf


def bw_geodesic(
    m0: jnp.ndarray, S0: jnp.ndarray,
    m1: jnp.ndarray, S1: jnp.ndarray,
    t: float,
    entropy_reg: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Bures-Wasserstein geodesic between Gaussians at time ``t``.

    ::

        m_t = (1-t) m0 + t m1
        S_t = ((1-t) I + t T) S0 ((1-t) I + t T) + entropy_reg * t (1-t) I

    where ``T`` is the BW transport map from ``S0`` to ``S1``. The
    ``entropy_reg`` term is the Schrödinger-bridge entropic
    regularisation (Janati et al. 2020); ``0`` recovers exact OT.

    At ``t=0`` returns (m0, S0); at ``t=1`` returns (m1, S1).
    """
    d = m0.shape[0]
    m_t = (1.0 - t) * m0 + t * m1
    T_map = bures_wasserstein_map(S0, S1)
    M_t = (1.0 - t) * jnp.eye(d) + t * T_map
    S_t = M_t @ S0 @ M_t.T
    if entropy_reg > 0.0:
        S_t = S_t + entropy_reg * t * (1.0 - t) * jnp.eye(d)
    # Symmetrise to dampen numerical asymmetry.
    return m_t, 0.5 * (S_t + S_t.T)


# ─────────────────────────────────────────────────────────────────────
# Importance-weighted target moment-match
# ─────────────────────────────────────────────────────────────────────

def estimate_target_gaussian(
    prev_particles: jnp.ndarray,
    log_w_unnorm: jnp.ndarray,
    *,
    floor_eff_n: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """Importance-weighted moment-match: estimate ``N(m1, S1)`` of the
    new posterior using prev particles weighted by
    ``log_w_unnorm = new_ld(u) - log q_0(u)``.

    The weights are normalised via softmax (numerically stable). When
    the effective sample size is degenerate (``< floor_eff_n``), falls
    back to uniform weighting so the moment-match doesn't collapse to
    a few particles.

    Returns:
        m1   : (d,) weighted mean
        S1   : (d, d) weighted covariance + small reg
        n_eff: effective sample size (float)

    The raw covariance gets a small ``1e-4 I`` regulariser to ensure
    positive-definiteness for the BW-square-root step downstream.
    """
    log_w_norm = log_w_unnorm - jax.scipy.special.logsumexp(log_w_unnorm)
    w = jnp.exp(log_w_norm)                 # (N,)
    n_eff = float(1.0 / jnp.sum(w ** 2))

    # ESS floor: revert to uniform weighting if too degenerate.
    use_uniform = n_eff < floor_eff_n
    w_safe = jnp.where(use_uniform,
                        jnp.ones_like(w) / w.shape[0],
                        w)

    m1 = jnp.sum(w_safe[:, None] * prev_particles, axis=0)
    diffs = prev_particles - m1[None, :]
    S1_raw = (w_safe[:, None] * diffs).T @ diffs
    d = m1.shape[0]
    S1 = S1_raw + 1e-4 * jnp.eye(d)
    return m1, S1, n_eff


# ─────────────────────────────────────────────────────────────────────
# Path B: annealed q1 estimation via mini-SMC
# ─────────────────────────────────────────────────────────────────────

def estimate_target_gaussian_annealed(
    prev_particles: jnp.ndarray,
    new_ld_fn: Callable,
    *,
    n_stages: int = 3,
    n_mh_steps: int = 2,
    proposal_scale: float = 0.4,
    rng_key=None,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float]:
    """K-stage annealed q1 estimation (Path B).

    A short tempered-SMC chain replaces the single importance-sampling
    step of ``estimate_target_gaussian``. At fractional inverse-
    temperatures ``beta_k = k/K`` for ``k = 1, ..., K``:

      1. Reweight by incremental log-likelihood
         ``(beta_k - beta_{k-1}) * new_ld(u) = (1/K) * new_ld(u)``.
      2. Systematic resample.
      3. Apply ``n_mh_steps`` random-walk Metropolis-Hastings moves at
         temperature ``beta_k`` with a Gaussian proposal scaled by
         ``proposal_scale * sqrt(emp_cov)``.

    After K stages the particles approximate samples from
    ``q_0 * pi_new = pi_new`` (up to a constant). q1 is the unweighted
    moments of the moved particles.

    Path A (single-step IS) collapses in high dimensions when the
    new-window likelihood is much sharper than the prev-posterior
    Gaussian. The K-stage chain breaks the same total likelihood into
    K manageable pieces, each one ``1/K`` of the total log-weight,
    keeping per-stage ESS healthy.

    Args:
        prev_particles: (N, d) prev-window posterior particles.
        new_ld_fn: callable u -> log pi_new(u).
        n_stages: number of tempering stages K.
        n_mh_steps: RW-MH moves per stage.
        proposal_scale: RW-MH step size; default 0.4 ~ 2.38/sqrt(35)
                        (Roberts-Gelman-Gilks optimal for 35-D).
        rng_key: JAX PRNG key.

    Returns:
        m1   : (d,) sample mean of final particles
        S1   : (d, d) sample covariance + 1e-4 reg
        n_eff_min: minimum per-stage ESS (diagnostic)
        accept_mean: mean MH acceptance across all stages (diagnostic)
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    particles = jnp.asarray(prev_particles, dtype=jnp.float64)
    N, d = particles.shape
    K = int(n_stages)

    # One initial new_ld eval so MH ratios at stage 1 have a baseline.
    log_p_curr = jax.vmap(new_ld_fn)(particles)

    n_eff_min = float(N)
    accept_sum = 0.0
    accept_count = 0

    for k in range(1, K + 1):
        # 1. Reweight by (1/K) * new_ld and resample
        log_w_unnorm = (1.0 / K) * log_p_curr
        log_w_norm = log_w_unnorm - jax.scipy.special.logsumexp(log_w_unnorm)
        w = jnp.exp(log_w_norm)
        n_eff = float(1.0 / jnp.sum(w ** 2))
        n_eff_min = min(n_eff_min, n_eff)

        rng_key, sub = jax.random.split(rng_key)
        idx = jax.random.choice(sub, N, shape=(N,), p=w)   # multinomial; cheap
        particles = particles[idx]
        log_p_curr = log_p_curr[idx]

        # 2. RW-MH at temperature beta_k = k/K with empirical-cov proposal
        beta_k = k / K
        emp_cov = jnp.cov(particles.T) + 1e-6 * jnp.eye(d)
        L_prop = jnp.linalg.cholesky(proposal_scale ** 2 * emp_cov)

        for _ in range(n_mh_steps):
            rng_key, sub_n, sub_u = jax.random.split(rng_key, 3)
            noise = jax.random.normal(sub_n, particles.shape, dtype=jnp.float64)
            proposals = particles + noise @ L_prop.T
            log_p_prop = jax.vmap(new_ld_fn)(proposals)
            log_alpha = beta_k * (log_p_prop - log_p_curr)
            log_u = jnp.log(jax.random.uniform(sub_u, (N,), dtype=jnp.float64))
            accept = log_u < log_alpha
            particles = jnp.where(accept[:, None], proposals, particles)
            log_p_curr = jnp.where(accept, log_p_prop, log_p_curr)
            accept_sum += float(jnp.mean(accept.astype(jnp.float64)))
            accept_count += 1

    m1 = jnp.mean(particles, axis=0)
    diffs = particles - m1[None, :]
    S1_raw = diffs.T @ diffs / N
    S1 = S1_raw + 1e-4 * jnp.eye(d)
    accept_mean = accept_sum / max(accept_count, 1)
    return m1, S1, n_eff_min, accept_mean


# ─────────────────────────────────────────────────────────────────────
# SF base measure for the rolling-window bridge
# ─────────────────────────────────────────────────────────────────────

def fit_sf_base(
    prev_particles: jnp.ndarray,
    new_ld_fn: Callable,
    *,
    blend: float = 0.5,
    entropy_reg: float = 0.0,
    lw_shrinkage: float = 1e-2,
    q1_mode: str = 'is',
    annealed_n_stages: int = 3,
    annealed_n_mh_steps: int = 2,
    annealed_proposal_scale: float = 0.4,
    use_q0_cov: bool = False,
    info_aware: bool = False,
    info_lambda_thresh_quantile: float = 0.5,
    info_blend_temperature: float = 1.0,
    rng_key=None,
):
    """Compute the SF bridge base measure for tempered SMC.

    Pipeline:

      1. Gaussian fit ``q_0 = N(m_0, S_0)`` of prev posterior with
         Ledoit-Wolf-style shrinkage at level ``lw_shrinkage``.
      2. Importance-weighted Gaussian estimate
         ``q_1 = N(m_1, S_1)`` of the new posterior, using prev
         particles weighted by ``new_ld(u) - log q_0(u)``.
      3. Bures-Wasserstein geodesic between ``q_0`` and ``q_1`` at
         time ``t = blend``, with optional entropic regularisation.
      4. Return ``(m_blend, S_blend)`` plus diagnostic info.

    ``blend = 0`` recovers the prev-posterior Gaussian (current
    bridge_type='gaussian'). ``blend = 1`` jumps directly to the
    importance-matched estimate of the new posterior. ``blend = 0.5``
    (default) sits at the BW midpoint — closer to π_new than the
    Gaussian bridge but still anchored on the prev-posterior shape.

    Args:
        prev_particles: (N, d) prev-window posterior particles
                        (in unconstrained space).
        new_ld_fn: callable u -> log π_new(u). Vectorised internally.
        blend: t parameter for BW geodesic, in [0, 1].
        entropy_reg: Schrödinger entropic regularisation. 0 = exact OT.
        lw_shrinkage: ridge factor for Gaussian fits (additive
                      ``lw_shrinkage * tr(S)/d * I``).

    Returns:
        dict with keys:
          'm', 'S':         the SF-base mean and covariance
          'L_chol':         Cholesky of S
          'L_inv':          inverse of L_chol (for log-density evals)
          'log_det':        log|S|
          'q0_mean', 'q0_cov': Gaussian fit of prev posterior
          'q1_mean', 'q1_cov': moment-matched new-posterior estimate
          'n_eff':          effective sample size of the IS weights
          'blend':          t value used
          'entropy_reg':    entropy_reg used
    """
    prev = jnp.asarray(prev_particles, dtype=jnp.float64)
    N, d = prev.shape

    # ── q_0: prev-posterior Gaussian fit (with LW-style shrinkage) ───
    m0 = jnp.mean(prev, axis=0)
    diffs0 = prev - m0[None, :]
    S0_raw = diffs0.T @ diffs0 / N
    tr_S0 = jnp.trace(S0_raw)
    S0 = (1.0 - lw_shrinkage) * S0_raw + lw_shrinkage * (tr_S0 / d) * jnp.eye(d)
    S0 = S0 + 1e-4 * jnp.eye(d)

    # log q_0 evaluated at each prev particle (for IS weights)
    L0 = jnp.linalg.cholesky(S0)
    L0_inv = jax.scipy.linalg.solve_triangular(L0, jnp.eye(d), lower=True)
    log_det_S0 = 2.0 * jnp.sum(jnp.log(jnp.diag(L0)))
    const0 = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det_S0)

    def log_q0(u):
        diff = u - m0
        return const0 - 0.5 * jnp.sum((L0_inv @ diff) ** 2)

    # ── q_1: estimate of π_new ───────────────────────────────────────
    if q1_mode == 'is':
        # Path A: single-step importance-sampling moment-match
        log_q0_vec = jax.vmap(log_q0)(prev)
        log_p_vec = jax.vmap(new_ld_fn)(prev)
        log_w_unnorm = log_p_vec - log_q0_vec
        m1, S1, n_eff = estimate_target_gaussian(prev, log_w_unnorm)
        accept_mean = float('nan')   # not applicable for IS
    elif q1_mode == 'annealed':
        # Path B: K-stage tempered-SMC with RW-MH mutation
        m1, S1, n_eff, accept_mean = estimate_target_gaussian_annealed(
            prev, new_ld_fn,
            n_stages=annealed_n_stages,
            n_mh_steps=annealed_n_mh_steps,
            proposal_scale=annealed_proposal_scale,
            rng_key=rng_key,
        )
    else:
        raise ValueError(
            f"q1_mode must be 'is' or 'annealed', got {q1_mode!r}")

    # ── Bridge base: BW geodesic, optionally decoupled (cov from q0) ──
    info_diagnostics = None
    if use_q0_cov:
        # Decoupled mode (issue #3 fix 2): trust q1's mean (where the
        # new posterior lives) but reuse q0's narrow, well-conditioned
        # cov. Sidesteps the BW-geodesic over-inflation that arises
        # when the annealed q1 cov is dominated by RW-MH noise rather
        # than genuine posterior breadth.
        if info_aware:
            # Information-aware mean update: per-eigenvector blend keyed
            # off the new-window FIM at q0's mean. In well-identified
            # eigendirections the bridge moves toward q1's mean (same as
            # scalar `blend`); in weakly-identified directions it stays
            # at m0, preventing the cross-window drift that compounds
            # in a wholly information-naive bridge.
            F_hat = estimate_fim_hessian(new_ld_fn, m0)
            eigvecs_F, blend_per_eig = per_eigenvector_blend(
                F_hat,
                sf_blend=float(blend),
                lambda_thresh_quantile=float(info_lambda_thresh_quantile),
                blend_temperature=float(info_blend_temperature),
            )
            m_blend = info_aware_bridge_mean(m0, m1, eigvecs_F, blend_per_eig)
            # Eigenvalues for diagnostics (recompute — cheap, d ~ 9-35):
            eigvals_F = jnp.linalg.eigvalsh(F_hat)
            info_diagnostics = {
                'fim_eigvals': eigvals_F,
                'blend_per_eig': blend_per_eig,
                'lambda_thresh_quantile': float(info_lambda_thresh_quantile),
                'blend_temperature': float(info_blend_temperature),
            }
        else:
            m_blend = (1.0 - float(blend)) * m0 + float(blend) * m1
        S_blend = S0
    else:
        m_blend, S_blend = bw_geodesic(
            m0, S0, m1, S1, t=float(blend),
            entropy_reg=float(entropy_reg),
        )
    S_blend = S_blend + 1e-4 * jnp.eye(d)
    L_blend = jnp.linalg.cholesky(S_blend)
    L_inv_blend = jax.scipy.linalg.solve_triangular(
        L_blend, jnp.eye(d), lower=True)
    log_det_blend = 2.0 * jnp.sum(jnp.log(jnp.diag(L_blend)))

    return {
        'm': m_blend,
        'S': S_blend,
        'L_chol': L_blend,
        'L_inv': L_inv_blend,
        'log_det': log_det_blend,
        'q0_mean': m0,
        'q0_cov': S0,
        'q1_mean': m1,
        'q1_cov': S1,
        'n_eff': n_eff,
        'accept_mean': accept_mean,
        'q1_mode': q1_mode,
        'use_q0_cov': bool(use_q0_cov),
        'info_aware': bool(info_aware),
        'info_diagnostics': info_diagnostics,
        'blend': float(blend),
        'entropy_reg': float(entropy_reg),
    }


# ─────────────────────────────────────────────────────────────────────
# Information-aware bridge: per-eigenvector blend keyed off local FIM
# ─────────────────────────────────────────────────────────────────────

def estimate_fim_hessian(
    new_ld_fn: Callable,
    location: jnp.ndarray,
    *,
    eps_clip: float = 0.0,
) -> jnp.ndarray:
    """Estimate the Fisher information matrix at ``location`` via the
    negative Hessian of ``new_ld_fn``.

    For a log-likelihood :math:`L(\\theta) = \\log p(y \\mid \\theta)`,
    the observed Fisher information is :math:`F = -\\nabla^2 L`. At the
    posterior peak this matches the expected FIM under regularity; off-
    peak it is the local quadratic approximation of identifiability.

    Args:
        new_ld_fn: callable θ → log p(y_window | θ). JAX-traceable; can
            be a stochastic estimator (e.g. inner-PF marginal LL) — the
            Hessian holds the RNG fixed via closure, so gradients are
            deterministic.
        location: (d,) point at which to evaluate the Hessian. Typically
            the prev-posterior mean.
        eps_clip: numerical floor on eigenvalues to ensure PSD; eigvals
            below this are set to ``eps_clip`` after eigendecomposition.

    Returns:
        F: (d, d) symmetric PSD Fisher information estimate.
    """
    H = jax.hessian(new_ld_fn)(location)
    F = -0.5 * (H + H.T)        # negate + symmetrise
    # Clip eigenvalues to non-negative (PSD enforcement)
    eigvals, eigvecs = jnp.linalg.eigh(F)
    eigvals_clipped = jnp.maximum(eigvals, eps_clip)
    return (eigvecs * eigvals_clipped[None, :]) @ eigvecs.T


def per_eigenvector_blend(
    F: jnp.ndarray,
    *,
    sf_blend: float = 0.7,
    lambda_thresh_quantile: float = 0.5,
    blend_temperature: float = 1.0,
    floor_eps: float = 1e-30,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the per-eigenvector blend matrix from a FIM estimate.

    Given a FIM ``F = V Λ Vᵀ``, the per-eigenvector blend is::

        blend_i = sf_blend · sigmoid( log(λ_i / λ_thresh) / τ )

    where ``λ_thresh`` is the configured quantile of the eigenvalue
    spectrum. Directions with ``λ_i ≫ λ_thresh`` get blend ≈ sf_blend
    (move toward q1 mean as in the standard SF bridge); directions with
    ``λ_i ≪ λ_thresh`` get blend ≈ 0 (hold at q0 mean — preventing the
    accumulation of drift in weakly-informed directions across rolling
    windows).

    Args:
        F: (d, d) symmetric PSD FIM estimate (e.g. from
            ``estimate_fim_hessian``).
        sf_blend: scalar blend used in the standard SF bridge (default 0.7).
        lambda_thresh_quantile: which eigenvalue quantile to use as the
            soft threshold. 0.5 = median; 0 = smallest; 1 = largest.
        blend_temperature: τ in the sigmoid; smaller = sharper transition.
            Default 1.0 ≈ half-decade soft threshold in log-eigenvalue space.
        floor_eps: numerical floor on eigenvalues before log (handles
            structurally null directions).

    Returns:
        eigvecs:    (d, d) eigenvector matrix V (columns sorted ascending λ).
        blend_diag: (d,)  per-eigenvector blend coefficients (in same
                          eigenvector order).
    """
    eigvals, eigvecs = jnp.linalg.eigh(F)
    eigvals_safe = jnp.maximum(eigvals, floor_eps)
    log_lam = jnp.log(eigvals_safe)
    log_thresh = jnp.quantile(log_lam, float(lambda_thresh_quantile))
    z = (log_lam - log_thresh) / float(blend_temperature)
    blend_per_eig = float(sf_blend) * jax.nn.sigmoid(z)
    return eigvecs, blend_per_eig


def info_aware_bridge_mean(
    m0: jnp.ndarray,
    m1: jnp.ndarray,
    eigvecs: jnp.ndarray,
    blend_per_eig: jnp.ndarray,
) -> jnp.ndarray:
    """Apply per-eigenvector blend in the eigenbasis::

        m_bridge = m0 + V diag(blend_i) Vᵀ (m1 - m0)

    In the well-identified subspace (high λ → blend → sf_blend) this
    reduces to the standard SF mean update. In the weakly-identified
    subspace (low λ → blend → 0) the bridge holds q0's mean fixed,
    which prevents random-walk drift in unidentified directions across
    rolling windows.
    """
    delta = m1 - m0
    delta_in_eig = eigvecs.T @ delta
    delta_blended = blend_per_eig * delta_in_eig
    delta_back = eigvecs @ delta_blended
    return m0 + delta_back


__all__ = [
    "bw_geodesic",
    "bures_wasserstein_map",
    "estimate_target_gaussian",
    "estimate_target_gaussian_annealed",
    "fit_sf_base",
    "estimate_fim_hessian",
    "per_eigenvector_blend",
    "info_aware_bridge_mean",
]
