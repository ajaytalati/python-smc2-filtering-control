"""Private kernel helpers for the Gaussian-kernel DPF.

Date:    18 April 2026
Version: 1.2  (adds ESS-scaled-bandwidth resamplers for v1/v2 tuner-regression fix)

Changelog
---------
v1.2 (18 Apr 2026)
    - Added ``smooth_resample_ess_scaled`` — used by the rewritten v1.
      Multiplies the Silverman bandwidth by an ESS-dependent factor
      f(ESS/K) = (1 - ESS/K)^2.  When the cloud is healthy (ESS ≈ K),
      the effective bandwidth → 0 and the blend becomes a near-identity;
      when it is degenerate (ESS ≈ 1), the effective bandwidth equals the
      Silverman default and the full v0 blend fires.

      Crucially this replaces the *discontinuous* branch
          if ESS < K/2: resample     else: pass-through
      with a *continuous* mapping ESS → bandwidth → blend amount.  The
      XLA graph shape is therefore identical to v0 (always-on blend),
      fixing the MCLMC tuner regression that killed the original v1
      (documented in TESTING_GK_DPF_VERSIONS.md, Bug 1).

    - Added ``smooth_resample_ess_scaled_lw`` — used by the rewritten v2.
      Same ESS-scaled-bandwidth structure as above, but with the Liu-West
      shrinkage correction applied after the blend.

    - No changes to v0.  ``smooth_resample_basic`` and ``smooth_resample``
      are left untouched so v0 and the existing ``gk_dpf.py`` continue to
      work identically.

v1.1 (17 Apr 2026)
    - Added ``compute_ess`` — computes Effective Sample Size from log-weights.
    - Added Liu-West shrinkage correction inside ``smooth_resample``.

v1.0 (17 Apr 2026) — initial implementation.

Why ESS-scaled bandwidth fixes the tuner regression
---------------------------------------------------
In v1/v2 as originally written, the resample decision was
    do_resample = (ess < K/2) & (has_obs > 0.5)
    particles_next = jnp.where(do_resample, blended, new_particles)

The problem: as theta varies during MCLMC tuning, different scan indices
cross the ESS threshold at different theta values.  The log-density is
therefore a piecewise-smooth function of theta with jump discontinuities
at each threshold crossing.  MCLMC's step-size tuner, faced with apparent
infinite-variance gradients, picks tiny step sizes (eps = 0.012 vs v0's
eps = 2.39 — 200× smaller).  Result: ~3.8× more log-density evaluations
per proposal, 0.1 steps/s vs v0's 0.3, fails the 30-min budget.

The fix replaces the discontinuous gate with a continuous ESS-scaled
bandwidth.  The blend runs at every step (same graph shape as v0), but
when ESS is high the bandwidth is near-zero and the blend is a near-
identity — zero bandwidth contamination.  When ESS drops, the bandwidth
smoothly increases and the cloud is resampled.  No discontinuity →
tuner behaves as in v0.

This directly addresses the two scientific goals of v1/v2:
    1. Reduce bandwidth contamination in healthy regimes (v0's bug).
    2. Still resample when the cloud degenerates.
Both goals are achieved by a single smooth operator instead of a branch.

Public functions
----------------
compute_ess(log_w)
    Effective sample size from un-normalised log-weights.

silverman_bandwidth(particles, stochastic_idx, K, scale) -> h vector
log_kernel_matrix(particles, stochastic_idx, h) -> (K, K) log-kernel

smooth_resample_basic(particles, log_w, stochastic_idx, K, scale)
    [v0, v1-old] Kernel blend with no Liu-West correction.

smooth_resample(particles, log_w, stochastic_idx, K, scale)
    [v2-old] Kernel blend with Liu-West correction.  Fixed Silverman bandwidth.

smooth_resample_ess_scaled(particles, log_w, stochastic_idx, K, scale)
    [v1.2 — rewritten v1] Kernel blend with ESS-scaled bandwidth.
    No Liu-West.  Always-on, no branching.

smooth_resample_ess_scaled_lw(particles, log_w, stochastic_idx, K, scale)
    [v1.2 — rewritten v2] Kernel blend with ESS-scaled bandwidth and
    Liu-West correction.  Always-on, no branching.

References
----------
Liu, J. & West, M. (2001). Combined Parameter and State Estimation in
    Simulation-Based Filtering.  Sequential Monte Carlo Methods in Practice.

GK_DPF_ALGORITHM.md §5.6, §6, §12 — algorithm specification.
TESTING_GK_DPF_VERSIONS.md Bug 1 — tuner-regression diagnosis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


# ── Module-level constants ──────────────────────────────────────────────────

# Bandwidth floor: prevents divide-by-zero on near-degenerate state components.
_MIN_BW = 1e-6

# ESS threshold for the OLD (branching) resampling trigger.  Kept for
# backwards compatibility with the old v1/v2 code paths that may still
# reference it.  The new ESS-scaled code paths do not use a threshold.
_ESS_FRAC = 0.5


# ── Public functions ────────────────────────────────────────────────────────


def compute_ess(log_w: Array) -> Array:
    """Effective Sample Size from un-normalised log-weights.

    ESS = 1 / Σ_i w̃_i²   where w̃_i = w_i / Σ_j w_j  are normalised weights.

    Interpretation:
        ESS = K  — all particles have equal weight (maximum diversity).
        ESS = 1  — all probability mass on a single particle (degenerate).

    Args:
        log_w: shape (K,) — un-normalised log-weights.

    Returns:
        Scalar ESS in [1, K].
    """
    log_w_norm = log_w - jax.nn.logsumexp(log_w)
    log_sum_w2 = jax.nn.logsumexp(2.0 * log_w_norm)
    return jnp.exp(-log_sum_w2)


def silverman_bandwidth(particles: Array,
                         stochastic_idx: Array,
                         K: int,
                         scale: float) -> Array:
    """Per-component Silverman-rule bandwidth for the Gaussian kernel.

    h_d = scale × (4 / (n_st + 2))^{1/(n_st+4)} × K^{-1/(n_st+4)} × σ_d

    Deterministic state dimensions are assigned h_d = 1e6 so they do not
    contribute to the pairwise kernel.

    Args:
        particles:      shape (K, n_s).
        stochastic_idx: shape (n_st,) — indices of stochastic state dims.
        K:              int — particle count.
        scale:          scalar multiplier on the Silverman default.

    Returns:
        h: shape (n_s,) — per-component bandwidths.
    """
    n_s  = particles.shape[1]
    n_st = stochastic_idx.shape[0]

    silverman_factor = (4.0 / (n_st + 2.0)) ** (1.0 / (n_st + 4.0))
    k_factor         = K ** (-1.0 / (n_st + 4.0))
    factor           = silverman_factor * k_factor * scale

    std = jnp.std(particles, axis=0)
    h_st = jnp.maximum(factor * std, _MIN_BW)

    h_full = jnp.full((n_s,), 1e6, dtype=particles.dtype)
    h_full = h_full.at[stochastic_idx].set(h_st[stochastic_idx])
    return h_full


def log_kernel_matrix(particles: Array,
                       stochastic_idx: Array,
                       h: Array) -> Array:
    """Pairwise log-Gaussian kernel matrix on the stochastic subspace.

    L_ij = −½ Σ_{d ∈ stochastic} ((x_i^d − x_j^d) / h_d)²
    """
    sub    = particles[:, stochastic_idx]
    h_sub  = h[stochastic_idx]
    scaled = sub / h_sub[None, :]
    diff   = scaled[:, None, :] - scaled[None, :, :]
    sq     = jnp.sum(diff * diff, axis=-1)
    return -0.5 * sq


def smooth_resample_basic(particles: Array,
                          log_w: Array,
                          stochastic_idx: Array,
                          K: int,
                          bandwidth_scale: float) -> Array:
    """Steps 1-4 only: Silverman + log-kernel + blend.  NO Liu-West correction.

    Used by gk_dpf_v0 (baseline).  Preserved for backwards compatibility
    with the originally-failed gk_dpf_v1 for diagnostic comparisons.
    """
    h       = silverman_bandwidth(particles, stochastic_idx, K, bandwidth_scale)
    L       = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = log_w[None, :] + L
    log_A   = log_w_b - jax.nn.logsumexp(log_w_b, axis=1, keepdims=True)
    A       = jnp.exp(log_A)
    return A @ particles


def _ess_bandwidth_factor(log_w: Array, K: int) -> Array:
    """Smooth ESS-based scaling factor for the bandwidth.

        factor(ESS) = (1 − ESS/K)^2

    Properties:
        - factor → 0 as ESS → K  (healthy cloud: no blending needed)
        - factor → 1 as ESS → 1  (degenerate cloud: full Silverman blending)
        - derivative vanishes at ESS = K (maximum smoothness at the
          "no blending" endpoint)
        - monotonic on [1, K]
        - smooth and differentiable in log_w (hence in theta)

    The quadratic form is chosen so the derivative with respect to ESS
    is zero at ESS = K.  In the normal operating regime (healthy cloud,
    ESS near K) the gradient of the blend with respect to theta via this
    path is near-zero, so the MCLMC tuner behaves as in v0.

    No stop_gradient is applied: the gradient must flow through this
    factor because ESS enters the blend smoothly (not as a binary trigger).

    Args:
        log_w: shape (K,) — un-normalised log-weights.
        K:     int — particle count.

    Returns:
        Scalar in [0, 1].  Multiplies ``bandwidth_scale`` to produce the
        effective bandwidth.
    """
    ess      = compute_ess(log_w)
    ess_frac = ess / jnp.float64(K)                    # ∈ [1/K, 1]
    ess_frac = jnp.clip(ess_frac, 0.0, 1.0)            # safety clip
    return (1.0 - ess_frac) ** 2                       # scalar ∈ [0, 1]


def smooth_resample_ess_scaled(particles: Array,
                                log_w: Array,
                                stochastic_idx: Array,
                                K: int,
                                bandwidth_scale: float) -> Array:
    """[v1.2] Kernel blend with ESS-scaled bandwidth.  NO Liu-West.

    Used by the rewritten gk_dpf_v1.  Replaces the originally-failed
    ESS-threshold-triggered v1:
        OLD:  if ESS < K/2: blend     (discontinuous, breaks MCLMC tuner)
        NEW:  always blend with bandwidth × (1 − ESS/K)^2   (continuous)

    When ESS is near K (healthy cloud), the effective bandwidth is nearly
    zero and the blend is nearly the identity — no bandwidth contamination.
    When ESS drops, the bandwidth grows smoothly toward the full Silverman
    value and the cloud is properly resampled.

    The XLA graph shape is identical to v0's (always-on blend, no branch),
    so the MCLMC tuner finds the same large step sizes as in v0.

    Args:
        particles:       shape (K, n_s).
        log_w:           shape (K,) — un-normalised log-weights.
        stochastic_idx:  shape (n_st,).
        K:               int — particle count.
        bandwidth_scale: scalar multiplier on the Silverman default.

    Returns:
        new_particles: shape (K, n_s).
    """
    ess_factor       = _ess_bandwidth_factor(log_w, K)
    effective_scale  = bandwidth_scale * ess_factor

    h       = silverman_bandwidth(particles, stochastic_idx, K, effective_scale)
    L       = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = log_w[None, :] + L
    log_A   = log_w_b - jax.nn.logsumexp(log_w_b, axis=1, keepdims=True)
    A       = jnp.exp(log_A)
    return A @ particles


def smooth_resample(particles: Array,
                     log_w: Array,
                     stochastic_idx: Array,
                     K: int,
                     bandwidth_scale: float) -> Array:
    """[v2-old] Kernel blend with Liu-West correction.  Fixed bandwidth.

    Used by the original gk_dpf_v2 and gk_dpf.py.  Preserved unchanged.
    For the tuner-regression-fixed v2 use ``smooth_resample_ess_scaled_lw``.
    """
    h = silverman_bandwidth(particles, stochastic_idx, K, bandwidth_scale)
    L = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = log_w[None, :] + L
    log_A   = log_w_b - jax.nn.logsumexp(log_w_b, axis=1, keepdims=True)
    A = jnp.exp(log_A)
    blended = A @ particles

    n_st = stochastic_idx.shape[0]
    silverman_factor = (4.0 / (n_st + 2.0)) ** (1.0 / (n_st + 4.0))
    k_factor         = jnp.float64(K) ** (-1.0 / (n_st + 4.0))
    h_norm           = silverman_factor * k_factor * bandwidth_scale
    a = jnp.sqrt(jnp.clip(1.0 - h_norm ** 2, 0.0, 1.0))

    w_norm = jnp.exp(log_w - jax.nn.logsumexp(log_w))
    mu_w   = jnp.sum(w_norm[:, None] * particles, axis=0)

    corrected = a * blended + (1.0 - a) * mu_w[None, :]
    return corrected


def smooth_resample_ess_scaled_lw(particles: Array,
                                   log_w: Array,
                                   stochastic_idx: Array,
                                   K: int,
                                   bandwidth_scale: float) -> Array:
    """[v1.2] Kernel blend with ESS-scaled bandwidth AND Liu-West correction.

    Used by the rewritten gk_dpf_v2.  Combines the two fixes:

    1. ESS-scaled bandwidth (from v1 fix): factor = (1 − ESS/K)^2 applied
       to the Silverman bandwidth.  Gives near-identity blend in healthy
       regimes; smooth in theta so the MCLMC tuner behaves as in v0.

    2. Liu-West shrinkage correction: deterministic rescaling that
       reverses the variance compression caused by the blend.  The
       shrinkage factor ``a`` is computed using the *effective* bandwidth
       (bandwidth_scale × ess_factor), which is the same scale used in
       the blend.  When the blend is near-identity, the Liu-West
       correction is also near-identity — self-consistent.

    Args:
        particles:       shape (K, n_s).
        log_w:           shape (K,) — un-normalised log-weights.
        stochastic_idx:  shape (n_st,).
        K:               int — particle count.
        bandwidth_scale: scalar multiplier on the Silverman default.

    Returns:
        new_particles: shape (K, n_s).
    """
    # ── ESS-scaled effective bandwidth ─────────────────────────────────────
    ess_factor      = _ess_bandwidth_factor(log_w, K)
    effective_scale = bandwidth_scale * ess_factor

    # ── Standard Silverman + kernel + blend with the scaled bandwidth ─────
    h = silverman_bandwidth(particles, stochastic_idx, K, effective_scale)
    L = log_kernel_matrix(particles, stochastic_idx, h)
    log_w_b = log_w[None, :] + L
    log_A   = log_w_b - jax.nn.logsumexp(log_w_b, axis=1, keepdims=True)
    A = jnp.exp(log_A)
    blended = A @ particles

    # ── Liu-West correction, calibrated to the EFFECTIVE bandwidth ────────
    # When effective_scale → 0 (healthy ESS), h_norm → 0, a → 1,
    # corrected = blended ≈ particles.  Liu-West is near-identity.
    n_st = stochastic_idx.shape[0]
    silverman_factor = (4.0 / (n_st + 2.0)) ** (1.0 / (n_st + 4.0))
    k_factor         = jnp.float64(K) ** (-1.0 / (n_st + 4.0))
    h_norm           = silverman_factor * k_factor * effective_scale
    a = jnp.sqrt(jnp.clip(1.0 - h_norm ** 2, 0.0, 1.0))

    w_norm = jnp.exp(log_w - jax.nn.logsumexp(log_w))
    mu_w   = jnp.sum(w_norm[:, None] * particles, axis=0)

    corrected = a * blended + (1.0 - a) * mu_w[None, :]
    return corrected
