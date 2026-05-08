"""FSA v1.5 estimation — purely-functional, typed records.

Functional rewrite of `version_1_5_Python_JAX/models/fsa_high_res/estimation.py`.
Two refactors vs the imperative source:

  (1) `obs_log_weight` and `propagate` no longer use `for m in range(M)`
      loops with `np.empty` index-assignment; both are now `jax.vmap`'d
      over the (M, 3) particle array.
  (2) `params` is a `ParamsV15` NamedTuple at every model-side call
      site. The framework boundary (the `EstimationModel` instance)
      still receives `frozen_params` as a dict — built once at module
      load via `PinnedParams._asdict()` plus the obs sigmas.

The filter side has just enough to:
  - know the 10-param prior config (PARAM_PRIOR_CONFIG)
  - evaluate the obs log-weight at a given (particles, obs) pair
  - propagate particles forward by one bin (prior-predictive)

All functions pure: same inputs → same outputs.

Note: `propagate` uses the prior-predictive (no locally-guided
proposal) because the obs is so informative (direct Gaussian on each
latent every bin) that prior-predictive does not degenerate. This is
deliberately simpler than v2's Kalman-fused proposal — see
`estimation.jl:11-13` for the rationale on the Julia side.

Also exposes `HIGH_RES_FSA_V15_ESTIMATION` — an `EstimationModel`
instance that the `smc2fc` framework's outer SMC² loop consumes
(same contract as v2's `HIGH_RES_FSA_V2_ESTIMATION`).
"""

from __future__ import annotations

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp

from smc2fc.estimation_model import EstimationModel
from smc2fc._likelihood_constants import HALF_LOG_2PI

from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS, INIT_STATE, PINNED_PARAMS,
    ParamsV15, EstimatedDynParams, Obs,
    params_v15_to_v1, DT_BIN_DAYS,
)


# ── Frozen pinned values (broadcast at every call site) ────────────────
TAU_B_FROZEN     = PINNED_PARAMS.tau_B
ETA_FROZEN       = PINNED_PARAMS.eta
EPSILON_A_FROZEN = PINNED_PARAMS.epsilon_A
MU_FF_FROZEN     = PINNED_PARAMS.mu_FF

SIGMA_B_OBS_FROZEN = DEFAULT_PARAMS.sigma_B_obs
SIGMA_F_OBS_FROZEN = DEFAULT_PARAMS.sigma_F_obs
SIGMA_A_OBS_FROZEN = DEFAULT_PARAMS.sigma_A_obs

# Initial state when the filter cold-starts. Matches the canonical
# (B=0.05, F=0.30, A=0.10) used by the plant.
COLD_START_INIT = jnp.array([INIT_STATE.B, INIT_STATE.F, INIT_STATE.A],
                            dtype=jnp.float64)


# ── 10 estimated v1.5 dynamics-parameter names (filter-vector order) ──
# Matches `estimation.jl:37-42`.
PARAM_NAMES = [
    'tau_F', 'B_inf', 'F_inf',
    'lambda_A',
    'mu_0', 'mu_B', 'mu_F',
    'sigma_B', 'sigma_F', 'sigma_A',
]


# ── Prior config — 10 lognormal entries centred on truth ───────────────
# `(name, (kind, (mu, sigma)))` with kind ∈ {'lognormal', 'normal'}.
# All 10 are lognormal in v1.5 (positivity required everywhere).
# Matches `estimation.jl:62-73`.
PARAM_PRIOR_CONFIG = OrderedDict([
    ('tau_F',    ('lognormal', (math.log(DEFAULT_PARAMS.tau_F),    0.30))),
    ('B_inf',    ('lognormal', (math.log(DEFAULT_PARAMS.B_inf),    0.30))),
    ('F_inf',    ('lognormal', (math.log(DEFAULT_PARAMS.F_inf),    0.30))),
    ('lambda_A', ('lognormal', (math.log(DEFAULT_PARAMS.lambda_A), 0.30))),
    ('mu_0',     ('lognormal', (math.log(DEFAULT_PARAMS.mu_0),     0.30))),
    ('mu_B',     ('lognormal', (math.log(DEFAULT_PARAMS.mu_B),     0.30))),
    ('mu_F',     ('lognormal', (math.log(DEFAULT_PARAMS.mu_F),     0.30))),
    ('sigma_B',  ('lognormal', (math.log(DEFAULT_PARAMS.sigma_B),  0.30))),
    ('sigma_F',  ('lognormal', (math.log(DEFAULT_PARAMS.sigma_F),  0.30))),
    ('sigma_A',  ('lognormal', (math.log(DEFAULT_PARAMS.sigma_A),  0.30))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict()  # init state pinned at COLD_START_INIT

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


def _estimated_from_filter_vector(theta_constrained) -> EstimatedDynParams:
    """Build a 10-field `EstimatedDynParams` from a length-10 constrained
    filter vector. Field order matches `PARAM_NAMES` / `_PI`.
    """
    return EstimatedDynParams(
        tau_F=theta_constrained[_PI['tau_F']],
        B_inf=theta_constrained[_PI['B_inf']],
        F_inf=theta_constrained[_PI['F_inf']],
        lambda_A=theta_constrained[_PI['lambda_A']],
        mu_0=theta_constrained[_PI['mu_0']],
        mu_B=theta_constrained[_PI['mu_B']],
        mu_F=theta_constrained[_PI['mu_F']],
        sigma_B=theta_constrained[_PI['sigma_B']],
        sigma_F=theta_constrained[_PI['sigma_F']],
        sigma_A=theta_constrained[_PI['sigma_A']],
    )


def _params_v15_from_filter_vector(theta_constrained) -> ParamsV15:
    """Build a `ParamsV15` from a length-10 constrained filter vector;
    pinned values come from the `_FROZEN` constants and obs sigmas from
    `DEFAULT_PARAMS`.
    """
    return ParamsV15(
        tau_B=TAU_B_FROZEN,
        tau_F=theta_constrained[_PI['tau_F']],
        B_inf=theta_constrained[_PI['B_inf']],
        F_inf=theta_constrained[_PI['F_inf']],
        epsilon_A=EPSILON_A_FROZEN,
        lambda_A=theta_constrained[_PI['lambda_A']],
        mu_0=theta_constrained[_PI['mu_0']],
        mu_B=theta_constrained[_PI['mu_B']],
        mu_F=theta_constrained[_PI['mu_F']],
        mu_FF=MU_FF_FROZEN,
        eta=ETA_FROZEN,
        sigma_B=theta_constrained[_PI['sigma_B']],
        sigma_F=theta_constrained[_PI['sigma_F']],
        sigma_A=theta_constrained[_PI['sigma_A']],
        sigma_B_obs=SIGMA_B_OBS_FROZEN,
        sigma_F_obs=SIGMA_F_OBS_FROZEN,
        sigma_A_obs=SIGMA_A_OBS_FROZEN,
    )


# ── Pure obs log-weight (per-particle) ────────────────────────────────
def obs_log_weight_one(B, F, A, obs: Obs, params: ParamsV15):
    """Per-particle Gaussian log-weight at one bin.

    Returns
        Σ_X  -0.5·log(2π) − log σ_X − 0.5·((obs_X − X)/σ_X)²
    over X ∈ {B, F, A}. Pure scalar.
    """
    sigma_B = params.sigma_B_obs
    sigma_F = params.sigma_F_obs
    sigma_A = params.sigma_A_obs
    log_norm = (-3.0 * HALF_LOG_2PI
                - jnp.log(sigma_B) - jnp.log(sigma_F) - jnp.log(sigma_A))
    inv2sB2 = 0.5 / (sigma_B * sigma_B)
    inv2sF2 = 0.5 / (sigma_F * sigma_F)
    inv2sA2 = 0.5 / (sigma_A * sigma_A)
    dB = obs.obs_B - B
    dF = obs.obs_F - F
    dA = obs.obs_A - A
    return (log_norm
            - inv2sB2 * dB * dB
            - inv2sF2 * dF * dF
            - inv2sA2 * dA * dA)


def obs_log_weight(particles, obs: Obs, params: ParamsV15):
    """Vectorised over particles. `particles` shape (M, 3).
    Returns jnp.ndarray shape (M,). Pure: replaces the imperative
    `for m in range(M)` over `np.empty` of the dict-based source with
    a single `jax.vmap(obs_log_weight_one, ...)` over the particle
    array.
    """
    parts = jnp.asarray(particles, dtype=jnp.float64)
    return jax.vmap(
        lambda b, f, a: obs_log_weight_one(b, f, a, obs, params)
    )(parts[:, 0], parts[:, 1], parts[:, 2])


# ── Pure propagate (prior-predictive, vectorised over particles) ──────
def propagate(particles, Phi_t, params: ParamsV15, dt: float, key):
    """Advance every particle by ONE Euler-Maruyama step under control
    `Phi_t`. Returns a NEW (M, 3) jnp.ndarray; does NOT mutate
    `particles`.

    Pure prior-predictive — no obs information is folded in (v1.5's obs
    is so informative directly on each latent that prior-predictive
    does not degenerate; matches the rationale at `estimation.jl:11-13`).

    Refactor vs the dict-based source: replaces the imperative
    `for m in range(M)` over `np.empty` with a `jax.vmap` over a
    per-particle one-step EM body.

    Args:
        particles: ndarray of shape (M, 3) in v1.5 latent space.
        Phi_t: scalar — control input.
        params: `ParamsV15` (14 dynamics + 3 obs noise).
        dt: bin width in days.
        key: JAX PRNGKey; per-particle sub-keys are split
            deterministically (same sequence as the imperative version).
    """
    parts = jnp.asarray(particles, dtype=jnp.float64)
    M = parts.shape[0]
    params_v1 = params_v15_to_v1(params)
    sqrt_dt = math.sqrt(dt)
    keys = jax.random.split(key, M)

    def one(y, k):
        d = drift_jax(y, params_v1, Phi_t)
        sigma = diffusion_state_dep(y, params_v1)
        noise = jax.random.normal(k, (3,), dtype=jnp.float64)
        y_pred = y + d * dt + sigma * sqrt_dt * noise
        B = jnp.where(y_pred[0] < 0.0, -y_pred[0],
                      jnp.where(y_pred[0] > 1.0, 2.0 - y_pred[0], y_pred[0]))
        F = jnp.abs(y_pred[1])
        A = jnp.abs(y_pred[2])
        return jnp.array([B, F, A])

    return jax.vmap(one)(parts, keys)


# ── EstimationModel instance for the smc2fc framework ─────────────────
# Same contract as v2's `HIGH_RES_FSA_V2_ESTIMATION`. The framework
# integrates this with its outer SMC² loop and inner GK-DPF.

# Adapter: framework-style `propagate_fn(y, t, dt, params, grid_obs, k,
# sigma_diag, noise, rng_key)` over a SINGLE particle vs the simpler
# v1.5 `propagate(particles, Phi_t, params, dt, key)`.

def _propagate_fn_framework(y, t, dt, params, grid_obs, k,
                              sigma_diag, noise, rng_key):
    """Framework-API single-particle propagate with **Kalman-fused
    proposal** over v1.5's 3 direct-Gaussian obs channels.

    Mirrors v2's `propagate_fn` pattern (sequential-scalar Kalman
    fusion + Cholesky sample + Radon-Nikodym pred_lw correction).

    `params` here is the constrained filter vector (a JAX array of
    length 10). We build a `ParamsV15` from it via
    `_params_v15_from_filter_vector`, then convert to `ParamsV1`.
    """
    del t, sigma_diag, rng_key

    # ── v1.5 Banister Euler drift prediction (v1 basis after rotation) ──
    p_v15 = _params_v15_from_filter_vector(params)
    p_v1 = params_v15_to_v1(p_v15)
    Phi_k = grid_obs['Phi'][k]
    d = drift_jax(y, p_v1, Phi_k)            # (3,) [dB, dF, dA] /day
    B_pred = y[0] + dt * d[0]
    F_pred = y[1] + dt * d[1]
    A_pred = y[2] + dt * d[2]
    mu_prior = jnp.array([B_pred, F_pred, A_pred])

    # ── State-dependent process-noise variance per dimension ──
    # σ_B(B) = sigma_B · √(B(1-B));  similarly for F, A (CIR-style).
    B_cl = jnp.clip(y[0], 1e-4, 1.0 - 1e-4)
    F_cl = jnp.maximum(y[1], 0.0)
    A_cl = jnp.maximum(y[2], 0.0)
    sigma_B_dyn = p_v1.sigma_B
    sigma_F_dyn = p_v1.sigma_F
    sigma_A_dyn = p_v1.sigma_A
    var_B = jnp.maximum(sigma_B_dyn ** 2 * B_cl * (1.0 - B_cl) * dt, 1e-12)
    var_F = jnp.maximum(sigma_F_dyn ** 2 * F_cl * dt, 1e-12)
    var_A = jnp.maximum(sigma_A_dyn ** 2 * (A_cl + 1e-4) * dt, 1e-12)
    P_prior = jnp.diag(jnp.array([var_B, var_F, var_A]))

    # ── Linear obs model: H = I, bias = 0, R = diag(σ_*_obs²) ──
    H = jnp.eye(3, dtype=mu_prior.dtype)
    bias = jnp.zeros(3, dtype=mu_prior.dtype)
    R_diag = jnp.array([SIGMA_B_OBS_FROZEN ** 2,
                         SIGMA_F_OBS_FROZEN ** 2,
                         SIGMA_A_OBS_FROZEN ** 2], dtype=mu_prior.dtype)
    obs_vals = jnp.array([grid_obs['obs_B'][k],
                           grid_obs['obs_F'][k],
                           grid_obs['obs_A'][k]], dtype=mu_prior.dtype)
    # All channels always present in v1.5 (no gating).
    obs_pres = jnp.ones(3, dtype=mu_prior.dtype)

    # ── Sequential scalar Kalman fusion (mirrors v2 line 217-235) ──
    def _kalman_step(carry, ch):
        mu, P, lp = carry
        h_i, b_i, r_i, y_i, pres_i = ch
        innov = y_i - (h_i @ mu + b_i)
        Ph    = P @ h_i
        S_i   = h_i @ Ph + r_i
        K_i   = Ph / S_i
        ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov ** 2 / S_i
        mu = mu + pres_i * K_i * innov
        P  = P  - pres_i * jnp.outer(K_i, Ph)
        lp = lp + pres_i * ll_i
        return (mu, P, lp), None

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
        _kalman_step,
        (mu_prior, P_prior, jnp.asarray(0.0, dtype=mu_prior.dtype)),
        (H, bias, R_diag, obs_vals, obs_pres),
    )

    # ── Sample x_new from fused Gaussian posterior ──
    P_safe = P_fused + jnp.asarray(1e-10, dtype=P_fused.dtype) \
                       * jnp.eye(3, dtype=P_fused.dtype)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise

    # ── Physical bounds (B ∈ [0, 1], F ≥ 0, A ≥ 0) ──
    B_new = jnp.clip(x_new[0], 1e-4, 1.0 - 1e-4)
    F_new = jnp.maximum(x_new[1], 0.0)
    A_new = jnp.maximum(x_new[2], 0.0)
    y_new = jnp.array([B_new, F_new, A_new])

    # ── Weight correction: pred_lw = log_pred_total - obs_ll(y_new) ──
    # Framework re-adds `obs_log_weight_fn(y_new, ...)` at line 156 of
    # gk_dpf_v3_lite.py, so subtracting it here cancels the double-count.
    preds_new = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5 * resids_new ** 2 / R_diag
                                      - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


def _diffusion_fn_framework(params):
    """Diagonal σ for the framework's noise scaling. Reads from the
    constrained filter vector (a JAX array, indexed via `_PI`)."""
    return jnp.array([params[_PI['sigma_B']],
                       params[_PI['sigma_F']],
                       params[_PI['sigma_A']]])


def _obs_log_weight_fn_framework(x_new, grid_obs, k, params):
    """Framework-API per-particle obs log-weight at a single bin.
    `params` is the constrained filter vector (unused — obs sigmas are
    pinned at `_FROZEN` constants)."""
    del params
    obs = Obs(
        obs_B=grid_obs['obs_B'][k],
        obs_F=grid_obs['obs_F'][k],
        obs_A=grid_obs['obs_A'][k],
    )
    obs_params = ParamsV15(
        tau_B=TAU_B_FROZEN, tau_F=0.0, B_inf=0.0, F_inf=0.0,
        epsilon_A=EPSILON_A_FROZEN, lambda_A=0.0,
        mu_0=0.0, mu_B=0.0, mu_F=0.0, mu_FF=MU_FF_FROZEN,
        eta=ETA_FROZEN,
        sigma_B=0.0, sigma_F=0.0, sigma_A=0.0,
        sigma_B_obs=SIGMA_B_OBS_FROZEN,
        sigma_F_obs=SIGMA_F_OBS_FROZEN,
        sigma_A_obs=SIGMA_A_OBS_FROZEN,
    )
    return obs_log_weight_one(x_new[0], x_new[1], x_new[2], obs, obs_params)


def align_obs_fn(obs_data, t_steps, dt):
    """Slice/align grid_obs for one window. v1.5 has 3 direct-on-latent
    obs channels (obs_B, obs_F, obs_A) plus the Φ schedule itself (the
    plant emits Phi alongside its obs since the controller pipes Phi
    in). Same shape contract as v2 — `grid_obs` stays a dict because
    the framework consumes it that way."""
    del dt
    return dict(
        obs_B=jnp.asarray(obs_data['obs_B'][:t_steps], dtype=jnp.float64),
        obs_F=jnp.asarray(obs_data['obs_F'][:t_steps], dtype=jnp.float64),
        obs_A=jnp.asarray(obs_data['obs_A'][:t_steps], dtype=jnp.float64),
        Phi=jnp.asarray(obs_data['Phi'][:t_steps], dtype=jnp.float64),
    )


def _shard_init_fn(time_offset, params, exogenous, global_init):
    """v1.5: no phase-conditioned init logic — pass-through."""
    del time_offset, params, exogenous
    return global_init


def _make_init_state_fn(init_estimates, params):
    """v1.5 has empty `INIT_STATE_PRIOR_CONFIG`, so init is pinned at
    `COLD_START_INIT` regardless of `init_estimates`."""
    del init_estimates, params
    return COLD_START_INIT


# ── Frozen-params dict for the framework boundary ──────────────────────
# `EstimationModel.frozen_params` is typed `Dict[str, float]` upstream,
# so we convert from typed records to dict here (only at this single
# boundary).
_FROZEN_PARAMS_DICT = {
    **PINNED_PARAMS._asdict(),
    'sigma_B_obs': SIGMA_B_OBS_FROZEN,
    'sigma_F_obs': SIGMA_F_OBS_FROZEN,
    'sigma_A_obs': SIGMA_A_OBS_FROZEN,
}


HIGH_RES_FSA_V15_ESTIMATION = EstimationModel(
    name='fsa_high_res_v15',
    version='1.5',
    n_states=3,
    n_stochastic=3,
    stochastic_indices=(0, 1, 2),
    state_bounds=((0.0, 1.0), (0.0, 10.0), (0.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params=_FROZEN_PARAMS_DICT,
    exogenous_keys=('Phi',),
    propagate_fn=_propagate_fn_framework,
    diffusion_fn=_diffusion_fn_framework,
    obs_log_weight_fn=_obs_log_weight_fn_framework,
    align_obs_fn=align_obs_fn,
    shard_init_fn=_shard_init_fn,
    make_init_state_fn=_make_init_state_fn,
)
