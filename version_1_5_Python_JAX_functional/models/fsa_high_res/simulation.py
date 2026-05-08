"""FSA v1.5 simulation — purely-functional, typed records.

Functional rewrite of `version_1_5_Python_JAX/models/fsa_high_res/simulation.py`.
All `dict`-keyed parameter containers are now `typing.NamedTuple` records
(`ParamsV15`, `PinnedParams`, `EstimatedDynParams`, `InitState`, `Obs`).
`ParamsV1` lives in `_dynamics.py` (re-exported here).

Owns:
  - BINS_PER_DAY (parsed from FSA_STEP_MINUTES env var at module load)
  - DT_BIN_DAYS = 1 / BINS_PER_DAY
  - DEFAULT_PARAMS — `ParamsV15` (14 dynamics + 3 obs noise) at truth
  - INIT_STATE — `InitState` with canonical (B=0.05, F=0.30, A=0.10)
  - PINNED_PARAMS — `PinnedParams` (tau_B, eta, epsilon_A, mu_FF) at truth
  - sample_obs_bfa — pure Gaussian obs sampler keyed by jax.random PRNGKey
  - params_v15_to_v1 — v1.5 → v1 basis adapter (★ @match site #1)
  - fill_pinned — merge 10-field estimated record + 4 pinned → ParamsV15

Re-parametrisation (option B from FIM gate):
  κ_B is replaced by B_inf = κ_B · τ_B  (steady-state B at Φ=1, no A coupling)
  κ_F is replaced by F_inf = κ_F · τ_F  (steady-state F at Φ=1, no A coupling)

The adapter `params_v15_to_v1` rotates the basis back at the call site
so `_dynamics.drift_jax` (which expects v1 basis) can stay verbatim.
The Julia source uses `Match.jl` to dispatch on Dict vs NamedTuple;
the typed-record port has a single explicit signature
(`ParamsV15` in, `ParamsV1` out) so no runtime dispatch is needed.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp

from models.fsa_high_res._dynamics import ParamsV1, TRUTH_PARAMS_V1


# ── Time-grid (resolves at module load) ────────────────────────────────
_STEP_MIN = int(os.environ.get('FSA_STEP_MINUTES', '60'))
if (60 * 24) % _STEP_MIN != 0:
    raise ValueError(f"FSA_STEP_MINUTES={_STEP_MIN} must divide 1440")
BINS_PER_DAY = (60 * 24) // _STEP_MIN
DT_BIN_DAYS = 1.0 / BINS_PER_DAY


# ── Typed parameter records ────────────────────────────────────────────
class ParamsV15(NamedTuple):
    """14 dynamics params in the v1.5 basis (B_inf, F_inf) + 3 obs sigmas."""
    tau_B:       float
    tau_F:       float
    B_inf:       float
    F_inf:       float
    epsilon_A:   float
    lambda_A:    float
    mu_0:        float
    mu_B:        float
    mu_F:        float
    mu_FF:       float
    eta:         float
    sigma_B:     float
    sigma_F:     float
    sigma_A:     float
    sigma_B_obs: float
    sigma_F_obs: float
    sigma_A_obs: float


class PinnedParams(NamedTuple):
    """The 4 pinned dynamics params (FIM-gate option B)."""
    tau_B:     float
    eta:       float
    epsilon_A: float
    mu_FF:     float


class EstimatedDynParams(NamedTuple):
    """The 10 estimated dynamics params, in the v1.5 basis. Order matches
    `estimation.PARAM_NAMES` so a constrained filter array can be loaded
    field-by-field.
    """
    tau_F:    float
    B_inf:    float
    F_inf:    float
    lambda_A: float
    mu_0:     float
    mu_B:     float
    mu_F:     float
    sigma_B:  float
    sigma_F:  float
    sigma_A:  float


class InitState(NamedTuple):
    """Canonical initial latent state (shared with v1 and v2)."""
    B: float
    F: float
    A: float


class Obs(NamedTuple):
    """One bin's three direct-Gaussian observations."""
    obs_B: float
    obs_F: float
    obs_A: float


# ── Default parameters: 14 dynamics (v1.5 basis) + 3 pinned obs-noise ──
# Truth values are computed from v1's TRUTH_PARAMS_V1 so the plant
# trajectories are bit-identical at truth — only the basis differs.
DEFAULT_PARAMS = ParamsV15(
    # Dynamics in v1.5 parametrisation
    tau_B=TRUTH_PARAMS_V1.tau_B,                                          # 42.0
    tau_F=TRUTH_PARAMS_V1.tau_F,                                          #  7.0
    B_inf=TRUTH_PARAMS_V1.kappa_B * TRUTH_PARAMS_V1.tau_B,                #  0.504
    F_inf=TRUTH_PARAMS_V1.kappa_F * TRUTH_PARAMS_V1.tau_F,                #  0.21
    epsilon_A=TRUTH_PARAMS_V1.epsilon_A,                                  #  0.40
    lambda_A=TRUTH_PARAMS_V1.lambda_A,                                    #  1.00
    mu_0=TRUTH_PARAMS_V1.mu_0,                                            #  0.02
    mu_B=TRUTH_PARAMS_V1.mu_B,                                            #  0.30
    mu_F=TRUTH_PARAMS_V1.mu_F,                                            #  0.10
    mu_FF=TRUTH_PARAMS_V1.mu_FF,                                          #  0.40
    eta=TRUTH_PARAMS_V1.eta,                                              #  0.20
    sigma_B=TRUTH_PARAMS_V1.sigma_B,                                      #  0.010
    sigma_F=TRUTH_PARAMS_V1.sigma_F,                                      #  0.012
    sigma_A=TRUTH_PARAMS_V1.sigma_A,                                      #  0.020

    # Observation noise — pinned. The filter does not estimate these;
    # they are constants of the experiment.
    sigma_B_obs=TRUTH_PARAMS_V1.sigma_B_obs,                              #  0.005
    sigma_F_obs=TRUTH_PARAMS_V1.sigma_F_obs,                              #  0.005
    sigma_A_obs=TRUTH_PARAMS_V1.sigma_A_obs,                              #  0.005
)


# Canonical initial state.
INIT_STATE = InitState(B=0.05, F=0.30, A=0.10)


# ── Pinned parameters ──────────────────────────────────────────────────
# Per FIM gate decision (option B + pin); 4 dynamics params not estimated:
#   tau_B   (chronic-B time constant)         — slow, can't be pinned in 14d
#   eta     (Stuart-Landau cubic damping)     — half of (eta, mu_FF) doublet
#   epsilon_A (B-gain autonomic coupling)     — partner in (tau_B, epsilon_A)
#   mu_FF   (Stuart-Landau F² curvature)      — partner in (mu_F, mu_FF) shear
PINNED_PARAMS = PinnedParams(
    tau_B=DEFAULT_PARAMS.tau_B,
    eta=DEFAULT_PARAMS.eta,
    epsilon_A=DEFAULT_PARAMS.epsilon_A,
    mu_FF=DEFAULT_PARAMS.mu_FF,
)


# ── Adapter: v1.5 (B_inf, F_inf, …) → v1 (kappa_B, kappa_F, …) ─────────
# ★ @match site #1 from writeup §7.1 / Julia `simulation.jl:110-143`.
# The Julia uses `@match p begin ::Dict => …; ::NamedTuple => … end` to
# dispatch on container type. The typed-record port has a single
# signature: `ParamsV15` in, `ParamsV1` out, no runtime dispatch.
def params_v15_to_v1(p: ParamsV15) -> ParamsV1:
    """Convert a v1.5 parameter record to a v1 record for `_dynamics`.

    Args:
        p: `ParamsV15` (14 dynamics in v1.5 basis + 3 obs sigmas).

    Returns:
        `ParamsV1` (14 dynamics in v1 basis + 3 obs sigmas). Only
        `kappa_B = B_inf / tau_B` and `kappa_F = F_inf / tau_F` change;
        all other fields pass through unchanged.
    """
    return ParamsV1(
        tau_B=p.tau_B,
        tau_F=p.tau_F,
        kappa_B=p.B_inf / p.tau_B,
        kappa_F=p.F_inf / p.tau_F,
        epsilon_A=p.epsilon_A,
        lambda_A=p.lambda_A,
        mu_0=p.mu_0,
        mu_B=p.mu_B,
        mu_F=p.mu_F,
        mu_FF=p.mu_FF,
        eta=p.eta,
        sigma_B=p.sigma_B,
        sigma_F=p.sigma_F,
        sigma_A=p.sigma_A,
        sigma_B_obs=p.sigma_B_obs,
        sigma_F_obs=p.sigma_F_obs,
        sigma_A_obs=p.sigma_A_obs,
    )


def fill_pinned(estimated: EstimatedDynParams,
                pinned: PinnedParams = PINNED_PARAMS,
                obs_sigmas: tuple = None) -> ParamsV15:
    """Merge a 10-field `estimated` v1.5 dynamics record with the 4
    pinned values (`tau_B`, `eta`, `epsilon_A`, `mu_FF`) and 3 obs sigmas
    into a full 17-field `ParamsV15`. Maps to `simulation.jl:159-177`
    (`fill_pinned_nt`).

    Args:
        estimated: `EstimatedDynParams` (10 estimated dynamics fields).
        pinned: `PinnedParams` (defaults to `PINNED_PARAMS`).
        obs_sigmas: optional 3-tuple `(sigma_B_obs, sigma_F_obs,
            sigma_A_obs)`; defaults to the truth values from
            `DEFAULT_PARAMS`.

    Returns:
        `ParamsV15` with all 17 fields populated.
    """
    if obs_sigmas is None:
        sB_obs = DEFAULT_PARAMS.sigma_B_obs
        sF_obs = DEFAULT_PARAMS.sigma_F_obs
        sA_obs = DEFAULT_PARAMS.sigma_A_obs
    else:
        sB_obs, sF_obs, sA_obs = obs_sigmas

    return ParamsV15(
        tau_B=pinned.tau_B,
        tau_F=estimated.tau_F,
        B_inf=estimated.B_inf,
        F_inf=estimated.F_inf,
        epsilon_A=pinned.epsilon_A,
        lambda_A=estimated.lambda_A,
        mu_0=estimated.mu_0,
        mu_B=estimated.mu_B,
        mu_F=estimated.mu_F,
        mu_FF=pinned.mu_FF,
        eta=pinned.eta,
        sigma_B=estimated.sigma_B,
        sigma_F=estimated.sigma_F,
        sigma_A=estimated.sigma_A,
        sigma_B_obs=sB_obs,
        sigma_F_obs=sF_obs,
        sigma_A_obs=sA_obs,
    )


# ── Pure obs sampler ───────────────────────────────────────────────────
def sample_obs_bfa(state, params: ParamsV15, key) -> Obs:
    """Sample one Gaussian obs per latent. Pure: same `(state, key)`
    always returns the same `Obs(obs_B, obs_F, obs_A)`.

    Args:
        state: array_like of shape (3,) — current [B, F, A].
        params: `ParamsV15` (uses the 3 sigma_*_obs fields).
        key: jax.random.PRNGKey.

    Returns:
        `Obs` NamedTuple with fields `obs_B`, `obs_F`, `obs_A`.
    """
    noise = jax.random.normal(key, (3,), dtype=jnp.float64)
    return Obs(
        obs_B=state[0] + params.sigma_B_obs * noise[0],
        obs_F=state[1] + params.sigma_F_obs * noise[1],
        obs_A=state[2] + params.sigma_A_obs * noise[2],
    )
