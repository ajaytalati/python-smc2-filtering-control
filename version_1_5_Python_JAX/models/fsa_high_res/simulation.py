"""FSA v1.5 simulation — pure, stateless. Direct port of
`version_1_5_Julia/models/fsa_high_res/simulation.jl`.

Owns:
  - BINS_PER_DAY (parsed from FSA_STEP_MINUTES env var at module load)
  - DT_BIN_DAYS = 1 / BINS_PER_DAY
  - DEFAULT_PARAMS — 14 dynamics + 3 obs-noise params, in v1.5 basis
  - INIT_STATE — canonical (B=0.05, F=0.30, A=0.10)
  - PINNED_PARAMS — {tau_B, eta, epsilon_A, mu_FF}, fixed at truth
  - sample_obs_bfa — pure Gaussian obs sampler keyed by jax.random PRNGKey
  - params_v15_to_v1 — v1.5 → v1 basis adapter (★ @match site #1)
  - fill_pinned — merge 10 estimated + 4 pinned → 14-field v1.5 dict

Re-parametrisation (option B from FIM gate):
  κ_B is replaced by B_inf = κ_B · τ_B  (steady-state B at Φ=1, no A coupling)
  κ_F is replaced by F_inf = κ_F · τ_F  (steady-state F at Φ=1, no A coupling)

The adapter `params_v15_to_v1` rotates the basis back at the call site
so `_dynamics.drift_jax` (which expects v1 basis) can stay verbatim.
The Julia source uses `Match.jl` to dispatch on Dict vs NamedTuple;
the Python equivalent uses a runtime `isinstance` check.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from models.fsa_high_res._dynamics import TRUTH_PARAMS as _V1_TRUTH


# ── Time-grid (resolves at module load) ────────────────────────────────
_STEP_MIN = int(os.environ.get('FSA_STEP_MINUTES', '60'))
if (60 * 24) % _STEP_MIN != 0:
    raise ValueError(f"FSA_STEP_MINUTES={_STEP_MIN} must divide 1440")
BINS_PER_DAY = (60 * 24) // _STEP_MIN
DT_BIN_DAYS = 1.0 / BINS_PER_DAY


# ── Default parameters: 14 dynamics (v1.5 basis) + 3 pinned obs-noise
# Truth values are computed from v1's TRUTH_PARAMS so the plant
# trajectories are bit-identical at truth — only the basis differs.
DEFAULT_PARAMS = dict(
    # Dynamics in v1.5 parametrisation
    tau_B=_V1_TRUTH['tau_B'],                                     # 42.0
    tau_F=_V1_TRUTH['tau_F'],                                     #  7.0
    B_inf=_V1_TRUTH['kappa_B'] * _V1_TRUTH['tau_B'],              #  0.504
    F_inf=_V1_TRUTH['kappa_F'] * _V1_TRUTH['tau_F'],              #  0.21
    epsilon_A=_V1_TRUTH['epsilon_A'],                             # 0.40
    lambda_A=_V1_TRUTH['lambda_A'],                               # 1.00
    mu_0=_V1_TRUTH['mu_0'],                                       # 0.02
    mu_B=_V1_TRUTH['mu_B'],                                       # 0.30
    mu_F=_V1_TRUTH['mu_F'],                                       # 0.10
    mu_FF=_V1_TRUTH['mu_FF'],                                     # 0.40
    eta=_V1_TRUTH['eta'],                                         # 0.20
    sigma_B=_V1_TRUTH['sigma_B'],                                 # 0.010
    sigma_F=_V1_TRUTH['sigma_F'],                                 # 0.012
    sigma_A=_V1_TRUTH['sigma_A'],                                 # 0.020

    # Observation noise — pinned. The filter does not estimate these;
    # they are constants of the experiment.
    sigma_B_obs=0.005,
    sigma_F_obs=0.005,
    sigma_A_obs=0.005,
)


# Canonical initial state (shared with v1 and v2).
INIT_STATE = dict(B=0.05, F=0.30, A=0.10)


# ── Pinned parameters ──────────────────────────────────────────────────
# Per FIM gate decision (option B + pin); 4 dynamics params not estimated:
#   tau_B   (chronic-B time constant)         — slow, can't be pinned in 14d
#   eta     (Stuart-Landau cubic damping)     — half of (eta, mu_FF) doublet
#   epsilon_A (B-gain autonomic coupling)     — partner in (tau_B, epsilon_A)
#   mu_FF   (Stuart-Landau F² curvature)      — partner in (mu_F, mu_FF) shear

PINNED_PARAMS = dict(
    tau_B=DEFAULT_PARAMS['tau_B'],
    eta=DEFAULT_PARAMS['eta'],
    epsilon_A=DEFAULT_PARAMS['epsilon_A'],
    mu_FF=DEFAULT_PARAMS['mu_FF'],
)


# ── Adapter: v1.5 (B_inf, F_inf, …) → v1 NamedTuple (kappa_B, kappa_F, …)
# ★ @match site #1 from writeup §7.1 / Julia `simulation.jl:110-143`.
# The Julia uses `@match p begin ::Dict => …; ::NamedTuple => … end`.
# Python equivalent: `isinstance` dispatch (Python has no @match for
# this purpose — `match/case` only got patterns in 3.10+ and dispatching
# on type is more idiomatically done with isinstance).
def params_v15_to_v1(p):
    """Convert a v1.5-form parameter container (Dict or attribute-keyed
    container) to a v1-form dict for `_dynamics.drift_jax` and
    `_dynamics.diffusion_state_dep`.

    Args:
        p: dict-like with v1.5 keys (`tau_B`, `tau_F`, `B_inf`, `F_inf`,
            `epsilon_A`, `lambda_A`, `mu_0`, `mu_B`, `mu_F`, `mu_FF`,
            `eta`, `sigma_B`, `sigma_F`, `sigma_A`). May be a regular
            `dict` or anything supporting `__getitem__` with string keys.

    Returns:
        dict with v1 keys (`kappa_B = B_inf / tau_B`,
        `kappa_F = F_inf / tau_F`; others unchanged).
    """
    if isinstance(p, dict):
        get = lambda k: p[k]
    else:
        # NamedTuple / dataclass / attribute-access fallback.
        get = lambda k: getattr(p, k)

    return dict(
        tau_B=get('tau_B'),
        tau_F=get('tau_F'),
        kappa_B=get('B_inf') / get('tau_B'),
        kappa_F=get('F_inf') / get('tau_F'),
        epsilon_A=get('epsilon_A'),
        lambda_A=get('lambda_A'),
        mu_0=get('mu_0'),
        mu_B=get('mu_B'),
        mu_F=get('mu_F'),
        mu_FF=get('mu_FF'),
        eta=get('eta'),
        sigma_B=get('sigma_B'),
        sigma_F=get('sigma_F'),
        sigma_A=get('sigma_A'),
    )


def fill_pinned(estimated, pinned=None):
    """Merge a 10-field `estimated` v1.5 param dict with the 4 pinned
    values (`tau_B`, `eta`, `epsilon_A`, `mu_FF`) into a full 14-field
    v1.5 dict. Maps to `simulation.jl:159-177` (`fill_pinned_nt`).

    Args:
        estimated: dict with the 10 estimated keys: `tau_F, B_inf,
            F_inf, lambda_A, mu_0, mu_B, mu_F, sigma_B, sigma_F, sigma_A`.
        pinned: dict with the 4 pinned keys; defaults to PINNED_PARAMS.

    Returns:
        dict with all 14 v1.5 dynamics fields.
    """
    pin = PINNED_PARAMS if pinned is None else pinned
    return dict(
        tau_B=pin['tau_B'],
        tau_F=estimated['tau_F'],
        B_inf=estimated['B_inf'],
        F_inf=estimated['F_inf'],
        epsilon_A=pin['epsilon_A'],
        lambda_A=estimated['lambda_A'],
        mu_0=estimated['mu_0'],
        mu_B=estimated['mu_B'],
        mu_F=estimated['mu_F'],
        mu_FF=pin['mu_FF'],
        eta=pin['eta'],
        sigma_B=estimated['sigma_B'],
        sigma_F=estimated['sigma_F'],
        sigma_A=estimated['sigma_A'],
    )


# ── Pure obs sampler ───────────────────────────────────────────────────
def sample_obs_bfa(state, params, key):
    """Sample one Gaussian obs per latent. Pure: same `(state, key)`
    always returns the same `(obs_B, obs_F, obs_A)`.

    Args:
        state: array_like of shape (3,) — current [B, F, A].
        params: dict containing `sigma_B_obs`, `sigma_F_obs`, `sigma_A_obs`.
        key: jax.random.PRNGKey.

    Returns:
        dict with keys `obs_B`, `obs_F`, `obs_A` (Python floats — match
        Julia's `NamedTuple{(:obs_B, :obs_F, :obs_A), ...}` shape).
    """
    sigma_B = params['sigma_B_obs']
    sigma_F = params['sigma_F_obs']
    sigma_A = params['sigma_A_obs']
    noise = jax.random.normal(key, (3,), dtype=jnp.float64)
    return dict(
        obs_B=float(state[0] + sigma_B * noise[0]),
        obs_F=float(state[1] + sigma_F * noise[1]),
        obs_A=float(state[2] + sigma_A * noise[2]),
    )
