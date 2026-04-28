"""Config dataclasses for SMC² + rolling-window estimation.

Replaces the tangle of module-level globals in the original monolithic driver
with explicit, plain-Python config objects that pass through the API.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SMCConfig:
    """Config for the outer tempered-SMC over parameters, plus the inner PF."""

    # Outer SMC
    n_smc_particles: int = 256
    target_ess_frac: float = 0.5
    num_mcmc_steps: int = 5          # cold-start HMC moves per tempering level
    max_lambda_inc: float = 0.05     # cold-start lambda clamp

    # Bridge (warm-start)
    num_mcmc_steps_bridge: int = 3
    max_lambda_inc_bridge: float = 0.10
    bridge_type: str = 'gaussian'      # 'gaussian' (single Gaussian + LW shrinkage)
                                       # 'mog' (2- or 3-component Gaussian mixture)
                                       # 'schrodinger_follmer' (BW geodesic between
                                       #   prev-posterior Gaussian fit and importance-
                                       #   matched new-posterior estimate; see
                                       #   smc2fc.core.sf_bridge)
    bridge_mog_components: int = 2     # only used when bridge_type == 'mog'
    sf_blend: float = 0.5              # only used when bridge_type == 'schrodinger_follmer';
                                       # t in [0, 1] along BW geodesic. 0 = prev posterior,
                                       # 1 = new-posterior moment-match, 0.5 = midpoint
    sf_entropy_reg: float = 0.0        # Schrödinger entropic regularisation; 0 = exact OT
    sf_q1_mode: str = 'is'             # 'is'      = single-step importance sampling (Path A;
                                       #             degenerates in high-D, see issue #1)
                                       # 'annealed'= K-stage tempered-SMC with RW-MH (Path B)
    sf_annealed_n_stages: int = 3      # K for Path B
    sf_annealed_n_mh_steps: int = 2    # RW-MH moves per Path B stage
    sf_annealed_proposal_scale: float = 0.4  # Roberts-Gelman-Gilks for d~35
    sf_use_q0_cov: bool = False        # Decoupled mode (issue #3 fix 2): mean from
                                       # BW interp, cov from q0 (LW-shrunk). Avoids
                                       # over-inflation when q1 cov is MCMC-noisy.
    sf_info_aware: bool = False        # Information-aware mean update: per-eigenvector
                                       # blend keyed off the local FIM (negative Hessian
                                       # of new_ld at q0 mean). Holds m0 in weakly-
                                       # identified directions, preventing cross-window
                                       # drift. Default off for bit-identical regression
                                       # on fsa_high_res / SWAT; opt in per-driver via
                                       # the model's rolling config.
    sf_info_lambda_thresh_quantile: float = 0.5
                                       # Quantile of FIM eigenvalues used as the soft
                                       # threshold between "well-identified" and
                                       # "weakly identified". 0.5 = median.
    sf_info_blend_temperature: float = 1.0
                                       # τ in sigmoid((log λ - log λ_thresh) / τ).
                                       # Smaller = sharper threshold. 1.0 ≈ half-decade
                                       # soft transition in log-eigenvalue space.

    # HMC kernel
    hmc_step_size: float = 0.025
    hmc_num_leapfrog: int = 8

    # Inner PF
    n_pf_particles: int = 400
    bandwidth_scale: float = 1.0

    # Optimal-transport rescue
    ot_ess_frac: float = 0.05
    ot_temperature: float = 5.0
    ot_max_weight: float = 0.01
    ot_rank: int = 5
    ot_n_iter: int = 2
    ot_epsilon: float = 0.5


@dataclass
class RollingConfig:
    """Rolling-window framing."""

    window_days: int = 120
    stride_days: int = 30
    dt: float = 1.0
    n_substeps: int = 10             # per-day sub-steps for SDE accuracy
    max_windows: int | None = None   # truncate if set


@dataclass
class MissingDataConfig:
    """Example missing-data corruption model.

    Models three gap patterns typical of consumer wearables + endurance-sport
    training logs:
      1. Rest days (weekly): mask active-measurement channels
      2. Random per-channel dropout: mask passive-measurement channels
      3. Continuous broken-watch gap: mask all channels

    This is an opinionated default. Adapt the parameters (or replace the
    function) for other sensor setups.
    """

    dropout_rate: float = 0.15
    broken_watch_days: int = 14
    rest_days_per_week: Tuple[int, int] = (2, 3)  # min, max

    # Channel groupings — depend on the observation model
    active_channels: Tuple[str, ...] = ()   # masked on rest days
    passive_channels: Tuple[str, ...] = ()  # subject to random dropout
    all_obs_channels: Tuple[str, ...] = ()  # masked during broken-watch gap
