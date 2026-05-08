"""Frozen config dataclasses for SMC² + rolling-window estimation.

Replaces the tangle of module-level globals in the original monolithic
driver with explicit, plain-Python config objects that pass through
the API. Every config in this module is ``@dataclass(frozen=True)`` —
fields are set at construction and never mutated.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SMCConfig:
    """Tuning knobs for the outer tempered-SMC over parameters + inner PF.

    Built once per bench, threaded through every filter call. Frozen
    so that no helper can accidentally mutate it after construction.

    Attributes:
        n_smc_particles: Outer SMC² particle count (number of HMC
            chains tracking the parameter posterior).
        target_ess_frac: ESS / N threshold for adaptive tempering;
            below this, λ is reduced.
        num_mcmc_steps: Cold-start HMC moves per tempering level.
        max_lambda_inc: Cold-start hard cap on λ-increment per
            tempering bisection.
        num_mcmc_steps_bridge: HMC moves per tempering level on the
            bridge (warm-start) path.
        max_lambda_inc_bridge: Bridge λ-increment cap.
        bridge_type: One of ``'gaussian'`` (single-Gaussian fit + LW
            shrinkage), ``'mog'`` (mixture of Gaussians), or
            ``'schrodinger_follmer'`` (Bures-Wasserstein geodesic
            between the previous-window posterior and a moment-matched
            new-window estimate; see :mod:`smc2fc.core.sf_bridge`).
        bridge_mog_components: Number of MoG components when
            ``bridge_type == 'mog'``.
        sf_blend: SF bridge interpolation parameter ``t in [0, 1]``.
            0 = previous posterior, 1 = new-posterior moment-match,
            0.5 = midpoint.
        sf_entropy_reg: Schrödinger entropic regularisation strength;
            0 = exact OT.
        sf_q1_mode: How to estimate the new-window posterior moments
            for the SF bridge: ``'is'`` (single-step importance
            sampling — degenerates in high-D) or ``'annealed'``
            (K-stage tempered-SMC with random-walk MH).
        sf_annealed_n_stages: K for the annealed Path-B mode.
        sf_annealed_n_mh_steps: RW-MH moves per Path-B stage.
        sf_annealed_proposal_scale: RW-MH proposal scale; the
            Roberts-Gelman-Gilks 2.38/√d scale becomes 0.4 for d~35.
        sf_use_q0_cov: If True, take the bridge mean from the BW
            interpolation but the covariance from the previous
            posterior's LW-shrunk fit. Avoids over-inflation when the
            new-window covariance estimate is MCMC-noisy.
        sf_info_aware: If True, blend per-eigenvector keyed off the
            local FIM (negative Hessian of the new log-density at the
            previous posterior mean). Holds the mean in weakly-
            identified directions across windows.
        sf_info_lambda_thresh_quantile: Quantile of FIM eigenvalues
            used as the soft threshold between well-identified and
            weakly-identified directions. 0.5 = median.
        sf_info_blend_temperature: Sigmoid temperature ``τ`` in
            ``sigmoid((log λ − log λ_thresh) / τ)``; smaller = sharper
            threshold.
        hmc_step_size: HMC leapfrog step size.
        hmc_num_leapfrog: HMC trajectory length (leapfrog steps).
        n_pf_particles: Inner particle-filter chain length.
        bandwidth_scale: Liu-West shrinkage bandwidth scale (1.0 =
            standard Silverman).
        ot_ess_frac: ESS/K value at which the OT-rescue interpolation
            weight is half-saturated.
        ot_temperature: Sigmoid sharpness for the OT-rescue
            interpolation.
        ot_max_weight: Maximum OT-rescue interpolation weight; 0
            disables OT entirely.
        ot_rank: Nyström anchor count for the low-rank Sinkhorn.
        ot_n_iter: Sinkhorn iterations per OT call.
        ot_epsilon: Sinkhorn entropic regularisation.
    """

    # Outer SMC
    n_smc_particles: int = 256
    target_ess_frac: float = 0.5
    num_mcmc_steps: int = 5
    max_lambda_inc: float = 0.05

    # Bridge (warm-start)
    num_mcmc_steps_bridge: int = 3
    max_lambda_inc_bridge: float = 0.10
    bridge_type: str = 'gaussian'
    bridge_mog_components: int = 2
    sf_blend: float = 0.5
    sf_entropy_reg: float = 0.0
    sf_q1_mode: str = 'is'
    sf_annealed_n_stages: int = 3
    sf_annealed_n_mh_steps: int = 2
    sf_annealed_proposal_scale: float = 0.4
    sf_use_q0_cov: bool = False
    sf_info_aware: bool = False
    sf_info_lambda_thresh_quantile: float = 0.5
    sf_info_blend_temperature: float = 1.0

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


@dataclass(frozen=True)
class RollingConfig:
    """Rolling-window framing parameters.

    Attributes:
        window_days: Filter window length in days.
        stride_days: Stride between consecutive windows in days.
        dt: Time-grid resolution in days (e.g. 1.0 = daily,
            1/24 = hourly).
        n_substeps: SDE sub-steps per grid bin.
        max_windows: Optional cap on the number of rolling windows
            (None = unlimited).
    """

    window_days: int = 120
    stride_days: int = 30
    dt: float = 1.0
    n_substeps: int = 10
    max_windows: Optional[int] = None


@dataclass(frozen=True)
class MissingDataConfig:
    """Synthetic missing-data corruption model.

    Models three gap patterns typical of consumer wearables +
    endurance-sport training logs:

        1. Rest days (weekly): mask the active-measurement channels.
        2. Random per-channel dropout: mask passive-measurement
           channels independently.
        3. Continuous broken-watch gap: mask all channels for a
           contiguous block.

    This is an opinionated default — adapt or replace it for other
    sensor setups.

    Attributes:
        dropout_rate: Per-bin dropout probability for passive
            channels.
        broken_watch_days: Length of the contiguous all-channel gap.
        rest_days_per_week: ``(min, max)`` rest-day count per week
            (uniformly sampled).
        active_channels: Channels masked on rest days.
        passive_channels: Channels subject to random dropout.
        all_obs_channels: Channels masked during the broken-watch
            gap.
    """

    dropout_rate: float = 0.15
    broken_watch_days: int = 14
    rest_days_per_week: Tuple[int, int] = (2, 3)
    active_channels: Tuple[str, ...] = ()
    passive_channels: Tuple[str, ...] = ()
    all_obs_channels: Tuple[str, ...] = ()
