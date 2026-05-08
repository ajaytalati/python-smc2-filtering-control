"""SMCControlConfig — knobs for the tempered-SMC outer loop on the
control side.

Mirrors smc2fc.core.config.SMCConfig but tuned for the control task
(typically smaller particle counts because cost surfaces are smoother,
and a target-nats knob for β_max calibration).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SMCControlConfig:
    """Tempered-SMC knobs for control. Defaults validated on A2/A3/B2/B3."""

    # Outer SMC
    n_smc: int = 128
    target_ess_frac: float = 0.5
    max_lambda_inc: float = 0.10
    max_temp_steps: int = 200

    # MC integration of the cost functional
    n_inner: int = 32

    # Schedule prior
    sigma_prior: float = 1.5

    # β_max auto-calibration
    beta_max_target_nats: float = 8.0
    n_calibration_samples: int = 256

    # HMC kernel
    num_mcmc_steps: int = 5
    hmc_step_size: float = 0.05
    hmc_num_leapfrog: int = 8

    # Logging
    log_every_n_steps: int = 5
