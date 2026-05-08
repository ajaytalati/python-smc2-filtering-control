"""
estimation_model.py — Interface Contract for Bayesian SDE Estimation
=====================================================================
Date:    15 April 2026
Version: 1.0

Frozen dataclass that defines what a model must provide for the
generic differentiable particle filter + MCLMC sampler.

Mirrors sde_model.SDEModel for the simulator side.
"""

from dataclasses import dataclass
from collections import OrderedDict
from typing import Callable, Optional, Dict, List, Tuple


@dataclass(frozen=True)
class EstimationModel:
    """Complete specification for Bayesian SDE estimation.

    The SINGLE OBJECT passed to all generic estimation functions.
    Every model-specific detail is encapsulated here.

    Attributes:
        name: Human-readable model name.
        version: Model version string.
        n_states: Total latent states (including deterministic).
        n_stochastic: Number of stochastic states (particle dimension).
        stochastic_indices: Tuple of indices for stochastic states.
        state_bounds: Tuple of (lo, hi) per stochastic state.
        param_prior_config: OrderedDict of {name: (type, args)} for
            estimated parameters. Types: 'lognormal', 'normal',
            'vonmises', 'beta'.
        init_state_prior_config: OrderedDict of {name: (type, args)}
            for initial state priors.
        frozen_params: Dict of {name: value} for non-estimated params.
        propagate_fn: JAX function called inside the PF scan.
            Signature: (y, t, dt, params, grid_obs, step_k,
                        sigma_diag, noise, rng_key) -> (x_new, pred_lw)
        diffusion_fn: JAX function returning noise diagonal.
            Signature: (params) -> array(n_states,)
        obs_log_weight_fn: JAX function for observation log-weight.
            Signature: (x_new, grid_obs, step_k, params) -> scalar
        align_obs_fn: Numpy function to grid-align observations.
            Signature: (obs_data, t_steps, dt_hours) -> dict_of_arrays
        shard_init_fn: JAX function for phase-conditioned init.
            Signature: (time_offset, params, exogenous, global_init)
                        -> init_states_array
        load_data_fn: I/O function to load raw observations.
            Signature: () -> obs_data (model-specific format)
        plot_trajectory_fn: I/O function for MAP trajectory plot.
            Signature: (t_hours, trajectory, obs_data, params_dict,
                        save_path) -> None
        plot_residuals_fn: I/O function for residual plots.
            Signature: (t_hours, trajectory, obs_data, params_dict,
                        save_path) -> None
        forward_sde_fn: JAX function for deterministic MAP trajectory.
            Signature: (init_state, params, exogenous, dt, n_steps)
                        -> trajectory array(n_steps, n_states)
        get_init_theta_fn: Function returning initial theta vector.
            Signature: () -> ndarray(n_dim,)
        exogenous_keys: Tuple of field names in grid_obs that are
            NOT time-indexed (broadcast to all shards unchanged).
    """

    # ── Metadata ──
    name: str
    version: str

    # ── State space ──
    n_states: int
    n_stochastic: int
    stochastic_indices: Tuple[int, ...]
    state_bounds: Tuple[Tuple[float, float], ...]

    # ── Parameters ──
    param_prior_config: OrderedDict
    init_state_prior_config: OrderedDict
    frozen_params: Dict[str, float]

    # ── Dynamics (JAX, inside jax.lax.scan) ──
    propagate_fn: Callable
    diffusion_fn: Callable

    # ── Observation model (JAX, vmapped over particles) ──
    obs_log_weight_fn: Callable

    # ── Grid alignment (numpy, called once at setup) ──
    align_obs_fn: Callable

    # ── Shard initialisation (JAX) ──
    shard_init_fn: Callable

    # ── Model-specific I/O (optional) ──
    load_data_fn: Optional[Callable] = None
    plot_trajectory_fn: Optional[Callable] = None
    plot_residuals_fn: Optional[Callable] = None
    forward_sde_fn: Optional[Callable] = None
    get_init_theta_fn: Optional[Callable] = None

    # ── Direct-scan log-density (NEW in v6.0) ──
    imex_step_fn: Optional[Callable] = None
    # (y, t, dt, params, grid_obs) -> y_next
    # Deterministic IMEX step (no noise, no particles)

    obs_log_prob_fn: Optional[Callable] = None
    # (y, grid_obs, k, params) -> scalar log-probability
    # Total observation log-prob at step k given state y

    make_init_state_fn: Optional[Callable] = None
    # (init_estimates, params) -> y0 array(n_states,)
    # Build full initial state vector from estimated init values

    # ── Synthetic data generation (NEW v6.3: simulator integration) ──
    obs_sample_fn: Optional[Callable] = None
    # (y, exog, k, params, rng_key) -> dict[str, jnp.ndarray]
    # Sample one observation from each channel at step k given state y.
    # MUST be the sampling counterpart of obs_log_prob_fn — i.e. the
    # densities they imply must be identical.

    # ── EKF-required functions (NEW v6.4: Kalman-filter likelihood) ──
    gaussian_obs_fn: Optional[Callable] = None
    # (y, grid_obs, k, params) -> dict with keys
    #   'mean'      shape (n_g,)  — predicted obs mean
    #   'value'     shape (n_g,)  — observed value
    #   'cov_diag'  shape (n_g,)  — diagonal of obs noise covariance
    #   'present'   shape (n_g,)  — per-channel 0/1 presence mask
    # Defines the Gaussian observation channels used by the EKF.
    # Required for likelihood_method='ekf' or 'ekf_hybrid'.

    init_cov_fn: Optional[Callable] = None
    # (params, init_estimates) -> P0 array(n_states, n_states)
    # Initial state covariance for the EKF.  If absent the EKF uses
    # diag(sigma_diag^2 * dt) as a sensible default.

    # ── Marginal-SGR kernel densities (NEW v6.4) ──
    dynamic_kernel_log_density_fn: Optional[Callable] = None
    # (x_new, x_prev, t, dt, params, grid_obs, sigma_diag) -> scalar
    # Evaluates log p(x_new | x_prev) under the bootstrap dynamic
    # kernel.  Required for sgr_marginal_pf.

    proposal_log_density_fn: Optional[Callable] = None
    # (x_new, x_prev, t, dt, params, grid_obs, k, sigma_diag) -> scalar
    # Evaluates log π(x_new | x_prev, y_k) under the proposal.
    # If None, marginal-SGR assumes bootstrap (π = p) and uses
    # dynamic_kernel_log_density_fn for both numerator and denominator.

    # ── Grid obs structure ──
    exogenous_keys: Tuple[str, ...] = ()

    # ── Derived properties ──
    @property
    def n_params(self) -> int:
        """Number of estimated parameters."""
        return len(self.param_prior_config)

    @property
    def n_init_states(self) -> int:
        """Number of estimated initial states."""
        return len(self.init_state_prior_config)

    @property
    def n_dim(self) -> int:
        """Total number of estimated dimensions."""
        return self.n_params + self.n_init_states

    @property
    def all_names(self) -> List[str]:
        """All estimated parameter + init state names."""
        return (list(self.param_prior_config.keys()) +
                list(self.init_state_prior_config.keys()))

    @property
    def param_keys(self) -> List[str]:
        """Estimated parameter names only."""
        return list(self.param_prior_config.keys())

    @property
    def param_idx(self) -> Dict[str, int]:
        """Parameter name -> index mapping."""
        return {k: i for i, k in enumerate(self.param_prior_config.keys())}
