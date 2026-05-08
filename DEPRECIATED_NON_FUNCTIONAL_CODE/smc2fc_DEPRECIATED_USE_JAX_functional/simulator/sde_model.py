"""
sde_model.py — Interface Contract for Coupled SDE Model Definitions
=====================================================================
Date:    19 April 2026
Version: 1.1

Frozen dataclasses that define what a model must provide.
No solver logic, no numpy — pure data structure definitions.

Changelog from 1.0:
  - Added optional noise_scale_fn / noise_scale_fn_jax fields to SDEModel
    so models declaring DIFFUSION_DIAGONAL_STATE can supply the state-
    dependent multipliers g_i(y, params).  The factorisation is
        sigma_i(x) = sigma_i · g_i(x)
    with sigma_i from diffusion_fn (unchanged contract) and g_i from
    noise_scale_fn (new).  DIFFUSION_DIAGONAL_CONSTANT ignores these
    new fields and behaves identically to 1.0.
"""

from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Tuple, Dict, List

# Diffusion type constants
DIFFUSION_DIAGONAL_CONSTANT = "diagonal_constant"
DIFFUSION_DIAGONAL_STATE = "diagonal_state"
DIFFUSION_MATRIX = "matrix"


@dataclass(frozen=True)
class StateSpec:
    """Specification of one state variable in the SDE system."""
    name: str
    lower_bound: float
    upper_bound: float
    is_deterministic: bool = False
    analytical_fn: Optional[Callable] = None
    # If is_deterministic=True, analytical_fn(t, params) -> float
    # gives the exact value.  Must use numpy/math (for scipy solver).
    analytical_fn_jax: Optional[Callable] = None
    # JAX-compatible version: analytical_fn_jax(t, params_jax) -> jnp scalar
    # Must use jnp.* operations (no math.sin, no float() casts) because
    # it is called inside a JIT-compiled scan.  Required if the Diffrax
    # solver is to handle this deterministic state.


@dataclass(frozen=True)
class ChannelSpec:
    """Specification of one observation channel."""
    name: str
    depends_on: Tuple[str, ...] = ()
    generate_fn: Optional[Callable] = None
    # Signature: generate_fn(trajectory, t_grid, params, aux,
    #                        prior_channels, seed) -> dict
    # prior_channels: dict of already-generated channel outputs.
    # Returns: dict with channel-specific keys (at minimum 't_idx' or 't_hours').


@dataclass(frozen=True)
class SDEModel:
    """Complete specification of a coupled nonlinear SDE system.

    This is the SINGLE OBJECT passed to all generic framework functions.
    Every model-specific detail is encapsulated here.
    """
    # ── Metadata ──
    name: str
    version: str

    # ── State space ──
    states: Tuple[StateSpec, ...]

    # ── Dynamics (pure functions) ──
    drift_fn: Callable
    # (t, y, params_dict, aux) -> ndarray(n_states,)

    diffusion_type: str = DIFFUSION_DIAGONAL_CONSTANT
    diffusion_fn: Optional[Callable] = None
    # Signature depends on diffusion_type (see module docstring).

    noise_scale_fn: Optional[Callable] = None
    # State-dependent diagonal noise multiplier g_i(y, params).
    # Signature: noise_scale_fn(y, params) -> ndarray(n_states,)
    # REQUIRED when diffusion_type == DIFFUSION_DIAGONAL_STATE.
    # IGNORED when diffusion_type == DIFFUSION_DIAGONAL_CONSTANT.
    # The per-step noise increment is
    #     sigma_i * g_i(y, params) * sqrt(dt) * xi_i,   xi_i ~ N(0, 1)
    # where sigma_i comes from diffusion_fn(params)[i] and g_i from
    # noise_scale_fn(y, params)[i].  Typical choices: sqrt(y_i*(1-y_i))
    # for Jacobi on [0, 1], sqrt(y_i) for CIR on [0, inf), sqrt(y_i+eps)
    # for regularised Landau at a reflecting boundary.

    noise_scale_fn_jax: Optional[Callable] = None
    # JAX-compatible variant of noise_scale_fn for the Diffrax solver.
    # Must use jnp.* ops (no numpy) because it is called inside a JIT-
    # compiled scan.  Takes the same arguments but `params` will be the
    # jnp-dict used by drift_fn_jax.

    # ── JAX variants (for Diffrax solver, optional) ──
    drift_fn_jax: Optional[Callable] = None
    # (t, y, jax_args) -> jnp.array(n_states,)
    make_aux_fn_jax: Optional[Callable] = None

    # ── Auxiliary data builder ──
    make_aux_fn: Optional[Callable] = None
    # (params, init_state, t_grid, exogenous) -> aux (opaque)

    # ── Initial state builder ──
    make_y0_fn: Optional[Callable] = None
    # (init_state_dict, params_dict) -> ndarray(n_states,)

    # ── Observation channels ──
    channels: Tuple[ChannelSpec, ...] = ()

    # ── Model-specific plotting ──
    plot_fn: Optional[Callable] = None
    # (trajectory, t_grid, channel_outputs, params, save_dir) -> None

    # ── Model-specific CSV export (for end-to-end pipeline testing) ──
    csv_writer_fn: Optional[Callable] = None
    # (trajectory, t_grid, channel_outputs, params, save_dir) -> dict

    # ── Test configurations ──
    param_sets: Optional[Dict[str, dict]] = None
    init_states: Optional[Dict[str, dict]] = None
    exogenous_inputs: Optional[Dict[str, dict]] = None

    # ── Physics verification ──
    verify_physics_fn: Optional[Callable] = None
    # (trajectory, t_grid, params) -> dict of {check_name: bool}

    # ── Derived properties ──
    @property
    def n_states(self) -> int:
        return len(self.states)

    @property
    def state_names(self) -> List[str]:
        return [s.name for s in self.states]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        return [(s.lower_bound, s.upper_bound) for s in self.states]

    @property
    def deterministic_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.states) if s.is_deterministic]

    @property
    def stochastic_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.states) if not s.is_deterministic]