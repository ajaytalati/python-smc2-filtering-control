"""Interface contract for coupled SDE model definitions.

Frozen dataclasses that define what a model must provide. No solver
logic, no numpy — pure data-structure definitions consumed by
:mod:`smc2fc.simulator.sde_solver_diffrax` and
:mod:`smc2fc.simulator.sde_observations`.

Already-functional in the imperative source; this module is
unchanged in semantics, with docstrings upgraded to Google style.

Diffusion type contract:

    * ``DIFFUSION_DIAGONAL_CONSTANT``: per-step noise increment is
      ``sigma_i * sqrt(dt) * xi_i``, where ``sigma_i`` comes from
      ``diffusion_fn(params)``.
    * ``DIFFUSION_DIAGONAL_STATE``: per-step noise increment is
      ``sigma_i * g_i(y, params) * sqrt(dt) * xi_i``, with ``g_i``
      from ``noise_scale_fn(y, params)``. Typical choices: ``sqrt(y *
      (1 - y))`` for Jacobi on ``[0, 1]``; ``sqrt(y)`` for CIR on
      ``[0, ∞)``.
    * ``DIFFUSION_MATRIX``: full-matrix noise (declared but not yet
      supported by the diffrax solver).
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Diffusion-type tag constants ───────────────────────────────────────
DIFFUSION_DIAGONAL_CONSTANT = "diagonal_constant"
DIFFUSION_DIAGONAL_STATE = "diagonal_state"
DIFFUSION_MATRIX = "matrix"


@dataclass(frozen=True)
class StateSpec:
    """Specification of one state variable in the SDE system.

    Attributes:
        name: Human-readable state name.
        lower_bound: Hard lower bound (used for clipping).
        upper_bound: Hard upper bound.
        is_deterministic: If True, this state has no diffusion and is
            overwritten with ``analytical_fn`` at every step.
        analytical_fn: Numpy-side analytical evaluator
            ``(t, params) -> float``. Required if ``is_deterministic``
            and the deterministic solver is used.
        analytical_fn_jax: JAX-compatible analytical evaluator
            ``(t, params_jax) -> jnp scalar``. Must use ``jnp.*`` ops
            (no ``math.sin``, no ``float()`` casts) because it is
            called inside a JIT-compiled scan.
    """
    name: str
    lower_bound: float
    upper_bound: float
    is_deterministic: bool = False
    analytical_fn: Optional[Callable] = None
    analytical_fn_jax: Optional[Callable] = None


@dataclass(frozen=True)
class ChannelSpec:
    """Specification of one observation channel.

    Attributes:
        name: Human-readable channel name; appears as a key in the
            output dict of :func:`generate_all_channels`.
        depends_on: Tuple of names of channels that must be generated
            first (their outputs are passed via ``prior_channels``).
        generate_fn: Pure function with the signature
            ``(trajectory, t_grid, params, aux, prior_channels, seed)
            -> dict``. Returns a dict containing whatever channel-
            specific arrays the consumer needs (typically at least
            ``t_idx`` or ``t_hours``).
    """
    name: str
    depends_on: Tuple[str, ...] = ()
    generate_fn: Optional[Callable] = None


@dataclass(frozen=True)
class SDEModel:
    """Complete specification of a coupled nonlinear SDE system.

    The single object passed to all generic framework functions.
    Every model-specific detail is encapsulated here.

    Attributes:
        name: Human-readable model name.
        version: Model version string (free-form).
        states: Tuple of :class:`StateSpec`, in canonical state order.
        drift_fn: Numpy-side drift,
            ``(t, y, params_dict, aux) -> ndarray(n_states,)``.
        diffusion_type: One of the ``DIFFUSION_*`` constants.
        diffusion_fn: ``params -> ndarray(n_states,)`` — the
            family-independent ``sigma_i`` vector. See module
            docstring for the per-type contract.
        noise_scale_fn: ``(y, params) -> ndarray(n_states,)`` — the
            state-dependent ``g_i`` multiplier. Required when
            ``diffusion_type == DIFFUSION_DIAGONAL_STATE``.
        noise_scale_fn_jax: JAX-compatible variant of
            ``noise_scale_fn`` for the diffrax solver.
        drift_fn_jax: JAX-compatible drift,
            ``(t, y, jax_args) -> jnp.array(n_states,)``.
        make_aux_fn_jax: Builds the JAX-side ``args_jax`` consumed by
            ``drift_fn_jax``.
        make_aux_fn: Numpy-side analogue of ``make_aux_fn_jax``.
        make_y0_fn: ``(init_state_dict, params_dict) -> ndarray
            (n_states,)`` — assembles the initial-state vector.
        channels: Tuple of :class:`ChannelSpec` declaring the model's
            observation channels.
        plot_fn: Optional model-specific plotter.
        csv_writer_fn: Optional model-specific CSV exporter.
        param_sets: Optional named parameter scenarios.
        init_states: Optional named initial-state scenarios.
        exogenous_inputs: Optional named exogenous-input scenarios.
        verify_physics_fn: Optional physics-invariant checker
            ``(traj, t_grid, params) -> dict[str, bool]``.
    """
    # Metadata
    name: str
    version: str

    # State space
    states: Tuple[StateSpec, ...]

    # Dynamics
    drift_fn: Callable

    diffusion_type: str = DIFFUSION_DIAGONAL_CONSTANT
    diffusion_fn: Optional[Callable] = None

    noise_scale_fn: Optional[Callable] = None
    noise_scale_fn_jax: Optional[Callable] = None

    # JAX variants
    drift_fn_jax: Optional[Callable] = None
    make_aux_fn_jax: Optional[Callable] = None

    # Auxiliary builders
    make_aux_fn: Optional[Callable] = None
    make_y0_fn: Optional[Callable] = None

    # Observation channels
    channels: Tuple[ChannelSpec, ...] = ()

    # Optional model-specific helpers
    plot_fn: Optional[Callable] = None
    csv_writer_fn: Optional[Callable] = None
    param_sets: Optional[Dict[str, dict]] = None
    init_states: Optional[Dict[str, dict]] = None
    exogenous_inputs: Optional[Dict[str, dict]] = None
    verify_physics_fn: Optional[Callable] = None

    @property
    def n_states(self) -> int:
        """Number of state dimensions (length of ``self.states``)."""
        return len(self.states)

    @property
    def state_names(self) -> List[str]:
        """List of state names in canonical order."""
        return [s.name for s in self.states]

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """List of ``(lower_bound, upper_bound)`` pairs per state."""
        return [(s.lower_bound, s.upper_bound) for s in self.states]

    @property
    def deterministic_indices(self) -> List[int]:
        """Indices of states whose ``is_deterministic`` is True."""
        return [i for i, s in enumerate(self.states) if s.is_deterministic]

    @property
    def stochastic_indices(self) -> List[int]:
        """Indices of states whose ``is_deterministic`` is False."""
        return [i for i, s in enumerate(self.states) if not s.is_deterministic]
