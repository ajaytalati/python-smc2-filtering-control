"""ControlSpec — the model-side contract for the SMC² control engine.

Mirrors `smc2fc.estimation_model.EstimationModel` for the filter side.
Each `models/<model>/control.py` builds and exports a `ControlSpec`
instance that the generic engine in `smc2fc.control` consumes.

The model-specific bits (cost coefficients, schedule semantics,
initial state, truth params, acceptance gates) live in the
`ControlSpec`; the generic tempered-SMC machinery lives in the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import jax.numpy as jnp


@dataclass(frozen=True)
class ControlSpec:
    """Complete specification for an SMC²-as-controller task.

    The single object passed to ``run_tempered_smc_loop``.

    Attributes:
        name: Human-readable model name.
        version: Model version string.

        dt: Outer time step (in the model's natural units).
        n_steps: Number of outer steps in the planning horizon.
        n_substeps: Number of deterministic-Euler sub-steps per outer
            step (for stiff dynamics; 1 = no substepping).

        initial_state: Starting state vector (n_states,) for the
            cost-evaluator forward simulator.
        truth_params: Dict of parameters fed to the dynamics. The
            control side uses fixed truth (not posterior); for closed-
            loop / MPC variants, truth_params is replaced at runtime
            with a filter-derived posterior mean.

        theta_dim: Dimension of the SMC² search space (the schedule's
            parameter vector).
        sigma_prior: Std of the Gaussian prior over θ (broad).

        cost_fn: JIT-compiled callable θ → scalar mean cost. Captures
            the schedule basis, dynamics, CRN noise grids, and cost
            coefficients internally. Built by the model's
            ``build_control_spec()`` helper.

        schedule_from_theta: Callable θ → schedule grid for diagnostic
            plotting. Same closure as in cost_fn but exposes the raw
            schedule (without rolling out the SDE).

        acceptance_gates: Dict {gate_name: callable(result_dict) →
            (passed: bool, value: float, message: str)}. Each gate is
            evaluated after the SMC² run on a result dict containing
            the final particle cloud, costs, sampled trajectories,
            and any model-specific summary stats.

        diagnostic_plot_fn: Optional callable for the model's headline
            diagnostic plot. Signature: (result_dict, out_path) → None.
    """

    name: str
    version: str

    # Time grid
    dt: float
    n_steps: int
    n_substeps: int = 1

    # Initial conditions + truth
    initial_state: jnp.ndarray = field(default=None)
    truth_params: Dict[str, float] = field(default_factory=dict)

    # Search space
    theta_dim: int = 0
    sigma_prior: float = 1.5

    # Core API
    cost_fn: Optional[Callable] = None
    schedule_from_theta: Optional[Callable] = None

    # Diagnostics
    acceptance_gates: Dict[str, Callable] = field(default_factory=dict)
    diagnostic_plot_fn: Optional[Callable] = None
