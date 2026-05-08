"""smc2fc.control — generic SMC²-as-controller engine.

Mirrors smc2fc.core (filter side). Per-model control specs live in
``models/<model>/control.py`` and supply a ControlSpec instance that
this engine consumes.

Public API:

    ControlSpec               — model-side contract (~ EstimationModel)
    SMCControlConfig          — outer-loop knobs
    run_tempered_smc_loop     — the headline driver
    calibrate_beta_max        — auto-set β_max from prior cost spread
    build_crn_noise_grids     — common-random-numbers helper
    RBFSchedule               — Gaussian-RBF schedule basis
    plot_cost_histogram       — diagnostic
    plot_schedule_comparison  — diagnostic
    plot_trajectories         — diagnostic
    print_smc_step            — tempering-progress log line
    evaluate_gates            — run the spec's acceptance_gates dict
"""

from smc2fc.control.config import SMCControlConfig
from smc2fc.control.control_spec import ControlSpec
from smc2fc.control.calibration import (
    calibrate_beta_max,
    build_crn_noise_grids,
)
from smc2fc.control.rbf_schedules import RBFSchedule
from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop
from smc2fc.control.diagnostics import (
    plot_cost_histogram,
    plot_schedule_comparison,
    plot_trajectories,
    print_smc_step,
    evaluate_gates,
)

__all__ = [
    'ControlSpec',
    'SMCControlConfig',
    'run_tempered_smc_loop',
    'calibrate_beta_max', 'build_crn_noise_grids',
    'RBFSchedule',
    'plot_cost_histogram', 'plot_schedule_comparison',
    'plot_trajectories', 'print_smc_step',
    'evaluate_gates',
]
