"""Stage B2: SMC²-as-controller on the bistable_controlled model.

Refactored to use smc2fc.control + models.bistable_controlled.control.
The model-specific bits (cost functional, schedule, gates) live in
models/bistable_controlled/control.py; this file is pure orchestration.

Run:
    PYTHONPATH=. python tools/bench_smc_control_bistable.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

from smc2fc.control import (
    SMCControlConfig, run_tempered_smc_loop, evaluate_gates,
)
from models.bistable_controlled.control import BISTABLE_CONTROL_SPEC


def main():
    print("=" * 72)
    print(f"  Stage B2 — SMC²-as-controller on {BISTABLE_CONTROL_SPEC.name}")
    print("=" * 72)
    print(f"  truth params: {BISTABLE_CONTROL_SPEC.truth_params}")
    print(f"  initial state: x={float(BISTABLE_CONTROL_SPEC.initial_state[0]):.2f}, "
          f"u={float(BISTABLE_CONTROL_SPEC.initial_state[1]):.2f}")
    print(f"  theta_dim:    {BISTABLE_CONTROL_SPEC.theta_dim} RBF anchors")
    print(f"  horizon:      {BISTABLE_CONTROL_SPEC.n_steps} steps × "
          f"{BISTABLE_CONTROL_SPEC.dt:.4f} h")
    print(f"  default cost: {BISTABLE_CONTROL_SPEC._default_cost:.2f} "
          f"(hand-coded 24h+u_on schedule)")
    print()

    cfg = SMCControlConfig(
        n_smc=128, n_inner=32, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        beta_max_target_nats=8.0,
    )
    result = run_tempered_smc_loop(
        spec=BISTABLE_CONTROL_SPEC, cfg=cfg, seed=42,
    )
    print()
    print(f"  done: {result['n_temp_levels']} levels in {result['elapsed_s']:.1f}s")
    print()

    # Evaluate acceptance gates
    evaluate_gates(spec=BISTABLE_CONTROL_SPEC, result=result, print_table=True)

    # Diagnostic plot
    out_path = "outputs/bistable_controlled/B2_control_diagnostic.png"
    BISTABLE_CONTROL_SPEC._diagnostic_plot(
        result, out_path,
        default_schedule=BISTABLE_CONTROL_SPEC._default_schedule,
        traj_sample_fn=BISTABLE_CONTROL_SPEC._traj_sample_fn,
    )
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
