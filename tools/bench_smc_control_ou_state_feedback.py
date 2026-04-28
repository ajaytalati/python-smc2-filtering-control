"""Stage A3: SMC²-as-controller with state feedback on scalar OU LQG.

Refactored to use smc2fc.control + models.scalar_ou_lqg.control.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import numpy as np
import matplotlib.pyplot as plt

from smc2fc.control import (
    SMCControlConfig, run_tempered_smc_loop, evaluate_gates,
    plot_cost_histogram,
)
from models.scalar_ou_lqg.control import SCALAR_OU_STATE_FEEDBACK_SPEC


def main():
    spec = SCALAR_OU_STATE_FEEDBACK_SPEC
    print("=" * 72)
    print(f"  Stage A3 — state-feedback SMC²-as-controller on {spec.name}")
    print("=" * 72)
    refs = spec._refs
    print(f"  analytical LQR (perfect state):    {refs['lqr_perfect']:.3f}")
    print(f"  MC LQG (Kalman + LQR):             {refs['lqg_mc']:.3f}")
    print(f"  MC open-loop (u=0):                {refs['open_loop_mc']:.3f}")
    print(f"  Riccati gains: {[f'{x:.2f}' for x in refs['riccati_gains'][:5]]} ... "
          f"{[f'{x:.2f}' for x in refs['riccati_gains'][-3:]]}")
    print()

    cfg = SMCControlConfig(
        n_smc=128, n_inner=64, sigma_prior=3.0,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=5,
        hmc_step_size=0.05, hmc_num_leapfrog=8,
        beta_max_target_nats=8.0,
    )
    result = run_tempered_smc_loop(spec=spec, cfg=cfg, seed=42)
    print()
    print(f"  done: {result['n_temp_levels']} levels in {result['elapsed_s']:.1f}s")
    print()

    evaluate_gates(spec=spec, result=result, print_table=True)

    # Diagnostic plot
    K_mean = result['mean_theta']
    K_std = result['particles'].std(axis=0)
    smc_cost = float(spec._cost_eval(K_mean))
    riccati_cost = float(spec._cost_eval(refs['riccati_gains']))

    out_path = "outputs/scalar_ou_lqg/A3_state_feedback_diagnostic.png"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    t = np.arange(spec.n_steps) * spec.dt
    axes[0].fill_between(t, K_mean - K_std, K_mean + K_std,
                            alpha=0.3, color='steelblue', label='SMC ±1σ')
    axes[0].plot(t, K_mean, 'o-', color='steelblue',
                   label='SMC² posterior mean K_k')
    axes[0].plot(t, refs['riccati_gains'], 's--', color='darkred',
                   label='LQR Riccati K_k*')
    axes[0].set_xlabel('time (s)')
    axes[0].set_ylabel('feedback gain K_k')
    axes[0].set_title('State-feedback gain: SMC² posterior vs Riccati')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    plot_cost_histogram(
        particle_costs=result['particle_costs'],
        references={
            'LQR perfect': refs['lqr_perfect'],
            'MC LQG':      refs['lqg_mc'],
            'open-loop':   refs['open_loop_mc'],
            'SMC² mean K': smc_cost,
            'Riccati K':   riccati_cost,
        },
        title='SMC² state-feedback cost vs analytical references',
        ax=axes[1],
    )

    import os as _os
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 72)


if __name__ == '__main__':
    main()
