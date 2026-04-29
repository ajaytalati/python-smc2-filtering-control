"""Stage D: SMC²-as-controller on the FSA-v2 model (Banister-coupled).

Drives a 3-state physiological SDE from a low-fitness / high-fatigue /
low-amplitude state toward sustained high amplitude by scheduling a
single training-strain rate Φ(t). Cost rewards cumulative ∫A(t)dt
(sustained adaptation, not peaking) with control-effort + overtraining
penalties.

Run:
    PYTHONPATH=. python tools/bench_smc_control_fsa.py [T_total_days]

T_total_days defaults to 42 (canonical Banister chronic time constant).
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from smc2fc.control import (
    SMCControlConfig, run_tempered_smc_loop, evaluate_gates,
)
from models.fsa_high_res.control import build_control_spec


def main():
    # Parse horizon (default canonical Banister τ_B = 42 d)
    T_total_days = float(sys.argv[1]) if len(sys.argv) > 1 else 42.0

    print("=" * 76)
    print(f"  Stage D — SMC²-as-controller on FSA-v2  (T = {T_total_days:.0f} days)")
    print("=" * 76)

    print(f"  Building FSA-v2 control spec (Banister-coupled, single Φ control;")
    print(f"  includes baseline + sedentary references — ~30s)...")
    spec = build_control_spec(T_total_days=T_total_days)
    print()
    print(f"  device:       {jax.devices()[0].platform.upper()}")
    print(f"  initial state: B={float(spec.initial_state[0]):.2f}, "
          f"F={float(spec.initial_state[1]):.2f}, "
          f"A={float(spec.initial_state[2]):.2f}")
    print(f"  horizon:      {spec.n_steps} outer × {spec.n_substeps} sub-steps "
          f"= {spec.n_steps * spec.dt:.1f} days")
    print(f"  theta_dim:    {spec.theta_dim} (Φ-only control, "
          f"{spec._n_anchors} RBF anchors)")
    print(f"  baseline mean ∫A/T: {spec._refs['baseline_mean_A']:.3f}  "
          f"(constant Φ={spec._refs['baseline_Phi']})")
    print(f"  sedentary mean ∫A/T: {spec._refs['sedentary_mean_A']:.3f}  "
          f"(constant Φ=0)")
    print()

    # HMC step size scales inversely with horizon: per-step cost
    # gradient grows with T (~ T · ∂_θ mean_A), so a fixed step that
    # mixes well at T=42 (acc ~0.95) collapses at T=84 (acc → 0). The
    # scaling below empirically gives acc > 0.4 across the sweep.
    if T_total_days >= 70.0:
        hmc_step_size = 0.05
    elif T_total_days >= 50.0:
        hmc_step_size = 0.12
    else:
        hmc_step_size = 0.30

    cfg = SMCControlConfig(
        n_smc=256, n_inner=32,
        sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=15,
        hmc_step_size=hmc_step_size, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0,
        max_temp_steps=100,
    )
    t0 = time.time()
    result = run_tempered_smc_loop(spec=spec, cfg=cfg, seed=42)
    elapsed_total = time.time() - t0
    print()
    print(f"  done: {result['n_temp_levels']} levels in "
          f"{result['elapsed_s']:.1f}s SMC + setup")
    print()

    # Acceptance gates
    evaluate_gates(spec=spec, result=result, print_table=True)

    # ── Diagnostic plot ──
    T_tag = f"T{int(T_total_days)}"
    out_path = f"outputs/fsa_high_res/D_v2_{T_tag}_diagnostic.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    Phi_schedule = np.asarray(result['mean_schedule'])    # (n_steps,)
    t_grid = np.arange(spec.n_steps) * spec.dt    # in days

    # Top-left: Φ(t) schedule (the only control surface in v2)
    ax = axes[0, 0]
    ax.plot(t_grid, Phi_schedule, '-', color='darkorange', lw=2,
              label='SMC² Φ(t)')
    ax.axhline(spec._refs['baseline_Phi'], color='gray', linestyle=':',
                 alpha=0.7,
                 label=f'baseline Φ = {spec._refs["baseline_Phi"]}')
    ax.axhline(0.0, color='red', linestyle=':', alpha=0.4,
                 label='sedentary Φ = 0')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('Φ (training strain rate)')
    ax.set_title('SMC²-derived Φ(t) schedule')
    ax.set_ylim(-0.1, max(spec._Phi_max, float(Phi_schedule.max())) + 0.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: cost histogram
    ax = axes[0, 1]
    ax.hist(np.asarray(result['particle_costs']), bins=30,
              color='steelblue', alpha=0.7,
              label='SMC² per-particle cost')
    ax.set_xlabel('cost')
    ax.set_ylabel('density')
    ax.set_title('SMC² per-particle cost distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sample trajectories under SMC² schedule
    rng_key = jax.random.PRNGKey(7)
    n_traj = 5
    trajs = []
    for i in range(n_traj):
        rng_key, sub = jax.random.split(rng_key)
        traj = spec._traj_sample_fn(jnp.asarray(result['mean_theta']), sub)
        trajs.append(np.asarray(traj))
    trajs = np.stack(trajs)    # (n_traj, n_steps, 3)

    # Middle row: B(t) and F(t)
    ax = axes[1, 0]
    for i in range(n_traj):
        ax.plot(t_grid, trajs[i, :, 0], alpha=0.4, lw=0.7, color='steelblue')
    ax.plot(t_grid, trajs[:, :, 0].mean(axis=0), '-', color='steelblue',
              lw=2, label='mean B(t)')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('B (fitness)')
    ax.set_title('B trajectory (Banister chronic) under SMC² schedule')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for i in range(n_traj):
        ax.plot(t_grid, trajs[i, :, 1], alpha=0.4, lw=0.7, color='darkred')
    ax.plot(t_grid, trajs[:, :, 1].mean(axis=0), '-', color='darkred',
              lw=2, label='mean F(t)')
    ax.axhline(spec._F_max, color='red', linestyle='--', alpha=0.5,
                 label=f'F_max = {spec._F_max}')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('F (strain / fatigue)')
    ax.set_title('F trajectory (Banister acute) under SMC² schedule')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom row: A(t) — the headline; aggregated comparison
    ax = axes[2, 0]
    for i in range(n_traj):
        ax.plot(t_grid, trajs[i, :, 2], alpha=0.4, lw=0.7, color='green')
    ax.plot(t_grid, trajs[:, :, 2].mean(axis=0), '-', color='green',
              lw=2, label='mean A(t)')
    ax.axhline(spec._refs['baseline_mean_A'], color='gray', linestyle=':',
                 alpha=0.7,
                 label=f'baseline mean A = {spec._refs["baseline_mean_A"]:.3f}')
    ax.axhline(spec._refs['sedentary_mean_A'], color='red', linestyle=':',
                 alpha=0.5,
                 label=f'sedentary mean A = {spec._refs["sedentary_mean_A"]:.3f}')
    ax.set_xlabel('time (days)')
    ax.set_ylabel('A (amplitude)')
    ax.set_title('A trajectory under SMC² schedule (headline)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Comparison: A under SMC² vs baseline vs sedentary
    ax = axes[2, 1]
    ax.bar(['sedentary\n(Φ=0)',
            f'baseline\n(Φ={spec._refs["baseline_Phi"]})',
            'SMC²\n(time-varying)'],
            [spec._refs['sedentary_mean_A'],
             spec._refs['baseline_mean_A'],
             float(jnp.mean(jnp.asarray(trajs[:, :, 2])))],
            color=['salmon', 'gray', 'steelblue'])
    ax.set_ylabel('mean ∫A / T  (time-averaged amplitude)')
    ax.set_title('Mean amplitude comparison')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Stage D — FSA-v2 (Banister) control:  T = {T_total_days:.0f} d, '
                  f'{result["n_temp_levels"]} tempering levels in '
                  f'{result["elapsed_s"]:.0f}s on {jax.devices()[0].platform.upper()}',
                  fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
