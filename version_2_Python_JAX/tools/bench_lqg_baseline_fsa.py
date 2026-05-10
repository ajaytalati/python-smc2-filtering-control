"""Stage H: LQG/Riccati exact-approximate baseline for FSA-v2 closed-loop MPC.

Builds an LQG controller via JAX-autodiff linearisation of the FSA-v2
drift around a deterministic operating point, solves the differential
Riccati equation backward over the planning horizon T_total_days, and
applies the resulting open-loop Phi schedule to the StepwisePlant.

Comparison value:
  - Cheap baseline against which the SMC^2 closed-loop MPC of
    bench_smc_full_mpc_fsa.py can be benchmarked. Does the
    nonlinear posterior-mean controller actually outperform the linear
    LQG approximation, and by how much?
  - Tests the Stuart-Landau nonlinearity hypothesis:
    LQG linearises away the cubic -eta A^3 term, so its absolute
    performance vs MPC scales with how much the cubic matters at
    long horizons (T=84 the linear approximation is far from the
    actual limit-cycle trajectory; T=14 it should be close).

Output: one folder per run under outputs/fsa_high_res/lqg_runs/
matching the G4 schema (manifest.json + data.npz) so the same
load_g4_run.py loader can read both for plotting.

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_lqg_baseline_fsa.py 14
    cd version_2 && PYTHONPATH=.:.. python tools/bench_lqg_baseline_fsa.py 84 --step-minutes 60
"""

from __future__ import annotations

import os
import sys


def _pop_step_minutes_from_argv() -> int:
    if '--step-minutes' in sys.argv:
        i = sys.argv.index('--step-minutes')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--step-minutes requires a value")
        val = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        return val
    return 15


_STEP_MINUTES = _pop_step_minutes_from_argv()
os.environ['FSA_STEP_MINUTES'] = str(_STEP_MINUTES)
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import json
import time

import numpy as np

from models.fsa_high_res._plant import StepwisePlant
from models.fsa_high_res.simulation import (
    DEFAULT_PARAMS, DEFAULT_INIT, BINS_PER_DAY,
)
from models.fsa_high_res._dynamics import (
    drift_jax as drift_jax_v2, A_TYP, F_TYP,
)

from smc2fc.control.lqg import build_lqg_open_loop_schedule


# Cost weights — see Section 8.8 (LaTex_docs).
#
# The default (w_A = w_F = 100, R = 1) was selected by a systematic
# weight sweep at T=42 (see PROGRESS_HI.md): heavier w_A relative to
# w_F drives bang-bang Phi schedules and 30%+ F-violation; lighter
# w_A under-uses the controller's leverage. Equal w_A == w_F gives
# the least pathological schedules (no F-violation at T<=56, modest
# A gain over baseline).
#
# x_ref components: F_ref = F_typ keeps the symmetric F-penalty
# centered at the natural operating-point F level, so deviations
# above (overtraining) and below (undertraining) are penalised
# equally.  A_ref = 0.30 is well above A_typ = 0.10, giving the
# Riccati a clear gradient to climb.
DEFAULT_W_A   = 100.0
DEFAULT_W_F   = 100.0
DEFAULT_W_PHI = 1.0

DEFAULT_A_REF = 0.30
DEFAULT_F_REF = 0.20    # = F_TYP


def main():
    # CLI:
    #   sys.argv[1]: T_total_days (default 14)
    T_total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 14

    print("=" * 76)
    print(f"  Stage H — LQG baseline for FSA-v2 (T = {T_total_days} d, "
          f"step = {_STEP_MINUTES} min)")
    print("=" * 76)

    n_steps = T_total_days * BINS_PER_DAY
    dt = 1.0 / BINS_PER_DAY

    print(f"  T_total:    {T_total_days} d")
    print(f"  step:       {_STEP_MINUTES} min ({BINS_PER_DAY} bins/day)")
    print(f"  n_steps:    {n_steps}")
    print(f"  dt (days):  {dt:.6f}")

    # Operating point (matches Section 8.8 sketch + G1 reparam constants)
    x_star = np.array([DEFAULT_INIT['B_0'], F_TYP, A_TYP], dtype=np.float64)
    phi_star = 1.0
    x_ref = np.array([DEFAULT_INIT['B_0'], DEFAULT_F_REF, DEFAULT_A_REF],
                     dtype=np.float64)
    phi_ref = phi_star

    Q   = np.diag([0.0, DEFAULT_W_F, DEFAULT_W_A])
    R   = np.array([[DEFAULT_W_PHI]])
    Q_T = np.zeros((3, 3))

    print(f"  x_star:     {x_star.tolist()}")
    print(f"  x_ref:      {x_ref.tolist()}")
    print(f"  Q diag:     {np.diag(Q).tolist()}")
    print(f"  R:          {R.flatten().tolist()}")
    print(f"  Q_T:        zeros")
    print()

    # ── Build LQG controller and the open-loop schedule ──
    print("  [1/3] Building LQG controller (linearise + Riccati) ...")
    t0 = time.time()
    phi_per_bin, ctrl = build_lqg_open_loop_schedule(
        drift_jax=drift_jax_v2, params=dict(DEFAULT_PARAMS),
        x_star=x_star, phi_star=phi_star,
        x_ref=x_ref, phi_ref=phi_ref,
        Q=Q, R=R, Q_T=Q_T,
        dt=dt, n_steps=n_steps,
        phi_min=0.0, phi_max=3.0,
    )
    elapsed_lqg_build = time.time() - t0
    print(f"      Riccati solved in {elapsed_lqg_build*1000:.1f} ms")
    print(f"      A_lin diag: {np.diag(ctrl.A_lin).round(4).tolist()}")
    print(f"      B_lin: {ctrl.B_lin.flatten().round(4).tolist()}")
    print(f"      Phi(t): mean={phi_per_bin.mean():.3f}, "
          f"min={phi_per_bin.min():.3f}, max={phi_per_bin.max():.3f}")
    print()

    # Aggregate to per-day for plotting / comparison with MPC daily plans
    daily_phi = phi_per_bin.reshape(T_total_days, BINS_PER_DAY).mean(axis=1)
    print(f"      daily Phi: day0={daily_phi[0]:.3f}, "
          f"day_mid={daily_phi[T_total_days // 2]:.3f}, "
          f"day_last={daily_phi[-1]:.3f}, "
          f"mean={daily_phi.mean():.3f}")

    # ── Apply to plant ──
    print("  [2/3] Running plant under LQG open-loop Phi ...")
    t0 = time.time()
    plant = StepwisePlant(seed_offset=42)
    out = plant.advance(n_steps, daily_phi)
    elapsed_plant = time.time() - t0
    traj = out['trajectory']    # (n_steps, 3)
    print(f"      plant integrated in {elapsed_plant:.1f} s")

    # ── Counterfactual: constant Phi=1.0 baseline (same plant seed) ──
    print("  [3/3] Counterfactual constant Phi=1.0 baseline ...")
    plant_b = StepwisePlant(seed_offset=42)
    plant_b.advance(n_steps, np.full(T_total_days, 1.0))
    traj_baseline = np.concatenate(plant_b.history['trajectory'])

    mean_A_lqg      = float(traj[:, 2].mean())
    mean_A_baseline = float(traj_baseline[:, 2].mean())
    F_viol_lqg      = float((traj[:, 1] > 0.40).mean())

    total_elapsed = elapsed_lqg_build + elapsed_plant
    print()
    print(f"  Total compute: {total_elapsed:.2f} s "
          f"(Riccati + plant; vs SMC^2 MPC ~hours)")
    print(f"  mean A (LQG open-loop):    {mean_A_lqg:.4f}")
    print(f"  mean A (constant Phi=1.0): {mean_A_baseline:.4f}")
    print(f"  ratio LQG/baseline:        {mean_A_lqg / mean_A_baseline:.3f}")
    print(f"  F-violation (LQG):         {F_viol_lqg:.2%}")

    # ── Checkpoint, mirror G4 schema ──
    h_suffix = '' if _STEP_MINUTES == 15 else f"_h{_STEP_MINUTES}min"
    run_name = f"T{T_total_days}d{h_suffix}_lqg"
    run_dir = f"outputs/fsa_high_res/lqg_runs/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        'schema_version': '1.0',
        'kind': 'lqg_baseline',
        'T_total_days': int(T_total_days),
        'step_minutes': int(_STEP_MINUTES),
        'BINS_PER_DAY': int(BINS_PER_DAY),
        'n_steps': int(n_steps),
        'dt_days': float(dt),
        'cost_weights': {
            'w_A': float(DEFAULT_W_A),
            'w_F': float(DEFAULT_W_F),
            'w_Phi': float(DEFAULT_W_PHI),
        },
        'x_star': x_star.tolist(),
        'x_ref':  x_ref.tolist(),
        'phi_star': float(phi_star),
        'phi_ref':  float(phi_ref),
        'truth_params': {k: float(v) for k, v in DEFAULT_PARAMS.items()},
        'A_lin_diag': [float(v) for v in np.diag(ctrl.A_lin)],
        'B_lin':      [float(v) for v in ctrl.B_lin.flatten()],
        'summary': {
            'mean_A_lqg': mean_A_lqg,
            'mean_A_baseline': mean_A_baseline,
            'ratio_lqg_baseline': mean_A_lqg / mean_A_baseline,
            'F_violation_frac_lqg': F_viol_lqg,
            'lqg_phi_mean': float(daily_phi.mean()),
            'lqg_phi_min':  float(daily_phi.min()),
            'lqg_phi_max':  float(daily_phi.max()),
            'compute_lqg_build_ms': elapsed_lqg_build * 1000,
            'compute_plant_s':      elapsed_plant,
            'total_compute_s':      total_elapsed,
        },
    }
    with open(f"{run_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    np.savez_compressed(
        f"{run_dir}/data.npz",
        trajectory_lqg=traj.astype(np.float32),
        trajectory_baseline=traj_baseline.astype(np.float32),
        Phi_per_bin=phi_per_bin.astype(np.float32),
        daily_phi=daily_phi.astype(np.float32),
        K_traj=ctrl.K_traj.astype(np.float32),
        P_traj=ctrl.P_traj.astype(np.float32),
        A_lin=ctrl.A_lin.astype(np.float64),
        B_lin=ctrl.B_lin.astype(np.float64),
        x_star=x_star.astype(np.float64),
        x_ref=x_ref.astype(np.float64),
    )
    print(f"  Checkpoint: {run_dir}/manifest.json + data.npz")
    print()


if __name__ == '__main__':
    main()
