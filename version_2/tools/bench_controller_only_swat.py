"""Controller-only SWAT MPC bench — bypasses the SMC² filter.

For *controller debugging* the filter is unnecessary: the controller's
two inputs are `init_state` and `params`, and in a synthetic test we
already know both — truth params live in `simulation.DEFAULT_PARAMS`
and the current state lives in `plant.state` after each `plant.advance`.

This mirrors `bench_smc_full_mpc_swat.py` but skips the entire filter
side. Used to iterate ~3-4× faster on controller-side fixes
(V_c bound, n_substeps, logit bias, sigma_prior, max_lambda_inc, λ_E).

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_controller_only_swat.py 7 \\
        experiments/<run_name> --scenario pathological --step-minutes 15

See claude_plans/controller_only_test_methodology.md for full rationale.
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

import pathlib as _pathlib
_CACHE_DIR = _pathlib.Path.home() / ".jax_compilation_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(_CACHE_DIR))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '1')

import json
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from models.swat.simulation import BINS_PER_DAY


# Replan cadence — match the full bench so the controller test is
# directly comparable. 3-hour stride, 1-day filter window equivalent,
# 6-hour wall-clock replan (every K=2 strides).
STRIDE_HOURS  = 3.0
REPLAN_HOURS  = 6.0
BINS_PER_HOUR = 60 // _STEP_MINUTES
STRIDE_BINS   = int(round(STRIDE_HOURS * BINS_PER_HOUR))
DT            = 1.0 / BINS_PER_DAY


def _replan_K_for_horizon(_T: int) -> int:
    return int(round(REPLAN_HOURS / STRIDE_HOURS))


def _hmc_step_for_horizon(T_total_days: int) -> float:
    if T_total_days >= 70.0:
        return 0.05
    if T_total_days >= 50.0:
        return 0.12
    return 0.20


def _particle_counts_for_horizon(T_total_days: int) -> tuple[int, int]:
    """(ctrl n_smc, ctrl n_inner).

    Memory-load on the controller's MPC rollout scales as
        n_smc · n_inner · T_total_days · BINS_PER_DAY · n_substeps.
    With n_substeps=10 (post-D2):

      T<=2  : 512 / 64  fits.
      T>=7  : 256 / 32  fits  (T=7 ran in ~28.7 GB).
      T>=14 : 128 / 16  fits  (T=14 OOMs at 256/32 — 41 GB working set).
    """
    if T_total_days >= 14:
        return 128, 16
    if T_total_days >= 7:
        return 256, 32
    return 512, 64


def _pop_scenario_from_argv() -> str:
    if '--scenario' in sys.argv:
        i = sys.argv.index('--scenario')
        if i + 1 >= len(sys.argv):
            raise SystemExit("--scenario requires a value")
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        if val not in ('pathological', 'set_A'):
            raise SystemExit(f"--scenario must be 'pathological' or 'set_A'")
        return val
    return 'pathological'


SCENARIO_CONFIGS = {
    'pathological': {
        'init_state':   np.array([0.5, 0.583, 0.5, 0.0], dtype=np.float64),
        'baseline_v_h': 0.0,
        'baseline_v_n': 4.0,
        'baseline_v_c': 12.0,
    },
    'set_A': {
        'init_state':   np.array([0.5, 0.583, 0.5, 0.0], dtype=np.float64),
        'baseline_v_h': 1.0,
        'baseline_v_n': 0.2,
        'baseline_v_c': 0.0,
    },
}


def _build_test_truth_params() -> dict:
    """Return a copy of DEFAULT_PARAMS with the optional tau_T override
    applied (env var SWAT_TAU_T_OVERRIDE_HOURS).

    Use case: at T=7 the truth tau_T=48h means T can't fully climb past
    T_floor before the run ends. Setting SWAT_TAU_T_OVERRIDE_HOURS=12
    quarters tau_T so 7-day controller-only runs show the same
    qualitative climb that the 14-day run does at truth tau_T. The
    override is applied identically to BOTH the plant and the
    controller's cost rollout — same model on both sides — so the test
    stays internally consistent. **Diagnostic only — do not commit
    runs with the override applied as production results.**
    """
    from models.swat.simulation import DEFAULT_PARAMS

    p = dict(DEFAULT_PARAMS)
    tau_T_h = os.environ.get('SWAT_TAU_T_OVERRIDE_HOURS')
    if tau_T_h is not None:
        p['tau_T'] = float(tau_T_h) / 24.0   # convert hours -> days
    return p


def _build_swat_control_spec(*, init_state, plan_horizon_days, lambda_E,
                              params: dict):
    """Build a ControlSpec from TRUTH params + actual current plant state."""
    from models.swat.control import build_control_spec

    n_steps = int(plan_horizon_days * BINS_PER_DAY)
    return build_control_spec(
        n_steps=n_steps,
        dt=DT,
        n_anchors=8,
        n_inner=64,
        n_substeps=10,            # match plant + estimator (D2)
        sigma_prior=1.5,
        lambda_E=lambda_E,
        init_state=np.asarray(init_state, dtype=np.float64),
        params=params,
        seed=42,
    )


def main():
    SCENARIO = _pop_scenario_from_argv()
    T_total_days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    run_name_override = sys.argv[2] if len(sys.argv) > 2 else None
    REPLAN_EVERY_K = _replan_K_for_horizon(T_total_days)

    sc = SCENARIO_CONFIGS[SCENARIO]
    print("=" * 76)
    print(f"  SWAT CONTROLLER-ONLY MPC (T = {T_total_days} d, "
          f"scenario = {SCENARIO})")
    print(f"  No filter; uses truth params + actual plant state.")
    print("=" * 76)

    from models.swat._plant import StepwisePlant
    from models.swat.simulation import DEFAULT_PARAMS
    from smc2fc.control import SMCControlConfig
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native

    n_strides = (T_total_days * BINS_PER_DAY) // STRIDE_BINS
    print(f"  device:    {jax.devices()[0].platform.upper()}")
    print(f"  T_total:   {T_total_days} days")
    print(f"  step:      {_STEP_MINUTES} min ({BINS_PER_DAY} bins/day)")
    print(f"  stride:    {STRIDE_BINS} bins ({STRIDE_HOURS:.0f}h)")
    print(f"  strides:   {n_strides}")
    print(f"  replan:    every K={REPLAN_EVERY_K} strides "
          f"= {REPLAN_EVERY_K * STRIDE_HOURS:.0f}h wall-clock")
    print()

    truth_params_test = _build_test_truth_params()
    tau_T_override_hours = os.environ.get('SWAT_TAU_T_OVERRIDE_HOURS')
    if tau_T_override_hours is not None:
        print(f"  ⚠️  tau_T OVERRIDE: {tau_T_override_hours} h "
              f"(truth = 48 h). Diagnostic only.")

    plant = StepwisePlant(seed_offset=42, state=sc['init_state'].copy(),
                            truth_params=dict(truth_params_test))

    daily_v_h_baseline = sc['baseline_v_h']
    daily_v_n_baseline = sc['baseline_v_n']
    daily_v_c_baseline = sc['baseline_v_c']
    daily_v_h_plan = np.full(T_total_days, daily_v_h_baseline, dtype=np.float64)
    daily_v_n_plan = np.full(T_total_days, daily_v_n_baseline, dtype=np.float64)
    daily_v_c_plan = np.full(T_total_days, daily_v_c_baseline, dtype=np.float64)
    last_replan_stride = 0

    print(f"  init state (W,Z,a,T) = {tuple(sc['init_state'])}")
    print(f"  pre-controller baseline: V_h={daily_v_h_baseline}, "
          f"V_n={daily_v_n_baseline}, V_c={daily_v_c_baseline}h")

    full_traj = []
    daily_v_h_per_stride = []
    daily_v_n_per_stride = []
    daily_v_c_per_stride = []
    replan_history = []

    c_n_smc, c_n_inner = _particle_counts_for_horizon(T_total_days)
    ctrl_cfg = SMCControlConfig(
        n_smc=c_n_smc, n_inner=c_n_inner,
        target_ess_frac=0.5, max_lambda_inc=0.1,
        num_mcmc_steps=5,
        hmc_step_size=_hmc_step_for_horizon(T_total_days),
        hmc_num_leapfrog=10,
    )
    lambda_E = float(os.environ.get('SWAT_LAMBDA_E', '1.0'))

    total_t0 = time.time()

    for s in range(n_strides):
        print(f"  Stride {s+1:>2}/{n_strides}: ", end='', flush=True)

        # Apply current plan
        day_in_plan = (s - last_replan_stride) // 2
        v_h_today = float(daily_v_h_plan[day_in_plan])
        v_n_today = float(daily_v_n_plan[day_in_plan])
        v_c_today = float(daily_v_c_plan[day_in_plan])

        n_days_advance = 2
        v_h_arr = np.full(n_days_advance, v_h_today)
        v_n_arr = np.full(n_days_advance, v_n_today)
        v_c_arr = np.full(n_days_advance, v_c_today)

        obs_stride = plant.advance(STRIDE_BINS, v_h_arr, v_n_arr, v_c_arr)
        full_traj.append(obs_stride['trajectory'].copy())
        daily_v_h_per_stride.append(v_h_today)
        daily_v_n_per_stride.append(v_n_today)
        daily_v_c_per_stride.append(v_c_today)

        # Replan every K strides — using truth params + actual plant state
        if s == 0 or (s + 1) % REPLAN_EVERY_K == 0:
            t_replan = time.time()
            spec = _build_swat_control_spec(
                init_state=plant.state.copy(),
                plan_horizon_days=T_total_days,
                lambda_E=lambda_E,
                params=truth_params_test,
            )
            res_ctrl = run_tempered_smc_loop_native(
                spec=spec, cfg=ctrl_cfg, seed=42 + s,
                print_progress=False,
            )
            schedule = np.asarray(res_ctrl['mean_schedule'])  # (n_steps, 3)
            n_days_in_plan = T_total_days
            if schedule.shape[0] >= n_days_in_plan * BINS_PER_DAY:
                sched_per_day = (schedule[:n_days_in_plan * BINS_PER_DAY]
                                  .reshape(n_days_in_plan, BINS_PER_DAY, 3)
                                  .mean(axis=1))
            else:
                sched_per_day = np.tile(schedule.mean(axis=0),
                                         (n_days_in_plan, 1))
            daily_v_h_plan = sched_per_day[:, 0].astype(np.float64)
            daily_v_n_plan = sched_per_day[:, 1].astype(np.float64)
            daily_v_c_plan = sched_per_day[:, 2].astype(np.float64)
            last_replan_stride = s + 1
            print(f"plan: {res_ctrl['n_temp_levels']}lvl/{time.time()-t_replan:.0f}s, "
                   f"V_h̄={daily_v_h_plan.mean():.2f}  "
                   f"V_n̄={daily_v_n_plan.mean():.2f}  "
                   f"V_c̄={daily_v_c_plan.mean():.2f}", flush=True)
            replan_history.append({
                'stride': int(s),
                'plan_v_h_per_day': daily_v_h_plan.copy(),
                'plan_v_n_per_day': daily_v_n_plan.copy(),
                'plan_v_c_per_day': daily_v_c_plan.copy(),
                'n_temp': int(res_ctrl['n_temp_levels']),
            })
        else:
            print(f"reuse plan day={day_in_plan}, "
                   f"V_h={v_h_today:.2f}/V_n={v_n_today:.2f}/V_c={v_c_today:.2f}",
                   flush=True)

    total_elapsed = time.time() - total_t0

    # Counterfactual baseline (status-quo controls)
    print()
    print(f"  Running counterfactual baseline (constant V_h={daily_v_h_baseline}, "
          f"V_n={daily_v_n_baseline}, V_c={daily_v_c_baseline}) ...")
    traj_full = np.concatenate(full_traj)
    n_total_bins = traj_full.shape[0]
    n_days_baseline = (n_total_bins + BINS_PER_DAY - 1) // BINS_PER_DAY
    plant_b = StepwisePlant(seed_offset=42, state=sc['init_state'].copy(),
                              truth_params=dict(truth_params_test))
    plant_b.advance(n_days_baseline * BINS_PER_DAY,
                     np.full(n_days_baseline, daily_v_h_baseline),
                     np.full(n_days_baseline, daily_v_n_baseline),
                     np.full(n_days_baseline, daily_v_c_baseline))
    traj_baseline = np.concatenate(plant_b.history['trajectory'])[:n_total_bins]

    mean_T_mpc = float(traj_full[:, 3].mean())
    mean_T_baseline = float(traj_baseline[:, 3].mean())
    T_floor_violation = float((traj_full[:, 3] < 0.05).mean())

    print(f"  Total compute: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s)")
    print(f"  mean T (MPC):       {mean_T_mpc:.4f}")
    print(f"  mean T (baseline):  {mean_T_baseline:.4f}")
    print(f"  T-floor violation:  {T_floor_violation:.2%}")

    print(f"\n  Acceptance gates (controller-only — id-cov gate dropped):")
    gates = {
        'mean_T_geq_0.95x_baseline':
            mean_T_mpc >= 0.95 * mean_T_baseline,
        'T_floor_violation_leq_5pct':
            T_floor_violation <= 0.05,
        'compute_leq_4h':
            total_elapsed <= 4 * 3600,
    }
    for gname, passed in gates.items():
        mark = "✓" if passed else "⛔"
        print(f"    {mark}  {gname}: {passed}")
    print(f"  {'✓ all gates pass' if all(gates.values()) else '⛔ gate fail'}")

    # Save outputs
    h_suffix = f"_h{int(_STEP_MINUTES)}min"
    auto_run_name = (f"swat_runs/swat_controller_only_T{T_total_days}d"
                      f"{h_suffix}_{SCENARIO}")
    run_name = run_name_override if run_name_override else auto_run_name
    run_dir = f"outputs/swat/{run_name}"
    os.makedirs(run_dir, exist_ok=True)

    manifest = {
        'schema_version': 1,
        'bench': 'controller_only',
        'T_total_days': T_total_days,
        'step_minutes': _STEP_MINUTES,
        'BINS_PER_DAY': BINS_PER_DAY,
        'STRIDE_BINS': STRIDE_BINS,
        'DT': DT,
        'n_strides': n_strides,
        'replan_K': REPLAN_EVERY_K,
        'n_replans': len(replan_history),
        'init_state': sc['init_state'].tolist(),
        'truth_params': {k: float(v) for k, v in truth_params_test.items()
                          if isinstance(v, (int, float))},
        'tau_T_override_hours': tau_T_override_hours,
        'lambda_E': lambda_E,
        'ctrl_cfg': ctrl_cfg.__dict__ if hasattr(ctrl_cfg, '__dict__') else {},
        'summary': {
            'mean_T_mpc': mean_T_mpc,
            'mean_T_baseline': mean_T_baseline,
            'T_floor_violation_frac_mpc': T_floor_violation,
            'total_compute_s': total_elapsed,
            'gates': gates,
        },
    }
    with open(f"{run_dir}/manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    np.savez(f"{run_dir}/data.npz",
              trajectory_mpc=traj_full,
              trajectory_baseline=traj_baseline,
              daily_v_h_per_stride=np.asarray(daily_v_h_per_stride),
              daily_v_n_per_stride=np.asarray(daily_v_n_per_stride),
              daily_v_c_per_stride=np.asarray(daily_v_c_per_stride))

    # Plot — same 4-panel layout as the full bench so plots are directly
    # comparable.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    t_days = np.arange(n_total_bins) * DT

    ax = axes[0, 0]
    ax.plot(t_days, traj_full[:, 3], 'C2', label='T (MPC)', lw=1.5)
    ax.plot(t_days, traj_baseline[:, 3], '--', color='grey',
             label='T (baseline)', lw=1.0)
    ax.axhline(0.05, color='red', ls=':', label='T_floor=0.05')
    ax.set_xlabel('time (days)'); ax.set_ylabel('T')
    ax.set_title(f'T trajectory (mean MPC: {mean_T_mpc:.3f}, '
                  f'baseline: {mean_T_baseline:.3f})')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_days, traj_full[:, 0], 'C0', label='W', lw=1.0)
    ax.plot(t_days, traj_full[:, 1], 'C1', label='Z', lw=1.0)
    ax.plot(t_days, traj_full[:, 2], 'C3', label='a', lw=1.0)
    ax.set_xlabel('time (days)'); ax.set_title('Other latent states')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    stride_t = np.arange(len(daily_v_h_per_stride)) * (STRIDE_HOURS / 24.0)
    ax.plot(stride_t, daily_v_h_per_stride, 'o-', color='C0', label='V_h applied')
    ax.plot(stride_t, daily_v_n_per_stride, 'o-', color='C3', label='V_n applied')
    ax.set_xlabel('time (days)'); ax.set_ylabel('V_h / V_n')
    ax.set_title('Applied V_h / V_n schedule')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(stride_t, daily_v_c_per_stride, 'o-', color='C2', label='V_c applied')
    ax.axhline(0, color='grey', ls='--', lw=0.5)
    ax.axhline(3, color='red', ls=':', label='V_c_max=3h')
    ax.axhline(-3, color='red', ls=':')
    ax.set_xlabel('time (days)'); ax.set_ylabel('V_c (hours)')
    ax.set_title('Applied V_c phase-shift schedule')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f'SWAT controller-only MPC, T={T_total_days}d  '
                  f'mean T={mean_T_mpc:.3f} vs baseline {mean_T_baseline:.3f}, '
                  f'T-floor viol={T_floor_violation:.1%}, '
                  f'{int(total_elapsed//60)} min')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = f"{run_dir}/controller_only_T{T_total_days}d_traces.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    print(f"\n  Checkpoint: {run_dir}/manifest.json + data.npz")
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
