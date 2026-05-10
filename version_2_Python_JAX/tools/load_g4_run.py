"""Loader for Stage G4 closed-loop MPC run checkpoints.

Each `bench_smc_full_mpc_fsa.py` invocation writes:

    outputs/fsa_high_res/g4_runs/{run_name}/
    ├── manifest.json   (settings + truth params + summary metrics)
    ├── data.npz        (bundled arrays — see bench source for keys)
    └── E5_full_mpc_T{T}d_traces.png   (diagnostic plot)

This module provides a thin loader so post-hoc analysis scripts can
load any run and produce arbitrary plots / metrics without re-running
the bench (which costs 1–15 hours per horizon).

Usage:
    from tools.load_g4_run import load_g4_run, list_g4_runs

    # Discover runs
    for path in list_g4_runs():
        print(path)

    # Load one
    run = load_g4_run("outputs/fsa_high_res/g4_runs/T14d_replanK2_no_infoaware")
    print(run['manifest']['summary']['mean_A_mpc'])
    traj = run['data']['trajectory_mpc']            # (n_bins, 3)
    phi = run['data']['daily_phi_per_stride']        # (n_strides,)
    posteriors = run['data']['posterior_particles']  # (n_strides, n_smc, n_params)

The full list of array keys in data.npz is documented at the bottom of
this file (and in the bench source).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

import numpy as np


def load_g4_run(run_dir: Union[str, Path]) -> dict:
    """Load a single G4 run checkpoint.

    Returns a dict with three top-level keys:

      - 'manifest'  : the parsed JSON config + summary
      - 'data'      : numpy NpzFile (lazy-loaded; access keys directly)
      - 'param_names': list of 30 parameter names in the order used by
                       posterior_particles' axis-2 indexing
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    data_path = run_dir / "data.npz"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json in {run_dir}")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.npz in {run_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # NB: np.load returns a lazy NpzFile by default; downstream callers
    # should use `with` or close() if they care about file handles.
    data = np.load(data_path, allow_pickle=False)

    return {
        'manifest':    manifest,
        'data':        data,
        'param_names': manifest.get('param_names', []),
        'run_dir':     run_dir,
    }


def list_g4_runs(root: Union[str, Path] = "outputs/fsa_high_res/g4_runs") -> List[Path]:
    """List all G4 run directories under `root`. Returns sorted paths.

    A directory counts as a run iff it contains both `manifest.json`
    and `data.npz`.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    runs = sorted(p for p in root.iterdir()
                   if p.is_dir()
                   and (p / "manifest.json").exists()
                   and (p / "data.npz").exists())
    return runs


def summarize_run(run: dict) -> str:
    """One-paragraph summary of a loaded run, for quick diagnostics."""
    m = run['manifest']
    s = m['summary']
    cfg = m['smc_cfg']
    return (
        f"T={m['T_total_days']}d, replan K={m['replan_K']}, "
        f"bridge={cfg['bridge_type']}/{cfg.get('sf_q1_mode', '-')} "
        f"info_aware={cfg.get('sf_info_aware', False)}\n"
        f"  mean A: {s['mean_A_mpc']:.4f} (MPC) vs {s['mean_A_baseline']:.4f} (baseline) "
        f"ratio={s['mean_A_mpc']/s['mean_A_baseline']:.3f}\n"
        f"  F-viol: {s['F_violation_frac_mpc']:.2%}, "
        f"id-cov: {s['n_windows_pass_id_cov_5_of_6']}/{m['n_strides']}, "
        f"compute: {s['total_compute_s']/60:.0f} min"
    )


# =========================================================================
# data.npz key reference
# =========================================================================
#
# Plant trajectory (B, F, A latent state at every 15-min bin):
#   trajectory_mpc                (n_bins, 3)   float32   — under MPC-applied Φ
#   trajectory_baseline           (n_bins, 3)   float32   — under constant Φ=1.0 (counterfactual)
#   Phi_per_bin                   (n_bins,)     float32   — the per-bin Φ applied to the plant
#                                                            (post-burst expansion of daily values)
#   C_per_bin                     (n_bins,)     float32   — circadian C(t)
#
# 4-channel observations from the MPC plant:
#   obs_HR_t_idx                  (n_HR,)       int32     — global bin indices, sleep-gated
#   obs_HR_value                  (n_HR,)       float32
#   obs_sleep_label               (n_bins,)     int32     — Bernoulli sleep label every bin
#   obs_stress_t_idx / value      (...)                    — wake-gated
#   obs_steps_t_idx  / value      (...)                    — wake-gated
#
# Per-stride applied Φ + per-replan controller plans:
#   daily_phi_per_stride          (n_strides,)  float32   — what was actually applied each stride
#   daily_phi_plan_per_replan     (n_replans,
#                                  T_total_days) float32  — the FULL T_total-day plan at each replan
#   replan_strides                (n_replans,)  int32     — stride index at which each replan happened
#   replan_n_temp                 (n_replans,)  int32     — controller tempering levels per replan
#
# Per-window filter posterior:
#   posterior_particles           (n_strides,
#                                  n_smc,
#                                  n_params)    float32   — full constrained particle cloud
#   posterior_window_mask         (n_strides,)  bool      — True where filter actually ran
#                                                            (False on warm-up stride 1)
#   n_temp_per_window             (n_strides,)  int32     — filter tempering levels
#   elapsed_per_window_s          (n_strides,)  float32   — wall-clock per filter window
#   id_cov_per_window             (n_strides,)  int32     — coverage on identifiable subset (0..6)
#
# n_params = 30 (see manifest['param_names'] for the order)
