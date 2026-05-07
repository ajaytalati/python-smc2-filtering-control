"""Diagnostic for the persistent ``weighted_violation_rate ~= 0.96``
the prior agent observed across all soft / soft_fast Stage-2 / Stage-3
runs. Read-only tool — does not run the controller, does not need GPU.

# What the diagnostic does

Given a completed bench run dir with ``trajectory.npz``, this script
computes — under the truth params (``TRUTH_PARAMS_V5``) — three
breakdowns of where the chance-constraint indicator
``(A_traj < A_sep_per_bin)`` is firing:

1. **Per-bin evaluator** (matches what the bench reports). For each of
   the ~1344 bins (T=14d × 96 bins/day) computes A_sep at the
   instantaneous Phi and the indicator. Reports the fraction of bins
   in each regime (mono-stable healthy / collapsed / bistable).

2. **Per-day evaluator** (alternative formulation). Averages Phi over
   each day (96 bins) and computes A_sep at the daily-mean Phi; same
   for A. Reports per-day regime classification and violation rate.

3. **Active-bin only** (a third option). Restricts the indicator to
   bins where ``Phi_B + Phi_S > threshold`` (default 0.05) — i.e.
   ignores rest periods.

# Why this matters

The bursty Phi-burst expansion (`_phi_burst.py`) means most per-bin
Phi values are 0.0 for both channels. At ``Phi=(0,0)``, the v5 closed
island is mono-stable collapsed: ``A_sep = +inf``, so the indicator
fires for every rest bin regardless of what A actually is. This
mechanically produces ~96% violation when the per-bin formulation is
used on a burst schedule. Run-16 (overtrained, full HMC) was confirmed
to land at exactly 0.9643 = 1296/1344 by this mechanism.

The per-day reformulation gives a much more informative reading on the
same data — for run-16 it drops to 0.214 (3/14 days collapsed), which
matches the visual evidence that the controller is trying (and mostly
succeeding) to keep the daily-mean schedule near the (0.30, 0.30)
healthy island.

This script does NOT change the production cost function — that's a
formulation decision for Ajay to make. It only surfaces the alternative
readings on existing data.

# Usage

    cd version_3
    PYTHONPATH=.:.. python tools/diagnose_violation_rate.py \\
        outputs/fsa_v5/experiments/old_experiments/run16_stage2_soft_fast_overtrained_T14_full_hmc

If no path is given, runs against the last numbered run dir under
``outputs/fsa_v5/experiments/``.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import json
import sys
from pathlib import Path

import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5
from version_3.models.fsa_v5.control_v5 import _jax_find_A_sep

BINS_PER_DAY = 96


def _classify(asep_array: np.ndarray) -> dict:
    return {
        'healthy_mono':     int(np.sum(np.isneginf(asep_array))),
        'bistable_finite':  int(np.sum(np.isfinite(asep_array))),
        'collapsed_mono':   int(np.sum(np.isposinf(asep_array))),
        'total':            len(asep_array),
    }


def _fmt(d: dict) -> str:
    n = d['total']
    return (f"healthy={d['healthy_mono']}/{n} ({100*d['healthy_mono']/n:.1f}%), "
            f"bistable={d['bistable_finite']}/{n} ({100*d['bistable_finite']/n:.1f}%), "
            f"collapsed={d['collapsed_mono']}/{n} ({100*d['collapsed_mono']/n:.1f}%)")


def _find_latest_run_dir() -> Path:
    exp_dir = Path("outputs/fsa_v5/experiments")
    if not exp_dir.exists():
        raise SystemExit(f"No experiments dir at {exp_dir} — run from version_3/")
    runs = sorted(exp_dir.iterdir(), key=lambda p: p.name)
    runs = [p for p in runs if p.is_dir() and p.name.startswith('run')
            and (p / 'trajectory.npz').exists()]
    if not runs:
        raise SystemExit(f"No completed run dirs under {exp_dir}")
    return runs[-1]


def diagnose(run_dir: Path):
    print(f"=== {run_dir.name} ===")
    manifest_path = run_dir / 'manifest.json'
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        sc = m.get('scenario', {})
        s = m.get('summary', {})
        print(f"  cost: {m.get('cost_variant')}, scenario: {sc.get('name')}, "
              f"baseline_phi: {sc.get('baseline_phi')}, T_days: {m.get('T_total_days')}")
        if 'weighted_violation_rate' in s:
            print(f"  reported weighted_violation_rate: {s['weighted_violation_rate']:.4f}")

    z = np.load(run_dir / 'trajectory.npz', allow_pickle=False)
    if 'full_phi' not in z or 'trajectory' not in z:
        raise SystemExit(f"Run dir {run_dir} lacks full_phi / trajectory keys "
                          f"in trajectory.npz; can't diagnose.")
    fphi = np.asarray(z['full_phi'], dtype=np.float64)        # (n_bins, 2)
    traj = np.asarray(z['trajectory'], dtype=np.float64)      # (n_bins, 6)
    A_traj = traj[:, 3]
    n_bins = fphi.shape[0]
    n_days = n_bins // BINS_PER_DAY
    print(f"  n_bins={n_bins}, n_days={n_days}")
    print(f"  per-bin Phi: PhiB median={np.median(fphi[:,0]):.3f} mean={fphi[:,0].mean():.3f} max={fphi[:,0].max():.3f}")
    print(f"               PhiS median={np.median(fphi[:,1]):.3f} mean={fphi[:,1].mean():.3f} max={fphi[:,1].max():.3f}")
    print(f"  A_traj: min={A_traj.min():.3f} max={A_traj.max():.3f} mean={A_traj.mean():.3f}")
    print()

    p = {k: jnp.asarray(float(v), dtype=jnp.float64)
         for k, v in TRUTH_PARAMS_V5.items()}

    def asep_at(pb_arr, ps_arr):
        return np.asarray(jax.vmap(
            lambda pb, ps: _jax_find_A_sep(pb, ps, p)
        )(jnp.asarray(pb_arr, dtype=jnp.float64),
          jnp.asarray(ps_arr, dtype=jnp.float64)))

    # 1. Per-bin (the production formulation)
    asep_bin = asep_at(fphi[:, 0], fphi[:, 1])
    cl = _classify(asep_bin)
    indicator_bin = (A_traj < asep_bin).astype(np.float64)
    print(f"[1] PER-BIN (production formulation)")
    print(f"    A_sep regime distribution: {_fmt(cl)}")
    print(f"    weighted_violation_rate (uniform per-bin weights): "
          f"{indicator_bin.mean():.4f}")
    fires = indicator_bin > 0.5
    if fires.any():
        n_collapsed_fires = int(np.sum(np.isposinf(asep_bin[fires])))
        n_bistable_fires = int(np.sum(np.isfinite(asep_bin[fires])))
        print(f"    Of {int(fires.sum())} firing bins: "
              f"{n_collapsed_fires} are A_sep=+inf (collapsed regime — vacuous), "
              f"{n_bistable_fires} are A_sep finite (real bistable violations)")
    print()

    # 2. Per-day (alternative formulation)
    days = n_bins // BINS_PER_DAY
    if days > 0:
        phi_d = fphi[:days * BINS_PER_DAY].reshape(days, BINS_PER_DAY, 2).mean(axis=1)
        A_d   = A_traj[:days * BINS_PER_DAY].reshape(days, BINS_PER_DAY).mean(axis=1)
        asep_d = asep_at(phi_d[:, 0], phi_d[:, 1])
        cl_d = _classify(asep_d)
        indicator_d = (A_d < asep_d).astype(np.float64)
        print(f"[2] PER-DAY (alternative: A_sep on daily-mean Phi)")
        print(f"    A_sep regime distribution: {_fmt(cl_d)}")
        print(f"    daily violation rate: {indicator_d.mean():.4f}  "
              f"({int(indicator_d.sum())}/{days} days)")
        print(f"    Per-day breakdown:")
        for i in range(days):
            phib, phis = phi_d[i]
            asep = asep_d[i]
            asep_str = ('-inf  HEALTHY' if np.isneginf(asep)
                        else '+inf  COLLAPSED' if np.isposinf(asep)
                        else f'{asep:6.3f} bistable')
            viol = 'VIOL' if indicator_d[i] > 0.5 else '    '
            print(f"      Day {i+1:2d}: PhiB={phib:.3f} PhiS={phis:.3f}  "
                  f"A_d={A_d[i]:.3f}  A_sep={asep_str}  {viol}")
        print()

    # 3. Active bins only (Phi above threshold)
    threshold = 0.05
    active = (fphi[:, 0] + fphi[:, 1]) > threshold
    if active.any():
        asep_act = asep_bin[active]
        A_act = A_traj[active]
        cl_a = _classify(asep_act)
        ind_a = (A_act < asep_act).astype(np.float64)
        print(f"[3] ACTIVE BINS ONLY  (sum(Phi) > {threshold}, "
              f"{int(active.sum())}/{n_bins} bins)")
        print(f"    A_sep regime distribution: {_fmt(cl_a)}")
        print(f"    active-bin violation rate: {ind_a.mean():.4f}")
        print()


def main(argv):
    if len(argv) > 1:
        run_dir = Path(argv[1])
        if not run_dir.exists():
            raise SystemExit(f"Run dir not found: {run_dir}")
    else:
        run_dir = _find_latest_run_dir()
        print(f"(No path arg; using latest: {run_dir})")
    diagnose(run_dir)


if __name__ == '__main__':
    main(sys.argv)
