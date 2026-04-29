"""Comparison helper for G4 (SMC^2 MPC) and LQG checkpoints.

Loads checkpoints from `outputs/fsa_high_res/g4_runs/` and
`outputs/fsa_high_res/lqg_runs/` and produces:

  1. Summary table CSV: T_total, mean_A_mpc, mean_A_lqg, mean_A_baseline,
     ratios, F-violations, wall-clock. Suitable for the LaTeX results
     section.

  2. Trajectory triptych at one chosen horizon: A(t), F(t), Phi(t)
     overlay of constant-Phi baseline, LQG open-loop, MPC closed-loop.

  3. Mean-A vs T plot — three curves, one per controller.

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/compare_g4_lqg.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[1].parent
G4_ROOT  = REPO / "version_2/outputs/fsa_high_res/g4_runs"
LQG_ROOT = REPO / "version_2/outputs/fsa_high_res/lqg_runs"
OUT_DIR  = REPO / "version_2/outputs/fsa_high_res"


def _load(run_dir: Path) -> dict:
    with open(run_dir / "manifest.json") as f:
        manifest = json.load(f)
    data = np.load(run_dir / "data.npz", allow_pickle=False)
    return {'manifest': manifest, 'data': data, 'run_dir': run_dir}


def _list_runs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "manifest.json").exists()
                  and (p / "data.npz").exists())


def _build_table() -> list[dict]:
    """Cross-join G4 and LQG runs by (T_total, step_minutes) and emit rows."""
    g4_runs = {}
    for p in _list_runs(G4_ROOT):
        m = json.load(open(p / "manifest.json"))
        key = (int(m['T_total_days']), int(m.get('step_minutes', 15)))
        g4_runs[key] = p
    lqg_runs = {}
    for p in _list_runs(LQG_ROOT):
        m = json.load(open(p / "manifest.json"))
        key = (int(m['T_total_days']), int(m.get('step_minutes', 15)))
        lqg_runs[key] = p

    keys = sorted(set(g4_runs) | set(lqg_runs))
    rows = []
    for key in keys:
        T, step_min = key
        row = {'T_total_days': T, 'step_minutes': step_min}
        if key in g4_runs:
            m = json.load(open(g4_runs[key] / "manifest.json"))
            s = m.get('summary', {})
            row['mpc_mean_A']   = s.get('mean_A_mpc')
            row['mpc_baseline'] = s.get('mean_A_baseline')
            row['mpc_F_viol']   = s.get('F_violation_frac_mpc')
            row['mpc_compute_min'] = (s.get('total_compute_s', 0.0) or 0.0) / 60.0
        if key in lqg_runs:
            m = json.load(open(lqg_runs[key] / "manifest.json"))
            s = m.get('summary', {})
            row['lqg_mean_A']   = s.get('mean_A_lqg')
            row['lqg_baseline'] = s.get('mean_A_baseline')
            row['lqg_F_viol']   = s.get('F_violation_frac_lqg')
            row['lqg_compute_s'] = s.get('total_compute_s')
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    print()
    print(f"  {'T_d':>4} {'h_min':>5}  {'A_const':>8} {'A_LQG':>8} {'A_MPC':>8}  "
          f"{'r_LQG':>6} {'r_MPC':>6}  {'F_LQG':>7} {'F_MPC':>7}  "
          f"{'comp_LQG':>9} {'comp_MPC':>9}")
    print(f"  {'---':>4} {'-----':>5}  {'-------':>8} {'-----':>8} {'-----':>8}  "
          f"{'-----':>6} {'-----':>6}  {'------':>7} {'------':>7}  "
          f"{'--------':>9} {'--------':>9}")
    for r in rows:
        T = r['T_total_days']; h = r['step_minutes']
        a_b = r.get('mpc_baseline') or r.get('lqg_baseline')
        a_l = r.get('lqg_mean_A')
        a_m = r.get('mpc_mean_A')
        r_l = (a_l / a_b) if (a_l and a_b) else None
        r_m = (a_m / a_b) if (a_m and a_b) else None
        f_l = r.get('lqg_F_viol')
        f_m = r.get('mpc_F_viol')
        c_l = r.get('lqg_compute_s')
        c_m = r.get('mpc_compute_min')

        def _fmt(v, w, prec):
            return f"{v:{w}.{prec}f}" if v is not None else f"{'-':>{w}}"

        print(f"  {T:>4} {h:>5}  "
              f"{_fmt(a_b, 8, 4)} {_fmt(a_l, 8, 4)} {_fmt(a_m, 8, 4)}  "
              f"{_fmt(r_l, 6, 3)} {_fmt(r_m, 6, 3)}  "
              f"{_fmt(f_l, 7, 4)} {_fmt(f_m, 7, 4)}  "
              f"{_fmt(c_l, 9, 1)} {_fmt(c_m, 9, 1)}")


def _save_csv(rows: list[dict], out_path: Path) -> None:
    cols = ['T_total_days', 'step_minutes',
            'mpc_baseline', 'lqg_mean_A', 'mpc_mean_A',
            'lqg_F_viol', 'mpc_F_viol',
            'lqg_compute_s', 'mpc_compute_min']
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(
                f"{r.get(c)}" if r.get(c) is not None else ''
                for c in cols
            ) + '\n')
    print(f"  CSV written: {out_path}")


def _plot_mean_A_vs_T(rows: list[dict], out_path: Path) -> None:
    Ts_baseline, mpc_xs, mpc_ys, lqg_xs, lqg_ys, base_xs, base_ys = (
        [], [], [], [], [], [], []
    )
    for r in rows:
        T = r['T_total_days']
        if r.get('mpc_baseline') is not None:
            base_xs.append(T); base_ys.append(r['mpc_baseline'])
        elif r.get('lqg_baseline') is not None:
            base_xs.append(T); base_ys.append(r['lqg_baseline'])
        if r.get('mpc_mean_A') is not None:
            mpc_xs.append(T); mpc_ys.append(r['mpc_mean_A'])
        if r.get('lqg_mean_A') is not None:
            lqg_xs.append(T); lqg_ys.append(r['lqg_mean_A'])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if base_xs:
        ax.plot(base_xs, base_ys, 'o-', color='gray', label=r'constant $\Phi=1.0$ baseline')
    if lqg_xs:
        ax.plot(lqg_xs, lqg_ys, 's--', color='steelblue', label='LQG open-loop')
    if mpc_xs:
        ax.plot(mpc_xs, mpc_ys, '^-', color='crimson',
                label=r'SMC$^2$ MPC closed-loop')
    ax.set_xlabel('horizon $T$ (days)')
    ax.set_ylabel(r'mean $\int A \, dt / T$')
    ax.set_title('Closed-loop performance vs horizon — FSA-v2')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot written: {out_path}")


def _plot_triptych(rows: list[dict], T_target: int, out_path: Path) -> None:
    """A(t), F(t), Phi(t) overlay at T=T_target, mixing whichever step grid is available."""
    target_g4 = next((r for r in rows
                       if r['T_total_days'] == T_target
                       and r.get('mpc_mean_A') is not None), None)
    target_lqg = next((r for r in rows
                        if r['T_total_days'] == T_target
                        and r.get('lqg_mean_A') is not None), None)
    if not target_g4 and not target_lqg:
        print(f"  [skip] no run available for T={T_target}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if target_g4:
        # Find the actual run dir for this row
        d = next((p for p in _list_runs(G4_ROOT)
                   if int(json.load(open(p / "manifest.json"))['T_total_days']) == T_target), None)
        if d:
            data = np.load(d / "data.npz", allow_pickle=False)
            traj_mpc = data['trajectory_mpc']
            traj_base = data['trajectory_baseline']
            phi_bin = data['Phi_per_bin']
            t = np.arange(traj_mpc.shape[0]) * float(json.load(
                open(d / "manifest.json"))['DT'])
            t_base = np.arange(min(traj_mpc.shape[0], traj_base.shape[0])) \
                * float(json.load(open(d / "manifest.json"))['DT'])
            axes[0].plot(t_base, traj_base[:len(t_base), 2],
                         color='gray', alpha=0.7, label='const baseline')
            axes[0].plot(t, traj_mpc[:, 2], color='crimson',
                         label='SMC$^2$ MPC')
            axes[1].plot(t_base, traj_base[:len(t_base), 1],
                         color='gray', alpha=0.7, label='const baseline')
            axes[1].plot(t, traj_mpc[:, 1], color='crimson',
                         label='SMC$^2$ MPC')
            axes[2].plot(t, phi_bin[:len(t)], color='crimson', alpha=0.7,
                         label=r'SMC$^2$ MPC $\Phi$')

    if target_lqg:
        d = next((p for p in _list_runs(LQG_ROOT)
                   if int(json.load(open(p / "manifest.json"))['T_total_days']) == T_target
                   and int(json.load(open(p / "manifest.json"))
                            .get('step_minutes', 15)) == target_lqg['step_minutes']), None)
        if d:
            data = np.load(d / "data.npz", allow_pickle=False)
            traj_lqg = data['trajectory_lqg']
            phi_bin = data['Phi_per_bin']
            dt = float(json.load(open(d / "manifest.json"))['dt_days'])
            t = np.arange(traj_lqg.shape[0]) * dt
            axes[0].plot(t, traj_lqg[:, 2], color='steelblue', linestyle='--',
                         label='LQG open-loop')
            axes[1].plot(t, traj_lqg[:, 1], color='steelblue', linestyle='--',
                         label='LQG open-loop')
            axes[2].plot(t, phi_bin, color='steelblue', linestyle='--',
                         label=r'LQG $\Phi$')

    axes[0].axhline(0.0, color='k', alpha=0.3, lw=0.5)
    axes[0].set_ylabel(r'$A(t)$')
    axes[0].set_title(f'Closed-loop trajectories — T = {T_target} days')
    axes[0].legend(loc='upper left', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.40, color='r', linestyle=':', alpha=0.5,
                     label=r'$F_{max} = 0.40$')
    axes[1].set_ylabel(r'$F(t)$')
    axes[1].legend(loc='upper left', fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].axhline(1.0, color='gray', linestyle=':', alpha=0.5,
                     label=r'$\Phi_\mathrm{default} = 1.0$')
    axes[2].set_ylabel(r'$\Phi(t)$')
    axes[2].set_xlabel('t (days)')
    axes[2].legend(loc='upper left', fontsize=9)
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Plot written: {out_path}")


def main():
    print("=" * 76)
    print("  Stage H+I — comparison: SMC^2 MPC vs LQG vs constant baseline")
    print("=" * 76)
    rows = _build_table()
    if not rows:
        print("  No checkpoints found.")
        return
    _print_table(rows)
    _save_csv(rows, OUT_DIR / "g4_lqg_summary.csv")
    _plot_mean_A_vs_T(rows, OUT_DIR / "g4_lqg_mean_A_vs_T.png")
    for T in [14, 42, 84]:
        _plot_triptych(rows, T,
                        OUT_DIR / f"g4_lqg_triptych_T{T}.png")


if __name__ == '__main__':
    main()
