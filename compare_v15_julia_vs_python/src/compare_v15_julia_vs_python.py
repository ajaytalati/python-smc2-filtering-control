"""Phase D of the v1.5 closed-loop comparison plan: load both bench
outputs + both GPU telemetry CSVs + both per-stride CSVs and produce:

  1. `comparison.png` — a 6-panel three-way diagnostic figure
       (Python MPC blue, Julia MPC red, baseline grey).
  2. A printed summary table with the **baseline as the gate** and
     the microbenchmark profiler numbers alongside the live macro
     telemetry.
  3. A short `summary.txt` capturing the same table.

Inputs (positional):
  parent_dir   directory produced by run_v15_T28d_compare.sh; expected
               to contain {julia,python}/ subdirs and *_profile.log files.

Usage:
  JAX_ENABLE_X64=True PYTHONPATH=.:.. \\
    python tools/compare_v15_julia_vs_python.py <parent_dir>

Notes:
  - JLD2 files are HDF5 under the hood; we read them with h5py.
  - Profile logs are parsed by simple regex; v1.5 profiler outputs the
    same structured lines (`median time per call`, `effective TFLOPS`,
    `utilisation`, etc.) on both stacks.
  - GPU telemetry CSV is the standard `nvidia-smi --format=csv`
    output (`timestamp, utilization.gpu [%], memory.used [MiB], …`).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


# ── Profile log parsing (matches the structured output of both
#    profile_gpu.{jl,py} for the lines we actually care about) ──────────

_FLOAT = r'([0-9]+\.[0-9]+|[0-9]+)'
_RE_KERNEL_HEADER = re.compile(r'PROFILE:\s+(filter|controller)', re.IGNORECASE)
_RE_TIME = re.compile(rf'median time per call\s*:\s*{_FLOAT}\s*s')
_RE_TFLOPS = re.compile(
    rf'effective TFLOPS\s*:\s*{_FLOAT}\s*/\s*{_FLOAT}\s*peak\s*=\s*{_FLOAT}%')


def parse_profile_log(path: Path) -> dict:
    """Pull (filter, controller) {time_s, tflops_eff, util_pct} from a
    profile_gpu.{jl,py} log. Returns dict with `filter` and `controller`
    keys; missing sections become None."""
    out = {'filter': None, 'controller': None}
    if not path.exists():
        return out
    text = path.read_text()
    sections = []
    last = None
    for line in text.splitlines():
        m = _RE_KERNEL_HEADER.search(line)
        if m:
            kind = m.group(1).lower()
            last = {'kind': kind}
            sections.append(last)
            continue
        if last is None:
            continue
        m = _RE_TIME.search(line)
        if m:
            last['time_s'] = float(m.group(1))
            continue
        m = _RE_TFLOPS.search(line)
        if m:
            last['tflops_eff'] = float(m.group(1))
            last['util_pct']   = float(m.group(3))
    for s in sections:
        out[s['kind']] = {k: v for k, v in s.items() if k != 'kind'}
    return out


# ── Loaders for the two output formats ────────────────────────────────

def load_python_run(run_dir: Path) -> dict:
    npz = np.load(run_dir / 'trajectory.npz', allow_pickle=True)
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    out = {
        'manifest': manifest,
        'trajectory_mpc':       np.asarray(npz['trajectory_mpc']),
        'trajectory_baseline':  np.asarray(npz['trajectory_baseline']),
        'applied_phi_per_stride': np.asarray(npz['applied_phi_per_stride']),
        'posterior_particles':  np.asarray(npz['posterior_particles']),
        'posterior_window_mask': np.asarray(npz['posterior_window_mask']),
        'param_names':          [str(n) for n in npz['param_names']],
        'BINS_PER_DAY':         int(manifest['BINS_PER_DAY']),
        'STRIDE_BINS':          int(manifest['STRIDE_BINS']),
        'WINDOW_BINS':          int(manifest['WINDOW_BINS']),
        'dt_days':              float(npz['dt_days']),
        'truth_params':         manifest.get('truth_params', {}),
        'wall_seconds':         float(manifest.get('total_elapsed_s', 0.0)),
    }
    out['per_stride'] = _read_per_stride(run_dir / 'per_stride.csv')
    return out


def load_julia_run(run_dir: Path) -> dict:
    """Read JLD2 via h5py. JLD2 is HDF5-based; numeric arrays map to
    top-level HDF5 datasets directly. Strings + Dicts use compound
    types that h5py doesn't read cleanly — for those we read from
    `manifest.json` instead."""
    import h5py
    h = h5py.File(run_dir / 'data.jld2', 'r')
    manifest = json.loads((run_dir / 'manifest.json').read_text())

    def _arr(key):
        return np.asarray(h[key])

    def _arr_T(key):
        """JLD2 stores Julia column-major arrays; h5py reads them with
        the dimensions swapped. For 2D arrays we want the Julia layout
        back — transpose."""
        a = np.asarray(h[key])
        return a.T if a.ndim == 2 else a

    def _arr_3d_julia(key):
        """For 3D Julia arrays (n_strides, n_smc, n_params), h5py reads
        them as (n_params, n_smc, n_strides) — reverse all three axes."""
        a = np.asarray(h[key])
        if a.ndim == 3:
            return a.transpose(2, 1, 0)
        return a

    # Param names + truth come from manifest, not JLD2.
    param_names = list(manifest.get('estimated_params',
                                      manifest.get('param_names', [])))
    truth_params = manifest.get('truth_params_dict',
                                  manifest.get('truth_params', {}))
    truth_params = {k: float(v) for k, v in truth_params.items()
                    if k in param_names}

    out = {
        'manifest': manifest,
        'trajectory_mpc':       _arr_T('trajectory_mpc'),
        'trajectory_baseline':  _arr_T('trajectory_baseline'),
        'applied_phi_per_stride': _arr('daily_phi_per_stride'),
        'posterior_particles':  _arr_3d_julia('posterior_particles'),
        'posterior_window_mask': _arr('posterior_window_mask').astype(bool),
        'param_names':          param_names,
        'BINS_PER_DAY':         int(_arr('BINS_PER_DAY')),
        'STRIDE_BINS':          int(_arr('STRIDE_BINS')),
        'WINDOW_BINS':          int(_arr('WINDOW_BINS')),
        'dt_days':              float(_arr('dt_days')),
        'truth_params':         truth_params,
        'wall_seconds':         float(_arr('wall_seconds')),
    }
    h.close()
    out['per_stride'] = _read_per_stride(run_dir / 'per_stride.csv')
    return out


def _read_per_stride(path: Path) -> dict:
    """Return per-stride telemetry as a dict of arrays. Empty if file
    missing."""
    if not path.exists():
        return {}
    rows = []
    with open(path, newline='') as fh:
        r = csv.DictReader(fh)
        for row in r:
            rows.append({k: float(v) if v not in ('', None) else float('nan')
                          for k, v in row.items()})
    if not rows:
        return {}
    return {k: np.array([row[k] for row in rows]) for k in rows[0].keys()}


def parse_gpu_telemetry(path: Path) -> dict:
    """Parse the standard `nvidia-smi --format=csv` output. Returns
    {util_pct: ndarray, mem_used_gb: ndarray, power_w: ndarray,
     n_samples: int} — empty fields if file missing."""
    if not path.exists():
        return dict(util_pct=np.array([]), mem_used_gb=np.array([]),
                    power_w=np.array([]), n_samples=0)
    util, mem, pwr = [], [], []
    with open(path, newline='') as fh:
        r = csv.reader(fh)
        try:
            header = next(r)
        except StopIteration:
            return dict(util_pct=np.array([]), mem_used_gb=np.array([]),
                        power_w=np.array([]), n_samples=0)
        # Locate the columns. nvidia-smi includes units in the header.
        def _find(substr):
            for i, h in enumerate(header):
                if substr in h.lower():
                    return i
            return -1
        i_util = _find('utilization.gpu')
        i_mem  = _find('memory.used')
        i_pwr  = _find('power.draw')
        for row in r:
            if not row:
                continue
            try:
                if i_util >= 0:
                    util.append(float(row[i_util].strip().rstrip('%').strip()))
                if i_mem >= 0:
                    # `memory.used` reports MiB; strip 'MiB' suffix.
                    mem.append(float(row[i_mem].strip().split()[0]) / 1024.0)
                if i_pwr >= 0:
                    pwr.append(float(row[i_pwr].strip().split()[0]))
            except Exception:
                continue
    return dict(util_pct=np.array(util), mem_used_gb=np.array(mem),
                power_w=np.array(pwr), n_samples=len(util))


# ── Diagnostic table + plot ────────────────────────────────────────────

def _mean_A_last_n_days(traj: np.ndarray, dt_days: float, n_days: int) -> float:
    if traj.size == 0:
        return float('nan')
    n_bins_total = traj.shape[0]
    n_bins_last  = min(n_bins_total, int(n_days / dt_days))
    return float(np.mean(traj[-n_bins_last:, 2]))


def _F_violation_frac(traj: np.ndarray, F_max: float = 0.40) -> float:
    if traj.size == 0:
        return float('nan')
    return float(np.mean(traj[:, 1] > F_max))


def _posterior_mse_to_truth(post: np.ndarray, mask: np.ndarray,
                              names: list, truth: dict) -> float:
    """Final-window MSE of posterior medians vs truth, log-relative.
    Returns nan if no valid windows or no truth values overlap."""
    if not mask.any():
        return float('nan')
    # Use the LAST valid window's posterior median.
    last_idx = int(np.where(mask)[0][-1])
    last_post = post[last_idx]                    # (n_smc, n_params)
    medians = np.median(last_post, axis=0)
    errs = []
    for i, name in enumerate(names):
        if name in truth and truth[name] != 0:
            errs.append(((medians[i] - truth[name]) / truth[name]) ** 2)
    return float(np.mean(errs)) if errs else float('nan')


def build_summary(py_run: dict, jl_run: dict,
                   py_prof: dict, jl_prof: dict,
                   py_gpu: dict, jl_gpu: dict, dt_days: float) -> str:
    """Format the diagnostic table as a string."""
    lines = []
    lines.append('=' * 78)
    lines.append('  v1.5 closed-loop three-way comparison summary')
    lines.append('=' * 78)
    lines.append('')

    # ── Algorithmic ──
    py_traj  = py_run['trajectory_mpc']
    jl_traj  = jl_run['trajectory_mpc']
    py_base  = py_run['trajectory_baseline']
    jl_base  = jl_run['trajectory_baseline']

    # The two stacks compute their own baselines from independent RNGs
    # — mean their baselines for a single reference number. Should be
    # statistically close.
    base_traj_pooled = np.concatenate([py_base, jl_base], axis=0) \
        if py_base.size and jl_base.size else (py_base if py_base.size else jl_base)

    py_meanA  = _mean_A_last_n_days(py_traj,  dt_days, 7)
    jl_meanA  = _mean_A_last_n_days(jl_traj,  dt_days, 7)
    bs_meanA  = _mean_A_last_n_days(base_traj_pooled, dt_days, 7)
    py_imp = (py_meanA / bs_meanA - 1.0) * 100 if bs_meanA else float('nan')
    jl_imp = (jl_meanA / bs_meanA - 1.0) * 100 if bs_meanA else float('nan')

    py_fviol = _F_violation_frac(py_traj)
    jl_fviol = _F_violation_frac(jl_traj)
    bs_fviol = _F_violation_frac(base_traj_pooled)

    py_mse = _posterior_mse_to_truth(py_run['posterior_particles'],
                                       py_run['posterior_window_mask'],
                                       py_run['param_names'],
                                       py_run['truth_params'])
    jl_mse = _posterior_mse_to_truth(jl_run['posterior_particles'],
                                       jl_run['posterior_window_mask'],
                                       jl_run['param_names'],
                                       jl_run['truth_params'])

    fmt = '  {:38s}  {:>16s}  {:>16s}  {:>16s}'
    lines.append(fmt.format('', 'Baseline (Φ=1.0)', 'Python+JAX MPC', 'Julia MPC'))
    lines.append('  ' + '─' * 76)
    lines.append(fmt.format('mean A (last 7 days)',
                             f'{bs_meanA:.4f}', f'{py_meanA:.4f}', f'{jl_meanA:.4f}'))
    lines.append(fmt.format('  improvement vs baseline',
                             '—', f'{py_imp:+.1f}%', f'{jl_imp:+.1f}%'))
    lines.append(fmt.format('F-violation rate (frac)',
                             f'{bs_fviol:.4f}', f'{py_fviol:.4f}', f'{jl_fviol:.4f}'))
    lines.append(fmt.format('posterior MSE to truth (final)',
                             '—',
                             'nan' if np.isnan(py_mse) else f'{py_mse:.4f}',
                             'nan' if np.isnan(jl_mse) else f'{jl_mse:.4f}'))

    # ── Macro telemetry ──
    lines.append('  ' + '─' * 76)
    lines.append('  ' + ' macro (during the closed-loop run) '.center(76, '─'))
    lines.append(fmt.format('total wall time (min)',
                             '~0 (plant-only)',
                             f'{py_run["wall_seconds"] / 60:.1f}',
                             f'{jl_run["wall_seconds"] / 60:.1f}'))
    lines.append(fmt.format('mean GPU utilisation (%)', '—',
                             f'{np.mean(py_gpu["util_pct"]):.1f}'
                                if py_gpu['n_samples'] else 'no nvidia-smi log',
                             f'{np.mean(jl_gpu["util_pct"]):.1f}'
                                if jl_gpu['n_samples'] else 'no nvidia-smi log'))
    lines.append(fmt.format('peak GPU memory (GB)', '—',
                             f'{np.max(py_gpu["mem_used_gb"]):.2f}'
                                if py_gpu['n_samples'] else '—',
                             f'{np.max(jl_gpu["mem_used_gb"]):.2f}'
                                if jl_gpu['n_samples'] else '—'))
    lines.append(fmt.format('mean GPU power (W)', '—',
                             f'{np.mean(py_gpu["power_w"]):.0f}'
                                if py_gpu['n_samples'] else '—',
                             f'{np.mean(jl_gpu["power_w"]):.0f}'
                                if jl_gpu['n_samples'] else '—'))

    # ── Micro (profilers) ──
    lines.append('  ' + '─' * 76)
    lines.append('  ' + ' micro (profile_gpu, isolated kernel timing) '.center(76, '─'))

    def _prof_row(label, py_field, jl_field, scale=1000.0, fmt_str='{:.1f}'):
        py_v = py_prof.get(label.split()[0]) or {}
        jl_v = jl_prof.get(label.split()[0]) or {}
        py_x = py_v.get(py_field, float('nan'))
        jl_x = jl_v.get(jl_field, float('nan'))
        py_x = py_x * scale if not np.isnan(py_x) else py_x
        jl_x = jl_x * scale if not np.isnan(jl_x) else jl_x
        return None  # placeholder; we emit explicit rows below

    pf_t = py_prof.get('filter') or {}
    jl_t = jl_prof.get('filter') or {}
    lines.append(fmt.format('filter kernel time / call (ms)', '—',
                             f'{pf_t.get("time_s", float("nan")) * 1000:.1f}',
                             f'{jl_t.get("time_s", float("nan")) * 1000:.1f}'))
    lines.append(fmt.format('  effective TFLOPS', '—',
                             f'{pf_t.get("tflops_eff", float("nan")):.2f}',
                             f'{jl_t.get("tflops_eff", float("nan")):.2f}'))
    lines.append(fmt.format('  util vs 5090 peak (%)', '—',
                             f'{pf_t.get("util_pct", float("nan")):.1f}',
                             f'{jl_t.get("util_pct", float("nan")):.1f}'))

    pc_t = py_prof.get('controller') or {}
    jc_t = jl_prof.get('controller') or {}
    lines.append(fmt.format('controller cost / call (ms)', '—',
                             f'{pc_t.get("time_s", float("nan")) * 1000:.1f}',
                             f'{jc_t.get("time_s", float("nan")) * 1000:.1f}'))
    lines.append(fmt.format('  effective TFLOPS', '—',
                             f'{pc_t.get("tflops_eff", float("nan")):.2f}',
                             f'{jc_t.get("tflops_eff", float("nan")):.2f}'))
    lines.append(fmt.format('  util vs 5090 peak (%)', '—',
                             f'{pc_t.get("util_pct", float("nan")):.1f}',
                             f'{jc_t.get("util_pct", float("nan")):.1f}'))

    lines.append('=' * 78)
    return '\n'.join(lines)


def make_comparison_plot(py_run, jl_run, py_gpu, jl_gpu, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), dpi=120)
    fig.suptitle('FSA v1.5 closed-loop three-way comparison '
                  '(Julia red / Python+JAX blue / baseline grey)',
                  fontsize=11)

    # ── Time grids ──
    py_dt = py_run['dt_days']
    jl_dt = jl_run['dt_days']
    py_t = np.arange(py_run['trajectory_mpc'].shape[0]) * py_dt
    jl_t = np.arange(jl_run['trajectory_mpc'].shape[0]) * jl_dt
    py_bt = np.arange(py_run['trajectory_baseline'].shape[0]) * py_dt
    jl_bt = np.arange(jl_run['trajectory_baseline'].shape[0]) * jl_dt

    # (0,0) A trajectory overlay
    ax = axes[0, 0]
    if py_bt.size:
        ax.plot(py_bt, py_run['trajectory_baseline'][:, 2],
                color='grey', lw=1.0, alpha=0.6, label='baseline (Python plant)')
    if jl_bt.size:
        ax.plot(jl_bt, jl_run['trajectory_baseline'][:, 2],
                color='dimgrey', lw=1.0, alpha=0.6, ls=':',
                label='baseline (Julia plant)')
    ax.plot(py_t, py_run['trajectory_mpc'][:, 2], color='steelblue',
            lw=1.6, label='Python+JAX MPC')
    ax.plot(jl_t, jl_run['trajectory_mpc'][:, 2], color='firebrick',
            lw=1.6, label='Julia MPC')
    ax.set_title('A trajectory'); ax.set_xlabel('time (days)')
    ax.set_ylabel('A'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (0,1) Applied daily Φ overlay
    ax = axes[0, 1]
    py_phi = py_run['applied_phi_per_stride']
    jl_phi = jl_run['applied_phi_per_stride']
    py_stride_t = np.arange(len(py_phi)) * py_run['STRIDE_BINS'] * py_dt
    jl_stride_t = np.arange(len(jl_phi)) * jl_run['STRIDE_BINS'] * jl_dt
    ax.axhline(1.0, color='grey', ls='--', lw=1.0, label='baseline Φ=1.0')
    ax.plot(py_stride_t, py_phi, marker='o', ms=4, lw=1.5,
            color='steelblue', label='Python+JAX MPC')
    ax.plot(jl_stride_t, jl_phi, marker='s', ms=4, lw=1.5,
            color='firebrick', label='Julia MPC')
    ax.set_title('applied daily Φ per stride')
    ax.set_xlabel('time (days)'); ax.set_ylabel('Φ')
    ax.set_ylim(0.0, max(1.5, max(np.max(py_phi) if py_phi.size else 0,
                                    np.max(jl_phi) if jl_phi.size else 0) * 1.1))
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # (1,0)/(1,1) Two example posterior medians
    common_names = [n for n in py_run['param_names']
                    if n in jl_run['param_names']
                    and n in py_run['truth_params']]
    panel_params = (common_names[:2] + ['', ''])[:2]
    for pi, name in enumerate(panel_params):
        ax = axes[1, pi]
        if not name:
            ax.axis('off'); continue
        i_py = py_run['param_names'].index(name)
        i_jl = jl_run['param_names'].index(name)
        py_mask = py_run['posterior_window_mask']
        jl_mask = jl_run['posterior_window_mask']
        py_xt = (np.arange(len(py_mask)) * py_run['STRIDE_BINS']
                  + py_run['WINDOW_BINS']) / py_run['BINS_PER_DAY']
        jl_xt = (np.arange(len(jl_mask)) * jl_run['STRIDE_BINS']
                  + jl_run['WINDOW_BINS']) / jl_run['BINS_PER_DAY']
        if py_mask.any():
            med = np.median(py_run['posterior_particles'][py_mask, :, i_py], axis=1)
            ax.plot(py_xt[py_mask], med, color='steelblue', lw=1.4, label='Python median')
        if jl_mask.any():
            med = np.median(jl_run['posterior_particles'][jl_mask, :, i_jl], axis=1)
            ax.plot(jl_xt[jl_mask], med, color='firebrick', lw=1.4, label='Julia median')
        ax.axhline(py_run['truth_params'][name], color='red', ls='--', lw=1.0,
                   label='truth')
        ax.set_title(f'posterior median: {name}')
        ax.set_xlabel('end of window (days)'); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # (2,0) GPU utilisation over wall time (relative seconds)
    ax = axes[2, 0]
    have_any_gpu = False
    if py_gpu['n_samples']:
        ax.plot(np.arange(py_gpu['n_samples']),
                py_gpu['util_pct'], color='steelblue', lw=1.0,
                label='Python+JAX')
        have_any_gpu = True
    if jl_gpu['n_samples']:
        ax.plot(np.arange(jl_gpu['n_samples']),
                jl_gpu['util_pct'], color='firebrick', lw=1.0,
                label='Julia')
        have_any_gpu = True
    if not have_any_gpu:
        ax.text(0.5, 0.5, 'no nvidia-smi telemetry', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='grey')
    ax.set_title('GPU SM utilisation over wall time')
    ax.set_xlabel('sample idx (1 Hz)'); ax.set_ylabel('util %')
    ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    if have_any_gpu:
        ax.legend(fontsize=8)

    # (2,1) Per-stride wall time
    ax = axes[2, 1]
    py_ps = py_run['per_stride']; jl_ps = jl_run['per_stride']
    width = 0.4
    if py_ps:
        ax.bar(py_ps['stride'] - width/2, py_ps['t_wall_s'],
               width=width, color='steelblue', label='Python+JAX')
    if jl_ps:
        # Julia strides are 1-indexed; align to 0-indexed for the plot
        jl_x = jl_ps['stride'] - 1
        ax.bar(jl_x + width/2, jl_ps['t_wall_s'],
               width=width, color='firebrick', label='Julia')
    ax.set_title('per-stride wall time'); ax.set_xlabel('stride idx')
    ax.set_ylabel('seconds'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='v1.5 three-way comparison: Julia vs Python+JAX vs baseline')
    ap.add_argument('parent_dir',
                    help='dir produced by run_v15_T28d_compare.sh '
                         '(must contain {julia,python}/ subdirs and '
                         '*_profile.log files)')
    args = ap.parse_args()

    parent = Path(args.parent_dir)
    if not parent.exists():
        sys.exit(f"parent_dir not found: {parent}")
    py_dir = parent / 'python'
    jl_dir = parent / 'julia'

    print(f"loading Python+JAX run from {py_dir}/ …")
    py_run  = load_python_run(py_dir)
    print(f"loading Julia run from {jl_dir}/ …")
    jl_run  = load_julia_run(jl_dir)
    print(f"loading profile logs …")
    py_prof = parse_profile_log(parent / 'python_profile.log')
    jl_prof = parse_profile_log(parent / 'julia_profile.log')
    print(f"loading GPU telemetry CSVs …")
    py_gpu  = parse_gpu_telemetry(py_dir / 'gpu_telemetry.csv')
    jl_gpu  = parse_gpu_telemetry(jl_dir / 'gpu_telemetry.csv')

    dt_days = py_run['dt_days']    # both stacks must agree; pick one

    summary = build_summary(py_run, jl_run, py_prof, jl_prof,
                              py_gpu, jl_gpu, dt_days)
    print()
    print(summary)
    (parent / 'summary.txt').write_text(summary + '\n')
    print(f"\n  wrote {parent / 'summary.txt'}")

    print()
    print("rendering comparison.png …")
    make_comparison_plot(py_run, jl_run, py_gpu, jl_gpu,
                          parent / 'comparison.png')


if __name__ == '__main__':
    main()
