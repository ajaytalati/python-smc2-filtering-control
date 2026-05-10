"""Per-horizon headline-metrics extractor for the Julia v1.5 sweep.

Reads one horizon's run dir (`data.jld2`, `manifest.json`,
`per_stride.csv`, `nvidia_smi.csv`) and writes:

    1. `<horizon>/results.json` — the full extracted metric dict.
    2. one row appended to `<sweep_root>/horizons_results.csv`
       (creating the file with a header on first call).

Called by the sweep launcher
(`version_1_5_Julia/tools/launchers/run_julia_horizon_sweep.sh`)
once per horizon after the bench finishes.

JLD2 is read via `h5py` (Julia's JLD2 backend is HDF5).

CLI:
    python extract_horizon_results.py \\
        --horizon-dir <DIR> --T-days <N> \\
        --sweep-csv <CSV> [--bench-exit-code <RC>]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np


# ── Field order for horizons_results.csv ────────────────────────────────
# The launcher's later "render to markdown" step prints these in this
# column order, so keep it stable.
FIELDS: tuple[str, ...] = (
    "T_days",
    "n_strides",
    "n_replans",
    "total_wall_s",
    "mean_stride_wall_s",
    "max_stride_wall_s",
    "mean_filter_n_temp",
    "mean_ctrl_n_temp",
    "mean_gpu_util_pct",
    "pct_time_gpu_above_90",
    "peak_vram_mib",
    "mean_power_w",
    "mean_A_mpc",
    "final_A_mpc",
    "mean_A_baseline",
    "final_A_baseline",
    "mean_B_mpc",
    "final_B_mpc",
    "final_F_mpc",
    "n_F_violations",
    "pct_F_violations",
    "max_F_overshoot",
    "mean_phi_per_stride",
    "final_phi_per_stride",
    "mean_phi_at_stride21",
    "bench_exit_code",
)


def _read_manifest(horizon_dir: Path) -> dict:
    """Loads `manifest.json` from the horizon's run dir."""
    with open(horizon_dir / "manifest.json") as fh:
        return json.load(fh)


def _read_per_stride_csv(horizon_dir: Path) -> list[dict]:
    """Parses `per_stride.csv` into a list of float-coerced dicts."""
    rows: list[dict] = []
    csv_path = horizon_dir / "per_stride.csv"
    if not csv_path.exists():
        return rows
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append({k: _to_float(v) for k, v in r.items()})
    return rows


def _to_float(s: str) -> float:
    """Parses a CSV cell to float; NaN on failure."""
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _read_jld2(horizon_dir: Path) -> dict[str, np.ndarray]:
    """Reads the Julia JLD2 file via h5py, returns the keys we need.

    JLD2 wraps HDF5; primitive numeric arrays come through h5py as
    plain ndarrays.

    Returns:
        Dict with `trajectory_mpc`, `trajectory_baseline`,
        `daily_phi_per_stride` arrays. Missing keys are not raised
        (caller decides how to handle absence).
    """
    out: dict[str, np.ndarray] = {}
    with h5py.File(horizon_dir / "data.jld2", "r") as fh:
        for k in ("trajectory_mpc", "trajectory_baseline",
                  "daily_phi_per_stride"):
            if k in fh:
                out[k] = np.asarray(fh[k])
    return out


def _read_nvidia_smi(horizon_dir: Path) -> Optional[np.ndarray]:
    """Reads `nvidia_smi.csv` into a (n_samples, 3) array.

    Columns expected: timestamp, gpu_util_percent, memory_used_mib,
    power_w. We drop the timestamp (string) and return floats for the
    last three.

    Returns:
        Array of shape ``(n_samples, 3)`` (util, mem, power), or
        ``None`` if the CSV doesn't exist or has no data rows.
    """
    p = horizon_dir / "nvidia_smi.csv"
    if not p.exists():
        return None
    rows: list[tuple[float, float, float]] = []
    with open(p) as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for r in reader:
            if len(r) < 4:
                continue
            try:
                rows.append((float(r[1]), float(r[2]), float(r[3])))
            except ValueError:
                continue
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _bfa_axes(traj: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Returns the trajectory in (3, n_bins) form regardless of input order.

    Julia is column-major; a Julia ``(n_bins, 3)`` array writes to HDF5
    as on-disk shape ``(3, n_bins)`` (and h5py reads it back that way).
    But a future version of the bench might transpose before writing,
    so this helper tolerates both: if the array's first axis has length
    3 we return as-is; if its second axis has length 3 we transpose.
    Any other shape returns ``None`` so the caller can NaN-fill.

    Args:
        traj: Trajectory ndarray as read from `data.jld2`, or ``None``.

    Returns:
        ``(3, n_bins)`` view of the input, or ``None`` if absent /
        unrecognised.
    """
    if traj is None or traj.ndim != 2:
        return traj
    if traj.shape[0] == 3:
        return traj
    if traj.shape[1] == 3:
        return traj.T
    return None


def _safe_mean(xs: np.ndarray) -> float:
    """Returns mean(xs) or NaN if empty."""
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.mean(xs)) if xs.size else float("nan")


def _safe_max(xs: np.ndarray) -> float:
    """Returns max(xs) or NaN if empty."""
    xs = np.asarray(xs, dtype=np.float64)
    return float(np.max(xs)) if xs.size else float("nan")


def extract_metrics(horizon_dir: Path, T_days: int,
                    bench_exit_code: int) -> dict[str, Any]:
    """Builds the headline-metrics dict for one horizon.

    Args:
        horizon_dir: Path to ``T<N>d_seed<S>/`` containing the bench
            artefacts.
        T_days: Planning horizon for this run (passed in by the
            launcher to avoid double-bookkeeping).
        bench_exit_code: Exit code of the bench process.

    Returns:
        Dict whose keys are the columns of `horizons_results.csv`.
    """
    manifest = _read_manifest(horizon_dir)
    stride_rows = _read_per_stride_csv(horizon_dir)
    jld = _read_jld2(horizon_dir)
    smi = _read_nvidia_smi(horizon_dir)

    # ── from manifest.json ─────────────────────────────────────────
    n_strides = int(manifest.get("n_strides", len(stride_rows)))
    total_wall_s = float(manifest.get("wall_seconds", float("nan")))

    # ── from per_stride.csv ────────────────────────────────────────
    stride_walls = np.array([r.get("t_wall_s", float("nan"))
                              for r in stride_rows], dtype=np.float64)
    n_temp_filter = np.array([r.get("n_temp_filter", 0.0)
                               for r in stride_rows], dtype=np.float64)
    n_temp_ctrl = np.array([r.get("n_temp_ctrl", 0.0)
                             for r in stride_rows], dtype=np.float64)
    mean_stride_wall_s = _safe_mean(stride_walls)
    max_stride_wall_s = _safe_max(stride_walls)
    # Means restricted to non-zero rows (where the filter / replan
    # actually fired) so warmup / off-cadence strides don't dilute.
    mask_filter_fired = n_temp_filter > 0
    mask_ctrl_fired = n_temp_ctrl > 0
    mean_filter_n_temp = (_safe_mean(n_temp_filter[mask_filter_fired])
                          if mask_filter_fired.any() else 0.0)
    mean_ctrl_n_temp = (_safe_mean(n_temp_ctrl[mask_ctrl_fired])
                        if mask_ctrl_fired.any() else 0.0)
    n_replans = int(mask_ctrl_fired.sum())

    # ── from nvidia_smi.csv ────────────────────────────────────────
    if smi is not None:
        util = smi[:, 0]
        mem = smi[:, 1]
        power = smi[:, 2]
        mean_gpu_util_pct = _safe_mean(util)
        peak_vram_mib = _safe_max(mem)
        mean_power_w = _safe_mean(power)
        pct_time_gpu_above_90 = (float(np.mean(util >= 90.0))
                                 if util.size else float("nan"))
    else:
        mean_gpu_util_pct = peak_vram_mib = mean_power_w = float("nan")
        pct_time_gpu_above_90 = float("nan")

    # ── from data.jld2 (state trajectories + applied Phi) ──────────
    # Julia is column-major; an (n_bins, 3) Julia array writes to HDF5
    # as on-disk shape (3, n_bins). h5py reads it back as (3, n_bins).
    # `_bfa_axes` normalises any 2-D array to a (3, n_bins) view so the
    # row-indexed access [0=B, 1=F, 2=A] is correct regardless of
    # which axis order the bench happened to write.
    traj_mpc = _bfa_axes(jld.get("trajectory_mpc"))
    traj_base = _bfa_axes(jld.get("trajectory_baseline"))
    daily_phi = jld.get("daily_phi_per_stride")

    F_max = 0.40  # hard-coded in the bench (gpu_control.jl line 364)
    if traj_mpc is not None and traj_mpc.size:
        # Now (3, n_bins): row 0 = B, row 1 = F, row 2 = A.
        mean_A_mpc = float(np.mean(traj_mpc[2, :]))
        final_A_mpc = float(traj_mpc[2, -1])
        mean_B_mpc = float(np.mean(traj_mpc[0, :]))
        final_B_mpc = float(traj_mpc[0, -1])
        final_F_mpc = float(traj_mpc[1, -1])
        F_traj = traj_mpc[1, :]
        n_F_violations = int(np.sum(F_traj > F_max))
        pct_F_violations = float(np.mean(F_traj > F_max))
        F_overshoot = np.maximum(F_traj - F_max, 0.0)
        max_F_overshoot = float(np.max(F_overshoot))
    else:
        mean_A_mpc = final_A_mpc = mean_B_mpc = final_B_mpc = final_F_mpc = float("nan")
        n_F_violations = 0
        pct_F_violations = 0.0
        max_F_overshoot = 0.0

    if traj_base is not None and traj_base.size:
        mean_A_baseline = float(np.mean(traj_base[2, :]))
        final_A_baseline = float(traj_base[2, -1])
    else:
        mean_A_baseline = final_A_baseline = float("nan")

    if daily_phi is not None and daily_phi.size:
        mean_phi_per_stride = float(np.mean(daily_phi))
        final_phi_per_stride = float(daily_phi[-1])
        # Julia tech guide's headline metric — "applied Φ̄ at stride 21".
        # Stride numbering in daily_phi_per_stride is 1-based-physical
        # (one entry per stride, in order), so index [20] gives stride 21.
        if daily_phi.size > 20:
            mean_phi_at_stride21 = float(daily_phi[20])
        else:
            mean_phi_at_stride21 = float("nan")
    else:
        mean_phi_per_stride = final_phi_per_stride = float("nan")
        mean_phi_at_stride21 = float("nan")

    return {
        "T_days":                 T_days,
        "n_strides":              n_strides,
        "n_replans":              n_replans,
        "total_wall_s":           round(total_wall_s, 1),
        "mean_stride_wall_s":     round(mean_stride_wall_s, 2),
        "max_stride_wall_s":      round(max_stride_wall_s, 2),
        "mean_filter_n_temp":     round(mean_filter_n_temp, 2),
        "mean_ctrl_n_temp":       round(mean_ctrl_n_temp, 2),
        "mean_gpu_util_pct":      round(mean_gpu_util_pct, 1),
        "pct_time_gpu_above_90":  round(pct_time_gpu_above_90, 3),
        "peak_vram_mib":          round(peak_vram_mib, 0),
        "mean_power_w":           round(mean_power_w, 1),
        "mean_A_mpc":             round(mean_A_mpc, 4),
        "final_A_mpc":            round(final_A_mpc, 4),
        "mean_A_baseline":        round(mean_A_baseline, 4),
        "final_A_baseline":       round(final_A_baseline, 4),
        "mean_B_mpc":             round(mean_B_mpc, 4),
        "final_B_mpc":            round(final_B_mpc, 4),
        "final_F_mpc":            round(final_F_mpc, 4),
        "n_F_violations":         int(n_F_violations),
        "pct_F_violations":       round(pct_F_violations, 4),
        "max_F_overshoot":        round(max_F_overshoot, 4),
        "mean_phi_per_stride":    round(mean_phi_per_stride, 4),
        "final_phi_per_stride":   round(final_phi_per_stride, 4),
        "mean_phi_at_stride21":   round(mean_phi_at_stride21, 4),
        "bench_exit_code":        bench_exit_code,
    }


def write_results_json(horizon_dir: Path, metrics: dict) -> None:
    """Writes per-horizon `results.json`."""
    with open(horizon_dir / "results.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


def append_sweep_row(sweep_csv: Path, metrics: dict) -> None:
    """Appends one row to the sweep-level CSV (creates header if new).

    Args:
        sweep_csv: Path to `horizons_results.csv` at the sweep root.
        metrics: Dict whose keys must include all `FIELDS`.
    """
    new_file = not sweep_csv.exists()
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_csv, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        if new_file:
            writer.writeheader()
        writer.writerow({k: metrics[k] for k in FIELDS})


def main() -> int:
    """Entry point. Parses CLI, extracts, writes both outputs."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon-dir", required=True, type=Path)
    ap.add_argument("--T-days", required=True, type=int)
    ap.add_argument("--sweep-csv", required=True, type=Path)
    ap.add_argument("--bench-exit-code", type=int, default=0)
    args = ap.parse_args()

    if not args.horizon_dir.is_dir():
        print(f"horizon dir does not exist: {args.horizon_dir}",
              file=sys.stderr)
        return 1

    metrics = extract_metrics(args.horizon_dir, args.T_days,
                               args.bench_exit_code)
    write_results_json(args.horizon_dir, metrics)
    append_sweep_row(args.sweep_csv, metrics)
    print(f"  results.json + sweep-csv row written for T={args.T_days}d")
    print(f"  total_wall_s={metrics['total_wall_s']}  "
          f"mean_gpu_util={metrics['mean_gpu_util_pct']}%  "
          f"peak_vram={metrics['peak_vram_mib']} MiB")
    print(f"  mean_A_mpc={metrics['mean_A_mpc']}  "
          f"final_A_mpc={metrics['final_A_mpc']}  "
          f"baseline mean_A={metrics['mean_A_baseline']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
