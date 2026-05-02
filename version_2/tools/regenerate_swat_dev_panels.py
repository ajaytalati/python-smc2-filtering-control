#!/usr/bin/env python3
"""Regenerate the dev-repo-style 3-panel plot set (latent_states /
observations / entrainment) for an existing SWAT closed-loop run dir.

Reads ``data.npz`` + ``manifest.json``, reconstructs per-bin V_h /
V_n / V_c from ``daily_v_*_per_stride`` and the manifest's
``STRIDE_BINS``, re-rolls the four obs samplers (RNG-stable), then
calls ``models.swat.sim_plots.plot_swat_panels``.

Use this when the bench was run with an older binary that didn't
emit the dev-style panels, or to regenerate after a plot-code tweak.

Usage:
    PYTHONPATH=.:.. python tools/regenerate_swat_dev_panels.py \
        outputs/swat/swat_runs/<run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from models.swat.sim_plots import plot_swat_panels
from models.swat.simulation import (
    BINS_PER_DAY, gen_obs_hr, gen_obs_sleep, gen_obs_steps, gen_obs_stress,
)


def regenerate(run_dir: Path) -> None:
    data = np.load(run_dir / 'data.npz')
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    truth = manifest['truth_params']
    traj = data['trajectory_mpc']
    n = traj.shape[0]
    t_days = np.arange(n) / BINS_PER_DAY
    stride_bins = int(manifest['STRIDE_BINS'])

    def to_per_bin(stride_arr):
        pb = np.repeat(np.asarray(stride_arr), stride_bins)
        if len(pb) >= n:
            return pb[:n]
        return np.pad(pb, (0, n - len(pb)),
                       constant_values=stride_arr[-1])

    v_h_pb = to_per_bin(data['daily_v_h_per_stride'])
    v_n_pb = to_per_bin(data['daily_v_n_per_stride'])
    v_c_pb = to_per_bin(data['daily_v_c_per_stride'])

    # Re-roll obs from the saved trajectory using the same seed offsets
    # the plant uses (offset+1..+4).
    seed = int(manifest.get('seeds', {}).get('plant', 42))
    hr     = gen_obs_hr   (traj, t_days, truth, seed=seed + 1)
    sleep  = gen_obs_sleep(traj, t_days, truth, seed=seed + 2)
    steps  = gen_obs_steps(traj, t_days, truth,
                            sleep_label=sleep['obs_label'], seed=seed + 3)
    stress = gen_obs_stress(traj, t_days, truth,
                             V_n_per_bin=v_n_pb, seed=seed + 4)

    p_lat, p_obs, p_ent = plot_swat_panels(
        trajectory=traj, t_grid_days=t_days,
        V_h_per_bin=v_h_pb, V_n_per_bin=v_n_pb, V_c_per_bin=v_c_pb,
        obs_HR=hr, obs_sleep=sleep, obs_steps=steps, obs_stress=stress,
        params=truth, save_dir=run_dir, suffix='',
    )
    print(f"wrote: {p_lat}")
    print(f"       {p_obs}")
    print(f"       {p_ent}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    regenerate(Path(sys.argv[1]))
