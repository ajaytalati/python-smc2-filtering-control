"""StepwisePlant — mutable simulator-as-plant for closed-loop MPC.

The architectural contribution of Stage E. Wraps the v2 SDE solver
+ obs-channel sampling in a stateful class that can be advanced one
stride at a time, accepting a freshly-planned daily Φ for that stride.

Why this exists
---------------
psim's `synthesise_scenario(model, truth, init_state, n_bins, dt, aux)`
takes the **complete** exogenous schedule up-front (a full Φ_arr).
For closed-loop MPC the controller decides the next stride's Φ from
the filter's posterior at the current time — we can't pre-compute the
whole horizon. The StepwisePlant exposes:

    plant = StepwisePlant(truth_params, init_state, dt, seed)
    for window_k in range(n_windows):
        obs_stride = plant.advance(stride_bins, Phi_daily_for_stride)
        posterior  = filter.update(obs_stride)
        Phi_next   = controller.plan(posterior, horizon=stride_bins)
        Phi_daily_for_stride = Phi_next   # decided online
    plant.finalise(out_dir)   # optional psim-format artifact for archival

The SDE integration is identical to the public-dev v1 simulator's
Euler-Maruyama with sqrt-Itô diffusion (Jacobi for B, CIR for F + A).
The 4 obs channels (HR / sleep / stress / steps) + circadian C(t) are
sampled at every advance.

A `plant.finalise(out_dir)` writes a psim-format artifact so the
closed-loop trajectory can be re-validated by psim's consistency checks
post-hoc — closing the loop with the existing validation discipline.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from models.fsa_high_res.simulation import (
    BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS,
    DEFAULT_PARAMS, DEFAULT_INIT,
    drift, noise_scale_fn,
    gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps,
    gen_Phi_channel, gen_C_channel, circadian,
)
from models.fsa_high_res._phi_burst import expand_daily_phi_to_subdaily


@dataclass
class StepwisePlant:
    """Mutable ground-truth FSA-v2 simulator for closed-loop MPC.

    Attributes
    ----------
    truth_params : dict
        Truth parameters (default: simulation.DEFAULT_PARAMS).
    state : np.ndarray, shape (3,)
        Current latent (B, F, A).
    t_bin : int
        Current global bin index (0 at construction, monotone increasing).
    seed_offset : int
        Base seed; per-channel sampling uses (seed_offset + bin_index +
        per-channel offset) so step-wise composition reproduces a
        single-shot run exactly.
    history : dict
        Accumulated trajectory + per-channel obs + per-bin Φ + C(t),
        appended each `advance` call.
    """

    truth_params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    state: np.ndarray = field(default_factory=lambda: np.array([
        DEFAULT_INIT['B_0'], DEFAULT_INIT['F_0'], DEFAULT_INIT['A_0'],
    ]))
    t_bin: int = 0
    seed_offset: int = 42
    dt: float = DT_BIN_DAYS

    history: dict = field(default_factory=lambda: {
        'trajectory':     [],
        'obs_HR_t_idx':   [], 'obs_HR_value':     [],
        'obs_sleep_label': [],
        'obs_stress_t_idx': [], 'obs_stress_value': [],
        'obs_steps_t_idx':  [], 'obs_steps_value':  [],
        'Phi_value':      [],
        'C_value':        [],
    })

    def advance(self, stride_bins: int,
                 Phi_daily: np.ndarray) -> dict:
        """Advance the plant by `stride_bins` 15-min bins, applying the
        daily Φ array (expanded into morning-loaded sub-bins) and
        returning the obs emitted during this stride.

        Parameters
        ----------
        stride_bins : int
            Number of 15-min outer bins to advance. Typically 48
            (12-hour stride) or 96 (1-day stride) for the E3-E5 demos.
        Phi_daily : np.ndarray, shape (n_days_in_stride,)
            Daily Φ values. Must have `len * 96 ≥ stride_bins`. The
            sub-bin expansion is applied across `len` days; only the
            first `stride_bins` of the resulting per-bin array are used.

        Returns
        -------
        dict
            'trajectory'    : (stride_bins, 3) latent (B, F, A) per bin
            'obs_HR'        : {'t_idx', 'obs_value'} (sleep-gated, indices
                              are GLOBAL bin indices, not stride-local)
            'obs_sleep'     : {'t_idx', 'sleep_label'} (always observed)
            'obs_stress'    : {'t_idx', 'obs_value'} (wake-gated)
            'obs_steps'     : {'t_idx', 'obs_value'} (wake-gated)
            'Phi'           : {'t_idx', 'Phi_value'}
            'C'             : {'t_idx', 'C_value'}
        """
        Phi_daily = np.asarray(Phi_daily, dtype=np.float64)

        # Expand to per-bin (length = len(Phi_daily) * 96), then slice
        Phi_subdaily_full = expand_daily_phi_to_subdaily(Phi_daily)
        if Phi_subdaily_full.shape[0] < stride_bins:
            raise ValueError(
                f"Phi_daily of length {len(Phi_daily)} expands to "
                f"{Phi_subdaily_full.shape[0]} bins but stride_bins="
                f"{stride_bins} requested. Ensure len(Phi_daily) * 96 "
                f"≥ stride_bins."
            )
        Phi_subdaily = Phi_subdaily_full[:stride_bins].astype(np.float32)

        # Global time grid for this stride
        t_grid_global = (np.arange(stride_bins, dtype=np.float64)
                          + self.t_bin) * self.dt

        # Forward Euler-Maruyama with sqrt-Itô diffusion
        rng = np.random.default_rng(self.seed_offset + self.t_bin)
        sigma = np.array([
            self.truth_params['sigma_B'],
            self.truth_params['sigma_F'],
            self.truth_params['sigma_A'],
        ])
        sqrt_dt = math.sqrt(self.dt)
        traj = np.zeros((stride_bins, 3), dtype=np.float32)
        y = self.state.copy()
        aux = (Phi_subdaily,)
        for k in range(stride_bins):
            d_y = drift(t_grid_global[k], y, self.truth_params, aux)
            g   = noise_scale_fn(y, self.truth_params)
            noise = rng.standard_normal(3)
            y = y + self.dt * d_y + sigma * g * sqrt_dt * noise
            y[0] = float(np.clip(y[0], 1e-4, 1.0 - 1e-4))
            y[1] = float(max(y[1], 0.0))
            y[2] = float(max(y[2], 0.0))
            traj[k] = y

        # Update state for next advance call
        self.state = y.copy()

        # Sample obs on this stride. The gen_obs_* functions take a
        # local t_grid (here: stride-local 0..stride_bins-1 in days)
        # but the circadian phase needs the GLOBAL t to preserve
        # phase across strides — pass `t_grid_global` for that.
        sleep_ch = gen_obs_sleep(traj, t_grid_global, self.truth_params,
                                  aux, None, seed=self.seed_offset + self.t_bin + 1)
        prior = {'obs_sleep': sleep_ch}
        hr_ch    = gen_obs_hr   (traj, t_grid_global, self.truth_params, aux,
                                   prior, seed=self.seed_offset + self.t_bin + 2)
        str_ch   = gen_obs_stress(traj, t_grid_global, self.truth_params, aux,
                                   prior, seed=self.seed_offset + self.t_bin + 3)
        steps_ch = gen_obs_steps (traj, t_grid_global, self.truth_params, aux,
                                   prior, seed=self.seed_offset + self.t_bin + 4)

        # Convert local t_idx within each channel to GLOBAL bin indices
        local_t = np.arange(stride_bins, dtype=np.int32)
        global_t = local_t + self.t_bin

        def _shift(ch: dict) -> dict:
            ch_out = dict(ch)
            ch_out['t_idx'] = (np.asarray(ch['t_idx']) + self.t_bin).astype(np.int32)
            return ch_out

        sleep_ch = _shift(sleep_ch)
        hr_ch    = _shift(hr_ch)
        str_ch   = _shift(str_ch)
        steps_ch = _shift(steps_ch)

        # Circadian C(t) on global grid
        phi = float(self.truth_params.get('phi', 0.0))
        C_val = np.cos(2.0 * np.pi * t_grid_global + phi).astype(np.float32)

        # Build canonical channel dicts at global grid
        phi_ch = {'t_idx': global_t, 'Phi_value': Phi_subdaily}
        c_ch   = {'t_idx': global_t, 'C_value':   C_val}

        # Append to history (for finalise)
        self.history['trajectory'].append(traj.copy())
        self.history['Phi_value'].append(Phi_subdaily)
        self.history['C_value'].append(C_val)
        self.history['obs_sleep_label'].append(sleep_ch['sleep_label'])
        self.history['obs_HR_t_idx'].append(hr_ch['t_idx'])
        self.history['obs_HR_value'].append(hr_ch['obs_value'])
        self.history['obs_stress_t_idx'].append(str_ch['t_idx'])
        self.history['obs_stress_value'].append(str_ch['obs_value'])
        self.history['obs_steps_t_idx'].append(steps_ch['t_idx'])
        self.history['obs_steps_value'].append(steps_ch['obs_value'])

        # Advance global bin index
        self.t_bin += stride_bins

        return {
            'trajectory': traj,
            'obs_HR':     hr_ch,
            'obs_sleep':  sleep_ch,
            'obs_stress': str_ch,
            'obs_steps':  steps_ch,
            'Phi':        phi_ch,
            'C':          c_ch,
        }

    def finalise(self, out_dir: str | Path,
                  scenario_name: str = "fsa_high_res_v2_closed_loop") -> Path:
        """Write a psim-format scenario artifact for the full closed-loop
        run. Includes manifest.json + trajectory.npz + obs/*.npz +
        exogenous/*.npz so the closed-loop trajectory can be re-loaded
        and re-validated by psim's consistency checks post-hoc.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate per-stride accumulated history
        trajectory = np.concatenate(self.history['trajectory'], axis=0)
        Phi_value  = np.concatenate(self.history['Phi_value'])
        C_value    = np.concatenate(self.history['C_value'])
        sleep_labels = np.concatenate(self.history['obs_sleep_label'])

        n_bins = trajectory.shape[0]

        # trajectory.npz
        np.savez(out_dir / "trajectory.npz", trajectory=trajectory)

        # obs/
        obs_dir = out_dir / "obs"; obs_dir.mkdir(exist_ok=True)
        np.savez(obs_dir / "obs_sleep.npz",
                  t_idx=np.arange(n_bins, dtype=np.int32),
                  sleep_label=sleep_labels)
        np.savez(obs_dir / "obs_HR.npz",
                  t_idx=np.concatenate(self.history['obs_HR_t_idx']),
                  obs_HR_value=np.concatenate(self.history['obs_HR_value']))
        np.savez(obs_dir / "obs_stress.npz",
                  t_idx=np.concatenate(self.history['obs_stress_t_idx']),
                  obs_stress_value=np.concatenate(self.history['obs_stress_value']))
        np.savez(obs_dir / "obs_steps.npz",
                  t_idx=np.concatenate(self.history['obs_steps_t_idx']),
                  obs_steps_value=np.concatenate(self.history['obs_steps_value']))

        # exogenous/
        exog_dir = out_dir / "exogenous"; exog_dir.mkdir(exist_ok=True)
        np.savez(exog_dir / "Phi.npz",
                  t_idx=np.arange(n_bins, dtype=np.int32),
                  Phi_value=Phi_value)
        np.savez(exog_dir / "C.npz",
                  t_idx=np.arange(n_bins, dtype=np.int32),
                  C_value=C_value)

        # manifest.json
        manifest = {
            "schema_version": "1.0",
            "model_name":     "fsa_high_res_v2",
            "model_version":  "2.0",
            "scenario_name":  scenario_name,
            "truth_params":   {k: float(v) for k, v in self.truth_params.items()},
            "init_state":     {'B_0': float(DEFAULT_INIT['B_0']),
                                'F_0': float(DEFAULT_INIT['F_0']),
                                'A_0': float(DEFAULT_INIT['A_0'])},
            "n_bins_total":   int(n_bins),
            "dt_days":        float(self.dt),
            "bins_per_day":   BINS_PER_DAY,
            "seed":           int(self.seed_offset),
            "state_names":    ["B", "F", "A"],
            "obs_channels":   ["obs_HR", "obs_sleep", "obs_stress", "obs_steps"],
            "exogenous_channels": ["Phi", "C"],
            "validation_summary": {
                "closed_loop": True,
                "stepwise_advances": len(self.history['trajectory']),
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return out_dir
