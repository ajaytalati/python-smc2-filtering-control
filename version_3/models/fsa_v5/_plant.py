"""StepwisePlant — mutable 6D FSA-v5 simulator-as-plant for closed-loop MPC.

This is the ground-truth simulator the closed-loop MPC controller runs
against during testing with synthetic feedback. It exposes a one-stride-
at-a-time interface so a controller can decide each next stride's $\\Phi$
schedule from the current filter posterior, advance the plant, observe,
and re-plan — exactly the receding-horizon protocol described in
LaTeX §6 (Open-Loop Limitation).

Why this exists (vs ``simulator.synthesise_scenario``)
-------------------------------------------------------
``simulator.synthesise_scenario`` takes the **complete** exogenous
schedule up-front (a full ``Phi_arr``). For closed-loop MPC the
controller decides the next stride's $\\Phi$ from the filter posterior
at the current time — we cannot pre-compute the whole horizon. The
``StepwisePlant`` exposes a per-stride API:

    plant = StepwisePlant(truth_params=TRUTH_PARAMS_V5)
    for k in range(n_windows):
        obs = plant.advance(stride_bins, Phi_daily_for_stride)
        posterior = filter.update(obs)
        Phi_next  = controller.plan(posterior, horizon=stride_bins)
    plant.finalise(out_dir)   # optional psim-format archival artifact

What changed in the FSA-v5 promotion
-------------------------------------
The previous (v2) version of this file held a 3D state ``(B, F, A)`` and
hard-coded a 3-channel diffusion. v5 uses the full 6D state
``(B, S, F, A, K_FB, K_FS)`` and the state-dependent diffusion structure
of ``models/fsa_high_res/_dynamics.py:diffusion_state_dep``. The drift
is delegated entirely to ``_dynamics.drift_jax`` — single source of
truth — which automatically picks up the v5 Hill-deconditioning term
(LaTeX §10.2) when the truth params dict contains ``mu_dec_* > 0``.

Single source of truth
----------------------
Drift  → ``_dynamics.drift_jax`` (NEVER inline)
Diffusion structure → mirrors ``_dynamics.diffusion_state_dep``
Initial state → ``simulation.DEFAULT_INIT`` (6D dict)
Truth params  → use ``_dynamics.TRUTH_PARAMS_V5`` for closed-island,
              or ``simulation.DEFAULT_PARAMS`` for v4 numerics

LaTeX cross-references
----------------------
- 6D state spec   → §11.1
- v5 deconditioning → §10
- Closed-loop motivation → §6 (open-loop limitation argument)
- SMC$^2$ controller integration → §11–§12
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from version_3.models.fsa_v5.simulation import (
    BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS,
    DEFAULT_PARAMS, DEFAULT_INIT,
    drift, noise_scale_fn,
    gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps, gen_obs_volumeload,
    gen_Phi_channel, gen_C_channel, circadian,
    EPS_A_FROZEN, EPS_B_FROZEN, EPS_S_FROZEN,
)
from version_3.models.fsa_v5._dynamics import drift_jax as _drift_jax_canonical
from version_3.models.fsa_v5._phi_burst import expand_daily_phi_to_subdaily


# ===========================================================================
# Inner integrator: one stride of forward Euler--Maruyama on GPU.
# ===========================================================================
# 6D state, state-dependent diffusion, no sub-stepping (one EM step per
# 15-min bin). Mirrors the production protocol described in LaTeX §11.3
# and matches the FIM-analysis integration grid.

@jax.jit
def _plant_em_step(
    initial_state,         # (6,) f64 — current [B, S, F, A, K_FB, K_FS]
    Phi_subdaily,          # (stride_bins, 2) f64 — per-bin (Phi_B, Phi_S)
    p_jax,                 # dict of f64 scalars — truth params (must include v5 Hill keys)
    sigma_diag,            # (6,) f64 — per-state diffusion scales
    dt,                    # f64 scalar — bin width (=DT_BIN_DAYS)
    rng_key,               # PRNGKey
):
    """Forward Euler--Maruyama with state-dependent sqrt-Itô diffusion.

    Each bin: one drift step (delegated to ``_drift_jax_canonical``) plus
    one diffusion step (sqrt(state-clipped) * noise, see §11.1
    diffusion paragraph). State is reflected/clipped back into its
    physical domain after each bin.

    Returns:
        ``(final_state, traj)`` where traj is shape (stride_bins, 6).
    """
    sqrt_dt = jnp.sqrt(dt)
    stride_bins = Phi_subdaily.shape[0]
    EPS_B = jnp.float64(EPS_B_FROZEN)
    EPS_S = jnp.float64(EPS_S_FROZEN)
    EPS_A = jnp.float64(EPS_A_FROZEN)

    def step(carry, k):
        y, key = carry
        key, sub = jax.random.split(key)
        Phi_t = Phi_subdaily[k]   # shape (2,)

        # Drift via canonical implementation — picks up v5 Hill term
        # automatically when p_jax has nonzero mu_dec_B / mu_dec_S.
        d_y = _drift_jax_canonical(y, p_jax, Phi_t)

        # State-dependent diffusion scales (mirrors _dynamics.diffusion_state_dep).
        # Jacobi-style for B, S; CIR-style for F, A, K_FB, K_FS. Clipping
        # happens before the sqrt to avoid NaN at boundaries.
        B_cl = jnp.clip(y[0], EPS_B, 1.0 - EPS_B)
        S_cl = jnp.clip(y[1], EPS_S, 1.0 - EPS_S)
        F_cl = jnp.maximum(y[2], 0.0)
        A_cl = jnp.maximum(y[3], 0.0)
        KFB_cl = jnp.maximum(y[4], 0.0)
        KFS_cl = jnp.maximum(y[5], 0.0)
        g = jnp.array([
            jnp.sqrt(B_cl * (1.0 - B_cl)),
            jnp.sqrt(S_cl * (1.0 - S_cl)),
            jnp.sqrt(F_cl),
            jnp.sqrt(A_cl + EPS_A),     # +EPS_A: floor for stability
            jnp.sqrt(KFB_cl),
            jnp.sqrt(KFS_cl),
        ])

        noise = jax.random.normal(sub, (6,), dtype=jnp.float64)
        y_new = y + dt * d_y + sigma_diag * g * sqrt_dt * noise

        # Boundary handling — same as in _dynamics.imex_step_substepped.
        # Reflection in [0, 1] for B, S (Jacobi); abs() for F, A, K_*.
        y_new = y_new.at[0].set(jnp.clip(y_new[0], EPS_B, 1.0 - EPS_B))
        y_new = y_new.at[1].set(jnp.clip(y_new[1], EPS_S, 1.0 - EPS_S))
        y_new = y_new.at[2].set(jnp.maximum(y_new[2], 0.0))
        y_new = y_new.at[3].set(jnp.maximum(y_new[3], 0.0))
        y_new = y_new.at[4].set(jnp.maximum(y_new[4], 0.0))
        y_new = y_new.at[5].set(jnp.maximum(y_new[5], 0.0))
        return (y_new, key), y_new

    init_carry = (initial_state, rng_key)
    (final_state, _), traj = lax.scan(step, init_carry, jnp.arange(stride_bins))
    return final_state, traj


# ===========================================================================
# StepwisePlant — the public API
# ===========================================================================

@dataclass
class StepwisePlant:
    """Mutable ground-truth FSA-v5 simulator for closed-loop MPC.

    Attributes
    ----------
    truth_params : dict
        Truth parameters (default: ``simulation.DEFAULT_PARAMS`` which has
        ``mu_dec_* = 0`` and is therefore numerically v4). For the v5
        closed-island regime, pass in
        ``models.fsa_high_res._dynamics.TRUTH_PARAMS_V5``.
    state : np.ndarray, shape (6,)
        Current 6D latent ``(B, S, F, A, K_FB, K_FS)``.
    t_bin : int
        Current global bin index (0 at construction, monotone increasing).
    seed_offset : int
        Base PRNG seed; per-channel sampling uses
        ``(seed_offset + t_bin + per-channel-offset)`` so step-wise
        composition reproduces a single-shot run exactly.
    dt : float
        Bin width in days (default = 1/96 ≈ 15 min).
    history : dict
        Accumulated 6D trajectory + per-channel obs + per-bin Phi + C(t).
        Appended each ``advance`` call. Used by ``finalise()``.
    """

    truth_params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    state: np.ndarray = field(default_factory=lambda: np.array([
        DEFAULT_INIT['B_0'],   DEFAULT_INIT['S_0'],
        DEFAULT_INIT['F_0'],   DEFAULT_INIT['A_0'],
        DEFAULT_INIT['KFB_0'], DEFAULT_INIT['KFS_0'],
    ]))
    t_bin: int = 0
    seed_offset: int = 42
    dt: float = DT_BIN_DAYS

    history: dict = field(default_factory=lambda: {
        'trajectory':            [],
        'obs_HR_t_idx':          [], 'obs_HR_value':         [],
        'obs_sleep_label':       [],
        'obs_stress_t_idx':      [], 'obs_stress_value':     [],
        'obs_steps_t_idx':       [], 'obs_steps_value':      [],
        'obs_volumeload_t_idx':  [], 'obs_volumeload_value': [],
        'Phi_value':             [],
        'C_value':               [],
    })

    def advance(self, stride_bins: int, Phi_daily: np.ndarray) -> dict:
        """Advance the plant by ``stride_bins`` 15-min bins under the
        supplied per-day stimulus schedule, sampling all 5 obs channels.

        Parameters
        ----------
        stride_bins : int
            Number of 15-min bins to advance. Typical values: 96 (1 day),
            48 (12 hours).
        Phi_daily : np.ndarray
            Either shape ``(n_days_in_stride, 2)`` for bimodal
            ``(Phi_B, Phi_S)`` per day, or shape ``(n_days_in_stride,)``
            for an aerobic-only legacy schedule (auto-broadcast as
            ``(Phi, 0)`` for back-compat with v2/v3 callers). Must have
            enough days to cover the stride after sub-bin expansion.

        Returns
        -------
        dict
            Per-channel observations on this stride, with global bin
            indices. Keys: ``trajectory`` (shape (stride_bins, 6)),
            ``obs_HR``, ``obs_sleep``, ``obs_stress``, ``obs_steps``,
            ``obs_volumeload``, ``Phi``, ``C``.
        """
        Phi_daily = np.asarray(Phi_daily, dtype=np.float64)

        # Back-compat: if caller passed a 1D aerobic-only schedule, lift
        # it to bimodal (aerobic-only) by appending a zero strength column.
        if Phi_daily.ndim == 1:
            Phi_daily = np.stack([Phi_daily, np.zeros_like(Phi_daily)], axis=-1)
        assert Phi_daily.ndim == 2 and Phi_daily.shape[-1] == 2, (
            f"Phi_daily must have shape (n_days, 2); got {Phi_daily.shape}")

        # Expand daily Phi → per-bin Phi (one column at a time, then stack).
        Phi_B_daily, Phi_S_daily = Phi_daily[:, 0], Phi_daily[:, 1]
        Phi_B_subdaily = expand_daily_phi_to_subdaily(Phi_B_daily)
        Phi_S_subdaily = expand_daily_phi_to_subdaily(Phi_S_daily)
        if Phi_B_subdaily.shape[0] < stride_bins:
            raise ValueError(
                f"Phi_daily of length {len(Phi_daily)} expands to "
                f"{Phi_B_subdaily.shape[0]} bins but stride_bins="
                f"{stride_bins} requested. Ensure len(Phi_daily) * "
                f"{BINS_PER_DAY} ≥ stride_bins.")
        Phi_subdaily = np.stack(
            [Phi_B_subdaily[:stride_bins], Phi_S_subdaily[:stride_bins]],
            axis=-1).astype(np.float32)

        # Global time grid for this stride (in days)
        t_grid_global = (np.arange(stride_bins, dtype=np.float64) + self.t_bin) * self.dt

        # ── Forward Euler-Maruyama on GPU via _plant_em_step ──
        # 6D diffusion vector (matches _dynamics.diffusion_state_dep ordering).
        #
        # IMPORTANT: We do NOT read these from ``self.truth_params`` because
        # ``DEFAULT_PARAMS`` (and ``DEFAULT_PARAMS_V5``) suffer from a key
        # collision: ``sigma_S`` is used for both the latent-S diffusion
        # scale (~0.008) and the stress-channel observation noise (~4.0),
        # and Python takes whichever was assigned last (the obs one). Reading
        # ``self.truth_params['sigma_S']`` would therefore put 4.0 into the
        # latent diffusion and blow S up by ~500x. The production
        # ``estimation.py`` sidesteps this by hard-coding the same
        # ``SIGMA_*_FROZEN`` constants — we mirror that policy here.
        sigma = np.array([
            0.010,    # sigma_B   — Jacobi diffusion scale for aerobic fitness B
            0.008,    # sigma_S   — Jacobi diffusion scale for strength S
            0.012,    # sigma_F   — CIR diffusion scale for unified fatigue F
            0.020,    # sigma_A   — CIR diffusion scale for autonomic A
            0.005,    # sigma_K   — shared CIR scale for K_FB
            0.005,    # sigma_K   — shared CIR scale for K_FS
        ])
        # JIT-friendly params dict — only the dynamics keys; the obs-coef
        # keys aren't used by the drift, but pass-through is harmless.
        p_jax = {k: jnp.asarray(float(v), dtype=jnp.float64)
                  for k, v in self.truth_params.items()}
        sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)
        Phi_jax   = jnp.asarray(Phi_subdaily, dtype=jnp.float64)
        y0_jax    = jnp.asarray(self.state, dtype=jnp.float64)
        rng_key   = jax.random.PRNGKey(int(self.seed_offset + self.t_bin))

        final_state_jax, traj_jax = _plant_em_step(
            y0_jax, Phi_jax, p_jax, sigma_jax,
            jnp.float64(self.dt), rng_key,
        )
        # Pull back to numpy for obs samplers + history bookkeeping
        traj = np.asarray(traj_jax, dtype=np.float32)
        y    = np.asarray(final_state_jax, dtype=np.float64)

        # Update state for next advance call
        self.state = y.copy()

        # Obs samplers expect aux = (Phi_arr,) and read Phi[k] for the
        # bin's stimulus. Same as v2/v3 contract; no change needed.
        aux = (Phi_subdaily,)

        # Sample all 5 obs channels. Sleep first because HR / Stress /
        # Steps depend on the sleep label. VL gets sleep too (for its
        # 1-per-day-during-wake gating).
        sleep_ch = gen_obs_sleep(
            traj, t_grid_global, self.truth_params, aux,
            None, seed=self.seed_offset + self.t_bin + 1)
        prior = {'obs_sleep': sleep_ch}
        hr_ch    = gen_obs_hr(
            traj, t_grid_global, self.truth_params, aux,
            prior, seed=self.seed_offset + self.t_bin + 2)
        str_ch   = gen_obs_stress(
            traj, t_grid_global, self.truth_params, aux,
            prior, seed=self.seed_offset + self.t_bin + 3)
        steps_ch = gen_obs_steps(
            traj, t_grid_global, self.truth_params, aux,
            prior, seed=self.seed_offset + self.t_bin + 4)
        vl_ch    = gen_obs_volumeload(
            traj, t_grid_global, self.truth_params, aux,
            prior, seed=self.seed_offset + self.t_bin + 5)

        # Convert local t_idx within each channel to GLOBAL bin indices.
        local_t  = np.arange(stride_bins, dtype=np.int32)
        global_t = local_t + self.t_bin

        def _shift(ch: dict) -> dict:
            ch_out = dict(ch)
            ch_out['t_idx'] = (np.asarray(ch['t_idx']) + self.t_bin).astype(np.int32)
            return ch_out

        sleep_ch = _shift(sleep_ch)
        hr_ch    = _shift(hr_ch)
        str_ch   = _shift(str_ch)
        steps_ch = _shift(steps_ch)
        vl_ch    = _shift(vl_ch)

        # Circadian C(t) on the global grid — used by the obs LL inside the filter.
        phi = float(self.truth_params.get('phi', 0.0))
        C_val = np.cos(2.0 * np.pi * t_grid_global + phi).astype(np.float32)

        # Canonical channel dicts at the global grid
        phi_ch = {'t_idx': global_t, 'Phi_value': Phi_subdaily}
        c_ch   = {'t_idx': global_t, 'C_value':   C_val}

        # Append to history (used by finalise() for psim-format archival)
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
        self.history['obs_volumeload_t_idx'].append(vl_ch['t_idx'])
        self.history['obs_volumeload_value'].append(vl_ch['obs_value'])

        # Advance global bin index
        self.t_bin += stride_bins

        return {
            'trajectory':     traj,
            'obs_HR':         hr_ch,
            'obs_sleep':      sleep_ch,
            'obs_stress':     str_ch,
            'obs_steps':      steps_ch,
            'obs_volumeload': vl_ch,
            'Phi':            phi_ch,
            'C':              c_ch,
        }

    def finalise(self, out_dir: str | Path,
                  scenario_name: str = "fsa_high_res_v5_closed_loop") -> Path:
        """Write a psim-format scenario artifact for the full closed-loop run.

        Includes ``manifest.json``, ``trajectory.npz``, ``obs/*.npz``, and
        ``exogenous/*.npz`` so the trajectory can be re-loaded and
        re-validated by psim's consistency checks post-hoc.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate per-stride accumulated history
        trajectory   = np.concatenate(self.history['trajectory'], axis=0)
        Phi_value    = np.concatenate(self.history['Phi_value'])
        C_value      = np.concatenate(self.history['C_value'])
        sleep_labels = np.concatenate(self.history['obs_sleep_label'])

        n_bins = trajectory.shape[0]

        # trajectory.npz (6 columns: B, S, F, A, K_FB, K_FS)
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
        np.savez(obs_dir / "obs_volumeload.npz",
                  t_idx=np.concatenate(self.history['obs_volumeload_t_idx']),
                  obs_volumeload_value=np.concatenate(self.history['obs_volumeload_value']))

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
            "schema_version":     "1.0",
            "model_name":         "fsa_high_res_v5",
            "model_version":      "5.0",
            "scenario_name":      scenario_name,
            "truth_params":       {k: float(v) for k, v in self.truth_params.items()},
            "init_state":         {k: float(v) for k, v in DEFAULT_INIT.items()},
            "n_bins_total":       int(n_bins),
            "dt_days":            float(self.dt),
            "bins_per_day":       BINS_PER_DAY,
            "seed":               int(self.seed_offset),
            "state_names":        ["B", "S", "F", "A", "KFB", "KFS"],
            "obs_channels":       ["obs_HR", "obs_sleep", "obs_stress",
                                    "obs_steps", "obs_volumeload"],
            "exogenous_channels": ["Phi", "C"],
            "validation_summary": {
                "closed_loop":         True,
                "stepwise_advances":   len(self.history['trajectory']),
            },
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return out_dir
