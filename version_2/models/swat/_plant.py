"""StepwisePlant — mutable simulator-as-plant for closed-loop SWAT MPC.

Mirrors FSA-v2's `_plant.py` but for the 4-state SWAT model with
**three exogenous controls** (V_h, V_n, V_c) instead of FSA-v2's
single Φ.

Why this exists
---------------
The closed-loop bench advances the plant one stride at a time, with
the controller deciding the **next stride's V_h, V_n, V_c daily
schedules** from the filter's posterior at the current time. We
can't pre-compute the whole horizon because the schedules are chosen
online. The StepwisePlant exposes:

    plant = StepwisePlant(truth_params=..., init_state=..., dt=..., seed=...)
    for window_k in range(n_windows):
        obs_stride = plant.advance(stride_bins,
                                    V_h_daily, V_n_daily, V_c_daily)
        posterior  = filter.update(obs_stride)
        V_h_next, V_n_next, V_c_next = controller.plan(
            posterior, horizon=stride_bins)
        # ... repeat
    plant.finalise(out_dir)

The SDE integration is identical to ``_dynamics.imex_step_substepped``
applied per bin via ``lax.scan`` — Stage-N pattern matching FSA-v2's
GPU plant. Obs samplers stay numpy (cheaper, plus they need integer
sampling for Sleep ordinal / Steps Poisson).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from version_2.models.swat._dynamics import (
    A_SCALE_FROZEN,
    diffusion_state_dep,
    drift_jax,
    state_clip,
)
from version_2.models.swat._v_schedule import (
    BINS_PER_DAY,
    DT_BIN_DAYS,
    expand_three_schedules,
)
from version_2.models.swat.simulation import (
    DEFAULT_INIT,
    DEFAULT_PARAMS,
    gen_obs_hr,
    gen_obs_sleep,
    gen_obs_steps,
    gen_obs_stress,
)


# =========================================================================
# Plant SDE (Stage N): Euler-Maruyama scan on GPU.
# =========================================================================

@jax.jit
def _plant_em_step(
    initial_state,         # (4,) f64 — current [W, Z, a, T]
    u_per_bin,             # (stride_bins, 3) f64 — per-bin (V_h, V_n, V_c)
    p_jax,                 # dict of f64 scalars — truth params
    t_start_days,          # f64 scalar — start of stride in days
    dt,                    # f64 scalar
    rng_key,               # PRNGKey
):
    """Sub-stepped Euler-Maruyama for SWAT 4-state plant.
    Uses 10x sub-stepping to resolve the fast flip-flop.
    """
    n_substeps = 10
    sub_dt = dt / float(n_substeps)
    sqrt_sub_dt = jnp.sqrt(sub_dt)
    stride_bins = u_per_bin.shape[0]

    def step(carry, k):
        y_bin, key = carry
        u_t = u_per_bin[k]
        t_bin = t_start_days + dt * k

        # Sub-stepping within the bin
        def sub_step(sub_carry, k_sub):
            y_curr, k_in = sub_carry
            k_in, sub = jax.random.split(k_in)
            t_sub = t_bin + k_sub * sub_dt
            d_y = drift_jax(y_curr, p_jax, t_sub, u_t)
            sigma_diag = diffusion_state_dep(y_curr, p_jax)
            noise = jax.random.normal(sub, (4,), dtype=jnp.float64)
            y_next = y_curr + sub_dt * d_y + sigma_diag * sqrt_sub_dt * noise
            return (state_clip(y_next), k_in), None

        (y_final, key_final), _ = jax.lax.scan(
            sub_step, (y_bin, key), jnp.arange(n_substeps))
            
        return (y_final, key_final), y_final

    init_carry = (initial_state, rng_key)
    (final_state, _), traj = lax.scan(
        step, init_carry, jnp.arange(stride_bins))
    return final_state, traj


# =========================================================================
# StepwisePlant
# =========================================================================

@dataclass
class StepwisePlant:
    """Mutable ground-truth SWAT simulator for closed-loop MPC.

    Attributes
    ----------
    truth_params : dict
        Truth parameters (default: ``simulation.DEFAULT_PARAMS``).
    state : np.ndarray, shape (4,)
        Current latent (W, Z, a, T).
    t_bin : int
        Current global bin index (0 at construction, monotone
        increasing).
    seed_offset : int
        Base seed; per-channel sampling uses
        (seed_offset + t_bin + per-channel offset) for reproducibility.
    dt : float
        Per-bin time step in days (default: DT_BIN_DAYS).
    history : dict
        Accumulated trajectory + obs + control schedules.
    """

    truth_params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    state: np.ndarray = field(
        default_factory=lambda: DEFAULT_INIT.copy())
    t_bin: int = 0
    seed_offset: int = 42
    dt: float = DT_BIN_DAYS

    history: dict = field(default_factory=lambda: {
        'trajectory':          [],
        'V_h_per_bin':         [],
        'V_n_per_bin':         [],
        'V_c_per_bin':         [],
        'obs_HR_t_idx':        [], 'obs_HR_value':     [],
        'obs_sleep_t_idx':     [], 'obs_sleep_label':  [],
        'obs_steps_t_idx':         [],
        'obs_steps_log_value':     [],
        'obs_steps_present_mask':  [],
        'obs_stress_t_idx':    [], 'obs_stress_value': [],
    })

    # ------------------------------------------------------------------
    def advance(self, stride_bins: int,
                 V_h_daily: np.ndarray,
                 V_n_daily: np.ndarray,
                 V_c_daily: np.ndarray) -> dict:
        """Advance the plant by ``stride_bins`` simulation bins.

        Args:
            stride_bins: number of bins to advance (typically half-day
                or one day at h=1h: 12 or 24).
            V_h_daily, V_n_daily, V_c_daily: each shape ``(n_days,)``,
                must satisfy ``n_days * BINS_PER_DAY >= stride_bins``.
                Each daily value is held piecewise-constant across
                its day.

        Returns:
            dict with global-indexed channel data — same keys as
            FSA-v2's plant for inter-model bench compatibility:
            ``trajectory``, ``obs_HR``, ``obs_sleep``, ``obs_steps``,
            ``obs_stress``, plus ``V_h``, ``V_n``, ``V_c`` (the three
            applied controls per-bin).
        """
        V_h_daily = np.asarray(V_h_daily, dtype=np.float64)
        V_n_daily = np.asarray(V_n_daily, dtype=np.float64)
        V_c_daily = np.asarray(V_c_daily, dtype=np.float64)

        u_full = expand_three_schedules(V_h_daily, V_n_daily, V_c_daily)
        if u_full.shape[0] < stride_bins:
            raise ValueError(
                f"Schedules expand to {u_full.shape[0]} bins, but "
                f"stride_bins={stride_bins} requested. Provide enough "
                f"daily values: n_days * BINS_PER_DAY >= stride_bins.")
        u_per_bin = u_full[:stride_bins].astype(np.float64)

        t_grid_global_days = (np.arange(stride_bins, dtype=np.float64)
                               + self.t_bin) * self.dt

        # Build params dict as JAX-friendly scalars
        p_jax = {k: jnp.asarray(float(v), dtype=jnp.float64)
                  for k, v in self.truth_params.items()
                  if isinstance(v, (int, float))}

        u_jax = jnp.asarray(u_per_bin, dtype=jnp.float64)
        y0_jax = jnp.asarray(self.state, dtype=jnp.float64)
        rng_key = jax.random.PRNGKey(int(self.seed_offset + self.t_bin))

        final_state_jax, traj_jax = _plant_em_step(
            y0_jax, u_jax, p_jax,
            jnp.float64(t_grid_global_days[0]),
            jnp.float64(self.dt), rng_key,
        )
        traj = np.asarray(traj_jax, dtype=np.float64)
        y = np.asarray(final_state_jax, dtype=np.float64)

        # Update state for next advance
        self.state = y.copy()

        # Obs samplers (numpy) — sample on this stride at global time
        local_t = np.arange(stride_bins, dtype=np.int32)
        global_t = local_t + self.t_bin

        # gen_obs_sleep MUST come before gen_obs_steps so the wake-mask
        # is available for steps wake-gating.
        hr_ch = gen_obs_hr(traj, t_grid_global_days, self.truth_params,
                            seed=self.seed_offset + self.t_bin + 1)
        sleep_ch = gen_obs_sleep(traj, t_grid_global_days,
                                   self.truth_params,
                                   seed=self.seed_offset + self.t_bin + 2)
        steps_ch = gen_obs_steps(traj, t_grid_global_days,
                                   self.truth_params,
                                   sleep_label=sleep_ch['obs_label'],
                                   seed=self.seed_offset + self.t_bin + 3)
        stress_ch = gen_obs_stress(traj, t_grid_global_days,
                                    self.truth_params,
                                    V_n_per_bin=u_per_bin[:, 1],
                                    seed=self.seed_offset + self.t_bin + 4)

        # Shift local t_idx to global indices
        def _shift(ch: dict) -> dict:
            ch_out = dict(ch)
            ch_out['t_idx'] = (np.asarray(ch['t_idx']) + self.t_bin
                                ).astype(np.int32)
            return ch_out

        hr_ch     = _shift(hr_ch)
        sleep_ch  = _shift(sleep_ch)
        steps_ch  = _shift(steps_ch)
        stress_ch = _shift(stress_ch)

        # Append to history
        self.history['trajectory'].append(traj.copy())
        self.history['V_h_per_bin'].append(u_per_bin[:, 0].copy())
        self.history['V_n_per_bin'].append(u_per_bin[:, 1].copy())
        self.history['V_c_per_bin'].append(u_per_bin[:, 2].copy())
        self.history['obs_HR_t_idx'].append(hr_ch['t_idx'])
        self.history['obs_HR_value'].append(hr_ch['obs_value'])
        self.history['obs_sleep_t_idx'].append(sleep_ch['t_idx'])
        self.history['obs_sleep_label'].append(sleep_ch['obs_label'])
        self.history['obs_steps_t_idx'].append(steps_ch['t_idx'])
        self.history['obs_steps_log_value'].append(steps_ch['log_value'])
        self.history['obs_steps_present_mask'].append(steps_ch['present_mask'])
        self.history['obs_stress_t_idx'].append(stress_ch['t_idx'])
        self.history['obs_stress_value'].append(stress_ch['obs_value'])

        # Advance global bin index
        self.t_bin += stride_bins

        return {
            'trajectory': traj,
            'obs_HR':     hr_ch,
            'obs_sleep':  sleep_ch,
            'obs_steps':  steps_ch,
            'obs_stress': stress_ch,
            'V_h':        {'t_idx': global_t, 'value': u_per_bin[:, 0]},
            'V_n':        {'t_idx': global_t, 'value': u_per_bin[:, 1]},
            'V_c':        {'t_idx': global_t, 'value': u_per_bin[:, 2]},
        }

    # ------------------------------------------------------------------
    def finalise(self, out_dir: str | Path,
                  scenario_name: str = "swat_closed_loop") -> Path:
        """Write a npz-format closed-loop trajectory artefact.

        Concatenates per-stride accumulated history and saves a single
        compact archive per channel.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.history['trajectory']:
            raise RuntimeError("plant.finalise() called before any advance()")

        trajectory = np.concatenate(self.history['trajectory'], axis=0)
        V_h = np.concatenate(self.history['V_h_per_bin'])
        V_n = np.concatenate(self.history['V_n_per_bin'])
        V_c = np.concatenate(self.history['V_c_per_bin'])

        np.savez(out_dir / "trajectory.npz",
                  trajectory=trajectory,
                  V_h_per_bin=V_h, V_n_per_bin=V_n, V_c_per_bin=V_c)

        obs_dir = out_dir / "obs"
        obs_dir.mkdir(exist_ok=True)
        np.savez(obs_dir / "obs_HR.npz",
                  t_idx=np.concatenate(self.history['obs_HR_t_idx']),
                  obs_value=np.concatenate(self.history['obs_HR_value']))
        np.savez(obs_dir / "obs_sleep.npz",
                  t_idx=np.concatenate(self.history['obs_sleep_t_idx']),
                  obs_label=np.concatenate(self.history['obs_sleep_label']))
        np.savez(obs_dir / "obs_steps.npz",
                  t_idx=np.concatenate(self.history['obs_steps_t_idx']),
                  log_value=np.concatenate(self.history['obs_steps_log_value']),
                  present_mask=np.concatenate(self.history['obs_steps_present_mask']))
        np.savez(obs_dir / "obs_stress.npz",
                  t_idx=np.concatenate(self.history['obs_stress_t_idx']),
                  obs_value=np.concatenate(self.history['obs_stress_value']))

        return out_dir
