"""SWAT diagnostic plots — bench-side port of SWAT_model_dev sim_plots.

Adapts the three reference plots that the dev repo emits per scenario
(``outputs/swat/set_<X>_<scen>_14d/{latent_states,observations,entrainment}.png``)
to the bench's 4-state-control representation:

  - dev:   trajectory shape (T_len, 7)  =  (W, Zt, a, T, C, V_h, V_n)
           V_c lives in params (constant per scenario)
  - bench: trajectory shape (T_len, 4)  =  (W, Z, a, T)
           V_h, V_n, V_c are per-bin arrays from the controller

This module accepts the bench shapes directly. The output PNGs are
visually equivalent to the dev's so they can be diffed eyeball-by-eyeball.

Public entry point: ``plot_swat_panels(...)`` — produces all three
files in ``save_dir``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Mapping

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


PHI_0_FROZEN = -math.pi / 3.0


def _sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


# =========================================================================
# PUBLIC ENTRY POINT
# =========================================================================

def plot_swat_panels(
    *,
    trajectory: np.ndarray,         # (T_len, 4): W, Z, a, T
    t_grid_days: np.ndarray,        # (T_len,) in DAYS
    V_h_per_bin: np.ndarray,        # (T_len,)
    V_n_per_bin: np.ndarray,        # (T_len,)
    V_c_per_bin: np.ndarray,        # (T_len,) hours
    obs_HR: Mapping,                # {'t_idx', 'obs_value'}
    obs_sleep: Mapping,             # {'t_idx', 'obs_label'}
    obs_steps: Mapping,             # {'t_idx', 'log_value', 'present_mask'}
    obs_stress: Mapping,            # {'t_idx', 'obs_value'}
    params: Mapping,
    save_dir: str | Path,
    suffix: str = '',
) -> tuple[Path, Path, Path]:
    """Emit ``latent_states.png``, ``observations.png``, ``entrainment.png``.

    Args:
        trajectory: bench's (T_len, 4) state array.
        t_grid_days: per-bin time grid in days (matches the bench's
            ``np.arange(T_len) / BINS_PER_DAY`` convention).
        V_h_per_bin, V_n_per_bin, V_c_per_bin: per-bin controls.
        obs_HR/obs_sleep/obs_steps/obs_stress: bench obs samplers'
            output dicts.
        params: SWAT param dict (TRUTH_PARAMS ∪ obs params).
        save_dir: output directory; created if missing.
        suffix: optional filename suffix (e.g. ``'_mpc'`` to disambiguate
            from the baseline rollout's plots).

    Returns:
        (latents_path, observations_path, entrainment_path).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    p1 = save_dir / f"latent_states{suffix}.png"
    _plot_latent(trajectory, t_grid_days,
                 V_h_per_bin, V_n_per_bin, V_c_per_bin,
                 params, p1)

    p2 = save_dir / f"observations{suffix}.png"
    _plot_observations(trajectory, t_grid_days,
                       V_n_per_bin,
                       obs_HR, obs_sleep, obs_steps, obs_stress,
                       params, p2)

    p3 = save_dir / f"entrainment{suffix}.png"
    _plot_entrainment(trajectory, t_grid_days,
                      V_h_per_bin, V_n_per_bin, V_c_per_bin,
                      params, p3)

    return p1, p2, p3


# =========================================================================
# ENTRAINMENT QUANTITIES
# =========================================================================

def _compute_E_dynamics(trajectory, V_h, V_n, V_c, params):
    """Numpy port of ``_dynamics.entrainment_quality`` driven by per-bin
    V_h, V_n, V_c arrays from the bench (rather than dev's params['V_c']).
    """
    a = trajectory[:, 2]
    T = trajectory[:, 3]
    A_W = params['lambda_amp_W'] * V_h
    A_Z = params['lambda_amp_Z'] * V_h
    B_W = V_n - a + params['alpha_T'] * T
    B_Z = -V_n + params['beta_Z'] * a
    amp_W = _sigmoid(B_W + A_W) - _sigmoid(B_W - A_W)
    amp_Z = _sigmoid(B_Z + A_Z) - _sigmoid(B_Z - A_Z)
    damp = np.exp(-V_n / params['V_n_scale'])
    V_c_max = params.get('V_c_max', 3.0)
    V_c_eff = np.minimum(np.abs(V_c), V_c_max)
    phase = np.cos(np.pi * V_c_eff / (2.0 * V_c_max))
    return damp * amp_W * amp_Z * phase


def _safe_corr(x, y):
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _compute_E_obs(trajectory, t_grid_days, params):
    """Windowed amp × phase-correlation diagnostic (port of dev's _compute_E)."""
    W = trajectory[:, 0]
    Zt = trajectory[:, 1]
    # Reference is the EXTERNAL light cycle (no V_c).
    C = np.sin(2.0 * np.pi * t_grid_days + PHI_0_FROZEN)
    n = len(t_grid_days)
    if n < 3:
        return np.zeros(n)
    dt_days = float(t_grid_days[1] - t_grid_days[0])
    win = max(int(round(1.0 / dt_days)), 3)   # 24h window in bins

    E = np.zeros(n)
    for i in range(n):
        lo = max(i - win + 1, 0)
        W_w = W[lo:i + 1]; Z_w = Zt[lo:i + 1]; C_w = C[lo:i + 1]
        if len(W_w) < 3:
            continue
        amp_W = (W_w.max() - W_w.min()) / 1.0
        amp_Z = (Z_w.max() - Z_w.min()) / 1.0   # Z is in [0, 1] in bench
        phase_W = max(_safe_corr(W_w, C_w),  0.0)
        phase_Z = max(_safe_corr(Z_w, -C_w), 0.0)
        E[i] = (amp_W * phase_W) * (amp_Z * phase_Z)
    return np.clip(E, 0.0, 1.0)


# =========================================================================
# LATENT STATES (6-panel)
# =========================================================================

def _plot_latent(trajectory, t_days, V_h, V_n, V_c, params, save_path):
    fig, axes = plt.subplots(6, 1, figsize=(12, 12), sharex=True)

    # W
    axes[0].plot(t_days, trajectory[:, 0], lw=0.6, color='steelblue')
    axes[0].set_ylabel('W (wakefulness)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, alpha=0.3)

    # Zt
    axes[1].plot(t_days, trajectory[:, 1], lw=0.6, color='indigo')
    c1 = params['c_tilde']
    axes[1].axhline(c1, ls='--', color='red', alpha=0.6,
                    label=f"c_tilde = {c1:.3f}")
    axes[1].set_ylabel('Z (sleep depth)')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # adenosine
    axes[2].plot(t_days, trajectory[:, 2], lw=0.6, color='darkorange')
    axes[2].set_ylabel('a (adenosine)')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    # TESTOSTERONE
    T_amp = trajectory[:, 3]
    axes[3].plot(t_days, T_amp, lw=0.8, color='crimson')
    mu_max = params['mu_E'] * (1.0 - params['E_crit'])
    if mu_max > 0:
        T_star_max = math.sqrt(mu_max / params['eta'])
        axes[3].axhline(T_star_max, ls=':', color='green', alpha=0.5,
                        label=f"T*(E=1) = {T_star_max:.2f}")
    axes[3].axhline(0, ls=':', color='gray', alpha=0.5,
                    label="T*=0 (flatline)")
    axes[3].set_ylabel('T (testosterone)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # External circadian C(t) — analytical (matches what u_W sees minus V_c shift)
    C_t = np.sin(2.0 * np.pi * t_days + PHI_0_FROZEN)
    axes[4].plot(t_days, C_t, lw=0.6, color='seagreen')
    axes[4].set_ylabel('C(t)')
    axes[4].set_ylim(-1.1, 1.1)
    axes[4].grid(True, alpha=0.3)

    # V_h / V_n (and V_c on twin axis since hours have different scale)
    axes[5].plot(t_days, V_h, lw=1.0, color='forestgreen', label='V_h')
    axes[5].plot(t_days, V_n, lw=1.0, color='firebrick',    label='V_n')
    axes[5].set_ylabel('Potentials (V_h / V_n)')
    axes[5].set_xlabel('Time (days)')
    axes[5].grid(True, alpha=0.3)
    ax_vc = axes[5].twinx()
    ax_vc.plot(t_days, V_c, lw=0.8, color='royalblue', alpha=0.6,
               label='V_c (h)')
    ax_vc.set_ylabel('V_c (hours)', color='royalblue')
    ax_vc.tick_params(axis='y', labelcolor='royalblue')
    h1, l1 = axes[5].get_legend_handles_labels()
    h2, l2 = ax_vc.get_legend_handles_labels()
    axes[5].legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8)

    basin = _basin_label(float(V_h[0]), float(V_n[0]), float(V_c[0]))
    fig.suptitle(
        f"SWAT SDE  —  latent states  "
        f"(V_h₀={V_h[0]:.2f}, V_n₀={V_n[0]:.2f}, V_c₀={V_c[0]:+.1f}h  ->  "
        f"{basin};  T_0={T_amp[0]:.2f})",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# OBSERVATIONS (HR / sleep / steps / stress)
# =========================================================================

def _plot_observations(trajectory, t_days, V_n_per_bin,
                       obs_HR, obs_sleep, obs_steps, obs_stress,
                       params, save_path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 2, 2]})
    n = len(t_days)
    W = trajectory[:, 0]
    # External (ground-truth, V_c-independent) circadian overlay.
    C_t_external = np.sin(2.0 * np.pi * t_days + PHI_0_FROZEN)

    # ── Panel 1: HR ──
    hr_idx = np.asarray(obs_HR.get('t_idx', []), dtype=np.int64)
    hr_val = np.asarray(obs_HR.get('obs_value', []))
    hr_pred = params['HR_base'] + params['alpha_HR'] * W
    axes[0].plot(t_days, hr_pred, color='crimson', lw=0.6, alpha=0.6,
                 label='HR mean (from W)')
    if hr_idx.size:
        valid = (hr_idx >= 0) & (hr_idx < n)
        axes[0].scatter(t_days[hr_idx[valid]], hr_val[valid], s=2, alpha=0.35,
                        color='navy',
                        label=f"HR obs (sigma_HR={params['sigma_HR']:.1f})")
    axes[0].set_ylabel('HR (bpm)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # ── Panel 2: 3-level sleep ──
    sl_idx = np.asarray(obs_sleep.get('t_idx', []), dtype=np.int64)
    sl_lvl = np.asarray(obs_sleep.get('obs_label', []), dtype=np.int64)
    has_sleep = False
    if sl_idx.size:
        valid = (sl_idx >= 0) & (sl_idx < n)
        if valid.any():
            axes[1].fill_between(t_days[sl_idx[valid]], 0, sl_lvl[valid],
                                 step='mid', color='midnightblue', alpha=0.7,
                                 label='sleep level')
            has_sleep = True
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(['wake', 'light+rem', 'deep'])
    axes[1].set_ylim(-0.3, 2.3)
    axes[1].set_ylabel('Sleep stage')
    if has_sleep:
        axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # ── Panel 3: steps (log-Gaussian, wake-gated) ──
    st_idx = np.asarray(obs_steps.get('t_idx', []), dtype=np.int64)
    st_log = np.asarray(obs_steps.get('log_value', []))
    st_present = np.asarray(obs_steps.get('present_mask', []))
    if st_idx.size:
        valid = (st_idx >= 0) & (st_idx < n)
        wake = (st_present > 0.5) & valid if st_present.size else valid
        st_counts = np.expm1(st_log)
        axes[2].scatter(t_days[st_idx[wake]], st_counts[wake], s=2.5, alpha=0.5,
                        color='seagreen',
                        label='steps obs (wake bins, log-Gaussian)')
    log_mean = params['mu_step0'] + params['beta_W_steps'] * W
    sigma = params['sigma_step']
    mean_count = np.exp(log_mean + 0.5 * sigma * sigma) - 1.0
    axes[2].plot(t_days, mean_count, lw=0.6, color='darkgreen', alpha=0.6,
                 label='E[steps | W]')
    axes[2].set_ylabel('Steps per bin (wake-gated)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    # ── Panel 4: Garmin stress ──
    sr_idx = np.asarray(obs_stress.get('t_idx', []), dtype=np.int64)
    sr_val = np.asarray(obs_stress.get('obs_value', []))
    sr_pred = (params['s_base']
               + params['alpha_s'] * W
               + params['beta_s'] * V_n_per_bin)
    axes[3].plot(t_days, sr_pred, color='purple', lw=0.6, alpha=0.6,
                 label='stress mean (from W, V_n)')
    if sr_idx.size:
        valid = (sr_idx >= 0) & (sr_idx < n)
        axes[3].scatter(t_days[sr_idx[valid]], sr_val[valid],
                        s=1.5, alpha=0.35, color='darkviolet',
                        label=f"stress obs (sigma_s={params['sigma_s']:.1f})")
    axes[3].set_ylabel('Stress (0-100)')
    axes[3].set_xlabel('Time (days)')
    axes[3].set_ylim(-5, 105)
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    # External C(t) overlay on every panel (twin axis), so each obs
    # channel can be read against time-of-day. V_c-independent — see
    # `simulation.circadian()` in the dev repo for why this is the
    # objective wall-clock reference, not the V_c-shifted subject drive.
    for axi in axes:
        ax_c = axi.twinx()
        ax_c.plot(t_days, C_t_external, color='grey', lw=0.5, alpha=0.4)
        ax_c.set_ylim(-1.15, 1.15)
        ax_c.set_yticks([-1, 0, 1])
        ax_c.tick_params(axis='y', labelcolor='grey', labelsize=7)
        ax_c.set_ylabel('C(t) ext', color='grey', fontsize=8)

    fig.suptitle('SWAT SDE  —  Observations (HR, sleep 3-level, steps, stress)  '
                 'with external C(t) overlay',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# ENTRAINMENT DIAGNOSTIC PLOT (3-panel)
# =========================================================================

def _plot_entrainment(trajectory, t_days, V_h, V_n, V_c, params, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    E_dyn = _compute_E_dynamics(trajectory, V_h, V_n, V_c, params)
    E_obs = _compute_E_obs(trajectory, t_days, params)
    mu = params['mu_E'] * (E_dyn - params['E_crit'])
    E_crit = params['E_crit']

    T_actual = trajectory[:, 3]
    T_star = np.where(mu > 0,
                       np.sqrt(np.maximum(mu, 0.0) / params['eta']),
                       0.0)

    axes[0].plot(t_days, E_dyn, lw=1.0, color='darkviolet',
                 label='E_dyn (drives μ in SDE)')
    axes[0].plot(t_days, E_obs, lw=0.8, ls='--', color='darkorange',
                 label='E_obs (24h windowed, diagnostic)')
    axes[0].axhline(E_crit, ls=':', color='red', alpha=0.7,
                    label=f"E_crit = {E_crit:.2f}")
    axes[0].set_ylabel('E(t)  (entrainment quality)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_days, mu, lw=0.8, color='darkgreen')
    axes[1].axhline(0, ls='--', color='red', alpha=0.6,
                    label='mu = 0 (pitchfork)')
    axes[1].fill_between(t_days, 0, mu, where=(mu > 0),
                         color='green', alpha=0.15,
                         label='mu > 0 (pulsatile)')
    axes[1].fill_between(t_days, 0, mu, where=(mu < 0),
                         color='red', alpha=0.15,
                         label='mu < 0 (flatline)')
    axes[1].set_ylabel('mu(E_dyn)')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_days, T_actual, lw=0.8, color='crimson',
                 label='T (actual)')
    axes[2].plot(t_days, T_star, lw=0.8, ls='--', color='green',
                 label='T* = sqrt(mu/eta) when mu>0 else 0')
    axes[2].set_ylabel('T (pulsatility amplitude)')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylim(bottom=-0.1)
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.3)

    basin = _basin_label(float(V_h[0]), float(V_n[0]), float(V_c[0]))
    fig.suptitle(
        f"SWAT — Entrainment → Bifurcation → Testosterone  "
        f"({basin};  tau_T = {params['tau_T']*24:.1f}h)",
        fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# =========================================================================
# BASIN LABEL
# =========================================================================

def _basin_label(Vh, Vn, V_c=0.0):
    if abs(V_c) >= 2.0:
        if abs(V_c) >= 8.0:
            return f"phase-inverted (V_c={V_c:+.1f}h)"
        return f"phase-shifted (V_c={V_c:+.1f}h)"
    Vh_high = Vh >= 0.6
    Vn_high = Vn >= 1.0
    if Vh_high and not Vn_high:
        return "healthy"
    if not Vh_high and Vn_high:
        return "hyperarousal-insomnia"
    if not Vh_high and not Vn_high:
        return "hypoarousal-hypersomnia"
    return "allostatic overload"
