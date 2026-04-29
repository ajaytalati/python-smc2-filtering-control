"""models/bistable_controlled/sim_plots.py — Controlled Double-Well Plots.

Date:    18 April 2026
Version: 1.0

Produces:
  1. latent_states.png  — two-panel plot:
        top:    x(t) with wells at +/-a, barrier at 0, intervention
                windows shaded
        bottom: u(t) (the tilt/barrier process) with u_target(t)
                overlay and critical tilt u_c marked
  2. observations.png   — observed y vs true x
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_bistable_ctrl(trajectory, t_grid, channel_outputs, params, save_dir):
    """Generate diagnostic plots for the controlled-bistable model."""
    os.makedirs(save_dir, exist_ok=True)

    # Extract u_target schedule from the channel output
    u_tgt_channel = channel_outputs.get('u_target', {})
    u_target_values = u_tgt_channel.get('u_target_value', None)

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent(trajectory, t_grid, u_target_values, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p2)
    print(f"  Plot: {p2}")


def _plot_latent(trajectory, t_grid, u_target_values, params, save_path):
    """Two-panel plot: x(t) above, u(t) below."""
    fig, (ax_x, ax_u) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    a = params['a']
    u_c = 2.0 * params['alpha'] * a**3 / (3.0 * math.sqrt(3.0))

    # --- Top panel: x trajectory ---
    ax_x.plot(t_grid, trajectory[:, 0], linewidth=0.6, alpha=0.9,
              color='steelblue')
    ax_x.axhline(+a, color='forestgreen', linestyle='--', alpha=0.6,
                 label=f'healthy well: x = +{a}')
    ax_x.axhline(-a, color='firebrick', linestyle='--', alpha=0.6,
                 label=f'unhealthy well: x = -{a}')
    ax_x.axhline(0.0, color='grey', linestyle=':', alpha=0.5,
                 label='barrier: x = 0')

    # Shade the active-intervention window if we have the schedule
    if u_target_values is not None:
        u_tgt = np.asarray(u_target_values)
        # Find contiguous regions where u_target > 0 -- shade them lightly,
        # darker for supercritical regions (u_target > u_c).
        for ax_to_shade in (ax_x, ax_u):
            _shade_intervention(ax_to_shade, t_grid, u_tgt, u_c)

    ax_x.set_ylabel('x (health state)')
    ax_x.legend(loc='upper right', fontsize=8, ncol=3)
    ax_x.grid(True, alpha=0.3)
    ax_x.set_title('Controlled Double-Well — Health State  x(t)',
                   fontsize=12)

    # --- Bottom panel: u trajectory + u_target overlay ---
    ax_u.plot(t_grid, trajectory[:, 1], linewidth=0.8, alpha=0.9,
              color='darkorange', label='u (tilt process)')
    if u_target_values is not None:
        ax_u.plot(t_grid, u_target_values, color='black', linewidth=1.3,
                  alpha=0.7, drawstyle='steps-post',
                  label='u_target (schedule)')
    ax_u.axhline(+u_c, color='purple', linestyle=':', alpha=0.7,
                 label=f'u_c = {u_c:.3f} (critical tilt)')
    ax_u.axhline(-u_c, color='purple', linestyle=':', alpha=0.7)
    ax_u.axhline(0.0, color='grey', linestyle='-', alpha=0.3)
    ax_u.set_xlabel('Time (hours)')
    ax_u.set_ylabel('u (control / barrier process)')
    ax_u.legend(loc='upper right', fontsize=8)
    ax_u.grid(True, alpha=0.3)
    ax_u.set_title('Controlled Double-Well — Control Process  u(t)',
                   fontsize=12)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _shade_intervention(ax, t_grid, u_tgt, u_c):
    """Shade intervention windows on an axis.  Darker for supercritical."""
    # Subcritical active intervention (0 < u_target <= u_c)
    mask_sub = (u_tgt > 0) & (u_tgt <= u_c)
    # Supercritical active intervention (u_target > u_c)
    mask_sup = (u_tgt > u_c)

    _shade_contiguous(ax, t_grid, mask_sub, color='gold',   alpha=0.12)
    _shade_contiguous(ax, t_grid, mask_sup, color='tomato', alpha=0.18)


def _shade_contiguous(ax, t, mask, color, alpha):
    """Shade every contiguous True-region of `mask` on axis `ax`."""
    if not np.any(mask):
        return
    # Find rising and falling edges of the boolean mask
    edges = np.diff(mask.astype(np.int8), prepend=0, append=0)
    starts = np.where(edges ==  1)[0]
    ends   = np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        t_lo = t[s]
        # e indexes the first False after the True run; clip to valid range
        t_hi = t[min(e, len(t) - 1)] if e < len(t) else t[-1]
        ax.axvspan(t_lo, t_hi, color=color, alpha=alpha, zorder=0)


def _plot_observations(trajectory, t_grid, channel_outputs, params, save_path):
    """Overlay observed y on true x."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    obs = channel_outputs.get('obs', {})
    t_idx = obs.get('t_idx', np.arange(len(t_grid)))
    y = obs.get('obs_value', np.zeros(len(t_grid)))

    # Shade intervention windows for context
    u_tgt_ch = channel_outputs.get('u_target', {})
    u_tgt = u_tgt_ch.get('u_target_value', None)
    if u_tgt is not None:
        u_c = 2.0 * params['alpha'] * params['a']**3 / (3.0 * math.sqrt(3.0))
        _shade_intervention(ax, t_grid, np.asarray(u_tgt), u_c)

    ax.plot(t_grid, trajectory[:, 0], linewidth=0.8, alpha=0.7,
            label='True x', color='blue')
    ax.scatter(t_grid[t_idx], y, s=3, alpha=0.4, color='orange',
               label=f'Observed y (σ_obs={params["sigma_obs"]:.2f})', zorder=2)
    ax.axhline(+params['a'], color='forestgreen', linestyle='--', alpha=0.5)
    ax.axhline(-params['a'], color='firebrick',   linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle('Controlled Double-Well — Observations', fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
