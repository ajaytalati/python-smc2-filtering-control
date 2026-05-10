"""Plot Stage 2 controller schedule-anchor traces across replans.

Stage 2 has no filter (no SDE-param posterior to trace), but the
controller's HMC posterior over its 16 RBF schedule anchors IS a
posterior trace -- it's the controller's "decision parameters" evolving
across the 14 replans. This is the Stage 2 analogue of Stage 3's
`Stage3_param_traces.png`.

Layout: 16 panels in a 4x4 grid. Top half (panels 0-7) = the 8 Phi_B
anchor params; bottom half (panels 8-15) = the 8 Phi_S anchor params.
Each panel shows the mean theta (across the 256 SMC particles) at each
of the 14 replans, with a reference line at zero.

Also produces an "applied schedule" trace: for each replan, the
daily-mean (Phi_B, Phi_S) of the FIRST DAY of the planned schedule
(which is what actually gets applied before the next replan).

Usage:
    PYTHONPATH=.:.. JAX_PLATFORMS=cpu python tools/plot_stage2_theta_traces.py \
        outputs/fsa_v5/experiments/run16_stage2_soft_fast_overtrained_T14_full_hmc/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    rec = np.load(run_dir / "replan_records.npz", allow_pickle=True)
    mean_thetas = rec["mean_thetas"]  # (n_replans, 2*n_anchors)
    mean_schedules = rec["mean_schedules"]  # (n_replans, n_steps, 2)
    strides = rec["stride"]
    n_replans, theta_dim = mean_thetas.shape
    n_anchors = theta_dim // 2

    # ---- panel 1: 16 theta-anchor traces ---------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(15, 11), sharex=True)
    fig.suptitle(
        f"Stage 2 ({run_dir.name})\n"
        f"controller HMC posterior mean of theta_ctrl across "
        f"{n_replans} replans  (n_anchors={n_anchors}, "
        f"theta_dim={theta_dim})",
        fontsize=10,
    )
    rep_idx = np.arange(n_replans)
    for k in range(theta_dim):
        ax = axes.flat[k]
        ax.plot(rep_idx, mean_thetas[:, k], "-o", color="C0", lw=1.2, ms=4)
        ax.axhline(0.0, color="0.5", lw=0.5, ls=":")
        anchor_idx = k % n_anchors
        dim_label = "Phi_B" if k < n_anchors else "Phi_S"
        ax.set_title(f"theta[{k}]  = {dim_label} anchor #{anchor_idx}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        if k >= 12:
            ax.set_xlabel("replan idx", fontsize=8)
        if k % 4 == 0:
            ax.set_ylabel("mean theta", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = run_dir / "Stage2_theta_traces.png"
    plt.savefig(out1, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out1}")

    # ---- panel 2: applied (Phi_B, Phi_S) daily-mean trace ----------------
    # Each replan plans `n_steps` bins. Take the first day (96 bins) of
    # each plan -- that's the chunk that gets APPLIED before the next
    # replan -- and average it.
    bins_per_day = 96
    n_steps = mean_schedules.shape[1]
    if n_steps < bins_per_day:
        bins_per_day = n_steps
    first_day = mean_schedules[:, :bins_per_day, :]  # (n_replans, 96, 2)
    daily_mean = first_day.mean(axis=1)  # (n_replans, 2)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    axes2[0].plot(rep_idx, daily_mean[:, 0], "-o", color="C0", lw=1.5, ms=5)
    axes2[0].axhline(0.30, color="g", lw=0.8, ls="--",
                     label="healthy island (0.30)")
    axes2[0].set_title("applied Phi_B (daily mean of first-day plan)",
                       fontsize=10)
    axes2[0].set_xlabel("replan idx")
    axes2[0].set_ylabel("Phi_B")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(fontsize=8)
    axes2[1].plot(rep_idx, daily_mean[:, 1], "-o", color="C1", lw=1.5, ms=5)
    axes2[1].axhline(0.30, color="g", lw=0.8, ls="--",
                     label="healthy island (0.30)")
    axes2[1].set_title("applied Phi_S (daily mean of first-day plan)",
                       fontsize=10)
    axes2[1].set_xlabel("replan idx")
    axes2[1].set_ylabel("Phi_S")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(fontsize=8)
    fig2.suptitle(
        f"Stage 2 ({run_dir.name}) -- applied Phi daily mean across "
        f"{n_replans} replans",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out2 = run_dir / "Stage2_applied_phi_trace.png"
    plt.savefig(out2, dpi=110, bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote {out2}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
