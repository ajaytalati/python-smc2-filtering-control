"""Generate two figures comparing LQG and SMC^2 MPC against the
constant-Phi=1 baseline. Numbers from PROGRESS_HI.md (Stage H/I).

Outputs:
  figures/fig_t14_comparison.pdf  -- bar chart at T=14
  figures/fig_lqg_horizon.pdf     -- LQG ratio and F-violation vs horizon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ---- Figure 1: T=14 head-to-head ---------------------------------
fig, ax = plt.subplots(figsize=(5.5, 3.4))

controllers = ["Constant\nΦ ≡ 1", "LQG\n(open-loop)", "SMC$^2$ MPC\n(closed-loop)"]
mean_A = [0.0824, 0.0830, 0.0954]
ratios = [1.000, 1.008, 1.157]
colors = ["lightgrey", "C1", "C0"]

bars = ax.bar(controllers, mean_A, color=colors, edgecolor="black", lw=0.8)
for bar, r, mA in zip(bars, ratios, mean_A):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            h + 0.0015,
            f"$\\bar A = {mA:.4f}$\nratio $= {r:.3f}$",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel(r"mean autonomic amplitude $\bar A$ over $T = 14$ days")
ax.set_ylim(0, 0.115)
ax.set_title("Closed-loop performance at $T = 14$ days, FSA-v2 plant")
# Wall-clock annotations along bottom
wallclock = ["< 1 s", "1.3 s", "124 min"]
for i, wc in enumerate(wallclock):
    ax.text(i, -0.008, "wall-clock: " + wc,
            ha="center", va="top", fontsize=8, color="dimgrey", style="italic")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("/home/ajay/Repos/python-smc2-filtering-control-master/LaTex_docs/figures/fig_t14_comparison.pdf",
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_t14_comparison.pdf")


# ---- Figure 2: LQG ratio across horizons -------------------------
T_grid = np.array([14, 28, 42, 56, 84])
ratio_h15  = np.array([1.008, 1.002, 1.039, 1.182, 0.958])
ratio_h60  = np.array([1.008, 1.002, 1.038, 1.155, 0.944])
fviol_h15  = np.array([0.0, 0.0, 0.0, 0.0, 10.23])  # %
mpc_T_known = np.array([14])
mpc_ratio_known = np.array([1.157])

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4),
                          gridspec_kw=dict(width_ratios=[1.2, 1.0]))
ax_r, ax_f = axes

ax_r.axhline(1.0, color="grey", lw=0.7, ls="--",
             label=r"constant baseline ($\Phi \equiv 1$)")
ax_r.plot(T_grid, ratio_h15, "o-", color="C1", lw=1.6, ms=6,
          label=r"LQG, $h{=}15$ min")
ax_r.plot(T_grid, ratio_h60, "s--", color="C1", lw=1.0, ms=5, alpha=0.6,
          label=r"LQG, $h{=}60$ min")
ax_r.plot(mpc_T_known, mpc_ratio_known, "^", color="C0", ms=10,
          label=r"SMC$^2$ MPC ($T{=}14$ checkpoint only)")
# annotate bifurcation
ax_r.fill_between([56, 90], 0.9, 1.3, color="C3", alpha=0.10)
ax_r.text(70, 1.22, "linearisation\nbreaks", ha="center", color="C3",
          fontsize=9, alpha=0.9)
ax_r.set_xlabel(r"horizon $T$ (days)")
ax_r.set_ylabel(r"ratio of mean $A$ over constant baseline")
ax_r.set_title("(a)  LQG vs constant baseline across horizons")
ax_r.set_xlim(10, 90)
ax_r.set_ylim(0.9, 1.25)
ax_r.set_xticks(T_grid)
ax_r.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
ax_r.grid(True, alpha=0.3)

# F-violation vs T
ax_f.axhline(5.0, color="C3", lw=0.7, ls="--", label=r"acceptance gate $5\%$")
ax_f.bar(T_grid - 1.5, fviol_h15, width=3, color="C1",
         edgecolor="black", lw=0.6, label=r"LQG $h{=}15$ min")
ax_f.set_xlabel(r"horizon $T$ (days)")
ax_f.set_ylabel(r"F-violation fraction (\%)")
ax_f.set_title("(b)  Soft-barrier violation")
ax_f.set_xlim(10, 90)
ax_f.set_ylim(0, 12)
ax_f.set_xticks(T_grid)
ax_f.legend(loc="upper left", framealpha=0.95, fontsize=8.5)
ax_f.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("/home/ajay/Repos/python-smc2-filtering-control-master/LaTex_docs/figures/fig_lqg_horizon.pdf",
            bbox_inches="tight")
plt.close(fig)
print("Saved fig_lqg_horizon.pdf")
