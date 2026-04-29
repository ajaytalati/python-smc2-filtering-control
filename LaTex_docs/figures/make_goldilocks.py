"""Generate the Goldilocks profile figure for Section 4 of the lecture notes.

Plots the deterministic-skeleton growth rate mu(Phi_c) and the
oscillatory amplitude A*_osc(Phi_c) = sqrt(mu/eta) as a function of
constant training load Phi_c, at the G1-reparametrized truth values.

Output: figures/fig_goldilocks.pdf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Truth values (G1-reparametrized; see Table 1 of the notes)
mu_0 = 0.036
mu_B = 0.30
mu_F = 0.26
mu_FF = 0.40
F_typ = 0.20
A_typ = 0.10
eta = 0.20

kappa_B = 0.01248
tau_B = 42.0
eps_A = 0.40
kappa_F = 0.030
tau_F_eff = 6.364  # G1: 7/(1+1*0.1)
lam_A = 1.00


def equilibria(Phi_c):
    """Sedentary-branch (B*, F*) at constant Phi_c.

    B saturates at 1 once kappa_B*tau_B*Phi/(1+eps*A_typ) > 1.
    """
    B_unclamped = kappa_B * tau_B * Phi_c / (1.0 + eps_A * A_typ)
    B_star = np.minimum(B_unclamped, 1.0)
    F_star = kappa_F * tau_F_eff * (1.0 + lam_A * A_typ) * Phi_c
    return B_star, F_star


def mu_of_Phi(Phi_c):
    B, F = equilibria(Phi_c)
    return mu_0 + mu_B * B - mu_F * F - mu_FF * (F - F_typ) ** 2


# Grid of training loads
Phi = np.linspace(0.0, 5.0, 1001)
mu_vals = mu_of_Phi(Phi)
# Oscillatory amplitude where mu > 0; zero (= sedentary stable) elsewhere
A_osc = np.where(mu_vals > 0, np.sqrt(np.maximum(mu_vals, 0) / eta), 0.0)

# Critical points
from scipy.optimize import brentq
Phi_crit = brentq(mu_of_Phi, 3.0, 5.0)
# Phi_opt = argmax mu(Phi); maximum is at the saturation point
i_opt = int(np.argmax(mu_vals))
Phi_opt = Phi[i_opt]
mu_opt = mu_vals[i_opt]
A_opt = A_osc[i_opt]

# Plot styling
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
ax_mu, ax_A = axes

# Left: growth rate mu(Phi_c)
ax_mu.plot(Phi, mu_vals, color="C0", lw=1.6)
ax_mu.axhline(0, color="grey", lw=0.6, ls="--")
ax_mu.axvline(Phi_opt, color="C2", lw=0.9, ls=":",
              label=fr"$\Phi_{{\mathrm{{opt}}}}\approx{Phi_opt:.2f}$")
ax_mu.axvline(Phi_crit, color="C3", lw=0.9, ls=":",
              label=fr"$\Phi_c^{{\mathrm{{crit}}}}\approx{Phi_crit:.2f}$")
ax_mu.fill_between(Phi, 0, mu_vals, where=(mu_vals > 0), alpha=0.10,
                   color="C0", label=r"$\mu>0$ (oscillatory branch exists)")
ax_mu.set_xlabel(r"constant training load $\Phi_c$ (TRIMP/day)")
ax_mu.set_ylabel(r"Stuart--Landau growth rate $\mu(B^*_0,F^*_0)$")
ax_mu.set_title("(a)  Growth rate")
ax_mu.set_xlim(0, 5)
ax_mu.set_ylim(-0.4, 0.3)
ax_mu.legend(loc="upper right", framealpha=0.95)
ax_mu.grid(True, alpha=0.25)

# Right: oscillatory amplitude A*_osc(Phi_c)
ax_A.plot(Phi[mu_vals > 0], A_osc[mu_vals > 0], color="C0", lw=1.8,
          label=r"$A^*_{\mathrm{osc}}(\Phi_c)$")
ax_A.plot(Phi[mu_vals <= 0], np.zeros_like(Phi[mu_vals <= 0]),
          color="C3", lw=1.8, label="sedentary (overtraining collapse)")
ax_A.axvline(Phi_opt, color="C2", lw=0.9, ls=":",
             label=fr"$\Phi_{{\mathrm{{opt}}}}\approx{Phi_opt:.2f}$")
ax_A.axvline(Phi_crit, color="C3", lw=0.9, ls=":",
             label=fr"$\Phi_c^{{\mathrm{{crit}}}}\approx{Phi_crit:.2f}$")
# Mark the constant-Phi=1 baseline (Stage D reference)
A_at_1 = float(np.sqrt(max(mu_of_Phi(1.0), 0) / eta))
ax_A.scatter([1.0], [A_at_1], color="grey", zorder=5)
ax_A.annotate(fr"baseline $\Phi=1$" + "\n" + fr"$A^*\approx{A_at_1:.2f}$",
              xy=(1.0, A_at_1), xytext=(0.3, 1.05),
              fontsize=8.5, color="dimgrey",
              arrowprops=dict(arrowstyle="-", color="dimgrey", lw=0.7))
ax_A.scatter([Phi_opt], [A_opt], color="C2", zorder=5)
ax_A.annotate(fr"optimum  $A^*\approx{A_opt:.2f}$",
              xy=(Phi_opt, A_opt), xytext=(2.4, 1.10),
              fontsize=8.5, color="C2",
              arrowprops=dict(arrowstyle="-", color="C2", lw=0.7))
ax_A.set_xlabel(r"constant training load $\Phi_c$ (TRIMP/day)")
ax_A.set_ylabel(r"oscillatory amplitude $A^*_{\mathrm{osc}}(\Phi_c)$")
ax_A.set_title("(b)  Goldilocks profile")
ax_A.set_xlim(0, 5)
ax_A.set_ylim(0, 1.3)
ax_A.legend(loc="upper right", framealpha=0.95, fontsize=8.5)
ax_A.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("/home/ajay/Repos/python-smc2-filtering-control-master/LaTex_docs/figures/fig_goldilocks.pdf",
            bbox_inches="tight")
print(f"Saved figures/fig_goldilocks.pdf  (Phi_opt={Phi_opt:.3f}, "
      f"Phi_crit={Phi_crit:.3f}, A_opt={A_opt:.3f})")
