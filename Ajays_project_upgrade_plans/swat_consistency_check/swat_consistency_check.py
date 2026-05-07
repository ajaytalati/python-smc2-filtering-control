"""Cross-repo SWAT consistency check.

Integrates the 7-state form (Repo A: Python-Model-Development-Simulation)
and the 4-state form (Repo C: Python-Model-Validation, vendored_dynamics)
under matching V_h, V_n, V_c values for three canonical scenarios, and
checks the testosterone trajectories agree to numerical precision.

Set A : V_h=1.0, V_n=0.3, V_c=0    (healthy)
Set B : V_h=0.2, V_n=3.5, V_c=0    (amplitude collapse)
Set D : V_h=1.0, V_n=0.3, V_c=6    (phase shift)
"""
from __future__ import annotations
import sys, os
import numpy as np
import jax
import jax.numpy as jnp
import diffrax

jax.config.update("jax_enable_x64", True)

REPO_A = "/home/ajay/Repos/Python-Model-Development-Simulation/version_1"
REPO_C = "/home/ajay/Repos/Python-Model-Validation/src"
sys.path[0:0] = [REPO_A, REPO_C]

from models.swat import _dynamics as repoa_dyn          # noqa
from models.swat.simulation import PARAM_SET_A          # noqa
from model_validation.models.swat import vendored_dynamics as repoc_dyn  # noqa
from model_validation.models.swat.vendored_parameters import default_swat_parameters  # noqa


# Repo A: drift takes (y, t, params_array, pi_dict).  Need params as ARRAY +
# pi index map, just like estimation.py builds it.  Construct a simple pi.
from models.swat.estimation import PARAM_PRIOR_CONFIG, PI

def repoa_param_array(p_dict, V_c_hours):
    """Build the parameter ARRAY in PI order from a dict + V_c override."""
    arr = np.zeros(len(PI), dtype=np.float64)
    for name, idx in PI.items():
        if name == 'V_c':
            arr[idx] = V_c_hours
        elif name in p_dict:
            arr[idx] = p_dict[name]
        else:
            ptype, pargs = PARAM_PRIOR_CONFIG[name]
            if ptype == 'lognormal':
                arr[idx] = float(np.exp(pargs[0]))
            elif ptype == 'normal':
                arr[idx] = float(pargs[0])
            elif ptype == 'beta':
                arr[idx] = float(pargs[0] / (pargs[0] + pargs[1]))
            else:
                arr[idx] = 0.0
    return jnp.asarray(arr)


def repoc_params_hours(V_c_hours):
    """Build Repo C's param dict converted back to per-hour units."""
    p = default_swat_parameters()
    HOURS = 24.0
    p_h = dict(p)
    p_h['tau_W'] *= HOURS
    p_h['tau_Z'] *= HOURS
    p_h['tau_a'] *= HOURS
    p_h['tau_T'] *= HOURS
    p_h['T_W']   /= HOURS
    p_h['T_Z']   /= HOURS
    p_h['T_a']   /= HOURS
    p_h['T_T']   /= HOURS
    return p_h


SCENARIOS = [
    ('A', 1.0, 0.3, 0.0),
    ('B', 0.2, 3.5, 0.0),
    ('D', 1.0, 0.3, 6.0),
]
T0 = 0.5
W0, Zt0, a0 = 0.5, 3.5, 0.5
T_TOTAL_HOURS = 14 * 24.0
DT = 0.05  # hours
N_PTS = int(T_TOTAL_HOURS / DT) + 1


def repoa_traj(V_h, V_n, V_c):
    """Deterministic ODE on Repo A's 7-state form."""
    p_dict = dict(PARAM_SET_A)
    p_array = repoa_param_array(p_dict, V_c)
    y0 = jnp.array([W0, Zt0, a0, T0, np.sin(repoa_dyn.PHI_MORNING_TYPE),
                     V_h, V_n], dtype=jnp.float64)

    def vf(t, y, args):
        return repoa_dyn.drift(y, t, p_array, PI)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Tsit5(),
        t0=0.0, t1=T_TOTAL_HOURS, dt0=0.01,
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, T_TOTAL_HOURS, N_PTS)),
        max_steps=2_000_000,
    )
    t = np.asarray(sol.ts)
    T_traj = np.asarray(sol.ys[:, 3])
    return t, T_traj


def repoc_traj(V_h, V_n, V_c):
    """Deterministic ODE on Repo C's 4-state form (native days form)."""
    p = default_swat_parameters()
    u = jnp.array([V_h, V_n, V_c], dtype=jnp.float64)
    x0 = jnp.array([W0, Zt0, a0, T0], dtype=jnp.float64)

    def vf(t, x, args):
        return repoc_dyn.swat_drift(t, x, u, p)

    T_TOTAL_DAYS = T_TOTAL_HOURS / 24.0
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf), diffrax.Tsit5(),
        t0=0.0, t1=T_TOTAL_DAYS, dt0=0.01 / 24.0,
        y0=x0,
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, T_TOTAL_DAYS, N_PTS)),
        max_steps=2_000_000,
    )
    t = np.asarray(sol.ts) * 24.0
    T_traj = np.asarray(sol.ys[:, 3])
    return t, T_traj


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_dir = Path("/tmp/swat_consistency_plots")
    out_dir.mkdir(exist_ok=True)

    print("Cross-repo SWAT consistency check")
    print(f"Horizon {T_TOTAL_HOURS}h ({T_TOTAL_HOURS/24}d), dt={DT}h, init T0={T0}")
    print("=" * 75)
    print(f"{'Scenario':<8} {'V_h':>5} {'V_n':>5} {'V_c':>5}  "
          f"{'A: T(D)':>9} {'C: T(D)':>9} {'max|dT|':>9} {'mean|dT|':>9}")
    print("-" * 75)

    descriptions = {
        'A': 'healthy basin (V_h=1, V_n=0.3, V_c=0)',
        'B': 'amplitude collapse (V_h=0.2, V_n=3.5, V_c=0)',
        'D': 'phase shift (V_h=1, V_n=0.3, V_c=6h)',
    }

    for name, V_h, V_n, V_c in SCENARIOS:
        t, T_a = repoa_traj(V_h, V_n, V_c)
        _, T_c = repoc_traj(V_h, V_n, V_c)
        delta = np.abs(T_a - T_c)
        t_days = t / 24.0
        print(f"Set {name:<4} {V_h:>5.2f} {V_n:>5.2f} {V_c:>5.1f}  "
              f"{T_a[-1]:>9.6f} {T_c[-1]:>9.6f} {delta.max():>9.2e} {delta.mean():>9.2e}")

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]},
        )
        ax1.plot(t_days, T_a, color="tab:blue", lw=2.0,
                  label="Repo A (7-state, estimation form)")
        ax1.plot(t_days, T_c, color="tab:orange", lw=1.5, ls="--",
                  label="Repo C (4-state, control form)")
        ax1.set_ylabel("Testosterone amplitude T(t)")
        ax1.set_title(f"Set {name} - {descriptions[name]}")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)

        ax2.semilogy(t_days, np.maximum(delta, 1e-18), color="tab:red", lw=1.0)
        ax2.set_ylabel("|T_A - T_C|")
        ax2.set_xlabel("time (days)")
        ax2.set_ylim(1e-18, 1e-3)
        ax2.grid(alpha=0.3, which="both")
        ax2.text(0.02, 0.85, f"max |dT| = {delta.max():.2e}",
                  transform=ax2.transAxes, fontsize=9, family="monospace")

        fig.tight_layout()
        out_path = out_dir / f"set_{name}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"   wrote {out_path}")


if __name__ == '__main__':
    main()
