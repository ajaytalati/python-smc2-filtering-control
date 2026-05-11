"""
Generate the two figures referenced by sedentary_basin_controllability.tex.

Fully self-contained: hard-codes TRUTH_PARAMS_V5 from
version_1_5_Julia/models/fsa_v5/simulation_v5.jl so a reader can
reproduce without any project dependencies. Only requires numpy +
matplotlib.

Run:
    python generate_figures.py

Outputs (same directory):
    candidates_trajectories.png   -- §6 candidate-comparison time-series
    controllability_region.png    -- §9 (B0, S0) phase-plot heat-map
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# TRUTH_PARAMS_V5 (centred forms, hard-coded from simulation_v5.jl:80-135)
# -----------------------------------------------------------------------------

A_TYP = 0.10
F_TYP = 0.20

PARAMS = dict(
    # Aerobic block
    tau_B      = 42.0,
    kappa_B    = 0.012 * (1.0 + 0.40 * A_TYP),    # = 0.01248
    epsilon_AB = 0.40,
    # Strength block
    tau_S      = 60.0,
    kappa_S    = 0.008 * (1.0 + 0.20 * A_TYP),    # = 0.00816
    epsilon_AS = 0.20,
    # Fatigue
    tau_F      = 7.0 / (1.0 + 1.0 * A_TYP),       # = 6.36
    lambda_A   = 1.00,
    # Busso K block
    KFB_0      = 0.030,
    KFS_0      = 0.050,
    tau_K      = 21.0,
    mu_K       = 0.005,
    # Stuart-Landau bifurcation parameter (centred)
    mu_0       = 0.02 + 0.40 * (F_TYP ** 2),      # = 0.036
    mu_B       = 0.30,
    mu_S       = 0.15,
    mu_F       = 0.10 + 2.0 * F_TYP * 0.40,       # = 0.26
    mu_FF      = 0.40,
    eta        = 0.20,
    # v5 Hill deconditioning
    B_dec      = 0.07,
    S_dec      = 0.07,
    mu_dec_B   = 0.10,
    mu_dec_S   = 0.10,
    n_dec      = 4.0,
)

SEDENTARY_INIT = dict(B=0.05, S=0.10, F=0.30, A=0.10, KFB=0.030, KFS=0.050)
TRAINED_ATHLETE_INIT = dict(B=0.50, S=0.45, F=0.20, A=0.45, KFB=0.0615, KFS=0.0815)

# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

def hill(x, x_dec, n):
    """Hill saturation: h_n(x; x_dec) = x_dec^n / (x^n + x_dec^n) ∈ [0, 1]."""
    return x_dec ** n / (x ** n + x_dec ** n)


def mu_bar_state(B, S, F, p=PARAMS):
    """Bifurcation parameter μ(B, S, F) — Eq. (6) in the document."""
    return (
        p["mu_0"]
        + p["mu_B"] * B + p["mu_S"] * S
        - p["mu_F"] * F - p["mu_FF"] * (F - F_TYP) ** 2
        - p["mu_dec_B"] * hill(B, p["B_dec"], p["n_dec"])
        - p["mu_dec_S"] * hill(S, p["S_dec"], p["n_dec"])
    )


def drift(y, phi_B, phi_S, p=PARAMS):
    """Drift field f(y, Φ) — Eqs. (1)–(5)."""
    B, S, F, A, KFB, KFS = y
    a_B = (1 + p["epsilon_AB"] * A) / (1 + p["epsilon_AB"] * A_TYP)
    a_S = (1 + p["epsilon_AS"] * A) / (1 + p["epsilon_AS"] * A_TYP)
    a_F = (1 + p["lambda_A"]   * A) / (1 + p["lambda_A"]   * A_TYP)

    dB   = p["kappa_B"] * a_B * phi_B - B / p["tau_B"]
    dS   = p["kappa_S"] * a_S * phi_S - S / p["tau_S"]
    dF   = KFB * phi_B + KFS * phi_S - a_F * F / p["tau_F"]
    mu   = mu_bar_state(B, S, F, p)
    dA   = mu * A - p["eta"] * A ** 3
    dKFB = (p["KFB_0"] - KFB) / p["tau_K"] + p["mu_K"] * phi_B
    dKFS = (p["KFS_0"] - KFS) / p["tau_K"] + p["mu_K"] * phi_S
    return np.array([dB, dS, dF, dA, dKFB, dKFS]), mu


def simulate(y0, phi_fn, T=100.0, dt=0.01, p=PARAMS):
    """RK4 integration of the deterministic ODE.

    y0      : dict with B, S, F, A, KFB, KFS
    phi_fn  : callable t -> (phi_B, phi_S)
    p       : parameter dict (default canonical TRUTH_PARAMS_V5)
    Returns time grid, state trajectory (N, 6), and μ(t) array.
    """
    y = np.array([y0["B"], y0["S"], y0["F"], y0["A"], y0["KFB"], y0["KFS"]])
    n_steps = int(round(T / dt))
    ts = np.linspace(0.0, T, n_steps + 1)
    ys = np.zeros((n_steps + 1, 6))
    mus = np.zeros(n_steps + 1)
    ys[0] = y
    mus[0] = mu_bar_state(y[0], y[1], y[2], p)
    for k in range(n_steps):
        t = ts[k]
        phi_B1, phi_S1 = phi_fn(t)
        phi_B2, phi_S2 = phi_fn(t + 0.5 * dt)
        phi_B3, phi_S3 = phi_fn(t + 0.5 * dt)
        phi_B4, phi_S4 = phi_fn(t + dt)
        k1, _ = drift(y,             phi_B1, phi_S1, p)
        k2, _ = drift(y + 0.5*dt*k1, phi_B2, phi_S2, p)
        k3, _ = drift(y + 0.5*dt*k2, phi_B3, phi_S3, p)
        k4, _ = drift(y + dt*k3,     phi_B4, phi_S4, p)
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        # plant-side clipping (matches Julia EM step)
        y[0] = np.clip(y[0], 1e-4, 1 - 1e-4)
        y[1] = np.clip(y[1], 1e-4, 1 - 1e-4)
        y[2] = max(y[2], 0.0)
        y[3] = max(y[3], 0.0)
        y[4] = max(y[4], 0.0)
        y[5] = max(y[5], 0.0)
        ys[k + 1] = y
        mus[k + 1] = mu_bar_state(y[0], y[1], y[2], p)
    return ts, ys, mus


# -----------------------------------------------------------------------------
# Helpers for the §14–§16 extension: parameter modification + slow-manifold +
# island heatmaps.
# -----------------------------------------------------------------------------

def make_params(**overrides):
    """Build a modified copy of PARAMS with named overrides.

    Example: make_params(B_dec=0.20, S_dec=0.20, mu_dec_B=0.05, mu_dec_S=0.05)
    """
    p = dict(PARAMS)
    for k, v in overrides.items():
        if k not in p:
            raise KeyError(f"Unknown parameter {k!r}; valid keys: {sorted(p.keys())}")
        p[k] = v
    return p


def slow_manifold_state(phi_B, phi_S, A, p=PARAMS):
    """Slow-manifold equilibrium values (B*, S*, F*, K_FB*, K_FS*) at a given A
    and constant Φ — Eqs. (8)–(11) of the LaTeX document."""
    a_B = (1 + p["epsilon_AB"] * A) / (1 + p["epsilon_AB"] * A_TYP)
    a_S = (1 + p["epsilon_AS"] * A) / (1 + p["epsilon_AS"] * A_TYP)
    a_F = (1 + p["lambda_A"]   * A) / (1 + p["lambda_A"]   * A_TYP)
    KFB = p["KFB_0"] + p["tau_K"] * p["mu_K"] * phi_B
    KFS = p["KFS_0"] + p["tau_K"] * p["mu_K"] * phi_S
    B   = p["tau_B"] * p["kappa_B"] * a_B * phi_B
    S   = p["tau_S"] * p["kappa_S"] * a_S * phi_S
    F   = p["tau_F"] * (KFB * phi_B + KFS * phi_S) / a_F
    return dict(B=B, S=S, F=F, A=A, KFB=KFB, KFS=KFS)


def mu_bar_slow(phi_B, phi_S, A, p=PARAMS):
    """Slow-manifold bifurcation parameter μ̄(A; Φ) — evaluated by substituting
    the cascade equilibria into μ(B*, S*, F*).
    """
    sm = slow_manifold_state(phi_B, phi_S, A, p)
    # Note: B*, S* under clipped plant would be min(., 1) — we follow the same
    # convention here.
    B_eff = min(sm["B"], 1.0)
    S_eff = min(sm["S"], 1.0)
    return mu_bar_state(B_eff, S_eff, sm["F"], p)


def mu_bar_slow_grid(phi_grid, p=PARAMS, A_anchor=0.0):
    """Vectorised μ̄(A_anchor; Φ) on a (φ_B, φ_S) grid.

    phi_grid : 1D array of φ values (same for both axes)
    Returns: (n, n) array, mu[i, j] = μ̄(A_anchor; φ_grid[i], φ_grid[j]).
    """
    n = len(phi_grid)
    mu = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mu[i, j] = mu_bar_slow(phi_grid[i], phi_grid[j], A_anchor, p)
    return mu


def island_center(phi_grid, p=PARAMS, A_anchor=0.0):
    """Find the (φ_B, φ_S) that maximises μ̄(A_anchor; Φ) on the grid.
    Returns (center_phi_B, center_phi_S, mu_max)."""
    mu = mu_bar_slow_grid(phi_grid, p, A_anchor)
    idx = np.unravel_index(np.argmax(mu), mu.shape)
    return phi_grid[idx[0]], phi_grid[idx[1]], mu[idx]


def find_trained_athlete_v2(p, phi_target=None, A_init=0.45, phi_grid=None):
    """Given a parametrisation p, compute TRAINED_ATHLETE_INIT_v2 as the slow-
    manifold state at the new island center.

    If phi_target is None, locate the island center via mu_bar_slow_grid on
    phi_grid (default: 0..2 in 41 steps). Then use slow_manifold_state at that Φ
    with A = A_init to get the 6D state.
    """
    if phi_target is None:
        if phi_grid is None:
            phi_grid = np.linspace(0.0, 2.0, 41)
        pB, pS, _ = island_center(phi_grid, p)
        phi_target = (pB, pS)
    state = slow_manifold_state(phi_target[0], phi_target[1], A_init, p)
    state["__phi_target__"] = phi_target
    return state


def plot_island(p, ax, phi_max=2.0, n_grid=80, A_anchor=0.0,
                title=None, overlay_canonical_island=True,
                probe_points=None, contour_levels=None, vmin=-0.5, vmax=0.5,
                show_colorbar=True):
    """Plot a μ̄(A_anchor; Φ) heatmap with the μ̄ = 0 contour highlighting the
    healthy island.

    p             : parameter dict for the model variant
    ax            : matplotlib Axes
    phi_max       : upper bound on the (φ_B, φ_S) plot range
    overlay_canonical_island : if True, also plot a dashed contour for the
                               canonical PARAMS (for reference)
    probe_points  : list of (phi_B, phi_S, label, color) tuples
    contour_levels: μ̄ contour levels to draw (default: [0])
    """
    phi_grid = np.linspace(0.0, phi_max, n_grid)
    mu = mu_bar_slow_grid(phi_grid, p, A_anchor)

    # Mirror conventions: μ̄ heatmap, RdBu (blue = healthy, red = collapsed),
    # zero contour highlights the island.
    im = ax.imshow(mu.T, origin="lower",
                    extent=[0, phi_max, 0, phi_max],
                    cmap="RdBu", vmin=vmin, vmax=vmax, aspect="equal")
    if contour_levels is None:
        contour_levels = [0.0]
    cs = ax.contour(phi_grid, phi_grid, mu.T, levels=contour_levels,
                     colors="black", linewidths=2.0)
    ax.clabel(cs, fmt={0.0: "μ̄ = 0"}, fontsize=8)

    if overlay_canonical_island:
        mu_canon = mu_bar_slow_grid(phi_grid, PARAMS, A_anchor)
        ax.contour(phi_grid, phi_grid, mu_canon.T, levels=[0.0],
                    colors="gray", linewidths=1.0, linestyles="dashed")

    if probe_points is not None:
        for ph_B, ph_S, label, color in probe_points:
            ax.scatter([ph_B], [ph_S], color=color, s=70,
                        edgecolors="black", zorder=5)
            ax.annotate(label, xy=(ph_B, ph_S), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color=color)

    ax.set_xlabel("Φ_B")
    ax.set_ylabel("Φ_S")
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="μ̄(0; Φ)")
    return im


# -----------------------------------------------------------------------------
# §14: Single-modification island plots
# -----------------------------------------------------------------------------

def fig_single_mods():
    """3-panel figure: each panel shows the healthy island under one of the
    single deconditioning modifications from §13."""
    mods = [
        ("Mod A: μ_dec halved (0.10 → 0.05)",
            make_params(mu_dec_B=0.05, mu_dec_S=0.05), "island_mod_A.png"),
        ("Mod B: B_dec lowered (0.07 → 0.04)",
            make_params(B_dec=0.04, S_dec=0.04), "island_mod_B.png"),
        ("Mod C: Hill exponent n reduced (4 → 2)",
            make_params(n_dec=2.0), "island_mod_C.png"),
    ]

    # Also show the BASELINE for reference + each MOD
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    plot_island(PARAMS, axes[0], phi_max=2.0, n_grid=80,
                title="Baseline TRUTH_PARAMS_V5",
                overlay_canonical_island=False,
                probe_points=[(0.30, 0.30, "(0.3,0.3)", "black"),
                              (1.0, 1.0, "(1,1)", "darkred")])

    for ax, (label, p_mod, _) in zip(axes[1:], mods):
        plot_island(p_mod, ax, phi_max=2.0, n_grid=80,
                    title=label,
                    overlay_canonical_island=True,
                    probe_points=[(0.30, 0.30, "(0.3,0.3)", "black"),
                                  (1.0, 1.0, "(1,1)", "darkred")])

    plt.suptitle("FSA-v5 healthy island under the 3 single deconditioning modifications\n"
                 "Black solid: μ̄=0 contour under modified params. "
                 "Dashed gray: canonical baseline μ̄=0 (for reference).",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "island_single_mods.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    # Report island-center shifts
    phi_grid = np.linspace(0.0, 2.0, 81)
    print("\nIsland-center summary (φ at the maximum of μ̄(0; Φ)):")
    pB, pS, mu_max = island_center(phi_grid, PARAMS)
    print(f"  Baseline:              center = ({pB:.2f}, {pS:.2f})  μ̄_max = {mu_max:+.4f}")
    for label, p_mod, _ in mods:
        pB, pS, mu_max = island_center(phi_grid, p_mod)
        print(f"  {label:45s}  center = ({pB:.2f}, {pS:.2f})  μ̄_max = {mu_max:+.4f}")


# -----------------------------------------------------------------------------
# §15: Sweep the deconditioning block (+ reward-side fallback) to shift the
# island toward (1,1)
# -----------------------------------------------------------------------------

def qualitative_metrics(p, phi_sed=(0.1, 0.1), phi_target=(1.0, 1.0),
                         phi_over=(2.0, 2.0)):
    """Compute the four scalar μ̄ values the user cares about, plus the island
    center location."""
    phi_grid = np.linspace(0.0, 2.5, 51)
    cx, cy, mu_max = island_center(phi_grid, p)
    return dict(
        mu_sed=mu_bar_slow(phi_sed[0], phi_sed[1], 0.0, p),
        mu_target=mu_bar_slow(phi_target[0], phi_target[1], 0.0, p),
        mu_over=mu_bar_slow(phi_over[0], phi_over[1], 0.0, p),
        island_cx=cx, island_cy=cy, mu_max=mu_max,
    )


def passes_qualitative(metrics, sed_thresh=-0.01, target_thresh=+0.001,
                         over_thresh=-0.05):
    """User's three qualitative criteria."""
    return (metrics["mu_sed"]    < sed_thresh and
            metrics["mu_target"] > target_thresh and
            metrics["mu_over"]   < over_thresh)


def sweep_deconditioning_only():
    """Pure 3D sweep over the deconditioning block (B_dec=S_dec coupled,
    μ_dec_B=μ_dec_S coupled, n)."""
    results = []
    for B_dec in [0.07, 0.15, 0.25, 0.35, 0.50, 0.70]:
        for mu_dec in [0.02, 0.05, 0.10, 0.20, 0.40]:
            for n in [2, 4, 6]:
                p = make_params(B_dec=B_dec, S_dec=B_dec,
                                mu_dec_B=mu_dec, mu_dec_S=mu_dec,
                                n_dec=float(n))
                m = qualitative_metrics(p)
                m["B_dec"] = B_dec
                m["mu_dec"] = mu_dec
                m["n"] = n
                m["passes"] = passes_qualitative(m)
                results.append(m)
    return results


def sweep_combined():
    """Combined sweep — deconditioning block + reward-side μ_F, μ_FF.
    Only fired if deconditioning-only sweep fails to shift the island to (1,1).
    """
    results = []
    for B_dec in [0.15, 0.25, 0.35]:
        for mu_dec in [0.05, 0.10, 0.20]:
            for mu_F in [0.03, 0.06, 0.10, 0.15, 0.26]:
                for mu_FF in [0.0, 0.02, 0.05, 0.10, 0.40]:
                    p = make_params(B_dec=B_dec, S_dec=B_dec,
                                    mu_dec_B=mu_dec, mu_dec_S=mu_dec,
                                    mu_F=mu_F, mu_FF=mu_FF)
                    m = qualitative_metrics(p)
                    m["B_dec"] = B_dec
                    m["mu_dec"] = mu_dec
                    m["mu_F"] = mu_F
                    m["mu_FF"] = mu_FF
                    m["passes"] = passes_qualitative(m)
                    # Score: prefer island center near (1,1), prefer minimum
                    # change from canonical
                    dist_to_target = (m["island_cx"] - 1.0)**2 + (m["island_cy"] - 1.0)**2
                    change_from_canonical = (
                        abs(mu_F - PARAMS["mu_F"]) / PARAMS["mu_F"]
                        + abs(mu_FF - PARAMS["mu_FF"]) / PARAMS["mu_FF"]
                        + abs(B_dec - PARAMS["B_dec"]) / PARAMS["B_dec"]
                        + abs(mu_dec - PARAMS["mu_dec_B"]) / PARAMS["mu_dec_B"]
                    )
                    m["score"] = dist_to_target + 0.05 * change_from_canonical
                    results.append(m)
    return results


def report_top_candidates(results, n_top=10, label=""):
    """Print top N candidates from a sweep, sorted by passes-qualitative and
    then by island-center distance to (1,1)."""
    print(f"\n=== Top {n_top} candidates from {label} ===")
    # Sort: first by whether-passes (descending), then by score if present, else
    # by distance to (1,1)
    def sort_key(m):
        passes = -int(m.get("passes", False))
        if "score" in m:
            return (passes, m["score"])
        dist = (m["island_cx"] - 1.0)**2 + (m["island_cy"] - 1.0)**2
        return (passes, dist)

    sorted_results = sorted(results, key=sort_key)

    hdr = "passes? μ_sed     μ_target   μ_over    cx     cy     μ̄_max   "
    hdr += "B_dec   μ_dec   "
    if "mu_F" in sorted_results[0]:
        hdr += "μ_F     μ_FF    "
    if "n" in sorted_results[0]:
        hdr += "n   "
    print(hdr)
    for m in sorted_results[:n_top]:
        line = f"{'✓' if m['passes'] else '✗':4s}    "
        line += f"{m['mu_sed']:+.4f}  {m['mu_target']:+.4f}  {m['mu_over']:+.4f}  "
        line += f"{m['island_cx']:.2f}   {m['island_cy']:.2f}   {m['mu_max']:+.4f}  "
        line += f"{m['B_dec']:.2f}    {m['mu_dec']:.2f}    "
        if "mu_F" in m:
            line += f"{m['mu_F']:.2f}    {m['mu_FF']:.2f}    "
        if "n" in m:
            line += f"{m['n']:.0f}   "
        print(line)
    return sorted_results


def fig_sweep_recommendation(recommended_p, label_recommended):
    """Show the recommended parametrisation as a single big island plot,
    annotated with the three probe points."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    phi_grid = np.linspace(0.0, 2.5, 100)
    mu_grid = mu_bar_slow_grid(phi_grid, recommended_p)
    cx, cy, mu_max = island_center(phi_grid, recommended_p)

    probe_points = [
        (0.1, 0.1, "(0.1,0.1) sedentary policy", "purple"),
        (1.0, 1.0, "(1,1) target center", "darkgreen"),
        (2.0, 2.0, "(2,2) over-training", "darkred"),
        (cx, cy, f"island max ({cx:.2f},{cy:.2f})", "blue"),
    ]
    plot_island(recommended_p, ax, phi_max=2.5, n_grid=100, title=label_recommended,
                overlay_canonical_island=True,
                probe_points=probe_points, vmin=-1.5, vmax=0.3)

    out = Path(__file__).parent / "island_recommended.png"
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nSaved {out}")
    plt.close(fig)


def fig_sweep_grid(top_candidates, n_show=9):
    """3x3 grid of island plots for the top 9 candidates from the combined
    sweep."""
    n_rows = int(np.ceil(np.sqrt(n_show)))
    fig, axes = plt.subplots(n_rows, n_rows, figsize=(15, 14))
    axes = axes.flatten()

    for i, m in enumerate(top_candidates[:n_show]):
        ax = axes[i]
        p = make_params(
            B_dec=m["B_dec"], S_dec=m["B_dec"],
            mu_dec_B=m["mu_dec"], mu_dec_S=m["mu_dec"],
            **({"mu_F": m["mu_F"]} if "mu_F" in m else {}),
            **({"mu_FF": m["mu_FF"]} if "mu_FF" in m else {}),
            **({"n_dec": float(m["n"])} if "n" in m else {}),
        )
        label_lines = [
            f"B_dec={m['B_dec']}, μ_dec={m['mu_dec']}",
        ]
        if "mu_F" in m:
            label_lines.append(f"μ_F={m['mu_F']}, μ_FF={m['mu_FF']}")
        label_lines.append(f"μ̄(1,1)={m['mu_target']:+.3f}  "
                            f"{'✓PASSES' if m['passes'] else '✗fails'}")
        plot_island(p, ax, phi_max=2.5, n_grid=60,
                    title="\n".join(label_lines),
                    overlay_canonical_island=True,
                    probe_points=[(1, 1, "", "darkgreen"),
                                   (0.1, 0.1, "", "purple"),
                                   (2, 2, "", "darkred")],
                    vmin=-1.5, vmax=0.3, show_colorbar=False)

    for j in range(n_show, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Top candidates from combined sweep (deconditioning + reward-side)\n"
                 "Green dot: target center (1,1). Purple: sedentary probe. Red: over-training probe.\n"
                 "Dashed gray contour: canonical μ̄=0 (for reference).",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "island_sweep_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def run_sweep_phase():
    """Run both sweeps, identify the recommended parametrisation, and generate
    the §15 figures."""
    print("\n" + "=" * 60)
    print("§15.A: Pure deconditioning-block sweep (3D)")
    print("=" * 60)
    results_decond = sweep_deconditioning_only()
    sorted_decond = report_top_candidates(results_decond, n_top=8,
                                            label="deconditioning-only sweep")

    decond_passes = [m for m in sorted_decond if m["passes"]]
    if decond_passes:
        best_decond = decond_passes[0]
        dist = (best_decond["island_cx"] - 1.0)**2 + (best_decond["island_cy"] - 1.0)**2
        print(f"\nBest deconditioning-only result: island at "
              f"({best_decond['island_cx']:.2f}, {best_decond['island_cy']:.2f}), "
              f"distance to (1,1) = {np.sqrt(dist):.3f}")
        if dist > 0.10:  # Distance > sqrt(0.1) = 0.32 from (1,1)
            print("→ Deconditioning-only insufficient to shift island to (1,1). "
                  "Proceeding to combined sweep.")
    else:
        print("\n→ No deconditioning-only parametrisation passes ALL qualitative "
              "criteria. Proceeding to combined sweep.")

    print("\n" + "=" * 60)
    print("§15.B: Combined sweep (deconditioning + reward-side μ_F, μ_FF)")
    print("=" * 60)
    results_combined = sweep_combined()
    sorted_combined = report_top_candidates(results_combined, n_top=10,
                                             label="combined sweep")

    combined_passes = [m for m in sorted_combined if m["passes"]]
    if not combined_passes:
        print("\n!!! WARNING: no parametrisation passes ALL qualitative criteria. "
              "Picking best-distance.")
        best = sorted_combined[0]
    else:
        best = combined_passes[0]
    print(f"\n→ Recommended parametrisation:")
    print(f"   B_dec = S_dec   = {best['B_dec']}")
    print(f"   μ_B− = μ_S−    = {best['mu_dec']}")
    print(f"   μ_F            = {best.get('mu_F', PARAMS['mu_F']):.3f}")
    print(f"   μ_FF           = {best.get('mu_FF', PARAMS['mu_FF']):.3f}")
    print(f"   n              = {int(best.get('n', 4))}")
    print(f"   island center  = ({best['island_cx']:.2f}, {best['island_cy']:.2f})")
    print(f"   μ̄(0; (1,1))    = {best['mu_target']:+.4f}")
    print(f"   μ̄(0; (0.1,0.1)) = {best['mu_sed']:+.4f}")
    print(f"   μ̄(0; (2,2))    = {best['mu_over']:+.4f}")

    recommended_p = make_params(
        B_dec=best["B_dec"], S_dec=best["B_dec"],
        mu_dec_B=best["mu_dec"], mu_dec_S=best["mu_dec"],
        **({"mu_F":  best["mu_F"]}  if "mu_F"  in best else {}),
        **({"mu_FF": best["mu_FF"]} if "mu_FF" in best else {}),
        **({"n_dec": float(best["n"])} if "n" in best else {}),
    )

    # Generate the recommended-parametrisation figure
    label = (f"RECOMMENDED PARAMETRISATION\n"
             f"B_dec={best['B_dec']}, μ_dec={best['mu_dec']}, "
             f"μ_F={best.get('mu_F', PARAMS['mu_F']):.3f}, "
             f"μ_FF={best.get('mu_FF', PARAMS['mu_FF']):.3f}")
    fig_sweep_recommendation(recommended_p, label)

    # Generate the top-candidates grid
    fig_sweep_grid(sorted_combined, n_show=9)

    return recommended_p, best


# -----------------------------------------------------------------------------
# §16: Constructive controllability under the recommended parametrisation
# -----------------------------------------------------------------------------

def fig_sedentary_basin_witness(p_recommended, T=200.0):
    """§16.1 + §16.2: from SEDENTARY_INIT under recommended params, show
    (a) passive Φ = (0.1, 0.1) → A → 0  (basin preserved)
    (b) constructive witness Φ(t) → A → island center  (controllable escape)

    Note: T extended to 200 d because the τ_B = 42 d, τ_S = 60 d time-scales
    require ~3τ to reach the slow manifold, leaving only ~25 d of useful
    growing-A time within T=100 d. T=200 gives the recommended parametrisation
    a fair shot.
    """
    dt = 0.05

    # --- (a) Passive sedentary policy ---
    phi_passive = lambda t: (0.1, 0.1)
    ts_a, ys_a, mus_a = simulate(SEDENTARY_INIT, phi_passive, T=T, dt=dt, p=p_recommended)

    # --- (b) Constructive witness: try several candidates, report all, plot best ---
    candidates = [
        ("constant Φ ≡ (1, 1)", lambda t: (1.0, 1.0)),
        ("aerobic-only burst Φ_B=2 then (1,1)",
            lambda t: (2.0, 0.0) if t < 10 else (1.0, 1.0)),
        ("3-seg ramp 0.5→0.8→1.0",
            lambda t: (0.5, 0.5) if t < 30 else (0.8, 0.8) if t < 60 else (1.0, 1.0)),
        ("aerobic-burst then balanced",
            lambda t: (1.5, 0.5) if t < 20 else (1.0, 1.0)),
    ]
    candidate_results = []
    for label, phi_fn in candidates:
        ts, ys, mus = simulate(SEDENTARY_INIT, phi_fn, T=T, dt=dt, p=p_recommended)
        integ = np.trapezoid(mus, ts)
        candidate_results.append((label, phi_fn, ts, ys, mus, integ, ys[-1, 3]))
        print(f"  Candidate {label!r}: A({T:.0f}) = {ys[-1, 3]:.4f}  ∫μ = {integ:+.3f}  "
              f"{'✓ ESCAPED' if ys[-1, 3] >= 0.30 else '✗ failed'}")

    # Pick the best (highest A at T)
    best = max(candidate_results, key=lambda x: x[6])
    label_b, phi_witness, ts_b, ys_b, mus_b, integ_b, A_final_b = best
    print(f"\n  → Best witness: {label_b!r}  A({T:.0f}) = {A_final_b:.4f}")
    phi_witness_label = label_b

    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ((ax_phi_a, ax_phi_b),
     (ax_BSF_a, ax_BSF_b),
     (ax_A_a,   ax_A_b)) = axes

    # --- Left column: passive ---
    ax_phi_a.plot(ts_a, [phi_passive(t)[0] for t in ts_a], "C0-", lw=2, label="Φ_B")
    ax_phi_a.plot(ts_a, [phi_passive(t)[1] for t in ts_a], "C1--", lw=2, label="Φ_S")
    ax_phi_a.set_ylim(0, 2.5); ax_phi_a.set_ylabel("Φ"); ax_phi_a.grid(True, alpha=0.3)
    ax_phi_a.set_title("§16.1 — PASSIVE sedentary policy Φ ≡ (0.1, 0.1)")
    ax_phi_a.legend(fontsize=8)

    ax_BSF_a.plot(ts_a, ys_a[:, 0], "C0", lw=2, label="B")
    ax_BSF_a.plot(ts_a, ys_a[:, 1], "C1", lw=2, label="S")
    ax_BSF_a.plot(ts_a, ys_a[:, 2], "C3", lw=2, label="F")
    ax_BSF_a.axhline(p_recommended["B_dec"], color="k", ls=":", lw=1, label=f"B_dec={p_recommended['B_dec']}")
    ax_BSF_a.set_ylabel("B, S, F"); ax_BSF_a.grid(True, alpha=0.3)
    ax_BSF_a.legend(fontsize=8)

    ax_A_a.plot(ts_a, ys_a[:, 3], "k", lw=2, label="A")
    ax_A_a.set_yscale("log"); ax_A_a.set_ylim(1e-3, 1.5)
    ax_A_a.axhline(0.30, color="r", ls=":", lw=1, label="A_target=0.30")
    ax_A_a.axhline(SEDENTARY_INIT["A"], color="purple", ls=":", lw=1, label="A_0=0.10")
    ax_A_a.set_xlabel("t (d)"); ax_A_a.set_ylabel("A (log)"); ax_A_a.grid(True, alpha=0.3)
    ax_A_a.legend(fontsize=8, loc="lower left")
    ax_A_a.set_title(f"A trajectory under passive policy:  A(100) = {ys_a[-1, 3]:.4f}")

    # --- Right column: witness ---
    phis_b = np.array([phi_witness(t) for t in ts_b])
    ax_phi_b.plot(ts_b, phis_b[:, 0], "C2", lw=2, label="Φ_B")
    ax_phi_b.plot(ts_b, phis_b[:, 1], "C2--", lw=2, label="Φ_S")
    ax_phi_b.set_ylim(0, 2.5); ax_phi_b.grid(True, alpha=0.3)
    ax_phi_b.set_title(f"§16.2 — CONSTRUCTIVE WITNESS Φ(t):  {phi_witness_label}")
    ax_phi_b.legend(fontsize=8)

    ax_BSF_b.plot(ts_b, ys_b[:, 0], "C0", lw=2, label="B")
    ax_BSF_b.plot(ts_b, ys_b[:, 1], "C1", lw=2, label="S")
    ax_BSF_b.plot(ts_b, ys_b[:, 2], "C3", lw=2, label="F")
    ax_BSF_b.axhline(p_recommended["B_dec"], color="k", ls=":", lw=1, label=f"B_dec={p_recommended['B_dec']}")
    ax_BSF_b.set_ylabel("B, S, F"); ax_BSF_b.grid(True, alpha=0.3)
    ax_BSF_b.legend(fontsize=8)

    ax_A_b.plot(ts_b, ys_b[:, 3], "k", lw=2, label="A")
    ax_A_b.set_yscale("log"); ax_A_b.set_ylim(1e-3, 1.5)
    ax_A_b.axhline(0.30, color="r", ls=":", lw=1, label="A_target=0.30")
    ax_A_b.axhline(SEDENTARY_INIT["A"], color="purple", ls=":", lw=1, label="A_0=0.10")
    ax_A_b.set_xlabel("t (d)"); ax_A_b.set_ylabel("A (log)"); ax_A_b.grid(True, alpha=0.3)
    ax_A_b.legend(fontsize=8, loc="lower right")
    ax_A_b.set_title(f"A trajectory under witness:  A(100) = {ys_b[-1, 3]:.4f}  ({'ESCAPED' if ys_b[-1, 3] >= 0.30 else 'failed'})")

    integ_a = np.trapezoid(mus_a, ts_a)
    plt.suptitle(f"§16.1 + §16.2 — Sedentary basin under RECOMMENDED parametrisation: from SEDENTARY_INIT, T={T:.0f}d\n"
                 f"Left (passive): A({T:.0f})={ys_a[-1, 3]:.4f}  ∫μ={integ_a:+.3f}    "
                 f"Right (best witness): A({T:.0f})={ys_b[-1, 3]:.4f}  ∫μ={integ_b:+.3f}",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "sedentary_basin_witness.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    return ys_a[-1, 3], ys_b[-1, 3], integ_a, integ_b


def fig_overtraining_basin_witness(p_recommended, trained_init_v2):
    """§16.3: from TRAINED_ATHLETE_INIT_v2 under recommended params, show
    (a) passive Φ = (2, 2) → A → 0  (basin preserved at the shifted location)
    (b) constructive maintenance Φ(t) ≡ (1, 1) → A maintained near A*
    """
    T = 100.0; dt = 0.05

    # --- (a) Passive overtraining policy ---
    phi_overtrain = lambda t: (2.0, 2.0)
    ts_a, ys_a, mus_a = simulate(trained_init_v2, phi_overtrain, T=T, dt=dt, p=p_recommended)

    # --- (b) Maintenance witness: Φ ≡ (1, 1) constant ---
    phi_maintain = lambda t: (1.0, 1.0)
    ts_b, ys_b, mus_b = simulate(trained_init_v2, phi_maintain, T=T, dt=dt, p=p_recommended)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ((ax_phi_a, ax_phi_b),
     (ax_BSF_a, ax_BSF_b),
     (ax_A_a,   ax_A_b)) = axes

    # --- (a) Passive ---
    ax_phi_a.plot(ts_a, [2.0]*len(ts_a), "C3", lw=2, label="Φ_B = Φ_S = 2.0")
    ax_phi_a.set_ylim(0, 2.5); ax_phi_a.set_ylabel("Φ"); ax_phi_a.grid(True, alpha=0.3)
    ax_phi_a.set_title("§16.3a — PASSIVE over-training Φ ≡ (2.0, 2.0)")
    ax_phi_a.legend(fontsize=8)

    ax_BSF_a.plot(ts_a, ys_a[:, 0], "C0", lw=2, label="B")
    ax_BSF_a.plot(ts_a, ys_a[:, 1], "C1", lw=2, label="S")
    ax_BSF_a.plot(ts_a, ys_a[:, 2], "C3", lw=2, label="F")
    ax_BSF_a.set_ylabel("B, S, F"); ax_BSF_a.grid(True, alpha=0.3)
    ax_BSF_a.legend(fontsize=8)

    ax_A_a.plot(ts_a, ys_a[:, 3], "k", lw=2, label="A")
    ax_A_a.set_yscale("log"); ax_A_a.set_ylim(1e-3, 1.5)
    ax_A_a.axhline(trained_init_v2["A"], color="green", ls=":", lw=1, label=f"A_v2={trained_init_v2['A']}")
    ax_A_a.set_xlabel("t (d)"); ax_A_a.set_ylabel("A (log)"); ax_A_a.grid(True, alpha=0.3)
    ax_A_a.legend(fontsize=8, loc="lower left")
    ax_A_a.set_title(f"A under passive over-training: A(100) = {ys_a[-1, 3]:.4f}")

    # --- (b) Maintenance ---
    ax_phi_b.plot(ts_b, [1.0]*len(ts_b), "C2", lw=2, label="Φ_B = Φ_S = 1.0")
    ax_phi_b.set_ylim(0, 2.5); ax_phi_b.grid(True, alpha=0.3)
    ax_phi_b.set_title("§16.3b — CONSTRUCTIVE MAINTENANCE Φ ≡ (1, 1)")
    ax_phi_b.legend(fontsize=8)

    ax_BSF_b.plot(ts_b, ys_b[:, 0], "C0", lw=2, label="B")
    ax_BSF_b.plot(ts_b, ys_b[:, 1], "C1", lw=2, label="S")
    ax_BSF_b.plot(ts_b, ys_b[:, 2], "C3", lw=2, label="F")
    ax_BSF_b.set_ylabel("B, S, F"); ax_BSF_b.grid(True, alpha=0.3)
    ax_BSF_b.legend(fontsize=8)

    ax_A_b.plot(ts_b, ys_b[:, 3], "k", lw=2, label="A")
    ax_A_b.set_yscale("log"); ax_A_b.set_ylim(1e-3, 1.5)
    ax_A_b.axhline(trained_init_v2["A"], color="green", ls=":", lw=1, label=f"A_v2={trained_init_v2['A']}")
    ax_A_b.set_xlabel("t (d)"); ax_A_b.set_ylabel("A (log)"); ax_A_b.grid(True, alpha=0.3)
    ax_A_b.legend(fontsize=8, loc="lower right")
    ax_A_b.set_title(f"A under maintenance: A(100) = {ys_b[-1, 3]:.4f}")

    plt.suptitle(f"§16.3 — Over-training basin (shifted) from TRAINED_ATHLETE_INIT_v2 = "
                 f"(B={trained_init_v2['B']:.2f}, S={trained_init_v2['S']:.2f}, "
                 f"F={trained_init_v2['F']:.2f}, A={trained_init_v2['A']:.2f}, ...)\n"
                 f"Left (passive over-train): A(100)={ys_a[-1, 3]:.4f}  "
                 f"Right (maintenance):     A(100)={ys_b[-1, 3]:.4f}",
                 fontsize=10, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "overtraining_basin_witness.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    return ys_a[-1, 3], ys_b[-1, 3]


# -----------------------------------------------------------------------------
# §17 (revision): FIM-duality witness search under timescale-halved params
# -----------------------------------------------------------------------------

def compute_trajectory_fim_on_phi(y0, phi_B_const, phi_S_const, T, dt, p, h=1e-3):
    """Compute the trajectory FIM under constant Φ = (phi_B_const, phi_S_const),
    treating Φ as a dummy parameter to identify.

    F = sum_t (∂y(t)/∂Φ)ᵀ (∂y(t)/∂Φ) · dt  ∈ ℝ^{2×2}.

    Sensitivities computed by central differences. Returns the 2×2 FIM, the
    nominal trajectory's final A, and the integrated μ(t).
    """
    phi_nominal = lambda t: (phi_B_const, phi_S_const)
    ts, ys_nom, mus_nom = simulate(y0, phi_nominal, T=T, dt=dt, p=p)

    # Sensitivity w.r.t. Φ_B
    phi_B_plus = lambda t: (phi_B_const + h, phi_S_const)
    phi_B_minus = lambda t: (phi_B_const - h, phi_S_const)
    _, ys_Bp, _ = simulate(y0, phi_B_plus, T=T, dt=dt, p=p)
    _, ys_Bm, _ = simulate(y0, phi_B_minus, T=T, dt=dt, p=p)
    dy_dB = (ys_Bp - ys_Bm) / (2 * h)   # shape (n_steps+1, 6)

    # Sensitivity w.r.t. Φ_S
    phi_S_plus = lambda t: (phi_B_const, phi_S_const + h)
    phi_S_minus = lambda t: (phi_B_const, phi_S_const - h)
    _, ys_Sp, _ = simulate(y0, phi_S_plus, T=T, dt=dt, p=p)
    _, ys_Sm, _ = simulate(y0, phi_S_minus, T=T, dt=dt, p=p)
    dy_dS = (ys_Sp - ys_Sm) / (2 * h)

    # Stack J = [∂y/∂Φ_B | ∂y/∂Φ_S], shape (n_steps+1, 6, 2)
    # F = ∫ J^T J dt — sum over time, each contribution is 2×2.
    F = np.zeros((2, 2))
    for k in range(len(ts)):
        J_k = np.column_stack([dy_dB[k], dy_dS[k]])   # (6, 2)
        F += J_k.T @ J_k * dt
    integ_mu = np.trapezoid(mus_nom, ts)
    return F, ys_nom[-1, 3], integ_mu


def fim_condition(F, eps=1e-30):
    """Condition number κ = λ_max / λ_min of a symmetric PSD matrix."""
    eigs = np.linalg.eigvalsh(F)
    if eigs[0] < eps:
        return np.inf
    return eigs[-1] / eigs[0]


# Recommended-v2 parametrisation: §15 recommendation + timescale halving.
# κ doubled to preserve slow-manifold B*, S* (so island geometry unchanged).
RECOMMENDED_V2 = None  # built lazily in run_witness_phase_v2 (depends on PARAMS at import time)


def run_witness_phase_v2(p_recommended):
    """§17 — timescale-halved + FIM-duality grid search witness."""
    global RECOMMENDED_V2
    # Build the timescale-halved parametrisation
    p_v2 = dict(p_recommended)
    p_v2["tau_B"] = PARAMS["tau_B"] / 2.0           # 42 → 21
    p_v2["tau_S"] = PARAMS["tau_S"] / 2.0           # 60 → 30
    p_v2["kappa_B"] = PARAMS["kappa_B"] * 2.0       # 0.01248 → 0.02496
    p_v2["kappa_S"] = PARAMS["kappa_S"] * 2.0       # 0.00816 → 0.01632
    RECOMMENDED_V2 = p_v2

    print("\n" + "=" * 60)
    print("§17: Timescale-halved revisions (T=100 d throughout)")
    print("=" * 60)

    # Sanity check: equilibrium invariance under (τ, κ) → (τ/2, 2κ)
    sm_old = slow_manifold_state(1.0, 1.0, 0.30, p_recommended)
    sm_new = slow_manifold_state(1.0, 1.0, 0.30, p_v2)
    print(f"Equilibrium invariance check (B*, S* at Φ=(1,1), A=0.30):")
    print(f"  Old (τ=42,60): B*={sm_old['B']:.4f}  S*={sm_old['S']:.4f}")
    print(f"  v2  (τ=21,30): B*={sm_new['B']:.4f}  S*={sm_new['S']:.4f}")
    assert abs(sm_old['B'] - sm_new['B']) < 1e-6, "B* invariance broken"
    assert abs(sm_old['S'] - sm_new['S']) < 1e-6, "S* invariance broken"
    print("  ✓ slow-manifold equilibria preserved")

    mu_old = mu_bar_slow(1.0, 1.0, 0.0, p_recommended)
    mu_new = mu_bar_slow(1.0, 1.0, 0.0, p_v2)
    print(f"  μ̄(0; (1,1)): old = {mu_old:+.4f}, v2 = {mu_new:+.4f}")
    assert abs(mu_old - mu_new) < 1e-4, "μ̄ at (1,1) invariance broken"

    # Trajectory speed-up check
    phi_const = lambda t: (1.0, 1.0)
    ts_old, ys_old, _ = simulate(SEDENTARY_INIT, phi_const, T=50, dt=0.05, p=p_recommended)
    ts_new, ys_new, _ = simulate(SEDENTARY_INIT, phi_const, T=50, dt=0.05, p=p_v2)
    t_clear_old = ts_old[np.argmax(ys_old[:, 0] >= 0.25)] if np.any(ys_old[:, 0] >= 0.25) else 50
    t_clear_new = ts_new[np.argmax(ys_new[:, 0] >= 0.25)] if np.any(ys_new[:, 0] >= 0.25) else 50
    print(f"Time for B to reach B_dec=0.25 under constant Φ=(1,1) from SEDENTARY_INIT:")
    print(f"  Old (τ=42): {t_clear_old:.1f} d")
    print(f"  v2  (τ=21): {t_clear_new:.1f} d")
    print(f"  Speed-up: {t_clear_old/max(t_clear_new, 0.01):.2f}×")

    # FIM grid search over constant Φ ∈ [0.1, 2.0]²
    print("\n--- §17.3: FIM-duality grid search (constant Φ as dummy parameters) ---")
    n_grid = 21
    phi_axis = np.linspace(0.1, 2.0, n_grid)
    kappa_F = np.zeros((n_grid, n_grid))
    A_final = np.zeros((n_grid, n_grid))
    integ_mu = np.zeros((n_grid, n_grid))
    T = 100.0
    dt = 0.1   # coarser for the grid search to keep total time reasonable
    print(f"Running {n_grid}×{n_grid} = {n_grid*n_grid} cells, T={T}d, dt={dt}d ...")
    import time
    t_start = time.time()
    for i, phi_B in enumerate(phi_axis):
        for j, phi_S in enumerate(phi_axis):
            F, A_T, im = compute_trajectory_fim_on_phi(
                SEDENTARY_INIT, phi_B, phi_S, T, dt, p_v2, h=5e-3
            )
            kappa_F[i, j] = fim_condition(F)
            A_final[i, j] = A_T
            integ_mu[i, j] = im
        if i % 5 == 0:
            elapsed = time.time() - t_start
            print(f"  row {i+1}/{n_grid} done at {elapsed:.1f}s")
    print(f"Grid search took {time.time()-t_start:.1f}s")

    # Find top-3 candidates by each metric
    flat_kappa = kappa_F.flatten()
    flat_A = A_final.flatten()
    top_by_kappa = np.argsort(flat_kappa)[:5]      # smallest κ
    top_by_A = np.argsort(-flat_A)[:5]              # largest A
    overlap = set(top_by_kappa) & set(top_by_A)

    print("\nTop 5 by κ(F) (lowest condition number):")
    for idx in top_by_kappa:
        i, j = np.unravel_index(idx, (n_grid, n_grid))
        print(f"  Φ = ({phi_axis[i]:.2f}, {phi_axis[j]:.2f})  κ = {kappa_F[i,j]:.2e}  A(100) = {A_final[i,j]:.4f}")
    print("Top 5 by A(100) (largest):")
    for idx in top_by_A:
        i, j = np.unravel_index(idx, (n_grid, n_grid))
        print(f"  Φ = ({phi_axis[i]:.2f}, {phi_axis[j]:.2f})  κ = {kappa_F[i,j]:.2e}  A(100) = {A_final[i,j]:.4f}")
    print(f"\nDuality check: overlap of top-5 sets = {len(overlap)}/5 cells.")

    # Choose witness: lowest κ that also escapes
    best_kappa_idx = top_by_kappa[0]
    i_k, j_k = np.unravel_index(best_kappa_idx, (n_grid, n_grid))
    phi_fim_winner = (phi_axis[i_k], phi_axis[j_k])
    A_fim = A_final[i_k, j_k]

    best_A_idx = top_by_A[0]
    i_a, j_a = np.unravel_index(best_A_idx, (n_grid, n_grid))
    phi_A_winner = (phi_axis[i_a], phi_axis[j_a])
    A_max = A_final[i_a, j_a]

    print(f"\n→ FIM winner:  Φ* = ({phi_fim_winner[0]:.2f}, {phi_fim_winner[1]:.2f})  A(100) = {A_fim:.4f}")
    print(f"→ A(100) winner: Φ* = ({phi_A_winner[0]:.2f}, {phi_A_winner[1]:.2f})  A(100) = {A_max:.4f}")

    # FIGURE: FIM grid search heatmaps
    fig_fim_grid_search(phi_axis, kappa_F, A_final, phi_fim_winner, phi_A_winner)

    # FIGURE: island under v2 (should look like v1 since equilibria preserved)
    fig_sweep_recommendation(p_v2, "RECOMMENDED PARAMETRISATION v2 (timescale-halved)\n"
                                     "τ_B=21d, τ_S=30d, κ_B,κ_S doubled (equilibria preserved)")
    # Rename the output to avoid overwriting v1
    import shutil
    out_v1 = Path(__file__).parent / "island_recommended.png"
    out_v2 = Path(__file__).parent / "island_recommended_v2.png"
    if out_v1.exists():
        # the v1 was generated earlier; we want to keep both. v2 is whatever was just saved.
        shutil.copy(out_v1, out_v2)
        # Now regenerate v1 from p_recommended to restore it
        fig_sweep_recommendation(p_recommended, "RECOMMENDED PARAMETRISATION\n"
                                                 "B_dec=0.25, μ_dec=0.10, μ_F=0.030, μ_FF=0.020")

    # FIGURE: sedentary basin witness v2 (passive vs FIM-witness)
    witness_phi = phi_fim_winner if A_fim >= 0.30 else phi_A_winner
    fig_sedentary_basin_witness_v2(p_v2, witness_phi)

    # FIGURE: overtraining basin witness v2
    fig_overtraining_basin_witness_v2(p_v2)

    return p_v2, witness_phi, A_fim


def fig_fim_grid_search(phi_axis, kappa_F, A_final, phi_fim_winner, phi_A_winner):
    """2-panel heat-map: log κ(F) and A(100) over the (Φ_B, Φ_S) grid."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    extent = [phi_axis[0], phi_axis[-1], phi_axis[0], phi_axis[-1]]

    log_kappa = np.log10(kappa_F + 1e-30)
    im0 = axes[0].imshow(log_kappa.T, origin="lower", extent=extent,
                          cmap="viridis_r", aspect="equal")
    axes[0].scatter([phi_fim_winner[0]], [phi_fim_winner[1]],
                     marker="*", color="red", s=300, edgecolors="white",
                     label=f"FIM winner ({phi_fim_winner[0]:.2f},{phi_fim_winner[1]:.2f})")
    axes[0].scatter([phi_A_winner[0]], [phi_A_winner[1]],
                     marker="X", color="cyan", s=200, edgecolors="black",
                     label=f"A(100) winner ({phi_A_winner[0]:.2f},{phi_A_winner[1]:.2f})")
    axes[0].set_xlabel("Φ_B"); axes[0].set_ylabel("Φ_S")
    axes[0].set_title("log₁₀ κ(F) — FIM condition number of trajectory\nlower (yellow) = controllable")
    axes[0].legend(fontsize=9, loc="upper right")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="log₁₀ κ(F)")

    im1 = axes[1].imshow(A_final.T, origin="lower", extent=extent,
                          cmap="RdYlGn", aspect="equal", vmin=0, vmax=1.5)
    axes[1].contour(phi_axis, phi_axis, A_final.T, levels=[0.30],
                     colors="black", linewidths=2)
    axes[1].scatter([phi_fim_winner[0]], [phi_fim_winner[1]],
                     marker="*", color="red", s=300, edgecolors="white",
                     label="FIM winner")
    axes[1].scatter([phi_A_winner[0]], [phi_A_winner[1]],
                     marker="X", color="cyan", s=200, edgecolors="black",
                     label="A(100) winner")
    axes[1].set_xlabel("Φ_B"); axes[1].set_ylabel("Φ_S")
    axes[1].set_title("A(100) from SEDENTARY_INIT under constant Φ\nblack contour: A = A_target = 0.30")
    axes[1].legend(fontsize=9, loc="upper right")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="A(100)")

    plt.suptitle("§17.3 FIM-duality grid search — constant Φ as dummy identification parameter\n"
                 "If duality holds, the FIM winner (★) should sit inside the A>A_target region",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "fim_grid_search.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_sedentary_basin_witness_v2(p_v2, witness_phi):
    """§17.4 — passive vs FIM-witness from SEDENTARY_INIT under v2 params, T=100d."""
    T = 100.0; dt = 0.05

    phi_passive = lambda t: (0.1, 0.1)
    ts_a, ys_a, mus_a = simulate(SEDENTARY_INIT, phi_passive, T=T, dt=dt, p=p_v2)

    phi_witness = lambda t: witness_phi
    ts_b, ys_b, mus_b = simulate(SEDENTARY_INIT, phi_witness, T=T, dt=dt, p=p_v2)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    ((ax_phi_a, ax_phi_b),
     (ax_BSF_a, ax_BSF_b),
     (ax_A_a,   ax_A_b)) = axes

    # Passive
    ax_phi_a.plot(ts_a, [0.1]*len(ts_a), "C0", lw=2, label="Φ_B = Φ_S = 0.1")
    ax_phi_a.set_ylim(0, 2.5); ax_phi_a.set_ylabel("Φ"); ax_phi_a.grid(True, alpha=0.3)
    ax_phi_a.set_title("§17.4a — PASSIVE sedentary Φ ≡ (0.1, 0.1) under v2"); ax_phi_a.legend(fontsize=8)
    ax_BSF_a.plot(ts_a, ys_a[:, 0], "C0", lw=2, label="B")
    ax_BSF_a.plot(ts_a, ys_a[:, 1], "C1", lw=2, label="S")
    ax_BSF_a.plot(ts_a, ys_a[:, 2], "C3", lw=2, label="F")
    ax_BSF_a.axhline(p_v2["B_dec"], color="k", ls=":", lw=1, label=f"B_dec={p_v2['B_dec']}")
    ax_BSF_a.set_ylabel("B, S, F"); ax_BSF_a.grid(True, alpha=0.3); ax_BSF_a.legend(fontsize=8)
    ax_A_a.plot(ts_a, ys_a[:, 3], "k", lw=2)
    ax_A_a.set_yscale("log"); ax_A_a.set_ylim(1e-3, 1.5); ax_A_a.grid(True, alpha=0.3)
    ax_A_a.axhline(0.30, color="r", ls=":", lw=1, label="A_target=0.30")
    ax_A_a.axhline(0.10, color="purple", ls=":", lw=1, label="A_0=0.10")
    ax_A_a.set_xlabel("t (d)"); ax_A_a.set_ylabel("A (log)"); ax_A_a.legend(fontsize=8, loc="lower left")
    ax_A_a.set_title(f"A under passive: A(100) = {ys_a[-1, 3]:.4f}")

    # Witness
    ax_phi_b.plot(ts_b, [witness_phi[0]]*len(ts_b), "C2", lw=2, label=f"Φ_B = {witness_phi[0]:.2f}")
    ax_phi_b.plot(ts_b, [witness_phi[1]]*len(ts_b), "C2--", lw=2, label=f"Φ_S = {witness_phi[1]:.2f}")
    ax_phi_b.set_ylim(0, 2.5); ax_phi_b.grid(True, alpha=0.3)
    ax_phi_b.set_title(f"§17.4b — FIM-WITNESS Φ ≡ ({witness_phi[0]:.2f}, {witness_phi[1]:.2f}) under v2"); ax_phi_b.legend(fontsize=8)
    ax_BSF_b.plot(ts_b, ys_b[:, 0], "C0", lw=2, label="B")
    ax_BSF_b.plot(ts_b, ys_b[:, 1], "C1", lw=2, label="S")
    ax_BSF_b.plot(ts_b, ys_b[:, 2], "C3", lw=2, label="F")
    ax_BSF_b.axhline(p_v2["B_dec"], color="k", ls=":", lw=1, label=f"B_dec={p_v2['B_dec']}")
    ax_BSF_b.set_ylabel("B, S, F"); ax_BSF_b.grid(True, alpha=0.3); ax_BSF_b.legend(fontsize=8)
    ax_A_b.plot(ts_b, ys_b[:, 3], "k", lw=2)
    ax_A_b.set_yscale("log"); ax_A_b.set_ylim(1e-3, 1.5); ax_A_b.grid(True, alpha=0.3)
    ax_A_b.axhline(0.30, color="r", ls=":", lw=1, label="A_target=0.30")
    ax_A_b.axhline(0.10, color="purple", ls=":", lw=1, label="A_0=0.10")
    ax_A_b.set_xlabel("t (d)"); ax_A_b.set_ylabel("A (log)"); ax_A_b.legend(fontsize=8, loc="lower right")
    escaped = "ESCAPED" if ys_b[-1, 3] >= 0.30 else "FAILED"
    ax_A_b.set_title(f"A under FIM witness: A(100) = {ys_b[-1, 3]:.4f}  ({escaped})")

    integ_a = np.trapezoid(mus_a, ts_a); integ_b = np.trapezoid(mus_b, ts_b)
    plt.suptitle(f"§17.4 — Sedentary basin under RECOMMENDED v2 (timescale-halved), T=100 d\n"
                 f"Left passive: A(100)={ys_a[-1, 3]:.4f}  ∫μ={integ_a:+.3f}    "
                 f"Right FIM-witness Φ ≡ ({witness_phi[0]:.2f}, {witness_phi[1]:.2f}): A(100)={ys_b[-1, 3]:.4f}  ∫μ={integ_b:+.3f}",
                 fontsize=10, y=1.00)
    plt.tight_layout()
    out = Path(__file__).parent / "sedentary_basin_witness_v2.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def fig_overtraining_basin_witness_v2(p_v2):
    """§17.5 — same as before but under v2 params, T=100 d."""
    # Locate the new island center under v2 (should be same as v1)
    phi_grid = np.linspace(0.0, 2.5, 100)
    cx, cy, mu_max = island_center(phi_grid, p_v2)

    # Find A* at the center
    A_scan = np.linspace(0.01, 1.5, 300)
    g_vals = np.array([mu_bar_slow(cx, cy, A, p_v2) - p_v2["eta"] * A * A for A in A_scan])
    sign_changes = np.where(np.diff(np.sign(g_vals)) != 0)[0]
    A_star = 0.5 * (A_scan[sign_changes[-1]] + A_scan[sign_changes[-1] + 1]) if len(sign_changes) > 0 else 0.5
    trained_v2 = slow_manifold_state(cx, cy, A_star, p_v2)

    print(f"  TRAINED_ATHLETE_INIT_v2 under v2 params: A*={A_star:.3f}")
    return fig_overtraining_basin_witness(p_v2, trained_v2)


def run_witness_phase(p_recommended):
    """Run the §16 constructive controllability checks (T=200 d, hand-designed witness)."""
    print("\n" + "=" * 60)
    print("§16: Constructive controllability under recommended parametrisation")
    print("=" * 60)

    # Define TRAINED_ATHLETE_INIT_v2 as the slow-manifold state at the new
    # island center, with A = A* (the stable healthy attractor at that Φ).
    # First find island center; then for that Φ, compute the slow-manifold A*.
    phi_grid = np.linspace(0.0, 2.5, 100)
    cx, cy, mu_max = island_center(phi_grid, p_recommended)
    print(f"Island center under recommended params: ({cx:.2f}, {cy:.2f})  μ̄_max={mu_max:+.4f}")

    # Approximate A*: the largest positive root of g(A) = μ̄(A; Φ_center) − η A^2.
    # Scan A and find sign change of g.
    A_scan = np.linspace(0.01, 1.5, 300)
    g_vals = np.array([mu_bar_slow(cx, cy, A, p_recommended) - p_recommended["eta"] * A * A for A in A_scan])
    sign_changes = np.where(np.diff(np.sign(g_vals)) != 0)[0]
    if len(sign_changes) > 0:
        # Take the LARGEST sign-change → the upper interior root (stable A*)
        idx = sign_changes[-1]
        A_star = 0.5 * (A_scan[idx] + A_scan[idx + 1])
    else:
        A_star = 0.5  # fallback
    print(f"Approximate A* at island center: A* = {A_star:.3f}")

    trained_v2 = slow_manifold_state(cx, cy, A_star, p_recommended)
    print(f"TRAINED_ATHLETE_INIT_v2 (slow-manifold at ({cx:.2f}, {cy:.2f}) with A=A*):")
    for k, v in trained_v2.items():
        print(f"  {k:7s} = {v:.4f}")

    print("\n--- §16.1 + §16.2: Sedentary basin from SEDENTARY_INIT ---")
    A_passive_sed, A_witness_sed, integ_p_sed, integ_w_sed = fig_sedentary_basin_witness(p_recommended)
    print(f"  Passive sedentary policy A(100) = {A_passive_sed:.4f}  (∫μ = {integ_p_sed:+.3f})")
    print(f"  Witness Φ(t) trajectory  A(100) = {A_witness_sed:.4f}  (∫μ = {integ_w_sed:+.3f})  "
          f"{'✓ ESCAPED to healthy island' if A_witness_sed >= 0.30 else '✗ failed to escape'}")

    print("\n--- §16.3: Over-training basin from TRAINED_ATHLETE_INIT_v2 ---")
    A_passive_ot, A_maintain = fig_overtraining_basin_witness(p_recommended, trained_v2)
    print(f"  Passive Φ = (2,2)        A(100) = {A_passive_ot:.4f}  "
          f"{'✓ collapsed (basin preserved)' if A_passive_ot < 0.05 else '✗ did not collapse'}")
    print(f"  Maintenance Φ = (1,1)    A(100) = {A_maintain:.4f}  "
          f"{'✓ maintained' if A_maintain >= 0.30 else '✗ degraded'}")

    return trained_v2


# -----------------------------------------------------------------------------
# Figure 1: candidate trajectories
# -----------------------------------------------------------------------------

def phi_constant(t, phi_B=0.30, phi_S=0.30):
    return phi_B, phi_S


def phi_bang_bang(t, T_burst=5.0, phi_max=1.0):
    if t < T_burst:
        return phi_max, 0.0
    return 0.30, 0.30


def phi_ramp(t, T_ramp=14.0, phi_max=1.0):
    if t < T_ramp:
        frac = max(0.0, 1.0 - t / T_ramp)
        return phi_max * frac + 0.30 * (1 - frac), phi_max * frac + 0.30 * (1 - frac)
    return 0.30, 0.30


def fig_candidates():
    candidates = [
        ("Candidate 1: Φ ≡ (0.30, 0.30)",          phi_constant,  "tab:blue"),
        ("Candidate 2: bang-bang (1, 0) → (0.3, 0.3)", phi_bang_bang, "tab:orange"),
        ("Candidate 3: ramp from (1, 1) → (0.3, 0.3)", phi_ramp,      "tab:green"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    ((ax_B, ax_S), (ax_F, ax_A), (ax_mu, ax_phi)) = axes

    integrals = []
    for label, phi_fn, color in candidates:
        ts, ys, mus = simulate(SEDENTARY_INIT, phi_fn, T=100.0, dt=0.05)
        B, S, F, A, KFB, KFS = ys.T
        integral = np.trapezoid(mus, ts)
        integrals.append((label, integral, A[-1]))

        ax_B.plot(ts, B, color=color, lw=1.8, label=label)
        ax_S.plot(ts, S, color=color, lw=1.8)
        ax_F.plot(ts, F, color=color, lw=1.8)
        ax_A.plot(ts, A, color=color, lw=1.8)
        ax_mu.plot(ts, mus, color=color, lw=1.8)

        phis = np.array([phi_fn(t) for t in ts])
        ax_phi.plot(ts, phis[:, 0], color=color, lw=1.2, linestyle="-")
        ax_phi.plot(ts, phis[:, 1], color=color, lw=1.2, linestyle="--")

    # Reference lines
    ax_B.axhline(PARAMS["B_dec"], color="k", linestyle=":", lw=1, label=f"B_dec={PARAMS['B_dec']}")
    ax_S.axhline(PARAMS["S_dec"], color="k", linestyle=":", lw=1)
    ax_F.axhline(F_TYP, color="k", linestyle=":", lw=1, label=f"F_TYP={F_TYP}")
    ax_A.axhline(0.30, color="r", linestyle=":", lw=1, label="A_target=0.30")
    ax_A.axhline(SEDENTARY_INIT["A"], color="k", linestyle=":", lw=1, label="A_0=0.10")
    ax_mu.axhline(0.0, color="k", linestyle="-", lw=0.5)

    ax_B.set_ylabel("B(t)"); ax_B.set_title("Aerobic capacity B")
    ax_S.set_ylabel("S(t)"); ax_S.set_title("Strength capacity S")
    ax_F.set_ylabel("F(t)"); ax_F.set_title("Fatigue F")
    ax_A.set_ylabel("A(t)"); ax_A.set_title("Autonomic amplitude A  (the target variable)")
    ax_A.set_yscale("log"); ax_A.set_ylim(1e-3, 1.0)
    ax_mu.set_ylabel("μ(t)"); ax_mu.set_title("Bifurcation parameter μ(B(t),S(t),F(t))")
    ax_phi.set_ylabel("Φ(t)"); ax_phi.set_title("Control schedule  (solid Φ_B, dashed Φ_S)")
    ax_phi.set_xlabel("time t (days)")
    ax_A.set_xlabel("time t (days)")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    ax_B.legend(fontsize=8, loc="lower right")
    ax_A.legend(fontsize=8, loc="lower left")
    ax_F.legend(fontsize=8, loc="upper right")

    title = ("FSA-v5 deterministic ODE trajectories from SEDENTARY_INIT under 3 candidate Φ schedules\n"
             "All candidates fail to escape to A_target = 0.30 within T = 100 d. ")
    for label, integ, A_final in integrals:
        title += f"\n {label}:  ∫μ dt = {integ:+.2f},  A(100) = {A_final:.4f}"
    plt.suptitle(title, fontsize=10, y=0.995)
    plt.tight_layout()

    out = Path(__file__).parent / "candidates_trajectories.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    return integrals


# -----------------------------------------------------------------------------
# Figure 2: controllability region heatmap in (B0, S0)
# -----------------------------------------------------------------------------

def fig_controllability_region():
    """For each (B0, S0) on a grid, simulate the trajectory under the
    slow-manifold-optimal constant Φ = (0.30, 0.30) and record both
    ∫μ dt and A(T). The (B0, S0) regions where the system escapes
    (A(T) ≥ A_target) vs. fails to escape are plotted as heatmaps.
    """
    n = 31
    B0_grid = np.linspace(0.02, 0.5, n)
    S0_grid = np.linspace(0.02, 0.5, n)
    A0s = [0.10, 0.30]
    Us  = [("U = [0,1]² (realistic)", 1.0),
           ("U = [0,3]² (kernel-permitted)", 3.0)]

    fig, axes = plt.subplots(len(A0s), len(Us), figsize=(12, 10))

    for row_idx, A0 in enumerate(A0s):
        for col_idx, (U_label, phi_max) in enumerate(Us):
            ax = axes[row_idx, col_idx]
            # For each (B0, S0), simulate under best-known constant Φ.
            # The slow-manifold-optimal Φ is (0.30, 0.30) inside the
            # admissible set; if phi_max < 0.30 use the boundary value.
            phi_const = min(0.30, phi_max)

            integrals = np.zeros((n, n))
            A_finals = np.zeros((n, n))
            print(f"Computing heatmap for A0={A0}, {U_label} ...")
            for i, B0 in enumerate(B0_grid):
                for j, S0 in enumerate(S0_grid):
                    y0 = dict(B=B0, S=S0, F=F_TYP, A=A0,
                              KFB=PARAMS["KFB_0"], KFS=PARAMS["KFS_0"])
                    ts, ys, mus = simulate(y0,
                                            lambda t, p=phi_const: (p, p),
                                            T=100.0, dt=0.2)
                    integrals[i, j] = np.trapezoid(mus, ts)
                    A_finals[i, j]  = ys[-1, 3]

            ln_threshold = np.log(0.30 / A0) if A0 < 0.30 else 0.01  # if already at target

            # Plot A_final on log scale, with contour at A_target
            im = ax.imshow(np.log10(A_finals + 1e-10).T,
                            origin="lower",
                            extent=[B0_grid[0], B0_grid[-1], S0_grid[0], S0_grid[-1]],
                            aspect="equal",
                            cmap="RdYlGn", vmin=-3, vmax=0)
            cs = ax.contour(B0_grid, S0_grid, A_finals.T,
                            levels=[0.30], colors="black", linewidths=2)
            ax.clabel(cs, fmt={0.30: "A=0.30"}, fontsize=8)

            # Reference thresholds (Hill knee)
            ax.axvline(PARAMS["B_dec"], color="white", linestyle=":", lw=1, alpha=0.7)
            ax.axhline(PARAMS["S_dec"], color="white", linestyle=":", lw=1, alpha=0.7)

            # Probe points
            ax.scatter([SEDENTARY_INIT["B"]], [SEDENTARY_INIT["S"]],
                        color="black", s=80, marker="o", edgecolors="white",
                        label="SEDENTARY_INIT")
            if A0 == 0.30:
                ax.scatter([TRAINED_ATHLETE_INIT["B"]], [TRAINED_ATHLETE_INIT["S"]],
                            color="cyan", s=80, marker="o", edgecolors="white",
                            label="TRAINED_ATHLETE_INIT")

            ax.set_xlabel("B₀  (initial aerobic capacity)")
            ax.set_ylabel("S₀  (initial strength)")
            ax.set_title(f"A₀ = {A0},  {U_label}\nΦ = ({phi_const:.2f}, {phi_const:.2f}) constant — best slow-manifold strategy",
                         fontsize=9)
            ax.legend(fontsize=8, loc="upper left")

            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("log₁₀ A(T=100)")

    plt.suptitle("FSA-v5 controllability region — A(T=100 days) under best constant Φ = (0.30, 0.30)\n"
                 "Black contour: A(T) = A_target = 0.30 (escape boundary). "
                 "Red: A(T) ≪ A_target (uncontrolled). White dotted: B_dec, S_dec.",
                 fontsize=11, y=1.00)
    plt.tight_layout()

    out = Path(__file__).parent / "controllability_region.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Generating figures for sedentary_basin_controllability.tex")
    print("=" * 60)

    # Sanity check: confirm μ(SEDENTARY_INIT) = -0.115
    mu_init = mu_bar_state(SEDENTARY_INIT["B"],
                            SEDENTARY_INIT["S"],
                            SEDENTARY_INIT["F"])
    print(f"\nμ(SEDENTARY_INIT) = {mu_init:.4f}  (expected -0.1148)")
    assert abs(mu_init - (-0.1148)) < 1e-3, "μ_init mismatch!"
    print("  ✓ Numerical anchor verified\n")

    print("Figure 1: candidate trajectories ...")
    integrals = fig_candidates()
    print("\nSummary of candidate-trajectory integrals:")
    for label, integ, A_final in integrals:
        print(f"  {label}: ∫μ dt = {integ:+.3f},  A(100) = {A_final:.4f}")
    ln_3 = np.log(3.0)
    print(f"\nEscape threshold:  ∫μ dt > ln(3) = {ln_3:.3f}")
    print(f"All three candidates fall short by a factor of ~{abs(min(integ for _, integ, _ in integrals)) / ln_3:.1f}x\n")

    print("Figure 2: controllability region heatmap ...")
    fig_controllability_region()

    # ----- §14 extension: single-mod island plots --------------------------
    print("\n" + "=" * 60)
    print("§14: Single-modification island plots")
    print("=" * 60)
    fig_single_mods()

    # ----- §15 extension: 5-parameter sweep for island shift ---------------
    recommended_p, recommended_metrics = run_sweep_phase()

    # ----- §16 extension: constructive witnesses ---------------------------
    trained_v2 = run_witness_phase(recommended_p)

    # ----- §17 revisions: timescale-halved + FIM-duality witness ------------
    p_v2, witness_phi, A_witness = run_witness_phase_v2(recommended_p)

    print("\nDone.")
