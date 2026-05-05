"""Basin-overlay diagnostic plot: applied (Phi_B, Phi_S) path on the v5 closed-island.

Mirrors Figure 2 (`fig:full-bifurcation`) of LaTex_docs/FSA_version_5_technical_guide.tex.
Builds the regime classification numerically via `_jax_find_A_sep` from
`control_v5.py` (returns -inf / finite / +inf for healthy / bistable /
collapsed at each (Phi_B, Phi_S) grid point).

Overlays the controller's chosen daily-mean (Phi_B, Phi_S) path:
  * START = green dot at first applied (typically the scenario's baseline_phi)
  * PATH  = blue line connecting each daily mean
  * END   = red dot at last applied (should land inside the healthy island
            for a working controller)

Used by both Stage 2 and Stage 3 benches as their final save step. Run
standalone too:

    python tools/plot_basin_overlay.py outputs/fsa_v5/experiments/runNN_<tag>/

The plot is dropped into the run dir as ``basin_overlay.png``.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _classify_regime_grid(n_grid: int = 41,
                            phi_max: float = 1.5,
                            params: dict | None = None) -> tuple:
    """Build a (Phi_B, Phi_S) grid and classify each cell into one of three regimes.

    Returns (Phi_B_axis, Phi_S_axis, regime_codes) where:
      * regime_codes shape (n_grid, n_grid) with codes:
        0 = mono-stable healthy   (no separatrix, A=0 unstable)
        1 = bistable annulus      (finite separatrix)
        2 = mono-stable collapsed (no positive root, A=0 globally stable)

    Uses the JAX-jittable `_jax_find_A_sep` so a 41x41 grid takes ~2-3 s
    on GPU, ~5-10 s on CPU.
    """
    from version_3.models.fsa_v5.control_v5 import _jax_find_A_sep
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5

    if params is None:
        params = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS_V5.items()}

    phi_axis = jnp.linspace(0.0, phi_max, n_grid)
    PHI_B, PHI_S = jnp.meshgrid(phi_axis, phi_axis, indexing='xy')

    def find_one(pb, ps):
        return _jax_find_A_sep(pb, ps, params)

    sep_grid = jax.vmap(jax.vmap(find_one))(PHI_B, PHI_S)

    # Classify
    regime = jnp.where(jnp.isneginf(sep_grid), 0,
                       jnp.where(jnp.isposinf(sep_grid), 2, 1))
    return np.asarray(phi_axis), np.asarray(phi_axis), np.asarray(regime)


def plot_basin_overlay(applied_phi: np.ndarray,
                        out_path: Path | str,
                        *,
                        title: str | None = None,
                        baseline_phi: tuple | None = None,
                        n_grid: int = 41,
                        phi_max: float = 1.5):
    """Render the basin-overlay diagnostic.

    Args:
        applied_phi: shape (n_strides, 2) of (Phi_B, Phi_S) per stride.
            Will be averaged across daily blocks of `replan_K` strides
            implicitly by passing the daily-mean array (caller's choice).
        out_path: where to save the PNG.
        title: optional plot title (e.g. the run-tag).
        baseline_phi: optional 2-tuple (Phi_B0, Phi_S0) for a hollow
            "scenario baseline" marker -- useful when applied_phi[0] is
            already a controller-chosen value rather than the bench
            baseline.
        n_grid: regime-classification grid resolution.
        phi_max: maximum Phi axis value for both axes.
    """
    pb_axis, ps_axis, regime = _classify_regime_grid(n_grid=n_grid, phi_max=phi_max)

    fig, ax = plt.subplots(figsize=(8, 7))

    # Three-tone colourmap: healthy (light green), bistable (yellow), collapsed (red).
    cmap = mcolors.ListedColormap(['#a6dba0', '#fee08b', '#f4a582'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(regime, origin='lower',
              extent=[pb_axis[0], pb_axis[-1], ps_axis[0], ps_axis[-1]],
              cmap=cmap, norm=norm, aspect='auto', alpha=0.6)

    # Contour the boundaries
    ax.contour(pb_axis, ps_axis, regime, levels=[0.5, 1.5],
               colors=['black', 'magenta'], linewidths=[1.2, 1.0],
               linestyles=['solid', 'dashed'])

    # Overlay applied path
    applied_phi = np.atleast_2d(applied_phi)
    if applied_phi.shape[1] != 2:
        raise ValueError(f"applied_phi must be (n_strides, 2); got {applied_phi.shape}")

    ax.plot(applied_phi[:, 0], applied_phi[:, 1], 'o-', color='steelblue',
            lw=1.8, markersize=4, alpha=0.85, label='controller path')
    # Start + end markers
    ax.scatter([applied_phi[0, 0]], [applied_phi[0, 1]],
               s=180, c='#1b7837', edgecolors='black', linewidths=1.2,
               zorder=5, label='start')
    ax.scatter([applied_phi[-1, 0]], [applied_phi[-1, 1]],
               s=180, c='#762a83', edgecolors='black', linewidths=1.2,
               zorder=5, label='end')
    # Optional scenario-baseline marker
    if baseline_phi is not None:
        ax.scatter([baseline_phi[0]], [baseline_phi[1]],
                   s=140, marker='X', facecolors='none',
                   edgecolors='black', linewidths=1.5,
                   zorder=4, label=f'scenario baseline {baseline_phi}')

    # Region labels
    ax.text(0.20, 0.20, 'healthy\nisland', ha='center', fontsize=10,
            color='#205c39', fontweight='bold')
    ax.text(1.15, 0.10, 'collapsed', ha='center', fontsize=10,
            color='#9c2d1a', fontweight='bold')
    ax.text(0.55, 0.85, 'bistable\nannulus', ha='center', fontsize=9,
            color='#7c5e00', fontweight='bold')

    ax.set_xlabel(r'$\Phi_B$  (aerobic stimulus)')
    ax.set_ylabel(r'$\Phi_S$  (strength stimulus)')
    ax.set_title(title or 'Basin overlay: applied (Phi_B, Phi_S) path')
    ax.set_xlim(0.0, phi_max)
    ax.set_ylim(0.0, phi_max)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    out_path = Path(out_path)
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


# ── CLI helper: plot from a saved run dir's trajectory.npz ──────────

def _plot_from_run_dir(run_dir: Path):
    """Load applied_phi_per_stride from a run dir's trajectory.npz and plot."""
    import json
    npz_path = run_dir / "trajectory.npz"
    if not npz_path.exists():
        sys.exit(f"trajectory.npz not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    if 'applied_phi_per_stride' not in data:
        sys.exit(f"trajectory.npz missing 'applied_phi_per_stride' key: "
                  f"{list(data.keys())}")
    applied = data['applied_phi_per_stride']

    # Try to read scenario + cost-variant info from manifest.json for title
    title = run_dir.name
    baseline_phi = None
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        sce = m.get('scenario', {})
        cv = m.get('cost_variant', '?')
        title = (f"{run_dir.name}\n"
                 f"cost={cv}, scenario={sce.get('name', '?')}")
        bp = sce.get('baseline_phi')
        if bp is not None:
            baseline_phi = tuple(bp)

    out = plot_basin_overlay(applied, run_dir / "basin_overlay.png",
                             title=title, baseline_phi=baseline_phi)
    print(f"Wrote {out}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Usage: python tools/plot_basin_overlay.py <run-dir>")
    _plot_from_run_dir(Path(sys.argv[1]).resolve())
