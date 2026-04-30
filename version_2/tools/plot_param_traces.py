"""Plot posterior parameter traces vs truth across rolling windows.

Reads a run dir produced by bench_smc_full_mpc_fsa.py:
  manifest.json     — param_names + truth_params + n_strides + step_minutes
  data.npz          — posterior_particles (n_strides, n_smc, n_params),
                      posterior_window_mask (n_strides,)

Writes <run_dir>/E5_full_mpc_T<T>d_param_traces.png — one panel per
parameter: posterior median + 5/95 quantile band over windows, with the
truth value overlaid as a horizontal line.

Usage:
    python -m tools.plot_param_traces <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(run_dir: str) -> None:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    data = np.load(run_dir / "data.npz")

    param_names = list(manifest["param_names"])
    truth = manifest["truth_params"]
    n_strides = int(manifest["n_strides"])
    step_min = int(manifest["step_minutes"])
    T_days = int(manifest["T_total_days"])

    posterior = data["posterior_particles"]      # (n_strides, n_smc, n_params)
    mask = data["posterior_window_mask"]         # (n_strides,)

    bins_per_day = (24 * 60) // step_min
    stride_bins = int(manifest["STRIDE_BINS"])
    window_bins = int(manifest["WINDOW_BINS"])
    end_t_days = (np.arange(n_strides) * stride_bins + window_bins) / bins_per_day
    end_t_days = end_t_days[mask]

    valid = posterior[mask]                       # (n_valid, n_smc, n_params)
    q05 = np.quantile(valid, 0.05, axis=1)        # (n_valid, n_params)
    q50 = np.quantile(valid, 0.50, axis=1)
    q95 = np.quantile(valid, 0.95, axis=1)

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.0, n_rows * 2.2),
                              sharex=True)
    axes = axes.flatten()

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.fill_between(end_t_days, q05[:, i], q95[:, i], color="C0", alpha=0.3,
                         label="5-95%")
        ax.plot(end_t_days, q50[:, i], color="C0", lw=1.4, label="median")
        if name in truth:
            ax.axhline(truth[name], color="red", lw=1.0, ls="--", label="truth")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_params, len(axes)):
        axes[j].axis("off")

    for ax in axes[-n_cols:]:
        ax.set_xlabel("end of window (days)", fontsize=8)

    summary = manifest.get("summary", {})
    ratio = summary.get("mean_A_mpc", 0) / max(summary.get("mean_A_baseline", 1), 1e-9)
    fviol = 100.0 * summary.get("F_violation_frac_mpc", 0)
    idcov = summary.get("n_windows_pass_id_cov_5_of_6", 0)
    fig.suptitle(
        f"FSA-v2 posterior parameter traces — T={T_days}d, h={step_min}min, "
        f"n_strides={n_strides}, ratio={ratio:.2f}, F-viol={fviol:.1f}%, "
        f"id-cov={idcov}/{n_strides}",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = run_dir / f"E5_full_mpc_T{T_days}d_param_traces.png"
    fig.savefig(out_path, dpi=120)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m tools.plot_param_traces <run_dir>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
