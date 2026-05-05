"""Print a CHANGELOG-ready summary of an experiment's manifest.json.

Usage:
    python tools/summarize_run.py outputs/fsa_v5/experiments/runNN_<tag>/

The output is markdown, ready to paste under a `## Run NN ...` header in
`outputs/fsa_v5/CHANGELOG.md`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _format_pct(x: float | int, total: float | int) -> str:
    return f"{x}/{total} ({100.0*x/total:.0f}%)"


def summarise_stage1(m: dict) -> str:
    s = m["summary"]
    sce = m["scenario"]
    n_id = m["n_estimable_params"]
    n_w = m["n_windows"]
    cov_thresh = s["id_cov_threshold"]
    pass_thresh = s["pass_threshold_windows"]
    n_pass = s["n_windows_pass_id_cov"]
    final = s["trajectory_summary"]
    per_param = s.get("per_param_coverage_frac", {})
    well = sum(1 for v in per_param.values() if v >= 0.5) if per_param else None
    poorly = sum(1 for v in per_param.values() if v < 0.5) if per_param else None

    lines = [
        f"**Run {m['run_number']:02d} -- Stage 1 verification "
        f"(filter+plant only, {sce['name']} scenario)**",
        f"",
        f"  Driver: `bench_smc_filter_only_fsa_v5.py`",
        f"  Run dir: `experiments/run{m['run_number']:02d}_{m['run_tag']}/`",
        f"  Pin: `{m['fsa_model_dev_pin'][:7]}` (FSA_model_dev/claude/dev-sandbox-v4)",
        f"  Scenario: {sce['description']}",
        f"  T={m['T_total_days']}d, window={m['WINDOW_BINS']}b, stride={m['STRIDE_BINS']}b, "
        f"{n_w} rolling windows on {s['device']}",
        f"  Compute: {s['total_compute_s']:.0f}s = {s['total_compute_min']:.1f}min",
        f"",
        f"  **Final state:**  "
        + ", ".join(f"{k.replace('_final',''):s}={v:.3f}"
                    for k, v in final.items()),
        f"",
        f"  **Coverage gate:** {_format_pct(n_pass, n_w)} windows have "
        f">= {cov_thresh}/{n_id} estimable params with truth in 90% CI "
        f"(threshold: >= {pass_thresh}/{n_w} windows).",
    ]
    if well is not None:
        lines.append(
            f"  **Per-param breakdown:** {well}/{n_id} params covered in "
            f">= 50% of windows; {poorly}/{n_id} struggling (cov < 50%)."
        )
    lines.append("")
    lines.append("  **Gates:**")
    for k, v in s["gates"].items():
        lines.append(f"  - {'PASS' if v else 'FAIL'} `{k}`")
    lines.append("")
    return "\n".join(lines)


def summarise_stage2(m: dict) -> str:
    s = m["summary"]
    sce = m["scenario"]
    cv = m["cost_variant"]

    lines = [
        f"**Run {m['run_number']:02d} -- Stage 2 verification "
        f"({cv}, {sce['name']} scenario)**",
        f"",
        f"  Driver: `bench_controller_only_fsa_v5.py`",
        f"  Run dir: `experiments/run{m['run_number']:02d}_{m['run_tag']}/`",
        f"  Cost variant: `{cv}`"
        + (f"  (beta={m['cost_kwargs']['beta']})" if cv == 'soft' else ''),
        f"  Scenario: {sce['description']}",
        f"  T={m['T_total_days']}d, replan_K={m['replan_K']}, "
        f"{m['n_strides']} strides, {m['n_replans']} replans on {s['device']}",
        f"  Compute: {s['total_compute_s']:.0f}s = {s['total_compute_min']:.1f}min",
        f"",
        f"  **Trajectory summary:**",
        f"  - mean A = {s['mean_A_traj']:.3f}",
        f"  - integral A dt = {s['A_integral_observed']:.2f}",
        f"  - weighted violation rate (post-hoc) = {s['weighted_violation_rate']:.4f}",
        f"  - applied Phi range: [{s['applied_phi_min']:.2f}, {s['applied_phi_max']:.2f}]",
        f"  - final state = {s['final_state']}",
        f"",
        f"  **Gates:**",
    ]
    for k, v in s["gates"].items():
        lines.append(f"  - {'PASS' if v else 'FAIL'} `{k}`")
    lines.append("")
    return "\n".join(lines)


def summarise_stage3(m: dict) -> str:
    s = m["summary"]
    sce = m["scenario"]
    cv = m["cost_variant"]
    n_w = m["n_filter_windows"]
    cov_thresh = s["id_cov_threshold"]
    pass_thresh = s["pass_threshold_windows"]

    lines = [
        f"**Run {m['run_number']:02d} -- Stage 3 verification "
        f"({cv}, {sce['name']} scenario)**",
        f"",
        f"  Driver: `bench_smc_full_mpc_fsa_v5.py`",
        f"  Run dir: `experiments/run{m['run_number']:02d}_{m['run_tag']}/`",
        f"  Cost variant: `{cv}`"
        + (f"  (beta={m['cost_kwargs']['beta']})" if cv == 'soft' else ''),
        f"  Scenario: {sce['description']}",
        f"  T={m['T_total_days']}d, replan_K={m['replan_K']}, "
        f"{n_w} filter windows, {m['n_replans']} replans on {s['device']}",
        f"  Compute: {s['total_compute_s']:.0f}s = {s['total_compute_h']:.2f}h",
        f"",
        f"  **Trajectory summary:**",
        f"  - mean A = {s['mean_A_traj']:.3f}",
        f"  - integral A dt = {s['A_integral_observed']:.2f}",
        f"  - weighted violation rate (post-hoc) = {s['weighted_violation_rate']:.4f}",
        f"  - applied Phi range: [{s['applied_phi_min']:.2f}, {s['applied_phi_max']:.2f}]",
        f"  - final state = {s['final_state']}",
        f"",
        f"  **Filter coverage:** {s['n_pass_id_geq_threshold']}/{n_w} windows have "
        f">= {cov_thresh}/{len(m['param_names'])} params covered (threshold: "
        f">= {pass_thresh}/{n_w}).",
        f"",
        f"  **Gates (4-gate Stage 3 production sign-off):**",
    ]
    for k, v in s["gates"].items():
        lines.append(f"  - {'PASS' if v else 'FAIL'} `{k}`")
    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python tools/summarize_run.py <run-dir>")
    run_dir = Path(sys.argv[1]).resolve()
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"manifest.json not found at {manifest_path}")
    m = json.loads(manifest_path.read_text())
    stage = m.get("stage")
    if stage == 1:
        print(summarise_stage1(m))
    elif stage == 2:
        print(summarise_stage2(m))
    elif stage == 3:
        print(summarise_stage3(m))
    else:
        sys.exit(f"unknown stage in manifest: {stage}")


if __name__ == '__main__':
    main()
