"""Standalone T=28 acceptance-gate checker.

Exists so the sweep script can read the gate verdict via a Python
exit code instead of an inline bash block. Inline bash gate checks are
unsafe to hot-edit while a sweep is running — bash streams the script
from a file position, and editing the file in place shifts byte
offsets. This script can be safely modified at any time; the running
sweep only invokes it after T=28 has finished writing its manifest.

Usage:
    python tools/check_t28_gate.py <path-to-T28-manifest.json>

Exit codes:
    0 — all gates pass
    2 — at least one gate fails
"""
from __future__ import annotations

import json
import sys


def main(manifest_path: str) -> int:
    with open(manifest_path) as f:
        manifest = json.load(f)

    s = manifest["summary"]
    n_strides = manifest["n_strides"]   # top-level, NOT inside summary

    ratio = s["mean_A_mpc"] / max(s["mean_A_baseline"], 1e-9)
    fviol = s["F_violation_frac_mpc"]
    idcov = s["n_windows_pass_id_cov_5_of_6"]
    threshold = (n_strides * 24) // 27

    fails = []
    if ratio < 0.95:
        fails.append(f"ratio {ratio:.3f} < 0.95")
    if fviol > 0.05:
        fails.append(f"F-viol {fviol:.2%} > 5%")
    if idcov < threshold:
        fails.append(f"id-cov {idcov}/{n_strides} < {threshold}")

    print(
        f"  T=28 GATE: ratio={ratio:.3f}  F-viol={fviol:.2%}  "
        f"id-cov={idcov}/{n_strides}  (threshold {threshold})"
    )
    if fails:
        print(f"  T=28 GATE FAIL: {' | '.join(fails)}", file=sys.stderr)
        return 2
    print("  T=28 GATE PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python tools/check_t28_gate.py <manifest.json>",
              file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
