"""Sanity checks for the FSA_STEP_MINUTES grid coarsening.

Each test runs in its own subprocess so the FSA_STEP_MINUTES env var
governs simulation/_phi_burst module-import constants cleanly.

Invoke directly:
    cd version_2 && PYTHONPATH=.:.. python tests/test_h1h_grid.py

Exits 0 on all-pass.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1].parent  # python-smc2-filtering-control
V2_DIR    = REPO_ROOT / "version_2"


def _run(snippet: str, step_minutes: int) -> str:
    env = os.environ.copy()
    env['FSA_STEP_MINUTES'] = str(step_minutes)
    env['JAX_PLATFORMS'] = env.get('JAX_PLATFORMS', 'cpu')  # unit tests on CPU
    env['PYTHONPATH'] = f"{V2_DIR}:{REPO_ROOT}"
    res = subprocess.run([sys.executable, '-c', snippet],
                          cwd=str(V2_DIR), env=env,
                          capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        raise RuntimeError(
            f"subprocess (step_minutes={step_minutes}) failed:\n"
            f"  stdout: {res.stdout}\n  stderr: {res.stderr}"
        )
    return res.stdout.strip()


# -------------------------------------------------------------------------
# Test 1: BINS_PER_DAY and DT_BIN_HOURS round-trip.
# -------------------------------------------------------------------------

def test_grid_constants():
    for sm, bpd, dbh in [(15, 96, 0.25), (60, 24, 1.0), (30, 48, 0.5)]:
        out = _run(
            "from models.fsa_high_res.simulation import BINS_PER_DAY, DT_BIN_HOURS; "
            "print(f'{BINS_PER_DAY},{DT_BIN_HOURS}')",
            step_minutes=sm,
        )
        got_bpd, got_dbh = out.split(',')
        assert int(got_bpd) == bpd, f"step={sm}: BINS_PER_DAY={got_bpd} != {bpd}"
        assert abs(float(got_dbh) - dbh) < 1e-9, \
            f"step={sm}: DT_BIN_HOURS={got_dbh} != {dbh}"
    print("  [pass] grid constants")


# -------------------------------------------------------------------------
# Test 2: circadian phase alignment — C(t) at hour-aligned bins under
# both grids returns the same value.
# -------------------------------------------------------------------------

def test_circadian_phase_alignment():
    snippet = """
import numpy as np
from models.fsa_high_res.simulation import BINS_PER_DAY, DT_BIN_HOURS, circadian
# Sample C(t) at every hour mark for 1 day
t_hours = np.arange(0, 24, 1.0)         # 24 hour marks
t_days = t_hours / 24.0
vals = np.array([circadian(t) for t in t_days])
print(','.join(f'{v:.10f}' for v in vals))
"""
    vals_15 = [float(v) for v in _run(snippet, 15).split(',')]
    vals_60 = [float(v) for v in _run(snippet, 60).split(',')]
    diffs = [abs(a - b) for a, b in zip(vals_15, vals_60)]
    max_diff = max(diffs)
    assert max_diff < 1e-9, f"max C(t) diff = {max_diff}"
    print("  [pass] circadian phase alignment")


# -------------------------------------------------------------------------
# Test 3: sleep gating — total daily sleep duration the same fraction
# of day under both grids (8h sleep window: 23:00→07:00 = 1/3 of day).
# -------------------------------------------------------------------------

def test_sleep_gating_consistency():
    snippet = """
import numpy as np
from models.fsa_high_res.simulation import sleep_mask_from_hours, BINS_PER_DAY
m = sleep_mask_from_hours(n_days=1, sleep_hour_lo=23.0, sleep_hour_hi=7.0)
sleep_frac = float(m.mean())
print(f'{BINS_PER_DAY},{sleep_frac:.6f}')
"""
    out15 = _run(snippet, 15).split(',')
    out60 = _run(snippet, 60).split(',')
    bpd15, frac15 = int(out15[0]), float(out15[1])
    bpd60, frac60 = int(out60[0]), float(out60[1])
    assert bpd15 == 96 and bpd60 == 24
    # 8h sleep / 24h = 0.333... — should hold to 1/BINS_PER_DAY rounding
    expected = 8.0 / 24.0
    assert abs(frac15 - expected) < 1.0 / 96 + 1e-9, \
        f"15-min: sleep_frac={frac15} vs expected {expected}"
    assert abs(frac60 - expected) < 1.0 / 24 + 1e-9, \
        f"60-min: sleep_frac={frac60} vs expected {expected}"
    print(f"  [pass] sleep gating consistency (15min={frac15:.4f}, "
          f"60min={frac60:.4f})")


# -------------------------------------------------------------------------
# Test 4: plant — 1-day plant simulation at constant Φ=1.0 produces
# day-average B/F/A that agree across grids to ~5%. Same seed for both.
# -------------------------------------------------------------------------

def test_plant_day_means():
    snippet = """
import numpy as np
from models.fsa_high_res._plant import StepwisePlant
from models.fsa_high_res.simulation import BINS_PER_DAY
plant = StepwisePlant(seed_offset=42)
out = plant.advance(BINS_PER_DAY, np.array([1.0]))
traj = out['trajectory']  # (BINS_PER_DAY, 3)
mean = traj.mean(axis=0)
print(f'{mean[0]:.6f},{mean[1]:.6f},{mean[2]:.6f}')
"""
    means15 = [float(v) for v in _run(snippet, 15).split(',')]
    means60 = [float(v) for v in _run(snippet, 60).split(',')]
    rel_diffs = [abs(a - b) / max(abs(a), 1e-6)
                 for a, b in zip(means15, means60)]
    max_rel = max(rel_diffs)
    # Stochastic — different seeds per bin diverge — but day-averages of
    # the same SDE under EM with different h should agree to a few %.
    # 10% slack to absorb the noise.
    assert max_rel < 0.10, (
        f"plant day-means diverge: 15min={means15}, 60min={means60}, "
        f"max_rel_diff={max_rel:.3f}"
    )
    print(f"  [pass] plant day-means (max rel diff {max_rel*100:.2f}%)")
    print(f"      h=15min: B={means15[0]:.4f} F={means15[1]:.4f} A={means15[2]:.4f}")
    print(f"      h=1h:    B={means60[0]:.4f} F={means60[1]:.4f} A={means60[2]:.4f}")


# -------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Stage I1 — h=1h grid sanity tests")
    print("=" * 64)
    test_grid_constants()
    test_circadian_phase_alignment()
    test_sleep_gating_consistency()
    test_plant_day_means()
    print("-" * 64)
    print("  All tests passed.")


if __name__ == '__main__':
    main()
