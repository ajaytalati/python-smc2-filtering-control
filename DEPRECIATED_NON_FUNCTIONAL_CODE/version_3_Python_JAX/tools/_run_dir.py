"""Shared atomic run-directory allocator for the FSA-v5 bench scripts.

The previous in-bench helpers were a TOCTOU race: ``_next_run_number``
read max-existing+1 then ``mkdir(exist_ok=True)``. Two concurrent
processes both compute the same N, both pass the existence check, and
the second silently writes its artifacts on top of the first's
``manifest.json`` / ``trajectory.npz``.

This module replaces both halves with a single atomic-allocation loop
using ``mkdir(exist_ok=False)``; on collision we re-scan and retry. A
small monotonic counter caps the loop so we don't spin forever on
filesystem oddities.
"""

from __future__ import annotations

from pathlib import Path

_MAX_ALLOC_RETRIES = 256


def _scan_max_run_number(experiments_dir: Path) -> int:
    """Return the largest existing run number under ``experiments_dir``,
    or 0 if there are none. ``runNN_<tag>`` is the expected layout."""
    if not experiments_dir.exists():
        return 0
    nums = []
    for p in experiments_dir.iterdir():
        if p.is_dir() and p.name.startswith('run'):
            stem = p.name[3:].split('_', 1)[0]
            try:
                nums.append(int(stem))
            except ValueError:
                pass
    return max(nums, default=0)


def allocate_run_dir(repo_root: Path, run_tag: str) -> tuple[Path, int]:
    """Atomically reserve ``outputs/fsa_v5/experiments/runNN_<run_tag>/``.

    Returns ``(out_dir, NN)``. Raises ``RuntimeError`` after
    ``_MAX_ALLOC_RETRIES`` consecutive failures (e.g. filesystem refuses
    the create even though the dir doesn't exist — investigate
    permissions or fs-full).
    """
    exp_dir = repo_root / "outputs" / "fsa_v5" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    last_err: Exception | None = None
    for attempt in range(_MAX_ALLOC_RETRIES):
        n = _scan_max_run_number(exp_dir) + 1 + attempt
        out_dir = exp_dir / f"run{n:02d}_{run_tag}"
        try:
            out_dir.mkdir(exist_ok=False)
            return out_dir, n
        except FileExistsError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Could not allocate a run dir after {_MAX_ALLOC_RETRIES} "
        f"attempts under {exp_dir}. Last error: {last_err}"
    )
