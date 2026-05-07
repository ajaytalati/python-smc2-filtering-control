"""Tests for the shared atomic run-directory allocator
(``version_3/tools/_run_dir.py``).

Targets the TOCTOU race the previous in-bench helper had: two
concurrent processes both compute N from max-existing+1, both call
``mkdir(exist_ok=True)`` and the second silently writes its artifacts
into the first's directory. The new allocator uses
``mkdir(exist_ok=False)`` and re-scans+retries on collision.
"""

import multiprocessing as mp
import tempfile
from pathlib import Path

from version_3.tools._run_dir import allocate_run_dir, _scan_max_run_number


def test_allocate_returns_runNN_pair(tmp_path: Path):
    out_dir, n = allocate_run_dir(tmp_path, "smoke")
    assert n == 1
    assert out_dir.name == "run01_smoke"
    assert out_dir.is_dir()


def test_consecutive_allocations_increment(tmp_path: Path):
    d1, n1 = allocate_run_dir(tmp_path, "alpha")
    d2, n2 = allocate_run_dir(tmp_path, "beta")
    d3, n3 = allocate_run_dir(tmp_path, "gamma")
    assert (n1, n2, n3) == (1, 2, 3)
    assert d1 != d2 != d3
    assert all(p.is_dir() for p in (d1, d2, d3))


def _alloc_worker(tmpdir_str: str, tag: str, q: mp.Queue) -> None:
    out_dir, n = allocate_run_dir(Path(tmpdir_str), tag)
    q.put((n, str(out_dir)))


def test_concurrent_allocations_unique_paths_distinct_tags():
    """Spawn 8 processes with distinct tags that race for run dirs.
    Path uniqueness is the safety guarantee (clobber-prevention).
    Run-number uniqueness is NOT guaranteed when tags differ — two
    processes can both pick N=1 if they make different paths
    (run01_w0 vs run01_w1) — that's intentional and harmless."""
    n_workers = 8
    with tempfile.TemporaryDirectory() as tmp:
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_alloc_worker, args=(tmp, f"w{i}", q))
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited with {p.exitcode}"
        results = [q.get() for _ in range(n_workers)]
    paths = [p for _, p in results]
    assert len(set(paths)) == n_workers, (
        f"Path collision under concurrent allocation: {paths}"
    )


def test_concurrent_allocations_same_tag_get_distinct_paths():
    """The destructive original-race scenario: two processes launched
    with the SAME tag. Old helper let both ``mkdir(exist_ok=True)``
    succeed and the second silently clobbered the first's artifacts.
    New helper must raise FileExistsError internally, retry with a
    bumped N, and produce two distinct paths."""
    n_workers = 8
    with tempfile.TemporaryDirectory() as tmp:
        ctx = mp.get_context("spawn")
        q: mp.Queue = ctx.Queue()
        procs = [
            ctx.Process(target=_alloc_worker, args=(tmp, "shared_tag", q))
            for _ in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker exited with {p.exitcode}"
        results = [q.get() for _ in range(n_workers)]
    nums = [n for n, _ in results]
    paths = [p for _, p in results]
    # Same-tag → numbers AND paths must all be unique.
    assert len(set(nums)) == n_workers, (
        f"Same-tag collision: numbers {nums}"
    )
    assert len(set(paths)) == n_workers, (
        f"Same-tag collision: paths {paths}"
    )


def test_max_scan_handles_non_run_dirs(tmp_path: Path):
    (tmp_path / "outputs" / "fsa_v5" / "experiments").mkdir(parents=True)
    exp = tmp_path / "outputs" / "fsa_v5" / "experiments"
    (exp / "run01_a").mkdir()
    (exp / "run42_b").mkdir()
    (exp / "not_a_run").mkdir()
    (exp / "run_garbage_no_number").mkdir()
    assert _scan_max_run_number(exp) == 42
