"""Pytest conftest for Julia/CUDA parity tests.

These tests require the Julia toolchain (`julia` on PATH) and a working
CUDA install. They are gated on the `julia_parity` marker; default
`pytest` runs skip the entire directory.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "julia_parity: requires Julia + CUDA toolchain (skip if absent)"
    )


def _julia_executable() -> str | None:
    for candidate in (
        os.environ.get("JULIA"),
        shutil.which("julia"),
        os.path.expanduser("~/.juliaup/bin/julia"),
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def pytest_collection_modifyitems(config, items):
    """Auto-skip julia_parity tests if Julia is not available."""
    julia = _julia_executable()
    skip_marker = pytest.mark.skip(reason="julia executable not found on PATH")
    for item in items:
        if "julia_parity" in item.keywords and julia is None:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def julia_executable() -> str:
    julia = _julia_executable()
    if julia is None:
        pytest.skip("julia executable not found on PATH")
    return julia


@pytest.fixture(scope="session")
def smc2fcgpu_project_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "julia", "Smc2fcGPU"))


def run_julia(julia: str, project: str, code: str) -> subprocess.CompletedProcess:
    """Helper: run a Julia snippet inside the Smc2fcGPU project.
    Caller checks `.returncode` and parses stdout."""
    return subprocess.run(
        [julia, f"--project={project}", "-e", code],
        capture_output=True, text=True, check=False,
    )
