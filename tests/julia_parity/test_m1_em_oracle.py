"""M1 — JAX↔Julia parity test for the FSA-v2 SDE Euler-Maruyama oracle.

We run `smc2fc.simulator.sde_solver_diffrax.solve_sde_jax` on a fixed
seed, dump the (noise, params, init, Phi, t_grid, golden_trajectory)
to a single .npz file, then call a Julia script that loads the .npz,
runs `Smc2fcGPU.em_oracle_single`, and writes its output back. We then
compare both trajectories at fp64.

Tolerance gate (M1 in plan):
  - means RMSE ≤ 1e-5 per state
  - stds  RMSE ≤ 1e-4 per state
  - additionally: per-element max abs diff ≤ 1e-9 — because we feed the
    SAME noise tensor to both solvers, the trajectories should match to
    fp64 round-off. Anything larger means the math diverges.
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.julia_parity


def _golden_path(name: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "golden", name)


def _make_jax_golden(out_path: str, seed: int = 42, n_substeps: int = 10):
    """Run JAX/diffrax `solve_sde_jax` and dump everything needed for parity."""
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    import sys
    repo = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo not in sys.path:
        sys.path.insert(0, repo)
    v2 = os.path.join(repo, "version_2")
    if v2 not in sys.path:
        sys.path.insert(0, v2)

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from models.fsa_high_res.simulation import (
        HIGH_RES_FSA_V2_MODEL, DEFAULT_PARAMS, DEFAULT_INIT,
        BINS_PER_DAY, generate_phi_sub_daily,
    )
    from smc2fc.simulator.sde_solver_diffrax import solve_sde_jax
    from smc2fc.simulator.sde_model import DIFFUSION_DIAGONAL_STATE
    assert HIGH_RES_FSA_V2_MODEL.diffusion_type == DIFFUSION_DIAGONAL_STATE

    n_grid = BINS_PER_DAY  # 96 for 1 day
    dt = 1.0 / BINS_PER_DAY
    t_grid = np.arange(n_grid + 1, dtype=np.float64) * dt
    Phi_arr = generate_phi_sub_daily(np.array([1.0]), seed=42, noise_frac=0.0)
    exogenous = {"Phi_arr": Phi_arr}

    init_state = dict(DEFAULT_INIT)
    params = dict(DEFAULT_PARAMS)

    # Reproduce the noise tensor that solve_sde_jax computes internally,
    # so we can hand the SAME draws to Julia.
    n_states = HIGH_RES_FSA_V2_MODEL.n_states
    key = jax.random.PRNGKey(seed)
    total_substeps = n_grid * n_substeps
    all_noise = jax.random.normal(key, (total_substeps, n_states), dtype=jnp.float64)
    # Note: this is the RAW N(0,1) draw, BEFORE any sigma * sqrt_dt scaling
    # (matches how solve_sde_jax pre-generates `all_noise` then optionally
    # rescales by `sigma * sqrt_dt` in the constant-diffusion path; the
    # state-dependent path uses raw N(0,1) and does the scaling inside scan).
    noise_unit = np.array(all_noise, dtype=np.float64).reshape(n_grid, n_substeps, n_states)

    traj = solve_sde_jax(
        HIGH_RES_FSA_V2_MODEL, params, init_state, t_grid,
        exogenous=exogenous, seed=seed, n_substeps=n_substeps,
    )

    np.savez_compressed(
        out_path,
        traj_jax=traj.astype(np.float64),
        t_grid=t_grid,
        Phi_arr=Phi_arr.astype(np.float64),
        # Julia side will reshape (n_substeps, n_grid, 3) — column-major view
        # of the same memory layout JAX uses.
        noise_unit_NgNsNk=noise_unit,
        y0=np.array([init_state["B_0"], init_state["F_0"], init_state["A_0"]],
                    dtype=np.float64),
        n_substeps=np.int64(n_substeps),
    )
    # Params side-car: NPZ.jl cannot parse numpy unicode arrays, so we ship
    # params as a sibling JSON file.
    params_path = out_path.replace(".npz", "_params.json")
    with open(params_path, "w") as f:
        json.dump({k: float(v) for k, v in params.items()}, f)


def test_m1_em_oracle_parity(tmp_path, julia_executable, smc2fcgpu_project_dir):
    golden_npz = _golden_path("em_seed42_K1_T96.npz")
    os.makedirs(os.path.dirname(golden_npz), exist_ok=True)

    # Always regenerate from the live JAX code path — it's cheap.
    _make_jax_golden(golden_npz, seed=42, n_substeps=10)

    # Drive Julia: load .npz, run em_oracle_single, save Julia trajectory.
    julia_traj_path = tmp_path / "traj_julia.npy"
    julia_code = textwrap.dedent(f"""
        using Smc2fcGPU
        using StaticArrays
        using NPZ
        using JSON

        data = NPZ.npzread(raw"{golden_npz}")
        y0v = data["y0"]
        y0 = SVector{{3,Float64}}(y0v[1], y0v[2], y0v[3])
        t_grid = vec(data["t_grid"])
        Phi_arr = vec(data["Phi_arr"])
        # JAX reshape: (n_grid, n_substeps, 3); we transpose to (n_substeps, n_grid, 3)
        noise_jax = data["noise_unit_NgNsNk"]
        @assert ndims(noise_jax) == 3
        n_grid_jax, n_sub_jax, n_st_jax = size(noise_jax)
        # NPZ loads numpy arrays in their original shape (Julia row-major view of
        # numpy column-major data); pure permutation here.
        noise_jl = permutedims(noise_jax, (2, 1, 3))   # (n_substeps, n_grid, 3)
        @assert size(noise_jl) == (n_sub_jax, n_grid_jax, n_st_jax)
        params_path = replace(raw"{golden_npz}", ".npz" => "_params.json")
        params_dict = JSON.parsefile(params_path)
        p = Smc2fcGPU.FsaParams{{Float64}}(params_dict)
        n_sub = Int(data["n_substeps"][1])
        traj_jl = Smc2fcGPU.em_oracle_single(y0, p, t_grid, Phi_arr, noise_jl;
                                              n_substeps=n_sub)
        NPZ.npzwrite(raw"{julia_traj_path}", traj_jl)
        println("OK")
    """)

    # Make sure NPZ + JSON are available in the project; add them on first run.
    subprocess.run(
        [julia_executable, f"--project={smc2fcgpu_project_dir}", "-e",
         'using Pkg; for p in ("NPZ", "JSON"); haskey(Pkg.project().dependencies, p) || Pkg.add(p); end'],
        check=True, capture_output=True, text=True,
    )

    proc = subprocess.run(
        [julia_executable, f"--project={smc2fcgpu_project_dir}", "-e", julia_code],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"Julia M1 oracle run failed: rc={proc.returncode}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    assert "OK" in proc.stdout, proc.stdout

    traj_jl = np.load(str(julia_traj_path))
    data = np.load(golden_npz, allow_pickle=False)
    traj_jax = data["traj_jax"]

    assert traj_jl.shape == traj_jax.shape, (
        f"shape mismatch: julia={traj_jl.shape}, jax={traj_jax.shape}")

    abs_diff = np.abs(traj_jl - traj_jax)
    max_abs = abs_diff.max(axis=0)
    rmse = np.sqrt((abs_diff ** 2).mean(axis=0))
    means_jax = traj_jax.mean(axis=0)
    means_jl = traj_jl.mean(axis=0)
    stds_jax = traj_jax.std(axis=0)
    stds_jl = traj_jl.std(axis=0)

    print(f"\n[M1 parity] state-wise max-abs diff: {max_abs}")
    print(f"[M1 parity] state-wise RMSE:          {rmse}")
    print(f"[M1 parity] mean(JAX): {means_jax}, mean(JL): {means_jl}")
    print(f"[M1 parity] std(JAX):  {stds_jax}, std(JL):  {stds_jl}")

    # M1 plan tolerances on aggregate stats
    assert np.all(np.abs(means_jl - means_jax) < 1e-5), (
        f"mean diff exceeds 1e-5: {np.abs(means_jl - means_jax)}")
    assert np.all(np.abs(stds_jl - stds_jax) < 1e-4), (
        f"std diff exceeds 1e-4: {np.abs(stds_jl - stds_jax)}")
    # Same-noise byte-parity: per-element abs diff is fp64 round-off only
    assert np.all(max_abs < 1e-9), (
        f"per-element max-abs diff exceeds 1e-9: {max_abs}; algorithms diverge")
