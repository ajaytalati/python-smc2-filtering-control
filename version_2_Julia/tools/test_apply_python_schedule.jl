#!/usr/bin/env julia
# Apply Python's exact daily-Φ schedule to Julia's plant. If Julia's
# plant under Python's schedule produces mean A close to Python's 0.122,
# the gap is in Julia's controller (planning different schedule). If it
# stays near 0.087, the plants behave differently under the same input.

ENV["FSA_STEP_MINUTES"] = "60"

using NPZ, Random, Statistics, Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: StepwisePlant, advance!
using .FSAHighRes.Simulation: INIT_STATE
using .FSAHighRes.Dynamics: TRUTH_PARAMS

# Load Python's reference daily_phi_per_stride
py = NPZ.npzread("/home/ajay/Repos/python-smc2-filtering-control/version_2/outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/data.npz")
py_daily_phi = py["daily_phi_per_stride"]   # 27 strides × 1 daily Phi each
println("Python daily Φ per stride: ", py_daily_phi)

const BINS_PER_DAY = 24
const DT_DAYS = 1.0 / BINS_PER_DAY
const STRIDE_BINS = 12
const WINDOW_BINS = 24
const T_DAYS = 14
const n_strides = 27

# Julia plant
plant = StepwisePlant(seed_offset=42, dt=DT_DAYS)
println("Julia plant init state: B=$(plant.state[1]) F=$(plant.state[2]) A=$(plant.state[3])")

# Apply Python's schedule stride by stride
for s in 1:n_strides
    advance_bins = (s == 1) ? WINDOW_BINS : STRIDE_BINS
    daily_phi = Float64(py_daily_phi[s])  # Python is 0-indexed: py[s-1] is for Julia's s=1... wait
    advance!(plant, advance_bins, [daily_phi])
end

# Compute mean A over the trajectory
import .FSAHighRes.Plant
traj = vcat(plant.history[:trajectory]...)
A_traj = traj[:, 3]
B_traj = traj[:, 1]
F_traj = traj[:, 2]
println()
@printf("Julia plant under PYTHON schedule:\n")
@printf("  Final state: B=%.4f F=%.4f A=%.4f\n", traj[end,1], traj[end,2], traj[end,3])
@printf("  Mean B = %.4f, Mean F = %.4f, Mean A = %.4f\n", mean(B_traj), mean(F_traj), mean(A_traj))
@printf("  A peak = %.4f\n", maximum(A_traj))
println()
@printf("Python's reported: Mean A = 0.122, A peak ≈ 0.166, baseline = 0.081\n")
