#!/usr/bin/env julia
# Run the plant under three known per-bin Φ schedules and report
# (B, F, A) trajectory + mean A. Compares to analytic Banister equilibria.
# If plant numbers don't match analytics → plant SDE has a bug.
#
# Schedules tested:
#   1. Φ ≡ 1.0 (baseline)
#   2. Φ ≡ 0.20 (rest-leaning, what controller picks)
#   3. recovery→overload: Φ=0.20 days 1-5 then ramp to 1.20 by day 13

ENV["FSA_STEP_MINUTES"] = "15"

using Statistics
using Random
using Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: StepwisePlant, advance_subdaily!
using .FSAHighRes.Dynamics: TRUTH_PARAMS

const BINS_PER_DAY = 96
const N_DAYS       = 14
const N_BINS       = N_DAYS * BINS_PER_DAY
const DT           = 1.0 / BINS_PER_DAY


function make_phi_per_bin(daily_phi::Vector{Float64})
    n_bins = length(daily_phi) * BINS_PER_DAY
    out = zeros(Float64, n_bins)
    for d in 1:length(daily_phi), k in 1:BINS_PER_DAY
        out[(d-1)*BINS_PER_DAY + k] = daily_phi[d]
    end
    return out
end


function make_recovery_overload()
    # Recovery 0.20 for days 1-5, ramp 0.20→1.20 over days 6-13.
    daily = zeros(Float64, N_DAYS)
    for d in 1:5
        daily[d] = 0.20
    end
    for d in 6:N_DAYS
        # Linear ramp 0.20 at d=6, 1.20 at d=13
        daily[d] = 0.20 + (1.20 - 0.20) * (d - 6) / 7.0
    end
    return daily
end


function run_plant_under_phi(phi_per_bin::Vector{Float64}, label::String, seed::Int)
    plant = StepwisePlant(seed_offset=seed, dt=DT)
    n_bins = length(phi_per_bin)
    advance_subdaily!(plant, phi_per_bin)
    traj = vcat(plant.history[:trajectory]...)
    @assert size(traj, 1) == n_bins
    return traj
end


function summarize(traj::Matrix{Float64}, label::String, phi::Vector{Float64})
    B, F, A = traj[:, 1], traj[:, 2], traj[:, 3]
    println("="^72)
    println(label)
    println("-"^72)
    @printf("  daily Φ: min=%.3f mean=%.3f max=%.3f\n", minimum(phi), mean(phi), maximum(phi))
    @printf("  B:     start=%.3f  day7=%.3f  end=%.3f  mean=%.4f\n",
            B[1], B[7*BINS_PER_DAY], B[end], mean(B))
    @printf("  F:     start=%.3f  day7=%.3f  end=%.3f  mean=%.4f\n",
            F[1], F[7*BINS_PER_DAY], F[end], mean(F))
    @printf("  A:     start=%.3f  day7=%.3f  end=%.3f  mean=%.4f\n",
            A[1], A[7*BINS_PER_DAY], A[end], mean(A))
end


# Run multiple seeds + take MC mean to remove noise effects.
function mc_mean_trajectory(phi_per_bin::Vector{Float64}, n_seeds::Int = 32)
    traj_acc = zeros(Float64, length(phi_per_bin), 3)
    for s in 1:n_seeds
        plant = StepwisePlant(seed_offset = 42 + s * 37, dt = DT)
        advance_subdaily!(plant, phi_per_bin)
        traj_acc .+= vcat(plant.history[:trajectory]...)
    end
    return traj_acc ./ n_seeds
end


println("\nMonte-Carlo plant traces (n_seeds=32) under three schedules\n")

phi_baseline = make_phi_per_bin(fill(1.0, N_DAYS))
phi_rest     = make_phi_per_bin(fill(0.20, N_DAYS))
phi_recovery = make_phi_per_bin(make_recovery_overload())

for (lbl, phi) in [
    ("Φ ≡ 1.0 baseline",        phi_baseline),
    ("Φ ≡ 0.20 rest-leaning",   phi_rest),
    ("recovery → overload",     phi_recovery),
]
    traj = mc_mean_trajectory(phi, 32)
    summarize(traj, lbl, phi)
end

println()
println("Analytic Banister equilibria for sanity:")
println("  Φ=1.00:  B_eq = κ_B·Φ·τ_B = 0.0125·1.00·42 = 0.525   (T=14d reaches ~0.18)")
println("  Φ=0.20:  B_eq = κ_B·Φ·τ_B = 0.0125·0.20·42 = 0.105   (T=14d reaches ~0.07)")
println("  Φ=1.00:  F_eq = κ_F·Φ·τ_F = 0.030·1.00·6.36/1.10 = 0.173")
println("  Φ=0.20:  F_eq = κ_F·Φ·τ_F = 0.030·0.20·6.36/1.10 = 0.035")
println()
println("Spec §10.2: baseline mean A = 0.0823, MPC mean A = 0.0964 (+17%)")
println("User plot: baseline mean A = 0.081,  MPC mean A = 0.122  (+50%, recovery→overload)")
