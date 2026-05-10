# Quick smoke test of the GPU PF kernel.

ENV["FSA_STEP_MINUTES"] = "60"
const REPO = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUPF: FSAGPUTargetBatched, gpu_log_density_batched, update_window_obs!
using .FSAHighRes.Estimation: get_init_theta, build_estimation_model
using CUDA
import SMC2FC: constrained_to_unconstrained

println("Building target: K=200, M_max=64, T=24...")
target = FSAGPUTargetBatched(K_per_chain=200, M_max=64, T_steps=24, dt=1/24)
println("Memory: total=$(round(CUDA.totalmem(CUDA.device())/1e9, digits=2))GB free=$(round(CUDA.available_memory()/1e9, digits=2))GB")

T = 24
grid_obs = Dict{Symbol,Any}(
    :hr_value        => Float32.(60.0 .+ 2.0 .* randn(T)),
    :hr_present      => Float32.(rand(Bool, T)),
    :stress_value    => Float32.(30.0 .+ 4.0 .* randn(T)),
    :stress_present  => Float32.(rand(Bool, T)),
    :log_steps_value => Float32.(5.5 .+ 0.5 .* randn(T)),
    :steps_present   => Float32.(rand(Bool, T)),
    :sleep_label     => Int32.(rand(0:1, T)),
    :sleep_present   => Float32.(ones(T)),
    :Phi             => Float32.(ones(T)),
    :C               => Float32.(cos.(2π .* (0:T-1) ./ 24)),
)
update_window_obs!(target, grid_obs; B_init=0.05, F_init=0.30, A_init=0.10)

em = build_estimation_model()
init_theta = Float64.(get_init_theta())
priors = [last(p) for p in em.param_priors]
u0 = Float64.(constrained_to_unconstrained(init_theta, priors))
U = repeat(reshape(u0, 1, :), 4, 1)
println("U size: $(size(U))")

# Warm-up + timed call
@time lls = gpu_log_density_batched(target, U)
@time lls = gpu_log_density_batched(target, U)
println("log-likelihoods: $lls")
println("Smoke test OK")
