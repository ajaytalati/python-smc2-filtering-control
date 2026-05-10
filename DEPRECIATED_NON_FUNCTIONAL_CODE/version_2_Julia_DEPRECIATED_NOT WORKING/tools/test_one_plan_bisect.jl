#!/usr/bin/env julia
# Bisection diagnostic: take the light-config open-loop test and bump
# ONE knob at a time toward the heavy bench config, keeping everything
# else light. Goal: identify which knob (n_inner, n_smc, num_mcmc,
# hmc_leap) flips the controller's optimum from recovery→overload to
# flat-low. Empirically the kernel's precision was already ruled out by
# Steps 0/1/2-alt — three different precision setups all give flat-low
# in the heavy closed-loop, so the bug is config, not arithmetic.
#
# Usage:
#   julia --project=. tools/test_one_plan_bisect.jl <knob> <value>
# Examples:
#   julia --project=. tools/test_one_plan_bisect.jl n_inner 128
#   julia --project=. tools/test_one_plan_bisect.jl n_smc 1024
#   julia --project=. tools/test_one_plan_bisect.jl num_mcmc 10
#   julia --project=. tools/test_one_plan_bisect.jl hmc_leap 16

# Allow CLI to override FSA_STEP_MINUTES BEFORE module load.
# Pass `step_min <N>` as one of the CLI knob/value pairs to override.
let
    j = 1
    while j + 1 <= length(ARGS)
        if ARGS[j] == "step_min"
            ENV["FSA_STEP_MINUTES"] = ARGS[j+1]
            break
        end
        j += 2
    end
end
get!(ENV, "FSA_STEP_MINUTES", "15")

using CUDA, Random, Statistics, Printf, LinearAlgebra
using SMC2FC: run_tempered_smc_gpu

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn
using .FSAHighRes.Dynamics: TRUTH_PARAMS

# ── Light config defaults (matches test_one_plan_fp64.jl) ──
n_smc      = 64
n_inner    = 16
num_mcmc   = 3
hmc_leap   = 8
hmc_step   = 0.2
n_anchors  = 8
sigma_prior = 1.5
target_nats = 8.0
max_levels = 30
target_ess = 0.5
max_lambda = 0.20

# Override from CLI: pairs of <knob> <value>, multiple allowed.
let
    j = 1
    while j + 1 <= length(ARGS)
        knob = ARGS[j]
        if knob == "step_min"
            j += 2
            continue   # already consumed at module-load time
        end
        val  = parse(Int, ARGS[j+1])
        if knob == "n_inner";   global n_inner   = val
        elseif knob == "n_smc"; global n_smc     = val
        elseif knob == "num_mcmc"; global num_mcmc = val
        elseif knob == "hmc_leap"; global hmc_leap = val
        elseif knob == "n_substeps"; global n_substeps = val
        elseif knob == "max_levels"; global max_levels = val
        else; error("unknown knob: $knob"); end
        @info "OVERRIDE: $knob = $val"
        j += 2
    end
end

const T_DAYS       = 14
step_min_used = parse(Int, ENV["FSA_STEP_MINUTES"])
const BINS_PER_DAY = (60 * 24) ÷ step_min_used
const N_STEPS      = T_DAYS * BINS_PER_DAY
const DT           = 1.0 / BINS_PER_DAY
const PHI_MAX      = 3.0
const PHI_DEFAULT  = 1.0
M_max = n_smc * (1 + 2 * n_anchors)
n_substeps = max(1, BINS_PER_DAY ÷ 24)   # mirrors Python: 4 at h=15min, 1 at h=60min

# init_state can be overridden via env var:  INIT_STATE="0.075,0.092,0.090"
init_state_arg = if haskey(ENV, "INIT_STATE")
    parts = split(ENV["INIT_STATE"], ",")
    [parse(Float64, p) for p in parts]
else
    [0.05, 0.30, 0.10]
end
@info "Init state: $init_state_arg"

target = FSAControlGPUTarget(
    n_inner = n_inner, M_max = M_max, n_steps = N_STEPS,
    n_anchors = n_anchors, n_substeps = n_substeps, dt = DT,
    F_max = 0.40, Phi_max = PHI_MAX, Phi_default = PHI_DEFAULT,
    lam_F = 1.0,
    sigma_prior = sigma_prior,
    params = TRUTH_PARAMS,
    init_state = init_state_arg,
    noise_seed = parse(Int, get(ENV, "NOISE_SEED", "42")),
)

@info "Config: step_min=$(step_min_used), BINS_PER_DAY=$(BINS_PER_DAY), n_substeps=$(n_substeps), n_smc=$n_smc, n_inner=$n_inner, num_mcmc=$num_mcmc, hmc_leap=$hmc_leap, hmc_step=$hmc_step, max_levels=$max_levels"

t0 = time()
log_density_fn = make_log_density_fn(target)
rng = MersenneTwister(parse(Int, get(ENV, "HMC_SEED", "123")))
posterior, n_temp, β_max = run_tempered_smc_gpu(
    log_density_fn, M_max, n_smc, n_anchors,
    0.0, sigma_prior, rng;
    target_nats = target_nats,
    target_ess_frac = target_ess,
    max_lambda_inc = max_lambda,
    max_temp_levels = max_levels,
    num_mcmc_steps = num_mcmc,
    hmc_step_size = hmc_step,
    hmc_num_leapfrog = hmc_leap,
    chees_L_candidates = [16, 32, 64, 128, 256],
    h_fd = 1e-4, calib_n = 64,
    verbose = false,
)
t = time() - t0

theta_post_mean = vec(mean(posterior; dims=1))
c_Phi = log((PHI_DEFAULT / PHI_MAX) / (1.0 - PHI_DEFAULT / PHI_MAX))
T_total = N_STEPS * DT
t_grid = collect(0:N_STEPS-1) .* DT
anchors = collect(range(0.0, T_total; length=n_anchors))
σ_rbf = T_total / n_anchors
phi_per_bin = zeros(Float64, N_STEPS)
for k in 1:N_STEPS
    raw = c_Phi
    for j in 1:n_anchors
        raw += theta_post_mean[j] * exp(-0.5 * ((t_grid[k] - anchors[j]) / σ_rbf)^2)
    end
    phi_per_bin[k] = PHI_MAX / (1.0 + exp(-raw))
end

println("="^72)
@printf("Result: %d levels, β_max=%.2f, %.1fs wall\n", n_temp, β_max, t)
@printf("Posterior mean θ: %s\n",
        "[" * join([@sprintf("%+.3f", v) for v in theta_post_mean], ", ") * "]")
@printf("Φ at days 0,2,4,6,8,10,12,14: ")
for d in [0,2,4,6,8,10,12,14]
    k = clamp(d * BINS_PER_DAY + 1, 1, N_STEPS)
    @printf("%.2f ", phi_per_bin[k])
end
println()
@printf("Φ summary: min=%.3f mean=%.3f max=%.3f\n",
        minimum(phi_per_bin), mean(phi_per_bin), maximum(phi_per_bin))
println(maximum(phi_per_bin) > 0.6 ? "  → recovery→overload found" : "  → FLAT-LOW (broken)")
