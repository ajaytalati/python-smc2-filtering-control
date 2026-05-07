# B3 closed-loop with PARALLEL-CHAINS GPU HMC + R=20 SEGMENTED PF.
#
# Per-step resampling + Liu-West shrinkage every R=20 obs steps inside
# the GPU PF — option C from the user checkpoint. ~7× slower per
# gradient call than the unresampled fused kernel but matches Python's
# gk_dpf_v3_lite numerics much more closely (per-step weight collapse
# is bounded, partially-identified params get a data anchor through
# the resampled cloud).
#
# Run from version_1_Julia/:
#   julia --threads auto --project=. tools/bench_b3_gpu_segmented.jl

using Random, Statistics, Printf
using LogExpFunctions: logsumexp
using CUDA
using Plots
using Plots.PlotMeasures: mm
gr()

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "models", "bistable_controlled", "gpu_pf.jl"))
using .BistableGPUPF: BistableGPUTargetSegmented,
                       gpu_log_density_batched_segmented,
                       gpu_grads_parallel_chains_segmented
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))
push!(LOAD_PATH, joinpath(ROOT, "utils"))
using SMC2FC: SMCConfig, ControlSpec, run_tempered_smc_loop,
              RBFBasis, design_matrix, schedule_from_theta, SigmoidOutput
using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A, simulate_em
include(joinpath(ROOT, "utils", "diagnostics.jl"))
using .Diagnostics: STEELBLUE

println("="^70)
println("B3 — segmented GPU PF (R=20 per-step resample + Liu-West)")
println("="^70)

const T_steps_full = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))
const n1           = Int(round(EXOGENOUS_A.T_intervention / EXOGENOUS_A.dt))
const n2           = T_steps_full - n1
const U_CRIT       = 2.0 * PARAM_SET_A.alpha * PARAM_SET_A.a^3 / (3.0 * sqrt(3.0))
const N_SMC        = 64
const NUM_MCMC     = 3
const HMC_STEP     = 0.002   # with prior-scaled mass matrix, ε is the position-step
                              # fraction of σ. 0.002 with L=16 leapfrog gives ~3% of σ
                              # per move, ChEES picks L → 32% of σ traversed per HMC.
const HMC_LEAPFROG = 6
const MAX_LAMBDA   = 0.10
const TARGET_ESS   = 0.5
const K_PER_CHAIN  = 10_000
const R_SEGMENT    = 1   # match Python's gk_dpf_v3_lite per-step resample.
                          # Was 20 (option-C compromise) but R>1 lets weights
                          # collapse between resamples, flattening the LL
                          # surface for partially-identified params (σ_u, σ_x).
const M_TOTAL      = N_SMC * (1 + 2 * 8)

data = simulate_em(seed = 7)
data_p1 = (t_grid = data.t_grid[1:n1], trajectory = data.trajectory[1:n1, :], obs = data.obs[1:n1])
x_24 = data.trajectory[n1, 1]; u_24 = data.trajectory[n1, 2]

println("Building segmented GPU target: K_per_chain=$K_PER_CHAIN, M=$M_TOTAL, R=$R_SEGMENT")
target = BistableGPUTargetSegmented(K_per_chain = K_PER_CHAIN, M_max = M_TOTAL,
    T_steps = n1, R = R_SEGMENT,
    dt = EXOGENOUS_A.dt, T_intervention = 24.0, u_on = 0.0,
    x_init = INIT_STATE_A.x_0, u_init = INIT_STATE_A.u_0,
    obs_seq_cpu = Float64.(data_p1.obs), noise_seed = 0)

# Priors mirror Python `version_1/models/bistable_controlled/estimation.py`
# PARAM_PRIOR_CONFIG + INIT_STATE_PRIOR_CONFIG. The previous Julia setup
# used N(0, 0.5) for every parameter, which puts the σ_u prior median at
# 1.0 (truth = 0.05) — the prior penalty at log(σ_u) = -3 dominated the
# weak likelihood signal, biasing the posterior to the prior mode.
#                       α      a      σ_x    γ          σ_u    σ_obs   x_0   u_0
const PRIOR_MEANS  = [0.0,   0.0,   -2.0,   log(2.0),  -3.0,  -1.5,  -1.0,  0.0]
const PRIOR_SIGMAS = [0.5,   0.3,    0.5,   0.3,        0.5,   0.3,   0.5,  0.3]

function logprior_grad(U)
    return -(U .- PRIOR_MEANS') ./ (PRIOR_SIGMAS' .^ 2)
end
function logprior_val(U)
    return -0.5 .* vec(sum(((U .- PRIOR_MEANS') ./ PRIOR_SIGMAS') .^ 2; dims = 2))
end

# FD step for the gradient. Larger h → averages over more resample-flip
# events, lower noise, but larger Taylor bias. h=1e-3 was inherited from
# the unresampled-kernel bench; for segmented PF we need much larger.
const FD_H = 0.05

# Parallel HMC one move using the segmented gradient.
# Uses a diagonal mass matrix M = diag(1/PRIOR_SIGMAS²) → inv_mass =
# PRIOR_SIGMAS². This makes the leapfrog position step ε·M⁻¹·p have the
# SAME relative magnitude (in units of prior σ) across all dimensions —
# without it, dimensions with smaller prior σ get oversampled and the
# chain biases toward those dimensions' modes.
const INV_MASS = PRIOR_SIGMAS .^ 2
const SQRT_INV_MASS = sqrt.(INV_MASS)

function parallel_hmc_seg!(U::Matrix{Float64}, target, ε::Float64, L::Int, rng)
    M, d = size(U)
    # Sample momentum p ~ N(0, M) where M = 1/inv_mass = 1/PRIOR_SIGMAS²
    momentum = randn(rng, M, d) ./ SQRT_INV_MASS'

    function tempered(U_in)
        vals_data, grads_data = gpu_grads_parallel_chains_segmented(target, U_in; h = FD_H)
        return vals_data .+ logprior_val(U_in), grads_data .+ logprior_grad(U_in)
    end

    val_init, grad_init = tempered(U)
    p_old = copy(momentum)
    p = momentum .+ (ε / 2) .* grad_init
    U_new = U .+ ε .* p .* INV_MASS'
    for _ in 2:L
        _, g = tempered(U_new)
        p .= p .+ ε .* g
        U_new .= U_new .+ ε .* p .* INV_MASS'
    end
    val_final, grad_final = tempered(U_new)
    p .= p .+ (ε / 2) .* grad_final
    K0 = 0.5 .* vec(sum(p_old .^ 2 .* INV_MASS'; dims = 2))
    KN = 0.5 .* vec(sum(p     .^ 2 .* INV_MASS'; dims = 2))
    log_α = (val_final .- KN) .- (val_init .- K0)
    n_acc = 0
    for m in 1:M
        if log(rand(rng)) < log_α[m]
            U[m, :] = U_new[m, :]; n_acc += 1
        end
    end
    return n_acc
end

function ess_at(δ, ll)
    log_w = δ .* ll
    log_w_n = log_w .- logsumexp(log_w)
    return exp(-logsumexp(2.0 .* log_w_n))
end

# ChEES adapt L on a small subset of chains. Runs a probe HMC move at each
# candidate L, measures expected squared jumped distance per leapfrog step,
# picks the L that maximises ESJD/L. Mirrors `chees_pick_L_parallel` from
# bench_b3_gpu_parallel.jl, adapted to use the segmented gradient.
function chees_pick_L_seg(target, U_subset::AbstractMatrix{Float64};
                            ε::Float64 = HMC_STEP,
                            L_candidates::Vector{Int} = [2, 4, 8, 16],
                            n_steps::Int = 1,
                            rng::AbstractRNG = MersenneTwister(0))
    best_L = first(L_candidates)
    best_score = -Inf
    for L in L_candidates
        U_try = copy(U_subset)
        for _ in 1:n_steps
            parallel_hmc_seg!(U_try, target, ε, L, rng)
        end
        sqd = sum(abs2, U_try .- U_subset)
        score = sqd / (size(U_subset, 1) * L * ε)
        if score > best_score
            best_score = score; best_L = L
        end
    end
    return best_L, best_score
end

function run_outer_smc!(U::Matrix{Float64}, target)
    λ = 0.0; n_temp = 0
    L_used = HMC_LEAPFROG   # initial; ChEES will adapt per level
    while λ < 1.0 - 1e-6
        ll = gpu_log_density_batched_segmented(target, U)
        δ_max = min(1.0 - λ, MAX_LAMBDA)
        target_ess = TARGET_ESS * N_SMC
        δ = if ess_at(δ_max, ll) >= target_ess
            δ_max
        else
            lo, hi = 0.0, δ_max
            for _ in 1:30
                mid = 0.5 * (lo + hi)
                if ess_at(mid, ll) > target_ess; lo = mid; else; hi = mid; end
            end
            lo
        end
        next_λ = (λ + δ < 1.0 - 1e-6) ? λ + δ : 1.0
        Δλ = next_λ - λ
        log_w = Δλ .* ll
        w = exp.(log_w .- logsumexp(log_w))
        cumsum_w = cumsum(w)
        indices = Vector{Int}(undef, N_SMC)
        u_shift = 0.5 / N_SMC
        for i in 1:N_SMC
            t = (i - 1) / N_SMC + u_shift
            indices[i] = clamp(searchsortedfirst(cumsum_w, t), 1, N_SMC)
        end
        U_resampled = U[indices, :]

        # ChEES adapt L on a small subset (≤8 chains) BEFORE the main HMC moves.
        chees_subset = U_resampled[1:min(8, N_SMC), :]
        L_used, _ = chees_pick_L_seg(target, chees_subset; ε = HMC_STEP,
                                       L_candidates = [2, 4, 8, 16],
                                       n_steps = 1,
                                       rng = MersenneTwister(11 + n_temp))

        n_acc = 0
        for _ in 1:NUM_MCMC
            n_acc += parallel_hmc_seg!(U_resampled, target, HMC_STEP, L_used,
                                         MersenneTwister(101 * n_temp))
        end
        accept_frac = n_acc / (NUM_MCMC * N_SMC)
        copyto!(U, U_resampled)
        n_temp += 1
        println(@sprintf("[%2d] λ %.3f → %.3f  Δλ=%.3f  L=%d  accept=%.0f%%",
                          n_temp, λ, next_λ, Δλ, L_used, 100 * accept_frac))
        λ = next_λ
        n_temp >= 30 && break
    end
    return n_temp
end

SEED_INIT = parse(Int, get(ENV, "BENCH_SEED", "11"))
println("\n--- Outer SMC² with segmented GPU PF (seed=$SEED_INIT) ---")
rng_init = MersenneTwister(SEED_INIT)
U = Matrix{Float64}(undef, N_SMC, 8)
for m in 1:N_SMC, j in 1:8
    U[m, j] = PRIOR_MEANS[j] + PRIOR_SIGMAS[j] * randn(rng_init)
end

# Quick sanity check: log-density at the initial cloud — should be finite,
# vary across chains, and not be wildly off from a Gaussian-around-prior.
println("\n[sanity] evaluating log p(y|θ) on initial cloud + a single FD probe")
# Truth-vs-prior LL probe: compute LL at the truth-θ and at a few offset
# positions to check the LL surface peaks at truth (not at a wrong basin).
let
    U_truth = zeros(64, 8)
    U_truth[:, 1] .= log(PARAM_SET_A.alpha)
    U_truth[:, 2] .= log(PARAM_SET_A.a)
    U_truth[:, 3] .= log(PARAM_SET_A.sigma_x)
    U_truth[:, 4] .= log(PARAM_SET_A.gamma)
    U_truth[:, 5] .= log(PARAM_SET_A.sigma_u)
    U_truth[:, 6] .= log(PARAM_SET_A.sigma_obs)
    U_truth[:, 7] .= INIT_STATE_A.x_0
    U_truth[:, 8] .= INIT_STATE_A.u_0
    ll_truth = gpu_log_density_batched_segmented(target, U_truth)
    println("  LL at TRUTH (all 64 chains identical): " *
            "min=$(round(minimum(ll_truth); digits=2))  " *
            "max=$(round(maximum(ll_truth); digits=2))  " *
            "mean=$(round(mean(ll_truth); digits=2))  " *
            "spread=$(round(maximum(ll_truth)-minimum(ll_truth); digits=4))")

    # Probe σ_u in isolation: vary log(σ_u) over a finer grid around truth (-3.0)
    println("  σ_u LL surface (log p(y|θ_truth, σ_u replaced)):")
    for log_sigma_u in [-5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, 0.0]
        U_probe = copy(U_truth)
        U_probe[:, 5] .= log_sigma_u
        ll_probe = gpu_log_density_batched_segmented(target, U_probe)
        # Posterior log-density at this σ_u (LL minus prior penalty)
        prior_log = -0.5 * (log_sigma_u - (-3.0))^2 / (0.5^2)
        post_log  = mean(ll_probe) + prior_log
        println("    σ_u=$(round(exp(log_sigma_u); digits=4)) (log=$log_sigma_u): " *
                "LL=$(round(mean(ll_probe); digits=2))  " *
                "prior=$(round(prior_log; digits=2))  " *
                "post=$(round(post_log; digits=2))")
    end
    # Probe α direction
    println("  α LL surface:")
    for log_alpha in [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
        U_probe = copy(U_truth)
        U_probe[:, 1] .= log_alpha
        ll_probe = gpu_log_density_batched_segmented(target, U_probe)
        prior_log = -0.5 * log_alpha^2 / (0.5^2)
        post_log  = mean(ll_probe) + prior_log
        println("    α=$(round(exp(log_alpha); digits=4)) (log=$log_alpha): " *
                "LL=$(round(mean(ll_probe); digits=2))  " *
                "prior=$(round(prior_log; digits=2))  " *
                "post=$(round(post_log; digits=2))")
    end
end

let
    sanity_t0 = time()
    ll_init = gpu_log_density_batched_segmented(target, U)
    sanity_t1 = time()
    println("  initial-cloud ll: min=$(round(minimum(ll_init); digits=2))  " *
            "max=$(round(maximum(ll_init); digits=2))  " *
            "mean=$(round(mean(ll_init); digits=2))  " *
            "n_finite=$(count(isfinite, ll_init))/$(length(ll_init))  " *
            "wall=$(round(sanity_t1-sanity_t0; digits=2))s")
    vals, grads = gpu_grads_parallel_chains_segmented(target, U; h = FD_H)
    println("  primal vals: min=$(round(minimum(vals); digits=2)) " *
            "max=$(round(maximum(vals); digits=2)) " *
            "mean=$(round(mean(vals); digits=2))")
    println("  gradient norms (per chain): min=$(round(minimum(vec(sqrt.(sum(grads.^2; dims=2)))); digits=3)) " *
            "max=$(round(maximum(vec(sqrt.(sum(grads.^2; dims=2)))); digits=3)) " *
            "mean=$(round(mean(vec(sqrt.(sum(grads.^2; dims=2)))); digits=3))")
end

t_start = time()
n_temp = run_outer_smc!(U, target)
t_filter = time() - t_start
println("\nPhase 1 filter: $(n_temp) levels, $(round(t_filter; digits=1)) s wall (segmented R=$R_SEGMENT)")

# Posterior summary
truth = (alpha = PARAM_SET_A.alpha, a = PARAM_SET_A.a,
          sigma_x = PARAM_SET_A.sigma_x, gamma = PARAM_SET_A.gamma,
          sigma_u = PARAM_SET_A.sigma_u, sigma_obs = PARAM_SET_A.sigma_obs)
pcs = similar(U)
for m in 1:N_SMC; pcs[m, 1:6] = exp.(U[m, 1:6]); pcs[m, 7:8] = U[m, 7:8]; end
pm = vec(mean(pcs; dims = 1))
println("\nPhase 1 posterior means (rel.err vs truth):")
for (i, k) in enumerate([:alpha, :a, :sigma_x, :gamma, :sigma_u, :sigma_obs])
    rel = abs(pm[i] - truth[k]) / truth[k]
    println(@sprintf("  %-10s truth=%.4f  mean=%.4f  rel.err=%.1f %%",
                      string(k), truth[k], pm[i], 100 * rel))
end
let factor = round(t_filter / 23.0; digits=1)
    println("\nWall time vs unresampled GPU (was 23 s): $(round(t_filter; digits=1)) s = $(factor)x slower")
end
