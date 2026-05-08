#!/usr/bin/env julia
# Test the framework's GPU controller (parallel-chains tempered SMC² +
# ChEES-HMC + FD gradient) on three synthetic analytic costs with KNOWN
# optima. Pure CPU log-density closures — no model, no filter, no GPU
# kernel. Isolates whether the framework's controller logic itself is
# correct, independent of the FSA cost kernel.
#
# If the framework passes all three:
#   → controller logic is sound; the flat-Φ behaviour we see on FSA must
#     be either a real property of the FSA cost surface or a bug in the
#     model-specific GPU cost kernel.
#
# If the framework fails any test:
#   → the bug is in `julia/SMC2FC/src/Control/GPUControlSMC.jl` (HMC,
#     ChEES, FD batcher, or the outer tempered-SMC² loop).

using Random, Statistics, Printf, LinearAlgebra
using SMC2FC: run_tempered_smc_gpu


# ── Test 1: 8-D quadratic bowl ───────────────────────────────────────────
# J(θ) = (1/2) Σ ((θ_i − θ_i^*) / σ_J)²
# Trivial — global minimum at θ* with no other minima. If HMC can't find
# this, something fundamental is broken.

function make_quadratic_bowl(theta_star::Vector{Float64}, sigma_J::Float64)
    function log_density(U::AbstractMatrix{Float64})
        M = size(U, 1)
        out = Vector{Float64}(undef, M)
        @inbounds for m in 1:M
            δ = view(U, m, :) .- theta_star
            out[m] = -0.5 * sum(abs2, δ ./ sigma_J)   # log p = -cost
        end
        return out
    end
    return log_density
end


# ── Test 2: anti-symmetric optimum ───────────────────────────────────────
# J(θ) = -w·θ + γ ||θ||²
# Optimum θ* = w / (2γ).  Choose w anti-symmetric across the 8 anchors
# (negative for early, positive for late) so the optimum has the same
# spatial structure as the FSA "recovery → overload" pattern.

function make_anti_symmetric_reward(w::Vector{Float64}, γ::Float64)
    function log_density(U::AbstractMatrix{Float64})
        M = size(U, 1)
        out = Vector{Float64}(undef, M)
        @inbounds for m in 1:M
            θ = view(U, m, :)
            out[m] = dot(w, θ) - γ * sum(abs2, θ)   # log p = w·θ - γ‖θ‖²
        end
        return out
    end
    return log_density
end


# ── Test 3: bimodal — two competing Gaussian wells ───────────────────────
# log p(θ) = logsumexp( log w_a − ‖θ − θ_a‖²/(2σ²),  log w_b − ‖θ − θ_b‖²/(2σ²) )
# θ_a = uniform vector (matches "flat" Φ basin); θ_b = anti-symmetric
# (matches recovery → overload basin). w_b > w_a so θ_b is the GLOBAL mode.
# Tests whether the controller can escape the larger-volume basin to find
# the deeper but narrower one.

function make_bimodal(theta_a::Vector{Float64}, theta_b::Vector{Float64},
                      sigma::Float64, log_weight_a::Float64, log_weight_b::Float64)
    function log_density(U::AbstractMatrix{Float64})
        M = size(U, 1)
        out = Vector{Float64}(undef, M)
        @inbounds for m in 1:M
            θ = view(U, m, :)
            la = log_weight_a - 0.5 * sum(abs2, (θ .- theta_a) ./ sigma)
            lb = log_weight_b - 0.5 * sum(abs2, (θ .- theta_b) ./ sigma)
            mx = max(la, lb)
            out[m] = mx + log(exp(la - mx) + exp(lb - mx))
        end
        return out
    end
    return log_density
end


# ── Helper to summarize controller posterior vs known optimum ────────────
function report(name::String, posterior::Matrix{Float64}, theta_star::Vector{Float64})
    post_mean = vec(mean(posterior; dims=1))
    err = norm(post_mean .- theta_star)
    println("="^72)
    println(name)
    println("-"^72)
    @printf("  known optimum θ*   = %s\n",
            "[" * join([@sprintf("%+.3f", v) for v in theta_star], ", ") * "]")
    @printf("  posterior mean θ̂   = %s\n",
            "[" * join([@sprintf("%+.3f", v) for v in post_mean], ", ") * "]")
    @printf("  ‖θ̂ − θ*‖₂         = %.4f\n", err)
    @printf("  per-coord max-err  = %.4f\n", maximum(abs.(post_mean .- theta_star)))
    return err
end


# ── Common controller config (matches the FSA bench) ────────────────────
const D            = 8
const N_SMC        = 1024
const N_ANCHORS    = 8
const M_MAX        = N_SMC * (1 + 2 * N_ANCHORS)   # 17408
const SIGMA_PRIOR  = 1.5
const TARGET_NATS  = 8.0
const HMC_STEP     = 0.2
const HMC_LEAP     = 16
const NUM_MCMC     = 10
const MAX_LEVELS   = 30
const TARGET_ESS   = 0.5
const MAX_LAMBDA   = 0.20
const CHEES_LS     = [16, 32, 64, 128, 256]


function run_test(name::String, log_density_fn::Function, theta_star::Vector{Float64};
                  init::Union{Nothing,Vector{Float64}} = nothing)
    rng = MersenneTwister(123)
    t0 = time()
    posterior, n_temp, β_max = run_tempered_smc_gpu(
        log_density_fn, M_MAX, N_SMC, D,
        0.0, SIGMA_PRIOR, rng;
        target_nats        = TARGET_NATS,
        target_ess_frac    = TARGET_ESS,
        max_lambda_inc     = MAX_LAMBDA,
        max_temp_levels    = MAX_LEVELS,
        num_mcmc_steps     = NUM_MCMC,
        hmc_step_size      = HMC_STEP,
        hmc_num_leapfrog   = HMC_LEAP,
        chees_L_candidates = CHEES_LS,
        h_fd               = 1e-4,
        calib_n            = 64,
        init_particles     = init,
        verbose            = false,
    )
    t = time() - t0
    err = report(name, posterior, theta_star)
    @printf("  framework: %d levels, β_max=%.3f, %.1fs\n", n_temp, β_max, t)
    return err
end


# ── Run all three tests ──────────────────────────────────────────────────

println("\n" * "="^72)
println("FRAMEWORK CONTROLLER TEST SUITE")
println("Same config as FSA bench: n_smc=$N_SMC, n_inner is N/A (analytic costs)")
println("="^72)

# Test 1: quadratic bowl, optimum off-prior-mean
theta_star_quad = [+0.5, -0.3, +0.7, -0.4, +0.2, -0.6, +0.8, -0.1]
quad = make_quadratic_bowl(theta_star_quad, 1.0)
err1 = run_test("Test 1: 8-D quadratic bowl  (σ_J = 1.0)", quad, theta_star_quad)
println()

# Test 2: anti-symmetric reward (mirrors FSA "recovery → overload" structure)
w = [-1.0, -1.0, -1.0, -0.5, +0.5, +1.0, +1.0, +1.0]   # spatially anti-symmetric
γ = 0.5
theta_star_anti = w ./ (2γ)
anti = make_anti_symmetric_reward(w, γ)
err2 = run_test("Test 2: anti-symmetric reward  (J = -w·θ + γ‖θ‖², θ* = w/(2γ))",
                anti, theta_star_anti)
println()

# Test 3: bimodal — flat (uniform) basin AND anti-symmetric basin.
# Anti-symmetric basin is GLOBAL (lower cost = higher log_w_b)
θ_a = fill(-1.0, D)             # "flat-low Φ" mode (where my FSA HMC ends up)
θ_b = [-1.5, -1.5, -1.5, 0.0, 0.0, +1.5, +1.5, +1.5]   # "recovery-overload" mode
log_wa = 0.0
log_wb = 1.0                     # b is e¹ ≈ 2.7× more probable per particle
bimo = make_bimodal(θ_a, θ_b, 0.5, log_wa, log_wb)

println("-- Test 3a: bimodal, HMC initialised from prior N(0, σ²·I) --")
err3a = run_test("Test 3a", bimo, θ_b)   # report against the GLOBAL mode θ_b
println()
println("-- Test 3b: bimodal, HMC initialised at θ_a (the flat-low basin) --")
err3b = run_test("Test 3b", bimo, θ_b; init = θ_a)
println()
println("-- Test 3c: bimodal, HMC initialised at θ_b (the global basin) --")
err3c = run_test("Test 3c", bimo, θ_b; init = θ_b)
println()

println("="^72)
println("SUMMARY")
println("="^72)
println("Pass criterion: ‖θ̂ − θ*‖₂ within ~σ_prior/√n_smc (≈ 0.05) for")
println("convex tests (1, 2). For multi-modal (3a-c) interpret as:")
println("  3a from prior:  if found θ_b → framework escapes basins.")
println("                  if found θ_a → framework gets stuck in larger basin.")
println("  3b from θ_a:    if escapes to θ_b → framework finds global mode.")
println("                  if stays at θ_a → framework can't escape local mode.")
println("  3c from θ_b:    must stay at θ_b (sanity).")
@printf("  Test 1 (quadratic bowl):      err = %.4f\n", err1)
@printf("  Test 2 (anti-symmetric):      err = %.4f\n", err2)
@printf("  Test 3a (bimodal, prior init): err vs θ_b = %.4f\n", err3a)
@printf("  Test 3b (bimodal, init at θ_a): err vs θ_b = %.4f\n", err3b)
@printf("  Test 3c (bimodal, init at θ_b): err vs θ_b = %.4f\n", err3c)
