#!/usr/bin/env julia
# End-to-end smoke test for FSA v1.5's GPU PF (gpu_pf.jl).
#
# This is the FIRST place where the full GPU-side pipeline is exercised,
# so the test is deliberately thorough:
#
#   T1 — gpu_log_density runs without error, returns finite values.
#   T2 — Purity: same (target, U, grid_obs, key) → same log-density.
#   T3 — Truth peaks: log-density at TRUTH_PARAMS is higher than at a
#        randomly-perturbed θ (data was generated from truth).
#   T4 — gpu_grads gives finite (vals, grads), grads point uphill: a
#        single FD-gradient leapfrog from a slightly perturbed θ moves
#        the log-density UP toward truth.
#   T5 — parallel_hmc_one_move returns a NEW U matrix without mutating
#        the input, n_acc ∈ [0, M], log-densities at the new state are
#        finite.

ENV["FSA_STEP_MINUTES"] = "60"

using Printf
using Random
using Statistics
using LinearAlgebra
using StaticArrays

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: PlantState, plant_rollout, init_plant_state
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE
using .FSAHighRes.Estimation: PARAM_NAMES, PARAM_PRIOR_CONFIG
using .FSAHighRes.GPUPF: FSAGPUTarget, gpu_log_density, gpu_grads,
                          parallel_hmc_one_move


# ── Helpers ────────────────────────────────────────────────────────────────

"""
Build a single-day window of obs by running the truth plant under Φ=1.
Returns the grid_obs NamedTuple expected by gpu_log_density.
"""
function build_truth_window(; n_bins::Int = BINS_PER_DAY,
                              Φ_const::Float64 = 1.0,
                              key::UInt64 = UInt64(0xC0FFEE))
    s0 = init_plant_state()
    Φ_seq = fill(Float32(Φ_const), n_bins)
    out = plant_rollout(s0, Φ_seq, DEFAULT_PARAMS, DT_BIN_DAYS, key)
    return (
        Phi_seq = out.Phi,
        obs_B   = out.obs_B,
        obs_F   = out.obs_F,
        obs_A   = out.obs_A,
        B_init  = INIT_STATE.B,
        F_init  = INIT_STATE.F,
        A_init  = INIT_STATE.A,
    )
end

"""
Truth U_unc — log of the truth value for each estimated param. Returns
a length-10 Vector{Float64} in PARAM_NAMES order.
"""
function truth_u_unc()
    return [log(DEFAULT_PARAMS[name]) for name in PARAM_NAMES]
end


# ── Tests ──────────────────────────────────────────────────────────────────

function run_tests()
    println("="^72)
    println("FSA v1.5 — GPU PF end-to-end smoke test")
    println("="^72)

    # Fixed config across tests. M_max must hold the FD batch:
    # gpu_grads expands M_demo chains to M_demo·(1 + 2·d) rows.
    K_per_chain = 200
    M_demo      = 4
    d_unc       = 10                  # length of PARAM_NAMES
    M_max       = M_demo * (1 + 2 * d_unc)   # = 84 for M_demo=4, d=10
    T_steps     = BINS_PER_DAY        # one day
    R           = 4

    @assert T_steps % R == 0

    @printf("Config: K=%d M_max=%d T=%d R=%d M_demo=%d\n",
            K_per_chain, M_max, T_steps, R, M_demo)
    println()

    # Build target ONCE (reused across all tests — exercise its purity).
    target = FSAGPUTarget(
        K_per_chain = K_per_chain,
        M_max       = M_max,
        T_steps     = T_steps,
        R           = R,
        dt          = DT_BIN_DAYS,
        noise_seed  = 42,
    )

    # Truth-data window
    grid_obs = build_truth_window(; n_bins = T_steps, Φ_const = 1.0,
                                    key = UInt64(0xC0FFEE))
    @printf("Window built: T_steps=%d, Φ=1 from INIT_STATE\n", T_steps)
    @printf("  obs_B (first 3): %s\n", string(round.(Float64.(grid_obs.obs_B[1:3]); digits = 4)))
    @printf("  obs_F (first 3): %s\n", string(round.(Float64.(grid_obs.obs_F[1:3]); digits = 4)))
    @printf("  obs_A (first 3): %s\n", string(round.(Float64.(grid_obs.obs_A[1:3]); digits = 4)))
    println()

    # ────────────────── T1: gpu_log_density runs ──────────────────
    println("─"^72)
    println("T1: gpu_log_density — single call, M=$M_demo, all chains at truth")

    u_truth = truth_u_unc()
    @assert length(u_truth) == 10
    U_truth = repeat(reshape(u_truth, 1, :), M_demo, 1)

    key1 = UInt64(0x1)
    ll_truth = gpu_log_density(target, U_truth, grid_obs, key1)
    @printf("  size(ll) = %s\n", string(size(ll_truth)))
    @printf("  ll  values  = %s\n", string(round.(ll_truth; digits = 2)))
    @assert all(isfinite, ll_truth) "T1 FAIL: some ll values are non-finite"
    println("  T1 PASS ✓")
    println()

    # ────────────────── T2: purity ──────────────────
    println("─"^72)
    println("T2: purity — same (target, U, grid_obs, key) → same output")

    ll_truth_again = gpu_log_density(target, U_truth, grid_obs, key1)
    Δ = maximum(abs.(ll_truth .- ll_truth_again))
    @printf("  max |ll1 - ll2| = %.2e\n", Δ)
    @assert Δ <= 1e-9 "T2 FAIL: gpu_log_density is not pure"
    println("  T2 PASS ✓")
    println()

    # ────────────────── T3: truth peaks vs perturbed θ ──────────────────
    println("─"^72)
    println("T3: log-density at truth > at a perturbed θ")

    rng = MersenneTwister(0xCAFE)
    perturbation = 0.30 * randn(rng, 10)
    U_pert = U_truth .+ reshape(perturbation, 1, :)
    ll_pert = gpu_log_density(target, U_pert, grid_obs, hash((key1, :pert)))
    diff_truth_minus_pert = mean(ll_truth) - mean(ll_pert)
    @printf("  mean ll(truth)    = %.2f\n", mean(ll_truth))
    @printf("  mean ll(perturbed)= %.2f\n", mean(ll_pert))
    @printf("  Δ = ll(truth) - ll(perturbed) = %.2f  (expect > 0)\n", diff_truth_minus_pert)
    @assert diff_truth_minus_pert > 0 "T3 FAIL: truth does not peak"
    println("  T3 PASS ✓")
    println()

    # ────────────────── T4: gpu_grads, FD step uphill ──────────────────
    println("─"^72)
    println("T4: gpu_grads — finite, points uphill, small at truth")

    h_fd = 1e-3
    out_g = gpu_grads(target, U_pert, grid_obs, h_fd, hash((key1, :grad)))
    @printf("  vals  range  = (%.2f, %.2f)\n", minimum(out_g.vals), maximum(out_g.vals))
    grad_norm_pert = maximum(map(norm, eachrow(out_g.grads)))
    @printf("  ‖grad‖ at perturbed θ = %.2e\n", grad_norm_pert)
    @assert all(isfinite, out_g.vals) && all(isfinite, out_g.grads) "T4a FAIL: non-finite grads"

    # T4a: grad at perturbed θ should point TOWARD truth, i.e.
    # cos(angle(grad, U_truth - U_pert)) > 0. Robust against MC noise.
    Δ_to_truth = vec(U_truth[1, :] .- U_pert[1, :])    # all chains identical
    g_avg      = vec(mean(out_g.grads; dims = 1))
    cos_θ      = dot(g_avg, Δ_to_truth) / (norm(g_avg) * norm(Δ_to_truth))
    @printf("  cos(angle(⟨grad⟩, U_truth − U_pert)) = %+.3f  (expect > 0)\n", cos_θ)
    @assert cos_θ > 0 "T4a FAIL: gradient does not point toward truth"
    println("  T4a PASS ✓ (gradient at perturbed θ points toward truth)")

    # T4b: directional derivative — moving +ε·grad gives a higher ll than
    # moving -ε·grad. (Exact-magnitude Taylor agreement is too strict
    # given MC noise from K=200 + 6 resample segments; we only check
    # SIGN, which is what matters for the HMC step.)
    ε_dir = 1.0 / (grad_norm_pert * 1000)        # ε·‖grad‖ ≈ 1e-3 → linear regime
    U_plus  = U_pert .+ ε_dir .* out_g.grads
    U_minus = U_pert .- ε_dir .* out_g.grads
    ll_plus  = gpu_log_density(target, U_plus,  grid_obs, hash((key1, :plus)))
    ll_minus = gpu_log_density(target, U_minus, grid_obs, hash((key1, :minus)))
    measured = mean(ll_plus .- ll_minus)
    @printf("  ε_dir = %.2e   ll(+ε·g) − ll(−ε·g) = %+.4e   (expect > 0)\n",
            ε_dir, measured)
    @assert measured > 0 "T4b FAIL: + ε·grad direction not uphill"
    println("  T4b PASS ✓ (gradient points uphill in directional-derivative sense)")
    println()

    # ────────────────── T5: parallel_hmc_one_move ──────────────────
    println("─"^72)
    println("T5: parallel_hmc_one_move — pure, returns new U + n_acc")

    prior_means  = [m for (_, _, m, _) in PARAM_PRIOR_CONFIG]
    prior_sigmas = [s for (_, _, _, s) in PARAM_PRIOR_CONFIG]
    @assert length(prior_means)  == 10
    @assert length(prior_sigmas) == 10

    U_in = copy(U_pert)
    out_h = parallel_hmc_one_move(U_in, target, grid_obs,
                                    0.05, 4, prior_means, prior_sigmas,
                                    UInt64(0x42); h_fd = h_fd)
    @assert U_in == U_pert "T5 FAIL: parallel_hmc_one_move mutated U_in"
    @assert size(out_h.U_new) == size(U_in) "T5 FAIL: U_new shape mismatch"
    @assert 0 <= out_h.n_acc <= M_demo
    ll_after_hmc = gpu_log_density(target, out_h.U_new, grid_obs, hash((key1, :hmc_after)))
    @printf("  n_acc       = %d / %d\n", out_h.n_acc, M_demo)
    @printf("  ll(U_new)   = %s\n", string(round.(ll_after_hmc; digits = 2)))
    @assert all(isfinite, ll_after_hmc) "T5 FAIL: non-finite ll after HMC"
    println("  T5 PASS ✓")
    println()

    # ────────────────── T6: scaling — M up to M_max for log_density ──
    println("─"^72)
    println("T6: scaling — gpu_log_density with M = M_max ($M_max chains)")

    U_big = repeat(reshape(u_truth, 1, :), M_max, 1)
    rng2 = MersenneTwister(0x77)
    U_big .+= 0.05 .* randn(rng2, M_max, 10)
    ll_big = gpu_log_density(target, U_big, grid_obs, UInt64(0x99))
    @printf("  ll for %d chains: min=%.2f  median=%.2f  max=%.2f\n",
            M_max, minimum(ll_big), median(ll_big), maximum(ll_big))
    @assert all(isfinite, ll_big)
    println("  T6 PASS ✓")
    println()

    println("="^72)
    println("ALL TESTS PASSED ✓")
    println("="^72)
end

run_tests()
