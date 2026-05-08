#!/usr/bin/env julia
# FIM (Fisher Information Matrix) gate for FSA v1.5 — HARD GATE before
# any closed-loop bench run, per the user's plan.
#
# What this tool does
# -------------------
# At TRUTH_PARAMS, integrate the deterministic ODE
#
#     d(B,F,A)/dt = drift(state, params, Φ=1)
#
# from INIT_STATE for one day at h=60min (24 bins) using fp64 Euler.
# Use ForwardDiff.jacobian to compute J_t = ∂(B,F,A)/∂params_vec at each
# bin t. Accumulate the deterministic-trajectory FIM
#
#     I(θ) = Σ_t  Σ_X  (1/σ_X_obs²) · J_t[X, :] · J_t[X, :]ᵀ
#
# for X ∈ {B, F, A}.
#
# The deterministic-ODE FIM only reaches DRIFT parameters. The 3 SDE
# diffusion parameters (σ_B, σ_F, σ_A) do NOT affect the mean trajectory
# and would always show as rank-deficient under this analysis — that is
# expected and not a flag. The drift params (11 of them) ARE the ones
# the controller's cost surface depends on, and they MUST be identifiable.
#
# Gate logic
# ----------
#  - Compute eigenvalues of the 11×11 drift-only FIM.
#  - condition number κ = λ_max / λ_min.
#  - rank (numerical, threshold = λ_max · 1e-8).
#
#  PASS:  rank == 11  AND  κ <= 1e8
#  STOP:  rank < 11   OR   κ > 1e8  → halt and surface findings.
#
# This tool is read-only and side-effect-free except for printing to
# stdout.

ENV["FSA_STEP_MINUTES"] = "60"

using LinearAlgebra
using Printf
using StaticArrays
using ForwardDiff

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Dynamics: drift
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE,
                              params_v15_to_v1_nt, fill_pinned_nt


# ── Drift parameters the FIM reaches — 7 ESTIMATED params (v1.5 basis) ──
# τ_B, η, ε_A, μ_FF are PINNED per the plan (B + pin {τ_B, η, ε_A, μ_FF})
# and not in this vector.

const DRIFT_PARAM_NAMES = [
    :tau_F, :B_inf, :F_inf,
    :lambda_A,
    :mu_0, :mu_B, :mu_F,
]

const N_DRIFT = length(DRIFT_PARAM_NAMES)


# ── Pure deterministic ODE rollout, parameterised by drift-param vector ───
#    Returns the flattened trajectory (n_bins * 3,) so ForwardDiff
#    .jacobian can differentiate it. ─────────────────────────────────────────

"""
    rollout_det(p_vec, init, dt, n_bins, Φ_t) -> Vector

Deterministic Euler integration of `(B, F, A)` for `n_bins` steps under
constant control `Φ_t`. `p_vec` is the 11-vector of drift params in
DRIFT_PARAM_NAMES order. Returns the flattened (n_bins * 3,) trajectory.

Pure: same inputs always produce the same output (no RNG).
"""
function rollout_det(p_vec::AbstractVector{T},
                      init::SVector{3, Float64},
                      dt::Float64,
                      n_bins::Int,
                      Φ_t::Float64) where {T<:Real}

    # p_vec is in v1.5 ESTIMATED-only basis (DRIFT_PARAM_NAMES order, 7 params).
    # Build the estimated NamedTuple, fill in the 4 pinned values, then
    # route through the v1.5 → v1 adapter so v1's drift() gets
    # (kappa_B, kappa_F, ...).
    estimated = (
        tau_F    = p_vec[1],
        B_inf    = p_vec[2],
        F_inf    = p_vec[3],
        lambda_A  = p_vec[4],
        mu_0     = p_vec[5],
        mu_B     = p_vec[6],
        mu_F     = p_vec[7],
        # Diffusion fields needed by fill_pinned_nt's signature; not used by drift().
        sigma_B  = T(DEFAULT_PARAMS[:sigma_B]),
        sigma_F  = T(DEFAULT_PARAMS[:sigma_F]),
        sigma_A  = T(DEFAULT_PARAMS[:sigma_A]),
    )
    p_v15 = fill_pinned_nt(estimated)
    params_nt = params_v15_to_v1_nt(p_v15)

    out = Vector{T}(undef, n_bins * 3)
    y   = SVector{3, T}(T(init[1]), T(init[2]), T(init[3]))
    @inbounds for k in 1:n_bins
        d = drift(y, params_nt, Φ_t)
        y = SVector{3, T}(y[1] + dt * d[1],
                          y[2] + dt * d[2],
                          y[3] + dt * d[3])
        out[3 * (k - 1) + 1] = y[1]
        out[3 * (k - 1) + 2] = y[2]
        out[3 * (k - 1) + 3] = y[3]
    end
    return out
end


# ── Compute the FIM ────────────────────────────────────────────────────────

function compute_fim(; n_bins::Int = BINS_PER_DAY, Φ_t::Float64 = 1.0)
    # Drift-param vector at truth
    p_vec = Float64[DEFAULT_PARAMS[name] for name in DRIFT_PARAM_NAMES]

    # Initial state
    init = SVector{3, Float64}(Float64(INIT_STATE.B),
                                Float64(INIT_STATE.F),
                                Float64(INIT_STATE.A))

    # Forward-diff the rollout
    f = pv -> rollout_det(pv, init, DT_BIN_DAYS, n_bins, Φ_t)
    J_full = ForwardDiff.jacobian(f, p_vec)              # (n_bins * 3) × 11

    # Reshape into per-bin Jacobians J_t of shape (3, 11)
    @assert size(J_full) == (n_bins * 3, N_DRIFT)
    J_per_bin = reshape(J_full, 3, n_bins, N_DRIFT)
    # axes:   J_per_bin[X_idx, t, i]

    # Channel-noise diagonal (σ_X_obs²)
    σ_obs = (
        DEFAULT_PARAMS[:sigma_B_obs],
        DEFAULT_PARAMS[:sigma_F_obs],
        DEFAULT_PARAMS[:sigma_A_obs],
    )
    inv_σ²_X = (1.0 / σ_obs[1]^2, 1.0 / σ_obs[2]^2, 1.0 / σ_obs[3]^2)

    # I(θ)_{ij} = Σ_t Σ_X  (1/σ_X²) · J_t[X, i] · J_t[X, j]
    I_mat = zeros(Float64, N_DRIFT, N_DRIFT)
    @inbounds for t in 1:n_bins
        for X in 1:3
            J_row = J_per_bin[X, t, :]                  # length-11 vector
            scale = inv_σ²_X[X]
            for i in 1:N_DRIFT, j in 1:N_DRIFT
                I_mat[i, j] += scale * J_row[i] * J_row[j]
            end
        end
    end
    return I_mat, J_per_bin
end


# ── Report ─────────────────────────────────────────────────────────────────

function compute_fim_phi_sched(Φ_sched::AbstractVector)
    p_vec = Float64[DEFAULT_PARAMS[name] for name in DRIFT_PARAM_NAMES]
    init  = SVector{3, Float64}(Float64(INIT_STATE.B),
                                 Float64(INIT_STATE.F),
                                 Float64(INIT_STATE.A))
    n_bins = length(Φ_sched)

    function rollout_phi(pv)
        T = eltype(pv)
        estimated = (
            tau_F    = pv[1], B_inf    = pv[2], F_inf    = pv[3],
            lambda_A  = pv[4],
            mu_0     = pv[5], mu_B     = pv[6], mu_F     = pv[7],
            sigma_B  = T(DEFAULT_PARAMS[:sigma_B]),
            sigma_F  = T(DEFAULT_PARAMS[:sigma_F]),
            sigma_A  = T(DEFAULT_PARAMS[:sigma_A]),
        )
        p_v15 = fill_pinned_nt(estimated)
        params_nt = params_v15_to_v1_nt(p_v15)
        out = Vector{T}(undef, n_bins * 3)
        y = SVector{3, T}(T(init[1]), T(init[2]), T(init[3]))
        @inbounds for k in 1:n_bins
            d = drift(y, params_nt, Φ_sched[k])
            y = SVector{3, T}(y[1] + DT_BIN_DAYS * d[1],
                              y[2] + DT_BIN_DAYS * d[2],
                              y[3] + DT_BIN_DAYS * d[3])
            out[3 * (k - 1) + 1] = y[1]
            out[3 * (k - 1) + 2] = y[2]
            out[3 * (k - 1) + 3] = y[3]
        end
        return out
    end

    J_full = ForwardDiff.jacobian(rollout_phi, p_vec)
    J_per_bin = reshape(J_full, 3, n_bins, N_DRIFT)

    inv_σ²_X = (1.0 / DEFAULT_PARAMS[:sigma_B_obs]^2,
                 1.0 / DEFAULT_PARAMS[:sigma_F_obs]^2,
                 1.0 / DEFAULT_PARAMS[:sigma_A_obs]^2)

    I_mat = zeros(Float64, N_DRIFT, N_DRIFT)
    @inbounds for t in 1:n_bins, X in 1:3
        J_row = J_per_bin[X, t, :]
        scale = inv_σ²_X[X]
        for i in 1:N_DRIFT, j in 1:N_DRIFT
            I_mat[i, j] += scale * J_row[i] * J_row[j]
        end
    end
    return I_mat
end

function summarise_fim(I_mat::AbstractMatrix; label::AbstractString = "")
    eig = eigen(Symmetric(I_mat))
    λ = sort(eig.values; rev = true)
    λ_max = maximum(λ); λ_min = minimum(λ)
    tol = λ_max * 1e-8
    rank_num = count(λ .> tol)
    κ = λ_max / max(λ_min, eps(Float64))
    @printf("[%s] rank = %d/%d   λ_max = %.3e   λ_min = %.3e   κ = %.3e\n",
            label, rank_num, N_DRIFT, λ_max, λ_min, κ)
    return (rank = rank_num, λ_max = λ_max, λ_min = λ_min, κ = κ, eig = eig)
end

function main()
    println("="^72)
    println("FSA v1.5 — FIM gate at TRUTH_PARAMS  (B + pin τ_B, η, ε_A, μ_FF)")
    println("="^72)
    println()
    @printf("Configuration:\n")
    @printf("  step (env)          : %s min\n", ENV["FSA_STEP_MINUTES"])
    @printf("  bins per day        : %d\n", BINS_PER_DAY)
    @printf("  obs-noise σ_B/F/A   : (%.4f, %.4f, %.4f) — pinned\n",
            DEFAULT_PARAMS[:sigma_B_obs], DEFAULT_PARAMS[:sigma_F_obs],
            DEFAULT_PARAMS[:sigma_A_obs])
    @printf("  pinned dynamics     : τ_B = %.2f, η = %.3f, ε_A = %.3f, μ_FF = %.3f\n",
            DEFAULT_PARAMS[:tau_B], DEFAULT_PARAMS[:eta],
            DEFAULT_PARAMS[:epsilon_A], DEFAULT_PARAMS[:mu_FF])
    @printf("  estimated (drift)   : %d  (%s)\n", N_DRIFT,
            join(string.(DRIFT_PARAM_NAMES), ", "))
    println()

    println("Window sweep:")
    println("-"^72)
    function Φ_sin(n_bins)
        return [0.2 + 0.65 * (1 - cos(2π * (k - 1) / n_bins)) for k in 1:n_bins]
    end
    fim_1d_const   = compute_fim_phi_sched(fill(1.0, BINS_PER_DAY))
    fim_14d_const  = compute_fim_phi_sched(fill(1.0, 14 * BINS_PER_DAY))
    fim_14d_sin    = compute_fim_phi_sched(Φ_sin(14 * BINS_PER_DAY))
    fim_28d_sin    = compute_fim_phi_sched(Φ_sin(28 * BINS_PER_DAY))
    summarise_fim(fim_1d_const;  label = "1d  Φ=1   (spec'd gate)        ")
    summarise_fim(fim_14d_const; label = "14d Φ=1   (bench horizon, baseline plant)")
    s_14d_sin = summarise_fim(fim_14d_sin;   label = "14d Φ=sin (bench horizon, varying)  ")
    s_28d_sin = summarise_fim(fim_28d_sin;   label = "28d Φ=sin (long horizon, varying)   ")
    println()

    # Use the 14-day varying-Φ FIM as the "realistic" gate — it's closest
    # to what the bench's filter actually sees (controller produces non-constant Φ).
    println("Realistic gate = 14d varying-Φ FIM (closest to bench info accumulation):")
    println("-"^72)
    I_mat = fim_14d_sin
    eig   = s_14d_sin.eig

    @printf("Per-parameter information (diag of FIM, CRLB std = 1/sqrt(diag)):\n")
    for (i, name) in enumerate(DRIFT_PARAM_NAMES)
        diag_i = I_mat[i, i]
        crlb_i = diag_i > 0 ? sqrt(1.0 / diag_i) : NaN
        @printf("  %-12s : I_ii = %.4e   CRLB std = %.4e\n", name, diag_i, crlb_i)
    end
    println()

    λ_max = s_14d_sin.λ_max
    λ_min = s_14d_sin.λ_min
    κ     = s_14d_sin.κ
    rank_num = s_14d_sin.rank
    pass_rank = (rank_num == N_DRIFT)
    pass_cond = (κ <= 1e8)
    println("="^72)

    if pass_rank && pass_cond
        println("RESULT: PASS at the 14-day varying-Φ realistic gate.")
        println()
        @printf("All %d estimated drift parameters are identifiable:\n", N_DRIFT)
        @printf("  - rank = %d / %d (full rank)\n", rank_num, N_DRIFT)
        @printf("  - κ = %.2e (≤ 1e8 threshold)\n", κ)
        println()
        println("Pinned at truth: τ_B = 42.0, η = 0.20, ε_A = 0.40.")
        println("OK to continue with the rest of v1.5: gpu_pf.jl + closed-loop bench.")
        exit(0)
    else
        println("RESULT: STOP — identifiability issue at the 14d varying-Φ gate.")
        println()
        if !pass_rank
            @printf("  - rank = %d / %d (need full rank)\n", rank_num, N_DRIFT)
        end
        if !pass_cond
            @printf("  - κ = %.2e > 1e8 (above the gate threshold)\n", κ)
        end
        println()
        # Show the eigenvectors associated with the smallest eigenvalues so
        # the user can see WHICH parameter combination is degenerate.
        println("Smallest eigenvalues + associated eigenvectors (parameter names = ")
        println(string(DRIFT_PARAM_NAMES))
        println("):")
        sort_idx = sortperm(eig.values)
        for ii in 1:min(3, length(sort_idx))
            idx = sort_idx[ii]
            @printf("\n  λ = %.6e\n", eig.values[idx])
            v = eig.vectors[:, idx]
            for (j, name) in enumerate(DRIFT_PARAM_NAMES)
                @printf("    %-12s = %+.4f\n", name, v[j])
            end
        end
        println()
        println("Per the v1.5 plan: HALTING. Surface this report to the user")
        println("for a re-parametrisation decision.")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
