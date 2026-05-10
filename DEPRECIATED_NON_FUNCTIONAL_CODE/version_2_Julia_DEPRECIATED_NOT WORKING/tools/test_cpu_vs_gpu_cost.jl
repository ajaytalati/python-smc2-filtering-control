#!/usr/bin/env julia
# Mirror the GPU cost kernel as a plain CPU function (no @kernel, just
# Julia loops) and compare both against a hand-traced single trajectory.
# If CPU and GPU disagree → GPU kernel has fp32 / index / dispatch bug.
# If they agree but disagree with hand-trace → algorithm bug in both.

ENV["FSA_STEP_MINUTES"] = "15"

using Statistics
using Random
using Printf

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched
using .FSAHighRes.Dynamics: TRUTH_PARAMS, A_TYP, F_TYP

# CPU mirror of fsa_cost_kernel! (single chain, single trial, fp64).
function cpu_cost_one_trial(theta::AbstractVector{Float64},
                            init_state::Vector{Float64},
                            params, fixed_w::AbstractMatrix{Float64},
                            rbf_design::AbstractMatrix{Float64},
                            c_Phi::Float64, Phi_max::Float64,
                            F_max::Float64, lam_phi::Float64, lam_barrier::Float64,
                            dt::Float64, n_substeps::Int)
    n_steps, n_anchors = size(rbf_design)
    sub_dt = dt / n_substeps
    sqrt_dt = sqrt(dt)
    eps_B, eps_A = 1e-4, 1e-4

    B, F, A = init_state[1], init_state[2], init_state[3]
    A_acc = 0.0; Phi_acc = 0.0; barrier_acc = 0.0
    A_history = Float64[]

    for k in 1:n_steps
        # Decode Phi
        raw = c_Phi
        for a in 1:n_anchors
            raw += theta[a] * rbf_design[k, a]
        end
        Phi_t = Phi_max / (1 + exp(-raw))

        # PRE-step accumulation
        A_acc += A * dt
        Phi_acc += Phi_t * Phi_t * dt
        barrier_acc += max(F - F_max, 0.0)^2 * dt
        push!(A_history, A)

        # Substepped drift
        for sub in 1:n_substeps
            F_dev = F - F_TYP
            mu_bif = params.mu_0 + params.mu_B * B - params.mu_F * F - params.mu_FF * F_dev * F_dev
            a_factor_B = (1.0 + params.epsilon_A * A) / (1.0 + params.epsilon_A * A_TYP)
            a_factor_F = (1.0 + params.lambda_A  * A) / (1.0 + params.lambda_A  * A_TYP)
            drift_B = params.kappa_B * a_factor_B * Phi_t - B / params.tau_B
            drift_F = params.kappa_F * Phi_t - a_factor_F / params.tau_F * F
            drift_A = mu_bif * A - params.eta * A * A * A
            B += sub_dt * drift_B
            F += sub_dt * drift_F
            A += sub_dt * drift_A
        end

        # Diffusion at outer bin
        B_cl = max(eps_B, min(1 - eps_B, B))
        F_cl = max(0.0, F)
        A_cl = max(0.0, A)
        sigma_B = params.sigma_B * sqrt(B_cl * (1 - B_cl))
        sigma_F = params.sigma_F * sqrt(F_cl)
        sigma_A = params.sigma_A * sqrt(A_cl)

        nB = fixed_w[k, 1]
        nF = fixed_w[k, 2]
        nA = fixed_w[k, 3]
        B += sigma_B * sqrt_dt * nB
        F += sigma_F * sqrt_dt * nF
        A += sigma_A * sqrt_dt * nA

        # Reflection
        B = B < 0 ? -B : (B > 1 ? 2 - B : B)
        F = abs(F)
        A = abs(A)
    end

    cost = -A_acc + lam_phi * Phi_acc + lam_barrier * barrier_acc
    return cost, A_history
end


# ── Setup ────────────────────────────────────────────────────────────────

n_inner = 128
n_anchors = 8
n_steps = 14 * 96
T_total = 14.0
dt = 1/96
Phi_max = 3.0
Phi_default = 1.0
c_Phi = log((Phi_default/Phi_max) / (1 - Phi_default/Phi_max))
F_max_val = 0.40

# Build RBF design (CPU side, matches GPU kernel's row-norm Gaussian)
t_grid = collect(0:n_steps-1) .* dt
anchors = collect(range(0.0, T_total; length=n_anchors))
σ_rbf = T_total / n_anchors
rbf_design = Matrix{Float64}(undef, n_steps, n_anchors)
for k in 1:n_steps, j in 1:n_anchors
    d = t_grid[k] - anchors[j]
    rbf_design[k, j] = exp(-0.5 * (d / σ_rbf)^2)
end
for k in 1:n_steps
    s = sum(@view rbf_design[k, :])
    rbf_design[k, :] ./= s
end

# CRN noise (n_inner, n_steps, 3) — same seed as GPU target uses
rng = MersenneTwister(42)
fixed_w_full = randn(rng, n_inner, n_steps, 3)

init_state = [0.05, 0.30, 0.10]

# ── CPU evaluation across n_inner trials ────────────────────────────────

theta_uniform_1 = zeros(n_anchors)   # Φ=1 baseline

A_traj_means = Float64[]
costs = Float64[]
for t in 1:n_inner
    cost, A_hist = cpu_cost_one_trial(
        theta_uniform_1, init_state, TRUTH_PARAMS,
        fixed_w_full[t, :, :], rbf_design,
        c_Phi, Phi_max, F_max_val, 0.0, 1.0,
        dt, 4,
    )
    push!(costs, cost)
    push!(A_traj_means, mean(A_hist))
end

println("="^72)
println("CPU mirror, Φ ≡ 1.0 baseline, n_inner=$n_inner trials")
println("="^72)
@printf("  mean cost           = %+.6f\n", mean(costs))
@printf("  mean A across trials = %.4f\n", mean(A_traj_means))
@printf("  std A across trials  = %.4f\n", std(A_traj_means))

# ── Single deterministic trial (zero noise) — should match deterministic ODE ─
zero_noise = zeros(n_steps, 3)
cost_det, A_hist_det = cpu_cost_one_trial(
    theta_uniform_1, init_state, TRUTH_PARAMS,
    zero_noise, rbf_design,
    c_Phi, Phi_max, F_max_val, 0.0, 1.0,
    dt, 4,
)

println()
println("Deterministic (zero noise) trajectory at Φ ≡ 1.0:")
@printf("  A at day 0  = %.4f\n", A_hist_det[1])
@printf("  A at day 2  = %.4f\n", A_hist_det[Int(2*96)])
@printf("  A at day 5  = %.4f\n", A_hist_det[Int(5*96)])
@printf("  A at day 7  = %.4f\n", A_hist_det[Int(7*96)])
@printf("  A at day 10 = %.4f\n", A_hist_det[Int(10*96)])
@printf("  A at day 13 = %.4f\n", A_hist_det[Int(13*96)])
@printf("  mean A      = %.4f\n", mean(A_hist_det))
@printf("  cost        = %+.6f\n", cost_det)

println()
println("Now compare to GPU kernel at the same θ:")
target = FSAControlGPUTarget(
    n_inner=n_inner, M_max=4, n_steps=n_steps, n_anchors=n_anchors,
    n_substeps=4, dt=dt,
    F_max=F_max_val, Phi_max=Phi_max, Phi_default=Phi_default,
    lam_phi=0.0, lam_barrier=1.0,
    sigma_prior=1.5, params=TRUTH_PARAMS, init_state=init_state,
    noise_seed=42,
)
U_gpu = vcat(theta_uniform_1', theta_uniform_1', theta_uniform_1', theta_uniform_1')
ll_gpu = gpu_cost_log_density_batched(target, U_gpu)
@printf("  GPU kernel cost (Φ=1.0)      = %+.6f\n", -ll_gpu[1])
@printf("  GPU kernel mean A per trial  = %.4f\n", ll_gpu[1] / 14.0)

println()
@printf("Δ between CPU and GPU = %+.6f\n", mean(costs) - (-ll_gpu[1]))
