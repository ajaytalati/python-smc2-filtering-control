#!/usr/bin/env julia
# Sanity check: evaluate the GPU cost at three known Φ schedules.
# If recovery-overload < uniform-0.30 < uniform-1.0 (in cost) the cost surface
# is correct; HMC would just need to find recovery-overload. Otherwise the
# kernel has a bug.

ENV["FSA_STEP_MINUTES"] = "15"

using CUDA
using Statistics
using Printf
using Random

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUControl: FSAControlGPUTarget, gpu_cost_log_density_batched
using .FSAHighRes.Dynamics: TRUTH_PARAMS

n_inner = 128
n_anchors = 8
n_steps = 14 * 96         # T=14d at h=15min

# Build target with the canonical Banister default Φ=1.0 baseline.
target = FSAControlGPUTarget(
    n_inner=n_inner, M_max=4, n_steps=n_steps, n_anchors=n_anchors,
    n_substeps=4, dt=1/96,
    F_max=0.40, Phi_max=3.0, Phi_default=1.0,
    lam_phi=0.0, lam_barrier=1.0,
    sigma_prior=1.5, params=TRUTH_PARAMS,
    init_state=[0.05, 0.30, 0.10],
)

# Decoder formula:  Φ(t) = Phi_max · σ(c_Φ + Σ_a θ_a · design[t,a])
# Anchors are evenly spaced on [0, T_total]; design row-normalised.
# To produce a target Φ_target uniform across t, we need
#     σ(c_Φ + θ̄ · design row sum) = Φ_target / Phi_max
#     c_Φ + θ̄ = logit(Φ_target / Phi_max)
# With c_Φ = log(1/2) = -0.693 (since Φ_default/Phi_max = 1/3),
# uniform θ_a = (logit(Φ_target / 3.0) − c_Φ) for all a.

c_Phi = log((1.0/3.0) / (1.0 - 1.0/3.0))
function theta_uniform(Phi_target)
    θ̄ = log((Phi_target/3.0) / (1.0 - Phi_target/3.0)) - c_Phi
    return fill(θ̄, n_anchors)
end

# Recovery-overload schedule: anchors at days 0, 2, 4, 6, 8, 10, 12, 14.
# Want Φ ≈ 0.2 for days 1-5 (anchors 2, 4), Φ ≈ 0.7 for days 6-9 (anchors 6, 8),
# Φ ≈ 1.2 for days 10-14 (anchors 10, 12, 14).
function theta_for_phi_per_anchor(phi_per_anchor)
    return [log((p/3.0) / (1.0 - p/3.0)) - c_Phi for p in phi_per_anchor]
end

θ_baseline    = theta_uniform(1.0)             # Φ=1.0 everywhere
θ_uniform_03  = theta_uniform(0.3)             # Φ=0.3 everywhere
θ_uniform_045 = theta_uniform(0.45)            # Φ=0.45 — what my HMC found
θ_recovery    = theta_for_phi_per_anchor([1.0, 0.2, 0.2, 0.4, 0.7, 0.9, 1.1, 1.2])

U = vcat(θ_baseline', θ_uniform_03', θ_uniform_045', θ_recovery')
@info "Evaluating cost at 4 schedules..."
ll = gpu_cost_log_density_batched(target, Float64.(U))

println()
println("="^72)
println("Cost = -log_density = ∫(-A + λ_F·max(F-F_max, 0)²) dt    (lower is better)")
println("="^72)
labels = ["Φ ≡ 1.0 (baseline)", "Φ ≡ 0.30", "Φ ≡ 0.45 (my HMC fixed point)", "recovery→overload"]
for (lbl, l) in zip(labels, ll)
    @printf("  %-32s   cost = %+.6f   (mean A per trial ≈ %.4f)\n",
            lbl, -l, l / 14.0)
end
println()
best_idx = argmin(.-ll)
@printf("Lowest-cost schedule: %s\n", labels[best_idx])
println()
println("Sanity:")
println("  - if my kernel is correct, recovery→overload should be cheapest")
println("    (it's what Python's HMC finds and the doc reports +17% gain)")
println("  - if Φ=0.45 is cheapest, the kernel cost surface differs from the")
println("    spec — bug somewhere in the SDE rollout or the integration.")
