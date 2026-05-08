# GPU controller for FSA-v2 — MODEL-SPECIFIC parts only.
#
# This file contains ONLY the FSA-v2-specific cost kernel (the SDE rollout
# integrating -∫A dt + λ_F·∫max(F-F_max,0)² dt) and the target struct that
# holds per-replan state.
#
# fp32 throughout, with Kahan-compensated accumulators and FMA-everywhere
# drift. Designed to run at native consumer-Blackwell fp32 throughput
# (no fp64 penalty) while preserving fp64-equivalent precision on the
# two pieces that matter most:
#
#  - Cost accumulators (A_acc, barrier_acc): Kahan compensated summation
#    keeps a parallel `*_comp` register that tracks lost low bits, so
#    Σ(A·dt) over 1344 outer bins is precise to fp32 mantissa, not
#    fp32 mantissa minus log2(N_terms).
#
#  - Drift terms: fused multiply-adds throughout. fma(a, b, c) rounds
#    once at the end of a*b+c, vs two rounds in fp32. On the GPU this
#    also maps to a single hardware instruction so it's faster.
#
# Kept from §2.8 (commit b9e2280): Horner-form μ + precomputed
# a_typ_inv_*. Per-substep inv_tau_{B,F} hoisted out of the inner loop.
#
# Step 2 of the residual-fp32-bias localisation
# (claude_plans/Localising_residual_fp32_bias_in_the_v2_GPU_cost_kernel
# _2026-05-07_1956.md). Step 1 (just promoting the two cost accumulators
# to fp64, leaving everything else fp32) was empirically insufficient
# in the heavy closed-loop bench. This kernel substitutes Kahan +
# FMA for the fp64 promotion entirely.

module GPUControl

using CUDA
using KernelAbstractions
using LogExpFunctions: logsumexp
using Statistics: mean
using LinearAlgebra
using Random: AbstractRNG, MersenneTwister, randn

using ..Dynamics: A_TYP, F_TYP

export FSAControlGPUTarget, gpu_cost_log_density_batched
export make_log_density_fn


# ── Per-(chain, trial) cost kernel ───────────────────────────────────────

@kernel function fsa_cost_kernel!(
    cost_per_thread,
    theta_per_chain,
    rbf_design,
    init_state,
    p_tau_B, p_tau_F, p_kappa_B, p_kappa_F,
    p_epsilon_A, p_lambda_A,
    p_mu_0, p_mu_B, p_mu_F, p_mu_FF,
    p_eta, p_sigma_B, p_sigma_F, p_sigma_A,
    p_a_typ_inv_B, p_a_typ_inv_F,
    p_mu_const, p_mu_lin_F,
    fixed_w,
    c_Phi, Phi_max, F_max, lam_F,
    dt, n_substeps, n_steps, n_anchors, M, n_inner,
)
    i = @index(Global, Linear)
    if i <= M * n_inner
        m_chain = ((i - 1) ÷ n_inner) + 1
        t_trial = ((i - 1) % n_inner) + 1

        sub_dt  = dt / Float32(n_substeps)
        sqrt_dt = sqrt(dt)
        eps_B   = 1f-4

        # State as Float32
        B, F, A = init_state[1], init_state[2], init_state[3]

        # Compensated Accumulators (Kahan-like) for the integrals.
        # Tracks lost low bits in *_comp, giving fp32 the precision of
        # fp64 summation over the 1344 per-bin adds without paying the
        # consumer-Blackwell fp64 throughput penalty.
        A_acc, A_comp     = 0f0, 0f0
        bar_acc, bar_comp = 0f0, 0f0

        @inbounds for k in 1:n_steps
            # 1. RBF decode with FMA (single round per multiply-add)
            raw = c_Phi
            for a in 1:n_anchors
                raw = fma(theta_per_chain[m_chain, a], rbf_design[k, a], raw)
            end
            Phi_t = Phi_max / (1f0 + exp(-raw))

            # 2. Compensated integration: -∫A dt + λ_F·∫max(F-F_max,0)² dt
            # Term A_acc += A · dt
            y_A      = fma(A, dt, -A_comp)
            t_A      = A_acc + y_A
            A_comp   = (t_A - A_acc) - y_A
            A_acc    = t_A

            # Term barrier_acc += max(F-F_max,0)² · dt
            bdiff    = max(F - F_max, 0f0)
            bar_val  = bdiff * bdiff * dt
            y_B      = bar_val - bar_comp
            t_B      = bar_acc + y_B
            bar_comp = (t_B - bar_acc) - y_B
            bar_acc  = t_B

            # 3. Substepped EM. Pre-compute reciprocals once per outer bin.
            inv_tau_B = 1f0 / p_tau_B
            inv_tau_F = 1f0 / p_tau_F

            for sub in 1:n_substeps
                # Horner-form μ in F (still §2.8 algebraic-stable form):
                #   mu_bif = p_mu_const + p_mu_B*B + F*(p_mu_lin_F - p_mu_FF*F)
                mu_bif = fma(F, (p_mu_lin_F - p_mu_FF * F),
                             fma(p_mu_B, B, p_mu_const))

                a_factor_B = fma(p_epsilon_A, A, 1f0) * p_a_typ_inv_B
                a_factor_F = fma(p_lambda_A,  A, 1f0) * p_a_typ_inv_F

                drift_B = fma(p_kappa_B * a_factor_B, Phi_t, -B * inv_tau_B)
                drift_F = fma(p_kappa_F, Phi_t, -(a_factor_F * inv_tau_F * F))
                drift_A = fma(mu_bif, A, -(p_eta * A * A * A))

                B = fma(sub_dt, drift_B, B)
                F = fma(sub_dt, drift_F, F)
                A = fma(sub_dt, drift_A, A)
            end

            # 4. State-dep diffusion at outer-bin boundary
            B_cl        = max(eps_B, min(1f0 - eps_B, B))
            sigma_B_eff = p_sigma_B * sqrt(B_cl * (1f0 - B_cl))
            sigma_F_eff = p_sigma_F * sqrt(max(0f0, F))
            sigma_A_eff = p_sigma_A * sqrt(max(0f0, A))

            B = fma(sigma_B_eff * sqrt_dt, fixed_w[t_trial, k, 1], B)
            F = fma(sigma_F_eff * sqrt_dt, fixed_w[t_trial, k, 2], F)
            A = fma(sigma_A_eff * sqrt_dt, fixed_w[t_trial, k, 3], A)

            # 5. Boundary reflection
            B = B < 0f0 ? -B : (B > 1f0 ? 2f0 - B : B)
            F = abs(F)
            A = abs(A)
        end

        # Eq 37: J(Φ) = -∫A dt + λ_F · ∫max(F-F_max,0)² dt
        cost_per_thread[i] = Float64(fma(lam_F, bar_acc, -A_acc))
    end
end


# ── Target struct ────────────────────────────────────────────────────────

mutable struct FSAControlGPUTarget
    n_inner::Int
    M_max::Int
    n_steps::Int
    n_anchors::Int
    n_substeps::Int
    dt::Float32
    F_max::Float32
    Phi_max::Float32
    Phi_default::Float32
    c_Phi::Float32
    lam_F::Float32           # λ_F in Eq 37 — soft barrier weight on max(F-F_max, 0)²
    sigma_prior::Float64
    p_tau_B::Float32; p_tau_F::Float32
    p_kappa_B::Float32; p_kappa_F::Float32
    p_epsilon_A::Float32; p_lambda_A::Float32
    p_mu_0::Float32; p_mu_B::Float32; p_mu_F::Float32; p_mu_FF::Float32
    p_eta::Float32
    p_sigma_B::Float32; p_sigma_F::Float32; p_sigma_A::Float32
    # §2.8 precomputed algebraic-stable rearrangement constants
    p_a_typ_inv_B::Float32; p_a_typ_inv_F::Float32
    p_mu_const::Float32;    p_mu_lin_F::Float32
    rbf_design::CuArray{Float32,2}
    init_state::CuArray{Float32,1}
    fixed_w::CuArray{Float32,3}
    theta_per_chain::CuArray{Float32,2}
    cost_per_thread::CuArray{Float64,1}   # Step 1: fp64 cost output
    kernel::Any
end


function FSAControlGPUTarget(; n_inner::Int, M_max::Int,
                              n_steps::Int, n_anchors::Int = 8,
                              n_substeps::Int = 4,
                              dt::Real,
                              F_max::Real = 0.40,
                              Phi_max::Real = 3.0,
                              Phi_default::Real = 1.0,
                              lam_F::Real = 1.0,
                              sigma_prior::Real = 1.5,
                              params::NamedTuple,
                              init_state::AbstractVector,
                              noise_seed::Int = 42)
    p_ratio = Phi_default / Phi_max
    c_Phi   = log(p_ratio / (1.0 - p_ratio))

    # Build RBF design matrix — RAW Gaussian basis (NOT row-normalised),
    # matching Python's `smc2fc/control/rbf_schedules.py:design_matrix`.
    T_total = n_steps * dt
    t_grid  = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length=n_anchors))
    σ = T_total / n_anchors
    M_design = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for k in 1:n_steps, j in 1:n_anchors
        d = t_grid[k] - anchors[j]
        M_design[k, j] = exp(-0.5 * (d / σ)^2)
    end

    # Pre-generate CRN noise grid (n_inner, n_steps, 3) — fp32, reused.
    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, n_inner, n_steps, 3)

    # §2.8 precomputed algebraic-stable rearrangement constants.
    a_typ_inv_B = 1.0 / (1.0 + Float64(params.epsilon_A) * Float64(A_TYP))
    a_typ_inv_F = 1.0 / (1.0 + Float64(params.lambda_A)  * Float64(A_TYP))
    mu_const    = Float64(params.mu_0) - Float64(params.mu_FF) * Float64(F_TYP)^2
    mu_lin_F    = 2.0 * Float64(params.mu_FF) * Float64(F_TYP) - Float64(params.mu_F)

    return FSAControlGPUTarget(
        n_inner, M_max, n_steps, n_anchors, n_substeps,
        Float32(dt),
        Float32(F_max), Float32(Phi_max), Float32(Phi_default),
        Float32(c_Phi),
        Float32(lam_F),
        Float64(sigma_prior),
        Float32(params.tau_B), Float32(params.tau_F),
        Float32(params.kappa_B), Float32(params.kappa_F),
        Float32(params.epsilon_A), Float32(params.lambda_A),
        Float32(params.mu_0), Float32(params.mu_B), Float32(params.mu_F),
        Float32(params.mu_FF), Float32(params.eta),
        Float32(params.sigma_B), Float32(params.sigma_F), Float32(params.sigma_A),
        Float32(a_typ_inv_B), Float32(a_typ_inv_F),
        Float32(mu_const),    Float32(mu_lin_F),
        CuArray(Float32.(M_design)),
        CuArray(Float32.(init_state)),
        CuArray(noise_cpu),
        CUDA.zeros(Float32, M_max, n_anchors),
        CUDA.zeros(Float64, M_max * n_inner),     # fp64 cost
        fsa_cost_kernel!(CUDABackend(), 256),
    )
end


"""
    gpu_cost_log_density_batched(target, theta_unc::AbstractMatrix{Float64})
        -> Vector{Float64}

Evaluate the controller's log-density log p(θ_ctrl) ∝ exp(-cost(θ_ctrl))
at M chains in one kernel launch. Returns -cost per chain (so the outer
SMC² treats higher = better).
"""
function gpu_cost_log_density_batched(target::FSAControlGPUTarget,
                                       theta_unc::AbstractMatrix{Float64})
    M = size(theta_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    @assert size(theta_unc, 2) == target.n_anchors

    theta_cpu_f32 = Float32.(theta_unc)
    copyto!(view(target.theta_per_chain, 1:M, :), theta_cpu_f32)

    Ntot = M * target.n_inner
    target.kernel(
        view(target.cost_per_thread, 1:Ntot),
        view(target.theta_per_chain, 1:M, :),
        target.rbf_design,
        target.init_state,
        target.p_tau_B, target.p_tau_F,
        target.p_kappa_B, target.p_kappa_F,
        target.p_epsilon_A, target.p_lambda_A,
        target.p_mu_0, target.p_mu_B, target.p_mu_F, target.p_mu_FF,
        target.p_eta,
        target.p_sigma_B, target.p_sigma_F, target.p_sigma_A,
        target.p_a_typ_inv_B, target.p_a_typ_inv_F,
        target.p_mu_const,    target.p_mu_lin_F,
        target.fixed_w,
        target.c_Phi, target.Phi_max,
        target.F_max, target.lam_F,
        target.dt, target.n_substeps,
        target.n_steps, target.n_anchors,
        M, target.n_inner;
        ndrange = Ntot,
    )
    KernelAbstractions.synchronize(CUDABackend())

    # Per-chain mean cost: sum over trials / n_inner. Pull to CPU.
    cost_cpu = Array(view(target.cost_per_thread, 1:Ntot))
    cost_mat = reshape(cost_cpu, target.n_inner, M)
    out = Vector{Float64}(undef, M)
    @inbounds for m in 1:M
        s = 0.0
        for t in 1:target.n_inner
            s += cost_mat[t, m]
        end
        out[m] = -s / target.n_inner
    end
    return out
end


"""
    make_log_density_fn(target::FSAControlGPUTarget) -> Function

Build a closure `f :: Matrix{Float64} → Vector{Float64}` that maps an
(M, n_anchors) θ matrix to per-chain log-density (= -cost). This is the
contract the generic framework controller (`SMC2FC.run_tempered_smc_gpu`)
expects from any model.
"""
function make_log_density_fn(target::FSAControlGPUTarget)
    return function (U::AbstractMatrix{Float64})
        return gpu_cost_log_density_batched(target, U)
    end
end


end # module GPUControl
