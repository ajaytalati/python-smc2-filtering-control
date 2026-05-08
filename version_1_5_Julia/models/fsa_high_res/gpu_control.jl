# GPU controller for FSA-v1 (Banister-coupled, NO G1 reparametrisation).
#
# Direct port of v2's `version_2_Julia/models/fsa_high_res/gpu_control.jl`
# with the v1 drift formulas:
#
#   μ(B,F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F²
#   dB/dt  = κ_B·(1 + ε_A·A)·Φ           − B / τ_B
#   dF/dt  = κ_F·Φ − (1 + λ_A·A) / τ_F · F
#   dA/dt  = μ·A − η·A³
#
# (v2 had aB(A) = (1+εA·A)/(1+εA·A_typ) and μ written around F_typ — none
# of that here.)
#
# Cost is unchanged from v2 / Eq.37 of the spec:
#   J(Φ) = E[ −∫A·dt + λ_F·∫max(F − F_max, 0)²·dt ]
#
# fp32 inner loop, fp64 outer (matches the filter convention).

module GPUControl

using CUDA
using KernelAbstractions
using LogExpFunctions: logsumexp
using Statistics: mean
using LinearAlgebra
using Random: AbstractRNG, MersenneTwister, randn

using ..Dynamics: TRUTH_PARAMS

export FSAv1ControlGPUTarget, gpu_cost_log_density_batched
export make_log_density_fn, build_rbf_design


# ── RBF design matrix helper (raw Gaussian, NOT row-normalised) ─────────
# Mirrors `version_2_Julia/models/fsa_high_res/cpu_control.jl:build_rbf`
# and `version_1_Julia/models/fsa_high_res/control.jl:build_rbf`. Inlined
# here to keep gpu_control.jl self-contained (no cross-module reach).
function build_rbf_design(n_steps::Integer, dt::Real, n_anchors::Integer)
    T_total = n_steps * dt
    t_grid  = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length=n_anchors))
    σ = T_total / n_anchors
    M = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for k in 1:n_steps, j in 1:n_anchors
        d = t_grid[k] - anchors[j]
        M[k, j] = exp(-0.5 * (d / σ)^2)
    end
    return M
end


# ── Per-(chain, trial) cost kernel — v1 drift formulas ──────────────────

@kernel function fsa_v1_cost_kernel!(
    cost_per_thread,        # (M·n_inner,) Float32 — output cost per (chain, trial)
    theta_per_chain,        # (M, n_anchors) Float32
    rbf_design,             # (n_steps, n_anchors) Float32
    init_state,             # (3,) Float32 — shared init (B, F, A)
    p_tau_B::Float32, p_tau_F::Float32,
    p_kappa_B::Float32, p_kappa_F::Float32,
    p_epsilon_A::Float32, p_lambda_A::Float32,
    p_mu_0::Float32, p_mu_B::Float32, p_mu_F::Float32, p_mu_FF::Float32,
    p_eta::Float32,
    p_sigma_B::Float32, p_sigma_F::Float32, p_sigma_A::Float32,
    fixed_w,                # (n_inner, n_steps, 3) Float32 — CRN noise
    c_Phi::Float32, Phi_max::Float32,
    F_max::Float32, lam_F::Float32,
    dt::Float32, n_substeps::Int,
    n_steps::Int, n_anchors::Int,
    M::Int, n_inner::Int,
)
    i = @index(Global, Linear)
    if i <= M * n_inner
        m_chain = ((i - 1) ÷ n_inner) + 1
        t_trial = ((i - 1) % n_inner) + 1

        sub_dt = dt / Float32(n_substeps)
        sqrt_dt = sqrt(dt)

        eps_B = 1f-4

        B = init_state[1]
        F = init_state[2]
        A = init_state[3]

        # Two-term Eq.37 cost:
        #   J(Φ) = E[ −∫A·dt + λ_F · ∫max(F − F_max, 0)² · dt ]
        A_acc = 0f0
        barrier_acc = 0f0

        @inbounds for k in 1:n_steps
            # ── Decode Φ(t) via raw RBF basis (per-bin) ────────────────
            raw = c_Phi
            for a in 1:n_anchors
                raw += theta_per_chain[m_chain, a] * rbf_design[k, a]
            end
            Phi_t = Phi_max / (1f0 + exp(-raw))

            # ── PRE-step accumulation (matches Python control.py:cost_fn:
            # A_acc uses the state BEFORE em_step) ─────────────────────
            A_acc       = A_acc + A * dt
            barrier_acc = barrier_acc + max(F - F_max, 0f0)^2 * dt

            # ── Substepped EM (drift n_substeps times, then ONE Wiener) ──
            # v1 drift — direct multiplicative coupling, NO operating-point
            # normalisation, μ uses F² (not (F-F_typ)²).
            for sub in 1:n_substeps
                mu_bif = p_mu_0 + p_mu_B * B - p_mu_F * F - p_mu_FF * F * F
                a_factor_B = 1f0 + p_epsilon_A * A
                a_factor_F = 1f0 + p_lambda_A  * A
                drift_B = p_kappa_B * a_factor_B * Phi_t - B / p_tau_B
                drift_F = p_kappa_F * Phi_t - a_factor_F / p_tau_F * F
                drift_A = mu_bif * A - p_eta * A * A * A
                B = B + sub_dt * drift_B
                F = F + sub_dt * drift_F
                A = A + sub_dt * drift_A
            end

            # State-dep diffusion at outer-bin boundary. Matches Python's
            # `diffusion_state_dep` (sqrt-Itô, vanishes at each boundary).
            B_cl = max(eps_B, min(1f0 - eps_B, B))
            F_cl = max(0f0, F)
            A_cl = max(0f0, A)
            sigma_B_eff = p_sigma_B * sqrt(B_cl * (1f0 - B_cl))
            sigma_F_eff = p_sigma_F * sqrt(F_cl)
            sigma_A_eff = p_sigma_A * sqrt(A_cl)

            nB = fixed_w[t_trial, k, 1]
            nF = fixed_w[t_trial, k, 2]
            nA = fixed_w[t_trial, k, 3]
            B = B + sigma_B_eff * sqrt_dt * nB
            F = F + sigma_F_eff * sqrt_dt * nF
            A = A + sigma_A_eff * sqrt_dt * nA

            # Boundary reflection (matches em_step_substepped exactly)
            B = B < 0f0 ? -B : (B > 1f0 ? 2f0 - B : B)
            F = abs(F)
            A = abs(A)
        end

        cost_per_thread[i] = -A_acc + lam_F * barrier_acc
    end
end


# ── Target struct ────────────────────────────────────────────────────────

mutable struct FSAv1ControlGPUTarget
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
    lam_F::Float32
    sigma_prior::Float64
    p_tau_B::Float32; p_tau_F::Float32
    p_kappa_B::Float32; p_kappa_F::Float32
    p_epsilon_A::Float32; p_lambda_A::Float32
    p_mu_0::Float32; p_mu_B::Float32; p_mu_F::Float32; p_mu_FF::Float32
    p_eta::Float32
    p_sigma_B::Float32; p_sigma_F::Float32; p_sigma_A::Float32
    rbf_design::CuArray{Float32,2}
    init_state::CuArray{Float32,1}
    fixed_w::CuArray{Float32,3}
    theta_per_chain::CuArray{Float32,2}
    cost_per_thread::CuArray{Float32,1}
    kernel::Any
end


function FSAv1ControlGPUTarget(; n_inner::Int, M_max::Int,
                                 n_steps::Int, n_anchors::Int = 8,
                                 n_substeps::Int = 4,
                                 dt::Real,
                                 F_max::Real = 0.40,
                                 Phi_max::Real = 3.0,
                                 Phi_default::Real = 1.0,
                                 lam_F::Real = 1.0,
                                 sigma_prior::Real = 1.5,
                                 params::NamedTuple = TRUTH_PARAMS,
                                 init_state::AbstractVector = [0.05, 0.30, 0.10],
                                 noise_seed::Int = 42)
    p_ratio = Phi_default / Phi_max
    c_Phi   = log(p_ratio / (1.0 - p_ratio))

    M_design = build_rbf_design(n_steps, dt, n_anchors)

    # CRN noise grid (n_inner, n_steps, 3) — reused across all calls.
    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, n_inner, n_steps, 3)

    return FSAv1ControlGPUTarget(
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
        CuArray(Float32.(M_design)),
        CuArray(Float32.(init_state)),
        CuArray(noise_cpu),
        CUDA.zeros(Float32, M_max, n_anchors),
        CUDA.zeros(Float32, M_max * n_inner),
        fsa_v1_cost_kernel!(CUDABackend(), 256),
    )
end


"""
    gpu_cost_log_density_batched(target, theta_unc::AbstractMatrix{Float64})
        -> Vector{Float64}

Evaluate the controller's log-density log p(θ) ∝ exp(−cost(θ)) at M chains
in one kernel launch. Returns −cost per chain (so the outer SMC² treats
higher = better).
"""
function gpu_cost_log_density_batched(target::FSAv1ControlGPUTarget,
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
        target.fixed_w,
        target.c_Phi, target.Phi_max,
        target.F_max, target.lam_F,
        target.dt, target.n_substeps,
        target.n_steps, target.n_anchors,
        M, target.n_inner;
        ndrange = Ntot,
    )
    KernelAbstractions.synchronize(CUDABackend())

    cost_cpu = Array(view(target.cost_per_thread, 1:Ntot))
    cost_mat = reshape(cost_cpu, target.n_inner, M)
    out = Vector{Float64}(undef, M)
    @inbounds for m in 1:M
        s = 0.0
        for t in 1:target.n_inner
            s += Float64(cost_mat[t, m])
        end
        out[m] = -s / target.n_inner
    end
    return out
end


"""
    make_log_density_fn(target::FSAv1ControlGPUTarget) -> Function

Build a closure `f :: Matrix{Float64} → Vector{Float64}` mapping (M, n_anchors)
θ matrix to per-chain log-density (= −cost). This is the contract the generic
framework controller (`SMC2FC.run_tempered_smc_gpu`) expects from any model.
"""
function make_log_density_fn(target::FSAv1ControlGPUTarget)
    return function (U::AbstractMatrix{Float64})
        return gpu_cost_log_density_batched(target, U)
    end
end


end # module GPUControl
