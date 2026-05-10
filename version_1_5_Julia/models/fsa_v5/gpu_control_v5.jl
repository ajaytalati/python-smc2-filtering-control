# GPU controller for FSA v5 — bimodal Φ, 6D state, SOFT cost only.
#
# Mirrors v1.5's `gpu_control.jl` plumbing verbatim (per-thread cost
# kernel, fp32 inner loop, fp64 outer aggregate, CRN noise grid). What
# differs from v1.5:
#   - 6D state [B, S, F, A, K_FB, K_FS] (was 3D)
#   - Bimodal stimulus (Φ_B, Φ_S) decoded from a 2 · n_anchors RBF
#     coefficient vector (was scalar Φ from n_anchors)
#   - v5 drift/diffusion math (Hill deconditioning, Stuart-Landau on
#     B+S+F, Busso K dynamics) replacing v1.5's v1-form
#   - Boundary handling: clamp B,S; floor F,A,K (per tech guide §6.4)
#   - Cost is the SOFT relaxation of the chance-constrained form (tech
#     guide §5.3 + §5.5):
#       J_soft(θ) = λ_Φ · ∫ (Φ_B² + Φ_S²) dt        // effort
#                 - ∫ A dt - ∫ B dt - ∫ S dt         // rewards
#                 + λ_F · ∫ max(F − F_max, 0)² dt   // soft fatigue penalty - not used
#                 + λ_chance · ∫ σ(β·(A_thr − A)/scale) dt  // soft chance surrogate
#     The soft chance surrogate uses a CONSTANT A_thr (default 0.05)
#     rather than the per-bin A_sep(Φ_t) from `find_a_sep` (implemented
#     in `cost_v5.jl`). The latter requires a 64-grid + 40-bisection
#     root-find per (chain, bin) and is deferred — wiring it up needs a
#     CPU-side precompute step or a separate GPU prep kernel. The
#     constant-threshold form gives the same QUALITATIVE behaviour
#     (penalise trajectories that approach low A) and is sufficient for
#     closed-loop optimisation provided A_thr is set well below the
#     healthy attractor.
#
# HARD chance-constraint variant (with the indicator and per-bin A_sep)
# is deliberately not implemented — per project decision the SOFT
# variant is the production target.
#
# fp32 inner loop, fp64 outer (matches v1.5).

module GPUControlV5

using CUDA
using KernelAbstractions
using Random: AbstractRNG, MersenneTwister, randn

import ..SimulationV5: TRUTH_PARAMS_V5, FROZEN_PARAMS_V5,
                        TRAINED_ATHLETE_INIT, A_TYP, F_TYP

export FSAv5ControlGPUTarget, gpu_cost_log_density_batched_v5
export make_log_density_fn_v5, build_rbf_design_v5
export gpu_cost_one!     # single-thread debug entry for the diff test


# ── RBF design matrix helper ──────────────────────────────────────────
# Identical to v1.5's `build_rbf_design` and to `ScheduleV5.design_matrix`
# (the latter is the diff-tested CPU primitive); duplicated here so this
# module is self-contained and can be loaded without depending on the
# CPU schedule module's structure conventions. Bit-identical output.

function build_rbf_design_v5(n_steps::Integer, dt::Real, n_anchors::Integer)
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


# ── Per-(chain, trial) cost kernel — v5 math, bimodal Phi ─────────────

@kernel function fsa_v5_cost_kernel!(
    cost_per_thread,         # (M·n_inner,) Float32
    theta_per_chain,         # (M, 2·n_anchors) Float32
    rbf_design,              # (n_steps, n_anchors) Float32
    init_state,              # (6,) Float32
    # Estimated dynamics (15 — production controller uses TRUTH_PARAMS_V5
    # since the cost is evaluated under the filter's posterior-mean params)
    p_tau_B::Float32, p_kappa_B::Float32, p_epsilon_AB::Float32,
    p_tau_S::Float32, p_kappa_S::Float32, p_epsilon_AS::Float32,
    p_tau_F::Float32, p_lambda_A::Float32,
    p_mu_K::Float32,
    p_mu_0::Float32, p_mu_B::Float32, p_mu_S::Float32, p_mu_F::Float32,
    p_mu_FF::Float32, p_eta::Float32,
    # Frozen dynamics (8 + diffusion 5 = 13)
    p_KFB_0::Float32, p_KFS_0::Float32, p_tau_K::Float32,
    p_B_dec::Float32, p_S_dec::Float32,
    p_mu_dec_B::Float32, p_mu_dec_S::Float32,
    p_sigma_B::Float32, p_sigma_S::Float32, p_sigma_F::Float32,
    p_sigma_A::Float32, p_sigma_K::Float32,
    A_TYP_f32::Float32, F_TYP_f32::Float32,
    # Pre-drawn CRN noise (n_inner, n_steps, 6)
    fixed_w,
    # Schedule decoder bias + envelope
    c_Phi::Float32, Phi_max::Float32,
    # Cost-shaping coefficients
    lam_Phi::Float32, F_max::Float32, lam_F::Float32,
    A_thr::Float32, lam_chance::Float32,
    beta_chance::Float32, scale_chance::Float32,
    # Time-step
    dt::Float32, n_substeps::Int,
    n_steps::Int, n_anchors::Int,
    M::Int, n_inner::Int,
)
    i = @index(Global, Linear)
    if i <= M * n_inner
        m_chain = ((i - 1) ÷ n_inner) + 1
        t_trial = ((i - 1) % n_inner) + 1

        sub_dt  = dt / Float32(n_substeps)
        sqrt_dt = sqrt(dt)
        eps_B   = 1f-4
        eps_S   = 1f-4

        # Initial state.
        B   = init_state[1]
        S   = init_state[2]
        F   = init_state[3]
        A   = init_state[4]
        KFB = init_state[5]
        KFS = init_state[6]

        # Cost accumulators (fp32 throughout — fp64 sum is done CPU-side).
        effort_acc  = 0f0
        A_acc       = 0f0
        B_acc       = 0f0
        S_acc       = 0f0
        barrier_acc = 0f0
        chance_acc  = 0f0

        @inbounds for k in 1:n_steps
            # ── Decode Φ(t) from RBF coefs (theta has 2·n_anchors entries) ─
            raw_B = c_Phi
            raw_S = c_Phi
            for a in 1:n_anchors
                raw_B += theta_per_chain[m_chain, a]              * rbf_design[k, a]
                raw_S += theta_per_chain[m_chain, n_anchors + a]  * rbf_design[k, a]
            end
            Phi_B = Phi_max / (1f0 + exp(-raw_B))
            Phi_S = Phi_max / (1f0 + exp(-raw_S))

            # ── PRE-step accumulation (matches v1.5's cost convention:
            #    accumulators use the state BEFORE em_step) ───────────────
            effort_acc  += (Phi_B * Phi_B + Phi_S * Phi_S) * dt
            A_acc       += A * dt
            B_acc       += B * dt
            S_acc       += S * dt
            barrier_acc += max(F - F_max, 0f0) * max(F - F_max, 0f0) * dt
            soft_v       = 1f0 / (1f0 + exp(-beta_chance * (A_thr - A) / scale_chance))
            chance_acc  += soft_v * dt

            # ── Substepped EM (v5 drift, n_substeps drift sub-steps then ONE Wiener) ─
            for sub in 1:n_substeps
                # μ̄(B, S, F) — v5 form with Hill (n_dec hardcoded to 4)
                F_dev = F - F_TYP_f32
                Bn = max(B, 0f0); Bn = Bn * Bn * Bn * Bn
                Sn = max(S, 0f0); Sn = Sn * Sn * Sn * Sn
                Bdn = p_B_dec * p_B_dec * p_B_dec * p_B_dec
                Sdn = p_S_dec * p_S_dec * p_S_dec * p_S_dec
                dec_B = p_mu_dec_B * Bdn / (Bn + Bdn)
                dec_S = p_mu_dec_S * Sdn / (Sn + Sdn)
                mu_bif = p_mu_0 +
                          p_mu_B * B + p_mu_S * S -
                          p_mu_F * F - p_mu_FF * F_dev * F_dev -
                          dec_B - dec_S

                a_factor_B = (1f0 + p_epsilon_AB * A) / (1f0 + p_epsilon_AB * A_TYP_f32)
                a_factor_S = (1f0 + p_epsilon_AS * A) / (1f0 + p_epsilon_AS * A_TYP_f32)
                a_factor_F = (1f0 + p_lambda_A  * A) / (1f0 + p_lambda_A  * A_TYP_f32)
                drift_B   = p_kappa_B * a_factor_B * Phi_B - B / p_tau_B
                drift_S   = p_kappa_S * a_factor_S * Phi_S - S / p_tau_S
                drift_F   = KFB * Phi_B + KFS * Phi_S - a_factor_F / p_tau_F * F
                drift_A   = mu_bif * A - p_eta * A * A * A
                drift_KFB = (p_KFB_0 - KFB) / p_tau_K + p_mu_K * Phi_B
                drift_KFS = (p_KFS_0 - KFS) / p_tau_K + p_mu_K * Phi_S

                B   = B   + sub_dt * drift_B
                S   = S   + sub_dt * drift_S
                F   = F   + sub_dt * drift_F
                A   = A   + sub_dt * drift_A
                KFB = KFB + sub_dt * drift_KFB
                KFS = KFS + sub_dt * drift_KFS
            end

            # State-dep diffusion at outer-bin boundary.
            B_cl   = max(eps_B, min(1f0 - eps_B, B))
            S_cl   = max(eps_S, min(1f0 - eps_S, S))
            F_cl   = max(0f0, F)
            A_cl   = max(0f0, A)
            KFB_cl = max(0f0, KFB)
            KFS_cl = max(0f0, KFS)
            sigma_B_eff   = p_sigma_B * sqrt(B_cl   * (1f0 - B_cl))
            sigma_S_eff   = p_sigma_S * sqrt(S_cl   * (1f0 - S_cl))
            sigma_F_eff   = p_sigma_F * sqrt(F_cl)
            sigma_A_eff   = p_sigma_A * sqrt(A_cl)
            sigma_KFB_eff = p_sigma_K * sqrt(KFB_cl)
            sigma_KFS_eff = p_sigma_K * sqrt(KFS_cl)

            nB  = fixed_w[t_trial, k, 1]
            nS  = fixed_w[t_trial, k, 2]
            nF  = fixed_w[t_trial, k, 3]
            nA  = fixed_w[t_trial, k, 4]
            nKB = fixed_w[t_trial, k, 5]
            nKS = fixed_w[t_trial, k, 6]
            B   = B   + sigma_B_eff   * sqrt_dt * nB
            S   = S   + sigma_S_eff   * sqrt_dt * nS
            F   = F   + sigma_F_eff   * sqrt_dt * nF
            A   = A   + sigma_A_eff   * sqrt_dt * nA
            KFB = KFB + sigma_KFB_eff * sqrt_dt * nKB
            KFS = KFS + sigma_KFS_eff * sqrt_dt * nKS

            # Boundary handling — clamp/floor (matches em_step_v5 / tech guide §6.4).
            B = max(eps_B, min(1f0 - eps_B, B))
            S = max(eps_S, min(1f0 - eps_S, S))
            F = max(0f0, F)
            A = max(0f0, A)
            KFB = max(0f0, KFB)
            KFS = max(0f0, KFS)
        end

        cost_per_thread[i] = lam_Phi * effort_acc - A_acc - B_acc - S_acc +
                              lam_F * barrier_acc +
                              lam_chance * chance_acc
    end
end


# ── Target struct ─────────────────────────────────────────────────────

mutable struct FSAv5ControlGPUTarget
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
    lam_Phi::Float32
    lam_F::Float32
    lam_chance::Float32
    A_thr::Float32
    beta_chance::Float32
    scale_chance::Float32
    sigma_prior::Float64
    # Estimated dynamics (15)
    p_tau_B::Float32; p_kappa_B::Float32; p_epsilon_AB::Float32
    p_tau_S::Float32; p_kappa_S::Float32; p_epsilon_AS::Float32
    p_tau_F::Float32; p_lambda_A::Float32
    p_mu_K::Float32
    p_mu_0::Float32; p_mu_B::Float32; p_mu_S::Float32; p_mu_F::Float32
    p_mu_FF::Float32; p_eta::Float32
    # Frozen dynamics + diffusion (8 + 5 = 13)
    p_KFB_0::Float32; p_KFS_0::Float32; p_tau_K::Float32
    p_B_dec::Float32; p_S_dec::Float32
    p_mu_dec_B::Float32; p_mu_dec_S::Float32
    p_sigma_B::Float32; p_sigma_S::Float32; p_sigma_F::Float32
    p_sigma_A::Float32; p_sigma_K::Float32
    A_TYP_f32::Float32; F_TYP_f32::Float32
    rbf_design::CuArray{Float32, 2}
    init_state::CuArray{Float32, 1}
    fixed_w::CuArray{Float32, 3}                 # (n_inner, n_steps, 6)
    theta_per_chain::CuArray{Float32, 2}         # (M_max, 2·n_anchors)
    cost_per_thread::CuArray{Float32, 1}
    kernel::Any
end


function FSAv5ControlGPUTarget(; n_inner::Int, M_max::Int,
                                  n_steps::Int, n_anchors::Int = 8,
                                  n_substeps::Int = 4,
                                  dt::Real,
                                  F_max::Real = 0.40,
                                  Phi_max::Real = 3.0,
                                  Phi_default::Real = 1.0,
                                  lam_Phi::Real = 0.0,
                                  lam_F::Real = 1.0,
                                  lam_chance::Real = 0.0,
                                  A_thr::Real = 0.05,
                                  beta_chance::Real = 50.0,
                                  scale_chance::Real = 0.10,
                                  sigma_prior::Real = 1.5,
                                  params::AbstractDict = TRUTH_PARAMS_V5,
                                  init_state::AbstractVector = collect(values((
                                      TRAINED_ATHLETE_INIT.B, TRAINED_ATHLETE_INIT.S,
                                      TRAINED_ATHLETE_INIT.F, TRAINED_ATHLETE_INIT.A,
                                      TRAINED_ATHLETE_INIT.KFB, TRAINED_ATHLETE_INIT.KFS,
                                  ))),
                                  noise_seed::Int = 42)
    p_ratio = Phi_default / Phi_max
    c_Phi   = log(p_ratio / (1.0 - p_ratio))

    M_design = build_rbf_design_v5(n_steps, dt, n_anchors)

    # CRN noise grid (n_inner, n_steps, 6) — reused across all calls.
    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, n_inner, n_steps, 6)

    return FSAv5ControlGPUTarget(
        n_inner, M_max, n_steps, n_anchors, n_substeps,
        Float32(dt),
        Float32(F_max), Float32(Phi_max), Float32(Phi_default),
        Float32(c_Phi),
        Float32(lam_Phi), Float32(lam_F),
        Float32(lam_chance),
        Float32(A_thr), Float32(beta_chance), Float32(scale_chance),
        Float64(sigma_prior),
        # Estimated dynamics
        Float32(params[:tau_B]), Float32(params[:kappa_B]),
        Float32(params[:epsilon_AB]),
        Float32(params[:tau_S]), Float32(params[:kappa_S]),
        Float32(params[:epsilon_AS]),
        Float32(params[:tau_F]), Float32(params[:lambda_A]),
        Float32(params[:mu_K]),
        Float32(params[:mu_0]), Float32(params[:mu_B]),
        Float32(params[:mu_S]), Float32(params[:mu_F]),
        Float32(params[:mu_FF]), Float32(params[:eta]),
        # Frozen dynamics + diffusion
        Float32(params[:KFB_0]), Float32(params[:KFS_0]),
        Float32(params[:tau_K]),
        Float32(params[:B_dec]), Float32(params[:S_dec]),
        Float32(params[:mu_dec_B]), Float32(params[:mu_dec_S]),
        Float32(params[:sigma_B]), Float32(params[:sigma_S]),
        Float32(params[:sigma_F]), Float32(params[:sigma_A]),
        Float32(params[:sigma_K]),
        Float32(A_TYP), Float32(F_TYP),
        CuArray(Float32.(M_design)),
        CuArray(Float32.(init_state)),
        CuArray(noise_cpu),
        CUDA.zeros(Float32, M_max, 2 * n_anchors),
        CUDA.zeros(Float32, M_max * n_inner),
        fsa_v5_cost_kernel!(CUDABackend(), 256),
    )
end


"""
    gpu_cost_log_density_batched_v5(target, theta_unc::AbstractMatrix{Float64})
        -> Vector{Float64}

Evaluate the controller's log-density `log p(θ) ∝ exp(−cost(θ))` at M
chains in one kernel launch. `theta_unc` is `(M, 2·n_anchors)` (the
RBF coefficient vector per chain). Returns `−mean(cost)` per chain so
the outer SMC² treats higher = better.
"""
function gpu_cost_log_density_batched_v5(target::FSAv5ControlGPUTarget,
                                            theta_unc::AbstractMatrix{Float64})
    M = size(theta_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    @assert size(theta_unc, 2) == 2 * target.n_anchors

    theta_cpu_f32 = Float32.(theta_unc)
    copyto!(view(target.theta_per_chain, 1:M, :), theta_cpu_f32)

    Ntot = M * target.n_inner
    target.kernel(
        view(target.cost_per_thread, 1:Ntot),
        view(target.theta_per_chain, 1:M, :),
        target.rbf_design,
        target.init_state,
        # Estimated dynamics
        target.p_tau_B, target.p_kappa_B, target.p_epsilon_AB,
        target.p_tau_S, target.p_kappa_S, target.p_epsilon_AS,
        target.p_tau_F, target.p_lambda_A,
        target.p_mu_K,
        target.p_mu_0, target.p_mu_B, target.p_mu_S, target.p_mu_F,
        target.p_mu_FF, target.p_eta,
        # Frozen dynamics + diffusion
        target.p_KFB_0, target.p_KFS_0, target.p_tau_K,
        target.p_B_dec, target.p_S_dec,
        target.p_mu_dec_B, target.p_mu_dec_S,
        target.p_sigma_B, target.p_sigma_S, target.p_sigma_F,
        target.p_sigma_A, target.p_sigma_K,
        target.A_TYP_f32, target.F_TYP_f32,
        # Noise + cost shaping + time
        target.fixed_w,
        target.c_Phi, target.Phi_max,
        target.lam_Phi, target.F_max, target.lam_F,
        target.A_thr, target.lam_chance,
        target.beta_chance, target.scale_chance,
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
    make_log_density_fn_v5(target) -> Function

Build a closure `f :: Matrix{Float64} → Vector{Float64}` mapping
`(M, 2·n_anchors)` θ matrix to per-chain log-density (= −cost). This is
the contract the generic framework controller (`SMC2FC.run_tempered_smc_gpu`)
expects from any model.
"""
function make_log_density_fn_v5(target::FSAv5ControlGPUTarget)
    return function (U::AbstractMatrix{Float64})
        return gpu_cost_log_density_batched_v5(target, U)
    end
end


# ── Single-thread debug entry point — closes the GPU control gap ──────
#
# Runs the production cost kernel for ndrange=1 (one chain, one trial,
# n_steps bins) and returns the resulting cost. Lets the v5 differential
# test verify the per-thread cost accumulation against a CPU reference
# computed via `_cpu_cost_one_reference` (defined in the diff test).

"""
    gpu_cost_one!(target, theta, noise) -> Float32

Run the v5 cost kernel for ndrange=1 with one chain and one trial.
`theta` is length-`(2·n_anchors)`; `noise` is `(n_steps, 6)` standard
normal pre-drawn noise. Returns the scalar cost (single fp32 value).

For diff-test use only — allocates fresh GPU buffers each call.
"""
function gpu_cost_one!(target::FSAv5ControlGPUTarget,
                         theta::AbstractVector{<:Real},
                         noise::AbstractMatrix{<:Real})
    @assert length(theta) == 2 * target.n_anchors
    @assert size(noise) == (target.n_steps, 6)

    # 1-trial noise grid: shape (1, n_steps, 6).
    fixed_w_one = CuArray(reshape(Float32.(noise), 1, target.n_steps, 6))
    theta_one   = CuArray(reshape(Float32.(theta), 1, 2 * target.n_anchors))
    cost_out    = CUDA.zeros(Float32, 1)

    kernel = fsa_v5_cost_kernel!(CUDABackend(), 1)
    kernel(
        cost_out,
        theta_one,
        target.rbf_design,
        target.init_state,
        target.p_tau_B, target.p_kappa_B, target.p_epsilon_AB,
        target.p_tau_S, target.p_kappa_S, target.p_epsilon_AS,
        target.p_tau_F, target.p_lambda_A,
        target.p_mu_K,
        target.p_mu_0, target.p_mu_B, target.p_mu_S, target.p_mu_F,
        target.p_mu_FF, target.p_eta,
        target.p_KFB_0, target.p_KFS_0, target.p_tau_K,
        target.p_B_dec, target.p_S_dec,
        target.p_mu_dec_B, target.p_mu_dec_S,
        target.p_sigma_B, target.p_sigma_S, target.p_sigma_F,
        target.p_sigma_A, target.p_sigma_K,
        target.A_TYP_f32, target.F_TYP_f32,
        fixed_w_one,
        target.c_Phi, target.Phi_max,
        target.lam_Phi, target.F_max, target.lam_F,
        target.A_thr, target.lam_chance,
        target.beta_chance, target.scale_chance,
        target.dt, target.n_substeps,
        target.n_steps, target.n_anchors,
        1, 1;
        ndrange = 1,
    )
    KernelAbstractions.synchronize(CUDABackend())
    return Array(cost_out)[1]
end


end # module GPUControlV5
