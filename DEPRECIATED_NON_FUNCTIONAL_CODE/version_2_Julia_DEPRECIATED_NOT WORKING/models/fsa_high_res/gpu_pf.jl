# GPU PF for FSA-v2 — model-specific propagate kernel only.
#
# Generic PF infrastructure (per-chain resample + Liu-West shrinkage +
# OT sigmoid-blend rescue) lives in the framework at
# `julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl`. This file contains ONLY
# the FSA-specific propagate kernel: the locally-guided (Pitt-Shephard)
# proposal that runs R bins of (G1 drift + state-dep cov + 3-channel
# sequential Kalman fusion + Cholesky-3 sample + Bernoulli sleep ll) on
# the GPU in fp32 per particle.
#
# Mirror of `version_2/models/fsa_high_res/estimation.py:propagate_fn`.

module GPUPF

using CUDA
using KernelAbstractions
using LogExpFunctions: logsumexp
using Statistics: mean
using Random: AbstractRNG, MersenneTwister, randn

using ..Dynamics: A_TYP, F_TYP

import SMC2FC
using SMC2FC: GPUSegmentedBuffers, run_segmented_smc_step!

export FSAGPUTargetBatched, gpu_log_density_batched, gpu_grads_parallel_chains
export parallel_hmc_one_move!, update_window_obs!


# ── FSA-specific per-segment propagate kernel ────────────────────────────
#
# Each thread handles one (chain, particle). Reads `particles_in[i, :]` =
# (B, F, A), writes `particles_out[i, :]` = (B, F, A). For R bins:
# G1 drift → state-dep cov → 3-channel Kalman fusion → Cholesky-3 sample
# → predictive log-marginal accumulation → Bernoulli sleep ll.

@kernel function fsa_propagate_segment_kernel!(
    log_w_inout,                     # (M·K,) Float32
    particles_in,                    # (M·K, 3) Float32
    particles_out,                   # (M·K, 3) Float32
    params_per_chain,                # (M, 30) Float32
    Phi_seq, C_seq,
    obs_HR_value, obs_HR_present,
    obs_stress_value, obs_stress_present,
    obs_log_steps_value, obs_steps_present,
    obs_sleep_label, obs_sleep_present,
    noise_grid,                      # (K, T, 3) Float32
    dt::Float32, sqrt_dt::Float32,
    R::Int, K_per_chain::Int, seg_step_offset::Int,
)
    i = @index(Global, Linear)
    m_chain = ((i - 1) ÷ K_per_chain) + 1
    p_idx   = ((i - 1) % K_per_chain) + 1

    tau_B     = params_per_chain[m_chain,  1]
    tau_F     = params_per_chain[m_chain,  2]
    kappa_B   = params_per_chain[m_chain,  3]
    kappa_F   = params_per_chain[m_chain,  4]
    epsilon_A = params_per_chain[m_chain,  5]
    lambda_A  = params_per_chain[m_chain,  6]
    mu_0      = params_per_chain[m_chain,  7]
    mu_B_p    = params_per_chain[m_chain,  8]
    mu_F_p    = params_per_chain[m_chain,  9]
    mu_FF     = params_per_chain[m_chain, 10]
    eta       = params_per_chain[m_chain, 11]
    HR_base    = params_per_chain[m_chain, 12]
    kappa_B_HR = params_per_chain[m_chain, 13]
    alpha_A_HR = params_per_chain[m_chain, 14]
    beta_C_HR  = params_per_chain[m_chain, 15]
    sigma_HR   = params_per_chain[m_chain, 16]
    k_C        = params_per_chain[m_chain, 17]
    k_A        = params_per_chain[m_chain, 18]
    c_tilde    = params_per_chain[m_chain, 19]
    S_base     = params_per_chain[m_chain, 20]
    k_F_p      = params_per_chain[m_chain, 21]
    k_A_S      = params_per_chain[m_chain, 22]
    beta_C_S   = params_per_chain[m_chain, 23]
    sigma_S    = params_per_chain[m_chain, 24]
    mu_step0   = params_per_chain[m_chain, 25]
    beta_B_st  = params_per_chain[m_chain, 26]
    beta_F_st  = params_per_chain[m_chain, 27]
    beta_A_st  = params_per_chain[m_chain, 28]
    beta_C_st  = params_per_chain[m_chain, 29]
    sigma_st   = params_per_chain[m_chain, 30]

    sigma_B = 0.010f0
    sigma_F = 0.012f0
    sigma_A = 0.020f0
    eps_B   = 1f-4
    eps_A   = 1f-4
    A_typ32 = Float32(A_TYP)
    F_typ32 = Float32(F_TYP)
    half_log_2pi = 0.9189385f0
    two_pi = 6.2831855f0

    B = particles_in[i, 1]
    F = particles_in[i, 2]
    A = particles_in[i, 3]
    log_w = log_w_inout[i]

    @inbounds for k_local in 1:R
        k = seg_step_offset + k_local

        F_dev = F - F_typ32
        mu_bif = mu_0 + mu_B_p * B - mu_F_p * F - mu_FF * F_dev * F_dev
        a_factor_B = (1f0 + epsilon_A * A) / (1f0 + epsilon_A * A_typ32)
        a_factor_F = (1f0 + lambda_A  * A) / (1f0 + lambda_A  * A_typ32)
        Phi_t = Phi_seq[k]
        drift_B = kappa_B * a_factor_B * Phi_t - B / tau_B
        drift_F = kappa_F * Phi_t - a_factor_F / tau_F * F
        drift_A = mu_bif * A - eta * A * A * A
        B_pred = B + dt * drift_B
        F_pred = F + dt * drift_F
        A_pred = A + dt * drift_A

        B_cl = max(eps_B, min(1f0 - eps_B, B))
        F_cl = max(0f0, F)
        A_cl = max(0f0, A)
        var_B = max(sigma_B * sigma_B * B_cl * (1f0 - B_cl) * dt, 1f-12)
        var_F = max(sigma_F * sigma_F * F_cl * dt, 1f-12)
        var_A = max(sigma_A * sigma_A * (A_cl + eps_A) * dt, 1f-12)

        mu_b = B_pred; mu_f = F_pred; mu_a = A_pred
        P11 = var_B; P12 = 0f0; P13 = 0f0
        P22 = var_F; P23 = 0f0
        P33 = var_A
        C_k = C_seq[k]
        log_pred_total = 0f0

        # Ch1: HR
        h1 = -kappa_B_HR; h2 = 0f0; h3 = alpha_A_HR
        b  = HR_base + beta_C_HR * C_k
        r  = sigma_HR * sigma_HR
        y_obs = obs_HR_value[k]
        pres  = obs_HR_present[k]
        innov = y_obs - (h1 * mu_b + h2 * mu_f + h3 * mu_a + b)
        Ph1 = P11*h1 + P12*h2 + P13*h3
        Ph2 = P12*h1 + P22*h2 + P23*h3
        Ph3 = P13*h1 + P23*h2 + P33*h3
        S_i = h1*Ph1 + h2*Ph2 + h3*Ph3 + r
        K1 = Ph1 / S_i; K2 = Ph2 / S_i; K3 = Ph3 / S_i
        ll = -0.5f0 * log(two_pi * S_i) - 0.5f0 * (innov*innov) / S_i
        mu_b += pres * K1 * innov; mu_f += pres * K2 * innov; mu_a += pres * K3 * innov
        P11 -= pres * K1 * Ph1; P12 -= pres * K1 * Ph2; P13 -= pres * K1 * Ph3
        P22 -= pres * K2 * Ph2; P23 -= pres * K2 * Ph3; P33 -= pres * K3 * Ph3
        log_pred_total += pres * ll

        # Ch2: stress
        h1 = 0f0; h2 = k_F_p; h3 = -k_A_S
        b  = S_base + beta_C_S * C_k
        r  = sigma_S * sigma_S
        y_obs = obs_stress_value[k]
        pres  = obs_stress_present[k]
        innov = y_obs - (h1 * mu_b + h2 * mu_f + h3 * mu_a + b)
        Ph1 = P11*h1 + P12*h2 + P13*h3
        Ph2 = P12*h1 + P22*h2 + P23*h3
        Ph3 = P13*h1 + P23*h2 + P33*h3
        S_i = h1*Ph1 + h2*Ph2 + h3*Ph3 + r
        K1 = Ph1 / S_i; K2 = Ph2 / S_i; K3 = Ph3 / S_i
        ll = -0.5f0 * log(two_pi * S_i) - 0.5f0 * (innov*innov) / S_i
        mu_b += pres * K1 * innov; mu_f += pres * K2 * innov; mu_a += pres * K3 * innov
        P11 -= pres * K1 * Ph1; P12 -= pres * K1 * Ph2; P13 -= pres * K1 * Ph3
        P22 -= pres * K2 * Ph2; P23 -= pres * K2 * Ph3; P33 -= pres * K3 * Ph3
        log_pred_total += pres * ll

        # Ch3: log(Steps+1)
        h1 = beta_B_st; h2 = -beta_F_st; h3 = beta_A_st
        b  = mu_step0 + beta_C_st * C_k
        r  = sigma_st * sigma_st
        y_obs = obs_log_steps_value[k]
        pres  = obs_steps_present[k]
        innov = y_obs - (h1 * mu_b + h2 * mu_f + h3 * mu_a + b)
        Ph1 = P11*h1 + P12*h2 + P13*h3
        Ph2 = P12*h1 + P22*h2 + P23*h3
        Ph3 = P13*h1 + P23*h2 + P33*h3
        S_i = h1*Ph1 + h2*Ph2 + h3*Ph3 + r
        K1 = Ph1 / S_i; K2 = Ph2 / S_i; K3 = Ph3 / S_i
        ll = -0.5f0 * log(two_pi * S_i) - 0.5f0 * (innov*innov) / S_i
        mu_b += pres * K1 * innov; mu_f += pres * K2 * innov; mu_a += pres * K3 * innov
        P11 -= pres * K1 * Ph1; P12 -= pres * K1 * Ph2; P13 -= pres * K1 * Ph3
        P22 -= pres * K2 * Ph2; P23 -= pres * K2 * Ph3; P33 -= pres * K3 * Ph3
        log_pred_total += pres * ll

        # Cholesky-3 sample
        P11 += 1f-10; P22 += 1f-10; P33 += 1f-10
        L11 = sqrt(max(P11, 1f-12))
        L21 = P12 / L11
        L22 = sqrt(max(P22 - L21*L21, 1f-12))
        L31 = P13 / L11
        L32 = (P23 - L31 * L21) / L22
        L33 = sqrt(max(P33 - L31*L31 - L32*L32, 1f-12))

        nB = noise_grid[p_idx, k, 1]
        nF = noise_grid[p_idx, k, 2]
        nA = noise_grid[p_idx, k, 3]
        x_b = mu_b + L11 * nB
        x_f = mu_f + L21 * nB + L22 * nF
        x_a = mu_a + L31 * nB + L32 * nF + L33 * nA

        if !isfinite(x_b) || !isfinite(x_f) || !isfinite(x_a)
            x_b = 0.05f0; x_f = 0.30f0; x_a = 0.10f0
        else
            x_b = max(eps_B, min(1f0 - eps_B, x_b))
            x_f = max(0f0, x_f)
            x_a = max(0f0, x_a)
        end

        z = k_C * C_k + k_A * x_a - c_tilde
        p_sleep = 1f0 / (1f0 + exp(-z))
        p_safe = max(1f-7, min(1f0 - 1f-7, p_sleep))
        s_lab = Float32(obs_sleep_label[k])
        bern_ll = obs_sleep_present[k] * (
            s_lab * log(p_safe) + (1f0 - s_lab) * log(1f0 - p_safe)
        )

        log_w += log_pred_total + bern_ll
        B = x_b; F = x_f; A = x_a
    end

    particles_out[i, 1] = B
    particles_out[i, 2] = F
    particles_out[i, 3] = A
    log_w_inout[i] = log_w
end


# ── Constrained ↔ unconstrained map (matches Estimation.PARAM_PRIOR_CONFIG) ─

const LOGNORMAL_MASK_FSA = Bool[
    true, true, true, true, true, true,
    true, true, true, true, true,
    false, true, true, false, true,
    true, true, false,
    false, true, true, false, true,
    false, true, true, true, false, true,
]

@inline function _to_constrained_row!(out::AbstractVector{Float32}, u::AbstractVector{<:Real})
    @inbounds for j in 1:length(u)
        out[j] = LOGNORMAL_MASK_FSA[j] ?
                 Float32(exp(clamp(Float64(u[j]), -20.0, 20.0))) :
                 Float32(u[j])
    end
    return out
end


# ── FSA target struct (model-specific) ───────────────────────────────────

mutable struct FSAGPUTargetBatched
    K_per_chain::Int
    M_max::Int
    T_steps::Int
    R::Int
    n_params::Int
    n_states::Int
    dt::Float32
    sqrt_dt::Float32
    B_init::Float32
    F_init::Float32
    A_init::Float32
    a_shrink::Float32
    ot_max_weight::Float32
    ot_threshold::Float32
    ot_temperature::Float32
    Phi_seq::CuArray{Float32,1}
    C_seq::CuArray{Float32,1}
    obs_HR_value::CuArray{Float32,1}
    obs_HR_present::CuArray{Float32,1}
    obs_stress_value::CuArray{Float32,1}
    obs_stress_present::CuArray{Float32,1}
    obs_log_steps_value::CuArray{Float32,1}
    obs_steps_present::CuArray{Float32,1}
    obs_sleep_label::CuArray{Int32,1}
    obs_sleep_present::CuArray{Float32,1}
    noise_grid::CuArray{Float32,3}
    params_per_chain::CuArray{Float32,2}
    bufs::GPUSegmentedBuffers       # framework-shared: particles_a/b, log_w, weights, cumsum, mu, kernel
    propagate_kernel::Any
end


function FSAGPUTargetBatched(; K_per_chain::Int, M_max::Int, T_steps::Int,
                              R::Int = 4,
                              dt::Real,
                              B_init::Real = 0.05, F_init::Real = 0.30, A_init::Real = 0.10,
                              n_params::Int = 30, n_states::Int = 3,
                              a_shrink::Real = 0.98,
                              ot_max_weight::Real = 0.01,
                              ot_threshold_frac::Real = 0.05,
                              ot_temperature::Real = 5.0,
                              noise_seed::Int = 0)
    rng = MersenneTwister(noise_seed)
    noise_cpu = Array{Float32,3}(undef, K_per_chain, T_steps, 3)
    for k in 1:T_steps, p in 1:K_per_chain, c in 1:3
        noise_cpu[p, k, c] = Float32(randn(rng))
    end
    @assert T_steps % R == 0 "T_steps=$T_steps must be divisible by R=$R"

    bufs = GPUSegmentedBuffers(K_per_chain, M_max, n_states)
    return FSAGPUTargetBatched(
        K_per_chain, M_max, T_steps, R, n_params, n_states,
        Float32(dt), Float32(sqrt(dt)),
        Float32(B_init), Float32(F_init), Float32(A_init),
        Float32(a_shrink),
        Float32(ot_max_weight),
        Float32(K_per_chain * ot_threshold_frac),
        Float32(ot_temperature),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Int32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CuArray(noise_cpu),
        CUDA.zeros(Float32, M_max, n_params),
        bufs,
        fsa_propagate_segment_kernel!(CUDABackend(), 256),
    )
end


function update_window_obs!(target::FSAGPUTargetBatched, grid_obs::Dict;
                             B_init::Real = 0.05, F_init::Real = 0.30, A_init::Real = 0.10)
    target.B_init = Float32(B_init)
    target.F_init = Float32(F_init)
    target.A_init = Float32(A_init)
    copyto!(target.Phi_seq,            Float32.(grid_obs[:Phi]))
    copyto!(target.C_seq,              Float32.(grid_obs[:C]))
    copyto!(target.obs_HR_value,       Float32.(grid_obs[:hr_value]))
    copyto!(target.obs_HR_present,     Float32.(grid_obs[:hr_present]))
    copyto!(target.obs_stress_value,   Float32.(grid_obs[:stress_value]))
    copyto!(target.obs_stress_present, Float32.(grid_obs[:stress_present]))
    copyto!(target.obs_log_steps_value, Float32.(grid_obs[:log_steps_value]))
    copyto!(target.obs_steps_present,  Float32.(grid_obs[:steps_present]))
    copyto!(target.obs_sleep_label,    Int32.(grid_obs[:sleep_label]))
    copyto!(target.obs_sleep_present,  Float32.(grid_obs[:sleep_present]))
    return target
end


# ── Public API: drives the framework's segmented PF with the FSA propagate ──

"""
    gpu_log_density_batched(target, U_unc; rng, use_ot=true)
        -> Vector{Float64}

Run the FSA locally-guided PF with framework-shared per-segment resample
+ Liu-West shrinkage + OT sigmoid-blend rescue. Returns per-chain log p(y | θ).
"""
function gpu_log_density_batched(target::FSAGPUTargetBatched,
                                  U_unc::AbstractMatrix{Float64};
                                  rng::AbstractRNG = MersenneTwister(),
                                  use_ot::Bool = true)
    M = size(U_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    K = target.K_per_chain
    Ntot = M * K
    R = target.R
    n_segments = target.T_steps ÷ R
    n_states = target.n_states

    # Pack params per chain.
    params_cpu = Matrix{Float32}(undef, M, target.n_params)
    @inbounds for m in 1:M
        _to_constrained_row!(view(params_cpu, m, :), view(U_unc, m, :))
    end
    copyto!(view(target.params_per_chain, 1:M, :), params_cpu)

    bufs = target.bufs
    # Init particle cloud to (B_init, F_init, A_init), log_w to 0.
    fill!(view(bufs.particles_a, 1:Ntot, 1), target.B_init)
    fill!(view(bufs.particles_a, 1:Ntot, 2), target.F_init)
    fill!(view(bufs.particles_a, 1:Ntot, 3), target.A_init)
    fill!(view(bufs.log_w, 1:Ntot), 0f0)

    log_lik_acc = zeros(Float64, M)
    log_K = log(Float32(K))

    for seg in 1:n_segments
        seg_step_offset = (seg - 1) * R

        # Run R bins of FSA propagate on GPU: a → b
        target.propagate_kernel(
            view(bufs.log_w, 1:Ntot),
            view(bufs.particles_a, 1:Ntot, :),
            view(bufs.particles_b, 1:Ntot, :),
            view(target.params_per_chain, 1:M, :),
            target.Phi_seq, target.C_seq,
            target.obs_HR_value, target.obs_HR_present,
            target.obs_stress_value, target.obs_stress_present,
            target.obs_log_steps_value, target.obs_steps_present,
            target.obs_sleep_label, target.obs_sleep_present,
            target.noise_grid,
            target.dt, target.sqrt_dt,
            R, K, seg_step_offset;
            ndrange = Ntot,
        )
        KernelAbstractions.synchronize(CUDABackend())

        # End-of-segment SMC step: per-chain stats + Liu-West + OT
        # all GPU-resident (framework).
        is_last = (seg == n_segments)
        if !is_last
            run_segmented_smc_step!(bufs, M, target.a_shrink;
                                     ot_max = target.ot_max_weight,
                                     ot_threshold = target.ot_threshold,
                                     ot_temperature = target.ot_temperature,
                                     use_ot = use_ot)
            # Accumulate per-chain log-likelihood: log mean exp(log_w_segment).
            # The stats kernel already wrote log_max, log_z to GPU; copy
            # only those (M floats — tiny) to accumulate.
            log_max_cpu = Array(view(bufs.log_max, 1:M))
            log_z_cpu   = Array(view(bufs.log_z, 1:M))
            @inbounds for m in 1:M
                log_lik_acc[m] += Float64(log_max_cpu[m] + log_z_cpu[m] - log_K)
            end
        else
            # Last segment: just compute stats (no resample needed) for the
            # final log-likelihood accumulator.
            bufs.stats_kernel(
                view(bufs.log_max, 1:M),
                view(bufs.log_z, 1:M),
                view(bufs.ess, 1:M),
                view(bufs.mu_per_chain, 1:M, :),
                view(bufs.log_w, 1:Ntot),
                view(bufs.particles_b, 1:Ntot, :),
                M, K, n_states;
                ndrange = M,
            )
            KernelAbstractions.synchronize(CUDABackend())
            log_max_cpu = Array(view(bufs.log_max, 1:M))
            log_z_cpu   = Array(view(bufs.log_z, 1:M))
            @inbounds for m in 1:M
                log_lik_acc[m] += Float64(log_max_cpu[m] + log_z_cpu[m] - log_K)
            end
        end
    end

    return log_lik_acc
end


# ── FD gradient via batched call ─────────────────────────────────────────

function gpu_grads_parallel_chains(target::FSAGPUTargetBatched,
                                    U_unc::AbstractMatrix{Float64};
                                    h::Float64 = 1e-3,
                                    rng::AbstractRNG = MersenneTwister(),
                                    use_ot::Bool = true)
    M = size(U_unc, 1); d = size(U_unc, 2)
    n_perturb_per_chain = 1 + 2 * d
    n_total = M * n_perturb_per_chain
    n_total ≤ target.M_max || throw(ArgumentError(
        "n_total $(n_total) > M_max $(target.M_max)"))

    U_flat = Matrix{Float64}(undef, n_total, d)
    @inbounds for m in 1:M
        base_row = (m - 1) * n_perturb_per_chain
        U_flat[base_row + 1, :] = U_unc[m, :]
        for i in 1:d
            U_flat[base_row + 1 + 2*(i-1) + 1, :] = U_unc[m, :]
            U_flat[base_row + 1 + 2*(i-1) + 1, i] += h
            U_flat[base_row + 1 + 2*(i-1) + 2, :] = U_unc[m, :]
            U_flat[base_row + 1 + 2*(i-1) + 2, i] -= h
        end
    end

    lls_flat = gpu_log_density_batched(target, U_flat; rng=rng, use_ot=use_ot)

    vals  = Vector{Float64}(undef, M)
    grads = Matrix{Float64}(undef, M, d)
    @inbounds for m in 1:M
        base = (m - 1) * n_perturb_per_chain
        vals[m] = lls_flat[base + 1]
        for i in 1:d
            grads[m, i] = (lls_flat[base + 1 + 2*(i-1) + 1] -
                           lls_flat[base + 1 + 2*(i-1) + 2]) / (2h)
        end
    end
    return vals, grads
end


# ── Parallel-chains HMC move ─────────────────────────────────────────────

function parallel_hmc_one_move!(U::AbstractMatrix{Float64},
                                 target::FSAGPUTargetBatched,
                                 ε::Float64,
                                 L::Int,
                                 prior_mean::AbstractVector{Float64},
                                 prior_sigma::AbstractVector{Float64},
                                 rng::AbstractRNG;
                                 inv_mass::AbstractVector{Float64} = ones(size(U, 2)))
    M, d = size(U)
    momentum = randn(rng, M, d) ./ sqrt.(inv_mass)'
    p0 = copy(momentum)

    function tempered_grads(U_in)
        vals_data, grads_data = gpu_grads_parallel_chains(target, U_in;
                                                            rng=rng, use_ot=true)
        grads_prior = -(U_in .- prior_mean') ./ (prior_sigma' .^ 2)
        vals_prior  = -0.5 .* vec(sum(((U_in .- prior_mean') ./ prior_sigma') .^ 2; dims = 2))
        return vals_data .+ vals_prior, grads_data .+ grads_prior
    end

    val_init, grad_init = tempered_grads(U)
    p = momentum .+ (ε / 2) .* grad_init
    U_new = U .+ ε .* p .* inv_mass'
    for _ in 2:L
        _, grad = tempered_grads(U_new)
        p .= p .+ ε .* grad
        U_new .= U_new .+ ε .* p .* inv_mass'
    end
    val_final, grad_final = tempered_grads(U_new)
    p .= p .+ (ε / 2) .* grad_final

    K0    = 0.5 .* vec(sum(p0 .^ 2 .* inv_mass'; dims = 2))
    K_new = 0.5 .* vec(sum(p  .^ 2 .* inv_mass'; dims = 2))
    log_α = (val_final .- K_new) .- (val_init .- K0)

    n_acc = 0
    @inbounds for m in 1:M
        if log(rand(rng)) < log_α[m]
            U[m, :] = U_new[m, :]
            n_acc += 1
        end
    end
    return n_acc
end


end # module GPUPF
