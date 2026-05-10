# FSA v5 GPU particle filter — purely functional public API.
#
# Mirrors v1.5's `gpu_pf.jl` plumbing verbatim:
#   - one thread = one (chain, particle)
#   - inner R-bin propagate loop
#   - GPU-resident log-likelihood accumulator (no per-segment PCIe round-trip)
#   - same SMC2FC_functional resampling / Liu-West / OT framework calls
#
# What differs from v1.5 (the *math block* and *dimensions*):
#   - 6D particle state [B, S, F, A, K_FB, K_FS] (was 3D)
#   - 15 estimated drift params + 22 estimated obs-channel params per chain
#     (was 14 v1-form drift in a single matrix)
#   - 14 frozen params (5 diffusion + 8 dynamics-side + circadian phase)
#     packed as kernel scalar args
#   - 5 obs channels (HR / Sleep / Stress / Steps / VolumeLoad) with explicit
#     per-bin gating via Float32 masks (was 3 direct Gaussians, no gating)
#   - boundary handling: clamp B,S to [ε, 1-ε]; floor F,A,K at 0
#     (per tech guide §6.4; was reflect)
#   - 6D NaN-guard fallback to DEFAULT_INIT
#
# The CPU functions `drift_v5`, `diffusion_v5`, `em_step_v5`,
# `hr_mean`, `sleep_prob`, `stress_mean`, `steps_log_mean`,
# `volume_load_mean`, `obs_log_weight_v5` are diff-tested at machine
# precision against the Lean reference. The math block below
# transcribes those same expressions into a single fp32 GPU kernel
# body. A single-thread debug entry point (`gpu_propagate_one!`) lets
# the v5 differential test invoke the kernel for ndrange=1 and check
# bit-exact (modulo fp32 round-off) agreement against the CPU
# functions — closing the §8 GPU-coverage gap from
# `lean4_to_julia_pipeline.tex`.

module GPUPFv5

using CUDA
using KernelAbstractions
using Random: AbstractRNG, MersenneTwister
using StableRNGs

import ..SimulationV5: TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5,
                        FROZEN_PARAMS_V5, DEFAULT_INIT,
                        A_TYP, F_TYP
import ..EstimationV5: PARAM_NAMES_V5, PARAM_PRIOR_CONFIG_V5

# Same framework as v1.5 (bit-identical migration confirmed at commit
# 4a5c007). v5 reuses GPUSegmentedBuffers and run_segmented_smc_step!
# with `n_states = 6` instead of 3.
import SMC2FC_functional
using SMC2FC_functional: GPUSegmentedBuffers, run_segmented_smc_step!

export FSAv5GPUTarget, gpu_log_density_v5
export gpu_propagate_one!   # single-thread debug entry for the diff test


# ── Per-segment propagate kernel — v5 math, fp32 inner loop ───────────
#
# One thread = one (chain, particle). Runs R bins of the v5 6D SDE
# (drift + state-dep diffusion + clamp/floor) and adds the per-bin
# 5-channel obs log-weight (each channel multiplied by its gate to
# zero-out absent observations).
#
# n_dec is hard-coded to 4 (`B*B*B*B` instead of `pow(B, n_dec)`) per
# tech guide §7.2: n_dec is a structural shape parameter pinned at 4.0
# in production. If a future variant needs n_dec ≠ 4 the kernel will
# need rebuilding.

@kernel function propagate_segment_kernel_v5!(
    log_w_inout,                 # (M·K,)              Float32, in-out
    particles_in,                # (M·K, 6)            Float32
    particles_out,               # (M·K, 6)            Float32
    params_dyn_per_chain,        # (M, 15)             Float32 — estimated dynamics
    params_obs_per_chain,        # (M, 22)             Float32 — estimated obs-channel
    Phi_B_seq, Phi_S_seq,        # (T,)                Float32 each
    obs_HR_seq, obs_S_seq, obs_steps_seq, obs_VL_seq,   # (T,) Float32
    obs_sleep_seq,               # (T,)                Float32 — 0.0 or 1.0
    gate_HR_seq, gate_stress_seq, gate_steps_seq, gate_VL_seq,   # (T,) Float32
    gate_sleep_seq,              # (T,)                Float32
    C_seq,                       # (T,)                Float32 — circadian
    noise_grid,                  # (K, T, 6)           Float32 (CRN)
    # ── 14 frozen scalar params (constant across chains) ──────────────
    KFB_0::Float32, KFS_0::Float32, tau_K::Float32,
    B_dec::Float32, S_dec::Float32,
    mu_dec_B::Float32, mu_dec_S::Float32,
    sigma_B::Float32, sigma_S::Float32, sigma_F::Float32,
    sigma_A::Float32, sigma_K::Float32,
    # Operating-point constants:
    A_TYP_f32::Float32, F_TYP_f32::Float32,
    # NaN-guard fallback (tech guide §6.4): values used to reset a
    # particle to a safe interior point if its propagated state goes
    # non-finite. Threaded from the bench's --init-preset selection.
    nan_B::Float32, nan_S::Float32,
    nan_F::Float32, nan_A::Float32,
    nan_KFB::Float32, nan_KFS::Float32,
    # ── Time-step ──────────────────────────────────────────────────────
    dt::Float32, sqrt_dt::Float32,
    R::Int, K_per_chain::Int, seg_step_offset::Int,
)
    i = @index(Global, Linear)
    m_chain = ((i - 1) ÷ K_per_chain) + 1
    p_idx   = ((i - 1) % K_per_chain) + 1

    # ── Estimated dynamics (15) — same row layout as PARAM_NAMES_V5 first 15 ─
    tau_B      = params_dyn_per_chain[m_chain,  1]
    kappa_B    = params_dyn_per_chain[m_chain,  2]
    epsilon_AB = params_dyn_per_chain[m_chain,  3]
    tau_S      = params_dyn_per_chain[m_chain,  4]
    kappa_S    = params_dyn_per_chain[m_chain,  5]
    epsilon_AS = params_dyn_per_chain[m_chain,  6]
    tau_F      = params_dyn_per_chain[m_chain,  7]
    lambda_A   = params_dyn_per_chain[m_chain,  8]
    mu_K       = params_dyn_per_chain[m_chain,  9]
    mu_0       = params_dyn_per_chain[m_chain, 10]
    mu_B_p     = params_dyn_per_chain[m_chain, 11]
    mu_S_p     = params_dyn_per_chain[m_chain, 12]
    mu_F_p     = params_dyn_per_chain[m_chain, 13]
    mu_FF      = params_dyn_per_chain[m_chain, 14]
    eta        = params_dyn_per_chain[m_chain, 15]

    # ── Estimated obs-channel (22) — same row layout as OBS_PARAM_KEYS_V5 ─
    HR_base     = params_obs_per_chain[m_chain,  1]
    kappa_B_HR  = params_obs_per_chain[m_chain,  2]
    alpha_A_HR  = params_obs_per_chain[m_chain,  3]
    beta_C_HR   = params_obs_per_chain[m_chain,  4]
    sigma_HR    = params_obs_per_chain[m_chain,  5]
    k_C         = params_obs_per_chain[m_chain,  6]
    k_A         = params_obs_per_chain[m_chain,  7]
    c_tilde     = params_obs_per_chain[m_chain,  8]
    S_base      = params_obs_per_chain[m_chain,  9]
    k_F         = params_obs_per_chain[m_chain, 10]
    k_A_S       = params_obs_per_chain[m_chain, 11]
    beta_C_S    = params_obs_per_chain[m_chain, 12]
    sigma_S_obs = params_obs_per_chain[m_chain, 13]
    mu_step0    = params_obs_per_chain[m_chain, 14]
    beta_B_st   = params_obs_per_chain[m_chain, 15]
    beta_F_st   = params_obs_per_chain[m_chain, 16]
    beta_A_st   = params_obs_per_chain[m_chain, 17]
    beta_C_st   = params_obs_per_chain[m_chain, 18]
    sigma_st    = params_obs_per_chain[m_chain, 19]
    beta_S_VL   = params_obs_per_chain[m_chain, 20]
    beta_F_VL   = params_obs_per_chain[m_chain, 21]
    sigma_VL    = params_obs_per_chain[m_chain, 22]

    # Per-chain log-norm and inv-2σ² constants for each Gaussian channel
    inv_2sigma_HR² = 0.5f0 / (sigma_HR * sigma_HR)
    inv_2sigma_S²  = 0.5f0 / (sigma_S_obs * sigma_S_obs)
    inv_2sigma_st² = 0.5f0 / (sigma_st * sigma_st)
    inv_2sigma_VL² = 0.5f0 / (sigma_VL * sigma_VL)
    log2π_f32      = Float32(log(2.0 * pi))
    log_norm_HR    = -0.5f0 * log2π_f32 - log(sigma_HR)
    log_norm_S     = -0.5f0 * log2π_f32 - log(sigma_S_obs)
    log_norm_st    = -0.5f0 * log2π_f32 - log(sigma_st)
    log_norm_VL    = -0.5f0 * log2π_f32 - log(sigma_VL)

    # Particle state
    B   = particles_in[i, 1]
    S   = particles_in[i, 2]
    F   = particles_in[i, 3]
    A   = particles_in[i, 4]
    KFB = particles_in[i, 5]
    KFS = particles_in[i, 6]
    log_w = log_w_inout[i]

    eps_B = 1f-4
    eps_S = 1f-4

    @inbounds for k in 1:R
        t   = seg_step_offset + k
        Phi_B = Phi_B_seq[t]
        Phi_S = Phi_S_seq[t]

        # ── v5 bifurcation parameter μ(B, S, F) ───────────────────────
        # n_dec hardcoded to 4 (frozen per tech guide §7.2).
        F_dev = F - F_TYP_f32
        Bn   = max(B, 0f0)
        Bn   = Bn * Bn * Bn * Bn
        Sn   = max(S, 0f0)
        Sn   = Sn * Sn * Sn * Sn
        Bdn  = B_dec * B_dec * B_dec * B_dec
        Sdn  = S_dec * S_dec * S_dec * S_dec
        dec_B = mu_dec_B * Bdn / (Bn + Bdn)
        dec_S = mu_dec_S * Sdn / (Sn + Sdn)
        mu_bif = mu_0 +
                  mu_B_p * B + mu_S_p * S -
                  mu_F_p * F - mu_FF * F_dev * F_dev -
                  dec_B - dec_S

        # ── Drifts (per tech guide §2.2) ──────────────────────────────
        a_factor_B = (1f0 + epsilon_AB * A) / (1f0 + epsilon_AB * A_TYP_f32)
        a_factor_S = (1f0 + epsilon_AS * A) / (1f0 + epsilon_AS * A_TYP_f32)
        a_factor_F = (1f0 + lambda_A  * A) / (1f0 + lambda_A  * A_TYP_f32)
        dB   = kappa_B * a_factor_B * Phi_B - B / tau_B
        dS   = kappa_S * a_factor_S * Phi_S - S / tau_S
        dF   = KFB * Phi_B + KFS * Phi_S - a_factor_F / tau_F * F
        dA   = mu_bif * A - eta * A * A * A
        dKFB = (KFB_0 - KFB) / tau_K + mu_K * Phi_B
        dKFS = (KFS_0 - KFS) / tau_K + mu_K * Phi_S

        # ── State-dep diffusion (Itô) ─────────────────────────────────
        gB   = sigma_B * sqrt(max(B * (1f0 - B), 0f0))
        gS   = sigma_S * sqrt(max(S * (1f0 - S), 0f0))
        gF   = sigma_F * sqrt(max(F, 0f0))
        gA   = sigma_A * sqrt(max(A, 0f0))
        gKFB = sigma_K * sqrt(max(KFB, 0f0))
        gKFS = sigma_K * sqrt(max(KFS, 0f0))

        ξB  = noise_grid[p_idx, t, 1]
        ξS  = noise_grid[p_idx, t, 2]
        ξF  = noise_grid[p_idx, t, 3]
        ξA  = noise_grid[p_idx, t, 4]
        ξKB = noise_grid[p_idx, t, 5]
        ξKS = noise_grid[p_idx, t, 6]

        x_B   = B   + dt * dB   + gB   * sqrt_dt * ξB
        x_S   = S   + dt * dS   + gS   * sqrt_dt * ξS
        x_F   = F   + dt * dF   + gF   * sqrt_dt * ξF
        x_A   = A   + dt * dA   + gA   * sqrt_dt * ξA
        x_KFB = KFB + dt * dKFB + gKFB * sqrt_dt * ξKB
        x_KFS = KFS + dt * dKFS + gKFS * sqrt_dt * ξKS

        # ── NaN-guard + clamp/floor (tech guide §6.4) ────────────────
        if !isfinite(x_B) || !isfinite(x_S) || !isfinite(x_F) ||
           !isfinite(x_A) || !isfinite(x_KFB) || !isfinite(x_KFS)
            x_B   = nan_B
            x_S   = nan_S
            x_F   = nan_F
            x_A   = nan_A
            x_KFB = nan_KFB
            x_KFS = nan_KFS
        else
            x_B = max(eps_B, min(1f0 - eps_B, x_B))
            x_S = max(eps_S, min(1f0 - eps_S, x_S))
            x_F = max(0f0, x_F)
            x_A = max(0f0, x_A)
            x_KFB = max(0f0, x_KFB)
            x_KFS = max(0f0, x_KFS)
        end

        # ── 5-channel obs log-weight (gates zero out absent channels) ─
        Ct = C_seq[t]

        # HR — sleep-active Gaussian
        mu_HR = HR_base - kappa_B_HR * x_B + alpha_A_HR * x_A + beta_C_HR * Ct
        ΔHR   = obs_HR_seq[t] - mu_HR
        log_w += gate_HR_seq[t] * (log_norm_HR - inv_2sigma_HR² * ΔHR * ΔHR)

        # Stress — wake-active Gaussian
        mu_St = S_base + k_F * x_F - k_A_S * x_A + beta_C_S * Ct
        ΔSt   = obs_S_seq[t] - mu_St
        log_w += gate_stress_seq[t] * (log_norm_S - inv_2sigma_S² * ΔSt * ΔSt)

        # Steps — wake-active log-Gaussian (obs is in log-space; see CPU
        # `obs_log_weight_v5` / `steps_log_mean` for the convention)
        mu_step = mu_step0 + beta_B_st * x_B - beta_F_st * x_F +
                   beta_A_st * x_A + beta_C_st * Ct
        Δstep   = obs_steps_seq[t] - mu_step
        log_w  += gate_steps_seq[t] * (log_norm_st - inv_2sigma_st² * Δstep * Δstep)

        # VolumeLoad — training-session-only Gaussian, no circadian
        mu_VL = beta_S_VL * x_S - beta_F_VL * x_F
        ΔVL   = obs_VL_seq[t] - mu_VL
        log_w += gate_VL_seq[t] * (log_norm_VL - inv_2sigma_VL² * ΔVL * ΔVL)

        # Sleep — Bernoulli logistic.
        # Compute log(p) and log(1-p) with a clamp to avoid log(0).
        # The clamp matches the CPU `obs_log_weight_v5` (1e-12 guard).
        z_sleep = k_C * Ct + k_A * x_A - c_tilde
        p_sleep = 1f0 / (1f0 + exp(-z_sleep))
        p_safe  = max(1f-12, min(1f0 - 1f-12, p_sleep))
        log_w  += gate_sleep_seq[t] * (
            obs_sleep_seq[t] * log(p_safe) +
            (1f0 - obs_sleep_seq[t]) * log(1f0 - p_safe)
        )

        B = x_B; S = x_S; F = x_F; A = x_A; KFB = x_KFB; KFS = x_KFS
    end

    particles_out[i, 1] = B
    particles_out[i, 2] = S
    particles_out[i, 3] = F
    particles_out[i, 4] = A
    particles_out[i, 5] = KFB
    particles_out[i, 6] = KFS
    log_w_inout[i]      = log_w
end


# ── Per-chain log-lik accumulation kernel ─────────────────────────────
# Identical to v1.5's; kept in this module so v5 doesn't have to depend
# on GPUPF symbols.

@kernel function gpu_log_lik_accumulate_kernel_v5!(
    log_lik_acc, log_max, log_z, log_K::Float32, M::Int,
)
    m = @index(Global, Linear)
    if m <= M
        @inbounds log_lik_acc[m] += Float64(log_max[m] + log_z[m] - log_K)
    end
end


# ── Constrained ↔ unconstrained map for the 37 estimated params ───────
# All 37 are LogNormal per `PARAM_PRIOR_CONFIG_V5`; the constrained
# value is `exp(clamp(u, -20, 20))`. Returns two split vectors
# (dynamics first 15, obs last 22) matching the kernel row layouts.

@inline function _to_v5_constrained_split(u::AbstractVector{T}) where {T<:Real}
    @assert length(u) == 37 "expected 37 unconstrained params, got $(length(u))"
    constrained = exp.(clamp.(u, -20.0, 20.0))
    return (
        dyn = view(constrained, 1:15),
        obs = view(constrained, 16:37),
    )
end


# ── Immutable target struct ────────────────────────────────────────────

struct FSAv5GPUTarget
    K_per_chain::Int
    M_max::Int
    T_steps::Int
    R::Int
    n_states::Int                    # = 6
    dt::Float32
    sqrt_dt::Float32
    a_shrink::Float32
    ot_max_weight::Float32
    ot_threshold::Float32
    ot_temperature::Float32
    # Frozen params (kernel scalar args, constant across chains):
    KFB_0::Float32; KFS_0::Float32; tau_K::Float32
    B_dec::Float32; S_dec::Float32
    mu_dec_B::Float32; mu_dec_S::Float32
    sigma_B::Float32; sigma_S::Float32; sigma_F::Float32
    sigma_A::Float32; sigma_K::Float32
    A_TYP_f32::Float32; F_TYP_f32::Float32
    # NaN-guard fallback (tech guide §6.4): values to reset a particle to
    # if its propagated state goes non-finite. Threaded from the bench's
    # --init-preset selection so the rescue target matches the patient
    # the bench is simulating.
    nan_fallback_B::Float32; nan_fallback_S::Float32
    nan_fallback_F::Float32; nan_fallback_A::Float32
    nan_fallback_KFB::Float32; nan_fallback_KFS::Float32
    # GPU buffers (pre-allocated, written by kernels)
    params_dyn_per_chain::CuArray{Float32, 2}        # (M_max, 15)
    params_obs_per_chain::CuArray{Float32, 2}        # (M_max, 22)
    Phi_B_seq::CuArray{Float32, 1}
    Phi_S_seq::CuArray{Float32, 1}
    obs_HR_seq::CuArray{Float32, 1}
    obs_S_seq::CuArray{Float32, 1}
    obs_steps_seq::CuArray{Float32, 1}
    obs_VL_seq::CuArray{Float32, 1}
    obs_sleep_seq::CuArray{Float32, 1}
    gate_HR_seq::CuArray{Float32, 1}
    gate_stress_seq::CuArray{Float32, 1}
    gate_steps_seq::CuArray{Float32, 1}
    gate_VL_seq::CuArray{Float32, 1}
    gate_sleep_seq::CuArray{Float32, 1}
    C_seq::CuArray{Float32, 1}
    noise_grid::CuArray{Float32, 3}                  # (K, T, 6)
    bufs::GPUSegmentedBuffers
    propagate_kernel::Any
    log_lik_acc_gpu::CuArray{Float64, 1}
    log_lik_accum_kernel::Any
end


function FSAv5GPUTarget(; K_per_chain::Int, M_max::Int, T_steps::Int,
                          R::Int = 4, dt::Real,
                          a_shrink::Real = 0.98,
                          ot_max_weight::Real = 0.0,    # default OFF (v1.5 lesson)
                          ot_threshold_frac::Real = 0.05,
                          ot_temperature::Real = 5.0,
                          noise_seed::Int = 0,
                          # NaN-guard fallback values; default is DEFAULT_INIT
                          # for back-compat with any caller that doesn't
                          # pass them explicitly. The bench passes the
                          # --init-preset selection here.
                          nan_fallback_B::Real   = DEFAULT_INIT.B,
                          nan_fallback_S::Real   = DEFAULT_INIT.S,
                          nan_fallback_F::Real   = DEFAULT_INIT.F,
                          nan_fallback_A::Real   = DEFAULT_INIT.A,
                          nan_fallback_KFB::Real = DEFAULT_INIT.KFB,
                          nan_fallback_KFS::Real = DEFAULT_INIT.KFS)
    @assert T_steps % R == 0 "T_steps=$T_steps must be divisible by R=$R"

    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, K_per_chain, T_steps, 6)

    n_states = 6
    bufs = GPUSegmentedBuffers(K_per_chain, M_max, n_states)

    return FSAv5GPUTarget(
        K_per_chain, M_max, T_steps, R, n_states,
        Float32(dt), Float32(sqrt(dt)),
        Float32(a_shrink),
        Float32(ot_max_weight),
        Float32(K_per_chain * ot_threshold_frac),
        Float32(ot_temperature),
        # Frozen scalars from FROZEN_PARAMS_V5
        Float32(FROZEN_PARAMS_V5[:KFB_0]),
        Float32(FROZEN_PARAMS_V5[:KFS_0]),
        Float32(FROZEN_PARAMS_V5[:tau_K]),
        Float32(FROZEN_PARAMS_V5[:B_dec]),
        Float32(FROZEN_PARAMS_V5[:S_dec]),
        Float32(FROZEN_PARAMS_V5[:mu_dec_B]),
        Float32(FROZEN_PARAMS_V5[:mu_dec_S]),
        Float32(FROZEN_PARAMS_V5[:sigma_B]),
        Float32(FROZEN_PARAMS_V5[:sigma_S]),
        Float32(FROZEN_PARAMS_V5[:sigma_F]),
        Float32(FROZEN_PARAMS_V5[:sigma_A]),
        Float32(FROZEN_PARAMS_V5[:sigma_K]),
        Float32(A_TYP), Float32(F_TYP),
        # NaN-guard fallback (preset-selected)
        Float32(nan_fallback_B),   Float32(nan_fallback_S),
        Float32(nan_fallback_F),   Float32(nan_fallback_A),
        Float32(nan_fallback_KFB), Float32(nan_fallback_KFS),
        # GPU buffers
        CUDA.zeros(Float32, M_max, 15),
        CUDA.zeros(Float32, M_max, 22),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CuArray(noise_cpu),
        bufs,
        propagate_segment_kernel_v5!(CUDABackend(), 256),
        CUDA.zeros(Float64, M_max),
        gpu_log_lik_accumulate_kernel_v5!(CUDABackend(), 64),
    )
end


# ── Helper: pack a 15-row of v5 estimated dynamics from `params_dict_to_nt`
# style ordering. The 15 estimated dynamics names, in PARAM_NAMES_V5 order:
#   tau_B, kappa_B, epsilon_AB, tau_S, kappa_S, epsilon_AS,
#   tau_F, lambda_A, mu_K,
#   mu_0, mu_B, mu_S, mu_F, mu_FF, eta
@inline function _pack_dyn_row!(row::AbstractVector{Float32},
                                  vals::AbstractVector{<:Real})
    @assert length(vals) == 15 "_pack_dyn_row! expects 15 values, got $(length(vals))"
    @inbounds for i in 1:15
        row[i] = Float32(vals[i])
    end
end

# 22 estimated obs-channel names, in OBS_PARAM_KEYS_V5 order:
#   HR_base, kappa_B_HR, alpha_A_HR, beta_C_HR, sigma_HR,
#   k_C, k_A, c_tilde,
#   S_base, k_F, k_A_S, beta_C_S, sigma_S_obs,
#   mu_step0, beta_B_st, beta_F_st, beta_A_st, beta_C_st, sigma_st,
#   beta_S_VL, beta_F_VL, sigma_VL
@inline function _pack_obs_row!(row::AbstractVector{Float32},
                                  vals::AbstractVector{<:Real})
    @assert length(vals) == 22 "_pack_obs_row! expects 22 values, got $(length(vals))"
    @inbounds for i in 1:22
        row[i] = Float32(vals[i])
    end
end


# ── Pure log-density evaluator (functional public API) ────────────────
#
# `grid_obs` is a NamedTuple with fields:
#   :Phi_B          :: Vector{Float32}   length T_steps
#   :Phi_S          :: Vector{Float32}   length T_steps
#   :obs_HR         :: Vector{Float32}   length T_steps
#   :obs_S          :: Vector{Float32}   length T_steps
#   :obs_steps      :: Vector{Float32}   length T_steps  (LOG values)
#   :obs_VL         :: Vector{Float32}   length T_steps
#   :obs_sleep      :: Vector{Float32}   length T_steps  (0.0/1.0)
#   :gate_HR        :: Vector{Float32}   length T_steps  (0.0/1.0)
#   :gate_stress    :: Vector{Float32}   length T_steps
#   :gate_steps     :: Vector{Float32}   length T_steps
#   :gate_VL        :: Vector{Float32}   length T_steps
#   :gate_sleep     :: Vector{Float32}   length T_steps
#   :C              :: Vector{Float32}   length T_steps
#   :init_state     :: NTuple{6, Float64}    initial 6D state

"""
    gpu_log_density_v5(target, U_unc, grid_obs, key) -> Vector{Float64}

Compute per-chain log p(y | θ) for M chains under the v5 5-channel obs
model. M = size(U_unc, 1), must be ≤ target.M_max. `U_unc` is `(M, 37)`
in `PARAM_NAMES_V5` order (all 37 LogNormal-unconstrained).

Pure: same `(target, U_unc, grid_obs, key)` always returns the same
output. No public field of `target` is mutated.
"""
function gpu_log_density_v5(target::FSAv5GPUTarget,
                              U_unc::AbstractMatrix{Float64},
                              grid_obs::NamedTuple,
                              key::UInt64;
                              use_ot::Bool = true)
    M = size(U_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    @assert size(U_unc, 2) == 37 "U_unc must have 37 columns (PARAM_NAMES_V5 order)"
    K  = target.K_per_chain
    Ntot = M * K
    R  = target.R
    n_segments = target.T_steps ÷ R

    # ── Pack per-chain params (estimated only; frozen are kernel scalars) ─
    params_dyn_cpu = Matrix{Float32}(undef, M, 15)
    params_obs_cpu = Matrix{Float32}(undef, M, 22)
    @inbounds for m in 1:M
        u_row = view(U_unc, m, :)
        split = _to_v5_constrained_split(u_row)
        _pack_dyn_row!(view(params_dyn_cpu, m, :), split.dyn)
        _pack_obs_row!(view(params_obs_cpu, m, :), split.obs)
    end
    copyto!(view(target.params_dyn_per_chain, 1:M, :), params_dyn_cpu)
    copyto!(view(target.params_obs_per_chain, 1:M, :), params_obs_cpu)

    # ── Stage obs / gates / circadian onto GPU ────────────────────────
    copyto!(target.Phi_B_seq,        Float32.(grid_obs.Phi_B))
    copyto!(target.Phi_S_seq,        Float32.(grid_obs.Phi_S))
    copyto!(target.obs_HR_seq,       Float32.(grid_obs.obs_HR))
    copyto!(target.obs_S_seq,        Float32.(grid_obs.obs_S))
    copyto!(target.obs_steps_seq,    Float32.(grid_obs.obs_steps))
    copyto!(target.obs_VL_seq,       Float32.(grid_obs.obs_VL))
    copyto!(target.obs_sleep_seq,    Float32.(grid_obs.obs_sleep))
    copyto!(target.gate_HR_seq,      Float32.(grid_obs.gate_HR))
    copyto!(target.gate_stress_seq,  Float32.(grid_obs.gate_stress))
    copyto!(target.gate_steps_seq,   Float32.(grid_obs.gate_steps))
    copyto!(target.gate_VL_seq,      Float32.(grid_obs.gate_VL))
    copyto!(target.gate_sleep_seq,   Float32.(grid_obs.gate_sleep))
    copyto!(target.C_seq,            Float32.(grid_obs.C))

    # ── Init particle cloud + log_w ───────────────────────────────────
    bufs = target.bufs
    fill!(view(bufs.particles_a, 1:Ntot, 1), Float32(grid_obs.init_state[1]))
    fill!(view(bufs.particles_a, 1:Ntot, 2), Float32(grid_obs.init_state[2]))
    fill!(view(bufs.particles_a, 1:Ntot, 3), Float32(grid_obs.init_state[3]))
    fill!(view(bufs.particles_a, 1:Ntot, 4), Float32(grid_obs.init_state[4]))
    fill!(view(bufs.particles_a, 1:Ntot, 5), Float32(grid_obs.init_state[5]))
    fill!(view(bufs.particles_a, 1:Ntot, 6), Float32(grid_obs.init_state[6]))
    fill!(view(bufs.log_w, 1:Ntot), 0f0)

    fill!(view(target.log_lik_acc_gpu, 1:M), 0.0)
    log_K = Float32(log(Float32(K)))

    for seg in 1:n_segments
        seg_step_offset = (seg - 1) * R

        target.propagate_kernel(
            view(bufs.log_w, 1:Ntot),
            view(bufs.particles_a, 1:Ntot, :),
            view(bufs.particles_b, 1:Ntot, :),
            view(target.params_dyn_per_chain, 1:M, :),
            view(target.params_obs_per_chain, 1:M, :),
            target.Phi_B_seq, target.Phi_S_seq,
            target.obs_HR_seq, target.obs_S_seq,
            target.obs_steps_seq, target.obs_VL_seq,
            target.obs_sleep_seq,
            target.gate_HR_seq, target.gate_stress_seq,
            target.gate_steps_seq, target.gate_VL_seq,
            target.gate_sleep_seq,
            target.C_seq, target.noise_grid,
            target.KFB_0, target.KFS_0, target.tau_K,
            target.B_dec, target.S_dec,
            target.mu_dec_B, target.mu_dec_S,
            target.sigma_B, target.sigma_S, target.sigma_F,
            target.sigma_A, target.sigma_K,
            target.A_TYP_f32, target.F_TYP_f32,
            target.nan_fallback_B,   target.nan_fallback_S,
            target.nan_fallback_F,   target.nan_fallback_A,
            target.nan_fallback_KFB, target.nan_fallback_KFS,
            target.dt, target.sqrt_dt,
            R, K, seg_step_offset;
            ndrange = Ntot,
        )

        if seg == n_segments
            bufs.stats_kernel(
                view(bufs.log_max, 1:M),
                view(bufs.log_z, 1:M),
                view(bufs.ess, 1:M),
                view(bufs.mu_per_chain, 1:M, :),
                view(bufs.log_w, 1:Ntot),
                view(bufs.particles_b, 1:Ntot, :),
                M, K, target.n_states;
                ndrange = M,
            )
        else
            run_segmented_smc_step!(bufs, M, target.a_shrink;
                                     ot_max = target.ot_max_weight,
                                     ot_threshold = target.ot_threshold,
                                     ot_temperature = target.ot_temperature,
                                     use_ot = use_ot)
        end

        target.log_lik_accum_kernel(
            view(target.log_lik_acc_gpu, 1:M),
            view(bufs.log_max, 1:M),
            view(bufs.log_z,   1:M),
            log_K, M;
            ndrange = M,
        )
    end
    return Array(view(target.log_lik_acc_gpu, 1:M))
end


# ── Single-thread debug entry point — closes the §8 GPU coverage gap ──
#
# Allocates a tiny 1-particle, 1-bin scenario, runs the production
# kernel for ndrange=1 with all gates set to 1.0, returns the next
# state and accumulated log-weight. Lets the v5 differential test
# verify per-thread math against the diff-tested CPU functions
# (`drift_v5`, `diffusion_v5`, `obs_log_weight_v5`) at fp32 precision
# (~1e-5 relative).
#
# This is the production kernel — same code path as the bench — just
# invoked at minimum size so a unit test can compare individual
# thread output against the CPU reference.

"""
    gpu_propagate_one!(state0, phi, params_dyn, params_obs, frozen,
                        obs, gates, C, noise, dt) -> (next_state, log_w)

Run the v5 propagate kernel for ndrange=1 with one (chain, particle)
and one bin (R=1, T=1). Returns the next 6D state and the accumulated
log-weight for one bin.

Inputs (Vector{Float64} for ergonomic test invocation; cast to Float32
internally for the kernel):
  - `state0`     : length-6 initial state.
  - `phi`        : `(Phi_B, Phi_S)`.
  - `params_dyn` : length-15 estimated dynamics (already constrained, in
                    PARAM_NAMES_V5 first-15 order).
  - `params_obs` : length-22 estimated obs-channel (in OBS_PARAM_KEYS_V5
                    order).
  - `frozen`     : NamedTuple with keys `(KFB_0, KFS_0, tau_K, B_dec,
                    S_dec, mu_dec_B, mu_dec_S, sigma_B, sigma_S,
                    sigma_F, sigma_A, sigma_K)`.
  - `obs`        : length-5 vector `[obs_HR, obs_S, obs_steps, obs_VL, obs_sleep]`.
  - `gates`      : length-5 vector of 0.0/1.0 in same order.
  - `C`          : circadian scalar.
  - `noise`      : length-6 standard normal.
  - `dt`         : bin width.

Allocates fresh GPU buffers each call; **for diff-test use only**, not
production hot path.
"""
function gpu_propagate_one!(state0::AbstractVector{<:Real},
                              phi::Tuple{<:Real, <:Real},
                              params_dyn::AbstractVector{<:Real},
                              params_obs::AbstractVector{<:Real},
                              frozen::NamedTuple,
                              obs::AbstractVector{<:Real},
                              gates::AbstractVector{<:Real},
                              C::Real,
                              noise::AbstractVector{<:Real},
                              dt::Real)
    @assert length(state0)     == 6
    @assert length(params_dyn) == 15
    @assert length(params_obs) == 22
    @assert length(obs)        == 5
    @assert length(gates)      == 5
    @assert length(noise)      == 6

    # Allocate minimal GPU buffers — ndrange = 1, R = 1, T = 1, K = 1.
    log_w_gpu = CUDA.zeros(Float32, 1)
    p_in_gpu  = CuArray(reshape(Float32.(state0), 1, 6))
    p_out_gpu = CUDA.zeros(Float32, 1, 6)
    dyn_gpu   = CuArray(reshape(Float32.(params_dyn), 1, 15))
    obs_gpu   = CuArray(reshape(Float32.(params_obs), 1, 22))

    # T = 1 obs / gate / Phi / C arrays.
    Phi_B = CuArray(Float32[phi[1]])
    Phi_S = CuArray(Float32[phi[2]])
    obs_HR_g    = CuArray(Float32[obs[1]])
    obs_S_g     = CuArray(Float32[obs[2]])
    obs_steps_g = CuArray(Float32[obs[3]])
    obs_VL_g    = CuArray(Float32[obs[4]])
    obs_sleep_g = CuArray(Float32[obs[5]])
    gate_HR_g     = CuArray(Float32[gates[1]])
    gate_stress_g = CuArray(Float32[gates[2]])
    gate_steps_g  = CuArray(Float32[gates[3]])
    gate_VL_g     = CuArray(Float32[gates[4]])
    gate_sleep_g  = CuArray(Float32[gates[5]])
    C_g           = CuArray(Float32[C])

    # Noise grid (K=1, T=1, 6) with the supplied 6-vector noise.
    noise_gpu = CuArray(reshape(Float32.(noise), 1, 1, 6))

    kernel = propagate_segment_kernel_v5!(CUDABackend(), 1)
    kernel(
        log_w_gpu, p_in_gpu, p_out_gpu,
        dyn_gpu, obs_gpu,
        Phi_B, Phi_S,
        obs_HR_g, obs_S_g, obs_steps_g, obs_VL_g, obs_sleep_g,
        gate_HR_g, gate_stress_g, gate_steps_g, gate_VL_g, gate_sleep_g,
        C_g, noise_gpu,
        Float32(frozen.KFB_0),  Float32(frozen.KFS_0),  Float32(frozen.tau_K),
        Float32(frozen.B_dec),  Float32(frozen.S_dec),
        Float32(frozen.mu_dec_B), Float32(frozen.mu_dec_S),
        Float32(frozen.sigma_B), Float32(frozen.sigma_S),
        Float32(frozen.sigma_F), Float32(frozen.sigma_A),
        Float32(frozen.sigma_K),
        Float32(A_TYP), Float32(F_TYP),
        # NaN-guard fallback: hardcoded to DEFAULT_INIT here because this
        # is the diff-test single-bin entry point (`gpu_propagate_one!`),
        # not the production hot path. The rescue branch only fires on
        # numerical pathology and a one-bin diff test from a known good
        # state never trips it.
        Float32(DEFAULT_INIT.B),   Float32(DEFAULT_INIT.S),
        Float32(DEFAULT_INIT.F),   Float32(DEFAULT_INIT.A),
        Float32(DEFAULT_INIT.KFB), Float32(DEFAULT_INIT.KFS),
        Float32(dt), Float32(sqrt(dt)),
        1, 1, 0;
        ndrange = 1,
    )
    KernelAbstractions.synchronize(CUDABackend())

    next_state = Array(p_out_gpu)[1, :]    # 6-vector
    log_w      = Array(log_w_gpu)[1]       # scalar
    return (next_state = next_state, log_w = log_w)
end


end # module GPUPFv5
