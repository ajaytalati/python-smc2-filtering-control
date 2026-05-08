# FSA v1.5 GPU particle filter — purely functional public API.
#
# The public functions all return new values; the only mutation is hidden
# inside the KernelAbstractions kernel and the framework's
# `run_segmented_smc_step!` (both write to GPU arrays that are private to
# this module).
#
# Public API:
#   FSAGPUTarget(; ...)                                — immutable target
#   gpu_log_density(target, U_unc, grid_obs, key)      → Vector{Float64}
#   gpu_grads(target, U_unc, grid_obs, h, key)         → (vals, grads)
#   parallel_hmc_one_move(U, target, grid_obs,
#                          ε, L, prior_means, prior_sigmas, key)
#                                                      → (U_new, n_acc)
#
# v1.5 specifics:
#   - 3-channel diagonal Gaussian obs on (B, F, A) every bin, no gating.
#   - 7 estimated drift + 3 estimated diffusion = 10-D U_unc.
#   - 4 dynamics params pinned at truth (τ_B, η, ε_A, μ_FF).
#   - 3 obs noise params pinned at truth (σ_B_obs, σ_F_obs, σ_A_obs).

module GPUPF

using CUDA
using KernelAbstractions
using Random: AbstractRNG, MersenneTwister
using StableRNGs
using Match

import ..Simulation: DEFAULT_PARAMS, fill_pinned_nt, params_v15_to_v1_nt
import ..Estimation: PARAM_NAMES, PARAM_PRIOR_CONFIG

import SMC2FC
using SMC2FC: GPUSegmentedBuffers, run_segmented_smc_step!

export FSAGPUTarget, gpu_log_density, gpu_grads, parallel_hmc_one_move


# ── Per-segment propagate kernel (v1.5: 3-channel Gaussian obs) ──────────
#
# Each thread = one (chain, particle). One thread runs R bins of the v1
# G0 SDE (drift + state-dep diffusion + boundary reflection) and adds the
# 3-channel diagonal Gaussian obs log-weight. CRN noise via a fixed
# `noise_grid[p, t, c]` so identical (state, params, key) → identical
# trajectories — required for FD gradient batching.

@kernel function propagate_segment_kernel!(
    log_w_inout,              # (M·K,)            Float32
    particles_in,             # (M·K, 3)          Float32
    particles_out,            # (M·K, 3)          Float32
    params_per_chain,         # (M, 14)           Float32 — v1-form params
    Phi_seq,                  # (T,)              Float32
    obs_B_value,              # (T,)              Float32
    obs_F_value,              # (T,)              Float32
    obs_A_value,              # (T,)              Float32
    noise_grid,               # (K, T, 3)         Float32
    inv_2sigma_B²::Float32,   # 1/(2·σ_B_obs²)
    inv_2sigma_F²::Float32,
    inv_2sigma_A²::Float32,
    log_norm_obs::Float32,    # sum of -0.5·log(2π σ_X²) across X
    dt::Float32, sqrt_dt::Float32,
    R::Int, K_per_chain::Int, seg_step_offset::Int,
)
    i = @index(Global, Linear)
    m_chain = ((i - 1) ÷ K_per_chain) + 1
    p_idx   = ((i - 1) % K_per_chain) + 1

    # Drift + diffusion params (v1 form, packed at construction time).
    tau_B     = params_per_chain[m_chain,  1]
    tau_F     = params_per_chain[m_chain,  2]
    kappa_B   = params_per_chain[m_chain,  3]
    kappa_F   = params_per_chain[m_chain,  4]
    epsilon_A = params_per_chain[m_chain,  5]
    lambda_A  = params_per_chain[m_chain,  6]
    mu_0      = params_per_chain[m_chain,  7]
    mu_B      = params_per_chain[m_chain,  8]
    mu_F      = params_per_chain[m_chain,  9]
    mu_FF     = params_per_chain[m_chain, 10]
    eta       = params_per_chain[m_chain, 11]
    sigma_B   = params_per_chain[m_chain, 12]
    sigma_F   = params_per_chain[m_chain, 13]
    sigma_A   = params_per_chain[m_chain, 14]

    B = particles_in[i, 1]
    F = particles_in[i, 2]
    A = particles_in[i, 3]
    log_w = log_w_inout[i]

    @inbounds for k in 1:R
        t   = seg_step_offset + k
        Φ_t = Phi_seq[t]

        # G0 drift (v1 form)
        μ  = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
        dB = kappa_B * (1f0 + epsilon_A * A) * Φ_t - B / tau_B
        dF = kappa_F * Φ_t - (1f0 + lambda_A * A) / tau_F * F
        dA = μ * A - eta * A * A * A

        # State-dep diffusion (Itô)
        sB = sigma_B * sqrt(max(B * (1f0 - B), 0f0))
        sF = sigma_F * sqrt(max(F, 0f0))
        sA = sigma_A * sqrt(max(A, 0f0))

        ξB = noise_grid[p_idx, t, 1]
        ξF = noise_grid[p_idx, t, 2]
        ξA = noise_grid[p_idx, t, 3]

        x_b = B + dt * dB + sB * sqrt_dt * ξB
        x_f = F + dt * dF + sF * sqrt_dt * ξF
        x_a = A + dt * dA + sA * sqrt_dt * ξA

        # NaN-guard + boundary reflection.
        if !isfinite(x_b) || !isfinite(x_f) || !isfinite(x_a)
            x_b = 0.05f0; x_f = 0.30f0; x_a = 0.10f0
        else
            x_b = x_b < 0f0 ? -x_b : (x_b > 1f0 ? 2f0 - x_b : x_b)
            x_f = abs(x_f)
            x_a = abs(x_a)
        end

        # Three-channel diagonal Gaussian obs log-weight (every bin observed).
        ΔB = obs_B_value[t] - x_b
        ΔF = obs_F_value[t] - x_f
        ΔA = obs_A_value[t] - x_a
        log_w += log_norm_obs -
                  inv_2sigma_B² * ΔB * ΔB -
                  inv_2sigma_F² * ΔF * ΔF -
                  inv_2sigma_A² * ΔA * ΔA

        B = x_b; F = x_f; A = x_a
    end

    particles_out[i, 1] = B
    particles_out[i, 2] = F
    particles_out[i, 3] = A
    log_w_inout[i]      = log_w
end


# ── Constrained ↔ unconstrained map (matches PARAM_PRIOR_CONFIG order) ──
# All 10 estimated params in v1.5 are LogNormal — see Estimation.PARAM_PRIOR_CONFIG.
# Unconstrained `u` ↦ constrained `exp(u)`.

@inline function _to_v15_constrained_nt(u::AbstractVector{T}) where {T<:Real}
    # PARAM_NAMES order (10 estimated):
    # 1=tau_F, 2=B_inf, 3=F_inf, 4=lambda_A, 5=mu_0, 6=mu_B, 7=mu_F,
    # 8=sigma_B, 9=sigma_F, 10=sigma_A.
    #
    # Implemented via `@match` vector destructure — maps 1:1 to the
    # Lean4 port's `match u with | #[u1, u2, ..., u10] => ...`.
    return @match u begin
        [u1, u2, u3, u4, u5, u6, u7, u8, u9, u10] => (
            tau_F    = exp(clamp(u1, -20.0, 20.0)),
            B_inf    = exp(clamp(u2, -20.0, 20.0)),
            F_inf    = exp(clamp(u3, -20.0, 20.0)),
            lambda_A  = exp(clamp(u4, -20.0, 20.0)),
            mu_0     = exp(clamp(u5, -20.0, 20.0)),
            mu_B     = exp(clamp(u6, -20.0, 20.0)),
            mu_F     = exp(clamp(u7, -20.0, 20.0)),
            sigma_B  = exp(clamp(u8, -20.0, 20.0)),
            sigma_F  = exp(clamp(u9, -20.0, 20.0)),
            sigma_A  = exp(clamp(u10, -20.0, 20.0)),
        )
    end
end

# Pack a v1-form NamedTuple (14 fields) into the 14-element row layout
# the kernel expects.
@inline function _pack_v1_row!(row::AbstractVector{Float32}, p_v1::NamedTuple)
    row[1]  = Float32(p_v1.tau_B)
    row[2]  = Float32(p_v1.tau_F)
    row[3]  = Float32(p_v1.kappa_B)
    row[4]  = Float32(p_v1.kappa_F)
    row[5]  = Float32(p_v1.epsilon_A)
    row[6]  = Float32(p_v1.lambda_A)
    row[7]  = Float32(p_v1.mu_0)
    row[8]  = Float32(p_v1.mu_B)
    row[9]  = Float32(p_v1.mu_F)
    row[10] = Float32(p_v1.mu_FF)
    row[11] = Float32(p_v1.eta)
    row[12] = Float32(p_v1.sigma_B)
    row[13] = Float32(p_v1.sigma_F)
    row[14] = Float32(p_v1.sigma_A)
end


# ── Immutable target struct ──────────────────────────────────────────────
# Holds GPU-resident buffers. Pre-allocated at construction time. The
# buffers are mutable from the GPU's perspective (the kernel writes to
# them) but no public API mutates the struct fields — same `target`
# returned for the same constructor inputs.

struct FSAGPUTarget
    K_per_chain::Int
    M_max::Int
    T_steps::Int
    R::Int
    n_params_v1::Int                    # = 14 (drift+diffusion in v1 form)
    n_states::Int                       # = 3
    dt::Float32
    sqrt_dt::Float32
    a_shrink::Float32
    ot_max_weight::Float32
    ot_threshold::Float32
    ot_temperature::Float32
    # Pinned obs-noise constants (precomputed at construction):
    inv_2sigma_B²::Float32
    inv_2sigma_F²::Float32
    inv_2sigma_A²::Float32
    log_norm_obs::Float32
    # GPU buffers (pre-allocated, written by kernels; never read by callers
    # except via the public facade functions).
    params_per_chain::CuArray{Float32, 2}      # (M_max, 14) v1-form
    Phi_seq::CuArray{Float32, 1}               # (T_steps,)
    obs_B_value::CuArray{Float32, 1}
    obs_F_value::CuArray{Float32, 1}
    obs_A_value::CuArray{Float32, 1}
    noise_grid::CuArray{Float32, 3}            # (K, T, 3) — CRN
    bufs::GPUSegmentedBuffers                  # framework: particles + log_w + ...
    propagate_kernel::Any                      # KernelAbstractions handle
end


function FSAGPUTarget(; K_per_chain::Int, M_max::Int, T_steps::Int,
                       R::Int = 4,
                       dt::Real,
                       a_shrink::Real = 0.98,
                       ot_max_weight::Real = 0.01,
                       ot_threshold_frac::Real = 0.05,
                       ot_temperature::Real = 5.0,
                       noise_seed::Int = 0)
    @assert T_steps % R == 0 "T_steps=$T_steps must be divisible by R=$R"

    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, K_per_chain, T_steps, 3)

    σ_B_obs = Float32(DEFAULT_PARAMS[:sigma_B_obs])
    σ_F_obs = Float32(DEFAULT_PARAMS[:sigma_F_obs])
    σ_A_obs = Float32(DEFAULT_PARAMS[:sigma_A_obs])
    inv_2sigma_B² = 0.5f0 / (σ_B_obs * σ_B_obs)
    inv_2sigma_F² = 0.5f0 / (σ_F_obs * σ_F_obs)
    inv_2sigma_A² = 0.5f0 / (σ_A_obs * σ_A_obs)
    log_norm_obs  = Float32(
        -0.5 * log(2π) - log(σ_B_obs) +
        -0.5 * log(2π) - log(σ_F_obs) +
        -0.5 * log(2π) - log(σ_A_obs)
    )

    n_params_v1 = 14
    n_states    = 3
    bufs = GPUSegmentedBuffers(K_per_chain, M_max, n_states)

    return FSAGPUTarget(
        K_per_chain, M_max, T_steps, R, n_params_v1, n_states,
        Float32(dt), Float32(sqrt(dt)),
        Float32(a_shrink),
        Float32(ot_max_weight),
        Float32(K_per_chain * ot_threshold_frac),
        Float32(ot_temperature),
        inv_2sigma_B², inv_2sigma_F², inv_2sigma_A², log_norm_obs,
        CUDA.zeros(Float32, M_max, n_params_v1),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CUDA.zeros(Float32, T_steps),
        CuArray(noise_cpu),
        bufs,
        propagate_segment_kernel!(CUDABackend(), 256),
    )
end


# ── Pure log-density evaluator ───────────────────────────────────────────
#
# `grid_obs` is a NamedTuple with fields:
#   :Phi_seq     :: Vector{Float32}   length T_steps  — per-bin control input
#   :obs_B       :: Vector{Float32}   length T_steps  — Gaussian obs of B
#   :obs_F       :: Vector{Float32}   length T_steps  — Gaussian obs of F
#   :obs_A       :: Vector{Float32}   length T_steps  — Gaussian obs of A
#   :B_init      :: Float64
#   :F_init      :: Float64
#   :A_init      :: Float64

"""
    gpu_log_density(target, U_unc, grid_obs, key) -> Vector{Float64}

Compute per-chain log p(y | θ) for M chains under v1.5's three-channel
Gaussian obs model. M is `size(U_unc, 1)`, must be ≤ target.M_max.

`U_unc` is a (M, 10) Matrix{Float64} in PARAM_NAMES order
(unconstrained — all 10 params are LogNormal so the constrained value
is exp(unconstrained)).

Pure: same `(target, U_unc, grid_obs, key)` always returns the same
output. No public field of `target` is mutated.
"""
function gpu_log_density(target::FSAGPUTarget,
                          U_unc::AbstractMatrix{Float64},
                          grid_obs::NamedTuple,
                          key::UInt64;
                          use_ot::Bool = true)
    M = size(U_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    @assert size(U_unc, 2) == 10 "U_unc must have 10 columns (PARAM_NAMES order)"
    K  = target.K_per_chain
    Ntot = M * K
    R  = target.R
    n_segments = target.T_steps ÷ R
    n_states = target.n_states

    # ── Pack v1-form params for each chain ───────────────────────────────
    # u ∈ R^10 → constrained v1.5 NamedTuple (10 fields)
    #         → fill_pinned_nt → 14-field v1.5 NamedTuple
    #         → params_v15_to_v1_nt → 14-field v1 NamedTuple
    #         → row in params_cpu (Float32, length 14)
    params_cpu = Matrix{Float32}(undef, M, target.n_params_v1)
    @inbounds for m in 1:M
        u_row    = view(U_unc, m, :)
        p_est    = _to_v15_constrained_nt(u_row)
        p_v15    = fill_pinned_nt(p_est)
        p_v1     = params_v15_to_v1_nt(p_v15)
        _pack_v1_row!(view(params_cpu, m, :), p_v1)
    end
    copyto!(view(target.params_per_chain, 1:M, :), params_cpu)

    # ── Stage obs onto GPU ───────────────────────────────────────────────
    copyto!(target.Phi_seq,     Float32.(grid_obs.Phi_seq))
    copyto!(target.obs_B_value, Float32.(grid_obs.obs_B))
    copyto!(target.obs_F_value, Float32.(grid_obs.obs_F))
    copyto!(target.obs_A_value, Float32.(grid_obs.obs_A))

    # ── Init particle cloud + log_w ──────────────────────────────────────
    bufs = target.bufs
    fill!(view(bufs.particles_a, 1:Ntot, 1), Float32(grid_obs.B_init))
    fill!(view(bufs.particles_a, 1:Ntot, 2), Float32(grid_obs.F_init))
    fill!(view(bufs.particles_a, 1:Ntot, 3), Float32(grid_obs.A_init))
    fill!(view(bufs.log_w, 1:Ntot), 0f0)

    log_lik_acc = zeros(Float64, M)
    log_K = log(Float32(K))

    for seg in 1:n_segments
        seg_step_offset = (seg - 1) * R

        target.propagate_kernel(
            view(bufs.log_w, 1:Ntot),
            view(bufs.particles_a, 1:Ntot, :),
            view(bufs.particles_b, 1:Ntot, :),
            view(target.params_per_chain, 1:M, :),
            target.Phi_seq,
            target.obs_B_value, target.obs_F_value, target.obs_A_value,
            target.noise_grid,
            target.inv_2sigma_B², target.inv_2sigma_F², target.inv_2sigma_A²,
            target.log_norm_obs,
            target.dt, target.sqrt_dt,
            R, K, seg_step_offset;
            ndrange = Ntot,
        )
        KernelAbstractions.synchronize(CUDABackend())

        # Pattern-match on (seg, n_segments) — the last segment skips the
        # SMC resample step (no further use of the resampled cloud) and just
        # computes per-chain stats. Maps 1:1 to the Lean4 port's
        # `match (seg, n_segments) with | (s, n) => if s == n then ... else ...`.
        @match (seg, n_segments) begin
            (s, n), if s == n end => begin
                # Last segment: stats kernel only, no resample.
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
            end
            _ => begin
                # Non-last segment: run segmented-SMC step (resample + Liu-West + OT).
                run_segmented_smc_step!(bufs, M, target.a_shrink;
                                         ot_max = target.ot_max_weight,
                                         ot_threshold = target.ot_threshold,
                                         ot_temperature = target.ot_temperature,
                                         use_ot = use_ot)
            end
        end
        # Accumulate per-chain log-likelihood (same in both branches).
        log_max_cpu = Array(view(bufs.log_max, 1:M))
        log_z_cpu   = Array(view(bufs.log_z,   1:M))
        @inbounds for m in 1:M
            log_lik_acc[m] += Float64(log_max_cpu[m] + log_z_cpu[m] - log_K)
        end
    end
    return log_lik_acc
end


# ── Pure FD-gradient batcher ─────────────────────────────────────────────

"""
    gpu_grads(target, U_unc, grid_obs, h, key) -> (vals, grads)

Central-difference gradient. M chains × (1 + 2d) perturbation rows are
packed into a single `gpu_log_density` call. Returns `vals :: Vector{Float64}`
of length M and `grads :: Matrix{Float64}` of shape (M, d). Pure.
"""
function gpu_grads(target::FSAGPUTarget,
                    U_unc::AbstractMatrix{Float64},
                    grid_obs::NamedTuple,
                    h::Float64,
                    key::UInt64;
                    use_ot::Bool = true)
    M = size(U_unc, 1); d = size(U_unc, 2)
    n_perturb = 1 + 2 * d
    n_total = M * n_perturb
    n_total ≤ target.M_max || throw(ArgumentError(
        "n_total $(n_total) > M_max $(target.M_max)"))

    U_flat = Matrix{Float64}(undef, n_total, d)
    @inbounds for m in 1:M
        base = (m - 1) * n_perturb
        U_flat[base + 1, :] = U_unc[m, :]
        for i in 1:d
            U_flat[base + 1 + 2 * (i - 1) + 1, :] = U_unc[m, :]
            U_flat[base + 1 + 2 * (i - 1) + 1, i] += h
            U_flat[base + 1 + 2 * (i - 1) + 2, :] = U_unc[m, :]
            U_flat[base + 1 + 2 * (i - 1) + 2, i] -= h
        end
    end

    lls = gpu_log_density(target, U_flat, grid_obs, key; use_ot = use_ot)

    vals  = Vector{Float64}(undef, M)
    grads = Matrix{Float64}(undef, M, d)
    @inbounds for m in 1:M
        base = (m - 1) * n_perturb
        vals[m] = lls[base + 1]
        for i in 1:d
            grads[m, i] = (lls[base + 1 + 2 * (i - 1) + 1] -
                           lls[base + 1 + 2 * (i - 1) + 2]) / (2h)
        end
    end
    return (vals = vals, grads = grads)
end


# ── Pure HMC move ────────────────────────────────────────────────────────

"""
    parallel_hmc_one_move(U, target, grid_obs, ε, L,
                          prior_means, prior_sigmas, key) -> (U_new, n_acc)

Run one parallel-chains HMC move. Returns a NEW `U_new` Matrix{Float64}
(does not mutate `U`). `n_acc` is the number of accepted chains.
"""
function parallel_hmc_one_move(U::AbstractMatrix{Float64},
                                target::FSAGPUTarget,
                                grid_obs::NamedTuple,
                                ε::Float64,
                                L::Int,
                                prior_means::AbstractVector{Float64},
                                prior_sigmas::AbstractVector{Float64},
                                key::UInt64;
                                inv_mass::AbstractVector{Float64} = ones(size(U, 2)),
                                h_fd::Float64 = 1e-3)
    M, d = size(U)
    rng = StableRNG(key)
    momentum = randn(rng, M, d) ./ sqrt.(inv_mass)'
    p0 = copy(momentum)

    function tempered_grads(U_in, sub_key::UInt64)
        out = gpu_grads(target, U_in, grid_obs, h_fd, sub_key)
        grads_prior = -(U_in .- prior_means') ./ (prior_sigmas' .^ 2)
        vals_prior  = -0.5 .* vec(sum(((U_in .- prior_means') ./ prior_sigmas') .^ 2; dims = 2))
        return (out.vals .+ vals_prior, out.grads .+ grads_prior)
    end

    val_init, grad_init = tempered_grads(U, hash((key, :leap, 0)))
    p     = momentum .+ (ε / 2) .* grad_init
    U_new = U .+ ε .* p .* inv_mass'
    for k in 2:L
        _, grad = tempered_grads(U_new, hash((key, :leap, k - 1)))
        p     = p .+ ε .* grad
        U_new = U_new .+ ε .* p .* inv_mass'
    end
    val_final, grad_final = tempered_grads(U_new, hash((key, :leap, L)))
    p = p .+ (ε / 2) .* grad_final

    K0    = 0.5 .* vec(sum(p0 .^ 2 .* inv_mass'; dims = 2))
    K_new = 0.5 .* vec(sum(p  .^ 2 .* inv_mass'; dims = 2))
    log_α = (val_final .- K_new) .- (val_init .- K0)

    # Functional accept/reject — build U_out as a fresh matrix.
    # `@match` on the Bernoulli outcome maps 1:1 to the Lean4 port's
    # `match outcome with | .accept => ... | .reject => ...`.
    U_out = copy(U)
    n_acc = 0
    @inbounds for m in 1:M
        outcome = log(rand(rng)) < log_α[m] ? :accept : :reject
        @match outcome begin
            :accept => begin
                U_out[m, :] = U_new[m, :]
                n_acc += 1
            end
            :reject => begin
                U_out[m, :] = U[m, :]
            end
        end
    end
    return (U_new = U_out, n_acc = n_acc)
end

end # module GPUPF
