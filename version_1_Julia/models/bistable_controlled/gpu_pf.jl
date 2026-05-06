# GPU bootstrap PF for bistable — reusable wrapper around the fused
# KernelAbstractions kernel from bench_gpu_bistable_fused.jl. Used by the
# GPU end-to-end SMC² bench (option C in the user's checkpoint).
#
# Public surface:
#   `BistableGPUTarget` — pre-allocated noise grids + obs + scratch
#   `gpu_log_density(target, u_unc::Vector{Float64})`           → scalar
#   `gpu_log_density_with_grad(target, u; h=1e-3)`              → (scalar, gradient)
#
# Per-call cost on RTX 5090 (K = 200_000 fp32, T = 432):
#   primal      : ~2 ms
#   FD gradient : ~32 ms (16 PF evals, central differences, 8-D θ)
# Same noise grids reused across the 16 FD evals (CRN — common random
# numbers — so the gradient signal isn't drowned by PF stochastic noise).

module BistableGPUPF

using CUDA
using KernelAbstractions
using LogExpFunctions: logsumexp
using Statistics: mean
using Random: AbstractRNG, MersenneTwister

export BistableGPUTarget, gpu_log_density, gpu_log_density_with_grad
export BistableGPUTargetBatched, gpu_log_density_batched, gpu_fd_gradient_batched
export gpu_grads_parallel_chains, parallel_hmc_one_move!

# ── Fused per-particle PF kernel (lifted from bench_gpu_bistable_fused.jl) ─

@kernel function bootstrap_pf_kernel!(
    log_w_out,         # (K,)
    x_init, u_init,
    α, a_param, γ,
    sx_sd, su_sd, sigma_obs,
    dt, sqrt_dt,
    T_i_threshold, u_on,
    obs_seq,           # (T,)
    noise_x,           # (K, T+1)
    noise_u,           # (K, T+1)
    T_steps::Int,
)
    i = @index(Global, Linear)

    x = x_init + sx_sd * sqrt_dt * noise_x[i, 1]
    u = u_init + su_sd * sqrt_dt * noise_u[i, 1]

    log_w = zero(eltype(log_w_out))
    half_log_2pi = oftype(log_w, 0.9189385332046727)

    @inbounds for k in 1:T_steps
        t = (k - 1) * dt
        u_tgt = t < T_i_threshold ? zero(t) : u_on

        drift_x = α * x * (a_param * a_param - x * x) + u
        drift_u = -γ * (u - u_tgt)

        x = x + dt * drift_x + sx_sd * sqrt_dt * noise_x[i, k + 1]
        u = u + dt * drift_u + su_sd * sqrt_dt * noise_u[i, k + 1]

        ν = obs_seq[k] - x
        log_w += -oftype(log_w, 0.5) * (ν / sigma_obs)^2 -
                  log(sigma_obs) - half_log_2pi
    end

    log_w_out[i] = log_w
end


"""
    BistableGPUTarget(; K, T_steps, dt, T_intervention, u_on,
                         x_init, u_init, obs_seq_cpu)

Pre-allocates everything the GPU PF needs:
  - `noise_x`, `noise_u`         (K × (T+1) Float32 CuArray) — drawn ONCE,
                                   reused across every log-density call so
                                   FD gradients use common random numbers
  - `log_w_out`                  (K,)       Float32 CuArray
  - `obs_gpu`                    (T,)       Float32 CuArray
  - `kernel`                     compiled KernelAbstractions kernel

Fields are all internal — call `gpu_log_density(target, u_unc)` to use it.
"""
mutable struct BistableGPUTarget
    K::Int
    T_steps::Int
    dt::Float32
    sqrt_dt::Float32
    T_i_threshold::Float32
    u_on::Float32
    x_init::Float32
    u_init::Float32
    obs_gpu::CuArray{Float32,1}
    noise_x::CuArray{Float32,2}
    noise_u::CuArray{Float32,2}
    log_w_out::CuArray{Float32,1}
    kernel::Any   # KernelAbstractions kernel object
end

function BistableGPUTarget(; K::Int, T_steps::Int,
                            dt::Real, T_intervention::Real, u_on::Real,
                            x_init::Real, u_init::Real,
                            obs_seq_cpu::AbstractVector{Float64},
                            noise_seed::Int = 0)
    # Pre-generate noise grids on CPU (deterministic) then upload once.
    rng = Random.MersenneTwister(noise_seed)
    noise_x_cpu = Float32.(randn(rng, K, T_steps + 1))
    noise_u_cpu = Float32.(randn(rng, K, T_steps + 1))

    obs_gpu     = CuArray{Float32}(obs_seq_cpu)
    noise_x_gpu = CuArray(noise_x_cpu)
    noise_u_gpu = CuArray(noise_u_cpu)
    log_w_out   = CUDA.zeros(Float32, K)

    backend = CUDABackend()
    kernel  = bootstrap_pf_kernel!(backend, 256)

    return BistableGPUTarget(
        K, T_steps,
        Float32(dt), Float32(sqrt(dt)),
        Float32(T_intervention), Float32(u_on),
        Float32(x_init), Float32(u_init),
        obs_gpu, noise_x_gpu, noise_u_gpu, log_w_out, kernel,
    )
end

using Random: MersenneTwister
import Random


"""
    gpu_log_density(target::BistableGPUTarget, u_unc::AbstractVector{Float64})
        -> scalar Float64

Evaluate log p(y | θ) on the GPU. `u_unc` is the 8-D unconstrained
parameter vector [log α, log a, log σ_x, log γ, log σ_u, log σ_obs,
x_0_init_unused, u_0_init_unused].

Prior log-density is NOT included — return is the data log-likelihood
only (the SMC² outer loop adds the prior separately).
"""
function gpu_log_density(target::BistableGPUTarget, u_unc::AbstractVector{Float64})
    α       = Float32(exp(u_unc[1]))
    a       = Float32(exp(u_unc[2]))
    σ_x     = Float32(exp(u_unc[3]))
    γ       = Float32(exp(u_unc[4]))
    σ_u     = Float32(exp(u_unc[5]))
    σ_obs   = Float32(exp(u_unc[6]))
    sx_sd   = sqrt(2.0f0 * σ_x)
    su_sd   = sqrt(2.0f0 * σ_u)

    target.kernel(
        target.log_w_out,
        target.x_init, target.u_init,
        α, a, γ,
        sx_sd, su_sd, σ_obs,
        target.dt, target.sqrt_dt,
        target.T_i_threshold, target.u_on,
        target.obs_gpu, target.noise_x, target.noise_u,
        target.T_steps;
        ndrange = target.K,
    )
    KernelAbstractions.synchronize(CUDABackend())

    # logsumexp - log K — small reduction; bring weights to CPU once.
    log_w_cpu = Array(target.log_w_out)
    return Float64(logsumexp(log_w_cpu) - log(Float32(target.K)))
end


"""
    gpu_log_density_with_grad(target, u_unc; h=1e-3) -> (Float64, Vector{Float64})

Central finite-difference gradient: 16 PF evals (8 dims × ±h) + 1 primal
= 17 evals total. Same noise grids reused across all 17 → CRN. Returns
`(log p, ∇ log p)`.

Step `h = 1e-3` works well for our 8-D θ; the per-parameter scale is
~O(1) in unconstrained (log-of-truth) space.
"""
function gpu_log_density_with_grad(target::BistableGPUTarget,
                                     u_unc::AbstractVector{Float64};
                                     h::Float64 = 1e-3)
    d = length(u_unc)
    primal = gpu_log_density(target, u_unc)
    g = zeros(Float64, d)
    for i in 1:d
        u_p     = copy(u_unc); u_p[i] += h
        u_m     = copy(u_unc); u_m[i] -= h
        ll_p    = gpu_log_density(target, u_p)
        ll_m    = gpu_log_density(target, u_m)
        g[i]    = (ll_p - ll_m) / (2h)
    end
    return primal, g
end

# ── Batched kernel: M chains in parallel ───────────────────────────────────
#
# Each thread handles one (chain, particle) pair. Chains read their own
# (α, a, γ, σ_x, σ_u, σ_obs) from per-chain arrays. With M = 64 chains
# and K_per_chain = 5_000 particles, total threads = 320_000 — single
# kernel launch, saturates the SMs, replaces 64 sequential calls.
#
# Memory layout:
#   noise_x, noise_u : (M·K_per_chain, T+1)   shared across all chains
#                     OR per-chain if you want independent CRN streams
#   log_w_out        : (M·K_per_chain,)        flattened (chain m at
#                                              indices (m-1)*K+1 : m*K)
#   α_arr, a_arr, …  : (M,)                    per-chain parameter scalars

@kernel function bootstrap_pf_kernel_batched!(
    log_w_out,             # (M·K_per_chain,)
    x_init, u_init,
    α_arr, a_arr, γ_arr,
    sx_sd_arr, su_sd_arr, sigma_obs_arr,
    dt, sqrt_dt,
    T_i_threshold, u_on,
    obs_seq,
    noise_x,               # (K_per_chain, T+1)  — SHARED across chains (CRN)
    noise_u,
    T_steps::Int,
    K_per_chain::Int,
)
    i = @index(Global, Linear)
    # Decode (chain, particle-in-chain) from flat thread index.
    # Critical: noise indices use particle-in-chain only, NOT chain index,
    # so chain m and chain m' both see the same noise[p,:] for particle p.
    # That's the Common-Random-Numbers property FD gradient needs.
    m_chain          = ((i - 1) ÷ K_per_chain) + 1
    particle_in_chain = ((i - 1) % K_per_chain) + 1

    α       = α_arr[m_chain]
    a_param = a_arr[m_chain]
    γ       = γ_arr[m_chain]
    sx_sd   = sx_sd_arr[m_chain]
    su_sd   = su_sd_arr[m_chain]
    sigma_obs = sigma_obs_arr[m_chain]

    x = x_init + sx_sd * sqrt_dt * noise_x[particle_in_chain, 1]
    u = u_init + su_sd * sqrt_dt * noise_u[particle_in_chain, 1]

    log_w = zero(eltype(log_w_out))
    half_log_2pi = oftype(log_w, 0.9189385332046727)

    @inbounds for k in 1:T_steps
        t = (k - 1) * dt
        u_tgt = t < T_i_threshold ? zero(t) : u_on
        drift_x = α * x * (a_param * a_param - x * x) + u
        drift_u = -γ * (u - u_tgt)
        x = x + dt * drift_x + sx_sd * sqrt_dt * noise_x[particle_in_chain, k + 1]
        u = u + dt * drift_u + su_sd * sqrt_dt * noise_u[particle_in_chain, k + 1]
        ν = obs_seq[k] - x
        log_w += -oftype(log_w, 0.5) * (ν / sigma_obs)^2 -
                  log(sigma_obs) - half_log_2pi
    end

    log_w_out[i] = log_w
end


"""
    BistableGPUTargetBatched(; K_per_chain, M_max, T_steps, dt, T_intervention,
                                u_on, x_init, u_init, obs_seq_cpu, noise_seed)

Batched variant of `BistableGPUTarget`. Allocates noise grids of size
`(M_max · K_per_chain, T+1)` so up to `M_max` chains can be evaluated
in ONE kernel launch. Per-call you supply a `Matrix(M, 8)` of θ values
where `M ≤ M_max`; only the first `M·K_per_chain` rows of the noise
grids are used.
"""
mutable struct BistableGPUTargetBatched
    K_per_chain::Int
    M_max::Int
    T_steps::Int
    dt::Float32
    sqrt_dt::Float32
    T_i_threshold::Float32
    u_on::Float32
    x_init::Float32
    u_init::Float32
    obs_gpu::CuArray{Float32,1}
    noise_x::CuArray{Float32,2}    # (M_max·K_per_chain, T+1)
    noise_u::CuArray{Float32,2}
    log_w_out::CuArray{Float32,1}  # (M_max·K_per_chain,)
    α_arr::CuArray{Float32,1}      # (M_max,) scratch
    a_arr::CuArray{Float32,1}
    γ_arr::CuArray{Float32,1}
    sx_sd_arr::CuArray{Float32,1}
    su_sd_arr::CuArray{Float32,1}
    sigma_obs_arr::CuArray{Float32,1}
    kernel::Any
end

function BistableGPUTargetBatched(; K_per_chain::Int, M_max::Int, T_steps::Int,
                                     dt::Real, T_intervention::Real, u_on::Real,
                                     x_init::Real, u_init::Real,
                                     obs_seq_cpu::AbstractVector{Float64},
                                     noise_seed::Int = 0)
    rng = Random.MersenneTwister(noise_seed)
    # CRN noise grids are (K_per_chain, T+1) — SHARED across chains so all
    # M chains see the same noise per particle index. ~M× memory saving
    # vs the per-chain layout AND restores the FD-gradient correctness
    # property the chains otherwise lose.
    noise_x_cpu = Float32.(randn(rng, K_per_chain, T_steps + 1))
    noise_u_cpu = Float32.(randn(rng, K_per_chain, T_steps + 1))
    obs_gpu     = CuArray{Float32}(obs_seq_cpu)
    noise_x_gpu = CuArray(noise_x_cpu)
    noise_u_gpu = CuArray(noise_u_cpu)
    log_w_out   = CUDA.zeros(Float32, K_per_chain * M_max)
    α_arr       = CUDA.zeros(Float32, M_max)
    a_arr       = CUDA.zeros(Float32, M_max)
    γ_arr       = CUDA.zeros(Float32, M_max)
    sx_sd_arr   = CUDA.zeros(Float32, M_max)
    su_sd_arr   = CUDA.zeros(Float32, M_max)
    sigma_obs_arr = CUDA.zeros(Float32, M_max)
    backend     = CUDABackend()
    kernel      = bootstrap_pf_kernel_batched!(backend, 256)
    return BistableGPUTargetBatched(
        K_per_chain, M_max, T_steps,
        Float32(dt), Float32(sqrt(dt)),
        Float32(T_intervention), Float32(u_on),
        Float32(x_init), Float32(u_init),
        obs_gpu, noise_x_gpu, noise_u_gpu, log_w_out,
        α_arr, a_arr, γ_arr, sx_sd_arr, su_sd_arr, sigma_obs_arr,
        kernel,
    )
end


"""
    gpu_log_density_batched(target, U_unc::AbstractMatrix{Float64}) -> Vector{Float64}

Evaluate log p(y | θ_m) at M different θ values **in one kernel launch**.
`U_unc` is `(M, 8)` — each row is one θ. Returns `(M,)` vector of log-liks.

Throws if `M > target.M_max`.
"""
function gpu_log_density_batched(target::BistableGPUTargetBatched,
                                   U_unc::AbstractMatrix{Float64})
    M = size(U_unc, 1)
    M ≤ target.M_max || throw(ArgumentError(
        "M = $M exceeds target.M_max = $(target.M_max)"))

    # Pack per-chain params
    α_cpu     = Float32.(exp.(@view U_unc[:, 1]))
    a_cpu     = Float32.(exp.(@view U_unc[:, 2]))
    σ_x_cpu   = Float32.(exp.(@view U_unc[:, 3]))
    γ_cpu     = Float32.(exp.(@view U_unc[:, 4]))
    σ_u_cpu   = Float32.(exp.(@view U_unc[:, 5]))
    σ_obs_cpu = Float32.(exp.(@view U_unc[:, 6]))
    sx_sd_cpu = sqrt.(2.0f0 .* σ_x_cpu)
    su_sd_cpu = sqrt.(2.0f0 .* σ_u_cpu)

    # Upload to per-chain device arrays (reuse pre-allocated slots).
    copyto!(@view(target.α_arr[1:M]),     α_cpu)
    copyto!(@view(target.a_arr[1:M]),     a_cpu)
    copyto!(@view(target.γ_arr[1:M]),     γ_cpu)
    copyto!(@view(target.sx_sd_arr[1:M]), sx_sd_cpu)
    copyto!(@view(target.su_sd_arr[1:M]), su_sd_cpu)
    copyto!(@view(target.sigma_obs_arr[1:M]), σ_obs_cpu)

    Ntot = M * target.K_per_chain
    target.kernel(
        target.log_w_out,
        target.x_init, target.u_init,
        target.α_arr, target.a_arr, target.γ_arr,
        target.sx_sd_arr, target.su_sd_arr, target.sigma_obs_arr,
        target.dt, target.sqrt_dt,
        target.T_i_threshold, target.u_on,
        target.obs_gpu, target.noise_x, target.noise_u,
        target.T_steps, target.K_per_chain;
        ndrange = Ntot,
    )
    KernelAbstractions.synchronize(CUDABackend())

    # Per-chain logsumexp - log K_per_chain. Brings (M·K) weights to CPU
    # and reduces — for M·K up to ~10⁶ the transfer is ~4 MB, tens of µs.
    log_w_cpu = Array(target.log_w_out)[1:Ntot]
    log_w_mat = reshape(log_w_cpu, target.K_per_chain, M)
    log_K     = log(Float32(target.K_per_chain))
    out = Vector{Float64}(undef, M)
    for m in 1:M
        out[m] = Float64(logsumexp(@view log_w_mat[:, m]) - log_K)
    end
    return out
end


"""
    gpu_fd_gradient_batched(target, u_unc::AbstractVector{Float64};
                              h = 1e-3) -> (val::Float64, grad::Vector{Float64})

Finite-difference gradient via ONE batched kernel call.
Builds a (1 + 2d, 8) θ matrix [primal; +h·e_1; -h·e_1; …; +h·e_d; -h·e_d],
runs them all in one launch, computes central differences. For d=8 this
is 17 chains × K_per_chain particles in one launch — typically saturates
the SMs at K_per_chain ≥ 5_000.
"""
function gpu_fd_gradient_batched(target::BistableGPUTargetBatched,
                                   u_unc::AbstractVector{Float64};
                                   h::Float64 = 1e-3)
    d = length(u_unc)
    n_chains = 1 + 2 * d   # primal + (±h) per dim
    n_chains ≤ target.M_max || throw(ArgumentError(
        "n_chains $(n_chains) > M_max $(target.M_max). Build target with bigger M_max."))

    U = Matrix{Float64}(undef, n_chains, d)
    U[1, :] = u_unc
    for i in 1:d
        U[1 + 2*(i-1) + 1, :] = u_unc; U[1 + 2*(i-1) + 1, i] += h
        U[1 + 2*(i-1) + 2, :] = u_unc; U[1 + 2*(i-1) + 2, i] -= h
    end

    lls = gpu_log_density_batched(target, U)
    val = lls[1]
    g = zeros(Float64, d)
    for i in 1:d
        g[i] = (lls[1 + 2*(i-1) + 1] - lls[1 + 2*(i-1) + 2]) / (2h)
    end
    return val, g
end

# ── Parallel-chains gradient: M chains × (1 + 2d) FD perturbations in
#    ONE kernel launch ───────────────────────────────────────────────────────

"""
    gpu_grads_parallel_chains(target, U_unc::AbstractMatrix{Float64}; h=1e-3)
        -> (vals::Vector{Float64}, grads::Matrix{Float64})

Compute log-density value AND finite-difference gradient at M different
θ values in ONE kernel launch. `U_unc` is `(M, d)` — each row is one θ.

Layout of the batched call: M chains × (1 + 2d) perturbations =
M·(1+2d) chains in one launch. With M = 64 and d = 8, that's 1088 chains
× K_per_chain particles in a single GPU dispatch — saturates the SMs
per leapfrog step and replaces M×(1+2d) sequential calls.

Returns:
  vals  — `(M,)` primal log-densities (one per chain)
  grads — `(M, d)` FD gradients

Throws if `M·(1+2d) > target.M_max`.
"""
function gpu_grads_parallel_chains(target::BistableGPUTargetBatched,
                                     U_unc::AbstractMatrix{Float64};
                                     h::Float64 = 1e-3)
    M = size(U_unc, 1)
    d = size(U_unc, 2)
    n_perturb_per_chain = 1 + 2 * d
    n_total = M * n_perturb_per_chain
    n_total ≤ target.M_max || throw(ArgumentError(
        "n_total $(n_total) > M_max $(target.M_max). Build target with bigger M_max."))

    # Build the (n_total, d) flat θ matrix:
    #   chain m's perturbations occupy rows (m-1)*(1+2d)+1 .. m*(1+2d)
    #   row 1 of each block: primal
    #   rows 2..1+2d: ±h on dim 1, ±h on dim 2, …
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

    lls_flat = gpu_log_density_batched(target, U_flat)

    # Decode per-chain primal + FD gradient
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


"""
    parallel_hmc_one_move!(U::Matrix{Float64}, target, ε, L, prior_mean, prior_sigma, rng;
                            inv_mass = ones(d))

Take ONE HMC move on each of M chains in parallel. `U` is `(M, d)` —
positions for each chain. Mutates U in place; rejected moves keep the
old position. Returns the number of accepted moves.

Each leapfrog step does ONE batched kernel launch over M·(1+2d) sub-chains.
Total kernel launches per move: L (one per leapfrog step).
"""
function parallel_hmc_one_move!(U::AbstractMatrix{Float64},
                                  target::BistableGPUTargetBatched,
                                  ε::Float64,
                                  L::Int,
                                  prior_mean::AbstractVector{Float64},
                                  prior_sigma::AbstractVector{Float64},
                                  rng::AbstractRNG;
                                  inv_mass::AbstractVector{Float64} = ones(size(U, 2)))
    M, d = size(U)
    momentum = randn(rng, M, d) ./ sqrt.(inv_mass)'   # sample p ~ N(0, M)

    # Save initial state for MH accept
    U0  = copy(U)
    p0  = copy(momentum)

    # Initial gradient + value (for both step 0 and final accept check)
    function tempered_grads(U_in)
        # log p(y|θ) + log p(θ)
        vals_data, grads_data = gpu_grads_parallel_chains(target, U_in)
        # add prior: -0.5 sum((u-μ)/σ)² and grad -(u-μ)/σ²
        grads_prior = -(U_in .- prior_mean') ./ (prior_sigma' .^ 2)
        vals_prior  = -0.5 .* vec(sum(((U_in .- prior_mean') ./ prior_sigma') .^ 2; dims = 2))
        return vals_data .+ vals_prior, grads_data .+ grads_prior
    end

    val_init, grad_init = tempered_grads(U)

    # Half-step momentum
    p = momentum .+ (ε / 2) .* grad_init
    # Full position step
    U_new = U .+ ε .* p .* inv_mass'

    # L-1 inner full leapfrog steps
    for _ in 2:L
        _, grad = tempered_grads(U_new)
        p .= p .+ ε .* grad
        U_new .= U_new .+ ε .* p .* inv_mass'
    end

    # Final half-step momentum
    val_final, grad_final = tempered_grads(U_new)
    p .= p .+ (ε / 2) .* grad_final

    # MH accept per chain
    K0     = 0.5 .* vec(sum(p0       .^ 2 .* inv_mass'; dims = 2))
    K_new  = 0.5 .* vec(sum(p        .^ 2 .* inv_mass'; dims = 2))
    log_α  = (val_final .- K_new) .- (val_init .- K0)

    n_acc = 0
    @inbounds for m in 1:M
        if log(rand(rng)) < log_α[m]
            U[m, :] = U_new[m, :]
            n_acc += 1
        end
        # else: keep U[m, :] (already U0 unchanged in this method since we
        # mutated U_new not U)
    end
    return n_acc
end

end # module BistableGPUPF
