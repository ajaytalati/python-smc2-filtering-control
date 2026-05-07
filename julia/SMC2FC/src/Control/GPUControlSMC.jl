# Control/GPUControlSMC.jl — generic GPU parallel-chains tempered SMC²
# for the controller side. Model-independent.
#
# Mirrors the pattern from the filter side (Filtering/GPUSegmentedPF.jl):
# the framework provides the SMC² outer loop, parallel-chains HMC, ChEES-L
# picker, and FD-gradient batcher. A model supplies ONLY:
#
#   - a `cost_log_density_fn :: Matrix{Float64} -> Vector{Float64}`
#     mapping (M, d) θ matrix to per-chain log p(data|θ) = -cost(θ)
#
# Everything else (tempering, resampling, HMC, ChEES) is model-agnostic.

module GPUControlSMC

using LogExpFunctions: logsumexp
using Statistics: mean, std
using Random: AbstractRNG, MersenneTwister
using LinearAlgebra

export run_tempered_smc_gpu, parallel_hmc_one_move_generic!
export gpu_grads_parallel_chains_fd, chees_pick_L_generic


# ── Generic FD gradient via parallel-chains batched call ─────────────────

"""
    gpu_grads_parallel_chains_fd(log_density_fn, U_unc, M_max; h=1e-4)
        -> (vals::Vector{Float64}, grads::Matrix{Float64})

Compute per-chain log-density value and central-FD gradient at M chains
in parallel. Uses (1+2d) chains per primal — caller's `log_density_fn`
must support up to `M·(1+2d) ≤ M_max` chains in one call.

`log_density_fn :: Matrix{Float64} -> Vector{Float64}`.
"""
function gpu_grads_parallel_chains_fd(log_density_fn::Function,
                                       U_unc::AbstractMatrix{Float64},
                                       M_max::Int;
                                       h::Float64 = 1e-4)
    M, d = size(U_unc)
    n_perturb_per_chain = 1 + 2 * d
    n_total = M * n_perturb_per_chain
    n_total ≤ M_max || throw(ArgumentError(
        "n_total=$n_total > M_max=$M_max"))

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

    lls_flat = log_density_fn(U_flat)

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


# ── Generic parallel-chains HMC move with Gaussian prior ─────────────────

"""
    parallel_hmc_one_move_generic!(U, log_density_fn, M_max, ε, L,
                                     prior_mean, prior_sigma, rng;
                                     beta=1.0, h_fd=1e-4, inv_mass=ones(d))

One HMC move on each of M chains in parallel, against the tempered density
   log_q(θ) = log_prior(θ) + β · log_density_fn(θ),
with Gaussian prior `θ ~ N(prior_mean, diag(prior_sigma²))`.

`log_density_fn` is the model's batched log-density (e.g. -cost for control,
log-likelihood for filter).

`prior_mean` may be a scalar or a length-d vector. Returns # accepted moves.
"""
function parallel_hmc_one_move_generic!(U::AbstractMatrix{Float64},
                                          log_density_fn::Function,
                                          M_max::Int,
                                          ε::Float64,
                                          L::Int,
                                          prior_mean,
                                          prior_sigma,
                                          rng::AbstractRNG;
                                          beta::Float64 = 1.0,
                                          h_fd::Float64 = 1e-4,
                                          inv_mass::AbstractVector{Float64} = ones(size(U, 2)))
    M, d = size(U)
    momentum = randn(rng, M, d) ./ sqrt.(inv_mass)'
    p0 = copy(momentum)

    pmu = prior_mean isa Number ? fill(Float64(prior_mean), d) : Float64.(prior_mean)
    psg = prior_sigma isa Number ? fill(Float64(prior_sigma), d) : Float64.(prior_sigma)

    function tempered_grads(U_in)
        vals_data, grads_data = gpu_grads_parallel_chains_fd(
            log_density_fn, U_in, M_max; h=h_fd)
        # Diagonal Gaussian prior gradient
        diff = U_in .- pmu'
        grads_prior = -diff ./ (psg' .^ 2)
        vals_prior  = -0.5 .* vec(sum((diff ./ psg') .^ 2; dims = 2))
        return (vals_prior .+ beta .* vals_data),
               (grads_prior .+ beta .* grads_data)
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


# ── Generic ChEES-L picker ───────────────────────────────────────────────

"""
    chees_pick_L_generic(U_subset, log_density_fn, M_max, ε, L_candidates,
                          prior_mean, prior_sigma, rng; beta=1.0)
        -> (best_L::Int, best_score::Float64)

Run ONE HMC move at each candidate L on a small M' × d subset and pick L
that maximises expected squared jumped distance per ε·L.
"""
function chees_pick_L_generic(U_subset::AbstractMatrix{Float64},
                                log_density_fn::Function,
                                M_max::Int,
                                ε::Float64,
                                L_candidates::AbstractVector{<:Integer},
                                prior_mean, prior_sigma,
                                rng::AbstractRNG;
                                beta::Float64 = 1.0,
                                h_fd::Float64 = 1e-4)
    best_L = first(L_candidates)
    best_score = -Inf
    M_sub, _ = size(U_subset)
    for L in L_candidates
        U_try = copy(U_subset)
        parallel_hmc_one_move_generic!(U_try, log_density_fn, M_max,
                                         ε, L, prior_mean, prior_sigma, rng;
                                         beta=beta, h_fd=h_fd)
        sqd = sum(abs2, U_try .- U_subset)
        score = sqd / (M_sub * L * ε)
        if score > best_score
            best_score = score; best_L = L
        end
    end
    return best_L, best_score
end


# ── Generic outer tempered SMC² loop for the controller ─────────────────

"""
    run_tempered_smc_gpu(log_density_fn, M_max, n_smc, theta_dim,
                          prior_mean, prior_sigma, rng;
                          target_nats=8.0,
                          target_ess_frac=0.5,
                          max_lambda_inc=0.20,
                          max_temp_levels=30,
                          num_mcmc_steps=10,
                          hmc_step_size=0.2,
                          hmc_num_leapfrog=16,
                          chees_L_candidates=[16, 32, 64, 128, 256],
                          h_fd=1e-4,
                          calib_n=64,
                          verbose=true)
        -> (U_post::Matrix, n_temp::Int, β_max::Float64)

Run the parallel-chains tempered SMC² controller. `log_density_fn` is the
model's batched log-density (e.g. -cost(θ)). Mirrors the bistable B3 GPU
SMC² pattern.
"""
function run_tempered_smc_gpu(log_density_fn::Function,
                                M_max::Int,
                                n_smc::Int,
                                theta_dim::Int,
                                prior_mean,
                                prior_sigma,
                                rng::AbstractRNG;
                                target_nats::Real = 8.0,
                                target_ess_frac::Real = 0.5,
                                max_lambda_inc::Real = 0.20,
                                max_temp_levels::Int = 30,
                                num_mcmc_steps::Int = 10,
                                hmc_step_size::Real = 0.2,
                                hmc_num_leapfrog::Int = 16,
                                chees_L_candidates::AbstractVector{<:Integer} = [16, 32, 64, 128, 256],
                                h_fd::Real = 1e-4,
                                calib_n::Int = 64,
                                init_particles::Union{Nothing,AbstractArray{<:Real}} = nothing,
                                init_jitter::Real = 0.0,
                                verbose::Bool = true)
    pmu = prior_mean isa Number ? fill(Float64(prior_mean), theta_dim) : Float64.(prior_mean)
    psg = prior_sigma isa Number ? fill(Float64(prior_sigma), theta_dim) : Float64.(prior_sigma)

    # Step 1: auto-calibrate β_max from prior cost spread.
    U_calib = pmu' .+ psg' .* randn(rng, calib_n, theta_dim)
    ll_calib = log_density_fn(U_calib)
    cost_std = std(-ll_calib)
    β_max = Float64(target_nats) / max(Float64(cost_std), 1e-6)

    # Step 2: initial cloud — either from the prior or from a user-supplied
    # seed (with optional Gaussian jitter for cloud diversity).
    U = if init_particles === nothing
        pmu' .+ psg' .* randn(rng, n_smc, theta_dim)
    else
        # Broadcast init_particles to (n_smc, theta_dim). Accepts a single
        # row (1, theta_dim), a column vector (theta_dim,), or a full matrix.
        seed = if size(init_particles) == (n_smc, theta_dim)
            Float64.(init_particles)
        elseif length(init_particles) == theta_dim
            repeat(reshape(Float64.(init_particles), 1, :), n_smc, 1)
        else
            throw(ArgumentError("init_particles shape mismatch"))
        end
        seed .+ Float64(init_jitter) .* randn(rng, n_smc, theta_dim)
    end

    # Step 3: outer adaptive-tempering loop.
    β_curr = 0.0
    n_temp = 0
    L_used = hmc_num_leapfrog
    while β_curr < β_max - 1e-6
        ll = log_density_fn(U)
        δβ_max = min(β_max - β_curr, β_max * max_lambda_inc)
        target_ess = target_ess_frac * n_smc
        function ess_at(δ)
            log_w = δ .* ll
            log_wn = log_w .- logsumexp(log_w)
            return exp(-logsumexp(2.0 .* log_wn))
        end
        δβ = if ess_at(δβ_max) >= target_ess
            δβ_max
        else
            lo, hi = 0.0, δβ_max
            for _ in 1:30
                mid = 0.5 * (lo + hi)
                if ess_at(mid) > target_ess; lo = mid; else; hi = mid; end
            end
            lo
        end
        next_β = (β_curr + δβ < β_max - 1e-6) ? β_curr + δβ : β_max
        Δβ = next_β - β_curr

        # Reweight + systematic resample.
        log_w = Δβ .* ll
        w = exp.(log_w .- logsumexp(log_w))
        cumsum_w = cumsum(w)
        indices = Vector{Int}(undef, n_smc)
        u_shift = rand(rng) / n_smc
        for i in 1:n_smc
            t = (i - 1) / n_smc + u_shift
            indices[i] = clamp(searchsortedfirst(cumsum_w, t), 1, n_smc)
        end
        U = U[indices, :]

        # ChEES-L adapt on a small subset.
        chees_n = min(8, n_smc)
        chees_subset = U[1:chees_n, :]
        L_used, _ = chees_pick_L_generic(chees_subset, log_density_fn, M_max,
                                           Float64(hmc_step_size),
                                           chees_L_candidates,
                                           pmu, psg,
                                           MersenneTwister(rand(rng, UInt32));
                                           beta=next_β, h_fd=Float64(h_fd))

        # HMC moves at temperature next_β.
        n_acc = 0
        for _ in 1:num_mcmc_steps
            n_acc += parallel_hmc_one_move_generic!(U, log_density_fn, M_max,
                                                      Float64(hmc_step_size),
                                                      L_used, pmu, psg, rng;
                                                      beta=next_β,
                                                      h_fd=Float64(h_fd))
        end
        β_curr = next_β
        n_temp += 1
        if verbose
            accept_frac = n_acc / (num_mcmc_steps * n_smc)
            @info "  [ctrl-tempering $n_temp] β=$(round(β_curr, digits=3))/$(round(β_max, digits=3))  L=$L_used  accept=$(round(100*accept_frac, digits=0))%"
        end
        n_temp >= max_temp_levels && break
    end
    return U, n_temp, β_max
end


end # module GPUControlSMC
