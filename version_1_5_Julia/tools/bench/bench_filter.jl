# bench/bench_filter.jl
#
# Outer-SMC² filter implementation used by the bench
# `tools/bench_smc_full_mpc_fsa_gpu.jl`.
#
# Public functions (verbatim from the prior in-line definitions):
#   - run_outer_smc(target, grid_obs, n_smc, cfg, U_init, prior_means,
#                    prior_sigmas, key) -> (U_post, n_temp)
#       Tempered SMC² over θ with adaptive λ-ladder. Runs the framework's
#       generic ChEES HMC at each tempering level (β = next_λ). Optional
#       between-window Gaussian bridge and per-level Liu–West /
#       Silverman+KDE rejuvenation, gated by `cfg`.
#   - extract_xhat(target, M) -> SVector{3, Float64}
#       Mean state-vector across the M chains' inner-PF posterior; reads
#       `target.bufs.mu_per_chain` populated by the final
#       `gpu_log_density` call inside `run_outer_smc`.
#
# Extracted 2026-05-09 as Phase 1c of the bench refactor described in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
# Verbatim move; no signature or behaviour change.
#
# Dependencies (all loaded by the calling bench script before this file
# is `include`d): `Statistics.mean/std`, `LogExpFunctions.logsumexp`,
# `Random.MersenneTwister`, `StableRNGs.StableRNG`, `StaticArrays.SVector`,
# `Printf.@sprintf`, the FSA GPU kernels (`gpu_log_density`,
# `FSAGPUTarget`), and the framework's `parallel_hmc_one_move_generic!`,
# `chees_pick_L_generic`, `fit_gaussian`, `sample_from_gaussian`,
# `silverman_bandwidth`, `log_kernel_matrix`.


# ── Pure outer SMC² — tempered + parallel HMC ─────────────────────────────
#
# Mirrors v2's run_outer_smc_gpu but written in functional form:
#   - input particles `U_init` are an immutable Matrix{Float64}
#   - returns a NEW (U_post, n_temp) tuple; `U_init` is not mutated
#   - RNG via explicit `key::UInt64`

function run_outer_smc(target::FSAGPUTarget,
                       grid_obs::NamedTuple,
                       n_smc::Int,
                       cfg::NamedTuple,
                       U_init::Union{Nothing, Matrix{Float64}},
                       prior_means::Vector{Float64},
                       prior_sigmas::Vector{Float64},
                       key::UInt64)
    d = length(prior_means)
    rng = StableRNG(key)
    U = if U_init === nothing
        # Cold-start from prior
        let buf = Matrix{Float64}(undef, n_smc, d)
            for m in 1:n_smc, j in 1:d
                buf[m, j] = prior_means[j] + prior_sigmas[j] * randn(rng)
            end
            buf
        end
    elseif cfg.gaussian_bridge
        # Multivariate-Gaussian bridge between rolling windows: replace
        # the identity-copy of the previous posterior with a sample from
        # N(μ, Σ̂_reg) via Bridge.fit_gaussian / sample_from_gaussian.
        # `Σ̂_reg` is the sample covariance + 1e-6 Tikhonov regularisation
        # (Cholesky-stable for near-degenerate clouds). Cost: O(n_smc·d²)
        # for cov + O(d³) Cholesky + O(n_smc·d²) draw.
        # See [`SMC2FC_functional/src/SMC2/Bridge.jl:46-80`].
        μ, Σ = fit_gaussian(U_init)
        sample_from_gaussian(rng, μ, Σ, n_smc)
    else
        copy(U_init)
    end

    λ = 0.0
    n_temp = 0
    while λ < 1.0 - 1e-6
        ll_data = gpu_log_density(target, U, grid_obs, hash((key, :ll, n_temp)))

        δ_max = min(1.0 - λ, cfg.max_lambda_inc)
        target_ess = cfg.target_ess_frac * n_smc
        ess_at(δ) = begin
            log_w = δ .* ll_data
            log_w_n = log_w .- logsumexp(log_w)
            return exp(-logsumexp(2.0 .* log_w_n))
        end
        δ = if ess_at(δ_max) >= target_ess
            δ_max
        else
            lo, hi = 0.0, δ_max
            for _ in 1:30
                mid = 0.5 * (lo + hi)
                if ess_at(mid) > target_ess
                    lo = mid
                else
                    hi = mid
                end
            end
            lo
        end
        next_λ = (λ + δ < 1.0 - 1e-6) ? λ + δ : 1.0
        Δλ = next_λ - λ

        # Systematic resample
        log_w = Δλ .* ll_data
        w = exp.(log_w .- logsumexp(log_w))
        cumsum_w = cumsum(w)
        u_shift = rand(rng) / n_smc
        indices = Vector{Int}(undef, n_smc)
        for i in 1:n_smc
            t = (i - 1) / n_smc + u_shift
            indices[i] = clamp(searchsortedfirst(cumsum_w, t), 1, n_smc)
        end
        U_resampled = U[indices, :]

        # ── θ-cloud rejuvenation between resample and HMC.
        #
        # Two alternatives, gated by the cfg fields:
        #
        # (a) `cfg.smooth_resample_bw > 0` — Silverman-bandwidth KDE
        #     resample with a coupled Liu-West correction. Multivariate;
        #     uses the framework's `silverman_bandwidth` +
        #     `log_kernel_matrix` and applies
        #         a = √(1 - h_norm²)
        #     where h_norm is the dimensionless Silverman factor for the
        #     given (n_smc, d, scale). This subsumes per-dim Liu-West
        #     and is a richer rejuvenation; see
        #     `SMC2FC_functional/src/Filtering/Kernels.jl::smooth_resample`.
        #
        # (b) Otherwise, if `cfg.liu_west_a ∈ (0, 1)` — fall back to the
        #     cheap per-dim Liu-West:
        #         θ_i := a·θ_i + (1-a)·θ_mean + √(1-a²)·σ_d·ξ
        #     with a per-dimension std and identity correlation.
        #
        # Default (cfg.smooth_resample_bw = 0.0, cfg.liu_west_a = 0.97):
        # path (b) applies. Set `--smooth-resample-bw 1.0` to switch to
        # path (a).
        if cfg.smooth_resample_bw > 0.0
            stochastic_idx = collect(1:d)
            scale = cfg.smooth_resample_bw
            h     = silverman_bandwidth(U_resampled, stochastic_idx,
                                          n_smc, scale)
            L_kern = log_kernel_matrix(U_resampled, stochastic_idx, h)
            # Uniform weights post-systematic-resample, so log_w_b reduces
            # to L_kern. Renormalise rows to obtain the kernel-weight
            # matrix A.
            log_A   = L_kern .- logsumexp(L_kern; dims = 2)
            A       = exp.(log_A)
            blended = A * U_resampled
            # Liu-West correction: a = √(1 - h_norm²).
            silverman_factor = (4.0 / (d + 2.0)) ^ (1.0 / (d + 4.0))
            k_factor         = float(n_smc) ^ (-1.0 / (d + 4.0))
            h_norm           = silverman_factor * k_factor * scale
            a_lw = sqrt(clamp(1.0 - h_norm^2, 0.0, 1.0))
            μ    = vec(mean(U_resampled; dims = 1))
            U_resampled = a_lw .* blended .+
                           (1.0 - a_lw) .* reshape(μ, 1, :)
        elseif cfg.liu_west_a > 0.0 && cfg.liu_west_a < 1.0
            a = cfg.liu_west_a
            θ_mean = mean(U_resampled; dims = 1)
            θ_std  = std(U_resampled;  dims = 1)
            jitter = sqrt(1 - a^2) .* θ_std .* randn(rng, size(U_resampled))
            U_resampled = a .* U_resampled .+ (1 - a) .* θ_mean .+ jitter
        end

        # ── Filter HMC moves: framework ChEES path (2026-05-09 unification).
        #
        # Switched from the model-specific `parallel_hmc_one_move!` (in
        # `gpu_pf.jl`) to the framework's
        # `parallel_hmc_one_move_generic!` + `chees_pick_L_generic`.
        # Same algorithm the controller uses; β=next_λ is built into the
        # framework function so it correctly leaves π_{next_λ} invariant.
        #
        # `log_density_fn` wraps `gpu_log_density` into the
        # `Matrix{Float64} -> Vector{Float64}` shape the framework HMC
        # expects. The framework's `gpu_grads_parallel_chains_fd` builds
        # the (M·(1+2d), d) FD-batch matrix internally and calls this
        # closure once per gradient evaluation. `target.M_max` was sized
        # at `n_smc · (1 + 2·d) = 10752` at filter-target build time, so
        # the M·(1+2d) ≤ M_max check passes.
        U_curr = U_resampled    # reuse this buffer; mutated in place
        n_acc_total = 0
        log_density_fn = U_in -> gpu_log_density(target, U_in, grid_obs,
                                                    hash((key, :ll_hmc, n_temp)))
        # ChEES picks the leapfrog count L on a small subset for this
        # tempering level (mirrors the controller's pattern at
        # GPUControlSMC.jl:300-308).
        chees_n = min(8, n_smc)
        L_used, _ = chees_pick_L_generic(U_curr[1:chees_n, :],
                                            log_density_fn, target.M_max,
                                            cfg.hmc_step, cfg.chees_L_candidates,
                                            prior_means, prior_sigmas,
                                            MersenneTwister(rand(rng, UInt32));
                                            beta = next_λ, h_fd = cfg.h_fd)
        for k in 1:cfg.num_mcmc
            n_acc_total += parallel_hmc_one_move_generic!(U_curr, log_density_fn,
                                                             target.M_max,
                                                             cfg.hmc_step, L_used,
                                                             prior_means, prior_sigmas, rng;
                                                             beta = next_λ,
                                                             h_fd = cfg.h_fd)
        end
        accept_frac = n_acc_total / max(1, cfg.num_mcmc * n_smc)

        U = U_curr
        n_temp += 1
        @info @sprintf("    [%2d] λ %.3f → %.3f (Δλ=%.3f) L=%d accept=%.0f%%",
                       n_temp, λ, next_λ, Δλ, L_used, 100 * accept_frac)
        λ = next_λ
        n_temp >= cfg.max_levels && break
    end

    # Final gpu_log_density call on the posterior chains so that
    # `bufs.mu_per_chain[1:n_smc]` reflects the M=n_smc posterior particles
    # (not the M=n_smc·(1+2d) FD batch from the last HMC FD call).
    # extract_xhat reads from bufs after this call.
    _ = gpu_log_density(target, U, grid_obs, hash((key, :final)))

    return (U_post = U, n_temp = n_temp)
end


# ── State extraction from filter posterior ──────────────────────────────
# Importance-weighted mean of the inner-PF cloud at end-of-window. The
# stats kernel populates `bufs.mu_per_chain[1:M, :]` (M, 3) with weighted
# means per chain. We marginalise across M with uniform weights since
# each filter chain represents one θ posterior particle.

function extract_xhat(target::FSAGPUTarget, M::Int)
    mu_cpu = Array(view(target.bufs.mu_per_chain, 1:M, :))
    return SVector{3, Float64}(mean(mu_cpu[:, 1]),
                                mean(mu_cpu[:, 2]),
                                mean(mu_cpu[:, 3]))
end
