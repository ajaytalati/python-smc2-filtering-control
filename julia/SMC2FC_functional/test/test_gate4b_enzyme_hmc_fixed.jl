"""
    test_gate4b_enzyme_hmc_fixed.jl

Gate 4b: re-run of Gate 4 (Enzyme reverse-mode HMC step on the
bootstrap-PF target) **with** the Enzyme inactive-rules patch from
`_enzyme_inactive_rules.jl` loaded first.

# Pass criteria
1. `hmc_step_chain(...; ad_backend = :Enzyme, sampler = :HMC)` runs
    without throwing.
2. The returned position is finite and has the right shape.
3. Sanity: a `:ForwardDiff` control run also succeeds (so the test
    itself is well-formed).

# Why this is the same test as Gate 4
Gate 4's correctness criterion is unchanged. The only difference is
that the inactive-rules patch is loaded **before** Enzyme is invoked,
so the libdSFMT C call inside Random's `fill_array!` is treated as
non-active (which is mathematically correct — random samples are
independent of the AD variable `u`).

# Run
```
cd julia/SMC2FC_functional
julia --project=. test/test_gate4b_enzyme_hmc_fixed.jl
```
"""

using Test
using Random
using SMC2FC_functional
using SMC2FC_functional: HMC

# CRITICAL: load the Enzyme patch BEFORE any Enzyme call. Once loaded,
# the EnzymeRules registrations are global and any subsequent
# `hmc_step_chain(...; ad_backend = :Enzyme)` benefits from them.
include(joinpath(@__DIR__, "_enzyme_inactive_rules.jl"))

@testset "Gate 4b — HMC step under :Enzyme + :ForwardDiff (with patch)" begin
    # ── AR(1) toy bootstrap-PF target (same as Gate 1 / Gate 4) ─────────────
    a_truth, b_truth, ρ_truth = 0.85, 0.4, 0.3

    rng_obs = MersenneTwister(2026_05_06)
    T_obs = 30
    x = zeros(T_obs); y = zeros(T_obs)
    x[1] = randn(rng_obs)
    y[1] = x[1] + ρ_truth * randn(rng_obs)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng_obs)
        y[k] = x[k] + ρ_truth * randn(rng_obs)
    end

    function _propagate(y_old::AbstractVector, t, dt, params,
                         grid_obs, k, σ_diag, ξ, rng_)
        a_, b_, _ = params[1], params[2], params[3]
        return [a_ * y_old[1] + b_ * ξ[1]], 0.0
    end
    _diffusion(params) = [params[2]]
    function _obs_log_weight(x_new::AbstractVector, grid_obs, k, params)
        ρ_  = params[3]
        ν   = grid_obs[:y][k] - x_new[1]
        return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
    end

    model = EstimationModel(
        name = "AR1_for_Gate4b", version = "test",
        n_states = 1, n_stochastic = 1, stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = Tuple{Symbol,PriorType}[
            (:a, NormalPrior(0.0, 1.0)),
            (:b, LogNormalPrior(0.0, 1.0)),
            (:ρ, LogNormalPrior(0.0, 1.0)),
        ],
        init_state_priors = Tuple{Symbol,PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn      = _propagate,
        diffusion_fn      = _diffusion,
        obs_log_weight_fn = _obs_log_weight,
        align_obs_fn      = (args...) -> Dict(),
        shard_init_fn     = (t,p,e,init) -> init,
        exogenous_keys    = Symbol[],
    )
    priors    = all_priors(model)
    cfg       = SMCConfig(n_pf_particles = 200, bandwidth_scale = 0.0,
                          ot_max_weight = 0.0)
    grid_obs  = Dict(:y => y)
    fixed_init = [0.0]

    function build_lp(stride_seed::Int)
        return function lp(u::AbstractVector)
            pf_rng = MersenneTwister(stride_seed)
            return bootstrap_log_likelihood(
                model, u, grid_obs, fixed_init, priors, cfg, pf_rng;
                dt = 1.0, t_steps = T_obs, window_start_bin = 0,
            )
        end
    end

    u0       = [a_truth, log(b_truth), log(ρ_truth)]
    inv_mass = ones(3)

    # ── (1) Sanity: ForwardDiff path (control) ──────────────────────────────
    @info "Gate 4b: ForwardDiff control path"
    lp_fd  = build_lp(101)
    rng_fd = MersenneTwister(42)
    u_fd   = HMC.hmc_step_chain(u0, lp_fd, 1, 0.02, inv_mass, 4, rng_fd;
                                  ad_backend = :ForwardDiff, sampler = :HMC)
    @test all(isfinite, u_fd)
    @test length(u_fd) == 3
    @info "Gate 4b: ForwardDiff result" u_fd diff = u_fd .- u0

    # ── (2) Enzyme path with patch ──────────────────────────────────────────
    @info "Gate 4b: Enzyme reverse-mode path (with EnzymeRules.inactive patch)"
    lp_enz  = build_lp(202)
    rng_enz = MersenneTwister(43)
    local u_enz
    enzyme_ok = try
        u_enz = HMC.hmc_step_chain(u0, lp_enz, 1, 0.02, inv_mass, 4, rng_enz;
                                     ad_backend = :Enzyme, sampler = :HMC)
        true
    catch err
        @warn "Gate 4b — Enzyme HMC step still threw" err
        false
    end

    @test enzyme_ok
    if enzyme_ok
        @test all(isfinite, u_enz)
        @test length(u_enz) == 3
        @info "Gate 4b: Enzyme result" u_enz diff = u_enz .- u0
    end
end
