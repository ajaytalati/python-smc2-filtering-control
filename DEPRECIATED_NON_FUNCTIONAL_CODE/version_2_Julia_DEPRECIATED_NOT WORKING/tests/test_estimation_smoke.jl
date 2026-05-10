# Estimation-side smoke tests:
#
#   1. Priors sample without errors and have the right cardinality.
#   2. Constrained ↔ unconstrained transforms round-trip within tolerance.
#   3. propagate_fn runs on a single window without erroring and returns
#      a valid (x_new, pred_lw) pair.
#   4. obs_log_weight_fn returns a finite scalar.
#   5. forward_sde_stochastic runs n_steps without errors.

using Test
using Random
using LinearAlgebra
using Distributions: LogNormal, Normal

# Default 15-min step.
delete!(ENV, "FSA_STEP_MINUTES")

const MODEL_DIR = abspath(joinpath(@__DIR__, "..", "models", "fsa_high_res"))
isdefined(Main, :FSAHighRes) || include(joinpath(MODEL_DIR, "FSAHighRes.jl"))
const FSA = Main.FSAHighRes
const PARAM_NAMES         = FSA.Estimation.PARAM_NAMES
const _PI                 = FSA.Estimation._PI
const PARAM_PRIOR_CONFIG  = FSA.Estimation.PARAM_PRIOR_CONFIG
const propagate_fn        = FSA.Estimation.propagate_fn
const obs_log_weight_fn   = FSA.Estimation.obs_log_weight_fn
const forward_sde_stochastic = FSA.Estimation.forward_sde_stochastic
const get_init_theta      = FSA.Estimation.get_init_theta
const build_estimation_model = FSA.Estimation.build_estimation_model
using SMC2FC: LogNormalPrior, NormalPrior,
              to_unconstrained, to_constrained,
              constrained_to_unconstrained, unconstrained_to_constrained,
              log_prior_unconstrained


@testset "test_estimation_smoke" begin

    @testset "prior cardinality" begin
        @test length(PARAM_PRIOR_CONFIG) == 30
        @test length(PARAM_NAMES) == 30
        @test length(_PI) == 30
    end

    @testset "priors sample without error" begin
        rng = Random.MersenneTwister(0)
        for (name, prior) in PARAM_PRIOR_CONFIG
            sample = if prior isa LogNormalPrior
                rand(rng, LogNormal(prior.μ, prior.σ))
            elseif prior isa NormalPrior
                rand(rng, Normal(prior.μ, prior.σ))
            else
                0.0
            end
            @test isfinite(sample)
        end
    end

    @testset "constrained <-> unconstrained roundtrip" begin
        rng = Random.MersenneTwister(1)
        priors = [p for (_, p) in PARAM_PRIOR_CONFIG]
        # Sample θ from each prior in constrained space.
        θ = Float64[]
        for prior in priors
            if prior isa LogNormalPrior
                push!(θ, rand(rng, LogNormal(prior.μ, prior.σ)))
            elseif prior isa NormalPrior
                push!(θ, rand(rng, Normal(prior.μ, prior.σ)))
            else
                push!(θ, 0.5)
            end
        end
        u = constrained_to_unconstrained(θ, priors)
        θ_back = unconstrained_to_constrained(u, priors)
        @test all(isapprox.(θ, θ_back; rtol=1e-9))
    end

    @testset "log_prior_unconstrained returns finite scalar" begin
        priors = [p for (_, p) in PARAM_PRIOR_CONFIG]
        u = zeros(length(priors))
        lp = log_prior_unconstrained(u, priors)
        @test isfinite(lp)
    end

    @testset "propagate_fn runs and returns valid output" begin
        # Construct a single-bin grid_obs.
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[60.0],
            :hr_present      => Float32[1.0],
            :stress_value    => Float32[30.0],
            :stress_present  => Float32[1.0],
            :log_steps_value => Float32[5.5],
            :steps_present   => Float32[1.0],
            :sleep_label     => Int32[0],
            :sleep_present   => Float32[1.0],
            :Phi             => Float32[1.0],
            :C               => Float32[0.5],
            :has_any_obs     => Float32[1.0],
        )
        params_vec = get_init_theta()
        y = [0.05, 0.30, 0.10]
        noise = randn(MersenneTwister(2), 3)
        sigma_diag = [0.010, 0.012, 0.020]
        x_new, pred_lw = propagate_fn(y, 0.0, 1.0/96, Float64.(params_vec),
                                       grid_obs, 1, sigma_diag, noise, nothing)
        @test length(x_new) == 3
        @test all(isfinite, x_new)
        @test 0.0 < x_new[1] < 1.0
        @test x_new[2] >= 0.0
        @test x_new[3] >= 0.0
        @test isfinite(pred_lw)
    end

    @testset "obs_log_weight_fn returns finite scalar" begin
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[60.0],
            :hr_present      => Float32[1.0],
            :stress_value    => Float32[30.0],
            :stress_present  => Float32[0.0],
            :log_steps_value => Float32[5.5],
            :steps_present   => Float32[1.0],
            :sleep_label     => Int32[1],
            :sleep_present   => Float32[1.0],
            :Phi             => Float32[1.0],
            :C               => Float32[0.5],
            :has_any_obs     => Float32[1.0],
        )
        params_vec = Float64.(get_init_theta())
        y = [0.05, 0.30, 0.10]
        lp = obs_log_weight_fn(y, grid_obs, 1, params_vec)
        @test isfinite(lp)
    end

    @testset "forward_sde_stochastic runs N steps" begin
        params_vec = Float64.(get_init_theta())
        n_steps = 96
        exogenous = Dict(:Phi => Float64.(ones(n_steps)))
        traj = forward_sde_stochastic([0.05, 0.30, 0.10], params_vec,
                                       exogenous, 1.0/96, n_steps;
                                       rng=MersenneTwister(3))
        @test size(traj) == (n_steps, 3)
        @test all(isfinite, traj)
        @test all(0.0 .<= traj[:, 1] .<= 1.0)
    end

    @testset "build_estimation_model assembles" begin
        em = build_estimation_model()
        @test em.name == "fsa_high_res_v2"
        @test em.n_states == 3
        @test em.n_stochastic == 3
        @test em.stochastic_indices == [1, 2, 3]
        @test em.state_bounds == [(0.0, 1.0), (0.0, 10.0), (0.0, 5.0)]
        @test length(em.param_priors) == 30
    end

end
