"""
    test_gate1_kalman_pf.jl

Gate 1 of the SMC2FC_functional replacement-readiness audit:
language-independent correctness check of the bootstrap PF against the
closed-form Kalman log-likelihood on a linear-Gaussian AR(1) model.

Mirrors `julia/SMC2FC/test/test_phase2_bootstrap.jl` from the original
port, adapted to the new library:

- API surface: `BootstrapWorkspace` instead of `BootstrapBuffers` (no
    explicit workspace passed here — `bootstrap_log_likelihood` allocates
    one internally when `workspace=nothing`).
- Module name: `SMC2FC_functional` instead of `SMC2FC`.

# Pass criteria
1. PF log-likelihood is finite.
2. `|pf_ll - kalman_ll| < 3.0` nats over T = 50 obs (about 1 nat per
    17 obs — comfortable Monte Carlo tolerance with N = 4000 particles).
3. Rerunning at the same seed gives bit-identical output.

# Run
```
cd julia/SMC2FC_functional
julia --project=. test/test_gate1_kalman_pf.jl
```

This file is self-contained and does not include the broader unit-test
suite.
"""

using Test
using Random
using LogExpFunctions: logsumexp
using SMC2FC_functional

@testset "Gate 1 — Bootstrap PF vs Kalman closed-form (AR(1))" begin
    # ── Toy model: AR(1) + Gaussian observations ────────────────────────────
    #   x_k = a · x_{k-1} + b · ξ_k,    ξ_k ~ N(0, 1)
    #   y_k = x_k + ε_k,                 ε_k ~ N(0, ρ)

    a_truth = 0.85
    b_truth = 0.4
    ρ_truth = 0.3

    rng = MersenneTwister(2026_05_06)
    T_obs = 50
    x = zeros(T_obs)
    y = zeros(T_obs)
    x[1] = randn(rng) * 1.0
    y[1] = x[1] + ρ_truth * randn(rng)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng)
        y[k] = x[k] + ρ_truth * randn(rng)
    end

    # ── Closed-form Kalman log-likelihood ───────────────────────────────────
    function kalman_log_lik(y::Vector{Float64}, a::Float64, b::Float64, ρ::Float64,
                            m0::Float64, P0::Float64)
        T = length(y)
        m, P = m0, P0
        ll = 0.0
        for k in 1:T
            m_pred = a * m
            P_pred = a^2 * P + b^2
            S      = P_pred + ρ^2
            ν      = y[k] - m_pred
            ll    += -0.5 * (log(2π * S) + ν^2 / S)
            K_gain = P_pred / S
            m      = m_pred + K_gain * ν
            P      = (1 - K_gain) * P_pred
        end
        return ll
    end

    kalman_ll = kalman_log_lik(y, a_truth, b_truth, ρ_truth, 0.0, 1.0)

    # ── Build an EstimationModel for the bootstrap PF ───────────────────────

    function _propagate(y_old::AbstractVector, t, dt, params,
                         grid_obs, k, σ_diag, ξ, rng_)
        a_, b_, _ = params[1], params[2], params[3]
        x_new = [a_ * y_old[1] + b_ * ξ[1]]
        return x_new, 0.0
    end

    function _diffusion(params)
        return [params[2]]
    end

    function _obs_log_weight(x_new::AbstractVector, grid_obs, k, params)
        ρ_  = params[3]
        y_k = grid_obs[:y][k]
        ν   = y_k - x_new[1]
        return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
    end

    _shard_init(time_offset, params, exog, init) = init
    _align_obs(args...) = Dict()

    model = EstimationModel(
        name = "AR1_Gauss",
        version = "test",
        n_states = 1,
        n_stochastic = 1,
        stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = Tuple{Symbol,PriorType}[
            (:a, NormalPrior(0.0, 1.0)),
            (:b, LogNormalPrior(0.0, 1.0)),
            (:ρ, LogNormalPrior(0.0, 1.0)),
        ],
        init_state_priors = Tuple{Symbol,PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn       = _propagate,
        diffusion_fn       = _diffusion,
        obs_log_weight_fn  = _obs_log_weight,
        align_obs_fn       = _align_obs,
        shard_init_fn      = _shard_init,
        exogenous_keys     = Symbol[],
    )

    priors = all_priors(model)
    # Unconstrained-space (a, b, ρ) at truth: a is identity, b/ρ are log.
    u = [a_truth, log(b_truth), log(ρ_truth)]

    # Disable OT rescue and Liu-West kernel contamination for a clean test.
    cfg = SMCConfig(n_pf_particles  = 4000,
                    bandwidth_scale = 0.0,
                    ot_max_weight   = 0.0)

    grid_obs   = Dict(:y => y)
    fixed_init = [0.0]

    pf_rng    = MersenneTwister(11)
    pf_target = bootstrap_log_likelihood(
        model, u, grid_obs, fixed_init, priors, cfg, pf_rng;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )

    # bootstrap_log_likelihood returns log p(y) + log p(u). Subtract the prior.
    log_prior_u = SMC2FC_functional.log_prior_unconstrained(u, priors)
    pf_ll = pf_target - log_prior_u

    @info "Gate 1: bootstrap PF vs Kalman" pf_ll kalman_ll diff=(pf_ll - kalman_ll)

    @test isfinite(pf_ll)
    @test abs(pf_ll - kalman_ll) < 3.0

    # Determinism: re-run at the same seed → byte-identical.
    pf_rng2    = MersenneTwister(11)
    pf_target2 = bootstrap_log_likelihood(
        model, u, grid_obs, fixed_init, priors, cfg, pf_rng2;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )
    @test pf_target ≈ pf_target2
end
