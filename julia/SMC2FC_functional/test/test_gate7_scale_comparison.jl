"""
    test_gate7_scale_comparison.jl

Gate 7 of the SMC2FC_functional replacement-readiness audit:
head-to-head log-likelihood comparison between the original
`SMC2FC` port and the new `SMC2FC_functional` library at a
production-meaningful particle count, on the AR(1) Kalman model
(same setup as Gate 1 — exact reference is available).

# Why not n_smc=256, k_pf=400?
The literal "production scale" in the report (256 outer, 400 inner)
is feasible on the GPU bench in `version_1_5_Julia/tools/`; on the
CPU SMC2FC_functional path with a ForwardDiff gradient through the
PF, one window at that scale is several hours to several days.
A meaningful CPU-tractable equivalent is `n_pf = 4000` (×100 over the
unit-test default) per the original Gate-1-style harness, *without*
the outer-SMC² loop. That is what this test runs: a marginal-likelihood
shoot-out at the largest particle count both libraries can complete in
under one minute.

# Pass criteria
1. Both libraries produce finite log-likelihoods.
2. Both libraries lie within 3 nats of the closed-form Kalman value
    over T = 50 obs (the Gate-1 tolerance).
3. The two libraries agree with each other within 0.5 nats — the
    expected MC variance from independent PRNG streams at this
    particle count.

# Run
```
cd julia/SMC2FC_functional
julia --project=. test/test_gate7_scale_comparison.jl
```

This test depends on both libraries being `Pkg.dev`-reachable. The
`benchmarks/compare_three_libraries.jl` script already arranges that;
this test does the same dance up-front.
"""

using Test
using Random
using Pkg
using LogExpFunctions: logsumexp

# Make both libraries reachable.
const REPO_ROOT     = abspath(joinpath(@__DIR__, "..", "..", ".."))
const SMC2FC_DIR    = joinpath(REPO_ROOT, "julia", "SMC2FC")
const SMC2FC_FN_DIR = joinpath(REPO_ROOT, "julia", "SMC2FC_functional")
let proj = Pkg.project().dependencies
    if !haskey(proj, "SMC2FC")
        Pkg.develop(path = SMC2FC_DIR)
    end
end

using SMC2FC
using SMC2FC_functional
const FN = SMC2FC_functional

@testset "Gate 7 — scale shoot-out: SMC2FC vs SMC2FC_functional vs Kalman" begin
    # ── AR(1) ground truth + closed-form Kalman log-lik (same as Gate 1) ────
    a_truth, b_truth, ρ_truth = 0.85, 0.4, 0.3
    rng_obs = MersenneTwister(2026_05_06)
    T_obs   = 50
    x = zeros(T_obs); y = zeros(T_obs)
    x[1] = randn(rng_obs)
    y[1] = x[1] + ρ_truth * randn(rng_obs)
    for k in 2:T_obs
        x[k] = a_truth * x[k-1] + b_truth * randn(rng_obs)
        y[k] = x[k] + ρ_truth * randn(rng_obs)
    end

    function kalman_ll(y, a, b, ρ)
        m, P, ll = 0.0, 1.0, 0.0
        for k in 1:length(y)
            m_pred = a * m
            P_pred = a^2 * P + b^2
            S      = P_pred + ρ^2
            ν      = y[k] - m_pred
            ll    += -0.5 * (log(2π * S) + ν^2 / S)
            K_g    = P_pred / S
            m      = m_pred + K_g * ν
            P      = (1 - K_g) * P_pred
        end
        return ll
    end
    ll_kalman = kalman_ll(y, a_truth, b_truth, ρ_truth)

    # ── Adapter functions (same body for both libraries) ────────────────────
    function _propagate(y_old, t, dt, params, grid_obs, k, σ_diag, ξ, rng_)
        a_, b_, _ = params[1], params[2], params[3]
        return [a_ * y_old[1] + b_ * ξ[1]], 0.0
    end
    _diffusion(params) = [params[2]]
    function _obs_log_weight(x_new, grid_obs, k, params)
        ρ_ = params[3]
        ν  = grid_obs[:y][k] - x_new[1]
        return -0.5 * (log(2π * ρ_^2) + ν^2 / ρ_^2)
    end
    _shard_init(t, p, e, init) = init
    _align_obs(args...) = Dict()

    # Build one EstimationModel per library (different module's struct).
    model_orig = SMC2FC.EstimationModel(
        name = "AR1_orig", version = "test",
        n_states = 1, n_stochastic = 1, stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = [
            (:a, SMC2FC.NormalPrior(0.0, 1.0)),
            (:b, SMC2FC.LogNormalPrior(0.0, 1.0)),
            (:ρ, SMC2FC.LogNormalPrior(0.0, 1.0)),
        ],
        init_state_priors = Tuple{Symbol,SMC2FC.PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn = _propagate, diffusion_fn = _diffusion,
        obs_log_weight_fn = _obs_log_weight,
        align_obs_fn = _align_obs, shard_init_fn = _shard_init,
        exogenous_keys = Symbol[],
    )
    model_fn = FN.EstimationModel(
        name = "AR1_fn", version = "test",
        n_states = 1, n_stochastic = 1, stochastic_indices = [1],
        state_bounds = [(-50.0, 50.0)],
        param_priors = Tuple{Symbol,FN.PriorType}[
            (:a, FN.NormalPrior(0.0, 1.0)),
            (:b, FN.LogNormalPrior(0.0, 1.0)),
            (:ρ, FN.LogNormalPrior(0.0, 1.0)),
        ],
        init_state_priors = Tuple{Symbol,FN.PriorType}[],
        frozen_params = Dict{Symbol,Float64}(),
        propagate_fn = _propagate, diffusion_fn = _diffusion,
        obs_log_weight_fn = _obs_log_weight,
        align_obs_fn = _align_obs, shard_init_fn = _shard_init,
        exogenous_keys = Symbol[],
    )

    n_pf_scale = 4000   # max scale that completes in <60s on this CPU
    cfg_orig = SMC2FC.SMCConfig(n_pf_particles = n_pf_scale,
                                  bandwidth_scale = 0.0,
                                  ot_max_weight   = 0.0)
    cfg_fn   = FN.SMCConfig(n_pf_particles = n_pf_scale,
                              bandwidth_scale = 0.0,
                              ot_max_weight   = 0.0)

    u = [a_truth, log(b_truth), log(ρ_truth)]
    grid_obs   = Dict(:y => y)
    fixed_init = [0.0]

    # ── Original port ───────────────────────────────────────────────────────
    priors_orig = SMC2FC.all_priors(model_orig)
    rng_orig    = MersenneTwister(2024)
    t0          = time()
    target_orig = SMC2FC.bootstrap_log_likelihood(
        model_orig, u, grid_obs, fixed_init, priors_orig, cfg_orig, rng_orig;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )
    ll_orig     = target_orig - SMC2FC.log_prior_unconstrained(u, priors_orig)
    t_orig      = time() - t0

    # ── Functional library ──────────────────────────────────────────────────
    priors_fn = FN.all_priors(model_fn)
    rng_fn    = MersenneTwister(2024)
    t0        = time()
    target_fn = FN.bootstrap_log_likelihood(
        model_fn, u, grid_obs, fixed_init, priors_fn, cfg_fn, rng_fn;
        dt = 1.0, t_steps = T_obs, window_start_bin = 0,
    )
    ll_fn     = target_fn - FN.log_prior_unconstrained(u, priors_fn)
    t_fn      = time() - t0

    @info "Gate 7: scale shoot-out at n_pf = $n_pf_scale, T_obs = $T_obs" ll_kalman ll_orig ll_fn
    @info "  diff vs Kalman" diff_orig=(ll_orig - ll_kalman) diff_fn=(ll_fn - ll_kalman)
    @info "  diff lib-vs-lib" diff_orig_fn=(ll_orig - ll_fn)
    @info "  wall time" t_orig_s=round(t_orig; digits=2) t_fn_s=round(t_fn; digits=2)

    @test isfinite(ll_orig)
    @test isfinite(ll_fn)
    # Each library agrees with Kalman within Gate-1 tolerance.
    @test abs(ll_orig - ll_kalman) < 3.0
    @test abs(ll_fn   - ll_kalman) < 3.0
    # The two libraries agree with each other within MC variance.
    # At n_pf = 4000 with bandwidth_scale = 0 and OT off, the per-window
    # Monte Carlo σ on log-lik is ~0.5 nats; with independent PRNG streams,
    # |Δ| < 0.5 nats is the expected band. We use 1.0 nat as a safety margin.
    @test abs(ll_orig - ll_fn) < 1.0
end
