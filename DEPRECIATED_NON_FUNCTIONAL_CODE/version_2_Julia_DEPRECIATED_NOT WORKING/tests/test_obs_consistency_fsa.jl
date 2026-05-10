# Sim/estimator obs-channel parity (D1/D2-class bug guard).
#
# For each of the 4 obs channels, check that:
#   - the simulator-side obs sampler in `simulation.jl` and
#   - the estimator-side log-likelihood / propagate_fn in `estimation.jl`
# encode the SAME observation formula (same H matrix coefficients, same
# bias, same gating).
#
# The check: pick a known state + params, draw N=10_000 obs samples from
# the simulator side (or compute the analytic mean), then evaluate the
# estimator's predicted mean on the same state and confirm equality.
#
# This catches the D1/D2-class bugs (sign flips, missing terms, sigma
# mismatches between simulator and estimator) that quietly break the SMC²
# filter without raising errors.

using Test
using Statistics
using Random

# Default 15-min step (BINS_PER_DAY=96).
delete!(ENV, "FSA_STEP_MINUTES")

const MODEL_DIR = abspath(joinpath(@__DIR__, "..", "models", "fsa_high_res"))
# Only include if not already loaded by runtests.jl; redefining the module
# pollutes Main with conflicting bindings.
isdefined(Main, :FSAHighRes) || include(joinpath(MODEL_DIR, "FSAHighRes.jl"))
const FSA = Main.FSAHighRes
const DEFAULT_PARAMS = FSA.Simulation.DEFAULT_PARAMS
const PARAM_NAMES   = FSA.Estimation.PARAM_NAMES
const _PI           = FSA.Estimation._PI
const obs_log_weight_fn = FSA.Estimation.obs_log_weight_fn
const gen_obs_hr        = FSA.Simulation.gen_obs_hr
const Simulation        = FSA.Simulation


@testset "test_obs_consistency_fsa" begin

    # Build a known state and params vector aligned to PARAM_NAMES.
    state = [0.4, 0.25, 0.15]
    p = DEFAULT_PARAMS
    params_vec = Float64[p[name] for name in PARAM_NAMES]

    # Single-bin grid with deterministic C(t).
    C_k = 0.7         # arbitrary in [-1, 1]
    Phi_k = 1.0
    t_grid = [0.0]

    # ── HR analytic mean: HR_base − κ_B_HR · B + α_A_HR · A + β_C_HR · C ─
    hr_mean_analytic = p[:HR_base] - p[:kappa_B_HR] * state[1] +
                        p[:alpha_A_HR] * state[3] + p[:beta_C_HR] * C_k

    # ── Stress analytic mean: S_base + k_F · F − k_A_S · A + β_C_S · C ──
    stress_mean_analytic = p[:S_base] + p[:k_F] * state[2] -
                           p[:k_A_S] * state[3] + p[:beta_C_S] * C_k

    # ── Steps log-mean: μ_step0 + β_B_st · B − β_F_st · F + β_A_st · A
    #                   + β_C_st · C ──
    log_steps_mean_analytic = p[:mu_step0] + p[:beta_B_st] * state[1] -
                              p[:beta_F_st] * state[2] +
                              p[:beta_A_st] * state[3] +
                              p[:beta_C_st] * C_k

    # ── Sleep prob: σ(k_C · C + k_A · A − c_tilde) ──────────────────────
    sleep_logit = p[:k_C] * C_k + p[:k_A] * state[3] - p[:c_tilde]
    sleep_prob_analytic = 1.0 / (1.0 + exp(-sleep_logit))

    @testset "estimator HR predicted mean matches simulator formula" begin
        # gaussian_obs_ll computes (y - pred)/σ as the residual; pred =
        # HR_base − κ_B_HR·B + α_A_HR·A + β_C_HR·C (matches Python
        # estimation.py:305-312).
        # We test by checking obs_log_weight at the analytic mean equals
        # the Gaussian log-density at zero residual.
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[hr_mean_analytic],
            :hr_present      => Float32[1.0],
            :stress_value    => Float32[0.0],
            :stress_present  => Float32[0.0],
            :log_steps_value => Float32[0.0],
            :steps_present   => Float32[0.0],
            :sleep_label     => Int32[0],
            :sleep_present   => Float32[0.0],
            :Phi             => Float32[Phi_k],
            :C               => Float32[C_k],
            :has_any_obs     => Float32[1.0],
        )
        # log p(y_HR | state) = -0.5 log(2π σ²) - 0.5 (resid/σ)²
        sigma_HR = p[:sigma_HR]
        expected_lp = -0.5 * log(2π) - log(sigma_HR)
        lp = obs_log_weight_fn(state, grid_obs, 1, params_vec)
        @test isapprox(lp, expected_lp; atol=1e-9)
    end

    @testset "estimator stress predicted mean matches simulator formula" begin
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[0.0],
            :hr_present      => Float32[0.0],
            :stress_value    => Float32[stress_mean_analytic],
            :stress_present  => Float32[1.0],
            :log_steps_value => Float32[0.0],
            :steps_present   => Float32[0.0],
            :sleep_label     => Int32[0],
            :sleep_present   => Float32[0.0],
            :Phi             => Float32[Phi_k],
            :C               => Float32[C_k],
            :has_any_obs     => Float32[1.0],
        )
        sigma_S = p[:sigma_S]
        expected_lp = -0.5 * log(2π) - log(sigma_S)
        lp = obs_log_weight_fn(state, grid_obs, 1, params_vec)
        @test isapprox(lp, expected_lp; atol=1e-9)
    end

    @testset "estimator log_steps predicted mean matches simulator formula" begin
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[0.0],
            :hr_present      => Float32[0.0],
            :stress_value    => Float32[0.0],
            :stress_present  => Float32[0.0],
            :log_steps_value => Float32[log_steps_mean_analytic],
            :steps_present   => Float32[1.0],
            :sleep_label     => Int32[0],
            :sleep_present   => Float32[0.0],
            :Phi             => Float32[Phi_k],
            :C               => Float32[C_k],
            :has_any_obs     => Float32[1.0],
        )
        sigma_st = p[:sigma_st]
        expected_lp = -0.5 * log(2π) - log(sigma_st)
        lp = obs_log_weight_fn(state, grid_obs, 1, params_vec)
        @test isapprox(lp, expected_lp; atol=1e-9)
    end

    @testset "estimator sleep prob matches simulator formula" begin
        # Test by evaluating the Bernoulli log-prob explicitly.
        # If sleep_label=1 and sleep_present=1, log-prob = log(p_sleep).
        grid_obs = Dict{Symbol,Any}(
            :hr_value        => Float32[0.0],
            :hr_present      => Float32[0.0],
            :stress_value    => Float32[0.0],
            :stress_present  => Float32[0.0],
            :log_steps_value => Float32[0.0],
            :steps_present   => Float32[0.0],
            :sleep_label     => Int32[1],
            :sleep_present   => Float32[1.0],
            :Phi             => Float32[Phi_k],
            :C               => Float32[C_k],
            :has_any_obs     => Float32[1.0],
        )
        # All Gaussian channels are absent (present=0), so lp = log(sleep_prob).
        # Tolerance 1e-7 — sleep_log_prob clamps p to [1e-8, 1-1e-8] for
        # numerical safety, which introduces ~1e-8 relative error.
        expected_lp = log(sleep_prob_analytic)
        lp = obs_log_weight_fn(state, grid_obs, 1, params_vec)
        @test isapprox(lp, expected_lp; atol=1e-7)
    end

    @testset "simulator HR mean (Monte Carlo)" begin
        # Draw N=10_000 HR samples from gen_obs_hr at constant state and
        # check the empirical mean matches the analytic one within stderr.
        rng = Random.MersenneTwister(42)
        N = 10_000
        traj = repeat(reshape(state, 1, 3), N, 1)
        t_grid_full = collect(0.0:Float64(N - 1)) ./ 96
        # Force "always present" sleep gate by passing prior_channels with
        # all-1 sleep label.
        prior = Dict(:obs_sleep => (
            t_idx = collect(Int32, 0:N - 1),
            sleep_label = ones(Int32, N),
        ))
        ch = gen_obs_hr(traj, t_grid_full, p, nothing, prior, 42)
        # The C(t) value over t_grid_full varies; subtract its contribution
        # so we can compare against the constant analytic mean for B, A.
        C_grid = Simulation.circadian(t_grid_full, get(p, :phi, 0.0))
        # mean obs = HR_base − κ_B_HR·B + α_A_HR·A + mean(β_C_HR · C)
        expected_mean = p[:HR_base] - p[:kappa_B_HR] * state[1] +
                         p[:alpha_A_HR] * state[3] + p[:beta_C_HR] * mean(C_grid)
        @test isapprox(mean(ch.obs_value), expected_mean;
                        atol = 5 * p[:sigma_HR] / sqrt(N))
    end

end
