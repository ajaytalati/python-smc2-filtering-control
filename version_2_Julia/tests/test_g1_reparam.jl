# Stage G1 verification — port of `version_2/tests/test_g1_reparam.py`.
#
# The G1 reparametrization is a coordinate change. Drift_repram(y, truth_repram, Φ)
# must equal v2-original drift(y, truth_v2_orig, Φ) for any state y and Φ.
# We assert this to ≤ 1e-10 fp precision.

using Test
using Random
using LinearAlgebra

const MODEL_DIR = abspath(joinpath(@__DIR__, "..", "models", "fsa_high_res"))
isdefined(Main, :FSAHighRes) || include(joinpath(MODEL_DIR, "FSAHighRes.jl"))
const FSA = Main.FSAHighRes
const drift = FSA.Dynamics.drift
const TRUTH_PARAMS = FSA.Dynamics.TRUTH_PARAMS
const A_TYP        = FSA.Dynamics.A_TYP
const F_TYP        = FSA.Dynamics.F_TYP


# Original v2 truth (BEFORE reparametrization) for parity check.
const ORIGINAL_V2_TRUTH = (
    tau_B    = 42.0,
    tau_F    =  7.0,
    kappa_B  = 0.012,
    kappa_F  = 0.030,
    epsilon_A = 0.40,
    lambda_A  = 1.00,
    mu_0  = 0.02,
    mu_B  = 0.30,
    mu_F  = 0.10,
    mu_FF = 0.40,
    eta   = 0.20,
    sigma_B = 0.010,
    sigma_F = 0.012,
    sigma_A = 0.020,
)


"""
    drift_v2_original(y, params, Phi_t)

The ORIGINAL v2 drift formulation (BEFORE G1 reparametrization). Reproduced
inline as the reference for parity testing.
"""
function drift_v2_original(y::AbstractVector, params, Phi_t::Real)
    B, F, A = y[1], y[2], y[3]
    μ = params.mu_0 + params.mu_B * B - params.mu_F * F - params.mu_FF * F * F
    dB = params.kappa_B * (1.0 + params.epsilon_A * A) * Phi_t - B / params.tau_B
    dF = params.kappa_F * Phi_t -
         (1.0 + params.lambda_A * A) / params.tau_F * F
    dA = μ * A - params.eta * A^3
    return [dB, dF, dA]
end


@testset "test_g1_reparam" begin

    @testset "drift_parity_at_typical_state" begin
        y = [0.05, 0.30, 0.10]
        Phi_t = 1.0
        d_rep  = drift(y, TRUTH_PARAMS, Phi_t)
        d_orig = drift_v2_original(y, ORIGINAL_V2_TRUTH, Phi_t)
        @test isapprox(d_rep, d_orig; atol=1e-10, rtol=1e-10)
    end

    @testset "drift_parity_grid" begin
        rng = MersenneTwister(0)
        for _ in 1:50
            B = 0.05 + 0.90 * rand(rng)
            F = 0.05 + 0.45 * rand(rng)
            A = 0.05 + 1.15 * rand(rng)
            Phi = 3.0 * rand(rng)
            y = [B, F, A]
            d_rep  = drift(y, TRUTH_PARAMS, Phi)
            d_orig = drift_v2_original(y, ORIGINAL_V2_TRUTH, Phi)
            @test isapprox(d_rep, d_orig; atol=1e-10, rtol=1e-10)
        end
    end

    @testset "truth_param_values_match_derivation" begin
        @test abs(TRUTH_PARAMS.kappa_B - 0.01248)         < 1e-10
        @test abs(TRUTH_PARAMS.tau_F   - 7.0 / 1.1)        < 1e-10
        @test abs(TRUTH_PARAMS.mu_F    - 0.26)             < 1e-10
        @test abs(TRUTH_PARAMS.mu_0    - 0.036)            < 1e-10
        @test abs(TRUTH_PARAMS.epsilon_A - 0.40)           < 1e-10
        @test abs(TRUTH_PARAMS.lambda_A  - 1.00)           < 1e-10
        @test abs(TRUTH_PARAMS.mu_FF   - 0.40)             < 1e-10
        @test abs(TRUTH_PARAMS.tau_B   - 42.0)             < 1e-10
    end

    @testset "reparametrization_isolates_residual_at_typical_point" begin
        # At (A=A_typ, F=F_typ), residual params (epsilon_A, lambda_A, mu_FF)
        # should drop out of drift.
        y_typ = [0.5, F_TYP, A_TYP]
        Phi_t = 1.0
        truth_pert = (
            tau_B    = TRUTH_PARAMS.tau_B,
            tau_F    = TRUTH_PARAMS.tau_F,
            kappa_B  = TRUTH_PARAMS.kappa_B,
            kappa_F  = TRUTH_PARAMS.kappa_F,
            epsilon_A = 0.0,                 # perturbed (was 0.40)
            lambda_A  = 0.0,                 # perturbed (was 1.00)
            mu_0  = TRUTH_PARAMS.mu_0,
            mu_B  = TRUTH_PARAMS.mu_B,
            mu_F  = TRUTH_PARAMS.mu_F,
            mu_FF = 0.0,                     # perturbed (was 0.40)
            eta   = TRUTH_PARAMS.eta,
            sigma_B = TRUTH_PARAMS.sigma_B,
            sigma_F = TRUTH_PARAMS.sigma_F,
            sigma_A = TRUTH_PARAMS.sigma_A,
        )
        d_typ  = drift(y_typ, TRUTH_PARAMS, Phi_t)
        d_pert = drift(y_typ, truth_pert, Phi_t)
        @test isapprox(d_typ, d_pert; atol=1e-10, rtol=1e-10)
    end

end
