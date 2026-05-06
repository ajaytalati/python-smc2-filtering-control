# Regression tests for the FSA-v5 param dictionaries.
#
# The historical Python `DEFAULT_PARAMS` had two `'sigma_S'` keys
# (state-noise vs stress obs noise). In Julia the structural fix is
# stronger than a guardrail: the two values live on TWO DIFFERENT
# ComponentVectors (`DynParams` vs `ObsParams`) with different field
# names. The historical bug cannot be expressed in Julia syntax.
#
# These tests confirm the structural separation holds.

using FSAv5
using Test

@testset "FSA-v5 param-dict guardrails" begin
    p  = FSAv5.TRUTH_PARAMS_V5
    op = FSAv5.DEFAULT_OBS_PARAMS_V5

    @testset "sigma_S vs sigma_S_obs structural separation" begin
        # State-noise scale (Bug 1: this used to be 4.0 silently)
        @test p.sigma_S == 0.008
        # Obs-channel noise (Bug 1: this used to clobber the state-noise key)
        @test op.sigma_S_obs == 4.0

        dyn_keys = keys(NamedTuple(p))
        obs_keys = keys(NamedTuple(op))

        @test :sigma_S in dyn_keys
        @test !(:sigma_S_obs in dyn_keys)   # bug 1 leak ⇒ fail
        @test :sigma_S_obs in obs_keys
        @test !(:sigma_S in obs_keys)       # bug 1 leak ⇒ fail
    end

    @testset "Field counts match LEAN reference" begin
        @test length(NamedTuple(p))  == 28
        @test length(NamedTuple(op)) == 22
        @test length(FSAv5.DYN_PARAM_KEYS) == 28
        @test length(FSAv5.OBS_PARAM_KEYS) == 22
    end

    @testset "Truth values match LaTeX §10.4 / Lean Types.lean" begin
        @test p.tau_B    == 42.0
        @test p.tau_S    == 60.0
        @test p.tau_K    == 21.0
        @test p.eta      == 0.20
        @test p.B_dec    == 0.07
        @test p.S_dec    == 0.07
        @test p.mu_dec_B == 0.10
        @test p.mu_dec_S == 0.10
        @test p.n_dec    == 4.0
    end

    @testset "Operating-point constants match LEAN" begin
        @test FSAv5.A_TYP   == 0.10
        @test FSAv5.F_TYP   == 0.20
        @test FSAv5.PHI_TYP == 1.0
    end

    @testset "make_dyn_params validates field set" begin
        # Missing key
        @test_throws ErrorException FSAv5.make_dyn_params(tau_B = 42.0)
        # Extra key — typo guard
        bad_kwargs = (k => 0.0 for k in FSAv5.DYN_PARAM_KEYS)
        @test_throws ErrorException FSAv5.make_dyn_params(; bad_kwargs..., bogus_key = 1.0)
    end
end
