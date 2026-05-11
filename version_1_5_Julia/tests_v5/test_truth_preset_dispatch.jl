# tests_v5/test_truth_preset_dispatch.jl
#
# Regression tests for the v2 truth/frozen dispatch.
#
# This file exists because of the bug discovered on 2026-05-11: under
# `--truth-preset v2` the plant was correctly using v2 dynamics, but the
# filter inner-PF and the closed-loop controller's posterior-mean dict were
# silently using the canonical `FROZEN_PARAMS_V5` (B_dec=0.07, S_dec=0.07)
# instead of the v2 versions (B_dec=0.25, S_dec=0.25). The controller's
# internal cost model thought the healthy island sat at Φ ≈ (0.30, 0.30)
# while the actual plant's v2 island sat at Φ ≈ (1.06, 0.78) — so the
# closed-loop bench drifted INTO the sedentary basin instead of away from
# it, and A collapsed to ~0.02 instead of escaping to ~1.16.
#
# Open-loop was unaffected because the open-loop controller call uses
# `full_params_v5` directly and never goes through `posterior_mean_v5` or
# the filter target. So the bug was invisible under open-loop and only
# manifested under closed-loop.
#
# The fix routes `--truth-preset` through `select_truth_preset` (in
# SimulationV5) which returns the (truth, frozen) PAIR together, and that
# pair is threaded through to the filter target (`FSAv5GPUTarget(; frozen=)`)
# and to `posterior_mean_v5(; frozen=)`. These tests assert each link in
# that chain so the next person who edits the dispatch can't break it
# silently.
#
# Run with:
#   cd version_1_5_Julia && julia --project=. tests_v5/test_truth_preset_dispatch.jl

using Test
using Statistics

# Load the v5 model module (don't include the bench driver — it needs CUDA).
include(joinpath(@__DIR__, "..", "models", "fsa_v5", "FSAv5.jl"))
using .FSAv5.SimulationV5: TRUTH_PARAMS_V5, TRUTH_PARAMS_V5_RECOMMENDED_V2,
                            FROZEN_PARAMS_V5, FROZEN_PARAMS_V5_RECOMMENDED_V2,
                            DEFAULT_OBS_PARAMS_V5,
                            SEDENTARY_INIT, TRAINED_ATHLETE_INIT_V2, MIDDLE_INIT_V2,
                            select_truth_preset
using .FSAv5.EstimationV5: PARAM_NAMES_V5,
                            PARAM_PRIOR_CONFIG_V5,
                            PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2,
                            build_prior_config_v5,
                            select_prior_config_v5
using .FSAv5.PlantV5: init_plant_state_from

# `posterior_mean_v5` is defined in bench_glue_v5.jl (not currently a
# member of any sub-module, included into the bench driver's namespace
# at runtime). For the test it's enough to include the same file at
# top-level — it picks up PARAM_NAMES_V5 / FROZEN_PARAMS_V5 from the
# `using` lines above, and Statistics.mean from the import below.
include(joinpath(@__DIR__, "..", "models", "fsa_v5", "bench_glue_v5.jl"))


@testset "v2 truth/frozen dispatch — regression tests" begin

    # =========================================================================
    # 1. Constants themselves match the PDF and reference_numerics.jl
    # =========================================================================
    @testset "TRUTH_PARAMS_V5_RECOMMENDED_V2 — 8 v2 overrides correct" begin
        # The 8 numerical entries that differ from canonical (PDF §2.3).
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:tau_B]   == 21.0f0
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:kappa_B] ≈ 0.02496f0  atol=1f-6
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:tau_S]   == 30.0f0
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:kappa_S] ≈ 0.01632f0  atol=1f-6
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:B_dec]   == 0.25f0
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:S_dec]   == 0.25f0
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:mu_F]    == 0.030f0
        @test TRUTH_PARAMS_V5_RECOMMENDED_V2[:mu_FF]   == 0.020f0
    end

    @testset "TRUTH_PARAMS_V5_RECOMMENDED_V2 — all other 20 keys inherit unchanged" begin
        # Every key NOT in the 8-override list must equal canonical.
        overridden = (:tau_B, :kappa_B, :tau_S, :kappa_S,
                       :B_dec, :S_dec, :mu_F, :mu_FF)
        for k in keys(TRUTH_PARAMS_V5)
            if !(k in overridden)
                @test TRUTH_PARAMS_V5_RECOMMENDED_V2[k] == TRUTH_PARAMS_V5[k]
            end
        end
    end

    @testset "FROZEN_PARAMS_V5_RECOMMENDED_V2 — B_dec/S_dec match v2, others match canonical" begin
        @test FROZEN_PARAMS_V5_RECOMMENDED_V2[:B_dec] == 0.25f0
        @test FROZEN_PARAMS_V5_RECOMMENDED_V2[:S_dec] == 0.25f0
        # The other 11 frozen entries don't change under v2.
        for k in keys(FROZEN_PARAMS_V5)
            if k != :B_dec && k != :S_dec
                @test FROZEN_PARAMS_V5_RECOMMENDED_V2[k] == FROZEN_PARAMS_V5[k]
            end
        end
    end

    @testset "FROZEN_PARAMS_V5_RECOMMENDED_V2 — canonical leak guard" begin
        # If someone accidentally redefines FROZEN_PARAMS_V5_RECOMMENDED_V2
        # to inherit B_dec/S_dec from canonical (e.g. by copying lines and
        # not editing them), this test catches it.
        @test FROZEN_PARAMS_V5_RECOMMENDED_V2[:B_dec] != FROZEN_PARAMS_V5[:B_dec]
        @test FROZEN_PARAMS_V5_RECOMMENDED_V2[:S_dec] != FROZEN_PARAMS_V5[:S_dec]
    end

    # =========================================================================
    # 2. The dispatch helper returns the right (truth, frozen) PAIR
    # =========================================================================
    @testset "select_truth_preset(\"canonical\")" begin
        truth, frozen = select_truth_preset("canonical")
        @test truth  === TRUTH_PARAMS_V5
        @test frozen === FROZEN_PARAMS_V5
        @test frozen[:B_dec] == 0.07f0
    end

    @testset "select_truth_preset(\"v2\")" begin
        truth, frozen = select_truth_preset("v2")
        @test truth  === TRUTH_PARAMS_V5_RECOMMENDED_V2
        @test frozen === FROZEN_PARAMS_V5_RECOMMENDED_V2
        @test frozen[:B_dec] == 0.25f0
        @test frozen[:S_dec] == 0.25f0
        # The cardinal regression check: the truth dict's B_dec MUST equal
        # the frozen dict's B_dec. The 2026-05-11 bug was exactly this
        # invariant being violated (truth was v2 = 0.25, frozen was
        # canonical = 0.07).
        @test truth[:B_dec] == frozen[:B_dec]
        @test truth[:S_dec] == frozen[:S_dec]
    end

    @testset "select_truth_preset rejects unknown presets" begin
        # Silently falling back to canonical would re-introduce the kind of
        # bug this dispatch is here to prevent. Errors must be explicit.
        @test_throws ErrorException select_truth_preset("V2")        # case wrong
        @test_throws ErrorException select_truth_preset("recommended_v2")
        @test_throws ErrorException select_truth_preset("")
        @test_throws ErrorException select_truth_preset("canonicl")  # typo
    end

    # =========================================================================
    # 3. posterior_mean_v5 honours the `frozen` kwarg
    # =========================================================================
    @testset "posterior_mean_v5 — default frozen kwarg is canonical" begin
        # Build a degenerate posterior with one chain at all zeros (so the
        # unconstrained mean is 0, constrained mean = exp(0) = 1.0 per key).
        U_post = zeros(Float64, 1, length(PARAM_NAMES_V5))
        out = posterior_mean_v5(U_post)              # no frozen kwarg
        @test out[:B_dec]  == FROZEN_PARAMS_V5[:B_dec]   # i.e. 0.07
        @test out[:S_dec]  == FROZEN_PARAMS_V5[:S_dec]
    end

    @testset "posterior_mean_v5 — explicit frozen=FROZEN_..._V2 routes B_dec/S_dec=0.25" begin
        U_post = zeros(Float64, 1, length(PARAM_NAMES_V5))
        out = posterior_mean_v5(U_post;
                                  frozen = FROZEN_PARAMS_V5_RECOMMENDED_V2)
        @test out[:B_dec]    == 0.25f0
        @test out[:S_dec]    == 0.25f0
        @test out[:mu_dec_B] == FROZEN_PARAMS_V5_RECOMMENDED_V2[:mu_dec_B]
        @test out[:sigma_B]  == FROZEN_PARAMS_V5_RECOMMENDED_V2[:sigma_B]
        # All 13 frozen keys must show up.
        for k in keys(FROZEN_PARAMS_V5_RECOMMENDED_V2)
            @test out[k] == FROZEN_PARAMS_V5_RECOMMENDED_V2[k]
        end
    end

    @testset "posterior_mean_v5 — the dispatch round-trip (the real chain)" begin
        # This is the chain the bench actually executes per replan:
        #   truth_str ── select_truth_preset ──> frozen
        #   posterior_mean_v5(U_post; frozen=frozen) ──> dict the controller plans against
        # If any link breaks, the controller sees the wrong B_dec.
        U_post = zeros(Float64, 1, length(PARAM_NAMES_V5))

        _, frozen_v2 = select_truth_preset("v2")
        params_post_v2 = posterior_mean_v5(U_post; frozen = frozen_v2)
        @test params_post_v2[:B_dec] == 0.25f0
        @test params_post_v2[:S_dec] == 0.25f0

        _, frozen_can = select_truth_preset("canonical")
        params_post_can = posterior_mean_v5(U_post; frozen = frozen_can)
        @test params_post_can[:B_dec] == 0.07f0
        @test params_post_can[:S_dec] == 0.07f0
    end

    # =========================================================================
    # 4. MIDDLE_INIT_V2 is the per-component midpoint and is consistent with
    #    its constituents
    # =========================================================================
    @testset "MIDDLE_INIT_V2 — per-component average of SEDENTARY_INIT and TRAINED_ATHLETE_INIT_V2" begin
        for k in (:B, :S, :F, :A, :KFB, :KFS)
            expected = (getfield(SEDENTARY_INIT, k) +
                         getfield(TRAINED_ATHLETE_INIT_V2, k)) / 2.0f0
            @test getfield(MIDDLE_INIT_V2, k) ≈ expected  atol=1f-7
        end
        # And in particular the user-visible target: A_middle should sit
        # well above the sedentary basin's A=0.10 but below the trained A*.
        @test MIDDLE_INIT_V2.A > SEDENTARY_INIT.A
        @test MIDDLE_INIT_V2.A < TRAINED_ATHLETE_INIT_V2.A
        # Same shape: every component must lie strictly between the two
        # constituents (since both are finite and distinct on each axis).
        for k in (:B, :S, :F, :A, :KFB, :KFS)
            sed = getfield(SEDENTARY_INIT, k)
            tra = getfield(TRAINED_ATHLETE_INIT_V2, k)
            mid = getfield(MIDDLE_INIT_V2, k)
            @test min(sed, tra) < mid < max(sed, tra)
        end
    end

    @testset "init_plant_state_from — fn ↔ NamedTuple agreement (regression)" begin
        # The 2026-05-11 secondary bug: bench dispatch returned a parameter-
        # less `init_plant_state_trained` alongside `TRAINED_ATHLETE_INIT_V2`,
        # so the plant was actually initialised at CANONICAL trained values
        # while the rest of the bench saw v2 values via init_nt. This test
        # exercises the parameterised constructor that fixes that — feeding
        # in MIDDLE_INIT_V2 must produce a plant state whose 6 fields equal
        # MIDDLE_INIT_V2's 6 fields exactly.
        ps = init_plant_state_from(MIDDLE_INIT_V2)
        @test ps.state[1] == Float32(MIDDLE_INIT_V2.B)
        @test ps.state[2] == Float32(MIDDLE_INIT_V2.S)
        @test ps.state[3] == Float32(MIDDLE_INIT_V2.F)
        @test ps.state[4] == Float32(MIDDLE_INIT_V2.A)
        @test ps.state[5] == Float32(MIDDLE_INIT_V2.KFB)
        @test ps.state[6] == Float32(MIDDLE_INIT_V2.KFS)
        # Same check for each other init NamedTuple — guards against
        # someone hardcoding a single init source inside the constructor.
        for nt in (SEDENTARY_INIT, TRAINED_ATHLETE_INIT_V2)
            ps = init_plant_state_from(nt)
            @test ps.state[1] == Float32(nt.B)
            @test ps.state[4] == Float32(nt.A)
            @test ps.state[6] == Float32(nt.KFS)
        end
    end

    # =========================================================================
    # 5. Prior config — v2 prior tracks v2 truth + canonical/v2 dispatch
    # =========================================================================
    @testset "PARAM_PRIOR_CONFIG_V5 — canonical prior centres on canonical truth" begin
        # The 4-tuple format is (name, kind, μ, σ). For LogNormal priors,
        # μ = log(truth). So exp(μ) should equal the canonical truth value
        # for every estimated dynamics key.
        for (name, kind, μ, σ) in PARAM_PRIOR_CONFIG_V5
            @test kind === :LogNormal
            @test σ ≈ 0.30  atol=1e-9
            if haskey(TRUTH_PARAMS_V5, name)
                @test exp(μ) ≈ TRUTH_PARAMS_V5[name]  rtol=1e-5
            end
        end
    end

    @testset "PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2 — v2 prior centres on v2 truth" begin
        for (name, kind, μ, σ) in PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2
            @test kind === :LogNormal
            @test σ ≈ 0.30  atol=1e-9
            if haskey(TRUTH_PARAMS_V5_RECOMMENDED_V2, name)
                @test exp(μ) ≈ TRUTH_PARAMS_V5_RECOMMENDED_V2[name]  rtol=1e-5
            end
        end
    end

    @testset "PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2 — six v2 keys differ from canonical" begin
        # 6 of the 8 v2 overrides are estimated (B_dec/S_dec are frozen).
        # Their prior centres must differ between canonical and v2.
        v2_changed_estimated = (:tau_B, :kappa_B, :tau_S, :kappa_S, :mu_F, :mu_FF)
        canonical_by_name = Dict(name => μ for (name, _, μ, _) in PARAM_PRIOR_CONFIG_V5)
        v2_by_name        = Dict(name => μ for (name, _, μ, _) in PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2)
        for name in v2_changed_estimated
            @test canonical_by_name[name] != v2_by_name[name]
        end
        # Every key NOT in that set must have identical prior centre under
        # both presets (catches accidental drift in unrelated entries).
        for name in PARAM_NAMES_V5
            if !(name in v2_changed_estimated)
                @test canonical_by_name[name] ≈ v2_by_name[name]  atol=1e-12
            end
        end
    end

    @testset "select_prior_config_v5 — string dispatch" begin
        @test select_prior_config_v5("canonical") === PARAM_PRIOR_CONFIG_V5
        @test select_prior_config_v5("v2")         === PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2
        @test_throws ErrorException select_prior_config_v5("V2")
        @test_throws ErrorException select_prior_config_v5("recommended_v2")
        @test_throws ErrorException select_prior_config_v5("")
    end

    @testset "build_prior_config_v5 — helper is consistent with the named consts" begin
        # Building from canonical TRUTH_PARAMS_V5 must reproduce the canonical
        # const exactly (otherwise the refactor introduced an off-by-something).
        rebuilt_canonical = build_prior_config_v5(TRUTH_PARAMS_V5)
        @test rebuilt_canonical == PARAM_PRIOR_CONFIG_V5
        rebuilt_v2 = build_prior_config_v5(TRUTH_PARAMS_V5_RECOMMENDED_V2)
        @test rebuilt_v2 == PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2
    end

    @testset "Dispatch consistency: select_truth_preset ↔ select_prior_config_v5" begin
        # The cardinal invariant for the v2 wiring: for the same --truth-preset
        # string, the prior centres for every ESTIMATED dynamics key must equal
        # log(truth[k]). If this fails, the filter cold-starts off-centre.
        for preset in ("canonical", "v2")
            truth, _frozen = select_truth_preset(preset)
            prior          = select_prior_config_v5(preset)
            for (name, kind, μ, σ) in prior
                if haskey(truth, name)
                    @test exp(μ) ≈ truth[name]  rtol=1e-5
                end
            end
        end
    end

end  # outer @testset
