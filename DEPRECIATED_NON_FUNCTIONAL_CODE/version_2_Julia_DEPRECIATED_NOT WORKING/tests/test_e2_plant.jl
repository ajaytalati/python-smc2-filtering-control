# Stage E2 verification tests — port of `version_2/tests/test_e2_plant.py`.
#
#   1. Φ-burst integral preservation: ∫_day Φ_subdaily dt = 24 · Φ_daily.
#   2. Φ-burst shape correctness (peak around 10am, zero overnight).
#   3. StepwisePlant produces in-bounds trajectories under stride composition.
#   4. Subsequent advance!() calls accumulate t_bin correctly.
#   5. plant.finalise produces a psim-format artifact.
#
# Assumes FSA_STEP_MINUTES = 15 (default) so BINS_PER_DAY = 96.

using Test

# Ensure BINS_PER_DAY = 96 for these tests.
delete!(ENV, "FSA_STEP_MINUTES")

# Add the model dir to LOAD_PATH for `using FSAHighRes` style.
const MODEL_DIR = abspath(joinpath(@__DIR__, "..", "models", "fsa_high_res"))

isdefined(Main, :FSAHighRes) || include(joinpath(MODEL_DIR, "FSAHighRes.jl"))
const FSA = Main.FSAHighRes
const BINS_PER_DAY  = FSA.PhiBurst.BINS_PER_DAY
const DT_BIN_HOURS  = FSA.PhiBurst.DT_BIN_HOURS
const expand_daily_phi_to_subdaily = FSA.PhiBurst.expand_daily_phi_to_subdaily
const StepwisePlant = FSA.Plant.StepwisePlant
const advance!      = FSA.Plant.advance!
const finalise      = FSA.Plant.finalise

import JSON3
import NPZ


@testset "test_e2_plant" begin

    @testset "phi_burst_integral_preserved" begin
        daily = [0.0, 0.5, 1.0, 1.5, 2.5, 0.05]
        out = expand_daily_phi_to_subdaily(daily)
        @test length(out) == length(daily) * BINS_PER_DAY
        for (d, phi_d) in enumerate(daily)
            slice = out[(d - 1) * BINS_PER_DAY + 1 : d * BINS_PER_DAY]
            integral = sum(slice .* DT_BIN_HOURS)
            @test isapprox(integral, 24.0 * phi_d; atol=1e-3)
        end
    end

    @testset "phi_burst_zero_overnight_peak_morning" begin
        # Assumes BINS_PER_DAY=96, so 1 bin = 15 min, k=0 is 00:00.
        out = expand_daily_phi_to_subdaily([1.0])
        @test BINS_PER_DAY == 96
        # Index k corresponds to hour k * 0.25 (15-min bins). Julia 1-based.
        @test out[12 + 1] == 0.0f0    # 03:00 (sleep)
        @test out[28 + 1] == 0.0f0    # 07:00 (just woke up, t_post=0)
        @test out[40 + 1] > 1.0       # 10:00 (Gamma peak)
        @test out[92 + 1] == 0.0f0    # 23:00 (asleep)
    end

    @testset "stepwise_in_bounds_trajectories" begin
        # 2-day plan, stride = 1 day (96 bins).
        plant_single = StepwisePlant(seed_offset=99)
        out_single = advance!(plant_single, 2 * 96, [1.0, 1.5])
        @test plant_single.t_bin == 192

        plant_step = StepwisePlant(seed_offset=99)
        out_a = advance!(plant_step, 96, [1.0])
        out_b = advance!(plant_step, 96, [1.5])
        @test plant_step.t_bin == 192

        # Both runs produce in-bounds trajectories.
        @test size(plant_single.history[:trajectory][1]) == (192, 3)
        @test size(plant_step.history[:trajectory][1]) == (96, 3)
        @test size(plant_step.history[:trajectory][2]) == (96, 3)
        for traj in (plant_single.history[:trajectory][1],
                     plant_step.history[:trajectory][1],
                     plant_step.history[:trajectory][2])
            @test all(0.0 .<= traj[:, 1] .<= 1.0)
            @test all(traj[:, 2] .>= 0.0)
            @test all(traj[:, 3] .>= 0.0)
        end
    end

    @testset "stepwise_advances_global_bin_correctly" begin
        plant = StepwisePlant(seed_offset=7)
        out1 = advance!(plant, 48, [1.0])    # 12-hour stride
        @test plant.t_bin == 48
        out2 = advance!(plant, 48, [1.0])
        @test plant.t_bin == 96
        out3 = advance!(plant, 96, [1.0])
        @test plant.t_bin == 192

        # Global bin indices: 0-based per the Python convention.
        for i in 0:47
            @test out1.Phi.t_idx[i + 1] == Int32(i)
            @test out2.Phi.t_idx[i + 1] == Int32(i + 48)
        end
        for i in 0:95
            @test out3.Phi.t_idx[i + 1] == Int32(i + 96)
        end
    end

    @testset "finalise_writes_psim_artifact" begin
        plant = StepwisePlant(seed_offset=11)
        advance!(plant, 96, [1.0])
        advance!(plant, 96, [1.0])

        mktempdir() do tmp
            out_dir = finalise(plant, tmp; scenario_name="test_artifact")
            @test isfile(joinpath(out_dir, "manifest.json"))
            @test isfile(joinpath(out_dir, "trajectory.npz"))
            @test isfile(joinpath(out_dir, "obs", "obs_HR.npz"))
            @test isfile(joinpath(out_dir, "obs", "obs_sleep.npz"))
            @test isfile(joinpath(out_dir, "obs", "obs_stress.npz"))
            @test isfile(joinpath(out_dir, "obs", "obs_steps.npz"))
            @test isfile(joinpath(out_dir, "exogenous", "Phi.npz"))
            @test isfile(joinpath(out_dir, "exogenous", "C.npz"))

            manifest = JSON3.read(read(joinpath(out_dir, "manifest.json"), String))
            @test String(manifest.schema_version) == "1.0"
            @test String(manifest.model_name) == "fsa_high_res_v2"
            @test manifest.n_bins_total == 192
            @test manifest.validation_summary.closed_loop == true
            @test manifest.validation_summary.stepwise_advances == 2

            traj = NPZ.npzread(joinpath(out_dir, "trajectory.npz"))
            @test size(traj["trajectory"]) == (192, 3)
        end
    end

end
