# Smoke tests — port of `version_3/tests/test_fsa_v5_smoke.py`.
#
# Confirms the package's exports import cleanly, the plant forward
# pipeline runs, and the chance-constrained cost evaluators compile.

using FSAv5
using Test
using SMC2FC: HardIndicator, SoftSurrogate

@testset "FSA-v5 smoke" begin

    @testset "imports clean" begin
        # Public surface from each module — confirms exports are wired.
        @test isdefined(FSAv5, :FSAv5State)
        @test isdefined(FSAv5, :BimodalPhi)
        @test isdefined(FSAv5, :DynParams)
        @test isdefined(FSAv5, :ObsParams)
        @test isdefined(FSAv5, :TRUTH_PARAMS_V5)
        @test isdefined(FSAv5, :DEFAULT_OBS_PARAMS_V5)
        @test isdefined(FSAv5, :DEFAULT_INIT)
        @test isdefined(FSAv5, :HIGH_RES_FSA_V5_ESTIMATION)
        for fn in (:dynamics_drift, :dynamics_diffusion,
                   :mu_bar, :find_a_sep, :a_sep_grid, :chance_indicator,
                   :design_matrix, :schedule_from_theta,
                   :em_step, :StepwisePlant, :advance!,
                   :hr_mean, :sleep_prob, :stress_mean,
                   :steps_log_mean, :volume_load_mean,
                   :expand_daily_phi_to_subdaily)
            @test isdefined(FSAv5, fn)
        end
    end

    @testset "plant forward pipeline" begin
        p = FSAv5.TRUTH_PARAMS_V5
        sigma_diag = [p.sigma_B, p.sigma_S, p.sigma_F,
                       p.sigma_A, p.sigma_K, p.sigma_K]
        plant = StepwisePlant(state = FSAv5.DEFAULT_INIT,
                               params = p, sigma_diag = sigma_diag,
                               dt = DT_BIN_DAYS, seed_offset = 7)
        phi_daily = reshape([0.30, 0.30], 1, 2)
        traj = advance!(plant, BINS_PER_DAY, phi_daily)
        @test size(traj) == (BINS_PER_DAY, 6)
        @test plant.t_bin == BINS_PER_DAY
        # All trajectory values finite + within bounds
        @test all(isfinite, traj)
        @test all(0.0 .<= traj[:, 1] .<= 1.0)        # B
        @test all(0.0 .<= traj[:, 2] .<= 1.0)        # S
        @test all(traj[:, 3:6] .>= 0.0)              # F, A, K_FB, K_FS
    end

    @testset "chance-constrained cost smoke" begin
        # 3-particle ensemble, 8-step schedule
        p_base   = FSAv5.TRUTH_PARAMS_V5
        particles = [p_base, copy(p_base), copy(p_base)]
        schedule  = [BimodalPhi(0.30, 0.30) for _ in 1:8]
        sep_pp    = a_sep_grid(particles, schedule)
        @test size(sep_pp) == (3, 8)

        # Mock A_traj: A=0.5 everywhere
        A_traj_pp = fill(0.5, 3, 8)

        # Hard indicator: at healthy island A_sep = -inf, no violations
        ind_h = chance_indicator(HardIndicator(), A_traj_pp, sep_pp)
        @test all(==(0.0), ind_h)

        # Soft indicator: same expectation, sigmoid handles -inf cleanly
        ind_s = chance_indicator(SoftSurrogate(), A_traj_pp, sep_pp;
                                  beta=50.0, scale=0.1)
        @test all(<(1e-9), ind_s)
    end

    @testset "soft-to-hard limit" begin
        # As beta → ∞, soft indicator approaches the hard one. Pick a
        # bistable point so A_sep is finite.
        particles = [FSAv5.TRUTH_PARAMS_V5]
        schedule  = [BimodalPhi(0.45, 0.30)]
        sep_pp    = a_sep_grid(particles, schedule)
        @test all(isfinite, sep_pp)

        # A above the separator → no violation
        A_above = fill(sep_pp[1, 1] + 0.1, 1, 1)
        ind_h_above = chance_indicator(HardIndicator(), A_above, sep_pp)
        ind_s_above = chance_indicator(SoftSurrogate(), A_above, sep_pp;
                                        beta=1000.0, scale=0.01)
        @test ind_h_above[1,1] == 0.0
        @test ind_s_above[1,1] < 1e-3

        # A below the separator → violation
        A_below = fill(sep_pp[1, 1] - 0.05, 1, 1)
        ind_h_below = chance_indicator(HardIndicator(), A_below, sep_pp)
        ind_s_below = chance_indicator(SoftSurrogate(), A_below, sep_pp;
                                        beta=1000.0, scale=0.01)
        @test ind_h_below[1,1] == 1.0
        @test ind_s_below[1,1] > 1.0 - 1e-3
    end

    @testset "EstimationModel cardinality" begin
        using SMC2FC: n_params
        m = FSAv5.HIGH_RES_FSA_V5_ESTIMATION
        @test n_params(m) == 37
        @test m.n_states == 6
        @test length(m.frozen_params) == 14   # 6 sigmas+phi + 8 frozen dyn
    end
end
