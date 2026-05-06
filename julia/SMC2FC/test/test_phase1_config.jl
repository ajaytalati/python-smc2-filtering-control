@testset "Phase 1 — Config" begin
    @testset "SMCConfig defaults match Python" begin
        cfg = SMCConfig()
        # Spot-check: defaults must match smc2fc/core/config.py exactly
        @test cfg.n_smc_particles == 256
        @test cfg.target_ess_frac ≈ 0.5
        @test cfg.num_mcmc_steps == 5
        @test cfg.max_lambda_inc ≈ 0.05
        @test cfg.bridge_type == :gaussian
        @test cfg.sf_q1_mode == :is
        @test cfg.hmc_step_size ≈ 0.025
        @test cfg.hmc_num_leapfrog == 8
        @test cfg.n_pf_particles == 400
        @test cfg.bandwidth_scale ≈ 1.0
        @test cfg.ot_ess_frac ≈ 0.05
        @test cfg.ot_temperature ≈ 5.0
        @test cfg.backend == :cpu
    end

    @testset "RollingConfig defaults" begin
        cfg = RollingConfig()
        @test cfg.window_days == 120
        @test cfg.stride_days == 30
        @test cfg.dt ≈ 1.0
        @test cfg.n_substeps == 10
        @test cfg.max_windows === nothing
    end

    @testset "MissingDataConfig defaults" begin
        cfg = MissingDataConfig()
        @test cfg.dropout_rate ≈ 0.15
        @test cfg.broken_watch_days == 14
        @test cfg.rest_days_per_week == (2, 3)
    end

    @testset "Custom backend selection" begin
        cfg = SMCConfig(backend=:cuda, n_smc_particles=128)
        @test cfg.backend == :cuda
        @test cfg.n_smc_particles == 128
    end
end
