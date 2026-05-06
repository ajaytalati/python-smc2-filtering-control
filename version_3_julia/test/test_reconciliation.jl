# Plant ↔ Estimator reconciliation (mirror test).
#
# Port of `version_3/tests/test_reconciliation_v5.py`. Confirms the
# plant's `em_step` and the estimator's `propagate_fn_v5` agree on
# a single-bin update — i.e. they share the same drift implementation
# (no inlining of stale formulae).

using FSAv5
using Test
using Random: MersenneTwister

@testset "Plant ↔ Estimator reconciliation" begin
    p = FSAv5.TRUTH_PARAMS_V5
    sigma_diag = [p.sigma_B, p.sigma_S, p.sigma_F,
                   p.sigma_A, p.sigma_K, p.sigma_K]
    dt = DT_BIN_DAYS
    y0 = FSAv5.DEFAULT_INIT
    phi = BimodalPhi(0.30, 0.30)

    # ── Step via plant.em_step (deterministic, given noise) ─────────────
    rng = MersenneTwister(42)
    noise = randn(rng, 6)
    y_plant = em_step(y0, phi, p, sigma_diag, dt, noise)

    # ── Step via Estimation.propagate_fn_v5 (the SMC2FC contract) ───────
    # Build a θ vector from the truth params (matching PARAM_PRIORS_V5 order)
    theta = Float64[]
    for sym in FSAv5.PK_V5
        if hasproperty(p, sym)
            push!(theta, getproperty(p, sym))
        else
            # Obs-channel keys live on ObsParams, not DynParams
            op = FSAv5.DEFAULT_OBS_PARAMS_V5
            push!(theta, getproperty(op, sym))
        end
    end

    grid_obs = Dict(:Phi => reshape([phi.Phi_B, phi.Phi_S], 1, 2),
                     :C   => [0.0])

    y_vec_in = Float64[y0.B, y0.S, y0.F, y0.A, y0.KFB, y0.KFS]
    x_new, pred_lw = FSAv5.propagate_fn_v5(y_vec_in, 0.0, dt, theta,
                                              grid_obs, 1, sigma_diag,
                                              noise, MersenneTwister(0))

    # Both pipelines route through the same drift; same noise →
    # bit-identical state update.
    @test isapprox(y_plant.B,   x_new[1]; atol = 1e-12)
    @test isapprox(y_plant.S,   x_new[2]; atol = 1e-12)
    @test isapprox(y_plant.F,   x_new[3]; atol = 1e-12)
    @test isapprox(y_plant.A,   x_new[4]; atol = 1e-12)
    @test isapprox(y_plant.KFB, x_new[5]; atol = 1e-12)
    @test isapprox(y_plant.KFS, x_new[6]; atol = 1e-12)
    @test pred_lw == 0.0   # bootstrap PF → no Kalman correction
end
