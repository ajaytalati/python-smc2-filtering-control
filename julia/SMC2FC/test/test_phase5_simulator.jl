@testset "Phase 5 — Plant + Simulator" begin
    using Statistics: mean, var

    @testset "simulate_sde: 1-D OU process" begin
        # dx = -θ x dt + σ dW, x(0) = x0
        # Stationary distribution: N(0, σ²/(2θ))
        θ_truth = 1.0
        σ_truth = 0.4

        function drift!(du, u, p, t)
            du[1] = -p.θ * u[1]
        end
        function diff!(du, u, p, t)
            du[1] = p.σ
        end

        u0    = [0.0]
        tspan = (0.0, 200.0)
        params = (θ = θ_truth, σ = σ_truth)
        sol = simulate_sde(drift!, diff!, u0, tspan, params;
                            dt = 0.05, saveat = 0.5, seed = 42)

        traj = reduce(hcat, sol.u)'   # (T, 1)
        # Burn-in: discard first 20 % of samples to let chain reach equilibrium.
        post_burn = traj[Int(floor(0.2 * size(traj, 1))):end, 1]

        # Stationary variance σ²/(2θ) = 0.16/2 = 0.08
        @test abs(mean(post_burn))     < 0.1
        @test abs(var(post_burn) - 0.08) < 0.05
    end

    @testset "Observation channels: dependency-ordered generation" begin
        # 3-channel toy: latent → core → derived.
        T = 50
        traj   = randn(T, 2)
        t_grid = collect(0.0:0.1:T*0.1 - 0.1)

        gen_latent(traj, t_grid, p, aux, prior, rng)  =
            (x = traj[:, 1] .+ 0.01 .* randn(rng, T),)
        gen_core(traj, t_grid, p, aux, prior, rng)    =
            (y = prior[:latent].x .* 2.0,)
        gen_derived(traj, t_grid, p, aux, prior, rng) =
            (z = prior[:core].y .+ prior[:latent].x,)

        channels = [
            ObsChannel(name = :latent,  generate_fn = gen_latent),
            # `derived` depends on both `core` and `latent`
            ObsChannel(name = :derived, depends_on = [:core, :latent],
                                          generate_fn = gen_derived),
            ObsChannel(name = :core,    depends_on = [:latent],
                                          generate_fn = gen_core),
        ]

        out = generate_all_channels(channels, traj, t_grid, nothing; seed=7)
        @test haskey(out, :latent) && haskey(out, :core) && haskey(out, :derived)
        @test out[:core].y ≈ out[:latent].x .* 2.0
        @test out[:derived].z ≈ out[:core].y .+ out[:latent].x
    end

    @testset "Cyclic dependency raises error" begin
        ch = [
            ObsChannel(name = :a, depends_on = [:b], generate_fn = (args...) -> 0),
            ObsChannel(name = :b, depends_on = [:a], generate_fn = (args...) -> 0),
        ]
        @test_throws ErrorException generate_all_channels(
            ch, randn(2, 1), [0.0, 1.0], nothing; seed=0)
    end
end
