# Regression test for Bug 2 (particle-0 separator template).
#
# The historical Python `_compute_cost_internals` collapsed the SMC²
# particle ensemble to particle-0 before computing `A_sep`, then
# broadcast that single template across all particles' trajectories
# — mathematically wrong because the separator depends on each
# particle's bifurcation parameters.
#
# The Julia type signature `a_sep_grid : AbstractVector{<:DynParams}
# → AbstractVector{BimodalPhi} → Matrix{Float64}` of shape
# `(n_particles, n_steps)` makes the buggy collapse to a 1-D vector
# impossible to express. These tests confirm that runtime behaviour
# matches the type contract.

using FSAv5
using Test

# Construct two particles where bifurcation parameters DIVERGE so the
# separator differs.
function _two_particle_ensemble()
    base = FSAv5.TRUTH_PARAMS_V5
    p1 = base                                              # healthy at (0.30, 0.30)
    p2 = copy(base)
    p2.mu_0 = -10.0                                        # collapsed at any Phi
    return [p1, p2]
end

@testset "Bug 2: per-particle separator" begin
    @testset "find_a_sep differs per particle" begin
        particles = _two_particle_ensemble()
        phi = BimodalPhi(0.30, 0.30)
        a1 = find_a_sep(phi, particles[1])
        a2 = find_a_sep(phi, particles[2])
        @test a1 == -Inf       # particle 1: mono-stable healthy
        @test a2 == Inf        # particle 2: mono-stable collapsed (mu_0 = -10)
    end

    @testset "a_sep_grid returns (n_particles, n_steps)" begin
        particles = _two_particle_ensemble()
        schedule  = [BimodalPhi(0.30, 0.30) for _ in 1:8]
        grid      = a_sep_grid(particles, schedule)
        @test size(grid) == (2, 8)
        # Particle 1: all -inf (healthy); particle 2: all +inf (collapsed).
        @test all(==(-Inf), grid[1, :])
        @test all(==(Inf),  grid[2, :])
    end

    @testset "Hard indicator uses per-particle separator" begin
        using SMC2FC: HardIndicator
        particles = _two_particle_ensemble()
        schedule  = [BimodalPhi(0.30, 0.30) for _ in 1:16]
        sep_pp    = a_sep_grid(particles, schedule)
        # Trajectory: A=0.5 everywhere
        traj_pp   = fill(0.5, 2, 16)
        ind       = chance_indicator(HardIndicator(), traj_pp, sep_pp)
        # Particle 0 (healthy, A_sep=-inf): indicator = 0 everywhere.
        @test all(==(0.0), ind[1, :])
        # Particle 1 (collapsed, A_sep=+inf): indicator = 1 everywhere.
        @test all(==(1.0), ind[2, :])
    end

    @testset "Soft indicator uses per-particle separator" begin
        using SMC2FC: SoftSurrogate
        particles = _two_particle_ensemble()
        schedule  = [BimodalPhi(0.30, 0.30) for _ in 1:16]
        sep_pp    = a_sep_grid(particles, schedule)
        traj_pp   = fill(0.5, 2, 16)
        ind       = chance_indicator(SoftSurrogate(), traj_pp, sep_pp;
                                       beta=50.0, scale=0.1)
        # sigmoid(50*(-inf - 0.5)/0.1) → 0; sigmoid(50*(+inf - 0.5)/0.1) → 1
        @test all(<(1e-9), ind[1, :])
        @test all(>(1.0 - 1e-9), ind[2, :])
    end
end
