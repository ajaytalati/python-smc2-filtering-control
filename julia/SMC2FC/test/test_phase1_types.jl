@testset "Phase 1 — Types" begin
    using StaticArrays: SVector
    using ComponentArrays: ComponentVector
    using StructArrays: StructArray
    using CUDA: CuArray, CUDA

    @testset "State / DynParams / ParticleCloud" begin
        s = State{6,Float64}(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        @test s isa SVector{6,Float64}
        @test length(s) == 6

        # ComponentVector with named fields
        p = ComponentVector(k_FB=1.0, k_FS=2.0, σ_HR=4.5)
        @test p isa DynParams
        @test p.k_FB == 1.0 && p.k_FS == 2.0 && p.σ_HR == 4.5

        cloud = StructArray([Particle(SVector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), -log(10.0))
                             for _ in 1:10])
        @test cloud isa ParticleCloud{6,Float64}
        @test length(cloud) == 10
    end

    @testset "Hybrid type boundaries (charter §14.4)" begin
        # CPU side always works
        cpu = CPUParameterCloud(rand(8, 37), zeros(8))
        @test cpu.theta isa Matrix{Float64}
        @test cpu.log_lik isa Vector{Float64}

        # GPU side only if CUDA is functional
        if CUDA.functional()
            d_state, n_pf = 6, 32
            gpu = GPUFilterState(CUDA.zeros(Float32, n_pf, d_state),
                                 CUDA.zeros(Float32, n_pf))
            @test gpu.particles isa CuArray{Float32,2}
            @test gpu.weights   isa CuArray{Float32,1}
            @test size(gpu.particles) == (n_pf, d_state)
        else
            @info "CUDA not functional — skipping GPU type-boundary test"
        end
    end

    @testset "Marker types dispatch" begin
        # Verify the marker types are distinct types so multiple-dispatch works
        @test GaussianBridge() !== SchrodingerFollmerBridge()
        @test typeof(GaussianBridge()) != typeof(SchrodingerFollmerBridge())
        @test SoftSurrogate() isa ChanceConstraintMode
        @test HardIndicator() isa ChanceConstraintMode
        @test CPUBackend() isa AbstractBackend
        @test CUDABackend() isa AbstractBackend
    end
end
