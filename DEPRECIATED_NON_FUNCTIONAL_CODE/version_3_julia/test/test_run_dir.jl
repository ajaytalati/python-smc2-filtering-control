# Tests for the atomic run-directory allocator (port of
# `version_3/tests/test_run_dir_atomic.py`).
#
# Targets the same TOCTOU race the Python helper had: two concurrent
# processes both compute N from max-existing+1, both call
# mkdir(exist_ok=True), and the second silently writes into the
# first's directory. The Julia allocator uses `mkdir` with retry
# (Julia's `mkdir` raises if the dir already exists — that's the
# atomic flag).

using Test

# Load the tools module — it's not part of the FSAv5 package, but it
# does live alongside it. We just include the file directly.
include(joinpath(@__DIR__, "..", "tools", "_RunDir.jl"))
using .RunDir: scan_max_run_number, allocate_run_dir

@testset "atomic run_dir allocator" begin
    @testset "first allocation returns runNN_<tag>" begin
        mktempdir() do tmp
            out_dir, n = allocate_run_dir(tmp, "smoke")
            @test n == 1
            @test endswith(out_dir, "run01_smoke")
            @test isdir(out_dir)
        end
    end

    @testset "consecutive allocations increment" begin
        mktempdir() do tmp
            d1, n1 = allocate_run_dir(tmp, "alpha")
            d2, n2 = allocate_run_dir(tmp, "beta")
            d3, n3 = allocate_run_dir(tmp, "gamma")
            @test (n1, n2, n3) == (1, 2, 3)
            @test d1 != d2 != d3
            @test all(isdir, (d1, d2, d3))
        end
    end

    @testset "concurrent same-tag allocations get distinct paths" begin
        mktempdir() do tmp
            n_workers = 8
            results = Vector{Tuple{String,Int}}(undef, n_workers)
            tasks = [Threads.@spawn allocate_run_dir(tmp, "shared_tag") for _ in 1:n_workers]
            for (i, t) in enumerate(tasks)
                results[i] = fetch(t)
            end
            paths = [r[1] for r in results]
            nums  = [r[2] for r in results]
            @test length(unique(paths)) == n_workers
            @test length(unique(nums))  == n_workers
        end
    end

    @testset "scan_max_run_number ignores non-runNN dirs" begin
        mktempdir() do tmp
            exp = joinpath(tmp, "outputs", "fsa_v5", "experiments")
            mkpath(exp)
            mkdir(joinpath(exp, "run01_a"))
            mkdir(joinpath(exp, "run42_b"))
            mkdir(joinpath(exp, "not_a_run"))
            mkdir(joinpath(exp, "run_garbage_no_number"))
            @test scan_max_run_number(exp) == 42
        end
    end
end
