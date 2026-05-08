"""
    runtests.jl

Top-level test entry point for `SMC2FC_functional`. Each individual test
file `include`s `fixtures.jl` so they can run in isolation.
"""

using SMC2FC_functional
using Test
using Random

@testset "SMC2FC_functional — unit tests" begin
    include("test_transforms.jl")
    include("test_filtering.jl")
    include("test_smc2.jl")
    include("test_control.jl")
end

# The end-to-end / Python-comparison harness lives in
# `benchmarks/compare_three_libraries.jl` — it is not part of the unit
# test suite because it requires Python + smc2fc on PATH.
