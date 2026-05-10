# Test orchestrator for the FSA-v5 Julia port.
#
# Per the joint LEAN4-and-Julia charter audit checklist (Part II §18):
#   - Pkg.test() must pass on both CPU and CUDA paths.
#   - Differential tests against the LEAN binary AND Python pass at 1e-6.
#   - JET.jl reports 0 type uncertainties.
#
# This file orchestrates all the above.

using Test
using FSAv5

@testset "FSA-v5 Julia port" begin
    include("test_param_dict.jl")
    include("test_smoke.jl")
    include("test_per_particle_separator.jl")
    include("test_reconciliation.jl")
    include("test_run_dir.jl")
    include("test_lean_diff.jl")
    include("test_python_diff.jl")
end
