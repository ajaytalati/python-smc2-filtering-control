# version_2_Julia test runner.
#
# Usage:
#     cd version_2_Julia
#     julia --project=. -e 'using Pkg; Pkg.test()'
#
# Or to run a single test file:
#     julia --project=. tests/test_e2_plant.jl
#
# Tests assume FSA_STEP_MINUTES=15 (default) so BINS_PER_DAY=96.

using Test

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

# Load FSAHighRes once into Main; each test file picks up the cached
# module via `Main.FSAHighRes` and avoids the
# "WARNING: replacing module FSAHighRes" / namespace-conflict spiral.
delete!(ENV, "FSA_STEP_MINUTES")    # default 15-min bins for tests
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))

@testset "version_2_Julia FSA-v2 port" begin
    include("test_g1_reparam.jl")
    include("test_e2_plant.jl")
    include("test_obs_consistency_fsa.jl")
    include("test_estimation_smoke.jl")
    include("test_ot_parity.jl")
end
