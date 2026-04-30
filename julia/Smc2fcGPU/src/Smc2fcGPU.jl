module Smc2fcGPU

using CUDA
using StaticArrays

include("FSAv2.jl")
include("SDESolver.jl")
include("EnsembleGPU.jl")

export FsaParams, FsaInit, FSAv2_DEFAULT_PARAMS, FSAv2_DEFAULT_INIT
export expand_phi_lut, run_em_oracle!
export FsaSDEParams, run_em_ensemble_gpu

function __init__()
    if !CUDA.functional()
        @warn "Smc2fcGPU loaded without functional CUDA — kernels will not work."
    end
end

end # module
