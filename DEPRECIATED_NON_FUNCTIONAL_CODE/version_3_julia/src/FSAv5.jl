"""
    FSAv5

Julia port of the FSA-v5 model for use with `SMC2FC.jl`.

The math is fixed by the LEAN4 reference at
`FSA_model_dev/lean/Fsa/V5/*.lean`; every function in this package
maps to a LEAN4 function with the same signature shape and is
differentially tested against the LEAN4 binary at `1e-6`.

The Python production at `version_3/models/fsa_v5/` continues to
exist as the differentially-tested fast path.

See `claude_plans/Julia_port_FSA_v5_model_plan_2026-05-06_1921.md`
for the port plan and audit checklist.
"""
module FSAv5

include("FSAv5/Types.jl")
include("FSAv5/DefaultParams.jl")
include("FSAv5/Dynamics.jl")
include("FSAv5/Cost.jl")
include("FSAv5/Schedule.jl")
include("FSAv5/PhiBurst.jl")
include("FSAv5/Plant.jl")
include("FSAv5/Obs.jl")
include("FSAv5/Estimation.jl")
# include("FSAv5/Estimation.jl")

end # module FSAv5
