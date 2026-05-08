"""
    test_gate5_multithread.jl

Gate 5 of the SMC2FC_functional replacement-readiness audit:
verify the unit-test suite stays green under
`JULIA_NUM_THREADS=4` and `=8`. The library uses
`Threads.@threads` internally in `SMC2/TemperedSMC.jl` and seeds per-
thread sub-RNGs *before* the parallel loop because `MersenneTwister` is
not thread-safe; this test confirms that discipline holds and the
public API does not regress under multi-threaded execution.

This file does not run the suite itself — that would invoke the
compiler twice and balloon wall time. Instead it should be invoked from
the shell with the thread count set, e.g.

```
cd julia/SMC2FC_functional
JULIA_NUM_THREADS=4 julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=8 julia --project=. -e 'using Pkg; Pkg.test()'
```

Or, more directly, run the included shell script `run_gate5.sh` which
does both runs back-to-back and prints a one-line summary of each.
"""

using Test

@testset "Gate 5 — sanity check on Threads.nthreads()" begin
    n = Threads.nthreads()
    @info "Gate 5: running under JULIA_NUM_THREADS=$n"
    @test n >= 1
    # If user invoked us with the env var set we should observe it.
    if haskey(ENV, "JULIA_NUM_THREADS")
        requested = parse(Int, ENV["JULIA_NUM_THREADS"])
        @test n == requested
    end
end
