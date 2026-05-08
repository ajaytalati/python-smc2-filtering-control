"""
    test_gate6_jet.jl

Gate 6 of the SMC2FC_functional replacement-readiness audit:
JET.jl static analysis on the public API.

Mirrors the original port's `test_jet.jl`. Restricts reports to
modules inside the `SMC2FC_functional` tree so that internal type
instabilities in upstream deps (CUDA, AdvancedHMC, GPUCompiler, …)
are not surfaced — those are not actionable from this library.

# Pass criteria
`JET.report_package(SMC2FC_functional)` returns no reports when the
`target_modules` filter is applied to our own modules.

# Run
```
cd julia/SMC2FC_functional
julia --project=. test/test_gate6_jet.jl
```
"""

using Test
using SMC2FC_functional
using JET

@testset "Gate 6 — JET static type analysis" begin
    target_mods = (
        SMC2FC_functional,
        SMC2FC_functional.Kernels,
        SMC2FC_functional.OT,
        SMC2FC_functional.Bootstrap,
    )

    rep = JET.report_package(SMC2FC_functional;
                              target_modules = target_mods,
                              toplevel_logger = nothing)
    reports = JET.get_reports(rep)

    @info "Gate 6: JET reports = $(length(reports)) (after target_modules filter)"

    if !isempty(reports)
        @info "  reports:"
        for r in reports
            show(stdout, MIME("text/plain"), r)
            println()
        end
    end

    @test isempty(reports)
end
