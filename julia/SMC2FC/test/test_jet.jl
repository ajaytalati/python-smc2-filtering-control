# JET static analysis pass — charter §18 audit gate.
# Runs `report_package(SMC2FC)` and fails the test suite if any type
# uncertainties or instabilities are flagged in the public API.

using JET

@testset "JET — static type analysis" begin
    # Restrict reports to errors found INSIDE the SMC2FC module tree.
    # Without `target_modules`, JET follows calls into upstream packages
    # (CUDA, GPUCompiler, LLVM, AdvancedHMC, ...) and surfaces internal
    # type instabilities in those deps — which are not actionable from
    # SMC2FC and are not what charter §18 audits.
    target_mods = (SMC2FC,
                    SMC2FC.Kernels,
                    SMC2FC.OT,
                    SMC2FC.Bootstrap)

    rep = JET.report_package(SMC2FC;
                              target_modules = target_mods,
                              toplevel_logger = nothing)
    reports = JET.get_reports(rep)

    if !isempty(reports)
        @info "JET reports inside SMC2FC ($(length(reports))):"
        for r in reports
            show(stdout, MIME("text/plain"), r)
            println()
        end
    end

    # Charter §18: any Union{...} return in a hot inner loop is a type
    # instability. We fail the gate on any non-trivial report.
    @test isempty(reports)
end
