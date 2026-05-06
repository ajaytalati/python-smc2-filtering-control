# tools/BenchControllerOnly.jl — Stage 2 (FSA-v5): controller-only.
# Filter NOT run.
#
# Port of `version_3/tools/bench_controller_only_fsa_v5.py`. Goal: isolate
# controller behaviour from filter quality. The plant runs under the
# controller's chosen schedule; truth params are KNOWN; current state is
# the actual `plant.state` (no posterior inference). Bugs in the
# controller side — RBF schedule, cost composition, integrator, tempering
# schedule — surface here without paying the filter's per-stride cost.
#
# Status: SKELETON. The plant + Phi-burst pipeline is wired. The
# controller loop goes through `SMC2FC.run_tempered_smc_loop` (Phase 4
# of `julia/SMC2FC/`); this scaffold leaves the inner loop as a TODO so
# the GPU controller integration (charter §15.6) can be exercised
# end-to-end in a separate pass.
#
# Cost variants (`--cost`):
#   * `soft`         → SoftSurrogate marker (HMC-friendly, default β=50.0)
#   * `hard`         → HardIndicator marker (no HMC, weighting-only)
#   * `gradient_ot`  → back-compat fallback; not shipped as production

module BenchControllerOnly

using JSON3
using NPZ
using Random
using Printf
using Statistics
using Plots

using ..FSAv5: FSAv5State, BimodalPhi, StepwisePlant, advance!,
                BINS_PER_DAY, DT_BIN_DAYS,
                TRUTH_PARAMS_V5, DEFAULT_INIT,
                SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                SIGMA_A_FROZEN, SIGMA_K_FROZEN

using ..RunDir: allocate_run_dir

# ── Window structure ─────────────────────────────────────────────────────
const DT             = 1.0 / 96.0    # 15 min in days
const STRIDE_BINS    = 48            # 12 hours
const DEFAULT_T_DAYS = 14
const DEFAULT_REPLAN_K = 2           # replan every 2 strides = 24h
const DEFAULT_BETA   = 50.0          # soft variant temperature
const DEFAULT_ALPHA  = 0.05
const DEFAULT_A_TARGET = 2.0

# ── CLI parsing ──────────────────────────────────────────────────────────
"""
    parse_args(argv) -> NamedTuple

Tiny CLI parser. Supports `--cost`, `--scenario`, `--T-days`, `--replan-K`,
`--beta`, `--run-tag`. Defaults match the Python bench.
"""
function parse_args(argv::Vector{<:AbstractString})
    cost      = "soft"
    scenario  = "healthy"
    T_days    = DEFAULT_T_DAYS
    replan_K  = DEFAULT_REPLAN_K
    beta      = DEFAULT_BETA
    run_tag   = ""
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--cost"
            cost = argv[i+1]; i += 2
        elseif a == "--scenario"
            scenario = argv[i+1]; i += 2
        elseif a == "--T-days"
            T_days = parse(Int, argv[i+1]); i += 2
        elseif a == "--replan-K"
            replan_K = parse(Int, argv[i+1]); i += 2
        elseif a == "--beta"
            beta = parse(Float64, argv[i+1]); i += 2
        elseif a == "--run-tag"
            run_tag = argv[i+1]; i += 2
        else
            i += 1
        end
    end
    if isempty(run_tag)
        run_tag = "stage2_controller_only_$(cost)_$(scenario)_T$(T_days)"
    end
    return (; cost, scenario, T_days, replan_K, beta, run_tag)
end

# ── Main entry ──────────────────────────────────────────────────────────
function main(argv::Vector{<:AbstractString} = ARGS)
    args = parse_args(argv)
    repo_root = dirname(dirname(@__FILE__))
    out_dir, run_num = allocate_run_dir(repo_root, args.run_tag)

    println("=" ^ 76)
    @printf "  Stage 2 (FSA-v5, Julia) — controller-only\n"
    @printf "  cost=%s, scenario=%s, T=%dd, replan_K=%d, β=%.1f\n" args.cost args.scenario args.T_days args.replan_K args.beta
    @printf "  run dir: %s\n" basename(out_dir)
    println("=" ^ 76)

    # Plant + scenario init
    sigma_diag = [SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                   SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN]
    plant = StepwisePlant(state=DEFAULT_INIT, params=TRUTH_PARAMS_V5,
                           sigma_diag=sigma_diag)
    n_strides = (args.T_days * BINS_PER_DAY) ÷ STRIDE_BINS
    n_replans = n_strides ÷ args.replan_K

    @printf "  n_strides=%d, n_replans=%d (every %dh)\n" n_strides n_replans 6*args.replan_K

    # ── Replan loop ───────────────────────────────────────────────────
    # The inner controller call goes through:
    #   spec = ControlSpec(... cost_fn = build cost from FSAv5.Cost,
    #                          schedule_from_theta = RBFSchedule.schedule_from_theta)
    #   result = SMC2FC.run_tempered_smc_loop(spec, ctrl_cfg, key)
    # then `result.mean_schedule` is decoded to per-day BimodalPhi and
    # applied to `plant` for `replan_K` strides before the next replan.
    @info "Stage-2 replan loop goes through SMC2FC.run_tempered_smc_loop. " *
          "Skeleton: GPU controller not invoked in this scaffold."

    # ── Save artifacts ─────────────────────────────────────────────────
    state_names = ["B", "S", "F", "A", "K_FB", "K_FS"]
    # Skeleton: until the replan loop runs the controller, write an empty
    # placeholder trajectory so downstream summarise/diagnose tools have
    # something to consume.
    npzwrite(joinpath(out_dir, "trajectory.npz"),
             Dict("trajectory"  => zeros(Float64, 0, 6),
                  "state_names" => state_names))

    manifest = Dict(
        "schema_version" => 1,
        "stage"          => 2,
        "bench"          => "bench_controller_only_fsa_v5",
        "language"       => "julia",
        "run_tag"        => args.run_tag,
        "run_number"     => run_num,
        "cost_variant"   => args.cost,
        "T_total_days"   => args.T_days,
        "step_minutes"   => 15,
        "BINS_PER_DAY"   => BINS_PER_DAY,
        "STRIDE_BINS"    => STRIDE_BINS,
        "n_strides"      => n_strides,
        "n_replans"      => n_replans,
        "ctrl_cfg"       => Dict(
            "beta"             => args.beta,
            "alpha"            => DEFAULT_ALPHA,
            "A_target"         => DEFAULT_A_TARGET,
            "replan_K"         => args.replan_K,
        ),
        "scenario"       => Dict("name" => args.scenario),
    )
    open(joinpath(out_dir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    println()
    @printf "  Artifacts written under %s/\n" basename(out_dir)
    println("=" ^ 76)
    return out_dir
end

export main, parse_args

end # module BenchControllerOnly
