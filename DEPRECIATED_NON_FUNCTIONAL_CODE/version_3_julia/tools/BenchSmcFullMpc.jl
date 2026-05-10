# tools/BenchSmcFullMpc.jl — Stage 3 (FSA-v5): closed-loop SMC²-MPC.
# Filter + controller composed. Posterior over (params, state) feeds
# the controller; controller's planned schedule advances the plant;
# next window's filter sees the data.
#
# Port of `version_3/tools/bench_smc_full_mpc_fsa_v5.py`. The two pillars
# share the same outer tempered-SMC kernel (`SMC2FC.TemperedSMC`) — one
# instance for the filter side (target = posterior over params + state),
# one for the controller side (target = exp(-β·J(u))).
#
# Status: SKELETON. The plant + replan dispatch is wired; the filter
# loop and controller loop both punt to SMC2FC.jl entry points, the same
# way the two single-pillar skeletons do. End-to-end integration is the
# final task in the Julia port plan (charter Part II §15.7).

module BenchSmcFullMpc

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
                SIGMA_A_FROZEN, SIGMA_K_FROZEN,
                HIGH_RES_FSA_V5_ESTIMATION

using ..RunDir: allocate_run_dir

# ── Window structure (matches Stage 1 + Stage 2)
const DT             = 1.0 / 96.0
const WINDOW_BINS    = 96
const STRIDE_BINS    = 48
const DEFAULT_T_DAYS = 14
const DEFAULT_REPLAN_K = 2
const DEFAULT_BETA   = 50.0

# ── CLI parsing (mirrors Stage 2 plus filter knobs)
function parse_args(argv::Vector{<:AbstractString})
    cost     = "soft"
    scenario = "healthy"
    T_days   = DEFAULT_T_DAYS
    replan_K = DEFAULT_REPLAN_K
    beta     = DEFAULT_BETA
    run_tag  = ""
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
    isempty(run_tag) && (run_tag = "stage3_full_mpc_$(cost)_$(scenario)_T$(T_days)")
    return (; cost, scenario, T_days, replan_K, beta, run_tag)
end

# ── Main entry ──────────────────────────────────────────────────────────
function main(argv::Vector{<:AbstractString} = ARGS)
    args = parse_args(argv)
    repo_root = dirname(dirname(@__FILE__))
    out_dir, run_num = allocate_run_dir(repo_root, args.run_tag)

    println("=" ^ 76)
    @printf "  Stage 3 (FSA-v5, Julia) — closed-loop SMC²-MPC\n"
    @printf "  cost=%s, scenario=%s, T=%dd, replan_K=%d, β=%.1f\n" args.cost args.scenario args.T_days args.replan_K args.beta
    @printf "  run dir: %s\n" basename(out_dir)
    println("=" ^ 76)

    sigma_diag = [SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                   SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN]
    plant = StepwisePlant(state=DEFAULT_INIT, params=TRUTH_PARAMS_V5,
                           sigma_diag=sigma_diag)
    n_strides = (args.T_days * BINS_PER_DAY) ÷ STRIDE_BINS
    n_replans = n_strides ÷ args.replan_K

    @printf "  n_strides=%d, n_replans=%d (every %dh)\n" n_strides n_replans 6*args.replan_K

    em = HIGH_RES_FSA_V5_ESTIMATION
    @printf "  estimation model: %d params (37 estimable, 14 frozen by charter)\n" length(em.all_names)

    # ── Closed-loop replan loop ────────────────────────────────────────
    # Per stride:
    #   1. plant.advance(stride_bins, current_phi_daily) → obs
    #   2. obs goes into SMC²-filter window → posterior over (params, state)
    #   3. every K strides: controller plans new schedule from posterior
    #      via SMC2FC.run_tempered_smc_loop
    #   4. apply new schedule for next K strides
    @info "Stage-3 closed loop wires SMC2FC.run_smc_window (filter) + " *
          "SMC2FC.run_tempered_smc_loop (controller). Skeleton: GPU loop " *
          "not invoked in this scaffold."

    # ── Save artifacts ─────────────────────────────────────────────────
    state_names = ["B", "S", "F", "A", "K_FB", "K_FS"]
    npzwrite(joinpath(out_dir, "trajectory.npz"),
             Dict("trajectory"  => zeros(Float64, 0, 6),
                  "state_names" => state_names))

    manifest = Dict(
        "schema_version" => 1,
        "stage"          => 3,
        "bench"          => "bench_smc_full_mpc_fsa_v5",
        "language"       => "julia",
        "run_tag"        => args.run_tag,
        "run_number"     => run_num,
        "cost_variant"   => args.cost,
        "T_total_days"   => args.T_days,
        "step_minutes"   => 15,
        "BINS_PER_DAY"   => BINS_PER_DAY,
        "WINDOW_BINS"    => WINDOW_BINS,
        "STRIDE_BINS"    => STRIDE_BINS,
        "n_strides"      => n_strides,
        "n_replans"      => n_replans,
        "ctrl_cfg"       => Dict(
            "beta"     => args.beta,
            "replan_K" => args.replan_K,
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

end # module BenchSmcFullMpc
