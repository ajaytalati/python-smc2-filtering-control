#!/usr/bin/env julia
# Closed-loop SMC²-MPC bench for FSA-v2 — Julia port of
# `version_2/tools/bench_smc_full_mpc_fsa.py`.
#
# Pinned config (T=14d run, matching `outputs/.../T14d_replanK2_h60min_no_infoaware/manifest.json`):
#   T_total_days = 14,  step_minutes = 60,  BINS_PER_DAY = 24
#   WINDOW_BINS  = 24,  STRIDE_BINS = 12,   n_strides = 27,  replan_K = 2
#   N_SMC = 1024, N_PF = 800, target_ess_frac = 0.5
#   Bridge: schrodinger_follmer (annealed, q0_cov=true, blend=0.7,
#           n_stages=3, n_mh_steps=5, info_aware=false)
#   Filter HMC: step_size=0.025, num_leapfrog=8, num_mcmc_steps=5
#   Control HMC: step_size=0.2, num_leapfrog=16, num_mcmc_steps=10
#   Seeds: plant=42, filter_base=42, ctrl_base=42
#
# CLI:
#   julia --project=. tools/bench_smc_full_mpc_fsa.jl \
#       --T-days 14 --step-minutes 60 --replan-K 2 \
#       --N-smc 1024 --N-pf 800 --seed 42 [--smoke]
#
# `--step-minutes` MUST be set BEFORE model imports — `FSA_STEP_MINUTES`
# env var is read at module-load time to set BINS_PER_DAY (mirror Python).


# ── Minimal CLI parser (no ArgParse dep) ─────────────────────────────────

function _parse_args(argv::Vector{String})
    defaults = Dict{String,Any}(
        "T-days"        => 14,
        "step-minutes"  => 60,
        "replan-K"      => 2,
        "N-smc"         => 1024,
        "N-pf"          => 800,
        "seed"          => 42,
        "output-dir"    => "",
        "smoke"         => false,
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--smoke"
            defaults["smoke"] = true; i += 1
        elseif startswith(a, "--")
            key = a[3:end]
            if !haskey(defaults, key)
                error("Unknown flag: $a (known: $(sort(collect(keys(defaults)))))")
            end
            i += 1
            i <= length(argv) || error("Missing value for $a")
            v = argv[i]
            if defaults[key] isa Int
                defaults[key] = parse(Int, v)
            else
                defaults[key] = v
            end
            i += 1
        else
            error("Unrecognized arg: $a")
        end
    end
    return defaults
end


# ── Parse args FIRST, THEN set FSA_STEP_MINUTES, THEN load model ─────────

const ARGS_DICT = _parse_args(copy(ARGS))
ENV["FSA_STEP_MINUTES"] = string(ARGS_DICT["step-minutes"])

@info "loading model + framework (FSA_STEP_MINUTES=$(ENV["FSA_STEP_MINUTES"]))..."

using Dates

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: StepwisePlant, advance!, finalise
using .FSAHighRes.Estimation: build_estimation_model
using SMC2FC


function main(args::Dict{String,Any})
    BINS_PER_DAY = (60 * 24) ÷ args["step-minutes"]
    WINDOW_BINS  = BINS_PER_DAY                   # 1-day window
    STRIDE_BINS  = WINDOW_BINS ÷ 2                # 12-h stride
    DT_DAYS      = 1.0 / BINS_PER_DAY
    n_total_bins = args["T-days"] * BINS_PER_DAY
    n_strides    = (n_total_bins - WINDOW_BINS) ÷ STRIDE_BINS + 1
    # NOTE: n_replans here is a STATIC upper bound used for log layout. The
    # Python manifest reports the DYNAMIC replan count from the loop body
    # (len(replan_history)) which can be lower if the last window is too
    # short to merit a fresh plan. For T=14d K=2: static=14, dynamic=13.
    n_replans    = (n_strides + args["replan-K"] - 1) ÷ args["replan-K"]

    @info "config: T=$(args["T-days"])d, step=$(args["step-minutes"])min, " *
          "BINS_PER_DAY=$BINS_PER_DAY, WINDOW=$WINDOW_BINS, STRIDE=$STRIDE_BINS, " *
          "n_strides=$n_strides, n_replans=$n_replans"

    out_dir = isempty(args["output-dir"]) ?
        joinpath(REPO_ROOT, "outputs", "fsa_high_res", "g4_runs",
                  "T$(args["T-days"])d_replanK$(args["replan-K"])_h$(args["step-minutes"])min_no_infoaware") :
        args["output-dir"]
    mkpath(out_dir)

    @info "building plant..."
    plant = StepwisePlant(seed_offset = args["seed"], dt = DT_DAYS)

    @info "building estimation model..."
    em = build_estimation_model()
    @info "  n_states=$(em.n_states), n_params=$(length(em.param_priors))"

    if args["smoke"]
        @info "smoke mode: advancing plant by 1 stride..."
        out = advance!(plant, STRIDE_BINS, [1.0])
        @info "  state after 1 stride: B=$(round(plant.state[1], digits=4)), " *
              "F=$(round(plant.state[2], digits=4)), A=$(round(plant.state[3], digits=4))"
        @info "smoke mode complete — long SMC² loop skipped."

        open(joinpath(out_dir, "experiment_run.md"), "w") do io
            println(io, "# FSA-v2 Julia bench — smoke run")
            println(io)
            println(io, "- **Mode:** smoke (imports + 1 plant.advance, NO SMC² loop, NO controller)")
            println(io, "- **Timestamp:** $(Dates.now())")
            println(io, "- **T_total_days:** $(args["T-days"])")
            println(io, "- **step_minutes:** $(args["step-minutes"])")
            println(io, "- **BINS_PER_DAY:** $BINS_PER_DAY")
            println(io, "- **WINDOW_BINS:** $WINDOW_BINS, **STRIDE_BINS:** $STRIDE_BINS")
            println(io, "- **n_strides:** $n_strides, **n_replans:** $n_replans")
            println(io, "- **N_SMC:** $(args["N-smc"]), **N_PF:** $(args["N-pf"])")
            println(io, "- **Plant state after 1 stride:** B=$(plant.state[1]), F=$(plant.state[2]), A=$(plant.state[3])")
            println(io)
            println(io, "## Status")
            println(io)
            println(io, "Mechanical port complete. Full SMC²-MPC loop body is scaffolded but not yet wired — see TODO in tools/bench_smc_full_mpc_fsa.jl.")
        end
        return
    end

    # ── Full SMC² loop (scaffold) ─────────────────────────────────────────
    # The full closed-loop body is a port of `bench_smc_full_mpc_fsa.py`.
    # Pseudocode:
    #
    #   for stride_idx in 1:n_strides
    #       if (stride_idx == 1) || ((stride_idx - 1) % replan_K == 0)
    #           Phi_daily = controller_plan(em, posterior, horizon=H_remaining)
    #       else
    #           Phi_daily = last_planned_Phi
    #       end
    #
    #       obs = advance!(plant, STRIDE_BINS, Phi_daily)
    #       grid_obs = em.align_obs_fn(obs, WINDOW_BINS, DT_DAYS)
    #       result = SMC2FC.run_smc_window_bridge(em, grid_obs, T_arr=window_T_grid,
    #                                               cfg=smc_cfg, prior_particles=prev_particles, ...)
    #       prev_particles = result.particles
    #       record_param_means(stride_idx, prev_particles)
    #   end
    #
    # The framework calls live in:
    #   - julia/SMC2FC/src/SMC2/TemperedSMC.jl :: run_smc_window_bridge
    #   - julia/SMC2FC/src/Control/TemperedSMC.jl :: run_tempered_smc_loop
    #
    # These are CPU-only (AdvancedHMC.jl) and slow at the manifest config.
    # Stage B replaces the inner HMC kernel with a model-specific GPU
    # parallel-chains ChEES path (`gpu_pf.jl`).
    #
    # Per the autonomous-mode brief: no long CPU/GPU runs. Use `--smoke`.

    error("Full SMC² loop not yet wired. Use `--smoke` to verify imports + plant.advance.")
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS_DICT)
end
