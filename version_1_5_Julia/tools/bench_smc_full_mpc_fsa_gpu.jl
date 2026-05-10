#!/usr/bin/env julia
# FSA v1.5 closed-loop SMC²-MPC bench — purely functional foldl over strides.
#
# Per stride:
#   1. Slice next stride's Φ from the current plan.
#   2. plant_rollout(state, Φ_slice, params, dt, key) → trajectory + obs.
#   3. Build window grid_obs from the accumulated obs history.
#   4. Run outer SMC² on the filter (tempered + parallel HMC).
#   5. Maybe replan: framework's run_tempered_smc_gpu on the controller's RBF θ.
#
# All ingredients are pure. The bench accumulator IS the history; mutation
# only happens (a) inside KernelAbstractions kernels (private to gpu_pf.jl)
# and (b) at the single I/O write at the end.
#
# Modes:
#   --open-loop false   default — closed-loop with replanning every K strides.
#   --open-loop true    one up-front plan from INIT_STATE+TRUTH_PARAMS, no replan.

using Dates


# CLI parsing: defaults dict + filter aliases + argv parser.
# Refactored 2026-05-09 — see Phase 1 of the bench refactor plan
# (claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md).
# Pure CLI logic; no dependencies on framework or model.
include(joinpath(@__DIR__, "bench", "bench_args.jl"))




const ARGS_DICT = _parse_args(copy(ARGS))
ENV["FSA_STEP_MINUTES"] = string(ARGS_DICT["step-minutes"])

@info "loading model + framework (FSA_STEP_MINUTES=$(ENV["FSA_STEP_MINUTES"]))..."

using Random, Statistics, Printf, LinearAlgebra
using LogExpFunctions: logsumexp
using JLD2
using JSON3
using CUDA
using StaticArrays
using StableRNGs
using Plots                # for the auto-generated param-traces plot

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: PlantState, plant_rollout, init_plant_state
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE,
                              PINNED_PARAMS, params_v15_to_v1_nt, fill_pinned_nt
using .FSAHighRes.Estimation: PARAM_NAMES, PARAM_PRIOR_CONFIG
using .FSAHighRes.GPUPF: FSAGPUTarget, gpu_log_density, gpu_grads,
                          parallel_hmc_one_move, parallel_hmc_one_move!
using .FSAHighRes.GPUControl: FSAv1ControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn
# Framework migration (2026-05-08): from SMC2FC (renamed to deprecated
# location) to the now-default SMC2FC_functional. Symbol is byte-
# identical between the two; see commit 4a5c007 + writeup §6.9.
#
# 2026-05-09: also import the framework's generic parallel-chains HMC
# (parallel_hmc_one_move_generic!) and ChEES picker (chees_pick_L_generic)
# so the filter side can use the same HMC algorithm as the controller.
using SMC2FC_functional: run_tempered_smc_gpu,
                          fit_gaussian, sample_from_gaussian,
                          silverman_bandwidth, log_kernel_matrix,
                          parallel_hmc_one_move_generic!,
                          chees_pick_L_generic


# FSA-v1.5 model-specific glue (posterior_mean_v15, window_grid_obs).
# Refactored 2026-05-09 — see Phase 1b of the bench refactor plan.
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "bench_glue.jl"))


# Bench filter SMC² (run_outer_smc + extract_xhat). Tempered λ-ladder with
# the framework's ChEES HMC at every level. Refactored 2026-05-09 — see
# Phase 1c of the bench refactor plan.
include(joinpath(@__DIR__, "bench", "bench_filter.jl"))


# Bench controller-side wrapper (controller_plan). Builds an
# FSAv1ControlGPUTarget and calls the framework's run_tempered_smc_gpu;
# decodes posterior-mean RBF θ → per-bin Φ schedule. Refactored
# 2026-05-09 — see Phase 1d of the bench refactor plan.
include(joinpath(@__DIR__, "bench", "bench_controller.jl"))


# Bench post-processing (save_bench_outputs + _plot_param_traces_v15).
# Writes data.jld2, per_stride.csv, controller_diagnostics.csv,
# manifest.json, experiment_run.md, state-traces PNG, param-traces PNG.
# Refactored 2026-05-09 — see Phase 1e of the bench refactor plan.
include(joinpath(@__DIR__, "bench", "bench_postproc.jl"))


# Closed-loop foldl over strides (run_one_stride + run_closed_loop_bench).
# Replaces the previously-inline stride_step closure with explicit
# helpers + a NamedTuple `ctx` that carries the captured outer vars.
# Refactored 2026-05-09 — see Phase 2 of the bench refactor plan.
include(joinpath(@__DIR__, "bench", "bench_loop.jl"))


# ── Main bench ───────────────────────────────────────────────────────────

function main(args::Dict{String,Any})
    bins_per_day = (60 * 24) ÷ args["step-minutes"]
    window_bins  = bins_per_day
    stride_bins  = window_bins ÷ 2
    dt_days      = 1.0 / bins_per_day
    T_total_bins = args["T-days"] * bins_per_day
    n_strides    = (T_total_bins - window_bins) ÷ stride_bins + 1
    H_plan_bins  = T_total_bins                       # FIXED, per §2.11

    is_open_loop = lowercase(args["open-loop"]) in ("true", "1", "yes")

    @info "config: T=$(args["T-days"])d step=$(args["step-minutes"])min " *
          "BINS_PER_DAY=$bins_per_day WINDOW=$window_bins STRIDE=$stride_bins " *
          "n_strides=$n_strides  open_loop=$is_open_loop"

    out_dir = isempty(args["output-dir"]) ?
              joinpath(REPO_ROOT, "outputs", "fsa_high_res", "g4_runs",
                        "T$(args["T-days"])d_$(is_open_loop ? "OL" : "CL_K$(args["replan-K"])")_h$(args["step-minutes"])min") :
              args["output-dir"]
    mkpath(out_dir)

    @info "GPU device: $(CUDA.name(CUDA.device()))"

    # ── Build filter target ──
    n_smc = args["N-smc"]
    K_per_chain = args["K-per-chain"]
    d = length(PARAM_NAMES)
    M_max = n_smc * (1 + 2 * d)
    @info "filter target: N_SMC=$n_smc K_per_chain=$K_per_chain d=$d M_max=$M_max"
    # NB: ot-max-weight defaults to 0.0 (OT rescue DISABLED — fast path).
    # Setting it to 0.01 enables the framework's `gpu_ot_blend_chain!`
    # rescue, which carries a ~12× wall-time penalty at this config
    # (per-chain Julia loop with PCIe round-trips). See the heavily-
    # commented default in `_parse_args` above for the full story.
    target = FSAGPUTarget(
        K_per_chain = K_per_chain, M_max = M_max,
        T_steps = window_bins, R = 4, dt = dt_days,
        noise_seed = 0,
        ot_max_weight = args["ot-max-weight"],
    )
    if args["ot-max-weight"] >= 1e-6
        @warn "OT rescue ENABLED (ot-max-weight = $(args["ot-max-weight"])). " *
              "Expect ~12× slowdown on Julia vs the default OT-off path. " *
              "See the comment block on `ot-max-weight` in this script."
    else
        @info "filter target: OT rescue DISABLED (ot-max-weight = 0.0, fast path)"
    end

    # ── Filter prior (unconstrained, v1.5 PARAM_PRIOR_CONFIG) ──
    prior_means  = Float64[m for (_, _, m, _) in PARAM_PRIOR_CONFIG]
    prior_sigmas = Float64[s for (_, _, _, s) in PARAM_PRIOR_CONFIG]
    @assert length(prior_means)  == d
    @assert length(prior_sigmas) == d

    # Filter-side ChEES-HMC candidate list: powers of 2 from
    # `--filter-chees-min` up to `--filter-chees-max` inclusive. Mirrors
    # the controller's chees_list pattern below.
    filter_chees_min = args["filter-chees-min"]
    filter_chees_max = args["filter-chees-max"]
    filter_chees_list = [filter_chees_min]
    while filter_chees_list[end] * 2 <= filter_chees_max
        push!(filter_chees_list, filter_chees_list[end] * 2)
    end

    filter_cfg = (
        max_lambda_inc    = args["max-lambda-inc"],
        target_ess_frac   = args["target-ess-frac"],
        num_mcmc          = args["num-mcmc"],
        hmc_step          = args["hmc-step-size"],
        hmc_leap          = args["hmc-leapfrog"],
        max_levels        = args["max-temp-levels"],
        h_fd              = args["h-fd"],
        liu_west_a        = args["liu-west-a"],
        smooth_resample_bw = args["smooth-resample-bw"],
        gaussian_bridge   = lowercase(string(args["gaussian-bridge"])) == "true",
        chees_L_candidates = filter_chees_list,
    )

    # ── Controller config — all knobs CLI-exposed ──
    ctrl_n_anchors = args["ctrl-n-anchors"]
    ctrl_n_smc     = args["ctrl-n-smc"]
    ctrl_n_inner   = args["ctrl-n-inner"]
    ctrl_M_max     = ctrl_n_smc * (1 + 2 * ctrl_n_anchors)
    # ChEES-HMC candidate list: powers of 2 from 16 up to ctrl-chees-max.
    chees_max  = args["ctrl-chees-max"]
    chees_list = [16]
    while chees_list[end] * 2 <= chees_max
        push!(chees_list, chees_list[end] * 2)
    end
    ctrl_cfg = (
        n_smc       = ctrl_n_smc,
        n_inner     = ctrl_n_inner,
        n_anchors   = ctrl_n_anchors,
        M_max       = ctrl_M_max,
        sigma_prior = args["ctrl-sigma-prior"],
        target_nats = args["ctrl-target-nats"],
        target_ess_frac = args["ctrl-target-ess-frac"],
        max_lambda_inc  = args["ctrl-max-lambda-inc"],
        max_levels      = args["ctrl-max-levels"],
        num_mcmc        = args["ctrl-num-mcmc"],
        hmc_step        = args["ctrl-hmc-step"],
        hmc_leap        = args["ctrl-hmc-leap"],
        chees_L_candidates = chees_list,
    )
    @info "controller cfg: n_smc=$ctrl_n_smc n_inner=$ctrl_n_inner num_mcmc=$(ctrl_cfg.num_mcmc) hmc_leap=$(ctrl_cfg.hmc_leap) chees=$chees_list max_levels=$(ctrl_cfg.max_levels)"

    # ── Pre-roll the BASELINE plant (Φ=1) for the full bench ──
    # The baseline plant produces obs-of-truth that we'll compare against
    # the MPC-plant. It also drives the FILTER (we filter on the
    # baseline obs to recover params, while the MPC plant uses the
    # controller's plan.). For v1.5 we filter on the MPC plant's obs
    # which is what the closed-loop bench actually does.

    base_seed = UInt64(args["seed"])
    base_plant_state = init_plant_state()
    base_phi  = fill(Float32(1.0), T_total_bins)
    base_out  = plant_rollout(base_plant_state, base_phi, DEFAULT_PARAMS, dt_days,
                                 hash((base_seed, :base_plant)))

    # ── Open-loop initial plan ──
    initial_plan = if is_open_loop
        @info "OPEN-LOOP: building one initial plan from INIT_STATE+TRUTH..."
        params_truth_v1 = params_v15_to_v1_nt(fill_pinned_nt((
            tau_F   = DEFAULT_PARAMS[:tau_F],
            B_inf   = DEFAULT_PARAMS[:B_inf],
            F_inf   = DEFAULT_PARAMS[:F_inf],
            lambda_A = DEFAULT_PARAMS[:lambda_A],
            mu_0    = DEFAULT_PARAMS[:mu_0],
            mu_B    = DEFAULT_PARAMS[:mu_B],
            mu_F    = DEFAULT_PARAMS[:mu_F],
            sigma_B = DEFAULT_PARAMS[:sigma_B],
            sigma_F = DEFAULT_PARAMS[:sigma_F],
            sigma_A = DEFAULT_PARAMS[:sigma_A],
        )))
        s0 = SVector{3,Float64}(Float64(INIT_STATE.B),
                                  Float64(INIT_STATE.F),
                                  Float64(INIT_STATE.A))
        out = controller_plan(params_truth_v1, s0, H_plan_bins, max(1, bins_per_day ÷ 24),
                               dt_days, ctrl_cfg, hash((base_seed, :ctrl_init)))
        @info @sprintf("  open-loop plan ready: Φ̄=%.3f  Φ_max=%.3f  Φ_min=%.3f  n_temp=%d",
                       mean(out.Phi_plan), maximum(out.Phi_plan), minimum(out.Phi_plan),
                       out.n_temp_ctrl)
        out.Phi_plan
    else
        fill(Float32(1.0), T_total_bins)
    end

    # ── Controller-HMC diagnostics flag ──
    # When ON (the 2026-05-09 study default), each replan's
    # `run_tempered_smc_gpu` returns a per-tempering-level diagnostics
    # vector; we flatten it into a single CSV row per (stride, level).
    collect_ctrl_diag = lowercase(string(args["collect-ctrl-diagnostics"])) == "true"

    # ── Closed-loop foldl over strides ──
    # `all_filter_posts` is a per-stride list of unconstrained posterior
    # particle clouds (one (n_smc, n_params) Matrix per stride; Nothing
    # for warmup strides). Keeps the param-traces plot's input cheap to
    # build at end-of-bench.
    init_acc = (
        plant_state    = init_plant_state(),
        plan_phi       = initial_plan,
        plan_offset    = 0,
        filter_post    = nothing,
        all_filter_posts = Vector{Union{Nothing,Matrix{Float64}}}(),
        traj_history   = Matrix{Float64}(undef, 0, 3),
        obs_history    = (B = Float32[], F = Float32[], A = Float32[],
                          Phi = Float32[]),
        per_stride_log = NamedTuple[],
        # Flat list of per-tempering-level controller diagnostics rows.
        # Each row tagged with the firing stride_idx (bench-side) so the
        # downstream plotter can group by replan.
        controller_diagnostics = NamedTuple[],
        last_xhat      = SVector{3, Float64}(INIT_STATE.B, INIT_STATE.F, INIT_STATE.A),
    )

    # Captured-vars NamedTuple for run_one_stride / run_closed_loop_bench.
    # Replaces the previous closure-capture pattern; every previously-
    # captured outer-scope variable is an explicit field here. Phase 3
    # of the refactor will swap `target` for a model-agnostic
    # `log_density_fn::Function` callback.
    ctx = (
        target            = target,
        n_smc             = n_smc,
        filter_cfg        = filter_cfg,
        ctrl_cfg          = ctrl_cfg,
        bins_per_day      = bins_per_day,
        stride_bins       = stride_bins,
        window_bins       = window_bins,
        dt_days           = dt_days,
        H_plan_bins       = H_plan_bins,
        n_strides         = n_strides,
        is_open_loop      = is_open_loop,
        base_seed         = base_seed,
        replan_K          = args["replan-K"],
        collect_ctrl_diag = collect_ctrl_diag,
        prior_means       = prior_means,
        prior_sigmas      = prior_sigmas,
    )

    t_total_start = time()
    final = run_closed_loop_bench(ctx, init_acc)
    t_total = time() - t_total_start

    @info @sprintf("Total wall time: %.1f s", t_total)

    # ── Post-processing: write all artefacts + plots ──
    # Refactored 2026-05-09 — see Phase 1e of the bench refactor plan.
    # save_bench_outputs is defined in tools/bench/bench_postproc.jl;
    # its `cfg` NamedTuple bundles the bench context vars it needs.
    save_bench_outputs(out_dir, final, t_total, args, (
        bins_per_day      = bins_per_day,
        stride_bins       = stride_bins,
        window_bins       = window_bins,
        dt_days           = dt_days,
        n_strides         = n_strides,
        is_open_loop      = is_open_loop,
        collect_ctrl_diag = collect_ctrl_diag,
        n_smc             = n_smc,
        K_per_chain       = K_per_chain,
        filter_cfg        = filter_cfg,
        ctrl_cfg          = ctrl_cfg,
        base_out          = base_out,
        base_phi          = base_phi,
    ))

    return final
end


main(ARGS_DICT)
