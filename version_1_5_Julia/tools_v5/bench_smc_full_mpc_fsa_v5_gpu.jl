#!/usr/bin/env julia
# FSA v5 closed-loop SMC²-MPC bench — purely functional foldl over strides.
#
# Mirrors `tools/bench_smc_full_mpc_fsa_gpu.jl` (the v1.5 driver) but
# wires up the v5 surface:
#   - 6D state, bimodal (Φ_B, Φ_S) stimulus, 5-channel obs (HR / Sleep /
#     Stress / Steps / VolumeLoad) with explicit per-bin gates, 15-min
#     bin grid (BINS_PER_DAY = 96, dt = 1/96 days).
#   - Filter target: `FSAv5GPUTarget` (37-D estimated θ; 14 frozen).
#   - Controller target: `FSAv5ControlGPUTarget` with SOFT cost only
#     (HARD chance-constraint variant deliberately out of scope).
#
# Per stride:
#   1. Slice next stride's bimodal (Φ_B, Φ_S) from the current plan.
#   2. plant_rollout_v5(state, phi_seq, full_params, dt, key) → trajectory + obs.
#   3. Build window grid_obs from the accumulated obs history.
#   4. Run outer SMC² on the filter (tempered + framework HMC).
#   5. Maybe replan: framework's run_tempered_smc_gpu on the controller's
#      RBF θ (2·n_anchors-dim).
#
# Modes:
#   --open-loop false   default — closed-loop with replanning every K strides.
#   --open-loop true    one up-front plan from the selected --init-preset
#                        + TRUTH_PARAMS_V5, no replan.

using Dates


# CLI parsing: same defaults / aliases / parser as v1.5 with one tweak
# (step-minutes default 60 → 15). Pure CLI logic; no model dependencies.
include(joinpath(@__DIR__, "bench", "bench_args.jl"))


const ARGS_DICT = _parse_args(copy(ARGS))
ENV["FSA_STEP_MINUTES"] = string(ARGS_DICT["step-minutes"])

@info "loading v5 model + framework (FSA_STEP_MINUTES=$(ENV["FSA_STEP_MINUTES"]))..."

using Random, Statistics, Printf, LinearAlgebra
using LogExpFunctions: logsumexp
using JLD2
using JSON3
using CUDA
using StaticArrays
using StableRNGs
using TensorBoardLogger: TBLogger, tb_overwrite, log_value

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_v5", "FSAv5.jl"))
using .FSAv5
using .FSAv5.PlantV5: PlantState6D, plant_rollout_v5, init_plant_state_sedentary,
                       init_plant_state_trained, init_plant_state_from
using .FSAv5.SimulationV5: BINS_PER_DAY, DT_BIN_DAYS,
                            TRUTH_PARAMS_V5, TRUTH_PARAMS_V5_RECOMMENDED_V2,
                            DEFAULT_OBS_PARAMS_V5,
                            FROZEN_PARAMS_V5, FROZEN_PARAMS_V5_RECOMMENDED_V2,
                            SEDENTARY_INIT,
                            TRAINED_ATHLETE_INIT, TRAINED_ATHLETE_INIT_V2,
                            MIDDLE_INIT_V2,
                            PARAM_KEYS_V5, OBS_PARAM_KEYS_V5,
                            select_truth_preset
using .FSAv5.EstimationV5: PARAM_NAMES_V5, PARAM_PRIOR_CONFIG_V5,
                            PARAM_PRIOR_CONFIG_V5_RECOMMENDED_V2,
                            select_prior_config_v5
using .FSAv5.GPUPFv5: FSAv5GPUTarget, gpu_log_density_v5
using .FSAv5.GPUControlV5: FSAv5ControlGPUTarget,
                            gpu_cost_log_density_batched_v5,
                            make_log_density_fn_v5,
                            eval_cost_decomp_v5
# Same framework as v1.5; HMC + ChEES picker + bridge sampler.
using SMC2FC_functional: run_tempered_smc_gpu,
                          fit_gaussian, sample_from_gaussian,
                          silverman_bandwidth, log_kernel_matrix,
                          parallel_hmc_one_move_generic!,
                          chees_pick_L_generic


# Bench's `FULL_PARAMS_V5` constant used by the post-processor: the
# merged 50-key Dict (28 dynamics + 22 obs-channel + frozen). Exposed
# as a top-level constant because `bench_postproc.jl` references it
# by name (mirrors v1.5's pattern of referencing `DEFAULT_PARAMS`).
const FULL_PARAMS_V5 = merge(TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5)


# v5 model-specific glue (posterior_mean_v5, window_grid_obs_v5).
include(joinpath(REPO_ROOT, "models", "fsa_v5", "bench_glue_v5.jl"))


# Live TensorBoard logging helper (log_stride_to_tb!). Sees PARAM_NAMES_V5
# from the bench's namespace where it's already imported.
include(joinpath(REPO_ROOT, "models", "fsa_v5", "bench_tb_logging.jl"))


# Bench filter SMC². Same algorithm as v1.5 (tempered λ-ladder + ChEES
# HMC at every level) but wired to FSAv5GPUTarget and gpu_log_density_v5.
include(joinpath(@__DIR__, "bench", "bench_filter.jl"))


# Bench controller-side wrapper. Builds an FSAv5ControlGPUTarget and
# calls run_tempered_smc_gpu on the 2·n_anchors-D RBF θ; decodes
# posterior-mean θ → bimodal (Φ_B, Φ_S) plan.
include(joinpath(@__DIR__, "bench", "bench_controller.jl"))


# Bench post-processing — writes data.jld2, per_stride.csv,
# controller_diagnostics.csv, manifest.json, experiment_run.md.
# Plotting is deferred for v5 (the v1.5 plotters assume 3-state /
# scalar-Φ shapes; v5 needs a separate plotter design).
include(joinpath(@__DIR__, "bench", "bench_postproc.jl"))


# Closed-loop foldl over strides — v5-shaped accumulator + ctx.
include(joinpath(@__DIR__, "bench", "bench_loop.jl"))


# ── Main bench ───────────────────────────────────────────────────────────

function main(args::Dict{String,Any})
    bins_per_day = (60 * 24) ÷ args["step-minutes"]
    window_bins  = bins_per_day
    stride_bins  = window_bins ÷ 2
    dt_days      = 1.0 / bins_per_day
    T_total_bins = args["T-days"] * bins_per_day
    n_strides    = (T_total_bins - window_bins) ÷ stride_bins + 1
    if haskey(ENV, "FSA_V5_SWEEP_TARGET_STRIDE")
        n_strides = min(n_strides, parse(Int, ENV["FSA_V5_SWEEP_TARGET_STRIDE"]))
        @info "FSA_V5_SWEEP_TARGET_STRIDE detected: overriding n_strides to $n_strides"
    end
    H_plan_bins  = T_total_bins                       # FIXED, per §2.11

    is_open_loop = lowercase(args["open-loop"]) in ("true", "1", "yes")

    # ── Truth-side parameter preset selection ──
    # `full_params_v5` is the local merged dynamics+obs dict that downstream
    # sites consume (plant rollout, open-loop initial controller plan, and
    # ctx.full_params for the closed-loop foldl). The top-level
    # `FULL_PARAMS_V5` const stays canonical-only — bench_postproc.jl reads
    # truth params from ctx.full_params instead (see ctx wiring below).
    truth_preset_str = string(args["truth-preset"])
    # Single dispatch helper (SimulationV5.select_truth_preset) — keeps the
    # truth dict and the frozen dict consistent. Inline branching here was
    # the original source of the canonical-B_dec leak under v2.
    truth_params_v5_local, frozen_params_v5_local =
        select_truth_preset(truth_preset_str)
    # Prior-config dispatch — parallel to truth/frozen. Cold-starts the
    # filter centred on the active truth regime instead of always on
    # canonical, so the posterior doesn't have to migrate the full
    # canonical→v2 gap (which previously cost many strides of tempering
    # and left the controller planning against canonical-shape posteriors).
    prior_config_v5_local = select_prior_config_v5(truth_preset_str)
    full_params_v5 = merge(truth_params_v5_local, DEFAULT_OBS_PARAMS_V5)

    # ── Startup dump of the EXACT parameter vector being used ──
    # Prints for every run (not launcher-specific). Use stdout (println)
    # so the block is easy to grep in logs and reads cleanly. The 28
    # dynamics keys are listed in PARAM_KEYS_V5 canonical order; the 22
    # observation keys in OBS_PARAM_KEYS_V5 order.
    println("=" ^ 76)
    println("Truth parameter vector  (--truth-preset $truth_preset_str)")
    println("=" ^ 76)
    println("Dynamics ($(length(PARAM_KEYS_V5))):")
    for k in PARAM_KEYS_V5
        @printf("  %-12s = %-14.6g  (raw: %s)\n",
                string(k), Float64(full_params_v5[k]), repr(full_params_v5[k]))
    end
    println("Observation ($(length(OBS_PARAM_KEYS_V5))):")
    for k in OBS_PARAM_KEYS_V5
        @printf("  %-12s = %-14.6g  (raw: %s)\n",
                string(k), Float64(full_params_v5[k]), repr(full_params_v5[k]))
    end
    println("Frozen ($(length(frozen_params_v5_local))) — fed to filter inner-PF +")
    println("                            controller posterior_mean_v5:")
    for k in sort(collect(keys(frozen_params_v5_local)))
        @printf("  %-12s = %-14.6g  (raw: %s)\n",
                string(k), Float64(frozen_params_v5_local[k]),
                repr(frozen_params_v5_local[k]))
    end
    # Prior centres for the 6 v2-affected dynamics keys — the filter cold-
    # starts here, so a mismatch with the truth dict would mean the filter
    # has to migrate the posterior across many tempering levels.
    println("Prior LogNormal centres (6 dynamics keys whose v2 value differs)")
    println("  format: name = centre  (log(centre) = μ_prior, σ_prior = 0.30)")
    _prior_means = Dict(name => μ for (name, _, μ, _) in prior_config_v5_local)
    for k in (:tau_B, :kappa_B, :tau_S, :kappa_S, :mu_F, :mu_FF)
        μ = _prior_means[k]
        @printf("  %-12s = %-14.6g  (log = %-10.5g)\n", string(k), exp(μ), μ)
    end
    println("=" ^ 76)
    flush(stdout)

    @info "config: T=$(args["T-days"])d step=$(args["step-minutes"])min " *
          "BINS_PER_DAY=$bins_per_day WINDOW=$window_bins STRIDE=$stride_bins " *
          "n_strides=$n_strides  open_loop=$is_open_loop  truth_preset=$truth_preset_str"

    out_dir = isempty(args["output-dir"]) ?
              joinpath(REPO_ROOT, "outputs", "fsa_v5", "g4_runs",
                        "T$(args["T-days"])d_$(is_open_loop ? "OL" : "CL_K$(args["replan-K"])")_h$(args["step-minutes"])min") :
              args["output-dir"]
    mkpath(out_dir)

    @info "GPU device: $(CUDA.name(CUDA.device()))"

    # ── Live TensorBoard logging (opt-in via --tensorboard true) ──
    # Lenient flag parsing mirrors --open-loop above. When enabled,
    # writes per-stride scalars directly to <out_dir>. The user
    # launches `tensorboard --logdir <SWEEP_ROOT>` separately.
    tb_enabled = lowercase(string(args["tensorboard"])) in ("true", "1", "yes")
    tb_logger = if tb_enabled
        @info "TensorBoard live logging ENABLED → $out_dir"
        @info "    Watch live with:  tensorboard --logdir $(dirname(out_dir))"
        TBLogger(out_dir, tb_overwrite)
    else
        nothing
    end

    # ── Plant initial-state preset selection ──
    # Single source of truth: every downstream site reads from
    # `(init_state_fn, init_nt)`, never from SEDENTARY_INIT /
    # TRAINED_ATHLETE_INIT directly. Enforces "all sites use the same
    # preset" by construction.
    init_preset = string(args["init-preset"])
    # Single source of truth: `init_nt` is the NamedTuple. The plant-state
    # constructor is derived from it via `init_plant_state_from`, so the
    # plant init and the NaN-fallback values can never disagree.
    init_nt = if init_preset == "TRAINED_ATHLETE_INIT"
        TRAINED_ATHLETE_INIT
    elseif init_preset == "TRAINED_ATHLETE_INIT_V2"
        TRAINED_ATHLETE_INIT_V2
    elseif init_preset == "SEDENTARY_INIT"
        SEDENTARY_INIT
    elseif init_preset == "MIDDLE_INIT_V2"
        MIDDLE_INIT_V2
    else
        error("--init-preset must be one of " *
              "\"TRAINED_ATHLETE_INIT\", \"TRAINED_ATHLETE_INIT_V2\", " *
              "\"SEDENTARY_INIT\", \"MIDDLE_INIT_V2\"; got: \"$init_preset\"")
    end
    init_state_fn = () -> init_plant_state_from(init_nt)

    # v2-only presets — both are v2-specific reference states.
    if init_preset in ("TRAINED_ATHLETE_INIT_V2", "MIDDLE_INIT_V2") &&
       truth_preset_str != "v2"
        error("--init-preset $init_preset requires --truth-preset v2 " *
              "(this init is a v2-specific reference state).")
    end
    if init_preset == "TRAINED_ATHLETE_INIT" && truth_preset_str == "v2"
        @warn "Canonical TRAINED_ATHLETE_INIT used with --truth-preset v2 — " *
              "not on v2 slow manifold; consider TRAINED_ATHLETE_INIT_V2."
    end
    @info "init preset: $init_preset"

    # ── Build filter target ──
    n_smc = args["filt-n-smc"]
    K_per_chain = args["filt-k-per-chain"]
    d = length(PARAM_NAMES_V5)        # = 37
    M_max = n_smc * (1 + 2 * d)
    @info "filter target: N_SMC=$n_smc K_per_chain=$K_per_chain d=$d M_max=$M_max"
    target = FSAv5GPUTarget(
        K_per_chain = K_per_chain, M_max = M_max,
        T_steps = window_bins, R = 4, dt = dt_days,
        noise_seed = 0,
        ot_max_weight = args["filt-ot-max-weight"],
        nan_fallback_B   = Float32(init_nt.B),
        nan_fallback_S   = Float32(init_nt.S),
        nan_fallback_F   = Float32(init_nt.F),
        nan_fallback_A   = Float32(init_nt.A),
        nan_fallback_KFB = Float32(init_nt.KFB),
        nan_fallback_KFS = Float32(init_nt.KFS),
        frozen           = frozen_params_v5_local,
    )
    if args["filt-ot-max-weight"] >= 1e-6
        @warn "OT rescue ENABLED — expect ~12× slowdown (v1.5 lesson)."
    else
        @info "filter target: OT rescue DISABLED (fast path)"
    end

    # ── Filter prior (unconstrained) ──
    # `prior_config_v5_local` was already dispatched on --truth-preset
    # earlier in main() (alongside truth/frozen). Extract the (μ, σ) per
    # estimated param here.
    prior_means  = Float64[m for (_, _, m, _) in prior_config_v5_local]
    prior_sigmas = Float64[s for (_, _, _, s) in prior_config_v5_local]
    @assert length(prior_means)  == d
    @assert length(prior_sigmas) == d

    filter_chees_min = args["filt-chees-min"]
    filter_chees_max = args["filt-chees-max"]
    filter_chees_list = [filter_chees_min]
    while filter_chees_list[end] * 2 <= filter_chees_max
        push!(filter_chees_list, filter_chees_list[end] * 2)
    end

    filter_cfg = (
        max_lambda_inc    = args["filt-max-lambda-inc"],
        target_ess_frac   = args["filt-target-ess-frac"],
        num_mcmc          = args["filt-num-mcmc"],
        hmc_step          = args["filt-hmc-step-size"],
        hmc_leap          = args["filt-hmc-leapfrog"],
        max_levels        = args["filt-max-temp-levels"],
        h_fd              = args["filt-h-fd"],
        liu_west_a        = args["filt-liu-west-a"],
        smooth_resample_bw = args["filt-smooth-resample-bw"],
        gaussian_bridge   = lowercase(string(args["filt-gaussian-bridge"])) == "true",
        chees_L_candidates = filter_chees_list,
    )

    # ── Controller config ──
    ctrl_n_anchors = args["ctrl-n-anchors"]
    ctrl_n_smc     = args["ctrl-n-smc"]
    ctrl_n_inner   = args["ctrl-n-inner"]
    ctrl_theta_dim = 2 * ctrl_n_anchors      # bimodal: B + S channels
    ctrl_M_max     = ctrl_n_smc * (1 + 2 * ctrl_theta_dim)
    chees_max  = args["ctrl-chees-max"]
    chees_list = [16]
    while chees_list[end] * 2 <= chees_max
        push!(chees_list, chees_list[end] * 2)
    end
    ctrl_cfg = (
        n_smc       = ctrl_n_smc,
        n_inner     = ctrl_n_inner,
        n_anchors   = ctrl_n_anchors,    # per-channel (decoder doubles internally)
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
        phi_default     = args["ctrl-phi-default"],
        lam_phi         = args["ctrl-lam-phi"],
        lam_a           = args["ctrl-lam-a"],
        lam_b           = args["ctrl-lam-b"],
        lam_s           = args["ctrl-lam-s"],
        lam_f           = args["ctrl-lam-f"],
        lam_chance      = args["ctrl-lam-chance"],
        a_thr           = args["ctrl-a-thr"],
        lam_chance_b    = args["ctrl-lam-chance-b"],
        b_thr           = args["ctrl-b-thr"],
        lam_chance_s    = args["ctrl-lam-chance-s"],
        s_thr           = args["ctrl-s-thr"],
        lam_island      = args["ctrl-lam-island"],
        beta_island     = args["ctrl-beta-island"],
    )
    @info "controller cfg: n_smc=$ctrl_n_smc n_inner=$ctrl_n_inner per-channel n_anchors=$ctrl_n_anchors θ_dim=$ctrl_theta_dim chees=$chees_list"

    # ── Initial Φ override (--init-phi-B / --init-phi-S) ──
    # Single source of truth for the baseline plant rollout AND the
    # closed-loop initial plan. Validates against the controller's
    # Phi_max=3.0 cap. In open-loop mode (--open-loop true) only the
    # baseline rollout uses these values; the controller plans the
    # full horizon up front and overrides the closed-loop fallback.
    init_phi_B = Float32(args["init-phi-B"])
    init_phi_S = Float32(args["init-phi-S"])
    phi_max = 3.0f0   # mirrors gpu_control_v5.jl Phi_max default
    (0.0f0 <= init_phi_B <= phi_max) || error(
        "--init-phi-B must be in [0, $phi_max], got $init_phi_B")
    (0.0f0 <= init_phi_S <= phi_max) || error(
        "--init-phi-S must be in [0, $phi_max], got $init_phi_S")
    @info "initial Φ (baseline + closed-loop init plan): B=$init_phi_B S=$init_phi_S"

    # ── Pre-roll the BASELINE plant under (init_phi_B, init_phi_S) ──
    # Constant Φ over the full horizon; produces the grey reference
    # curves on end-of-run plots and `state/baseline/*` series in TB.
    base_seed = UInt64(args["seed"])
    base_plant_state = init_state_fn()
    base_phi_seq = [(init_phi_B, init_phi_S) for _ in 1:T_total_bins]
    base_out  = plant_rollout_v5(base_plant_state, base_phi_seq,
                                    full_params_v5, Float32(dt_days),
                                    hash((base_seed, :base_plant)))
    base_phi  = (Phi_B = base_out.Phi_B, Phi_S = base_out.Phi_S)

    # ── Open-loop initial plan ──
    initial_plan_B, initial_plan_S = if is_open_loop
        @info "OPEN-LOOP: building one initial plan from $init_preset + truth_preset=$truth_preset_str..."
        s0 = SVector{6, Float32}(
            init_nt.B,   init_nt.S,
            init_nt.F,   init_nt.A,
            init_nt.KFB, init_nt.KFS,
        )
        out = controller_plan_v5(full_params_v5, s0, H_plan_bins,
                                    max(1, bins_per_day ÷ 24),
                                    Float32(dt_days), ctrl_cfg,
                                    hash((base_seed, :ctrl_init)))
        @info @sprintf("  open-loop plan ready: Φ̄_B=%.3f Φ̄_S=%.3f  n_temp=%d",
                       mean(out.Phi_B_plan), mean(out.Phi_S_plan),
                       out.n_temp_ctrl)
        (out.Phi_B_plan, out.Phi_S_plan)
    else
        (fill(init_phi_B, T_total_bins),
         fill(init_phi_S, T_total_bins))
    end

    collect_ctrl_diag = lowercase(string(args["collect-ctrl-diagnostics"])) == "true"

    # ── Closed-loop foldl over strides ──
    init_acc = (
        plant_state    = init_state_fn(),
        plan_phi_B     = initial_plan_B,
        plan_phi_S     = initial_plan_S,
        plan_offset    = 0,
        filter_post    = nothing,
        all_filter_posts = Vector{Union{Nothing,Matrix{Float64}}}(),
        traj_history   = Matrix{Float64}(undef, 0, 6),
        obs_history    = (
            obs_HR    = Float32[], obs_S    = Float32[],
            obs_steps = Float32[], obs_VL   = Float32[],
            obs_sleep = Float32[],
            gate_HR   = Float32[], gate_stress = Float32[],
            gate_steps= Float32[], gate_VL  = Float32[],
            gate_sleep = Float32[],
            Phi_B = Float32[], Phi_S = Float32[],
            C     = Float32[],
        ),
        per_stride_log = NamedTuple[],
        controller_diagnostics = NamedTuple[],
        last_xhat      = SVector{6, Float64}(
            init_nt.B,   init_nt.S,
            init_nt.F,   init_nt.A,
            init_nt.KFB, init_nt.KFS),
    )

    ctx = (
        target            = target,
        n_smc             = n_smc,
        full_params       = full_params_v5,
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
        init_nt           = init_nt,
        tb_logger         = tb_logger,
        base_traj         = base_out.trajectory,
        init_phi_B        = init_phi_B,
        init_phi_S        = init_phi_S,
        frozen_params     = frozen_params_v5_local,
    )

    t_total_start = time()
    final = run_closed_loop_bench_v5(ctx, init_acc)
    t_total = time() - t_total_start

    @info @sprintf("Total wall time: %.1f s", t_total)

    # ── Post-processing ──
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
        full_params       = full_params_v5,
        truth_preset      = truth_preset_str,
        frozen_params     = frozen_params_v5_local,
    ))

    return final
end


# `FSA_V5_DRY_RUN=1` skips the actual bench (the include-and-link smoke
# test that the diff-test harness relies on). Production runs leave the
# env var unset.
if !haskey(ENV, "FSA_V5_DRY_RUN")
    main(ARGS_DICT)
else
    @info "FSA_V5_DRY_RUN=1 — driver loaded but main() skipped"
end
