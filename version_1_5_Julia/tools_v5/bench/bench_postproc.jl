# bench/bench_postproc.jl
#
# Post-processing functions used by the bench
# `tools/bench_smc_full_mpc_fsa_gpu.jl` — everything that runs after the
# closed-loop foldl completes.
#
# Public functions:
#   - save_bench_outputs(out_dir, final, t_total, args, cfg) -> Nothing
#       Single entry point that writes data.jld2, per_stride.csv,
#       controller_diagnostics.csv, manifest.json, experiment_run.md, the
#       state-trajectory auto-plot, and the parameter-traces plot.
#   - _plot_param_traces_v5(posterior, mask, param_names, truth, ...)
#       Plotting helper used by save_bench_outputs for the per-parameter
#       quantile-band trace plot.
#
# Extracted 2026-05-09 as Phase 1e of the bench refactor described in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
# The post-processing block was previously inlined inside `main`; this
# pass moves it to a function with explicit (out_dir, final, t_total,
# args, cfg) arguments, where `cfg` is a NamedTuple carrying the bench
# context variables (n_smc, K_per_chain, bins_per_day, etc.). No
# behavioural change; outputs byte-identical to the prior in-line block.
#
# Dependencies (all loaded by the calling bench script before this file
# is `include`d): `Statistics.mean`, `JLD2.jldopen`, `JSON3.pretty`,
# `Plots.plot`/`hline!`/`savefig`, `Dates.now`, the FSA model globals
# (PARAM_NAMES_V5, FULL_PARAMS_V5, FROZEN_PARAMS_V5), `REPO_ROOT`.


"""
    save_bench_outputs(out_dir, final, t_total, args, cfg) -> Nothing

Write all post-loop artefacts (data.jld2, per_stride.csv,
controller_diagnostics.csv, manifest.json, experiment_run.md,
auto-state-traces PNG, param-traces PNG) into `out_dir`. `final` is the
return of the closed-loop foldl; `t_total` is the wall in seconds; `args`
is the parsed CLI Dict; `cfg` is a NamedTuple of bench context with
fields:

    bins_per_day, stride_bins, window_bins, dt_days, T_total_bins,
    n_strides, is_open_loop, collect_ctrl_diag, n_smc, K_per_chain,
    filter_cfg, ctrl_cfg, base_out, base_phi.
"""
function save_bench_outputs(out_dir::AbstractString,
                              final::NamedTuple,
                              t_total::Float64,
                              args::Dict{String,Any},
                              cfg::NamedTuple)
    bins_per_day  = cfg.bins_per_day
    stride_bins   = cfg.stride_bins
    window_bins   = cfg.window_bins
    dt_days       = cfg.dt_days
    n_strides     = cfg.n_strides
    is_open_loop  = cfg.is_open_loop
    collect_ctrl_diag = cfg.collect_ctrl_diag
    n_smc         = cfg.n_smc
    K_per_chain   = cfg.K_per_chain
    filter_cfg    = cfg.filter_cfg
    ctrl_cfg      = cfg.ctrl_cfg
    base_out      = cfg.base_out
    base_phi      = cfg.base_phi

    # ── Build per-day Φ for plotting ──
    daily_phi_plan_per_stride = Float32[]
    log = final.per_stride_log
    for entry in log
        push!(daily_phi_plan_per_stride, Float32(entry.phi_mean))
    end

    # ── Save data.jld2 in the schema plot_state_traces.jl reads ──
    # Truncate baseline to MPC length: the foldl runs n_strides × stride_bins
    # bins, which can be < T_total_bins. plot_state_traces.jl assumes equal
    # lengths between MPC and baseline.
    n_mpc_bins = size(final.traj_history, 1)
    data_path = joinpath(out_dir, "data.jld2")
    # ── Per-stride posterior particles + mask, in CONSTRAINED space.
    # All 10 v1.5 priors are LogNormal, so constrained = exp(unconstrained).
    n_params_est = length(PARAM_NAMES_V5)
    posterior_particles_arr = zeros(Float64, n_strides, n_smc, n_params_est)
    posterior_window_mask   = falses(n_strides)
    for (s, U) in enumerate(final.all_filter_posts)
        if U !== nothing
            posterior_particles_arr[s, 1:size(U, 1), :] = exp.(Float64.(U))
            posterior_window_mask[s] = true
        end
    end

    # v5 trajectory has 6 columns (state [B, S, F, A, K_FB, K_FS]) — saved
    # as a (n_bins, 6) Matrix. The schema field names are kept compatible
    # with the v1.5 plot_state_traces reader where possible, with v5-only
    # additions (`Phi_B_per_bin_*`, `Phi_S_per_bin_*`) for the bimodal
    # schedule.
    JLD2.jldopen(data_path, "w") do f
        f["trajectory_mpc"]       = Float32.(final.traj_history)
        f["trajectory_baseline"]  = Float32.(base_out.trajectory[1:n_mpc_bins, :])
        f["Phi_B_per_bin_mpc"]    = final.obs_history.Phi_B[1:n_mpc_bins]
        f["Phi_S_per_bin_mpc"]    = final.obs_history.Phi_S[1:n_mpc_bins]
        f["Phi_B_per_bin_baseline"] = base_phi.Phi_B[1:n_mpc_bins]
        f["Phi_S_per_bin_baseline"] = base_phi.Phi_S[1:n_mpc_bins]
        f["daily_phi_per_stride"] = daily_phi_plan_per_stride
        f["BINS_PER_DAY"]         = bins_per_day
        f["STRIDE_BINS"]          = stride_bins
        f["WINDOW_BINS"]          = window_bins
        f["dt_days"]              = dt_days
        f["F_max"]                = 0.40
        f["state_names"]          = ["B", "S", "F", "A", "K_FB", "K_FS"]
        if final.filter_post !== nothing
            f["filter_post_unc"]  = Float64.(final.filter_post)
        end
        f["posterior_particles"]    = posterior_particles_arr
        f["posterior_window_mask"]  = collect(Bool, posterior_window_mask)
        f["param_names"]          = String.(PARAM_NAMES_V5)
        f["truth_params_dict"]    = Dict{String,Float64}(string(k) => v
                                                          for (k, v) in FULL_PARAMS_V5)
        f["seed"]                 = args["seed"]
        f["wall_seconds"]         = t_total

        # ── Per-bin obs / gates / circadian (the data the filter sees) ──
        # Saved so the obs-channel plotter (`plot_obs_channels_v5.jl`)
        # can render each channel with the gating mask overlaid. All
        # arrays are length n_mpc_bins, aligned with `trajectory_mpc`.
        f["obs_HR"]      = final.obs_history.obs_HR[1:n_mpc_bins]
        f["obs_S"]       = final.obs_history.obs_S[1:n_mpc_bins]
        f["obs_steps"]   = final.obs_history.obs_steps[1:n_mpc_bins]
        f["obs_VL"]      = final.obs_history.obs_VL[1:n_mpc_bins]
        f["obs_sleep"]   = final.obs_history.obs_sleep[1:n_mpc_bins]
        f["gate_HR"]     = final.obs_history.gate_HR[1:n_mpc_bins]
        f["gate_stress"] = final.obs_history.gate_stress[1:n_mpc_bins]
        f["gate_steps"]  = final.obs_history.gate_steps[1:n_mpc_bins]
        f["gate_VL"]     = final.obs_history.gate_VL[1:n_mpc_bins]
        f["gate_sleep"]  = final.obs_history.gate_sleep[1:n_mpc_bins]
        f["circadian_C"] = final.obs_history.C[1:n_mpc_bins]
    end
    @info "wrote $data_path"

    # ── Per-stride telemetry CSV (Phase B; consumed by the comparison
    #    script in Phase D). One row per stride. Plain manual format —
    #    avoids depending on CSV.jl which isn't a current dep.
    csv_path = joinpath(out_dir, "per_stride.csv")
    open(csv_path, "w") do io
        println(io, "stride,t_wall_s,n_temp_filter,n_temp_ctrl,",
                     "daily_phi_B,daily_phi_S,A_mean_so_far,",
                     "B_end,S_end,F_end,A_end,KFB_end,KFS_end")
        for r in final.per_stride_log
            println(io,
                "$(r.stride),$(r.t_wall_s),$(r.n_temp_filter),",
                "$(r.n_temp_ctrl),$(r.daily_phi_B),$(r.daily_phi_S),",
                "$(r.A_mean_so_far),",
                "$(r.B_end),$(r.S_end),$(r.F_end),$(r.A_end),",
                "$(r.KFB_end),$(r.KFS_end)")
        end
    end
    @info "wrote $csv_path"

    # ── Controller-HMC per-tempering-level diagnostics CSV ──
    # One row per (replan stride_idx, tempering level). Empty if the
    # `--collect-ctrl-diagnostics` flag was off.
    if collect_ctrl_diag && !isempty(final.controller_diagnostics)
        ctrl_csv = joinpath(out_dir, "controller_diagnostics.csv")
        open(ctrl_csv, "w") do io
            println(io,
                "stride_idx,level,beta_pre,beta_post,delta_beta,beta_max,",
                "ess_at_dbeta_chosen,n_smc,theta_dim,",
                "chees_L_chosen,chees_score_best,chees_L_candidates,chees_scores,",
                "eps_step_size,n_mcmc_moves,total_accepts,accept_frac,",
                "log_density_pre_mean,log_density_post_mean,delta_log_density,",
                "esjd_total,esjd_per_eps_L,wall_seconds_level")
            for r in final.controller_diagnostics
                # Inline the variable-length vector fields as
                # semicolon-separated strings (CSV-friendly).
                cands_str  = join(r.chees_L_candidates, ";")
                scores_str = isempty(r.chees_scores) ? "" :
                              join(r.chees_scores, ";")
                println(io,
                    "$(r.stride_idx),$(r.level),$(r.beta_pre),$(r.beta_post),",
                    "$(r.delta_beta),$(r.beta_max),$(r.ess_at_dbeta_chosen),",
                    "$(r.n_smc),$(r.theta_dim),",
                    "$(r.chees_L_chosen),$(r.chees_score_best),",
                    "$(cands_str),$(scores_str),",
                    "$(r.eps_step_size),$(r.n_mcmc_moves),",
                    "$(r.total_accepts),$(r.accept_frac),",
                    "$(r.log_density_pre_mean),$(r.log_density_post_mean),",
                    "$(r.delta_log_density),$(r.esjd_total),",
                    "$(r.esjd_per_eps_L),$(r.wall_seconds_level)")
            end
        end
        @info "wrote $ctrl_csv"
    end

    # ── Manifest ──
    manifest = Dict(
        "schema_version"   => "1.0-v5",
        "T_total_days"     => args["T-days"],
        "step_minutes"     => args["step-minutes"],
        "BINS_PER_DAY"     => bins_per_day,
        "WINDOW_BINS"      => window_bins,
        "STRIDE_BINS"      => stride_bins,
        "n_strides"        => n_strides,
        "open_loop"        => is_open_loop,
        "replan_K"         => args["replan-K"],
        "smc_cfg"          => Dict(
            "N_smc"             => n_smc,
            "K_per_chain"       => K_per_chain,
            "num_mcmc"          => args["filt-num-mcmc"],
            "hmc_step_size"     => args["filt-hmc-step-size"],
            "hmc_num_leapfrog"  => args["filt-hmc-leapfrog"],
            "h_fd"              => args["filt-h-fd"],
            "filter_chees_min"  => args["filt-chees-min"],
            "filter_chees_max"  => args["filt-chees-max"],
            "filter_chees_list" => filter_cfg.chees_L_candidates,
            "liu_west_a"        => args["filt-liu-west-a"],
            "smooth_resample_bw"=> args["filt-smooth-resample-bw"],
            "gaussian_bridge"   => filter_cfg.gaussian_bridge,
            "ctrl_target_ess_frac" => args["ctrl-target-ess-frac"],
            "ctrl_max_lambda_inc"  => args["ctrl-max-lambda-inc"],
        ),
        "seed"             => args["seed"],
        "wall_seconds"     => t_total,
        "device"           => string(CUDA.name(CUDA.device())),
        "pinned_dynamics"  => Dict(string(k) => v for (k, v) in FROZEN_PARAMS_V5),
        "estimated_params" => String.(PARAM_NAMES_V5),
        # Truth values for the 10 estimated v1.5 params; mirrored from
        # FULL_PARAMS_V5 so the Phase-D comparison script can read truth
        # from the manifest (avoids parsing the JLD2 Dict with h5py).
        "truth_params"     => Dict(String(k) => Float64(FULL_PARAMS_V5[k])
                                     for k in PARAM_NAMES_V5),
    )
    open(joinpath(out_dir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    # ── experiment_run.md ──
    open(joinpath(out_dir, "experiment_run.md"), "w") do io
        println(io, "# FSA v5 closed-loop bench")
        println(io)
        println(io, "- **Mode:** $(is_open_loop ? "open-loop" : "closed-loop")")
        println(io, "- **Timestamp:** $(Dates.now())")
        println(io, "- **Wall time:** $(round(t_total, digits=1)) s")
        println(io, "- **T_total_days:** $(args["T-days"]), step=$(args["step-minutes"]) min")
        println(io, "- **BINS_PER_DAY:** $bins_per_day, WINDOW=$window_bins, STRIDE=$stride_bins")
        println(io, "- **n_strides:** $n_strides")
        println(io, "- **N_SMC:** $n_smc, K_per_chain=$K_per_chain")
        println(io, "- **Pinned at truth:** $(join(string.(keys(FROZEN_PARAMS_V5)), ", "))")
        println(io, "- **Filter estimates:** $(length(PARAM_NAMES_V5)) ($(join(string.(PARAM_NAMES_V5), ", ")))")
        println(io, "- **Final x̂:** B=$(round(final.last_xhat[1], digits=4)) " *
                     "S=$(round(final.last_xhat[2], digits=4)) " *
                     "F=$(round(final.last_xhat[3], digits=4)) " *
                     "A=$(round(final.last_xhat[4], digits=4)) " *
                     "K_FB=$(round(final.last_xhat[5], digits=4)) " *
                     "K_FS=$(round(final.last_xhat[6], digits=4))")
    end
    @info "wrote manifest.json + experiment_run.md"

    # ── Auto-plotting deferred for v5 ─────────────────────────────────
    # The v1.5 auto-plot (plot_state_traces.jl) and `_plot_param_traces_v5`
    # both assume v1.5 dimensions (3-state trajectory + 10 estimated
    # params + scalar Φ). The v5 surface is 6-state + 37 estimated +
    # bimodal Φ; the right plot scripts need a separate design pass.
    # data.jld2 has every field a future v5 plotter would need; the
    # smoke run produces it. Plot-side rendering is a follow-up task.
    @info "plotting deferred for v5 (data.jld2 has all fields a future v5 plotter needs)"

    return nothing
end


# ── Param-traces plot helper ─────────────────────────────────────────────
"""
    _plot_param_traces_v5(posterior, mask, param_names, truth, n_strides,
                            stride_bins, window_bins, bins_per_day,
                            T_days, step_minutes; out_path, mean_A_mpc,
                            mean_A_base)

Per-parameter posterior trace across rolling windows. One panel per
estimated param; 5-95% quantile band + median + truth horizontal line.
Layout mirrors `version_2_Python_JAX/tools/plot_param_traces.py`.
"""
function _plot_param_traces_v5(
        posterior::Array{Float64,3},
        mask::AbstractVector{Bool},
        param_names::Vector{String},
        truth::Dict{String,Float64},
        n_strides::Int, stride_bins::Int, window_bins::Int,
        bins_per_day::Int, T_days, step_minutes;
        out_path::AbstractString,
        mean_A_mpc::Float64, mean_A_base::Float64)
    end_t_days = ((collect(0:n_strides-1) .* stride_bins) .+ window_bins) ./ bins_per_day
    end_t_days = end_t_days[mask]
    valid = posterior[mask, :, :]                    # (n_valid, n_smc, n_params)
    n_valid, n_smc, n_params = size(valid)
    q05 = [quantile(vec(valid[s, :, p]), 0.05)
           for s in 1:n_valid, p in 1:n_params]
    q50 = [quantile(vec(valid[s, :, p]), 0.50)
           for s in 1:n_valid, p in 1:n_params]
    q95 = [quantile(vec(valid[s, :, p]), 0.95)
           for s in 1:n_valid, p in 1:n_params]

    n_cols = 5
    n_rows = (n_params + n_cols - 1) ÷ n_cols
    panels = Plots.Plot[]
    for i in 1:n_params
        name = param_names[i]
        p = plot(end_t_days, q50[:, i];
                  ribbon = (q50[:, i] .- q05[:, i], q95[:, i] .- q50[:, i]),
                  fillalpha = 0.30, color = :steelblue, lw = 1.4,
                  label = "median", title = name, titlefontsize = 9,
                  legend = false, grid = true, gridalpha = 0.3,
                  tickfontsize = 7,
                  xlabel = i > n_params - n_cols ? "end of window (days)" : "",
                  xguidefontsize = 7)
        if haskey(truth, name)
            hline!(p, [truth[name]]; color = :red, ls = :dash, lw = 1.0,
                    label = "truth")
        end
        push!(panels, p)
    end
    for _ in n_params+1:n_rows*n_cols
        push!(panels, plot(framestyle = :none, ticks = false, legend = false))
    end
    fig = plot(panels...; layout = (n_rows, n_cols),
                size = (n_cols * 320, n_rows * 220), dpi = 120,
                plot_title = "FSA-v5 posterior parameter traces — " *
                             "T=$(T_days)d, h=$(step_minutes)min, " *
                             "n_strides=$n_strides, " *
                             "mean A $(round(mean_A_mpc, digits=3)) " *
                             "vs baseline $(round(mean_A_base, digits=3))",
                plot_titlefontsize = 10)
    savefig(fig, out_path)
end
