# Live TensorBoard logging helper for the v5 closed-loop SMC²-MPC bench.
#
# Mirrors the series the end-of-run plotters consume (state-traces +
# param-traces) so the live feed is bit-equivalent to the eventual PNGs
# at matching strides. Called from `tools_v5/bench/bench_loop.jl` at
# the end of every `run_one_stride_v5` iteration when the bench was
# launched with `--tensorboard true`.
#
# This file is `include`d by `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`
# right after `bench_glue_v5.jl`, so it inherits the bench's namespace
# (in particular `PARAM_NAMES_V5` and `TBLogger` / `log_value`).
#
# Schema (per stride):
#   state/truth/{B,S,F,A,K_FB,K_FS}              ← p.final_state (MPC truth)
#   state/posterior_mean/{B,S,F,A,K_FB,K_FS}     ← new_xhat
#   state/baseline/{B,S,F,A,K_FB,K_FS}           ← base_traj sliced at
#                                                   end-of-stride bin —
#                                                   matches the grey-dashed
#                                                   baseline curves on the
#                                                   end-of-run state-traces PNG
#   phi/mean_B, phi/mean_S                       ← stride means of Φ (MPC)
#   phi/baseline_B, phi/baseline_S               ← constant 1.0 (matches the
#                                                   "baseline Φ=1.0" line on
#                                                   the end-of-run plot)
#   wall/stride_s                                ← stride wall time
#   tempering/n_temp_filter, tempering/n_temp_ctrl
#   params/<name>/{q05,q50,q95} for each of 37   ← exp.(filter_post)
#                                                  quantiles, only when
#                                                  filter is active
#   params/<name>/truth for each of 37           ← TRUTH_PARAMS_V5[name],
#                                                  matches the truth
#                                                  horizontal line on the
#                                                  end-of-run param-traces
#                                                  PNG (logged every stride
#                                                  so TB renders a flat line)
#
# Total scalars per stride: 65 during warmup; 173 once filter active.

using Statistics: mean, quantile

export log_stride_to_tb!


function log_stride_to_tb!(lg::TBLogger, stride_idx::Int,
                            final_state::AbstractVector,
                            xhat::AbstractVector,
                            filter_post::Union{Nothing, AbstractMatrix},
                            phi_B::AbstractVector,
                            phi_S::AbstractVector,
                            log_row::NamedTuple,
                            base_traj::AbstractMatrix,
                            stride_bins::Int,
                            base_phi_B::Real,
                            base_phi_S::Real)
    state_names = (:B, :S, :F, :A, :K_FB, :K_FS)

    # Baseline state at the same end-of-stride bin index as the MPC truth.
    # Clamp to the trajectory's last row in case stride_idx * stride_bins
    # ever overruns (defensive — shouldn't happen in normal operation).
    base_bin = min(stride_idx * stride_bins, size(base_traj, 1))

    for (i, name) in enumerate(state_names)
        log_value(lg, "state/truth/$(name)",          Float64(final_state[i]);    step = stride_idx)
        log_value(lg, "state/posterior_mean/$(name)", Float64(xhat[i]);           step = stride_idx)
        log_value(lg, "state/baseline/$(name)",       Float64(base_traj[base_bin, i]); step = stride_idx)
    end

    log_value(lg, "phi/mean_B",     Float64(mean(phi_B));  step = stride_idx)
    log_value(lg, "phi/mean_S",     Float64(mean(phi_S));  step = stride_idx)
    log_value(lg, "phi/baseline_B", Float64(base_phi_B);   step = stride_idx)
    log_value(lg, "phi/baseline_S", Float64(base_phi_S);   step = stride_idx)

    log_value(lg, "wall/stride_s",           Float64(log_row.t_wall_s);     step = stride_idx)
    log_value(lg, "tempering/n_temp_filter", Int(log_row.n_temp_filter);    step = stride_idx)
    log_value(lg, "tempering/n_temp_ctrl",   Int(log_row.n_temp_ctrl);      step = stride_idx)

    # Truth parameter horizontal lines — match the end-of-run param-traces
    # PNG. Logged every stride because TB scalar charts render a series
    # only at the steps where it was logged. Use FULL_PARAMS_V5 (the merge
    # of TRUTH_PARAMS_V5 + DEFAULT_OBS_PARAMS_V5) — same dict bench_postproc
    # serialises as `truth_params_dict` for the plotter, since the 37
    # estimated params include 22 obs-channel keys not in TRUTH_PARAMS_V5.
    for name in PARAM_NAMES_V5
        log_value(lg, "params/$(name)/truth", Float64(FULL_PARAMS_V5[name]); step = stride_idx)
    end

    if filter_post !== nothing
        # exp(U) is the same constrained-space conversion bench_postproc
        # does at end-of-run before the param-traces plotter.
        constrained = exp.(Float64.(filter_post))     # (n_smc, 37)
        for (p, name) in enumerate(PARAM_NAMES_V5)
            col = view(constrained, :, p)
            log_value(lg, "params/$(name)/q05", quantile(col, 0.05); step = stride_idx)
            log_value(lg, "params/$(name)/q50", quantile(col, 0.50); step = stride_idx)
            log_value(lg, "params/$(name)/q95", quantile(col, 0.95); step = stride_idx)
        end
    end
    return nothing
end
