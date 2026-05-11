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

export log_stride_to_tb!, log_cost_decomp_to_tb!


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
                            base_phi_S::Real,
                            truth_params::AbstractDict)
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

    # Bifurcation parameter μ̄(B, S, F) — state-level (PDF Eq. 6), evaluated
    # at the current trajectory state. μ̄ > 0 ⇒ healthy attractor exists at
    # this (B, S, F); μ̄ < 0 ⇒ Stuart-Landau drive is collapsing A toward 0.
    # Watching μ̄ rise over time is the cleanest "controller is doing
    # something correct" diagnostic: if A is on its way up, μ̄ must have
    # crossed into positive territory first. Three series for direct
    # comparison of plant truth vs filter belief vs do-nothing reference.
    truth_mu     = mu_bar_state(Float64(final_state[1]),   # B
                                  Float64(final_state[2]),   # S
                                  Float64(final_state[3]),   # F
                                  truth_params)
    posterior_mu = mu_bar_state(Float64(xhat[1]),
                                  Float64(xhat[2]),
                                  Float64(xhat[3]),
                                  truth_params)
    baseline_mu  = mu_bar_state(Float64(base_traj[base_bin, 1]),
                                  Float64(base_traj[base_bin, 2]),
                                  Float64(base_traj[base_bin, 3]),
                                  truth_params)
    log_value(lg, "bifurcation/truth_mu",          truth_mu;     step = stride_idx)
    log_value(lg, "bifurcation/posterior_mean_mu", posterior_mu; step = stride_idx)
    log_value(lg, "bifurcation/baseline_mu",       baseline_mu;  step = stride_idx)

    log_value(lg, "phi/mean_B",     Float64(mean(phi_B));  step = stride_idx)
    log_value(lg, "phi/mean_S",     Float64(mean(phi_S));  step = stride_idx)
    log_value(lg, "phi/baseline_B", Float64(base_phi_B);   step = stride_idx)
    log_value(lg, "phi/baseline_S", Float64(base_phi_S);   step = stride_idx)

    log_value(lg, "wall/stride_s",           Float64(log_row.t_wall_s);     step = stride_idx)
    log_value(lg, "tempering/n_temp_filter", Int(log_row.n_temp_filter);    step = stride_idx)
    log_value(lg, "tempering/n_temp_ctrl",   Int(log_row.n_temp_ctrl);      step = stride_idx)

    # Truth parameter horizontal lines — match the end-of-run param-traces
    # PNG. Logged every stride because TB scalar charts render a series
    # only at the steps where it was logged. Reads from the caller-supplied
    # `truth_params` dict (= cfg.full_params from the bench), so under
    # --truth-preset v2 the lines reflect the v2 overrides rather than the
    # top-level canonical FULL_PARAMS_V5 const.
    for name in PARAM_NAMES_V5
        log_value(lg, "params/$(name)/truth", Float64(truth_params[name]); step = stride_idx)
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


"""
    log_cost_decomp_to_tb!(lg, stride_idx, decomp, ctrl_cfg) -> Nothing

Push the per-replan cost decomposition (output of `eval_cost_decomp_v5`)
to TensorBoard. Three groups of series, all keyed by `stride_idx`:

  cost/raw/<term>_acc        ← the unweighted ∫ X dt accumulators
                                (effort, A, B, S, barrier, chance,
                                 chance_B, chance_S, island)
  cost/weighted/<term>        ← each term's signed contribution to J
                                (matches the kernel's cost-line algebra)
  cost/total/J                ← the full sum the controller is minimising
  cost/coef/<lam>             ← the active coefficient values (lam_phi,
                                 lam_b, lam_s, lam_F, lam_chance,
                                 lam_chance_b, lam_chance_s, lam_island)
                                 — logged every replan so TB renders flat
                                 lines and you can see at-a-glance which
                                 terms are switched on
"""
function log_cost_decomp_to_tb!(lg::TBLogger, stride_idx::Int,
                                  decomp::NamedTuple, ctrl_cfg::NamedTuple)
    # Raw unweighted integrals.
    log_value(lg, "cost/raw/effort_acc",   Float64(decomp.effort_acc);   step = stride_idx)
    log_value(lg, "cost/raw/A_acc",        Float64(decomp.A_acc);        step = stride_idx)
    log_value(lg, "cost/raw/B_acc",        Float64(decomp.B_acc);        step = stride_idx)
    log_value(lg, "cost/raw/S_acc",        Float64(decomp.S_acc);        step = stride_idx)
    log_value(lg, "cost/raw/barrier_acc",  Float64(decomp.barrier_acc);  step = stride_idx)
    log_value(lg, "cost/raw/chance_acc",   Float64(decomp.chance_acc);   step = stride_idx)
    log_value(lg, "cost/raw/chance_B_acc", Float64(decomp.chance_B_acc); step = stride_idx)
    log_value(lg, "cost/raw/chance_S_acc", Float64(decomp.chance_S_acc); step = stride_idx)
    log_value(lg, "cost/raw/island_acc",   Float64(decomp.island_acc);   step = stride_idx)

    # Weighted contributions (signed, sum to J_total).
    log_value(lg, "cost/weighted/effort",   Float64(decomp.w_effort);    step = stride_idx)
    log_value(lg, "cost/weighted/A_reward", Float64(decomp.w_A_reward);  step = stride_idx)
    log_value(lg, "cost/weighted/B_reward", Float64(decomp.w_B_reward);  step = stride_idx)
    log_value(lg, "cost/weighted/S_reward", Float64(decomp.w_S_reward);  step = stride_idx)
    log_value(lg, "cost/weighted/barrier",  Float64(decomp.w_barrier);   step = stride_idx)
    log_value(lg, "cost/weighted/chance",   Float64(decomp.w_chance);    step = stride_idx)
    log_value(lg, "cost/weighted/chance_B", Float64(decomp.w_chance_B);  step = stride_idx)
    log_value(lg, "cost/weighted/chance_S", Float64(decomp.w_chance_S);  step = stride_idx)
    log_value(lg, "cost/weighted/island",   Float64(decomp.w_island);    step = stride_idx)

    # Total cost.
    log_value(lg, "cost/total/J", Float64(decomp.J_total); step = stride_idx)

    # Active coefficients (so the chart legend tells the whole story).
    log_value(lg, "cost/coef/lam_phi",       Float64(ctrl_cfg.lam_phi);       step = stride_idx)
    log_value(lg, "cost/coef/lam_a",         Float64(ctrl_cfg.lam_a);         step = stride_idx)
    log_value(lg, "cost/coef/lam_b",         Float64(ctrl_cfg.lam_b);         step = stride_idx)
    log_value(lg, "cost/coef/lam_s",         Float64(ctrl_cfg.lam_s);         step = stride_idx)
    log_value(lg, "cost/coef/lam_F",         Float64(ctrl_cfg.lam_f);         step = stride_idx)
    log_value(lg, "cost/coef/lam_chance",    Float64(ctrl_cfg.lam_chance);    step = stride_idx)
    log_value(lg, "cost/coef/lam_chance_b",  Float64(ctrl_cfg.lam_chance_b);  step = stride_idx)
    log_value(lg, "cost/coef/lam_chance_s",  Float64(ctrl_cfg.lam_chance_s);  step = stride_idx)
    log_value(lg, "cost/coef/lam_island",    Float64(ctrl_cfg.lam_island);    step = stride_idx)

    return nothing
end
