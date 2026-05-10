# bench/bench_loop.jl
#
# Closed-loop foldl over strides — extracted from the monolithic bench
# `tools/bench_smc_full_mpc_fsa_gpu.jl` as Phase 2 of the refactor in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
#
# Replaces the 130-line `stride_step` closure with a composition of small
# pure helpers + an explicit `BenchContext` NamedTuple that carries the
# previously-captured outer-scope variables as fields. Behaviour is
# byte-identical to the closure — same `foldl` over `1:n_strides`, same
# accumulator shape, same per-stride log row.
#
# A NamedTuple-shaped context (rather than a typed struct) keeps the
# style consistent with `filter_cfg` / `ctrl_cfg` and avoids the need to
# rewrite type signatures when Phase 3 swaps `target::FSAGPUTarget` for
# a `log_density_fn::Function`.
#
# Public functions:
#   - run_one_stride(acc, stride_idx, ctx) -> new_acc
#   - run_closed_loop_bench(ctx, init_acc) -> final_acc
#
# Internal helpers (local to this file, exported only for testability):
#   - slice_phi_for_stride(plan_phi, plan_offset, stride_bins)
#   - accumulate_obs_history(obs_history, p)
#   - accumulate_traj_history(traj_history, p)
#   - should_run_filter(hist_end, window_bins)
#   - should_run_replan(stride_idx, replan_K, is_open_loop, filter_post_exists)
#   - compose_per_stride_log_row(stride_idx, t_wall_s, n_temp_filter,
#                                  n_temp_ctrl, phi, traj)
#
# Dependencies (all loaded by the calling bench before this file is
# `include`d): `Statistics.mean`, `StaticArrays.SVector`, `Printf.@sprintf`,
# `plant_rollout`, `INIT_STATE`, `DEFAULT_PARAMS`, `params_v15_to_v1_nt`,
# `fill_pinned_nt`, `posterior_mean_v15`, `window_grid_obs`,
# `run_outer_smc`, `extract_xhat`, `controller_plan`.


# ── Pure helpers ─────────────────────────────────────────────────────────

slice_phi_for_stride(plan_phi::AbstractVector, plan_offset::Int, stride_bins::Int) =
    plan_phi[plan_offset + 1 : plan_offset + stride_bins]

function accumulate_obs_history(obs_history::NamedTuple, p)
    return (
        B   = vcat(obs_history.B,   p.obs_B),
        F   = vcat(obs_history.F,   p.obs_F),
        A   = vcat(obs_history.A,   p.obs_A),
        Phi = vcat(obs_history.Phi, p.Phi),
    )
end

accumulate_traj_history(traj_history::AbstractMatrix, p) =
    vcat(traj_history, p.trajectory)

should_run_filter(hist_end::Int, window_bins::Int) = hist_end >= window_bins

should_run_replan(stride_idx::Int, replan_K::Int, is_open_loop::Bool,
                   filter_post_exists::Bool) =
    !is_open_loop && (stride_idx % replan_K == 0) && filter_post_exists

function compose_per_stride_log_row(stride_idx::Int, t_wall_s::Float64,
                                     n_temp_filter::Int, n_temp_ctrl::Int,
                                     phi::AbstractVector,
                                     traj::AbstractMatrix)
    A_mean_so_far = isempty(traj) ? NaN : mean(traj[:, 3])
    bfa_end       = (size(traj, 1) == 0) ?
                       (NaN, NaN, NaN) :
                       (traj[end, 1], traj[end, 2], traj[end, 3])
    return (
        stride         = stride_idx,
        t_wall_s       = t_wall_s,
        n_temp_filter  = n_temp_filter,
        n_temp_ctrl    = n_temp_ctrl,
        daily_phi      = mean(phi),
        A_mean_so_far  = A_mean_so_far,
        B_end          = bfa_end[1],
        F_end          = bfa_end[2],
        A_end          = bfa_end[3],
        # legacy keys kept for compat with the existing
        # daily_phi_plan_per_stride builder downstream.
        n_temp         = n_temp_filter,
        phi_mean       = mean(phi),
    )
end


# ── Single-stride step ──────────────────────────────────────────────────

function run_one_stride(acc, stride_idx::Int, ctx::NamedTuple)
    t_start = time()

    # 1. Slice next stride's Φ
    @assert acc.plan_offset + ctx.stride_bins <= length(acc.plan_phi)
    phi = slice_phi_for_stride(acc.plan_phi, acc.plan_offset, ctx.stride_bins)

    # 2. Plant rollout
    p = plant_rollout(acc.plant_state, phi, DEFAULT_PARAMS, ctx.dt_days,
                       hash((ctx.base_seed, :mpc_plant, stride_idx)))

    # 3. Append obs to bench history
    new_obs  = accumulate_obs_history(acc.obs_history, p)
    new_traj = accumulate_traj_history(acc.traj_history, p)

    # 4. Filter window — start at end-of-history minus window_bins
    #    (the most recent window_bins of obs).
    hist_end = size(new_traj, 1)
    new_filter_post, n_temp_filter, new_xhat = if !should_run_filter(hist_end, ctx.window_bins)
        # Not enough obs yet — skip filter for this stride.
        xhat = SVector{3,Float64}(p.final_state.bfa[1],
                                    p.final_state.bfa[2],
                                    p.final_state.bfa[3])
        @info @sprintf("[stride %2d/%2d] %.1fs  warming up (hist=%d/%d) Φ̄=%.3f x̂=(%.3f, %.3f, %.3f)",
                       stride_idx, ctx.n_strides, time() - t_start,
                       hist_end, ctx.window_bins, mean(phi),
                       xhat[1], xhat[2], xhat[3])
        (acc.filter_post, 0, xhat)
    else
        t0 = hist_end - ctx.window_bins
        grid_obs = window_grid_obs(new_obs, new_traj, t0, ctx.window_bins)
        filter_out = run_outer_smc(ctx.target, grid_obs, ctx.n_smc, ctx.filter_cfg,
                                     acc.filter_post, ctx.prior_means, ctx.prior_sigmas,
                                     hash((ctx.base_seed, :filter, stride_idx)))
        xhat = extract_xhat(ctx.target, ctx.n_smc)
        @info @sprintf("[stride %2d/%2d] %.1fs  %d filter levels  Φ̄=%.3f x̂=(%.3f, %.3f, %.3f)",
                       stride_idx, ctx.n_strides, time() - t_start,
                       filter_out.n_temp, mean(phi),
                       xhat[1], xhat[2], xhat[3])
        (filter_out.U_post, filter_out.n_temp, xhat)
    end

    # 5. Maybe replan (closed-loop only)
    new_plan, new_offset, n_temp_ctrl, ctrl_diag_rows =
        if should_run_replan(stride_idx, ctx.replan_K, ctx.is_open_loop,
                              new_filter_post !== nothing)
            t_plan = time()
            params_post_v15 = posterior_mean_v15(new_filter_post)
            params_post_v1  = params_v15_to_v1_nt(fill_pinned_nt(params_post_v15))
            ctrl_out = controller_plan(params_post_v1, new_xhat, ctx.H_plan_bins,
                                          max(1, ctx.bins_per_day ÷ 24), ctx.dt_days,
                                          ctx.ctrl_cfg,
                                          hash((ctx.base_seed, :ctrl, stride_idx));
                                          collect_diagnostics = ctx.collect_ctrl_diag)
            @info @sprintf("  [replan @ stride %2d] %.1fs  Φ̄_plan=%.3f  shape=[%.2f→%.2f→%.2f]  ctrl_n_temp=%d",
                           stride_idx, time() - t_plan, mean(ctrl_out.Phi_plan),
                           ctrl_out.Phi_plan[1],
                           ctrl_out.Phi_plan[length(ctrl_out.Phi_plan) ÷ 2],
                           ctrl_out.Phi_plan[end],
                           ctrl_out.n_temp_ctrl)
            (ctrl_out.Phi_plan, 0, Int(ctrl_out.n_temp_ctrl), ctrl_out.diagnostics)
        else
            (acc.plan_phi, acc.plan_offset + ctx.stride_bins, 0, NamedTuple[])
        end

    # Snapshot the latest filter posterior into the per-stride store.
    # Stride that ran the filter contributes a Matrix; warmup strides
    # contribute `nothing` (caller handles the mask).
    post_entry = should_run_filter(hist_end, ctx.window_bins) ? new_filter_post : nothing
    new_all_filter_posts = vcat(acc.all_filter_posts,
                                  Union{Nothing,Matrix{Float64}}[post_entry])

    # Tag each controller-diagnostics row with the firing stride
    # so we can group by replan downstream.
    new_controller_diagnostics = if isempty(ctrl_diag_rows)
        acc.controller_diagnostics
    else
        tagged = [merge(r, (stride_idx = stride_idx,)) for r in ctrl_diag_rows]
        vcat(acc.controller_diagnostics, tagged)
    end

    t_stride_s = time() - t_start
    log_row    = compose_per_stride_log_row(stride_idx, t_stride_s,
                                              n_temp_filter, n_temp_ctrl,
                                              phi, new_traj)

    return (
        plant_state    = p.final_state,
        plan_phi       = new_plan,
        plan_offset    = new_offset,
        filter_post    = new_filter_post,
        all_filter_posts = new_all_filter_posts,
        traj_history   = new_traj,
        obs_history    = new_obs,
        controller_diagnostics = new_controller_diagnostics,
        per_stride_log = vcat(acc.per_stride_log, [log_row]),
        last_xhat      = new_xhat,
    )
end


# ── Top-level loop driver ───────────────────────────────────────────────

function run_closed_loop_bench(ctx::NamedTuple, init_acc::NamedTuple)
    return foldl((acc, k) -> run_one_stride(acc, k, ctx),
                 1:ctx.n_strides; init = init_acc)
end
