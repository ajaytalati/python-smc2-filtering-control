# bench/bench_loop.jl  (v5)
#
# Closed-loop foldl over strides — copy of v1.5's `bench/bench_loop.jl`
# adapted for v5's 6D state, 5-channel obs (HR/Sleep/Stress/Steps/VL)
# with explicit gates, and bimodal (Φ_B, Φ_S) schedule.
#
# Identical control flow (same foldl, same warmup logic, same
# replan-K cadence, same per-stride log row layout); only the data-shape
# touchpoints have been edited:
#   - obs_history NamedTuple: 5 obs channels + 5 gates + Phi_B + Phi_S + C
#   - traj_history: (n, 6)
#   - plant_state: PlantState6D
#   - xhat: SVector{6, Float64}
#   - plan: TWO Φ vectors (Phi_B_plan, Phi_S_plan) per stride
#
# Public functions:
#   - run_one_stride_v5(acc, stride_idx, ctx) -> new_acc
#   - run_closed_loop_bench_v5(ctx, init_acc) -> final_acc
#
# Dependencies (loaded by the calling bench script before this file is
# `include`d): `Statistics.mean`, `StaticArrays.SVector`, `Printf.@sprintf`,
# the v5 model (`plant_rollout_v5`, `init_plant_state_v5`,
# `TRUTH_PARAMS_V5`, `DEFAULT_OBS_PARAMS_V5`, `FROZEN_PARAMS_V5`,
# `DEFAULT_INIT`, `BINS_PER_DAY`), `posterior_mean_v5` and
# `window_grid_obs_v5` from `bench_glue_v5.jl`, `run_outer_smc`,
# `extract_xhat`, `controller_plan_v5`.


# ── Pure helpers ─────────────────────────────────────────────────────────

"""
    slice_phi_for_stride_v5(plan_B, plan_S, plan_offset, stride_bins)
        -> Vector{NTuple{2, Float64}}

Slice both bimodal-Φ plan arrays into the next stride's bimodal sequence.
Returns a length-`stride_bins` vector of `(Phi_B, Phi_S)` Tuples (the
shape `plant_rollout_v5` consumes).
"""
function slice_phi_for_stride_v5(plan_B::AbstractVector,
                                   plan_S::AbstractVector,
                                   plan_offset::Int,
                                   stride_bins::Int)
    return [(Float64(plan_B[plan_offset + k]),
              Float64(plan_S[plan_offset + k])) for k in 1:stride_bins]
end

# Append rollout per-bin obs / gates / Φ / C arrays into the bench's
# running history NamedTuple. Implements the tech guide §3.2 production
# sleep / wake gating policy:
#
#   HR        — sleep-only:    bin in [22:00, 24:00) ∪ [00:00, 06:00)
#   Stress    — wake-only:     complement of HR
#   Steps     — wake-only:     complement of HR
#   VolumeLoad — one bin/day at the canonical 18:00 training-session bin
#   Sleep     — every bin (the sleep label IS the obs)
#
# Gate values are Float32 0.0 / 1.0 multiplied into the per-channel
# log-likelihood inside `propagate_segment_kernel_v5!`. Bin-within-day
# is computed from the global bin index (= cumulative obs-history length
# at the start of the new chunk + offset), modulo BINS_PER_DAY.
#
# `bins_per_day` defaults to `BINS_PER_DAY` (the module-level constant
# imported from `SimulationV5`); pass explicitly to override (e.g. for
# unit tests with different bin widths).
function accumulate_obs_history_v5(obs_history::NamedTuple, p;
                                     bins_per_day::Int = BINS_PER_DAY)
    n = length(p.obs_HR)
    start_bin = length(obs_history.obs_HR)            # 0-indexed
    hours_per_bin = 24.0 / bins_per_day
    # Tech guide §3.2: 22:00-06:00 sleep window. 18:00 VL session.
    sleep_lo_h = 22.0
    sleep_hi_h = 6.0
    vl_bin_within_day = (18 * bins_per_day) ÷ 24      # 18:00 = bin 72 at 15-min

    gate_HR     = Vector{Float32}(undef, n)
    gate_stress = Vector{Float32}(undef, n)
    gate_steps  = Vector{Float32}(undef, n)
    gate_VL     = Vector{Float32}(undef, n)
    @inbounds for i in 1:n
        b = start_bin + i - 1
        within = b - bins_per_day * (b ÷ bins_per_day)
        h = within * hours_per_bin
        is_sleep = (h >= sleep_lo_h) || (h < sleep_hi_h)
        gate_HR[i]     = is_sleep ? 1.0f0 : 0.0f0
        gate_stress[i] = is_sleep ? 0.0f0 : 1.0f0
        gate_steps[i]  = is_sleep ? 0.0f0 : 1.0f0
        gate_VL[i]     = (within == vl_bin_within_day) ? 1.0f0 : 0.0f0
    end
    gate_sleep = ones(Float32, n)                       # always observed

    return (
        obs_HR    = vcat(obs_history.obs_HR,    p.obs_HR),
        obs_S     = vcat(obs_history.obs_S,     p.obs_S),
        obs_steps = vcat(obs_history.obs_steps, p.obs_steps),
        obs_VL    = vcat(obs_history.obs_VL,    p.obs_VL),
        obs_sleep = vcat(obs_history.obs_sleep, p.obs_sleep),
        gate_HR     = vcat(obs_history.gate_HR,     gate_HR),
        gate_stress = vcat(obs_history.gate_stress, gate_stress),
        gate_steps  = vcat(obs_history.gate_steps,  gate_steps),
        gate_VL     = vcat(obs_history.gate_VL,     gate_VL),
        gate_sleep  = vcat(obs_history.gate_sleep,  gate_sleep),
        Phi_B = vcat(obs_history.Phi_B, p.Phi_B),
        Phi_S = vcat(obs_history.Phi_S, p.Phi_S),
        C     = vcat(obs_history.C,     p.C),
    )
end

accumulate_traj_history_v5(traj_history::AbstractMatrix, p) =
    vcat(traj_history, p.trajectory)

should_run_filter(hist_end::Int, window_bins::Int) = hist_end >= window_bins

should_run_replan(stride_idx::Int, replan_K::Int, is_open_loop::Bool,
                   filter_post_exists::Bool) =
    !is_open_loop && (stride_idx % replan_K == 0) && filter_post_exists

function compose_per_stride_log_row_v5(stride_idx::Int, t_wall_s::Float64,
                                         n_temp_filter::Int, n_temp_ctrl::Int,
                                         phi_B::AbstractVector,
                                         phi_S::AbstractVector,
                                         traj::AbstractMatrix)
    # v5 state index: [B, S, F, A, K_FB, K_FS]. A is column 4.
    A_mean_so_far = isempty(traj) ? NaN : mean(traj[:, 4])
    end_row = (size(traj, 1) == 0) ?
                ntuple(_ -> NaN, 6) :
                ntuple(i -> traj[end, i], 6)
    return (
        stride         = stride_idx,
        t_wall_s       = t_wall_s,
        n_temp_filter  = n_temp_filter,
        n_temp_ctrl    = n_temp_ctrl,
        daily_phi_B    = mean(phi_B),
        daily_phi_S    = mean(phi_S),
        A_mean_so_far  = A_mean_so_far,
        B_end          = end_row[1],
        S_end          = end_row[2],
        F_end          = end_row[3],
        A_end          = end_row[4],
        KFB_end        = end_row[5],
        KFS_end        = end_row[6],
        # Legacy keys kept for compat with downstream tooling that
        # consumed v1.5's `phi_mean` / `n_temp` field names.
        n_temp         = n_temp_filter,
        phi_mean       = (mean(phi_B) + mean(phi_S)) / 2,
    )
end


# ── Single-stride step ──────────────────────────────────────────────────

function run_one_stride_v5(acc, stride_idx::Int, ctx::NamedTuple)
    t_start = time()

    # 1. Slice next stride's bimodal Φ
    @assert acc.plan_offset + ctx.stride_bins <= length(acc.plan_phi_B)
    phi_seq = slice_phi_for_stride_v5(acc.plan_phi_B, acc.plan_phi_S,
                                         acc.plan_offset, ctx.stride_bins)

    # 2. Plant rollout
    p = plant_rollout_v5(acc.plant_state, phi_seq, ctx.full_params,
                           ctx.dt_days,
                           hash((ctx.base_seed, :mpc_plant, stride_idx)))

    # 3. Append obs to bench history
    new_obs  = accumulate_obs_history_v5(acc.obs_history, p)
    new_traj = accumulate_traj_history_v5(acc.traj_history, p)

    # 4. Filter window — the most recent `window_bins` of obs
    hist_end = size(new_traj, 1)
    new_filter_post, n_temp_filter, new_xhat = if !should_run_filter(hist_end, ctx.window_bins)
        # Not enough obs yet — skip filter for this stride.
        xhat = SVector{6, Float64}(p.final_state.state[1], p.final_state.state[2],
                                     p.final_state.state[3], p.final_state.state[4],
                                     p.final_state.state[5], p.final_state.state[6])
        @info @sprintf("[stride %2d/%2d] %.1fs  warming up (hist=%d/%d) Φ̄_B=%.3f Φ̄_S=%.3f x̂=(B=%.3f, S=%.3f, F=%.3f, A=%.3f)",
                       stride_idx, ctx.n_strides, time() - t_start,
                       hist_end, ctx.window_bins,
                       mean(p.Phi_B), mean(p.Phi_S),
                       xhat[1], xhat[2], xhat[3], xhat[4])
        (acc.filter_post, 0, xhat)
    else
        t0 = hist_end - ctx.window_bins
        grid_obs = window_grid_obs_v5(new_obs, new_traj, t0, ctx.window_bins;
                                        init_nt = ctx.init_nt)
        filter_out = run_outer_smc(ctx.target, grid_obs, ctx.n_smc, ctx.filter_cfg,
                                     acc.filter_post, ctx.prior_means, ctx.prior_sigmas,
                                     hash((ctx.base_seed, :filter, stride_idx)))
        xhat = extract_xhat(ctx.target, ctx.n_smc)
        @info @sprintf("[stride %2d/%2d] %.1fs  %d filter levels  Φ̄_B=%.3f Φ̄_S=%.3f x̂=(B=%.3f, S=%.3f, F=%.3f, A=%.3f)",
                       stride_idx, ctx.n_strides, time() - t_start,
                       filter_out.n_temp,
                       mean(p.Phi_B), mean(p.Phi_S),
                       xhat[1], xhat[2], xhat[3], xhat[4])
        (filter_out.U_post, filter_out.n_temp, xhat)
    end

    # 5. Maybe replan (closed-loop only)
    new_plan_B, new_plan_S, new_offset, n_temp_ctrl, ctrl_diag_rows =
        if should_run_replan(stride_idx, ctx.replan_K, ctx.is_open_loop,
                              new_filter_post !== nothing)
            t_plan = time()
            params_post = posterior_mean_v5(new_filter_post)
            ctrl_out = controller_plan_v5(params_post, new_xhat, ctx.H_plan_bins,
                                            max(1, ctx.bins_per_day ÷ 24), ctx.dt_days,
                                            ctx.ctrl_cfg,
                                            hash((ctx.base_seed, :ctrl, stride_idx));
                                            collect_diagnostics = ctx.collect_ctrl_diag)
            @info @sprintf("  [replan @ stride %2d] %.1fs  Φ̄_B_plan=%.3f Φ̄_S_plan=%.3f  ctrl_n_temp=%d",
                           stride_idx, time() - t_plan,
                           mean(ctrl_out.Phi_B_plan),
                           mean(ctrl_out.Phi_S_plan),
                           ctrl_out.n_temp_ctrl)
            (ctrl_out.Phi_B_plan, ctrl_out.Phi_S_plan, 0,
              Int(ctrl_out.n_temp_ctrl), ctrl_out.diagnostics)
        else
            (acc.plan_phi_B, acc.plan_phi_S,
              acc.plan_offset + ctx.stride_bins, 0, NamedTuple[])
        end

    # Snapshot the latest filter posterior into the per-stride store.
    post_entry = should_run_filter(hist_end, ctx.window_bins) ? new_filter_post : nothing
    new_all_filter_posts = vcat(acc.all_filter_posts,
                                  Union{Nothing,Matrix{Float64}}[post_entry])

    new_controller_diagnostics = if isempty(ctrl_diag_rows)
        acc.controller_diagnostics
    else
        tagged = [merge(r, (stride_idx = stride_idx,)) for r in ctrl_diag_rows]
        vcat(acc.controller_diagnostics, tagged)
    end

    t_stride_s = time() - t_start
    @info @sprintf("[stride %2d phase walls (s)] total=%.4f", stride_idx, t_stride_s)
    log_row    = compose_per_stride_log_row_v5(stride_idx, t_stride_s,
                                                 n_temp_filter, n_temp_ctrl,
                                                 p.Phi_B, p.Phi_S, new_traj)

    # Live TensorBoard scalars (no-op when --tensorboard was off).
    # Note: p.final_state is a PlantState6D struct; pass its `.state`
    # SVector to the helper so the signature stays AbstractVector-typed.
    # ctx.base_traj is the constant baseline trajectory (rolled once at
    # bench startup under Φ_B = Φ_S = 1.0); the helper slices it at the
    # end-of-stride bin to mirror the grey baseline curves on the
    # end-of-run state-traces PNG.
    if ctx.tb_logger !== nothing
        log_stride_to_tb!(ctx.tb_logger, stride_idx,
                          p.final_state.state, new_xhat, new_filter_post,
                          p.Phi_B, p.Phi_S,
                          log_row,
                          ctx.base_traj, ctx.stride_bins,
                          ctx.init_phi_B, ctx.init_phi_S)
    end

    return (
        plant_state    = p.final_state,
        plan_phi_B     = new_plan_B,
        plan_phi_S     = new_plan_S,
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

function run_closed_loop_bench_v5(ctx::NamedTuple, init_acc::NamedTuple)
    return foldl((acc, k) -> run_one_stride_v5(acc, k, ctx),
                 1:ctx.n_strides; init = init_acc)
end
