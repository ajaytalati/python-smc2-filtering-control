# models/fsa_high_res/bench_glue.jl
#
# FSA-v1.5 model-specific glue used by the bench
# `tools/bench_smc_full_mpc_fsa_gpu.jl`. This file is a sibling of the
# model's other files (Simulation.jl, Estimation.jl, gpu_pf.jl,
# gpu_control.jl) and is included by the bench AFTER the model has been
# imported (so DEFAULT_PARAMS, INIT_STATE, PARAM_NAMES are in scope).
#
# Extracted verbatim from the monolithic bench script 2026-05-09 as
# Phase 1b of the refactor described in
# `claude_plans/Refactor_v1_5_bench_into_5_modules_2026-05-09_2003.md`.
#
# Public functions:
#   - posterior_mean_v15(U_post) -> NamedTuple of v1.5 params (constrained
#       space; assumes LogNormal priors so exp(mean of unconstrained)).
#       Hardcoded at PARAM_NAMES order: tau_F, B_inf, F_inf, lambda_A,
#       mu_0, mu_B, mu_F, sigma_B, sigma_F, sigma_A.
#   - window_grid_obs(obs_acc, traj_acc, t0_bin, T_window) -> NamedTuple
#       of (Phi_seq, obs_B, obs_F, obs_A, B_init, F_init, A_init) suitable
#       for the FSA inner-PF gpu_log_density call. Slices the rolling
#       obs window from the bench's accumulated obs/traj history. The
#       channel field names (B, F, A, Phi) are FSA-specific.
#
# Both functions reference globals from the FSAHighRes model module
# (INIT_STATE, etc.). They were inline in the bench prior to this
# extraction; no behavioural change.


# ── Posterior → constrained v1.5 NamedTuple ─────────────────────────────
"""
    posterior_mean_v15(U_post::AbstractMatrix{Float64}) -> NamedTuple

Convert an unconstrained-space outer-SMC² posterior particle cloud (rows
are particles, columns are parameters in PARAM_NAMES order) into a
NamedTuple of constrained-space posterior means. Hardcoded for v1.5's
10 LogNormal-prior parameters; uses `exp(mean(u_mean))` per dimension.
"""
function posterior_mean_v15(U_post::AbstractMatrix{Float64})
    # PARAM_NAMES order, all LogNormal — exp(mean of unconstrained).
    u_mean = vec(mean(U_post; dims = 1))
    return (
        tau_F    = exp(u_mean[1]),
        B_inf    = exp(u_mean[2]),
        F_inf    = exp(u_mean[3]),
        lambda_A  = exp(u_mean[4]),
        mu_0     = exp(u_mean[5]),
        mu_B     = exp(u_mean[6]),
        mu_F     = exp(u_mean[7]),
        sigma_B  = exp(u_mean[8]),
        sigma_F  = exp(u_mean[9]),
        sigma_A  = exp(u_mean[10]),
    )
end


# ── Build a single-window grid_obs from the bench's accumulated history ─
"""
    window_grid_obs(obs_acc, traj_acc, t0_bin, T_window) -> NamedTuple

Slice the bench's accumulated obs/traj history into the `T_window`-bin
rolling window starting at `t0_bin`. Returns a NamedTuple in the shape
the FSA `gpu_log_density` expects (`Phi_seq`, `obs_B`, `obs_F`, `obs_A`,
`B_init`, `F_init`, `A_init`).

The init-state for `t0_bin == 0` (the first window after warmup) is read
from `INIT_STATE` rather than the trajectory, since the trajectory may
not yet have reached `t0_bin`.
"""
function window_grid_obs(obs_acc, traj_acc, t0_bin::Int, T_window::Int)
    obs_B = view(obs_acc.B, t0_bin + 1 : t0_bin + T_window)
    obs_F = view(obs_acc.F, t0_bin + 1 : t0_bin + T_window)
    obs_A = view(obs_acc.A, t0_bin + 1 : t0_bin + T_window)
    init  = t0_bin == 0 ?
            (Float64(INIT_STATE.B), Float64(INIT_STATE.F), Float64(INIT_STATE.A)) :
            (traj_acc[t0_bin, 1], traj_acc[t0_bin, 2], traj_acc[t0_bin, 3])
    Phi_seq = view(obs_acc.Phi, t0_bin + 1 : t0_bin + T_window)
    return (
        Phi_seq = collect(Float32, Phi_seq),
        obs_B   = collect(Float32, obs_B),
        obs_F   = collect(Float32, obs_F),
        obs_A   = collect(Float32, obs_A),
        B_init  = init[1], F_init = init[2], A_init = init[3],
    )
end
