# models/fsa_v5/bench_glue_v5.jl
#
# FSA v5 model-specific glue used by `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`.
# Sibling of the model's other files (simulation_v5.jl, _dynamics_v5.jl,
# _plant_v5.jl, obs_v5.jl, estimation_v5.jl, cost_v5.jl, schedule_v5.jl,
# gpu_pf_v5.jl, gpu_control_v5.jl). Included by the bench AFTER the model
# has been imported.
#
# Mirrors `models/fsa_high_res/bench_glue.jl` from v1.5 in spirit; the
# two functions have the same role but operate on v5 shapes.
#
# Public functions:
#   - posterior_mean_v5(U_post)
#       Convert a (M, 37) unconstrained-space outer-SMC² posterior cloud
#       to a Dict{Symbol, Float64} of constrained-space posterior means
#       merged with the 13 frozen v5 entries. The result is the full
#       50-key (28 dynamics + 22 obs-channel) Dict the controller GPU
#       target consumes via `FSAv5ControlGPUTarget(params=...)`.
#   - window_grid_obs_v5(obs_acc, traj_acc, t0_bin, T_window)
#       Slice the bench's accumulated obs / traj history into a single
#       T_window-bin window. Returns a NamedTuple in the shape
#       `gpu_log_density_v5` expects (`Phi_B`, `Phi_S`, 5 obs channels,
#       5 gates, `C`, `init_state`).

using StaticArrays


# ── Posterior → constrained v5 Dict ───────────────────────────────────

"""
    posterior_mean_v5(U_post::AbstractMatrix{Float64}) -> Dict{Symbol, Float64}

Convert the outer-SMC² posterior particle cloud (rows are particles,
columns are parameters in `PARAM_NAMES_V5` order, all 37 LogNormal-
unconstrained) into a Dict of constrained-space posterior means, with
the 13 frozen entries merged in.

The result is the 50-key Dict the v5 controller / plant consume:
- 15 estimated dynamics (PARAM_NAMES_V5[1:15])
- 22 estimated obs-channel (PARAM_NAMES_V5[16:37])
- 13 frozen entries from FROZEN_PARAMS_V5 (5 diffusion + 8 dynamics-side)

Hardcoded for v5's 37 LogNormal priors per `PARAM_PRIOR_CONFIG_V5`;
uses `exp(mean(u_mean))` per dimension.
"""
function posterior_mean_v5(U_post::AbstractMatrix{Float64})
    @assert size(U_post, 2) == 37 "expected 37 estimated params, got $(size(U_post, 2))"
    u_mean = vec(mean(U_post; dims = 1))
    constrained = exp.(u_mean)

    # Build a fresh dict from PARAM_NAMES_V5 order; merge frozen on top.
    out = Dict{Symbol, Float64}()
    for (i, k) in enumerate(PARAM_NAMES_V5)
        out[k] = constrained[i]
    end
    # Frozen entries (5 diffusion + 8 dynamics-side); these don't appear
    # in PARAM_NAMES_V5 since they aren't estimated. Tech guide §7.1.
    for (k, v) in FROZEN_PARAMS_V5
        out[k] = v
    end
    return out
end


# ── Window grid_obs builder ───────────────────────────────────────────

"""
    window_grid_obs_v5(obs_acc, traj_acc, t0_bin, T_window) -> NamedTuple

Slice the bench's accumulated obs / traj history into the `T_window`-bin
rolling window starting at `t0_bin`. Returns the NamedTuple
`gpu_log_density_v5` expects:

    Phi_B, Phi_S          — bimodal stimulus (Float32, length T_window)
    obs_HR, obs_S, obs_steps, obs_VL  — Gaussian channels (Float32)
    obs_sleep             — Bernoulli sleep label as Float32 0.0/1.0
    gate_HR, gate_stress, gate_steps, gate_VL, gate_sleep
                          — per-bin gates as Float32 0.0/1.0
    C                     — circadian regressor (Float32)
    init_state            — NTuple{6, Float64} initial 6D state for
                            the kernel's particle cloud at the start
                            of this window

For `t0_bin == 0` (first post-warmup window) the init state comes from
the `init_nt` keyword argument (defaults to `DEFAULT_INIT` for back-
compat); otherwise it's read from `traj_acc[t0_bin, :]`. The bench passes
the preset selected by `--init-preset` here so the first window's
init_state matches what the plant actually started from.
"""
function window_grid_obs_v5(obs_acc::NamedTuple,
                              traj_acc::AbstractMatrix,
                              t0_bin::Int,
                              T_window::Int;
                              init_nt::NamedTuple = DEFAULT_INIT)
    rng = (t0_bin + 1) : (t0_bin + T_window)

    Phi_B = collect(Float32, view(obs_acc.Phi_B, rng))
    Phi_S = collect(Float32, view(obs_acc.Phi_S, rng))
    obs_HR    = collect(Float32, view(obs_acc.obs_HR,    rng))
    obs_S     = collect(Float32, view(obs_acc.obs_S,     rng))
    obs_steps = collect(Float32, view(obs_acc.obs_steps, rng))
    obs_VL    = collect(Float32, view(obs_acc.obs_VL,    rng))
    obs_sleep = collect(Float32, view(obs_acc.obs_sleep, rng))
    gate_HR     = collect(Float32, view(obs_acc.gate_HR,     rng))
    gate_stress = collect(Float32, view(obs_acc.gate_stress, rng))
    gate_steps  = collect(Float32, view(obs_acc.gate_steps,  rng))
    gate_VL     = collect(Float32, view(obs_acc.gate_VL,     rng))
    gate_sleep  = collect(Float32, view(obs_acc.gate_sleep,  rng))
    C_seq       = collect(Float32, view(obs_acc.C,           rng))

    init_state = if t0_bin == 0
        (Float64(init_nt.B), Float64(init_nt.S),
          Float64(init_nt.F), Float64(init_nt.A),
          Float64(init_nt.KFB), Float64(init_nt.KFS))
    else
        (traj_acc[t0_bin, 1], traj_acc[t0_bin, 2],
          traj_acc[t0_bin, 3], traj_acc[t0_bin, 4],
          traj_acc[t0_bin, 5], traj_acc[t0_bin, 6])
    end

    return (
        Phi_B = Phi_B, Phi_S = Phi_S,
        obs_HR = obs_HR, obs_S = obs_S,
        obs_steps = obs_steps, obs_VL = obs_VL,
        obs_sleep = obs_sleep,
        gate_HR = gate_HR, gate_stress = gate_stress,
        gate_steps = gate_steps, gate_VL = gate_VL,
        gate_sleep = gate_sleep,
        C = C_seq,
        init_state = init_state,
    )
end
