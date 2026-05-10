#!/usr/bin/env julia
# v5 observation-channel plot — the 5 obs the filter sees, with the
# per-bin gating mask overlaid and the deterministic mean (from the
# truth parameters + plant trajectory + circadian regressor) drawn on
# top so the user can verify obs ≈ μ + noise channel-by-channel.
#
# This is the "is the synthetic data going into the filter correct?"
# diagnostic. Six panels stacked vertically (time on the x-axis):
#
#   1. HR        — sleep-active Gaussian
#   2. Stress    — wake-active Gaussian
#   3. Steps     — wake-active log-Gaussian (μ shown in log-space)
#   4. VolumeLoad — once-per-day-at-18:00 Gaussian
#   5. Sleep     — Bernoulli (binary)
#   6. Circadian — C(t) regressor used by every channel mean
#
# In each Gaussian panel:
#   - blue dots: obs at bins where the gate is 1.0 (filter sees them).
#   - faint grey dots: obs at gated-off bins (filter discards them; the
#     plant still emits a value but the gate zeroes its contribution).
#   - red dashed line: deterministic mean μ_channel(t) computed from
#     the v5 channel-mean formula (`hr_mean`, `stress_mean`, etc.) at
#     the trajectory state + truth obs params + circadian C.
#
# Reads a v5 `data.jld2` written by `tools_v5/bench/bench_postproc.jl`.
# Required keys: `trajectory_mpc`, `circadian_C`, `obs_HR`, `obs_S`,
# `obs_steps`, `obs_VL`, `obs_sleep`, `gate_HR`, `gate_stress`,
# `gate_steps`, `gate_VL`, `gate_sleep`, `truth_params_dict`,
# `BINS_PER_DAY`, `dt_days`.
#
# Usage:
#   julia --project=. tools_v5/plot_obs_channels_v5.jl <data.jld2> <out.png>

using JLD2
using Plots
using Statistics


function load_run_data(path::AbstractString)
    return JLD2.jldopen(path, "r") do f
        Dict(string(k) => f[k] for k in keys(f))
    end
end


# ── Channel-mean reproductions (read from truth_params_dict) ──────────
# Same arithmetic as `models/fsa_v5/obs_v5.jl`, but inlined here so the
# plotter doesn't need to import the model module. Keys match the
# OBS_PARAM_KEYS_V5 layout in `simulation_v5.jl`.
#
# `params` is a Dict{String, Float64} (the JLD2-saved
# `truth_params_dict`). State is the trajectory row `[B, S, F, A, K_FB, K_FS]`.

@inline _hr_mean(B, A, C, p) =
    p["HR_base"] - p["kappa_B_HR"] * B + p["alpha_A_HR"] * A + p["beta_C_HR"] * C

@inline _sleep_prob(A, C, p) =
    1.0 / (1.0 + exp(-(p["k_C"] * C + p["k_A"] * A - p["c_tilde"])))

@inline _stress_mean(F, A, C, p) =
    p["S_base"] + p["k_F"] * F - p["k_A_S"] * A + p["beta_C_S"] * C

@inline _steps_log_mean(B, F, A, C, p) =
    p["mu_step0"] + p["beta_B_st"] * B - p["beta_F_st"] * F +
     p["beta_A_st"] * A + p["beta_C_st"] * C

@inline _volume_load_mean(S, F, p) =
    p["beta_S_VL"] * S - p["beta_F_VL"] * F


# ── Helper: split obs into (active, inactive) by gate ─────────────────
function _split_by_gate(obs::AbstractVector, gate::AbstractVector,
                         t::AbstractVector)
    active_idx   = findall(==(1.0f0), Float32.(gate))
    inactive_idx = findall(==(0.0f0), Float32.(gate))
    t_active   = t[active_idx];   y_active   = obs[active_idx]
    t_inactive = t[inactive_idx]; y_inactive = obs[inactive_idx]
    return (t_active, y_active, t_inactive, y_inactive)
end


function plot_obs_channels_v5(data::Dict; out_path::AbstractString,
                                title_prefix::AbstractString = "FSA-v5 observation channels")
    traj = data["trajectory_mpc"]                       # (n_bins, 6)
    C    = data["circadian_C"]                          # (n_bins,)
    truth = data["truth_params_dict"]                   # Dict{String, Float64}

    obs_HR    = data["obs_HR"]
    obs_S     = data["obs_S"]
    obs_steps = data["obs_steps"]
    obs_VL    = data["obs_VL"]
    obs_sleep = data["obs_sleep"]
    gate_HR     = data["gate_HR"]
    gate_stress = data["gate_stress"]
    gate_steps  = data["gate_steps"]
    gate_VL     = data["gate_VL"]
    gate_sleep  = data["gate_sleep"]

    BINS_PER_DAY = data["BINS_PER_DAY"]
    dt_days = data["dt_days"]
    n_bins = size(traj, 1)
    t_days = collect(0:n_bins - 1) .* dt_days

    # ── Compute per-bin deterministic channel means from the trajectory
    # + truth params (the "ground truth" the obs sampler centred on).
    mu_HR    = Float64[_hr_mean(traj[i, 1], traj[i, 4], C[i], truth) for i in 1:n_bins]
    mu_S     = Float64[_stress_mean(traj[i, 3], traj[i, 4], C[i], truth) for i in 1:n_bins]
    mu_steps = Float64[_steps_log_mean(traj[i, 1], traj[i, 3], traj[i, 4], C[i], truth) for i in 1:n_bins]
    mu_VL    = Float64[_volume_load_mean(traj[i, 2], traj[i, 3], truth) for i in 1:n_bins]
    p_sleep  = Float64[_sleep_prob(traj[i, 4], C[i], truth) for i in 1:n_bins]

    # ── Palette (match v1.5 conventions) ──────────────────────────────
    blue   = RGB(0x1f/255, 0x77/255, 0xb4/255)
    grey   = RGB(0x7f/255, 0x7f/255, 0x7f/255)
    red    = RGB(0xd6/255, 0x27/255, 0x28/255)
    green  = RGB(0x2c/255, 0xa0/255, 0x2c/255)
    orange = RGB(0xff/255, 0x7f/255, 0x0e/255)

    plt = plot(layout = (6, 1), size = (1300, 1700), dpi = 120,
                fontfamily = "Helvetica", framestyle = :box,
                grid = true, gridalpha = 0.3,
                plot_title = "$title_prefix — T=$(round(t_days[end], digits=1))d, " *
                              "$n_bins bins ($(BINS_PER_DAY)/day)",
                plot_titlefontsize = 10)

    # ── 1. HR (sleep-only Gaussian) ───────────────────────────────────
    let
        t_a, y_a, t_i, y_i = _split_by_gate(obs_HR, gate_HR, t_days)
        scatter!(plt[1], t_i, y_i; markersize = 1.4, markerstrokewidth = 0,
                  color = grey, alpha = 0.35,
                  label = "obs (gated off)")
        scatter!(plt[1], t_a, y_a; markersize = 1.6, markerstrokewidth = 0,
                  color = blue,
                  label = "obs (gate=1, sleep window)")
        plot!(plt[1], t_days, mu_HR; color = red, ls = :dash, lw = 1.0,
                label = "μ_HR (truth params + traj)")
        title!(plt[1], "HR  (sleep-only; $(Int(sum(gate_HR)))/$n_bins active bins)";
                titlefontsize = 11)
        xlabel!(plt[1], "time (days)"; labelfontsize = 8)
        ylabel!(plt[1], "HR  (bpm)"; labelfontsize = 8)
    end

    # ── 2. Stress (wake-only Gaussian) ────────────────────────────────
    let
        t_a, y_a, t_i, y_i = _split_by_gate(obs_S, gate_stress, t_days)
        scatter!(plt[2], t_i, y_i; markersize = 1.4, markerstrokewidth = 0,
                  color = grey, alpha = 0.35, label = "obs (gated off)")
        scatter!(plt[2], t_a, y_a; markersize = 1.6, markerstrokewidth = 0,
                  color = orange, label = "obs (gate=1, wake)")
        plot!(plt[2], t_days, mu_S; color = red, ls = :dash, lw = 1.0,
                label = "μ_Stress")
        title!(plt[2], "Stress  (wake-only; $(Int(sum(gate_stress)))/$n_bins active)";
                titlefontsize = 11)
        xlabel!(plt[2], "time (days)"; labelfontsize = 8)
        ylabel!(plt[2], "Stress score"; labelfontsize = 8)
    end

    # ── 3. Steps (wake-only log-Gaussian; obs in log-space) ────────────
    let
        t_a, y_a, t_i, y_i = _split_by_gate(obs_steps, gate_steps, t_days)
        scatter!(plt[3], t_i, y_i; markersize = 1.4, markerstrokewidth = 0,
                  color = grey, alpha = 0.35, label = "obs (gated off)")
        scatter!(plt[3], t_a, y_a; markersize = 1.6, markerstrokewidth = 0,
                  color = green, label = "obs (gate=1, wake)")
        plot!(plt[3], t_days, mu_steps; color = red, ls = :dash, lw = 1.0,
                label = "μ_log-Steps")
        title!(plt[3], "Steps  (wake-only, log-Gaussian; $(Int(sum(gate_steps)))/$n_bins active)";
                titlefontsize = 11)
        xlabel!(plt[3], "time (days)"; labelfontsize = 8)
        ylabel!(plt[3], "log Steps"; labelfontsize = 8)
    end

    # ── 4. VolumeLoad (once-per-day Gaussian) ─────────────────────────
    let
        t_a, y_a, t_i, y_i = _split_by_gate(obs_VL, gate_VL, t_days)
        scatter!(plt[4], t_i, y_i; markersize = 1.0, markerstrokewidth = 0,
                  color = grey, alpha = 0.20, label = "obs (gated off)")
        scatter!(plt[4], t_a, y_a; markersize = 4.0, markerstrokewidth = 0,
                  color = blue, label = "obs (1/day @ 18:00)")
        plot!(plt[4], t_days, mu_VL; color = red, ls = :dash, lw = 1.0,
                label = "μ_VL")
        title!(plt[4], "VolumeLoad  (1/day at 18:00; $(Int(sum(gate_VL)))/$n_bins active)";
                titlefontsize = 11)
        xlabel!(plt[4], "time (days)"; labelfontsize = 8)
        ylabel!(plt[4], "VL"; labelfontsize = 8)
    end

    # ── 5. Sleep (Bernoulli; every bin) ────────────────────────────────
    let
        plot!(plt[5], t_days, p_sleep; color = red, ls = :dash, lw = 1.0,
                label = "p_sleep (truth)")
        scatter!(plt[5], t_days, obs_sleep; markersize = 1.0,
                  markerstrokewidth = 0, color = blue, alpha = 0.6,
                  label = "obs (Bernoulli draw)")
        title!(plt[5], "Sleep label  (Bernoulli, every bin)"; titlefontsize = 11)
        xlabel!(plt[5], "time (days)"; labelfontsize = 8)
        ylabel!(plt[5], "obs ∈ {0, 1}"; labelfontsize = 8)
        ylims!(plt[5], -0.1, 1.1)
    end

    # ── 6. Circadian C(t) ──────────────────────────────────────────────
    let
        plot!(plt[6], t_days, C; color = orange, lw = 1.2,
                label = "C(t) = cos(2π·t)")
        title!(plt[6], "Circadian regressor C(t)"; titlefontsize = 11)
        xlabel!(plt[6], "time (days)"; labelfontsize = 8)
        ylabel!(plt[6], "C"; labelfontsize = 8)
    end

    savefig(plt, out_path)
    return out_path
end


function main()
    if length(ARGS) < 2
        println("Usage: julia plot_obs_channels_v5.jl <data.jld2> <out.png>")
        exit(1)
    end
    data = load_run_data(ARGS[1])
    plot_obs_channels_v5(data; out_path = ARGS[2])
    @info "wrote: $(ARGS[2])"
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
