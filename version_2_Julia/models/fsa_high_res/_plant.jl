# StepwisePlant — port of `version_2/models/fsa_high_res/_plant.py`.
#
# Mutable simulator-as-plant for closed-loop MPC. Wraps the FSA-v2
# Euler-Maruyama solver + 4 obs-channel sampling in a stateful struct that
# can be advanced one stride at a time, accepting a freshly-planned daily
# Φ for that stride.
#
# Usage:
#     plant = StepwisePlant(truth_params, init_state, dt, seed)
#     for window in 1:n_windows
#         obs = advance!(plant, stride_bins, Phi_daily_for_stride)
#         posterior = filter_update(obs)
#         Phi_next  = controller_plan(posterior, stride_bins)
#     end
#     finalise(plant, out_dir)   # psim-format artifact

module Plant

using Random
using LinearAlgebra
using Printf
using JSON3
using NPZ

using ..PhiBurst: BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS,
                   expand_daily_phi_to_subdaily
using ..Dynamics: drift, A_TYP, F_TYP
using ..Simulation: DEFAULT_PARAMS, INIT_STATE,
                     EPS_A_FROZEN, EPS_B_FROZEN,
                     gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps,
                     circadian


# ── Plant SDE step (single bin, fp32 inner) ──────────────────────────────
# Mirrors `_plant.py:_plant_em_step` (but per-bin rather than scan).

@inline function _plant_em_step_one(y::Vector{Float64},
                                     Phi_t::Real,
                                     params,
                                     sigma_diag::Vector{Float64},
                                     dt::Real,
                                     noise::Vector{Float64})
    sqrt_dt = sqrt(dt)

    # Drift in fp64 (Python uses f64 here; outer accumulator dtype).
    d_y = drift(y, params, Phi_t)

    # sqrt-Itô diffusion scales (mirror noise_scale_fn)
    B_cl = clamp(y[1], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_cl = max(y[2], 0.0)
    A_cl = max(y[3], 0.0)
    g = (sqrt(B_cl * (1.0 - B_cl)),
         sqrt(F_cl),
         sqrt(A_cl + EPS_A_FROZEN))

    y_new1 = y[1] + dt * d_y[1] + sigma_diag[1] * g[1] * sqrt_dt * noise[1]
    y_new2 = y[2] + dt * d_y[2] + sigma_diag[2] * g[2] * sqrt_dt * noise[2]
    y_new3 = y[3] + dt * d_y[3] + sigma_diag[3] * g[3] * sqrt_dt * noise[3]

    # Boundary clip (mirror Python: B clipped to [1e-4, 1-1e-4])
    return [
        clamp(y_new1, 1e-4, 1.0 - 1e-4),
        max(y_new2, 0.0),
        max(y_new3, 0.0),
    ]
end


# ── StepwisePlant struct ─────────────────────────────────────────────────

"""
    StepwisePlant(; truth_params=DEFAULT_PARAMS, init_state=INIT_STATE,
                    dt=DT_BIN_DAYS, seed_offset=42)

Mutable ground-truth FSA-v2 simulator for closed-loop MPC.

Fields:
- `truth_params::Dict{Symbol,Float64}` (default `Simulation.DEFAULT_PARAMS`)
- `state::Vector{Float64}` — current `[B, F, A]`
- `t_bin::Int` — current global bin index (0 at construction, monotone)
- `seed_offset::Int` — base seed; per-channel uses `(seed_offset + bin + ch_offset)`
- `dt::Float64` — bin width in days (`= 1 / BINS_PER_DAY`)
- `history::Dict` — accumulated traj + obs + Phi + C(t) per advance call
"""
mutable struct StepwisePlant
    truth_params::Dict{Symbol,Float64}
    state::Vector{Float64}
    t_bin::Int
    seed_offset::Int
    dt::Float64
    history::Dict{Symbol,Vector{Any}}
end

# Truth-params NamedTuple view for drift() convenience.
function _params_nt(params::Dict{Symbol,Float64})
    return (
        tau_B    = params[:tau_B],
        tau_F    = params[:tau_F],
        kappa_B  = params[:kappa_B],
        kappa_F  = params[:kappa_F],
        epsilon_A = params[:epsilon_A],
        lambda_A  = params[:lambda_A],
        mu_0  = params[:mu_0],
        mu_B  = params[:mu_B],
        mu_F  = params[:mu_F],
        mu_FF = params[:mu_FF],
        eta   = params[:eta],
        sigma_B = params[:sigma_B],
        sigma_F = params[:sigma_F],
        sigma_A = params[:sigma_A],
    )
end


function StepwisePlant(; truth_params::Dict{Symbol,Float64} = copy(DEFAULT_PARAMS),
                         init_state = INIT_STATE,
                         dt::Real = DT_BIN_DAYS,
                         seed_offset::Integer = 42)
    state = init_state isa AbstractVector ?
            Vector{Float64}(init_state) :
            [Float64(init_state.B), Float64(init_state.F), Float64(init_state.A)]
    history = Dict{Symbol,Vector{Any}}(
        :trajectory      => Any[],
        :obs_HR_t_idx    => Any[], :obs_HR_value     => Any[],
        :obs_sleep_label => Any[],
        :obs_stress_t_idx => Any[], :obs_stress_value => Any[],
        :obs_steps_t_idx  => Any[], :obs_steps_value  => Any[],
        :Phi_value => Any[],
        :C_value   => Any[],
    )
    return StepwisePlant(truth_params, state, 0, seed_offset, dt, history)
end


# ── advance! — one stride forward ────────────────────────────────────────

"""
    advance!(plant::StepwisePlant, stride_bins::Integer,
             Phi_daily::AbstractVector) -> NamedTuple

Advance the plant by `stride_bins` bins, applying the daily Φ array
(expanded into morning-loaded sub-bins) and returning the obs emitted
during this stride.

Returns a NamedTuple with fields:
- `trajectory::Matrix(stride_bins, 3)` — latent (B, F, A) per bin
- `obs_HR :: NamedTuple{(:t_idx, :obs_value)}` (sleep-gated, GLOBAL bin idx)
- `obs_sleep :: NamedTuple{(:t_idx, :sleep_label)}` (always observed)
- `obs_stress :: NamedTuple{(:t_idx, :obs_value)}` (wake-gated)
- `obs_steps :: NamedTuple{(:t_idx, :obs_value)}` (wake-gated)
- `Phi :: NamedTuple{(:t_idx, :Phi_value)}`
- `C   :: NamedTuple{(:t_idx, :C_value)}`
"""
function advance!(plant::StepwisePlant, stride_bins::Integer,
                  Phi_daily::AbstractVector)
    Phi_daily_f = Vector{Float64}(Phi_daily)

    Phi_subdaily_full = expand_daily_phi_to_subdaily(Phi_daily_f)
    if length(Phi_subdaily_full) < stride_bins
        error("Phi_daily of length $(length(Phi_daily_f)) expands to " *
              "$(length(Phi_subdaily_full)) bins but stride_bins=" *
              "$stride_bins requested. Ensure len(Phi_daily) * BINS_PER_DAY " *
              "≥ stride_bins.")
    end
    Phi_subdaily = Float32.(Phi_subdaily_full[1:stride_bins])

    # Global time grid for this stride (in days)
    t_grid_global = ((0:stride_bins-1) .+ plant.t_bin) .* plant.dt

    # ── Forward EM rollout for stride_bins steps ─────────────────────────
    sigma_diag = [plant.truth_params[:sigma_B],
                  plant.truth_params[:sigma_F],
                  plant.truth_params[:sigma_A]]
    params_nt = _params_nt(plant.truth_params)

    rng = MersenneTwister(plant.seed_offset + plant.t_bin)
    traj = Matrix{Float64}(undef, stride_bins, 3)
    y = copy(plant.state)
    @inbounds for k in 1:stride_bins
        noise = randn(rng, 3)
        y = _plant_em_step_one(y, Phi_subdaily[k], params_nt, sigma_diag,
                                plant.dt, noise)
        traj[k, :] = y
    end
    plant.state = copy(y)

    aux = (Phi_subdaily,)

    # Obs samplers — gen_obs_sleep first so HR/stress/steps can use it.
    sleep_ch = gen_obs_sleep(traj, t_grid_global, plant.truth_params,
                              aux, nothing,
                              plant.seed_offset + plant.t_bin + 1)
    prior = Dict(:obs_sleep => sleep_ch)
    hr_ch = gen_obs_hr(traj, t_grid_global, plant.truth_params, aux,
                       prior, plant.seed_offset + plant.t_bin + 2)
    stress_ch = gen_obs_stress(traj, t_grid_global, plant.truth_params, aux,
                                prior, plant.seed_offset + plant.t_bin + 3)
    steps_ch = gen_obs_steps(traj, t_grid_global, plant.truth_params, aux,
                              prior, plant.seed_offset + plant.t_bin + 4)

    # Convert local t_idx within each channel to GLOBAL bin indices.
    function _shift_idx(ch::NamedTuple)
        idx = Int32.(ch.t_idx .+ plant.t_bin)
        if hasproperty(ch, :sleep_label)
            return (t_idx = idx, sleep_label = ch.sleep_label)
        else
            return (t_idx = idx, obs_value = ch.obs_value)
        end
    end

    sleep_ch_g  = _shift_idx(sleep_ch)
    hr_ch_g     = _shift_idx(hr_ch)
    stress_ch_g = _shift_idx(stress_ch)
    steps_ch_g  = _shift_idx(steps_ch)

    # Circadian C(t) on global grid (same dtype as Python: Float32 broadcast).
    phi = Float64(get(plant.truth_params, :phi, 0.0))
    C_val = Float32.(cos.(2π .* t_grid_global .+ phi))

    global_t = Int32.(0:stride_bins-1) .+ Int32(plant.t_bin)

    phi_ch = (t_idx = global_t, Phi_value = Phi_subdaily)
    c_ch   = (t_idx = global_t, C_value   = C_val)

    # Append to history.
    push!(plant.history[:trajectory], copy(traj))
    push!(plant.history[:Phi_value], Phi_subdaily)
    push!(plant.history[:C_value], C_val)
    push!(plant.history[:obs_sleep_label], sleep_ch_g.sleep_label)
    push!(plant.history[:obs_HR_t_idx], hr_ch_g.t_idx)
    push!(plant.history[:obs_HR_value], hr_ch_g.obs_value)
    push!(plant.history[:obs_stress_t_idx], stress_ch_g.t_idx)
    push!(plant.history[:obs_stress_value], stress_ch_g.obs_value)
    push!(plant.history[:obs_steps_t_idx], steps_ch_g.t_idx)
    push!(plant.history[:obs_steps_value], steps_ch_g.obs_value)

    plant.t_bin += stride_bins

    return (
        trajectory = traj,
        obs_HR     = hr_ch_g,
        obs_sleep  = sleep_ch_g,
        obs_stress = stress_ch_g,
        obs_steps  = steps_ch_g,
        Phi        = phi_ch,
        C          = c_ch,
    )
end


# ── finalise — write psim-format artifact ────────────────────────────────

"""
    finalise(plant::StepwisePlant, out_dir; scenario_name="fsa_high_res_v2_closed_loop")

Write a psim-format scenario artifact: `manifest.json` + `trajectory.npz` +
`obs/*.npz` + `exogenous/*.npz`. Format matches Python so the same psim
consistency checks can replay the Julia plant trajectory.
"""
function finalise(plant::StepwisePlant, out_dir::AbstractString;
                  scenario_name::AbstractString = "fsa_high_res_v2_closed_loop")
    out_path = abspath(out_dir)
    mkpath(out_path)

    trajectory = vcat(plant.history[:trajectory]...)
    Phi_value  = vcat(plant.history[:Phi_value]...)
    C_value    = vcat(plant.history[:C_value]...)
    sleep_labels = vcat(plant.history[:obs_sleep_label]...)
    n_bins = size(trajectory, 1)

    NPZ.npzwrite(joinpath(out_path, "trajectory.npz"),
                 Dict("trajectory" => trajectory))

    obs_dir = joinpath(out_path, "obs"); mkpath(obs_dir)
    NPZ.npzwrite(joinpath(obs_dir, "obs_sleep.npz"), Dict(
        "t_idx" => collect(Int32, 0:n_bins-1),
        "sleep_label" => Int32.(sleep_labels),
    ))
    NPZ.npzwrite(joinpath(obs_dir, "obs_HR.npz"), Dict(
        "t_idx" => Int32.(vcat(plant.history[:obs_HR_t_idx]...)),
        "obs_HR_value" => Float32.(vcat(plant.history[:obs_HR_value]...)),
    ))
    NPZ.npzwrite(joinpath(obs_dir, "obs_stress.npz"), Dict(
        "t_idx" => Int32.(vcat(plant.history[:obs_stress_t_idx]...)),
        "obs_stress_value" => Float32.(vcat(plant.history[:obs_stress_value]...)),
    ))
    NPZ.npzwrite(joinpath(obs_dir, "obs_steps.npz"), Dict(
        "t_idx" => Int32.(vcat(plant.history[:obs_steps_t_idx]...)),
        "obs_steps_value" => Float32.(vcat(plant.history[:obs_steps_value]...)),
    ))

    exog_dir = joinpath(out_path, "exogenous"); mkpath(exog_dir)
    NPZ.npzwrite(joinpath(exog_dir, "Phi.npz"), Dict(
        "t_idx" => collect(Int32, 0:n_bins-1),
        "Phi_value" => Float32.(Phi_value),
    ))
    NPZ.npzwrite(joinpath(exog_dir, "C.npz"), Dict(
        "t_idx" => collect(Int32, 0:n_bins-1),
        "C_value" => Float32.(C_value),
    ))

    manifest = Dict(
        "schema_version" => "1.0",
        "model_name"     => "fsa_high_res_v2",
        "model_version"  => "2.0",
        "scenario_name"  => scenario_name,
        "truth_params"   => Dict(string(k) => Float64(v)
                                  for (k, v) in plant.truth_params),
        "init_state"     => Dict("B_0" => Float64(INIT_STATE.B),
                                  "F_0" => Float64(INIT_STATE.F),
                                  "A_0" => Float64(INIT_STATE.A)),
        "n_bins_total"   => Int(n_bins),
        "dt_days"        => Float64(plant.dt),
        "bins_per_day"   => Int(BINS_PER_DAY),
        "seed"           => Int(plant.seed_offset),
        "state_names"    => ["B", "F", "A"],
        "obs_channels"   => ["obs_HR", "obs_sleep", "obs_stress", "obs_steps"],
        "exogenous_channels" => ["Phi", "C"],
        "validation_summary" => Dict(
            "closed_loop" => true,
            "stepwise_advances" => length(plant.history[:trajectory]),
        ),
    )
    open(joinpath(out_path, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    return out_path
end


"""
    advance_subdaily!(plant, Phi_subdaily::AbstractVector{<:Real}) -> NamedTuple

Advance the plant by `length(Phi_subdaily)` bins applying the supplied
per-bin Φ schedule directly (no daily→subdaily expansion). Use this when
the controller produces a per-bin schedule.
"""
function advance_subdaily!(plant::StepwisePlant, Phi_subdaily::AbstractVector{<:Real})
    stride_bins = length(Phi_subdaily)
    Phi_subdaily_f32 = Float32.(Phi_subdaily)

    t_grid_global = ((0:stride_bins-1) .+ plant.t_bin) .* plant.dt

    sigma_diag = [plant.truth_params[:sigma_B],
                  plant.truth_params[:sigma_F],
                  plant.truth_params[:sigma_A]]
    params_nt = _params_nt(plant.truth_params)

    rng = MersenneTwister(plant.seed_offset + plant.t_bin)
    traj = Matrix{Float64}(undef, stride_bins, 3)
    y = copy(plant.state)
    @inbounds for k in 1:stride_bins
        noise = randn(rng, 3)
        y = _plant_em_step_one(y, Phi_subdaily_f32[k], params_nt, sigma_diag,
                                plant.dt, noise)
        traj[k, :] = y
    end
    plant.state = copy(y)
    aux = (Phi_subdaily_f32,)

    sleep_ch = gen_obs_sleep(traj, t_grid_global, plant.truth_params,
                              aux, nothing,
                              plant.seed_offset + plant.t_bin + 1)
    prior = Dict(:obs_sleep => sleep_ch)
    hr_ch = gen_obs_hr(traj, t_grid_global, plant.truth_params, aux,
                       prior, plant.seed_offset + plant.t_bin + 2)
    stress_ch = gen_obs_stress(traj, t_grid_global, plant.truth_params, aux,
                                prior, plant.seed_offset + plant.t_bin + 3)
    steps_ch = gen_obs_steps(traj, t_grid_global, plant.truth_params, aux,
                              prior, plant.seed_offset + plant.t_bin + 4)

    function _shift_idx(ch::NamedTuple)
        idx = Int32.(ch.t_idx .+ plant.t_bin)
        if hasproperty(ch, :sleep_label)
            return (t_idx = idx, sleep_label = ch.sleep_label)
        else
            return (t_idx = idx, obs_value = ch.obs_value)
        end
    end

    sleep_ch_g  = _shift_idx(sleep_ch)
    hr_ch_g     = _shift_idx(hr_ch)
    stress_ch_g = _shift_idx(stress_ch)
    steps_ch_g  = _shift_idx(steps_ch)

    phi = Float64(get(plant.truth_params, :phi, 0.0))
    C_val = Float32.(cos.(2π .* t_grid_global .+ phi))

    global_t = Int32.(0:stride_bins-1) .+ Int32(plant.t_bin)
    phi_ch = (t_idx = global_t, Phi_value = Phi_subdaily_f32)
    c_ch   = (t_idx = global_t, C_value   = C_val)

    push!(plant.history[:trajectory], copy(traj))
    push!(plant.history[:Phi_value], Phi_subdaily_f32)
    push!(plant.history[:C_value], C_val)
    push!(plant.history[:obs_sleep_label], sleep_ch_g.sleep_label)
    push!(plant.history[:obs_HR_t_idx], hr_ch_g.t_idx)
    push!(plant.history[:obs_HR_value], hr_ch_g.obs_value)
    push!(plant.history[:obs_stress_t_idx], stress_ch_g.t_idx)
    push!(plant.history[:obs_stress_value], stress_ch_g.obs_value)
    push!(plant.history[:obs_steps_t_idx], steps_ch_g.t_idx)
    push!(plant.history[:obs_steps_value], steps_ch_g.obs_value)

    plant.t_bin += stride_bins

    return (
        trajectory = traj,
        obs_HR     = hr_ch_g,
        obs_sleep  = sleep_ch_g,
        obs_stress = stress_ch_g,
        obs_steps  = steps_ch_g,
        Phi        = phi_ch,
        C          = c_ch,
    )
end


export StepwisePlant, advance!, advance_subdaily!, finalise

end # module Plant
