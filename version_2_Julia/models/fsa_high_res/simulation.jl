# FSA-v2 simulator side — port of `version_2/models/fsa_high_res/simulation.py`.
#
# Provides:
#   - INIT_STATE (B=0.05, F=0.30, A=0.10)
#   - DEFAULT_PARAMS (full 30 estimated + 5 frozen)
#   - circadian C(t)
#   - 4 obs samplers (HR Gaussian sleep-gated, sleep Bernoulli, stress Gaussian
#     wake-gated, log-Gaussian steps wake-gated) — formulas verbatim from Python
#   - simulate_em forward roll-out under a Φ schedule
#
# Read FSA_STEP_MINUTES env var at import time (matches Python).

module Simulation

using Random
using ..PhiBurst: BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS
using ..Dynamics: A_TYP, F_TYP, drift, diffusion_state_dep, em_step_substepped


# ── Frozen constants ─────────────────────────────────────────────────────

const EPS_A_FROZEN = 1.0e-4
const EPS_B_FROZEN = 1.0e-4


# ── Default parameters (Set A v2, G1-reparametrized) ─────────────────────
# Matches `simulation.py:DEFAULT_PARAMS` — verbatim values.

const DEFAULT_PARAMS = Dict{Symbol,Float64}(
    # --- v2 Banister dynamics (G1-reparametrized) ---
    :tau_B     => 42.0,
    :tau_F     => 7.0 / (1.0 + 1.00 * A_TYP),    # 6.3636…
    :kappa_B   => 0.012 * (1.0 + 0.40 * A_TYP),  # 0.01248
    :kappa_F   => 0.030,
    :epsilon_A => 0.40,
    :lambda_A  => 1.00,
    :mu_0      => 0.02 + 0.40 * (F_TYP ^ 2),     # 0.036
    :mu_B      => 0.30,
    :mu_F      => 0.10 + 2.0 * F_TYP * 0.40,     # 0.26
    :mu_FF     => 0.40,
    :eta       => 0.20,

    # --- sqrt-Itô diffusion scales (frozen in estimation, free here) ---
    :sigma_B   => 0.010,
    :sigma_F   => 0.012,
    :sigma_A   => 0.020,

    # --- Circadian (frozen) ---
    :phi       => 0.0,    # morning chronotype

    # --- Ch1: HR (sleep-gated) ---
    :HR_base    => 62.0,
    :kappa_B_HR => 12.0,
    :alpha_A_HR =>  3.0,
    :beta_C_HR  => -2.5,
    :sigma_HR   =>  2.0,

    # --- Ch2: Sleep (Bernoulli) ---
    :k_C        =>  3.0,
    :k_A        =>  2.0,
    :c_tilde    =>  0.5,

    # --- Ch3: Stress (wake-gated) ---
    :S_base     => 30.0,
    :k_F        => 20.0,
    :k_A_S      =>  8.0,
    :beta_C_S   => -4.0,
    :sigma_S    =>  4.0,

    # --- Ch4: Steps (log-Gaussian, wake-gated) ---
    :mu_step0   =>  5.5,
    :beta_B_st  =>  0.8,
    :beta_F_st  =>  0.5,
    :beta_A_st  =>  0.3,
    :beta_C_st  => -0.8,
    :sigma_st   =>  0.5,
)


# ── Initial state (Stage-D: de-trained subject) ──────────────────────────

const INIT_STATE = (B = 0.05, F = 0.30, A = 0.10)


# ── EXOGENOUS defaults (used by control.jl) ──────────────────────────────

const EXOGENOUS = (
    T_total      = 42.0,           # days — canonical chronic time constant
    dt_days      = 1.0 / 96.0,     # 15-min outer step (overridden via FSA_STEP_MINUTES)
    n_substeps   = 4,
    F_max        = 0.40,           # overtraining-fatigue limit
    Phi_max      = 3.0,            # max training-strain rate
    Phi_default  = 1.0,            # canonical Banister default
)


# ── Circadian ────────────────────────────────────────────────────────────

"""
    circadian(t_days, phi=0.0) -> C(t) = cos(2π·t + φ)

Period 1 day. With φ=0 (morning chronotype): peak at midnight, trough at noon.
"""
@inline circadian(t_days::Real, phi::Real = 0.0) = cos(2π * t_days + phi)
circadian(t_days::AbstractVector, phi::Real = 0.0) = cos.(2π .* t_days .+ phi)


# ── Sleep-probability helper (Bernoulli, always observed) ────────────────

@inline function _sleep_prob(A::Real, C::Real, k_C::Real, k_A::Real, c_tilde::Real)
    z = k_C * C + k_A * A - c_tilde
    return 1.0 / (1.0 + exp(-z))
end

_sleep_prob(A::AbstractVector, C::AbstractVector, k_C, k_A, c_tilde) =
    1.0 ./ (1.0 .+ exp.(-(k_C .* C .+ k_A .* A .- c_tilde)))


# ── Observation samplers (4 channels) ────────────────────────────────────
# Each sampler takes (trajectory::Matrix, t_grid::Vector, params::Dict, aux,
# prior_channels::Union{Dict,Nothing}, seed::Integer)
# and returns a NamedTuple with at least `t_idx` and one of
# `obs_value` / `sleep_label`.

"""
    gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed)

Bernoulli sleep label at every bin. Always observed.
"""
function gen_obs_sleep(trajectory::AbstractMatrix, t_grid::AbstractVector,
                       params::Dict, aux, prior_channels, seed::Integer)
    rng = MersenneTwister(seed)
    A = trajectory[:, 3]
    C = circadian(t_grid, get(params, :phi, 0.0))
    p = _sleep_prob(A, C, params[:k_C], params[:k_A], params[:c_tilde])
    labels = Int32.(rand(rng, length(t_grid)) .< p)
    return (
        t_idx       = collect(Int32, 0:length(t_grid)-1),
        sleep_label = labels,
    )
end


"""
    gen_obs_hr(trajectory, t_grid, params, aux, prior_channels, seed)

HR Gaussian, observed during sleep only.
"""
function gen_obs_hr(trajectory::AbstractMatrix, t_grid::AbstractVector,
                    params::Dict, aux, prior_channels, seed::Integer)
    rng = MersenneTwister(seed)
    B = trajectory[:, 1]; A = trajectory[:, 3]
    C = circadian(t_grid, get(params, :phi, 0.0))
    hr_mean = params[:HR_base] .- params[:kappa_B_HR] .* B .+
              params[:alpha_A_HR] .* A .+ params[:beta_C_HR] .* C
    hr_obs = hr_mean .+ params[:sigma_HR] .* randn(rng, length(t_grid))

    if prior_channels !== nothing && haskey(prior_channels, :obs_sleep)
        sleep_label = prior_channels[:obs_sleep].sleep_label
        present = Int32.(sleep_label)
    else
        p = _sleep_prob(A, C, params[:k_C], params[:k_A], params[:c_tilde])
        present = Int32.(p .> 0.5)
    end
    idx_present = findall(==(Int32(1)), present) .- 1   # 0-based for parity
    return (
        t_idx     = Int32.(idx_present),
        obs_value = Float32.(hr_obs[idx_present .+ 1]),
    )
end


"""
    gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed)

Stress Gaussian, wake-gated.
"""
function gen_obs_stress(trajectory::AbstractMatrix, t_grid::AbstractVector,
                        params::Dict, aux, prior_channels, seed::Integer)
    rng = MersenneTwister(seed)
    F = trajectory[:, 2]; A = trajectory[:, 3]
    C = circadian(t_grid, get(params, :phi, 0.0))
    s_mean = params[:S_base] .+ params[:k_F] .* F .-
             params[:k_A_S] .* A .+ params[:beta_C_S] .* C
    s_obs = s_mean .+ params[:sigma_S] .* randn(rng, length(t_grid))

    if prior_channels !== nothing && haskey(prior_channels, :obs_sleep)
        sleep_label = prior_channels[:obs_sleep].sleep_label
        present = Int32.(1 .- sleep_label)
    else
        p = _sleep_prob(A, C, params[:k_C], params[:k_A], params[:c_tilde])
        present = Int32.(p .<= 0.5)
    end
    idx_present = findall(==(Int32(1)), present) .- 1
    return (
        t_idx     = Int32.(idx_present),
        obs_value = Float32.(s_obs[idx_present .+ 1]),
    )
end


"""
    gen_obs_steps(trajectory, t_grid, params, aux, prior_channels, seed)

Step count log-Gaussian, wake-gated. Returns raw step counts (NOT log).
"""
function gen_obs_steps(trajectory::AbstractMatrix, t_grid::AbstractVector,
                       params::Dict, aux, prior_channels, seed::Integer)
    rng = MersenneTwister(seed)
    B = trajectory[:, 1]; F = trajectory[:, 2]; A = trajectory[:, 3]
    C = circadian(t_grid, get(params, :phi, 0.0))
    log_mean = params[:mu_step0] .+ params[:beta_B_st] .* B .-
               params[:beta_F_st] .* F .+ params[:beta_A_st] .* A .+
               params[:beta_C_st] .* C
    log_obs = log_mean .+ params[:sigma_st] .* randn(rng, length(t_grid))
    step_count = max.(exp.(log_obs) .- 1.0, 0.0)

    if prior_channels !== nothing && haskey(prior_channels, :obs_sleep)
        sleep_label = prior_channels[:obs_sleep].sleep_label
        present = Int32.(1 .- sleep_label)
    else
        p = _sleep_prob(A, C, params[:k_C], params[:k_A], params[:c_tilde])
        present = Int32.(p .<= 0.5)
    end
    idx_present = findall(==(Int32(1)), present) .- 1
    return (
        t_idx     = Int32.(idx_present),
        obs_value = Float32.(step_count[idx_present .+ 1]),
    )
end


# ── Forward Euler-Maruyama roll-out ──────────────────────────────────────

"""
    simulate_em(init_state::AbstractVector, params, Phi_arr::AbstractVector,
                dt; n_substeps=4, rng=Random.GLOBAL_RNG) -> Matrix(n_steps, 3)

Forward roll-out under a per-bin Phi schedule. State-dependent sqrt-Itô
diffusion via `Dynamics.em_step_substepped`. Boundary reflection at
B∈[0,1], F≥0, A≥0.
"""
function simulate_em(init_state::AbstractVector, params,
                     Phi_arr::AbstractVector, dt::Real;
                     n_substeps::Integer = 4,
                     rng = Random.GLOBAL_RNG)
    n_steps = length(Phi_arr)
    traj = Matrix{Float64}(undef, n_steps, 3)
    y = collect(init_state)
    @inbounds for k in 1:n_steps
        noise = randn(rng, 3)
        y = em_step_substepped(y, params, noise, Phi_arr[k], dt;
                                n_substeps=n_substeps)
        traj[k, :] = y
    end
    return traj
end


export EPS_A_FROZEN, EPS_B_FROZEN
export DEFAULT_PARAMS, INIT_STATE, EXOGENOUS
export circadian, simulate_em
export gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps

end # module Simulation
