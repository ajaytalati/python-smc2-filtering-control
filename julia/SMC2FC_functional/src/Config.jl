"""
    Config.jl

Hyperparameter `@kwdef` structs for the outer SMC², the inner particle
filter, the warm-start bridge, the HMC kernel, the OT rescue, the rolling
window, and the missing-data simulator.

Mirrors `smc2fc/core/config.py` (Python reference) and the existing port's
`julia/SMC2FC/src/Config.jl`. All structs are immutable.
"""

# `Base.@kwdef` gives keyword constructors with default values, the Julia
# analogue of Python `@dataclass`.

"""
    SMCConfig

Hyperparameters for the outer SMC² loop, the warm-start bridge, the
HMC rejuvenation kernel, the inner particle filter, the OT rescue, and
the AD / sampler choices.

# Fields (selected — full list in the source)
- `n_smc_particles::Int = 256`: outer SMC² particle count.
- `target_ess_frac::Float64 = 0.5`: ESS target fraction (drives tempering λ).
- `num_mcmc_steps::Int = 5`: cold-start HMC moves per tempering level.
- `max_lambda_inc::Float64 = 0.05`: cold-start λ-step clamp.
- `bridge_type::Symbol = :gaussian`: `:gaussian | :mog | :schrodinger_follmer`.
- `n_pf_particles::Int = 400`: inner particle-filter cloud size.
- `bandwidth_scale::Float64 = 1.0`: Liu–West / Silverman bandwidth multiplier.
- `ot_ess_frac::Float64 = 0.05`: ESS threshold below which OT rescue triggers.
- `backend::Symbol = :cpu`: `:cpu | :cuda`.
- `ad_backend::Symbol = :ForwardDiff`: `:ForwardDiff | :Enzyme`.
- `sampler::Symbol = :HMC`: `:HMC | :NUTS | :MALA | :AutoMALA | :ChEES`.

# Notes
- Python defaults are preserved verbatim except the Julia-only `backend`,
    `ad_backend`, and `sampler` fields.
- For `:ChEES`, see the `chees_*` fields below.
"""
Base.@kwdef struct SMCConfig
    # ── Outer SMC ────────────────────────────────────────────────────────────
    n_smc_particles::Int          = 256
    target_ess_frac::Float64      = 0.5
    num_mcmc_steps::Int           = 5
    max_lambda_inc::Float64       = 0.05

    # ── Bridge (warm-start) ──────────────────────────────────────────────────
    num_mcmc_steps_bridge::Int    = 3
    max_lambda_inc_bridge::Float64 = 0.10
    bridge_type::Symbol           = :gaussian
    bridge_mog_components::Int    = 2

    sf_blend::Float64             = 0.5
    sf_entropy_reg::Float64       = 0.0
    sf_q1_mode::Symbol            = :is
    sf_annealed_n_stages::Int     = 3
    sf_annealed_n_mh_steps::Int   = 2
    sf_annealed_proposal_scale::Float64 = 0.4
    sf_use_q0_cov::Bool           = false
    sf_info_aware::Bool           = false
    sf_info_lambda_thresh_quantile::Float64 = 0.5
    sf_info_blend_temperature::Float64      = 1.0

    # ── HMC kernel ───────────────────────────────────────────────────────────
    hmc_step_size::Float64        = 0.025
    hmc_num_leapfrog::Int         = 8

    # ── Inner PF ─────────────────────────────────────────────────────────────
    n_pf_particles::Int           = 400
    bandwidth_scale::Float64      = 1.0

    # ── Optimal-transport rescue ─────────────────────────────────────────────
    ot_ess_frac::Float64          = 0.05
    ot_temperature::Float64       = 5.0
    ot_max_weight::Float64        = 0.01
    ot_rank::Int                  = 5
    ot_n_iter::Int                = 2
    ot_epsilon::Float64           = 0.5

    # ── Backend selector ─────────────────────────────────────────────────────
    backend::Symbol               = :cpu

    # ── AD backend for AdvancedHMC.jl gradients ──────────────────────────────
    ad_backend::Symbol            = :ForwardDiff

    # ── MCMC sampler kind for the per-tempering-level rejuvenation ───────────
    sampler::Symbol               = :HMC

    # ── ChEES adaptation knobs ───────────────────────────────────────────────
    chees_L_candidates::Vector{Int}    = [2, 4, 8, 16, 32]
    chees_calib_n_particles::Int       = 16
    chees_calib_n_steps::Int           = 10
end


"""
    RollingConfig

Hyperparameters for the rolling-window driver.

# Fields
- `window_days::Int = 120`: window length in days.
- `stride_days::Int = 30`: stride between successive windows.
- `dt::Float64 = 1.0`: integration step in hours.
- `n_substeps::Int = 10`: SDE sub-steps per `dt`.
- `max_windows::Union{Int,Nothing} = nothing`: optional cap on number of windows.
"""
Base.@kwdef struct RollingConfig
    window_days::Int      = 120
    stride_days::Int      = 30
    dt::Float64           = 1.0
    n_substeps::Int       = 10
    max_windows::Union{Int,Nothing} = nothing
end


"""
    MissingDataConfig

Hyperparameters for the missing-data / dropout simulator that perturbs
synthetic observation data.

# Fields
- `dropout_rate::Float64 = 0.15`: per-step dropout probability.
- `broken_watch_days::Int = 14`: contiguous-missing block length when the
    "watch broke" event fires.
- `rest_days_per_week::Tuple{Int,Int} = (2, 3)`: range of low-activity
    days per week.
- `active_channels::Vector{Symbol} = []`: channels considered "active".
- `passive_channels::Vector{Symbol} = []`: channels considered "passive".
- `all_obs_channels::Vector{Symbol} = []`: full channel set.
"""
Base.@kwdef struct MissingDataConfig
    dropout_rate::Float64    = 0.15
    broken_watch_days::Int   = 14
    rest_days_per_week::Tuple{Int,Int} = (2, 3)

    active_channels::Vector{Symbol}    = Symbol[]
    passive_channels::Vector{Symbol}   = Symbol[]
    all_obs_channels::Vector{Symbol}   = Symbol[]
end
