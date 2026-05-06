# Config.jl — translation of `smc2fc/core/config.py` to `@kwdef` structs.
#
# Charter: LaTex_docs/julia_port_charter.pdf §15.2.
# `backend::Symbol = :cpu` is added per the charter's CPU/GPU dispatch note;
# all other fields preserve the Python defaults verbatim.

# `@kwdef` (Base.@kwdef) gives keyword constructors with default values, the
# Julia analogue of Python `@dataclass`.

Base.@kwdef struct SMCConfig
    # ── Outer SMC ────────────────────────────────────────────────────────────
    n_smc_particles::Int          = 256
    target_ess_frac::Float64      = 0.5
    num_mcmc_steps::Int           = 5         # cold-start HMC moves per tempering level
    max_lambda_inc::Float64       = 0.05      # cold-start λ clamp

    # ── Bridge (warm-start) ──────────────────────────────────────────────────
    num_mcmc_steps_bridge::Int    = 3
    max_lambda_inc_bridge::Float64 = 0.10
    bridge_type::Symbol           = :gaussian
    # :gaussian            — single Gaussian + Liu–West shrinkage
    # :mog                 — 2- or 3-component Gaussian mixture
    # :schrodinger_follmer — Bures–Wasserstein geodesic, see SMC2/Bridge.jl
    bridge_mog_components::Int    = 2

    sf_blend::Float64             = 0.5
    sf_entropy_reg::Float64       = 0.0
    sf_q1_mode::Symbol            = :is        # :is | :annealed
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

    # ── Backend selector (Julia-only) ────────────────────────────────────────
    # :cpu  — Array         (CPU thread-parallel via Polyester / Threads)
    # :cuda — CUDA.CuArray  (GPU via CUDA.jl + KernelAbstractions.jl)
    backend::Symbol               = :cpu

    # ── AD backend for AdvancedHMC.jl gradients (Phase 6 follow-up) ─────────
    # :ForwardDiff — robust default; cost = (1 + d_θ) PF evals per gradient
    #               where d_θ is the active partial count. Good for d ≲ 8.
    # :Enzyme      — reverse-mode; cost = ~1 PF eval per gradient (matches
    #               JAX/Python). Charter §13's named production backend.
    #               Requires the inner PF to be Enzyme-mutation-clean.
    ad_backend::Symbol            = :ForwardDiff

    # ── MCMC sampler kind for the per-tempering-level rejuvenation ─────────
    # :HMC  — fixed leapfrog count = `hmc_num_leapfrog` (the v1 default).
    # :NUTS — No-U-Turn Sampler with adaptive trajectory length. Each
    #         transition picks its own leapfrog count by detecting when
    #         the trajectory makes a U-turn; per-call cost varies but
    #         total mixing per gradient eval is typically much higher
    #         than fixed-step HMC, especially on poorly-conditioned
    #         posteriors. `hmc_num_leapfrog` is unused under :NUTS;
    #         `hmc_step_size` becomes the integrator step size.
    sampler::Symbol               = :HMC
end


Base.@kwdef struct RollingConfig
    window_days::Int      = 120
    stride_days::Int      = 30
    dt::Float64           = 1.0
    n_substeps::Int       = 10
    max_windows::Union{Int,Nothing} = nothing
end


Base.@kwdef struct MissingDataConfig
    dropout_rate::Float64    = 0.15
    broken_watch_days::Int   = 14
    rest_days_per_week::Tuple{Int,Int} = (2, 3)

    # Channel groupings — depend on the observation model
    active_channels::Vector{Symbol}    = Symbol[]
    passive_channels::Vector{Symbol}   = Symbol[]
    all_obs_channels::Vector{Symbol}   = Symbol[]
end
