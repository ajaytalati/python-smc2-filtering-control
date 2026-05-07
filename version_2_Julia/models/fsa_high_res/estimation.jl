# FSA-v2 EstimationModel — port of `version_2/models/fsa_high_res/estimation.py`.
#
# Provides:
#   - PARAM_PRIOR_CONFIG (30 estimated + 5 frozen, matching Python order)
#   - propagate_fn — locally-guided sequential-scalar Kalman fusion across
#     the 3 Gaussian channels (HR / stress / log_steps), Cholesky-sample,
#     return predictive log-marginal (the Pitt-Shephard-style guided proposal)
#   - obs_log_weight_fn — Bernoulli sleep + Gaussian residual correction
#   - align_obs_fn — bin-aligned Float32 obs grid
#   - build_estimation_model() → SMC2FC.EstimationModel

module Estimation

using LinearAlgebra
using Random
using LogExpFunctions: log1pexp

using SMC2FC: EstimationModel, LogNormalPrior, NormalPrior, PriorType

using ..Dynamics: A_TYP, F_TYP


# ── Frozen constants ─────────────────────────────────────────────────────

const EPS_A_FROZEN   = 1.0e-4
const EPS_B_FROZEN   = 1.0e-4
const SIGMA_B_FROZEN = 0.010
const SIGMA_F_FROZEN = 0.012
const SIGMA_A_FROZEN = 0.020
const PHI_FROZEN     = 0.0       # circadian phase, morning chronotype
const HALF_LOG_2PI   = 0.5 * log(2π)


# ── Priors (Set A v2, G1-reparametrized) ─────────────────────────────────
# 30 estimated params, ordered EXACTLY as Python so the param-trace plot
# panels line up. Python source: estimation.py:53-98.

const PARAM_NAMES = Symbol[
    # --- v2 Banister dynamics (G1-reparametrized) ---
    :tau_B, :tau_F, :kappa_B, :kappa_F, :epsilon_A, :lambda_A,
    # --- Stuart-Landau bifurcation parameter ---
    :mu_0, :mu_B, :mu_F, :mu_FF, :eta,
    # --- Ch1: HR (sleep-gated, Gaussian) ---
    :HR_base, :kappa_B_HR, :alpha_A_HR, :beta_C_HR, :sigma_HR,
    # --- Ch2: Sleep (Bernoulli) ---
    :k_C, :k_A, :c_tilde,
    # --- Ch3: Stress (wake-gated, Gaussian) ---
    :S_base, :k_F, :k_A_S, :beta_C_S, :sigma_S,
    # --- Ch4: Steps (log-Gaussian, wake-gated) ---
    :mu_step0, :beta_B_st, :beta_F_st, :beta_A_st, :beta_C_st, :sigma_st,
]

const _PI = Dict(name => i for (i, name) in enumerate(PARAM_NAMES))


"""
    PARAM_PRIOR_CONFIG :: Vector{Tuple{Symbol, PriorType}}

Mirrors `estimation.py:PARAM_PRIOR_CONFIG`. Lognormal priors on rate-like
params, Normal on additive offsets / coefficients with mixed sign.
"""
const PARAM_PRIOR_CONFIG = Tuple{Symbol,PriorType}[
    (:tau_B,       LogNormalPrior(log(42.0),                 0.10)),
    (:tau_F,       LogNormalPrior(log(7.0 / 1.1),            0.15)),  # τ_F^eff
    (:kappa_B,     LogNormalPrior(log(0.012 * 1.04),         0.20)),  # κ_B^eff
    (:kappa_F,     LogNormalPrior(log(0.030),                0.20)),
    (:epsilon_A,   LogNormalPrior(log(0.40),                 0.05)),  # tightened residual
    (:lambda_A,    LogNormalPrior(log(1.00),                 0.05)),  # tightened residual

    (:mu_0,        LogNormalPrior(log(0.036),                0.20)),  # μ_0^eff
    (:mu_B,        LogNormalPrior(log(0.30),                 0.20)),
    (:mu_F,        LogNormalPrior(log(0.26),                 0.20)),  # μ_F^eff
    (:mu_FF,       LogNormalPrior(log(0.40),                 0.05)),  # tightened
    (:eta,         LogNormalPrior(log(0.20),                 0.15)),

    (:HR_base,     NormalPrior(   62.0,                      2.0 )),
    (:kappa_B_HR,  LogNormalPrior(log(12.0),                 0.15)),
    (:alpha_A_HR,  LogNormalPrior(log(3.0),                  0.20)),
    (:beta_C_HR,   NormalPrior(   -2.5,                      0.5 )),
    (:sigma_HR,    LogNormalPrior(log(2.0),                  0.20)),

    (:k_C,         LogNormalPrior(log(3.0),                  0.15)),
    (:k_A,         LogNormalPrior(log(2.0),                  0.25)),
    (:c_tilde,     NormalPrior(   0.5,                       0.25)),

    (:S_base,      NormalPrior(   30.0,                      3.0 )),
    (:k_F,         LogNormalPrior(log(20.0),                 0.20)),
    (:k_A_S,       LogNormalPrior(log(8.0),                  0.25)),
    (:beta_C_S,    NormalPrior(   -4.0,                      0.8 )),
    (:sigma_S,     LogNormalPrior(log(4.0),                  0.20)),

    (:mu_step0,    NormalPrior(   5.5,                       0.3 )),
    (:beta_B_st,   LogNormalPrior(log(0.8),                  0.20)),
    (:beta_F_st,   LogNormalPrior(log(0.5),                  0.25)),
    (:beta_A_st,   LogNormalPrior(log(0.3),                  0.25)),
    (:beta_C_st,   NormalPrior(   -0.8,                      0.2 )),
    (:sigma_st,    LogNormalPrior(log(0.5),                  0.15)),
]

# init-state priors empty (matches Python INIT_STATE_PRIOR_CONFIG = OrderedDict())
const INIT_STATE_PRIOR_CONFIG = Tuple{Symbol,PriorType}[]

const COLD_START_INIT = Float64[0.05, 0.30, 0.10]


# ── Sleep log-prob helper (Bernoulli, observed always) ───────────────────

"""
    sleep_log_prob(A, C_k, sleep_label, sleep_present, params)

Bernoulli sleep log-likelihood. `params` is positionally indexed.
"""
@inline function sleep_log_prob(A::Real, C_k::Real,
                                 sleep_label::Real, sleep_present::Real,
                                 params::AbstractVector)
    k_C     = params[_PI[:k_C]]
    k_A     = params[_PI[:k_A]]
    c_tilde = params[_PI[:c_tilde]]
    z = k_C * C_k + k_A * A - c_tilde
    p = 1.0 / (1.0 + exp(-z))
    p_safe = clamp(p, 1e-8, 1.0 - 1e-8)
    s = Float64(sleep_label)
    return sleep_present * (s * log(p_safe) + (1.0 - s) * log(1.0 - p_safe))
end


# ── Gaussian-channel log-likelihood (used by obs_log_weight_fn) ──────────

@inline function gaussian_obs_ll(y::AbstractVector, grid_obs::Dict, k::Integer,
                                  params::AbstractVector)
    B, F, A = y[1], y[2], y[3]
    C_k = grid_obs[:C][k]

    HR_base    = params[_PI[:HR_base]]
    kappa_B_HR = params[_PI[:kappa_B_HR]]
    alpha_A_HR = params[_PI[:alpha_A_HR]]
    beta_C_HR  = params[_PI[:beta_C_HR]]
    sigma_HR   = params[_PI[:sigma_HR]]

    S_base     = params[_PI[:S_base]]
    k_F        = params[_PI[:k_F]]
    k_A_S      = params[_PI[:k_A_S]]
    beta_C_S   = params[_PI[:beta_C_S]]
    sigma_S    = params[_PI[:sigma_S]]

    mu_step0   = params[_PI[:mu_step0]]
    beta_B_st  = params[_PI[:beta_B_st]]
    beta_F_st  = params[_PI[:beta_F_st]]
    beta_A_st  = params[_PI[:beta_A_st]]
    beta_C_st  = params[_PI[:beta_C_st]]
    sigma_st   = params[_PI[:sigma_st]]

    pred_HR = HR_base - kappa_B_HR * B + alpha_A_HR * A + beta_C_HR * C_k
    pred_S  = S_base + k_F * F - k_A_S * A + beta_C_S * C_k
    pred_ST = mu_step0 + beta_B_st * B - beta_F_st * F + beta_A_st * A +
              beta_C_st * C_k

    # HR
    resid_HR = grid_obs[:hr_value][k] - pred_HR
    lp = grid_obs[:hr_present][k] *
         (-0.5 * (resid_HR / sigma_HR)^2 - log(sigma_HR) - HALF_LOG_2PI)
    # Stress
    resid_S = grid_obs[:stress_value][k] - pred_S
    lp += grid_obs[:stress_present][k] *
          (-0.5 * (resid_S / sigma_S)^2 - log(sigma_S) - HALF_LOG_2PI)
    # log(steps+1)
    resid_ST = grid_obs[:log_steps_value][k] - pred_ST
    lp += grid_obs[:steps_present][k] *
          (-0.5 * (resid_ST / sigma_st)^2 - log(sigma_st) - HALF_LOG_2PI)

    return lp
end


# ── obs_log_weight_fn ────────────────────────────────────────────────────

"""
    obs_log_weight_fn(x_new, grid_obs, k, params) -> scalar

Total observation log-weight for the particle at x_new at bin k.
Adds Bernoulli sleep log-prob to the Gaussian-channel residual ll.
"""
function obs_log_weight_fn(x_new::AbstractVector, grid_obs::Dict, k::Integer,
                            params::AbstractVector)
    gauss_ll = gaussian_obs_ll(x_new, grid_obs, k, params)
    C_k = grid_obs[:C][k]
    bern_ll = sleep_log_prob(x_new[3], C_k,
                              grid_obs[:sleep_label][k],
                              grid_obs[:sleep_present][k],
                              params)
    return gauss_ll + bern_ll
end

# Same signature as obs_log_prob_fn (Python alias).
obs_log_prob_fn(y, grid_obs, k, params) = obs_log_weight_fn(y, grid_obs, k, params)


# ── propagate_fn — locally-guided sequential-scalar Kalman fusion ────────

"""
    propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key)
        -> (x_new, pred_lw)

Pitt-Shephard-style guided proposal (port of estimation.py:108-257).

1. Compute prior predictive mean from G1 drift + state-dependent variance.
2. Sequentially fuse the three Gaussian channels (HR, stress, log_steps)
   into a posterior Gaussian via scalar Kalman steps gated by `*_present`.
3. Sample x_new from N(μ_fused, P_fused) via Cholesky.
4. Return `pred_lw = log_pred_total - obs_ll(x_new)` so the outer weight
   correctly accounts for the proposal's importance ratio.

Sleep (Bernoulli) is NOT fused here — handled by obs_log_weight_fn.
"""
function propagate_fn(y::AbstractVector, t::Real, dt::Real,
                       params::AbstractVector, grid_obs::Dict, k::Integer,
                       sigma_diag, noise::AbstractVector, rng_key)
    # --- v2 dynamics params (positional) ---
    tau_B     = params[_PI[:tau_B]]
    tau_F     = params[_PI[:tau_F]]
    kappa_B   = params[_PI[:kappa_B]]
    kappa_F   = params[_PI[:kappa_F]]
    epsilon_A = params[_PI[:epsilon_A]]
    lambda_A  = params[_PI[:lambda_A]]
    mu_0      = params[_PI[:mu_0]]
    mu_B_p    = params[_PI[:mu_B]]
    mu_F_p    = params[_PI[:mu_F]]
    mu_FF     = params[_PI[:mu_FF]]
    eta       = params[_PI[:eta]]

    B, F, A = y[1], y[2], y[3]
    Phi_k = grid_obs[:Phi][k]
    C_k   = grid_obs[:C][k]

    # --- v2 G1-reparametrized Euler drift ---
    F_dev   = F - F_TYP
    mu_bif  = mu_0 + mu_B_p * B - mu_F_p * F - mu_FF * F_dev * F_dev
    a_factor_B = (1.0 + epsilon_A * A) / (1.0 + epsilon_A * A_TYP)
    a_factor_F = (1.0 + lambda_A  * A) / (1.0 + lambda_A  * A_TYP)
    drift_B = kappa_B * a_factor_B * Phi_k - B / tau_B
    drift_F = kappa_F * Phi_k - a_factor_F / tau_F * F
    drift_A = mu_bif * A - eta * A * A * A
    B_pred = B + dt * drift_B
    F_pred = F + dt * drift_F
    A_pred = A + dt * drift_A

    # --- Prior predictive covariance (state-dependent process noise) ---
    B_cl = clamp(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_cl = max(F, 0.0)
    A_cl = max(A, 0.0)
    var_B = max(SIGMA_B_FROZEN^2 * B_cl * (1.0 - B_cl) * dt, 1e-12)
    var_F = max(SIGMA_F_FROZEN^2 * F_cl * dt, 1e-12)
    var_A = max(SIGMA_A_FROZEN^2 * (A_cl + EPS_A_FROZEN) * dt, 1e-12)

    mu_prior = [B_pred, F_pred, A_pred]
    P_prior  = Matrix(Diagonal([var_B, var_F, var_A]))

    # --- Observation params ---
    HR_base    = params[_PI[:HR_base]]
    kappa_B_HR = params[_PI[:kappa_B_HR]]
    alpha_A_HR = params[_PI[:alpha_A_HR]]
    beta_C_HR  = params[_PI[:beta_C_HR]]
    sigma_HR   = params[_PI[:sigma_HR]]
    S_base     = params[_PI[:S_base]]
    k_F_p      = params[_PI[:k_F]]
    k_A_S      = params[_PI[:k_A_S]]
    beta_C_S   = params[_PI[:beta_C_S]]
    sigma_S    = params[_PI[:sigma_S]]
    mu_step0   = params[_PI[:mu_step0]]
    beta_B_st  = params[_PI[:beta_B_st]]
    beta_F_st  = params[_PI[:beta_F_st]]
    beta_A_st  = params[_PI[:beta_A_st]]
    beta_C_st  = params[_PI[:beta_C_st]]
    sigma_st   = params[_PI[:sigma_st]]

    # 3×3 obs-rows H — match Python ordering (HR, stress, log_steps).
    H = [
        -kappa_B_HR    0.0          alpha_A_HR ;
         0.0           k_F_p       -k_A_S     ;
         beta_B_st    -beta_F_st    beta_A_st ;
    ]
    bias = [
        HR_base  + beta_C_HR * C_k,
        S_base   + beta_C_S  * C_k,
        mu_step0 + beta_C_st * C_k,
    ]
    R_diag = [sigma_HR^2, sigma_S^2, sigma_st^2]

    obs_vals = [
        grid_obs[:hr_value][k],
        grid_obs[:stress_value][k],
        grid_obs[:log_steps_value][k],
    ]
    obs_pres = [
        grid_obs[:hr_present][k],
        grid_obs[:stress_present][k],
        grid_obs[:steps_present][k],
    ]

    # --- Sequential scalar Kalman fusion (3 channels) ---
    mu_state = copy(mu_prior)
    P = copy(P_prior)
    log_pred_total = 0.0
    @inbounds for ch in 1:3
        h_i = vec(H[ch, :])
        b_i = bias[ch]
        r_i = R_diag[ch]
        y_i = obs_vals[ch]
        pres_i = obs_pres[ch]
        innov = y_i - (dot(h_i, mu_state) + b_i)
        Ph    = P * h_i
        S_i   = dot(h_i, Ph) + r_i
        K_i   = Ph ./ S_i
        ll_i  = -0.5 * log(2π * S_i) - 0.5 * innov^2 / S_i
        mu_state .+= pres_i .* K_i .* innov
        P .-= pres_i .* (K_i * Ph')
        log_pred_total += pres_i * ll_i
    end

    # --- Sample from fused Gaussian posterior via Cholesky ---
    P_safe = P + 1e-10 * I(3)
    L = cholesky(Symmetric(P_safe)).L
    x_raw = mu_state .+ L * noise

    # --- Physical bounds ---
    x_new = [
        clamp(x_raw[1], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN),
        max(x_raw[2], 0.0),
        max(x_raw[3], 0.0),
    ]

    # --- Weight correction: pred_lw = log_pred_total - obs_ll_Gaussian(x_new) ---
    preds_new = H * x_new .+ bias
    resids_new = obs_vals .- preds_new
    obs_ll_new = 0.0
    @inbounds for ch in 1:3
        obs_ll_new += obs_pres[ch] * (
            -0.5 * resids_new[ch]^2 / R_diag[ch] -
             0.5 * log(R_diag[ch]) - HALF_LOG_2PI
        )
    end
    pred_lw = log_pred_total - obs_ll_new

    return x_new, pred_lw
end


# ── diffusion_fn ─────────────────────────────────────────────────────────

diffusion_fn(params) = [SIGMA_B_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN]


# ── align_obs_fn ─────────────────────────────────────────────────────────

"""
    align_obs_fn(obs_data, t_steps, dt) -> Dict{Symbol, Vector{Float32}}

Align the 4 obs channels + Phi exogenous + circadian C(t) to a Float32 grid
of length `t_steps`. `obs_data` follows the per-channel dict layout
(`:obs_HR`, `:obs_sleep`, …) with `:t_idx` and `:obs_value`/`:sleep_label`.

Output keys:
    :hr_value, :hr_present
    :stress_value, :stress_present
    :log_steps_value, :steps_present
    :sleep_label, :sleep_present
    :Phi, :C, :has_any_obs
"""
function align_obs_fn(obs_data, t_steps::Integer, dt::Real)
    T = Int(t_steps)

    function _get(name::Symbol)
        if obs_data isa Dict
            return get(obs_data, name, nothing)
        elseif obs_data isa NamedTuple
            return hasproperty(obs_data, name) ? getproperty(obs_data, name) : nothing
        else
            return nothing
        end
    end

    hr_val  = zeros(Float32, T); hr_pres  = zeros(Float32, T)
    s_val   = zeros(Float32, T); s_pres   = zeros(Float32, T)
    lst_val = zeros(Float32, T); st_pres  = zeros(Float32, T)
    sl_lab  = zeros(Int32, T);   sl_pres  = zeros(Float32, T)
    Phi_val = zeros(Float32, T)
    C_val   = zeros(Float32, T)

    function _populate_value!(arr_val::Vector{Float32}, arr_pres::Vector{Float32},
                              ch, value_key::Symbol, transform = identity)
        ch === nothing && return
        idx = ch isa Dict ? get(ch, :t_idx, nothing) :
              (hasproperty(ch, :t_idx) ? ch.t_idx : nothing)
        idx === nothing && return
        vals = ch isa Dict ? get(ch, value_key, nothing) :
              (hasproperty(ch, value_key) ? getproperty(ch, value_key) : nothing)
        vals === nothing && return
        for (j, gi) in enumerate(idx)
            i1 = Int(gi) + 1
            if 1 <= i1 <= T
                arr_val[i1]  = Float32(transform(vals[j]))
                arr_pres[i1] = 1.0f0
            end
        end
    end

    _populate_value!(hr_val,  hr_pres,  _get(:obs_HR),     :obs_value)
    _populate_value!(s_val,   s_pres,   _get(:obs_stress), :obs_value)
    _populate_value!(lst_val, st_pres,  _get(:obs_steps),  :obs_value, x -> log(x + 1.0))

    sl_ch = _get(:obs_sleep)
    if sl_ch !== nothing
        idx  = sl_ch isa Dict ? get(sl_ch, :t_idx, nothing) :
              (hasproperty(sl_ch, :t_idx) ? sl_ch.t_idx : nothing)
        labs = sl_ch isa Dict ? get(sl_ch, :sleep_label, nothing) :
              (hasproperty(sl_ch, :sleep_label) ? sl_ch.sleep_label : nothing)
        if idx !== nothing && labs !== nothing
            for (j, gi) in enumerate(idx)
                i1 = Int(gi) + 1
                if 1 <= i1 <= T
                    sl_lab[i1]  = Int32(labs[j])
                    sl_pres[i1] = 1.0f0
                end
            end
        end
    end

    p_ch = _get(:Phi)
    if p_ch !== nothing
        raw = p_ch isa Dict ? get(p_ch, :Phi_value, nothing) :
              (hasproperty(p_ch, :Phi_value) ? p_ch.Phi_value : nothing)
        if raw !== nothing
            n = min(length(raw), T)
            Phi_val[1:n] = Float32.(raw[1:n])
        end
    end

    c_ch = _get(:C)
    if c_ch !== nothing
        raw = c_ch isa Dict ? get(c_ch, :C_value, nothing) :
              (hasproperty(c_ch, :C_value) ? c_ch.C_value : nothing)
        if raw !== nothing
            n = min(length(raw), T)
            C_val[1:n] = Float32.(raw[1:n])
        end
    else
        # Fallback (matches Python).
        for i in 1:T
            C_val[i] = Float32(cos(2π * (i - 1) * Float64(dt) + PHI_FROZEN))
        end
    end

    has_any = max.(hr_pres, s_pres, st_pres, sl_pres)

    return Dict{Symbol,Any}(
        :hr_value        => hr_val,
        :hr_present      => hr_pres,
        :stress_value    => s_val,
        :stress_present  => s_pres,
        :log_steps_value => lst_val,
        :steps_present   => st_pres,
        :sleep_label     => sl_lab,
        :sleep_present   => sl_pres,
        :Phi             => Phi_val,
        :C               => C_val,
        :has_any_obs     => has_any,
    )
end


# ── shard_init / forward SDE / get_init_theta helpers ────────────────────

shard_init_fn(time_offset, params, exogenous, global_init) = global_init


"""
    forward_sde_stochastic(init_state, params, exogenous, dt, n_steps; rng=...)

Forward Euler-Maruyama on the v2 G1-reparametrized SDE. Used by the
framework for cold-start path construction. `params` is a positional
vector matching `_PI`.
"""
function forward_sde_stochastic(init_state::AbstractVector,
                                 params::AbstractVector,
                                 exogenous::Dict,
                                 dt::Real, n_steps::Integer;
                                 rng = Random.GLOBAL_RNG)
    tau_B     = params[_PI[:tau_B]]
    tau_F     = params[_PI[:tau_F]]
    kappa_B   = params[_PI[:kappa_B]]
    kappa_F   = params[_PI[:kappa_F]]
    epsilon_A = params[_PI[:epsilon_A]]
    lambda_A  = params[_PI[:lambda_A]]
    mu_0      = params[_PI[:mu_0]]
    mu_B      = params[_PI[:mu_B]]
    mu_F      = params[_PI[:mu_F]]
    mu_FF     = params[_PI[:mu_FF]]
    eta       = params[_PI[:eta]]
    sqrt_dt = sqrt(dt)
    Phi_arr = exogenous[:Phi]

    traj = Matrix{Float64}(undef, n_steps, 3)
    y = collect(init_state)
    @inbounds for i in 1:n_steps
        noise = randn(rng, 3)
        B, F, A = y[1], y[2], y[3]
        F_dev = F - F_TYP
        mu = mu_0 + mu_B * B - mu_F * F - mu_FF * F_dev * F_dev
        a_factor_B = (1.0 + epsilon_A * A) / (1.0 + epsilon_A * A_TYP)
        a_factor_F = (1.0 + lambda_A  * A) / (1.0 + lambda_A  * A_TYP)
        dB = kappa_B * a_factor_B * Phi_arr[i] - B / tau_B
        dF = kappa_F * Phi_arr[i] - a_factor_F / tau_F * F
        dA = mu * A - eta * A^3
        B_cl = clamp(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_cl = max(F, 0.0); A_cl = max(A, 0.0)
        B_new = B + dt*dB + SIGMA_B_FROZEN * sqrt(B_cl*(1-B_cl)) * sqrt_dt * noise[1]
        F_new = F + dt*dF + SIGMA_F_FROZEN * sqrt(F_cl) * sqrt_dt * noise[2]
        A_new = A + dt*dA + SIGMA_A_FROZEN * sqrt(A_cl + EPS_A_FROZEN) * sqrt_dt * noise[3]
        B_new = clamp(B_new, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_new = max(F_new, 0.0); A_new = max(A_new, 0.0)
        y = [B_new, F_new, A_new]
        traj[i, :] = y
    end
    return traj
end


# Mean of LogNormal/Normal in constrained space (used as cold-start θ).
function _prior_mean(p::PriorType)
    if p isa LogNormalPrior
        return exp(p.μ + p.σ^2 / 2)
    elseif p isa NormalPrior
        return p.μ
    else
        return 0.0
    end
end


"""
    get_init_theta() -> Vector{Float32}(n_params)

Cold-start θ: prior-mean for every estimated dimension. Order matches
`PARAM_NAMES` (and Python's `_PK`).
"""
function get_init_theta()
    return Float32[_prior_mean(p) for (_, p) in PARAM_PRIOR_CONFIG]
end


# ── Build EstimationModel ────────────────────────────────────────────────

"""
    build_estimation_model() -> SMC2FC.EstimationModel

Assembles the FSA-v2 EstimationModel for the framework's SMC²/PF + control
machinery. Mirror of `estimation.py:HIGH_RES_FSA_V2_ESTIMATION`.
"""
function build_estimation_model()
    return EstimationModel(
        name              = "fsa_high_res_v2",
        version           = "2.0",
        n_states          = 3,
        n_stochastic      = 3,
        stochastic_indices = [1, 2, 3],
        state_bounds      = [(0.0, 1.0), (0.0, 10.0), (0.0, 5.0)],
        param_priors      = PARAM_PRIOR_CONFIG,
        init_state_priors = INIT_STATE_PRIOR_CONFIG,
        frozen_params     = Dict(:eps_A => EPS_A_FROZEN,
                                  :eps_B => EPS_B_FROZEN,
                                  :sigma_B => SIGMA_B_FROZEN,
                                  :sigma_F => SIGMA_F_FROZEN,
                                  :sigma_A => SIGMA_A_FROZEN,
                                  :phi    => PHI_FROZEN),
        propagate_fn      = propagate_fn,
        diffusion_fn      = diffusion_fn,
        obs_log_weight_fn = obs_log_weight_fn,
        align_obs_fn      = align_obs_fn,
        shard_init_fn     = shard_init_fn,
        forward_sde_fn    = forward_sde_stochastic,
        get_init_theta_fn = get_init_theta,
        obs_log_prob_fn   = obs_log_prob_fn,
        exogenous_keys    = [:Phi],
    )
end


export PARAM_NAMES, _PI, PARAM_PRIOR_CONFIG, INIT_STATE_PRIOR_CONFIG, COLD_START_INIT
export propagate_fn, diffusion_fn, obs_log_weight_fn, obs_log_prob_fn, align_obs_fn
export forward_sde_stochastic, get_init_theta, shard_init_fn
export build_estimation_model

end # module Estimation
