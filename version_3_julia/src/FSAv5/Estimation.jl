# FSAv5/Estimation.jl — wires the FSA-v5 model into SMC2FC's
# EstimationModel contract.
#
# Maps to:
#   • Python : `models/fsa_high_res/estimation.py:55-630`
#              (HIGH_RES_FSA_V5_ESTIMATION + propagate_fn_v5 + obs_log_weight_fn)
#   • Julia framework: `julia/SMC2FC/src/EstimationModel.jl` (the typed contract)
#
# Design choice: the propagate_fn here is the **bootstrap-PF** form
# (drift + state-dependent diffusion noise, no Kalman fusion). This
# mirrors the canonical bootstrap PF in the Julia framework
# (`Bootstrap.jl`) and matches the LEAN4 reference's `em_step` plus
# noise. The Python production has a Kalman-fused variant (deferred
# follow-up #1).

using SMC2FC: EstimationModel, PriorType, LogNormalPrior, NormalPrior

# ── Frozen constants (mirror estimation.py:55-100) ─────────────────────────

# State-clipping epsilons (Bug-resistance: never read from params dict)
const EPS_A_FROZEN = 1.0e-4
const EPS_B_FROZEN = 1.0e-4
const EPS_S_FROZEN = 1.0e-4

# Frozen diffusion scales — the Bug 1 sigma_S dict-key collision is
# structurally impossible in this Julia code (sigma_S lives only on
# DynParams, sigma_S_obs only on ObsParams), but for parity with the
# Python `estimation.py:SIGMA_*_FROZEN` we keep these as named
# constants so the bench drivers can pass them as `sigma_diag`.
const SIGMA_B_FROZEN = 0.010
const SIGMA_S_FROZEN = 0.008    # state-noise scale, NOT the obs sigma
const SIGMA_F_FROZEN = 0.012
const SIGMA_A_FROZEN = 0.020
const SIGMA_K_FROZEN = 0.005
const PHI_PHASE_FROZEN = 0.0    # circadian phase reference

# v5 frozen dynamics keys (closed-island calibration §10.4)
const FROZEN_V5_DYNAMICS = Dict{Symbol,Float64}(
    :KFB_0    => 0.030,
    :KFS_0    => 0.050,
    :tau_K    => 21.0,
    :n_dec    => 4.0,
    :B_dec    => 0.07,
    :S_dec    => 0.07,
    :mu_dec_B => 0.10,
    :mu_dec_S => 0.10,
)

# v4 alias: identical to v5 except mu_dec_* = 0 (Hill silent)
const FROZEN_V4_DYNAMICS = let d = copy(FROZEN_V5_DYNAMICS)
    d[:mu_dec_B] = 0.0
    d[:mu_dec_S] = 0.0
    d
end

# Frozen params dict for SMC2FC EstimationModel — combines diffusion
# scales + frozen dynamics into the single Dict the framework expects.
const FROZEN_PARAMS_V5 = let d = Dict{Symbol,Float64}(
        :sigma_B => SIGMA_B_FROZEN,
        :sigma_S => SIGMA_S_FROZEN,
        :sigma_F => SIGMA_F_FROZEN,
        :sigma_A => SIGMA_A_FROZEN,
        :sigma_K => SIGMA_K_FROZEN,
        :phi     => PHI_PHASE_FROZEN,
    )
    merge!(d, FROZEN_V5_DYNAMICS)
    d
end

# ── Prior configuration (mirror estimation.py:113-162) ─────────────────────
# 15 dynamics + 22 obs = 37 estimated parameters. Order is significant —
# defines the layout of θ in the unconstrained vector that HMC/Enzyme
# operate on.

const PARAM_PRIORS_V5 = Tuple{Symbol,PriorType}[
    # ── Dynamics (15) ──
    (:tau_B,       LogNormalPrior(log(42.0),    0.10)),
    (:kappa_B,     LogNormalPrior(log(0.01248), 0.20)),
    (:epsilon_AB,  LogNormalPrior(log(0.40),    0.05)),
    (:tau_S,       LogNormalPrior(log(60.0),    0.10)),
    (:kappa_S,     LogNormalPrior(log(0.00816), 0.20)),
    (:epsilon_AS,  LogNormalPrior(log(0.20),    0.05)),
    (:tau_F,       LogNormalPrior(log(6.3636),  0.15)),
    (:lambda_A,    LogNormalPrior(log(1.00),    0.05)),
    (:mu_K,        LogNormalPrior(log(0.005),   0.20)),
    (:mu_0,        LogNormalPrior(log(0.036),   0.20)),
    (:mu_B,        LogNormalPrior(log(0.30),    0.20)),
    (:mu_S,        LogNormalPrior(log(0.15),    0.20)),
    (:mu_F,        LogNormalPrior(log(0.26),    0.20)),
    (:mu_FF,       LogNormalPrior(log(0.40),    0.05)),
    (:eta,         LogNormalPrior(log(0.20),    0.15)),
    # ── HR observation (5) ──
    (:HR_base,     NormalPrior(62.0, 2.0)),
    (:kappa_B_HR,  LogNormalPrior(log(12.0),    0.15)),
    (:alpha_A_HR,  LogNormalPrior(log(3.0),     0.20)),
    (:beta_C_HR,   NormalPrior(-2.5, 0.5)),
    (:sigma_HR,    LogNormalPrior(log(2.0),     0.20)),
    # ── Sleep Bernoulli (3) ──
    (:k_C,         LogNormalPrior(log(3.0),     0.15)),
    (:k_A,         LogNormalPrior(log(2.0),     0.25)),
    (:c_tilde,     NormalPrior(0.5, 0.25)),
    # ── Stress observation (5) — note `sigma_S_obs`, NOT `sigma_S`
    (:S_base,      NormalPrior(30.0, 3.0)),
    (:k_F,         LogNormalPrior(log(20.0),    0.20)),
    (:k_A_S,       LogNormalPrior(log(8.0),     0.25)),
    (:beta_C_S,    NormalPrior(-4.0, 0.8)),
    (:sigma_S_obs, LogNormalPrior(log(4.0),     0.20)),
    # ── Steps observation (6) ──
    (:mu_step0,    NormalPrior(5.5, 0.3)),
    (:beta_B_st,   LogNormalPrior(log(0.8),     0.20)),
    (:beta_F_st,   LogNormalPrior(log(0.5),     0.25)),
    (:beta_A_st,   LogNormalPrior(log(0.3),     0.25)),
    (:beta_C_st,   NormalPrior(-0.8, 0.2)),
    (:sigma_st,    LogNormalPrior(log(0.5),     0.15)),
    # ── Volume Load observation (3) ──
    (:beta_S_VL,   LogNormalPrior(log(100.0),   0.15)),
    (:beta_F_VL,   LogNormalPrior(log(20.0),    0.20)),
    (:sigma_VL,    LogNormalPrior(log(10.0),    0.20)),
]

# Init-state priors — none for the moment (use DEFAULT_INIT or hand-set).
const INIT_STATE_PRIORS_V5 = Tuple{Symbol,PriorType}[]

# Index map for fast positional lookup inside propagate_fn
const PK_V5 = first.(PARAM_PRIORS_V5)
const PI_V5 = Dict{Symbol,Int}(s => i for (i, s) in enumerate(PK_V5))

# ── Helper: extract a dynamics param from estimated θ + frozen ────────────

@inline function _full_dyn_param(theta::AbstractVector, sym::Symbol)
    if haskey(PI_V5, sym)
        return theta[PI_V5[sym]]
    elseif haskey(FROZEN_PARAMS_V5, sym)
        return FROZEN_PARAMS_V5[sym]
    else
        error("unknown param key: $sym")
    end
end

# ── Build a DynParams ComponentVector from the estimated θ ────────────────
# This is the bridge from SMC2FC's flat θ vector to the FSAv5 dynamics
# functions (which take a structured DynParams).

function _build_dyn_params(theta::AbstractVector{T}) where T<:Real
    return make_dyn_params(
        # Estimated dynamics keys (15)
        tau_B      = T(_full_dyn_param(theta, :tau_B)),
        kappa_B    = T(_full_dyn_param(theta, :kappa_B)),
        epsilon_AB = T(_full_dyn_param(theta, :epsilon_AB)),
        tau_S      = T(_full_dyn_param(theta, :tau_S)),
        kappa_S    = T(_full_dyn_param(theta, :kappa_S)),
        epsilon_AS = T(_full_dyn_param(theta, :epsilon_AS)),
        tau_F      = T(_full_dyn_param(theta, :tau_F)),
        lambda_A   = T(_full_dyn_param(theta, :lambda_A)),
        mu_K       = T(_full_dyn_param(theta, :mu_K)),
        mu_0       = T(_full_dyn_param(theta, :mu_0)),
        mu_B       = T(_full_dyn_param(theta, :mu_B)),
        mu_S       = T(_full_dyn_param(theta, :mu_S)),
        mu_F       = T(_full_dyn_param(theta, :mu_F)),
        mu_FF      = T(_full_dyn_param(theta, :mu_FF)),
        eta        = T(_full_dyn_param(theta, :eta)),
        # Frozen dynamics keys
        KFB_0      = T(FROZEN_V5_DYNAMICS[:KFB_0]),
        KFS_0      = T(FROZEN_V5_DYNAMICS[:KFS_0]),
        tau_K      = T(FROZEN_V5_DYNAMICS[:tau_K]),
        # Frozen diffusion scales (passed through; not consumed by drift)
        sigma_B    = T(SIGMA_B_FROZEN),
        sigma_S    = T(SIGMA_S_FROZEN),
        sigma_F    = T(SIGMA_F_FROZEN),
        sigma_A    = T(SIGMA_A_FROZEN),
        sigma_K    = T(SIGMA_K_FROZEN),
        # Frozen Hill (v5)
        B_dec      = T(FROZEN_V5_DYNAMICS[:B_dec]),
        S_dec      = T(FROZEN_V5_DYNAMICS[:S_dec]),
        mu_dec_B   = T(FROZEN_V5_DYNAMICS[:mu_dec_B]),
        mu_dec_S   = T(FROZEN_V5_DYNAMICS[:mu_dec_S]),
        n_dec      = T(FROZEN_V5_DYNAMICS[:n_dec]),
    )
end

# ── propagate_fn: bootstrap PF (drift + diffusion noise) ──────────────────
# Called by SMC2FC.Bootstrap with one particle's state. Returns
# (next-state, predict-log-weight). For bootstrap PF the predict
# log-weight is 0 (no Kalman correction).

function propagate_fn_v5(y::AbstractVector, t_global::Real, dt::Real,
                          params::AbstractVector, grid_obs::Dict,
                          k::Integer, σ_diag::AbstractVector,
                          noise::AbstractVector, rng)
    p = _build_dyn_params(params)
    Phi_k_arr = grid_obs[:Phi]
    Phi_B = Float64(Phi_k_arr[k, 1])
    Phi_S = Float64(Phi_k_arr[k, 2])
    phi = BimodalPhi(Phi_B, Phi_S)

    y_state = FSAv5State(Float64(y[1]), Float64(y[2]), Float64(y[3]),
                          Float64(y[4]), Float64(y[5]), Float64(y[6]))
    sigma_v = [σ_diag[i] for i in 1:6]
    noise_v = [noise[i] for i in 1:6]

    y_next = em_step(y_state, phi, p, sigma_v, dt, noise_v)
    x_new = [y_next.B, y_next.S, y_next.F, y_next.A, y_next.KFB, y_next.KFS]
    return x_new, 0.0
end

# ── obs_log_weight_fn: Gaussian likelihood across 4 wake/sleep channels ───

function obs_log_weight_fn_v5(x_new::AbstractVector, grid_obs::Dict,
                               k::Integer, params::AbstractVector)
    # Build full obs params from theta indices
    total = 0.0
    HALF_LOG_2PI = 0.5 * log(2.0 * pi)

    # HR channel
    HR_base    = params[PI_V5[:HR_base]]
    kappa_B_HR = params[PI_V5[:kappa_B_HR]]
    alpha_A_HR = params[PI_V5[:alpha_A_HR]]
    beta_C_HR  = params[PI_V5[:beta_C_HR]]
    sigma_HR   = params[PI_V5[:sigma_HR]]
    if haskey(grid_obs, :hr_present) && grid_obs[:hr_present][k] > 0.5
        C_k     = Float64(grid_obs[:C][k])
        hr_mean_k = HR_base - kappa_B_HR * x_new[1] +
                    alpha_A_HR * x_new[4] + beta_C_HR * C_k
        resid   = grid_obs[:hr_value][k] - hr_mean_k
        total  += -0.5 * (resid / sigma_HR)^2 - log(sigma_HR) - HALF_LOG_2PI
    end

    # Stress channel (wake-only is enforced by `stress_present` mask upstream)
    S_base    = params[PI_V5[:S_base]]
    k_F       = params[PI_V5[:k_F]]
    k_A_S     = params[PI_V5[:k_A_S]]
    beta_C_S  = params[PI_V5[:beta_C_S]]
    sigma_S_o = params[PI_V5[:sigma_S_obs]]
    if haskey(grid_obs, :stress_present) && grid_obs[:stress_present][k] > 0.5
        C_k = Float64(grid_obs[:C][k])
        s_mean_k = S_base + k_F * x_new[3] - k_A_S * x_new[4] + beta_C_S * C_k
        resid    = grid_obs[:stress_value][k] - s_mean_k
        total   += -0.5 * (resid / sigma_S_o)^2 - log(sigma_S_o) - HALF_LOG_2PI
    end

    # Steps channel (log-Gaussian)
    mu_step0   = params[PI_V5[:mu_step0]]
    beta_B_st  = params[PI_V5[:beta_B_st]]
    beta_F_st  = params[PI_V5[:beta_F_st]]
    beta_A_st  = params[PI_V5[:beta_A_st]]
    beta_C_st  = params[PI_V5[:beta_C_st]]
    sigma_st   = params[PI_V5[:sigma_st]]
    if haskey(grid_obs, :steps_present) && grid_obs[:steps_present][k] > 0.5
        C_k = Float64(grid_obs[:C][k])
        log_mean_k = mu_step0 + beta_B_st * x_new[1] - beta_F_st * x_new[3] +
                     beta_A_st * x_new[4] + beta_C_st * C_k
        resid      = grid_obs[:log_steps_value][k] - log_mean_k
        total     += -0.5 * (resid / sigma_st)^2 - log(sigma_st) - HALF_LOG_2PI
    end

    # VolumeLoad channel
    beta_S_VL = params[PI_V5[:beta_S_VL]]
    beta_F_VL = params[PI_V5[:beta_F_VL]]
    sigma_VL  = params[PI_V5[:sigma_VL]]
    if haskey(grid_obs, :vl_present) && grid_obs[:vl_present][k] > 0.5
        vl_mean_k = beta_S_VL * x_new[2] - beta_F_VL * x_new[3]
        resid     = grid_obs[:vl_value][k] - vl_mean_k
        total    += -0.5 * (resid / sigma_VL)^2 - log(sigma_VL) - HALF_LOG_2PI
    end

    # Sleep Bernoulli (logistic over A and C)
    k_C       = params[PI_V5[:k_C]]
    k_A_obs   = params[PI_V5[:k_A]]
    c_tilde   = params[PI_V5[:c_tilde]]
    if haskey(grid_obs, :sleep_present) && grid_obs[:sleep_present][k] > 0.5
        C_k = Float64(grid_obs[:C][k])
        z   = k_C * C_k + k_A_obs * x_new[4] - c_tilde
        p   = 1.0 / (1.0 + exp(-z))
        sleep_label = grid_obs[:sleep_value][k]
        total += sleep_label * log(max(p, 1e-12)) +
                 (1 - sleep_label) * log(max(1 - p, 1e-12))
    end

    return total
end

# ── diffusion_fn: returns the 6D σ_diag vector ─────────────────────────────

diffusion_fn_v5(params::AbstractVector) = [
    SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
    SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN,
]

# ── shard_init_fn: returns the initial state used to seed PF particles ────

shard_init_fn_v5(window_start_bin::Integer, params, exogenous,
                  global_init::AbstractVector) =
    Float64[global_init[1], global_init[2], global_init[3],
            global_init[4], global_init[5], global_init[6]]

# ── align_obs_fn: pass-through (caller supplies grid-aligned dict) ────────

align_obs_fn_v5(obs_data, t_steps, dt_hours) = obs_data

# ── Assembled EstimationModel ─────────────────────────────────────────────

"""
    HIGH_RES_FSA_V5_ESTIMATION

Canonical FSA-v5 EstimationModel. Plug into the SMC2FC machinery
(bootstrap_log_likelihood, run_smc_window, …) directly. Mirrors
Python's `HIGH_RES_FSA_V5_ESTIMATION` at `estimation.py:567+`.
"""
const HIGH_RES_FSA_V5_ESTIMATION = EstimationModel(
    name              = "fsa_high_res_v5_julia",
    version           = "0.1.0",
    n_states          = 6,
    n_stochastic      = 6,
    stochastic_indices = [1, 2, 3, 4, 5, 6],
    state_bounds       = [(0.0, 1.0), (0.0, 1.0), (0.0, 10.0),
                           (0.0, 5.0), (0.0, 1.0), (0.0, 1.0)],
    param_priors       = PARAM_PRIORS_V5,
    init_state_priors  = INIT_STATE_PRIORS_V5,
    frozen_params      = FROZEN_PARAMS_V5,
    propagate_fn       = propagate_fn_v5,
    diffusion_fn       = diffusion_fn_v5,
    obs_log_weight_fn  = obs_log_weight_fn_v5,
    align_obs_fn       = align_obs_fn_v5,
    shard_init_fn      = shard_init_fn_v5,
    exogenous_keys     = Symbol[:Phi, :C],
)

export HIGH_RES_FSA_V5_ESTIMATION
export FROZEN_V5_DYNAMICS, FROZEN_V4_DYNAMICS, FROZEN_PARAMS_V5
export PARAM_PRIORS_V5, INIT_STATE_PRIORS_V5
export PI_V5, PK_V5
export EPS_A_FROZEN, EPS_B_FROZEN, EPS_S_FROZEN
export SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN, SIGMA_K_FROZEN
export propagate_fn_v5, obs_log_weight_fn_v5, diffusion_fn_v5
