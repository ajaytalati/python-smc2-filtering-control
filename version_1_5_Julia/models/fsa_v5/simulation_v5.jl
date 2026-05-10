# FSA v5 simulation — pure, stateless. Mirrors the layout of
# v1.5's `simulation.jl`, but for the v5 surface:
#
#   - 6D state [B, S, F, A, K_FB, K_FS] instead of 3D
#   - 28 dynamics + diffusion params instead of 14
#   - 22 obs-channel params (HR / Sleep / Stress / Steps / VolumeLoad)
#   - 15-minute bin grid (BINS_PER_DAY = 96) instead of 60-minute
#
# Source of truth: `version_1_5_LEAN/Fsa/V5/Types.lean` (Params,
# ObsParams, TRUTH_PARAMS, TRUTH_PARAMS_V5) and the technical guide
# at `version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex`.
#
# Bit-equivalence with the Lean reference is checked at
# `version_1_5_Julia/diff_test/test_lean_diff_v5.jl`.

module SimulationV5

using StaticArrays
using StableRNGs

export BINS_PER_DAY, DT_BIN_DAYS
export A_TYP, F_TYP, PHI_TYP
export TRUTH_PARAMS_V4, TRUTH_PARAMS_V5
export DEFAULT_OBS_PARAMS_V5
export DEFAULT_INIT, TRAINED_ATHLETE_INIT
export FROZEN_PARAMS_V5
export PARAM_KEYS_V5, OBS_PARAM_KEYS_V5
export sample_obs_v5, params_dict_to_nt


# ── Time grid (mirrors v1.5; default 15 min for v5 per tech guide §6) ──

# Tech guide §6.1: production grid is 15-min bins (BINS_PER_DAY = 96).
# Override via the FSA_STEP_MINUTES env var if a coarser grid is wanted
# (e.g. for the basin-sweep tool). This must happen at module-load
# because `BINS_PER_DAY` is a `const` and the bench drivers read it
# at import time.

const _STEP_MIN     = parse(Int, get(ENV, "FSA_STEP_MINUTES", "15"))
const BINS_PER_DAY  = (60 * 24) ÷ _STEP_MIN
const DT_BIN_DAYS   = 1.0 / BINS_PER_DAY


# ── Operating-point reference constants (mirror Fsa.V5.Types) ─────────
# Tech guide §2.1, lines 206-209. These are baked into the G1
# reparametrisation and are NOT estimated.

const A_TYP   = 0.10   # typical autonomic amplitude
const F_TYP   = 0.20   # typical fatigue
const PHI_TYP = 1.0    # typical stimulus rate


# ── Truth parameters: v4-recovering defaults ──────────────────────────
# Mirrors `Fsa/V5/Types.lean::TRUTH_PARAMS` (lines 116-152).
# Setting `mu_dec_B = mu_dec_S = 0` recovers FSA-v4 exactly — the v5
# Hill-deconditioning terms vanish identically. Used as the v4-back-compat
# baseline.

const TRUTH_PARAMS_V4 = Dict{Symbol, Float64}(
    # Aerobic Fitness B
    :tau_B      => 42.0,
    :kappa_B    => 0.012 * (1.0 + 0.40 * A_TYP),
    :epsilon_AB => 0.40,
    # Strength Adaptation S
    :tau_S      => 60.0,
    :kappa_S    => 0.008 * (1.0 + 0.20 * A_TYP),
    :epsilon_AS => 0.20,
    # Unified Fatigue F
    :tau_F      => 7.0 / (1.0 + 1.00 * A_TYP),
    :lambda_A   => 1.00,
    # Busso Variable-Dose K
    :KFB_0      => 0.030,
    :KFS_0      => 0.050,
    :tau_K      => 21.0,
    :mu_K       => 0.005,
    # Stuart-Landau bifurcation parameter
    :mu_0       => 0.02 + 0.40 * (F_TYP * F_TYP),
    :mu_B       => 0.30,
    :mu_S       => 0.15,
    :mu_F       => 0.10 + 2.0 * F_TYP * 0.40,
    :mu_FF      => 0.40,
    :eta        => 0.20,
    # State-dependent diffusion (frozen in production)
    :sigma_B    => 0.010,
    :sigma_S    => 0.008,
    :sigma_F    => 0.012,
    :sigma_A    => 0.020,
    :sigma_K    => 0.005,
    # v5 Hill deconditioning — silent (mu_dec_* = 0) in v4 mode
    :B_dec      => 0.10,
    :S_dec      => 0.10,
    :mu_dec_B   => 0.0,
    :mu_dec_S   => 0.0,
    :n_dec      => 4.0,
)

# v5 truth: closed-island basin topology activated.
# Mirrors `Fsa/V5/Types.lean::TRUTH_PARAMS_V5` (lines 154-162).

const TRUTH_PARAMS_V5 = let
    p = copy(TRUTH_PARAMS_V4)
    p[:B_dec]    = 0.07
    p[:S_dec]    = 0.07
    p[:mu_dec_B] = 0.10
    p[:mu_dec_S] = 0.10
    p[:n_dec]    = 4.0
    p
end


# ── Default observation-channel parameters (placeholder) ──────────────
# The tech guide cites `\input{sections/appendix_v5_parameters.tex}`
# at the end of §10 but the file is missing from the repo. The values
# below are physiologically reasonable placeholders that match the
# 22-field `ObsParams` schema in `Fsa/V5/Obs.lean` (lines 41-71). For
# diff testing they are arbitrary — the same values get sent to both
# sides. For closed-loop production benches they should be tuned
# against the canonical Python source if/when available.

const DEFAULT_OBS_PARAMS_V5 = Dict{Symbol, Float64}(
    # HR channel — sleep-active
    :HR_base     => 60.0,    # bpm at zero load
    :kappa_B_HR  => 20.0,    # bpm decrease per unit B (fitness lowers HR)
    :alpha_A_HR  => 5.0,     # bpm increase per unit A
    :beta_C_HR   => 2.0,     # bpm circadian amplitude
    :sigma_HR    => 2.0,     # bpm obs noise
    # Sleep Bernoulli
    :k_C         => 4.0,
    :k_A         => 1.0,
    :c_tilde     => 1.0,
    # Stress channel — wake-active
    :S_base      => 50.0,
    :k_F         => 30.0,
    :k_A_S       => 10.0,
    :beta_C_S    => 5.0,
    :sigma_S_obs => 4.0,     # NOT the same as TRUTH_PARAMS[:sigma_S] (latent diffusion)
    # Steps channel — log-Gaussian
    :mu_step0    => 8.5,
    :beta_B_st   => 1.0,
    :beta_F_st   => 0.5,
    :beta_A_st   => 0.5,
    :beta_C_st   => 0.3,
    :sigma_st    => 0.4,
    # VolumeLoad channel — training-session-only
    :beta_S_VL   => 100.0,
    :beta_F_VL   => 50.0,
    :sigma_VL    => 10.0,
)


# ── Initial states ────────────────────────────────────────────────────
# `DEFAULT_INIT` (tech guide §9.10): "deconditioned but otherwise
# healthy" starting point. Forward-sim from this under moderate Φ
# takes weeks to settle.
# `TRAINED_ATHLETE_INIT` (tech guide §8.1): the canonical test-scenario
# starting point used in the §8 smoke tests.

const DEFAULT_INIT = (
    B   = 0.05,
    S   = 0.10,
    F   = 0.30,
    A   = 0.10,
    KFB = 0.030,
    KFS = 0.050,
)

const TRAINED_ATHLETE_INIT = (
    B   = 0.50,
    S   = 0.45,
    F   = 0.20,
    A   = 0.45,
    KFB = 0.06,
    KFS = 0.07,
)


# ── Frozen-parameter set (per tech guide §7.1) ────────────────────────
# 14 frozen total: 6 diffusion + circadian phase, 8 dynamics-side.
# Frozen here means: not estimated by the SMC² filter. The values
# come from TRUTH_PARAMS_V5 above.

const FROZEN_PARAMS_V5 = Dict{Symbol, Float64}(
    # Diffusion-side (6)
    :sigma_B  => TRUTH_PARAMS_V5[:sigma_B],
    :sigma_S  => TRUTH_PARAMS_V5[:sigma_S],
    :sigma_F  => TRUTH_PARAMS_V5[:sigma_F],
    :sigma_A  => TRUTH_PARAMS_V5[:sigma_A],
    :sigma_K  => TRUTH_PARAMS_V5[:sigma_K],
    # Dynamics-side (8): structurally non-id'able + need long-detraining
    :KFB_0    => TRUTH_PARAMS_V5[:KFB_0],
    :KFS_0    => TRUTH_PARAMS_V5[:KFS_0],
    :tau_K    => TRUTH_PARAMS_V5[:tau_K],
    :n_dec    => TRUTH_PARAMS_V5[:n_dec],
    :B_dec    => TRUTH_PARAMS_V5[:B_dec],
    :S_dec    => TRUTH_PARAMS_V5[:S_dec],
    :mu_dec_B => TRUTH_PARAMS_V5[:mu_dec_B],
    :mu_dec_S => TRUTH_PARAMS_V5[:mu_dec_S],
)


# ── Field key lists ───────────────────────────────────────────────────
# These match the JSON field order expected by `Main_v5.lean::getParams`
# and `Main_v5.lean::getObsParams`. Used by the diff test to serialise
# parameter dicts deterministically.

const PARAM_KEYS_V5 = (
    :tau_B, :kappa_B, :epsilon_AB,
    :tau_S, :kappa_S, :epsilon_AS,
    :tau_F, :lambda_A,
    :KFB_0, :KFS_0, :tau_K, :mu_K,
    :mu_0, :mu_B, :mu_S, :mu_F, :mu_FF, :eta,
    :sigma_B, :sigma_S, :sigma_F, :sigma_A, :sigma_K,
    :B_dec, :S_dec, :mu_dec_B, :mu_dec_S, :n_dec,
)

const OBS_PARAM_KEYS_V5 = (
    :HR_base, :kappa_B_HR, :alpha_A_HR, :beta_C_HR, :sigma_HR,
    :k_C, :k_A, :c_tilde,
    :S_base, :k_F, :k_A_S, :beta_C_S, :sigma_S_obs,
    :mu_step0, :beta_B_st, :beta_F_st, :beta_A_st, :beta_C_st, :sigma_st,
    :beta_S_VL, :beta_F_VL, :sigma_VL,
)


# ── Helpers ───────────────────────────────────────────────────────────

"""
    params_dict_to_nt(p::Dict{Symbol,Float64}) -> NamedTuple

Convert a params Dict to a NamedTuple with the canonical key order
from `PARAM_KEYS_V5`. Useful for hot inner loops where NamedTuple
field-access avoids the dict lookup overhead. Generic `eltype` for
ForwardDiff compatibility (per the v1.5 convention).
"""
@inline function params_dict_to_nt(p::Dict{Symbol, T}) where {T}
    NamedTuple{PARAM_KEYS_V5}(getindex.(Ref(p), PARAM_KEYS_V5))
end


# ── Pure obs sampler ──────────────────────────────────────────────────
# Stochastic helper used by the synthetic-data plant. The deterministic
# means are in `obs_v5.jl` and are diff-tested against the Lean reference;
# this function adds Gaussian noise / draws Bernoulli on top. Sleep
# gating (HR sleep-only, Stress + Steps wake-only, VolumeLoad
# session-only) is the caller's responsibility.

"""
    sample_obs_v5(state, C, params, key::UInt64) -> NamedTuple

Sample one observation per channel. Pure: same `(state, C, params,
key)` always returns the same result.

Returns `(obs_HR, obs_sleep, obs_S, obs_steps, obs_VL)`. Caller
applies sleep / wake gating to mask channels not active in the
current bin.
"""
function sample_obs_v5(state::SVector{6, Float64}, C::Float64,
                        params::Dict{Symbol, Float64}, key::UInt64)
    rng = StableRNG(key)
    B, S, F, A, _KFB, _KFS = state
    # HR — sleep-active Gaussian
    mu_HR  = params[:HR_base] - params[:kappa_B_HR] * B +
              params[:alpha_A_HR] * A + params[:beta_C_HR] * C
    obs_HR = mu_HR + params[:sigma_HR] * randn(rng)
    # Sleep — Bernoulli logistic
    z       = params[:k_C] * C + params[:k_A] * A - params[:c_tilde]
    p_sleep = 1.0 / (1.0 + exp(-z))
    obs_sleep = rand(rng) < p_sleep
    # Stress — wake-active Gaussian
    mu_S  = params[:S_base] + params[:k_F] * F -
             params[:k_A_S] * A + params[:beta_C_S] * C
    obs_S = mu_S + params[:sigma_S_obs] * randn(rng)
    # Steps — log-Gaussian (caller exponentiates if needed)
    mu_step = params[:mu_step0] + params[:beta_B_st] * B -
               params[:beta_F_st] * F + params[:beta_A_st] * A +
               params[:beta_C_st] * C
    obs_steps = mu_step + params[:sigma_st] * randn(rng)
    # VolumeLoad — Gaussian, no circadian
    mu_VL  = params[:beta_S_VL] * S - params[:beta_F_VL] * F
    obs_VL = mu_VL + params[:sigma_VL] * randn(rng)
    return (
        obs_HR    = obs_HR,
        obs_sleep = obs_sleep,
        obs_S     = obs_S,
        obs_steps = obs_steps,
        obs_VL    = obs_VL,
    )
end

end # module SimulationV5
