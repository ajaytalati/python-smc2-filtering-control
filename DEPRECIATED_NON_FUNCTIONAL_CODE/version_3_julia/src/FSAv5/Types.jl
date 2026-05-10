# FSAv5/Types.jl — typed declarations for the v5 model.
#
# Maps line-by-line to `FSA_model_dev/lean/Fsa/V5/Types.lean`. The
# LEAN4 reference is the formal source of truth for field names and
# physical units; this file is its Julia executable counterpart.
#
# Per the LEAN4-first charter (`lean4_first_charter.pdf` §4.1) and the
# Julia port charter (`julia_port_charter.pdf` §15.2): named fields
# make the historical `sigma_S` dict-key collision a compile error.
# State-noise `sigma_S` lives on `DynParams`; stress-channel obs noise
# `sigma_S_obs` lives on `ObsParams`. They are different fields on
# different ComponentVectors and CANNOT collide.

using StaticArrays: FieldVector
using ComponentArrays: ComponentVector

# ── 6D state vector ─────────────────────────────────────────────────────────
# Maps to `Fsa.V5.State6D` in `Types.lean`.
# `<: FieldVector{6, Float64}` makes it a stack-allocated SVector subtype
# with both named-field access (`y.B`) AND positional / iteration / linear-
# algebra interfaces. Zero-overhead in inner loops; per charter §15.2 a
# regression-prone cliff if we used heap-allocated Vectors instead.

struct FSAv5State <: FieldVector{6, Float64}
    B   :: Float64    # Aerobic fitness, Banister chronic, Jacobi diff in [0, 1]
    S   :: Float64    # Strength capacity, Banister chronic, Jacobi diff in [0, 1]
    F   :: Float64    # Unified fatigue pool, CIR diffusion in [0, ∞)
    A   :: Float64    # Autonomic amplitude, Stuart-Landau, CIR in [0, ∞)
    KFB :: Float64    # Aerobic fatigue gain (Busso variable), CIR in [0, ∞)
    KFS :: Float64    # Strength fatigue gain (Busso variable), CIR in [0, ∞)
end

# ── 2D bimodal stimulus rate Φ = [Φ_B, Φ_S] ─────────────────────────────────
# Maps to `Fsa.V5.BimodalPhi` in `Types.lean`.

struct BimodalPhi <: FieldVector{2, Float64}
    Phi_B :: Float64    # Aerobic stimulus rate (dimensionless)
    Phi_S :: Float64    # Strength stimulus rate (dimensionless)
end

# ── Operating-point reference constants ────────────────────────────────────
# Mirrors `_dynamics.py:81-83` and `Types.lean:A_TYP/F_TYP/PHI_TYP`. These
# are the "typical" values around which the G1 reparametrisation is centred.

const A_TYP   = 0.10    # typical autonomic amplitude (dimensionless)
const F_TYP   = 0.20    # typical unified fatigue (dimensionless)
const PHI_TYP = 1.0     # typical stimulus rate (dimensionless)

# ── Dynamics + diffusion + Hill parameters (28 fields) ─────────────────────
# Maps to `Fsa.V5.Params` in `Types.lean`. The historical Python dict had
# two `'sigma_S'` keys; here `sigma_S` is unambiguously the LATENT state-
# noise scale (default 0.008). The stress-channel obs noise is
# `ObsParams.sigma_S_obs` — a different field on a different record.
#
# `DynParams` is `ComponentVector{Float64}` aliased; AdvancedHMC and
# Enzyme accept it directly as a flat AbstractVector while consumers
# get dot-access (`p.tau_B`, `p.kappa_B`, …). Charter §15.2.
#
# The 28-key NamedTuple is the public ordered layout — every constructor
# uses these keys in this order.

const DYN_PARAM_KEYS = (
    # Aerobic Fitness B (3)
    :tau_B, :kappa_B, :epsilon_AB,
    # Strength Adaptation S (3)
    :tau_S, :kappa_S, :epsilon_AS,
    # Unified Fatigue F (2)
    :tau_F, :lambda_A,
    # Busso Variable-Dose K (4)
    :KFB_0, :KFS_0, :tau_K, :mu_K,
    # Stuart-Landau bifurcation parameter (6)
    :mu_0, :mu_B, :mu_S, :mu_F, :mu_FF, :eta,
    # State-dependent diffusion scales — frozen in production (5)
    :sigma_B, :sigma_S, :sigma_F, :sigma_A, :sigma_K,
    # FSA-v5 Hill deconditioning — silent when mu_dec_* = 0 (5)
    :B_dec, :S_dec, :mu_dec_B, :mu_dec_S, :n_dec,
)

"""
    DynParams

Type alias for the dynamics-parameter ComponentVector. 28 named fields;
constructed via `make_dyn_params(; kwargs...)` for type stability.
"""
const DynParams = ComponentVector

"""
    make_dyn_params(; kwargs...)

Build a `DynParams` ComponentVector with the canonical 28-field layout.
Keyword arguments must include every key in `DYN_PARAM_KEYS`; missing or
extra keys raise an error so the caller catches typos at construction
rather than silently picking up a stale default.
"""
function make_dyn_params(; kwargs...)
    nt = (; kwargs...)
    given = keys(nt)
    missing_keys = setdiff(DYN_PARAM_KEYS, given)
    extra_keys   = setdiff(given, DYN_PARAM_KEYS)
    if !isempty(missing_keys)
        error("make_dyn_params: missing keys: $(missing_keys)")
    end
    if !isempty(extra_keys)
        error("make_dyn_params: unknown keys: $(extra_keys)")
    end
    # Order the values per DYN_PARAM_KEYS so the underlying flat vector
    # always has the same layout regardless of kwarg order.
    ordered = NamedTuple{DYN_PARAM_KEYS}(getindex.(Ref(nt), DYN_PARAM_KEYS))
    return ComponentVector{Float64}(; ordered...)
end

# ── Observation-channel parameters (22 fields) ─────────────────────────────
# Maps to `Fsa.V5.ObsParams` in `Obs.lean`. SEPARATE structure from
# `DynParams` — that separation is the structural fix for Bug 1.

const OBS_PARAM_KEYS = (
    # HR channel (5)
    :HR_base, :kappa_B_HR, :alpha_A_HR, :beta_C_HR, :sigma_HR,
    # Sleep channel (3) — Bernoulli logistic
    :k_C, :k_A, :c_tilde,
    # Stress channel (5) — note `sigma_S_obs`, NOT `sigma_S`
    :S_base, :k_F, :k_A_S, :beta_C_S, :sigma_S_obs,
    # Steps channel (6) — log-Gaussian
    :mu_step0, :beta_B_st, :beta_F_st, :beta_A_st, :beta_C_st, :sigma_st,
    # VolumeLoad channel (3)
    :beta_S_VL, :beta_F_VL, :sigma_VL,
)

"""
    ObsParams

Type alias for the obs-channel-parameter ComponentVector. 22 named
fields; built via `make_obs_params(; kwargs...)`.
"""
const ObsParams = ComponentVector

"""
    make_obs_params(; kwargs...)

Build an `ObsParams` ComponentVector with the canonical 22-field layout.
"""
function make_obs_params(; kwargs...)
    nt = (; kwargs...)
    given = keys(nt)
    missing_keys = setdiff(OBS_PARAM_KEYS, given)
    extra_keys   = setdiff(given, OBS_PARAM_KEYS)
    if !isempty(missing_keys)
        error("make_obs_params: missing keys: $(missing_keys)")
    end
    if !isempty(extra_keys)
        error("make_obs_params: unknown keys: $(extra_keys)")
    end
    ordered = NamedTuple{OBS_PARAM_KEYS}(getindex.(Ref(nt), OBS_PARAM_KEYS))
    return ComponentVector{Float64}(; ordered...)
end

# ── Schedule type aliases ──────────────────────────────────────────────────
# `RBFCoeffs` is the controller's decision variable: a flat vector of
# length 2*n_anchors holding both channels' anchor weights. Distinct
# type from `DynParams` (the SMC² inference target). The retracted
# "Bug 5" pattern-matched these as the same scalar — Julia's type
# system makes that unifiable conflation impossible.

const RBFCoeffs       = Vector{Float64}
const SmoothSchedule  = Vector{BimodalPhi}
const AppliedSchedule = Vector{BimodalPhi}    # post-Φ-burst expansion

# ── Public exports ─────────────────────────────────────────────────────────

export FSAv5State, BimodalPhi
export A_TYP, F_TYP, PHI_TYP
export DYN_PARAM_KEYS, OBS_PARAM_KEYS
export DynParams, ObsParams, make_dyn_params, make_obs_params
export RBFCoeffs, SmoothSchedule, AppliedSchedule
