"""
    SimulationV5

Stateless module defining fundamental constants, physiological parameters, 
and observation models for the FSA-v5 stochastic differential equation (SDE).

This module serves as the central configuration hub for:
- Time discretization (15-minute bin grid).
- Parameter Centering around a Typical Operating Point.
- Ground-truth parameter sets for the v5 closed-island topology.
- Initial state presets for simulation scenarios.
- Observation likelihood parameters (HR, Sleep, Stress, etc.).

Implementation Note: All physiological and observation parameters are defined 
using `Float32` to ensure high-performance execution on GPU targets without 
redundant type-casting.

Source of truth: `version_1_5_LEAN/Fsa/V5/Types.lean`.
Bit-equivalence verified in: `version_1_5_Julia/diff_test/test_lean_diff_v5.jl`.
"""
module SimulationV5

using StaticArrays
using StableRNGs

export BINS_PER_DAY, DT_BIN_DAYS
export A_TYP, F_TYP
export TRUTH_PARAMS_V4, TRUTH_PARAMS_V5, TRUTH_PARAMS_V5_RECOMMENDED_V2
export select_truth_preset
export DEFAULT_OBS_PARAMS_V5
export DEFAULT_INIT, SEDENTARY_INIT, TRAINED_ATHLETE_INIT, TRAINED_ATHLETE_INIT_V2,
       MIDDLE_INIT_V2
export FROZEN_PARAMS_V5, FROZEN_PARAMS_V5_RECOMMENDED_V2
export PARAM_KEYS_V5, OBS_PARAM_KEYS_V5
export sample_obs_v5, params_dict_to_nt


# =============================================================================
# 1. TIME GRID CONFIGURATION
# =============================================================================

"""
    BINS_PER_DAY::Int
    DT_BIN_DAYS::Float32

The simulation uses a 15-minute discrete time grid by default (96 bins/day).
This resolution is required for model identifiability from overnight heart rate signals.
Adjustable via the `FSA_STEP_MINUTES` environment variable.
"""
const _STEP_MIN::Int        = parse(Int, get(ENV, "FSA_STEP_MINUTES", "15"))
const BINS_PER_DAY::Int     = (60 * 24) ÷ _STEP_MIN
const DT_BIN_DAYS::Float32  = Float32(1.0 / BINS_PER_DAY)


# =============================================================================
# 2. OPERATING POINT CONSTANTS (Parameter Centering)
# =============================================================================

"""
    A_TYP::Float32 = 0.10f0
    F_TYP::Float32 = 0.20f0

Reference constants for "Parameter Centering around a Typical Operating Point."
These are baked into the non-linear drift equations to ensure:
1. Identifiability: Reducing multi-collinearity between linear and quadratic terms.
2. Stability: Avoiding catastrophic cancellation in floating-point arithmetic.
3. Interpretation: Gains and penalties represent deviations from a healthy human baseline.
"""
const A_TYP::Float32 = 0.10f0   # Typical autonomic amplitude
const F_TYP::Float32 = 0.20f0   # Typical fatigue pool level


# =============================================================================
# 3. TRUTH PARAMETERS (Dynamics)
# =============================================================================

"""
    TRUTH_PARAMS_V4::Dict{Symbol, Float32}

Baseline parameters recovering the FSA-v4 model. Setting the Hill-deconditioning 
penalties (mu_dec_*) to zero results in an open quarter-disc basin topology where 
sedentary behavior is stable.
"""
const TRUTH_PARAMS_V4::Dict{Symbol, Float32} = Dict(
    # Aerobic Fitness B
    :tau_B      => 42.0f0,
    :kappa_B    => 0.012f0 * (1.0f0 + 0.40f0 * A_TYP),
    :epsilon_AB => 0.40f0,
    # Strength Adaptation S
    :tau_S      => 60.0f0,
    :kappa_S    => 0.008f0 * (1.0f0 + 0.20f0 * A_TYP),
    :epsilon_AS => 0.20f0,
    # Unified Fatigue F
    :tau_F      => 7.0f0 / (1.0f0 + 1.00f0 * A_TYP),
    :lambda_A   => 1.00f0,
    # Busso Variable-Dose K
    :KFB_0      => 0.030f0,
    :KFS_0      => 0.050f0,
    :tau_K      => 21.0f0,
    :mu_K       => 0.005f0,
    # Stuart-Landau bifurcation parameter (Centered)
    :mu_0       => 0.02f0 + 0.40f0 * (F_TYP * F_TYP),
    :mu_B       => 0.30f0,
    :mu_S       => 0.15f0,
    :mu_F       => 0.10f0 + 2.0f0 * F_TYP * 0.40f0,
    :mu_FF      => 0.40f0,
    :eta        => 0.20f0,
    # State-dependent diffusion magnitudes
    :sigma_B    => 0.010f0,
    :sigma_S    => 0.008f0,
    :sigma_F    => 0.012f0,
    :sigma_A    => 0.020f0,
    :sigma_K    => 0.005f0,
    # v5 Hill deconditioning amplitudes (silent in V4)
    :B_dec      => 0.10f0,
    :S_dec      => 0.10f0,
    :mu_dec_B   => 0.0f0,
    :mu_dec_S   => 0.0f0,
    :n_dec      => 4.0f0,
)

"""
    TRUTH_PARAMS_V5::Dict{Symbol, Float32}

Production parameters activating the v5 closed-island basin topology.
Incorporates Hill-deconditioning thresholds (B_dec, S_dec) to model 
autonomic collapse under chronic deconditioning.
"""
const TRUTH_PARAMS_V5::Dict{Symbol, Float32} = let
    p = copy(TRUTH_PARAMS_V4)
    p[:B_dec]    = 0.07f0
    p[:S_dec]    = 0.07f0
    p[:mu_dec_B] = 0.10f0
    p[:mu_dec_S] = 0.10f0
    p[:n_dec]    = 4.0f0
    p
end

"""
    TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32}

v2 re-parametrisation of TRUTH_PARAMS_V5 for the controllability_v2_proofs
theorems. Eight numerical entries differ from canonical; all other entries
(tau_F, mu_dec_*, n_dec, K^0_*, mu_K, tau_K, sigma_*, mu_0, mu_B, mu_S, eta,
epsilon_A*, lambda_A) are inherited unchanged.

The 8 v2 overrides (controllability_v2_proofs.pdf §2.3):
  tau_B    : 42 d        -> 21 d        (halved; B* invariant under (tau,kappa) -> (tau/2, 2 kappa))
  kappa_B  : 0.01248     -> 0.02496     (doubled, centred form)
  tau_S    : 60 d        -> 30 d
  kappa_S  : 0.00816     -> 0.01632     (doubled, centred form)
  B_dec    : 0.07        -> 0.25        (Hill threshold raised 3.6x; shifts island toward (1,1))
  S_dec    : 0.07        -> 0.25
  mu_F     : 0.26        -> 0.030       (reward-side fatigue penalty reduced 88%)
  mu_FF    : 0.40        -> 0.020       (reduced 95%)
"""
const TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32} = let
    p = copy(TRUTH_PARAMS_V5)
    p[:tau_B]   = 21.0f0
    p[:kappa_B] = 0.012f0 * (1.0f0 + 0.40f0 * A_TYP) * 2.0f0   # 0.02496f0
    p[:tau_S]   = 30.0f0
    p[:kappa_S] = 0.008f0 * (1.0f0 + 0.20f0 * A_TYP) * 2.0f0   # 0.01632f0
    p[:B_dec]   = 0.25f0
    p[:S_dec]   = 0.25f0
    p[:mu_F]    = 0.030f0
    p[:mu_FF]   = 0.020f0
    p
end


# =============================================================================
# 4. OBSERVATION PARAMETERS
# =============================================================================

"""
    DEFAULT_OBS_PARAMS_V5::Dict{Symbol, Float32}

Physiologically representative coefficients for the 5-channel observation model:
- HR: Heart Rate (bpm)
- Sleep: Bernoulli (logistic)
- Stress: Subjective Wellness Score
- Steps: Log-Gaussian activity
- VolumeLoad: Training load (session-only)
"""
const DEFAULT_OBS_PARAMS_V5::Dict{Symbol, Float32} = Dict(
    # HR channel (sleep-active)
    :HR_base     => 60.0f0,    # bpm at zero load
    :kappa_B_HR  => 20.0f0,    # bpm decrease per unit B (fitness lowers HR)
    :alpha_A_HR  => 5.0f0,     # bpm increase per unit A
    :beta_C_HR   => 2.0f0,     # bpm circadian amplitude
    :sigma_HR    => 2.0f0,     # bpm obs noise
    # Sleep Bernoulli (logit slopes)
    :k_C         => 4.0f0,
    :k_A         => 1.0f0,
    :c_tilde     => 1.0f0,
    # Stress channel (wake-active)
    :S_base      => 50.0f0,
    :k_F         => 30.0f0,
    :k_A_S       => 10.0f0,
    :beta_C_S    => 5.0f0,
    :sigma_S_obs => 4.0f0,
    # Steps channel (log-Gaussian means)
    :mu_step0    => 8.5f0,
    :beta_B_st   => 1.0f0,
    :beta_F_st   => 0.5f0,
    :beta_A_st   => 0.5f0,
    :beta_C_st   => 0.3f0,
    :sigma_st    => 0.4f0,
    # VolumeLoad channel (session-active)
    :beta_S_VL   => 100.0f0,
    :beta_F_VL   => 50.0f0,
    :sigma_VL    => 10.0f0,
)


# =============================================================================
# 5. INITIAL STATE PRESETS
# =============================================================================

"""
    SEDENTARY_INIT::NamedTuple

"Deconditioned but healthy" starting point. Characterized by low aerobic 
fitness (B=0.05) and low strength (S=0.10).
"""
const SEDENTARY_INIT = (
    B   = 0.05f0,
    S   = 0.10f0,
    F   = 0.30f0,
    A   = 0.10f0,
    KFB = 0.030f0,
    KFS = 0.050f0,
)

"""
    DEFAULT_INIT::NamedTuple

Alias for SEDENTARY_INIT. Used primarily by the particle filter as a generic 
NaN-guard fallback state rather than a specific physical scenario.
"""
const DEFAULT_INIT = SEDENTARY_INIT

"""
    TRAINED_ATHLETE_INIT::NamedTuple

Canonical "Trained Athlete" starting point used for closed-loop benchmark 
scenarios. Fitness and strength are balanced around 0.50.
"""
const TRAINED_ATHLETE_INIT = (
    B   = 0.50f0,
    S   = 0.45f0,
    F   = 0.20f0,
    A   = 0.45f0,
    KFB = 0.06f0,
    KFS = 0.07f0,
)

"""
    TRAINED_ATHLETE_INIT_V2::NamedTuple

Slow-manifold equilibrium of FSA-v5 under TRUTH_PARAMS_V5_RECOMMENDED_V2 at
the v2 island centre Φ = (1.06, 0.78), upper-stable autonomic root A* = 1.238.
Not on the slow manifold under canonical TRUTH_PARAMS_V5. See
controllability_v2_proofs.pdf §2.4.
"""
const TRAINED_ATHLETE_INIT_V2 = (
    B   = 0.7993f0,
    S   = 0.4688f0,
    F   = 0.7925f0,
    A   = 1.2384f0,
    KFB = 0.1414f0,
    KFS = 0.1322f0,
)

"""
    MIDDLE_INIT_V2::NamedTuple

Per-component average of SEDENTARY_INIT and TRAINED_ATHLETE_INIT_V2. A
"middle-of-the-road" starting point lying roughly halfway between the v2
sedentary basin and the v2 slow-manifold trained equilibrium — intended as
an easier launchpad for the SMC²-MPC controller when probing controllability
from a less extreme initial condition.

Only meaningful under --truth-preset v2 (it's averaged against the v2
trained-athlete state). Using it under canonical TRUTH_PARAMS_V5 mixes a
v2 reference point into a canonical run, which is not a coherent scenario.
"""
const MIDDLE_INIT_V2 = (
    B   = (SEDENTARY_INIT.B   + TRAINED_ATHLETE_INIT_V2.B)   / 2.0f0,
    S   = (SEDENTARY_INIT.S   + TRAINED_ATHLETE_INIT_V2.S)   / 2.0f0,
    F   = (SEDENTARY_INIT.F   + TRAINED_ATHLETE_INIT_V2.F)   / 2.0f0,
    A   = (SEDENTARY_INIT.A   + TRAINED_ATHLETE_INIT_V2.A)   / 2.0f0,
    KFB = (SEDENTARY_INIT.KFB + TRAINED_ATHLETE_INIT_V2.KFB) / 2.0f0,
    KFS = (SEDENTARY_INIT.KFS + TRAINED_ATHLETE_INIT_V2.KFS) / 2.0f0,
)


# =============================================================================
# 6. METADATA & FIELD KEYS
# =============================================================================

"""
    FROZEN_PARAMS_V5::Dict{Symbol, Float32}

The subset of 14 parameters that are NOT estimated by the SMC² filter. 
Includes diffusion scales (identifiability is weak) and deconditioning 
parameters (informable only via long detraining episodes).
"""
const FROZEN_PARAMS_V5::Dict{Symbol, Float32} = Dict(
    :sigma_B  => TRUTH_PARAMS_V5[:sigma_B],
    :sigma_S  => TRUTH_PARAMS_V5[:sigma_S],
    :sigma_F  => TRUTH_PARAMS_V5[:sigma_F],
    :sigma_A  => TRUTH_PARAMS_V5[:sigma_A],
    :sigma_K  => TRUTH_PARAMS_V5[:sigma_K],
    :KFB_0    => TRUTH_PARAMS_V5[:KFB_0],
    :KFS_0    => TRUTH_PARAMS_V5[:KFS_0],
    :tau_K    => TRUTH_PARAMS_V5[:tau_K],
    :n_dec    => TRUTH_PARAMS_V5[:n_dec],
    :B_dec    => TRUTH_PARAMS_V5[:B_dec],
    :S_dec    => TRUTH_PARAMS_V5[:S_dec],
    :mu_dec_B => TRUTH_PARAMS_V5[:mu_dec_B],
    :mu_dec_S => TRUTH_PARAMS_V5[:mu_dec_S],
)

"""
    FROZEN_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32}

v2 re-parametrisation of FROZEN_PARAMS_V5. Mirrors the v2 truth: B_dec and
S_dec are raised from 0.07 to 0.25 (matching TRUTH_PARAMS_V5_RECOMMENDED_V2).
The other 11 entries are identical to canonical FROZEN_PARAMS_V5.

Used by the bench driver under --truth-preset v2 to feed the correct frozen
values into the filter inner-PF and into `posterior_mean_v5` (which merges
frozen entries into the controller's params dict). Without this, the filter
and the controller's closed-loop cost rollout would silently use canonical
B_dec=0.07/S_dec=0.07 even when the plant uses v2 (=0.25), producing severe
model misspecification.
"""
const FROZEN_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32} = Dict(
    :sigma_B  => TRUTH_PARAMS_V5_RECOMMENDED_V2[:sigma_B],
    :sigma_S  => TRUTH_PARAMS_V5_RECOMMENDED_V2[:sigma_S],
    :sigma_F  => TRUTH_PARAMS_V5_RECOMMENDED_V2[:sigma_F],
    :sigma_A  => TRUTH_PARAMS_V5_RECOMMENDED_V2[:sigma_A],
    :sigma_K  => TRUTH_PARAMS_V5_RECOMMENDED_V2[:sigma_K],
    :KFB_0    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:KFB_0],
    :KFS_0    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:KFS_0],
    :tau_K    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:tau_K],
    :n_dec    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:n_dec],
    :B_dec    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:B_dec],   # 0.25 (v2)
    :S_dec    => TRUTH_PARAMS_V5_RECOMMENDED_V2[:S_dec],   # 0.25 (v2)
    :mu_dec_B => TRUTH_PARAMS_V5_RECOMMENDED_V2[:mu_dec_B],
    :mu_dec_S => TRUTH_PARAMS_V5_RECOMMENDED_V2[:mu_dec_S],
)

"""
    select_truth_preset(preset::AbstractString)
        -> (truth_params::Dict, frozen_params::Dict)

Resolve a `--truth-preset` string into the matching (truth, frozen) dict
pair the bench should use for the plant, filter, and controller. Lives
in SimulationV5 (rather than the bench driver) so the dispatch is
testable in isolation.

Legal preset values: `"canonical"`, `"v2"`. Any other value raises an
explicit error so a misspelled flag can't silently fall back to a default.

The two dicts returned MUST be used together: if the truth dict is v2,
the frozen dict must also be v2 (otherwise the filter inner-PF and the
closed-loop controller's posterior dict get canonical B_dec / S_dec
while the plant uses v2's — the exact bug this function exists to make
unrepeatable).
"""
function select_truth_preset(preset::AbstractString)
    if preset == "canonical"
        return (TRUTH_PARAMS_V5, FROZEN_PARAMS_V5)
    elseif preset == "v2"
        return (TRUTH_PARAMS_V5_RECOMMENDED_V2, FROZEN_PARAMS_V5_RECOMMENDED_V2)
    else
        error("--truth-preset must be \"canonical\" or \"v2\", got: \"$preset\"")
    end
end

"""
Canonical field order required for deterministic JSON serialization and 
bit-equivalence testing against the Lean4/C++ binary.
"""
const PARAM_KEYS_V5::NTuple{28, Symbol} = (
    :tau_B, :kappa_B, :epsilon_AB,
    :tau_S, :kappa_S, :epsilon_AS,
    :tau_F, :lambda_A,
    :KFB_0, :KFS_0, :tau_K, :mu_K,
    :mu_0, :mu_B, :mu_S, :mu_F, :mu_FF, :eta,
    :sigma_B, :sigma_S, :sigma_F, :sigma_A, :sigma_K,
    :B_dec, :S_dec, :mu_dec_B, :mu_dec_S, :n_dec,
)

const OBS_PARAM_KEYS_V5::NTuple{22, Symbol} = (
    :HR_base, :kappa_B_HR, :alpha_A_HR, :beta_C_HR, :sigma_HR,
    :k_C, :k_A, :c_tilde,
    :S_base, :k_F, :k_A_S, :beta_C_S, :sigma_S_obs,
    :mu_step0, :beta_B_st, :beta_F_st, :beta_A_st, :beta_C_st, :sigma_st,
    :beta_S_VL, :beta_F_VL, :sigma_VL,
)


# =============================================================================
# 7. HELPER FUNCTIONS
# =============================================================================

"""
    params_dict_to_nt(p::Dict{Symbol, T}) where {T} -> NamedTuple

Efficiently converts a parameter dictionary into a NamedTuple using the 
canonical `PARAM_KEYS_V5` order. Used in hot inner loops to avoid 
dictionary lookup overhead.
"""
@inline function params_dict_to_nt(p::Dict{Symbol, T}) where {T}
    NamedTuple{PARAM_KEYS_V5}(getindex.(Ref(p), PARAM_KEYS_V5))
end

"""
    sample_obs_v5(state, C, params, key::UInt64) -> NamedTuple

Generates a stochastic observation vector from a single 6D state bin.
Pure function: same inputs always produce identical results.
Caller is responsible for applying sleep/wake masks to the output.
"""
function sample_obs_v5(state::SVector{6, Float32}, C::Float32,
                        params::Dict{Symbol, Float32}, key::UInt64)
    rng = StableRNG(key)
    B, S, F, A, _KFB, _KFS = state
    # HR — sleep-active Gaussian
    mu_HR  = params[:HR_base] - params[:kappa_B_HR] * B +
              params[:alpha_A_HR] * A + params[:beta_C_HR] * C
    obs_HR = mu_HR + params[:sigma_HR] * randn(rng, Float32)
    # Sleep — Bernoulli logistic
    z       = params[:k_C] * C + params[:k_A] * A - params[:c_tilde]
    p_sleep = 1.0f0 / (1.0f0 + exp(-z))
    obs_sleep = rand(rng) < p_sleep
    # Stress — wake-active Gaussian
    mu_S  = params[:S_base] + params[:k_F] * F -
             params[:k_A_S] * A + params[:beta_C_S] * C
    obs_S = mu_S + params[:sigma_S_obs] * randn(rng, Float32)
    # Steps — log-Gaussian
    mu_step = params[:mu_step0] + params[:beta_B_st] * B -
               params[:beta_F_st] * F + params[:beta_A_st] * A +
               params[:beta_C_st] * C
    obs_steps = mu_step + params[:sigma_st] * randn(rng, Float32)
    # VolumeLoad — Gaussian, no circadian
    mu_VL  = params[:beta_S_VL] * S - params[:beta_F_VL] * F
    obs_VL = mu_VL + params[:sigma_VL] * randn(rng, Float32)
    return (
        obs_HR    = obs_HR,
        obs_sleep = obs_sleep,
        obs_S     = obs_S,
        obs_steps = obs_steps,
        obs_VL    = obs_VL,
    )
end

end # module SimulationV5
