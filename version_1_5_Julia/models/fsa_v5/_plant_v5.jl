# FSA-v5 plant — stateless trajectory simulator. Single-bin step is
# `em_step_v5` from `_dynamics_v5.jl` (which IS diff-tested against
# the Lean reference `Fsa.V5.emStep`); the rollout, the bin-time
# counter, and the keyed-RNG noise are all Julia-side wrappers
# without a Lean counterpart. Their correctness inherits from
# `em_step_v5`'s correctness.
#
# Mirrors v1.5's `_plant.jl` layout but with 6D state.

module PlantV5

using StaticArrays
using StableRNGs

import ..DynamicsV5: em_step_v5
import ..SimulationV5: DT_BIN_DAYS, BINS_PER_DAY, SEDENTARY_INIT,
                        TRAINED_ATHLETE_INIT, TRUTH_PARAMS_V5

export PlantState6D, plant_step_v5, plant_rollout_v5, init_plant_state_sedentary
export init_plant_state_trained, init_plant_state_from


# ── Plant state ────────────────────────────────────────────────────────
# Mirrors `Fsa.V5.PlantState` (`Fsa/V5/Types.lean::State6D`) plus a bin
# counter `t_bin` for the rollout. Immutable so plant_step always
# returns a fresh instance.


"""
    PlantState6D(state, t_bin)

Immutable 6D plant state with a bin counter. `state` is the latent
6-vector `[B, S, F, A, K_FB, K_FS]`; `t_bin` is the integer bin index
since simulation start (used to derive the circadian phase).
"""
struct PlantState6D
    state :: SVector{6, Float32}
    t_bin :: Int
end


# ── Sigma-diag extraction ──────────────────────────────────────────────
# The diffusion-scale sigma_diag passed to `em_step_v5` is the 6-vector
# of (sigma_B, sigma_S, sigma_F, sigma_A, sigma_K, sigma_K). It mirrors
# the v1.5 convention of looking up scalar sigmas from the params dict.

@inline function _sigma_diag_from_params(p)
    return SVector{6, Float32}(
        p[:sigma_B], p[:sigma_S], p[:sigma_F],
        p[:sigma_A], p[:sigma_K], p[:sigma_K],
    )
end


# ── Single-bin plant step (deterministic + obs sample) ─────────────────

"""
    plant_step_v5(state::PlantState6D, phi, params, dt, key) -> (state, obs)

Advance the plant by one bin. Returns a `NamedTuple` with the new
`PlantState6D` (`state` field) and a 5-channel observation
`NamedTuple` (`obs` field).

  - `phi`    : `(Phi_B, Phi_S)` per-bin stimulus.
  - `params` : Dict with all 28 dynamics + diffusion + 22 obs-channel
                fields (i.e. `merge(TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5)`).
                See tech guide §9.5 for the merge requirement.
  - `dt`     : bin width in days (typically `DT_BIN_DAYS`).
  - `key`    : `UInt64` RNG key. Two sub-keys are derived
                deterministically: one for the SDE noise, one for the
                obs noise.

Pure: same `(state, phi, params, dt, key)` always returns the same
result.
"""
function plant_step_v5(state::PlantState6D,
                        phi::Tuple{<:Real, <:Real},
                        params::AbstractDict,
                        dt::Real,
                        key::UInt64)
    # Two sub-keys for SDE / obs to keep them statistically
    # independent across calls with the same root key.
    sde_key = hash((key, :sde))
    obs_key = hash((key, :obs))

    # SDE noise (6-vector standard normal)
    sde_rng = StableRNG(sde_key)
    sde_noise = SVector{6, Float32}(
        randn(sde_rng, Float32), randn(sde_rng, Float32), randn(sde_rng, Float32),
        randn(sde_rng, Float32), randn(sde_rng, Float32), randn(sde_rng, Float32),
    )

    sigma_diag = _sigma_diag_from_params(params)
    phi_f32    = (Float32(phi[1]), Float32(phi[2]))
    new_state  = em_step_v5(state.state, phi_f32, params, sigma_diag, Float32(dt), sde_noise)

    # Obs sample. Sleep-gating logic is the bench driver's
    # responsibility; here we sample every channel and return all
    # five. Circadian regressor C(t) = cos(2π · t_day) at bin
    # t_bin: t_day = t_bin / BINS_PER_DAY (units of days).
    t_day = Float32(state.t_bin) / Float32(BINS_PER_DAY)
    C     = cos(2.0f0 * Float32(pi) * t_day)
    obs   = _sample_obs(new_state, C, params, obs_key)

    return (
        state = PlantState6D(new_state, state.t_bin + 1),
        obs   = obs,
    )
end


# ── Multi-bin rollout (deterministic; uses keyed RNG for reproducibility) ─

"""
    plant_rollout_v5(state0, phi_seq, params, dt, key)
        -> NamedTuple

Roll the plant forward over a sequence of bimodal per-bin stimuli.
Returns a NamedTuple with the bench-consumed fields (mirrors v1.5's
`Plant.plant_rollout` return shape but adapted for v5's 6D state +
5-channel obs + bimodal Φ + circadian regressor):

    final_state :: PlantState6D                   — state after last step
    trajectory  :: Matrix{Float32} (n, 6)         — post-step states
    obs_HR, obs_S, obs_steps, obs_VL :: Vector{Float32} (n,)
    obs_sleep   :: Vector{Float32} (n,)           — 0.0 / 1.0
    Phi_B, Phi_S :: Vector{Float32} (n,)
    C           :: Vector{Float32} (n,)           — circadian regressor

`phi_seq` is a Vector of `(Phi_B, Phi_S)` Tuples. `key` is the root RNG
key; per-bin sub-keys are derived deterministically. Pure: same inputs
always produce the same outputs.
"""
function plant_rollout_v5(state0::PlantState6D,
                           phi_seq::AbstractVector,
                           params::AbstractDict,
                           dt::Real,
                           key::UInt64)
    n = length(phi_seq)
    traj      = Matrix{Float32}(undef, n, 6)
    obs_HR    = Vector{Float32}(undef, n)
    obs_S_v   = Vector{Float32}(undef, n)
    obs_steps_v = Vector{Float32}(undef, n)
    obs_VL_v  = Vector{Float32}(undef, n)
    obs_sleep_v = Vector{Float32}(undef, n)
    Phi_B_v   = Vector{Float32}(undef, n)
    Phi_S_v   = Vector{Float32}(undef, n)
    C_v       = Vector{Float32}(undef, n)

    s = state0
    @inbounds for k in 1:n
        bin_key = hash((key, s.t_bin))
        out     = plant_step_v5(s, phi_seq[k], params, dt, bin_key)
        s       = out.state
        traj[k, 1] = s.state[1]; traj[k, 2] = s.state[2]
        traj[k, 3] = s.state[3]; traj[k, 4] = s.state[4]
        traj[k, 5] = s.state[5]; traj[k, 6] = s.state[6]
        obs_HR[k]      = out.obs.obs_HR
        obs_sleep_v[k] = out.obs.obs_sleep ? 1.0f0 : 0.0f0
        obs_S_v[k]     = out.obs.obs_S
        obs_steps_v[k] = out.obs.obs_steps
        obs_VL_v[k]    = out.obs.obs_VL
        Phi_B_v[k]     = Float32(phi_seq[k][1])
        Phi_S_v[k]     = Float32(phi_seq[k][2])
        # Circadian at the END of the bin (i.e. at the post-step time).
        # Matches the obs sampling convention inside plant_step_v5.
        t_day = Float32(s.t_bin) / Float32(BINS_PER_DAY)
        C_v[k] = cos(2.0f0 * Float32(pi) * t_day)
    end

    return (
        final_state = s,
        trajectory  = traj,
        obs_HR      = obs_HR,
        obs_S       = obs_S_v,
        obs_steps   = obs_steps_v,
        obs_VL      = obs_VL_v,
        obs_sleep   = obs_sleep_v,
        Phi_B       = Phi_B_v,
        Phi_S       = Phi_S_v,
        C           = C_v,
    )
end


# ── Constructors for typical initial states ────────────────────────────

"""
    init_plant_state_sedentary(; t_bin = 0) -> PlantState6D

Build a plant initial state from the canonical `SEDENTARY_INIT`
("deconditioned but otherwise healthy" per tech guide §9.10).
"""
function init_plant_state_sedentary(; t_bin::Int = 0)
    s = SVector{6, Float32}(
        Float32(SEDENTARY_INIT.B),   Float32(SEDENTARY_INIT.S),   Float32(SEDENTARY_INIT.F),
        Float32(SEDENTARY_INIT.A),   Float32(SEDENTARY_INIT.KFB), Float32(SEDENTARY_INIT.KFS),
    )
    return PlantState6D(s, t_bin)
end

"""
    init_plant_state_v5(; t_bin = 0) -> PlantState6D

Alias for init_plant_state_sedentary.
"""
const init_plant_state_v5 = init_plant_state_sedentary

"""
    init_plant_state_trained(; t_bin = 0) -> PlantState6D

Build a plant initial state from the trained-athlete reference
(tech guide §8.1, used in tests 1–4).
"""
function init_plant_state_trained(; t_bin::Int = 0)
    s = SVector{6, Float32}(
        Float32(TRAINED_ATHLETE_INIT.B),   Float32(TRAINED_ATHLETE_INIT.S),
        Float32(TRAINED_ATHLETE_INIT.F),   Float32(TRAINED_ATHLETE_INIT.A),
        Float32(TRAINED_ATHLETE_INIT.KFB), Float32(TRAINED_ATHLETE_INIT.KFS),
    )
    return PlantState6D(s, t_bin)
end

"""
    init_plant_state_from(init_nt; t_bin = 0) -> PlantState6D

Build a plant initial state from an arbitrary 6-field NamedTuple
(`(B, S, F, A, KFB, KFS)`). Used by the bench so that the plant-state
constructor and the `init_nt` reference dict can never get out of sync
(each --init-preset entry routes through this single function with the
right NamedTuple, rather than picking from a small set of hardcoded
fns and a separate NamedTuple).
"""
function init_plant_state_from(init_nt; t_bin::Int = 0)
    s = SVector{6, Float32}(
        Float32(init_nt.B),   Float32(init_nt.S),
        Float32(init_nt.F),   Float32(init_nt.A),
        Float32(init_nt.KFB), Float32(init_nt.KFS),
    )
    return PlantState6D(s, t_bin)
end


# ── Internal: per-bin obs sampler ──────────────────────────────────────
# Same as `SimulationV5.sample_obs_v5` but inlined here so the plant
# doesn't pull in the synthetic-data sampler module just for the
# inline call. The two are kept in sync by virtue of using the same
# 5-channel deterministic means (`hr_mean`, `sleep_prob`, etc.) from
# `obs_v5.jl` — but that module isn't loaded yet at this point in
# the dependency chain, so we do the per-channel formula inline here.

function _sample_obs(state::SVector{6, Float32}, C::Float32,
                      params::AbstractDict, key::UInt64)
    rng = StableRNG(key)
    B, S, F, A, _KFB, _KFS = state
    # HR (Gaussian)
    mu_HR  = Float32(params[:HR_base]) - Float32(params[:kappa_B_HR]) * B +
              Float32(params[:alpha_A_HR]) * A + Float32(params[:beta_C_HR]) * C
    obs_HR = mu_HR + Float32(params[:sigma_HR]) * randn(rng, Float32)
    # Sleep (Bernoulli logistic)
    z         = Float32(params[:k_C]) * C + Float32(params[:k_A]) * A - Float32(params[:c_tilde])
    p_sleep   = 1.0f0 / (1.0f0 + exp(-z))
    obs_sleep = rand(rng) < p_sleep
    # Stress (Gaussian)
    mu_S  = Float32(params[:S_base]) + Float32(params[:k_F]) * F -
             Float32(params[:k_A_S]) * A + Float32(params[:beta_C_S]) * C
    obs_S = mu_S + Float32(params[:sigma_S_obs]) * randn(rng, Float32)
    # Steps (log-Gaussian)
    mu_step = Float32(params[:mu_step0]) + Float32(params[:beta_B_st]) * B -
               Float32(params[:beta_F_st]) * F + Float32(params[:beta_A_st]) * A +
               Float32(params[:beta_C_st]) * C
    obs_steps = mu_step + Float32(params[:sigma_st]) * randn(rng, Float32)
    # VolumeLoad (Gaussian, no circadian)
    mu_VL  = Float32(params[:beta_S_VL]) * S - Float32(params[:beta_F_VL]) * F
    obs_VL = mu_VL + Float32(params[:sigma_VL]) * randn(rng, Float32)
    return (
        obs_HR    = obs_HR,
        obs_sleep = obs_sleep,
        obs_S     = obs_S,
        obs_steps = obs_steps,
        obs_VL    = obs_VL,
    )
end

end # module PlantV5
