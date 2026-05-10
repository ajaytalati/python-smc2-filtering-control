# FSA-v5 deterministic observation channel means — pure Julia
# transcription of `version_1_5_LEAN/Fsa/V5/Obs.lean`. Each function
# is the deterministic core of a stochastic observation channel
# (Gaussian or Bernoulli). The stochastic samplers add noise on top
# (in `simulation_v5.jl::sample_obs_v5` and `_plant_v5.jl::_sample_obs`).
#
# Differential testing of the channel means is at machine precision
# against the Lean reference; the random sampling on top is a
# Julia-only concern.
#
# Tech guide §3.1 (lines 339-350) gives the formulae; the Lean file
# Obs.lean (lines 79-104) is the line-by-line reference.

module ObsV5

using StaticArrays

export hr_mean, sleep_prob, stress_mean, steps_log_mean, volume_load_mean


# ── Channel means ──────────────────────────────────────────────────────
# `params` is any Dict-or-NamedTuple-like with the 22 obs-channel
# fields (see `SimulationV5.OBS_PARAM_KEYS_V5` for the canonical list).

"""
    hr_mean(y, C, params) -> Float64

HR channel mean (sleep-active Gaussian).
  μ_HR = HR_base − κ_B^HR · B + α_A^HR · A + β_C^HR · C

Mirrors `Fsa.V5.hrMean` (`Fsa/V5/Obs.lean:81-82`).
Tech guide eq:obs-HR (line 340).
"""
@inline function hr_mean(y::AbstractVector, C::Real, params)
    B = y[1]; A = y[4]
    return _get(params, :HR_base) -
            _get(params, :kappa_B_HR) * B +
            _get(params, :alpha_A_HR) * A +
            _get(params, :beta_C_HR)  * C
end

"""
    sleep_prob(y, C, params) -> Float64

Sleep Bernoulli probability — logistic of `k_C·C + k_A·A − c̃`.

Mirrors `Fsa.V5.sleepProb` (`Fsa/V5/Obs.lean:86-88`).
Tech guide eq:obs-sleep (line 348).
"""
@inline function sleep_prob(y::AbstractVector, C::Real, params)
    A = y[4]
    z = _get(params, :k_C) * C + _get(params, :k_A) * A -
         _get(params, :c_tilde)
    return 1.0 / (1.0 + exp(-z))
end

"""
    stress_mean(y, C, params) -> Float64

Stress channel mean (wake-active Gaussian).
  μ_S = S_base + k_F · F − k_{A,S} · A + β_C^S · C

Mirrors `Fsa.V5.stressMean` (`Fsa/V5/Obs.lean:92-93`).
Tech guide eq:obs-S (line 342).
"""
@inline function stress_mean(y::AbstractVector, C::Real, params)
    F = y[3]; A = y[4]
    return _get(params, :S_base) +
            _get(params, :k_F)      * F -
            _get(params, :k_A_S)    * A +
            _get(params, :beta_C_S) * C
end

"""
    steps_log_mean(y, C, params) -> Float64

Steps log-mean (wake-active log-Gaussian).
  μ_log = μ_step,0 + β_B^st · B − β_F^st · F + β_A^st · A + β_C^st · C

Mirrors `Fsa.V5.stepsLogMean` (`Fsa/V5/Obs.lean:97-99`).
Tech guide eq:obs-st (line 344).
"""
@inline function steps_log_mean(y::AbstractVector, C::Real, params)
    B = y[1]; F = y[3]; A = y[4]
    return _get(params, :mu_step0) +
            _get(params, :beta_B_st) * B -
            _get(params, :beta_F_st) * F +
            _get(params, :beta_A_st) * A +
            _get(params, :beta_C_st) * C
end

"""
    volume_load_mean(y, params) -> Float64

VolumeLoad channel mean (training-session-only Gaussian, no circadian).
  μ_VL = β_S^VL · S − β_F^VL · F

Mirrors `Fsa.V5.volumeLoadMean` (`Fsa/V5/Obs.lean:103-104`).
Tech guide eq:obs-VL (line 346).
"""
@inline function volume_load_mean(y::AbstractVector, params)
    S = y[2]; F = y[3]
    return _get(params, :beta_S_VL) * S - _get(params, :beta_F_VL) * F
end


# ── Internal: dual access for Dict / NamedTuple obs-params ────────────
@inline _get(p::Dict{Symbol, T}, k::Symbol) where {T}    = p[k]
@inline _get(p::NamedTuple, k::Symbol)                    = getproperty(p, k)

end # module ObsV5
