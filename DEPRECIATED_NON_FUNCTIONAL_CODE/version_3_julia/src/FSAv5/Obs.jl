# FSAv5/Obs.jl — deterministic observation-channel means + sleep prob.
#
# Maps line-by-line to:
#   • LEAN  spec : `FSA_model_dev/lean/Fsa/V5/Obs.lean`
#   • Python    : `models/fsa_high_res/simulation.py:240-319`
#                 (`_sleep_prob`, `gen_obs_*` mean computations)
#   • LaTeX     : §3 (observation model)
#
# The Python `gen_obs_*` functions are stochastic (add Gaussian noise
# + sleep-state masking). Here we expose the **deterministic core** —
# the means and the Bernoulli probability — as pure functions. The
# random sampling and masking are I/O concerns; they don't need
# formal verification (per LEAN4 charter §5 step 6 scope).
#
# Bug 1 prevention: this module's parameters live on `ObsParams`,
# which is a SEPARATE ComponentVector from `DynParams`. The
# stress-channel obs noise is `op.sigma_S_obs` — distinct from
# `DynParams.sigma_S` (latent state-noise). The historical Python
# dict-key collision is structurally impossible.

# ── Circadian regressor C(t) ───────────────────────────────────────────────
# Mirrors `simulation.circadian` at `simulation.py:44-45`:
# C(t) = cos(2π · t_days + phi). The phase `phi` is a model-level
# constant (almost always zero in production); kept as keyword for
# parity with Python's signature.

"""
    circadian(t_days::Float64; phi::Float64 = 0.0)::Float64

The circadian regressor `C(t) = cos(2π · t + φ)`. Used as the input
to several obs channels (HR, sleep, stress, steps). Mirrors
`simulation.circadian`.
"""
@inline circadian(t_days::Float64; phi::Float64 = 0.0)::Float64 =
    cos(2.0 * pi * t_days + phi)

# ── HR channel (sleep-active) ──────────────────────────────────────────────

"""
    hr_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64

HR observation mean: `μ_HR = HR_base - κ_B^HR · B + α_A^HR · A + β_C^HR · C`.

Mirrors `simulation.gen_obs_hr` (line 265) and `Fsa.V5.hrMean` in
`Obs.lean`.
"""
@inline hr_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64 =
    op.HR_base - op.kappa_B_HR * y.B + op.alpha_A_HR * y.A + op.beta_C_HR * C

# ── Sleep channel (Bernoulli logistic) ─────────────────────────────────────

"""
    sleep_prob(y::FSAv5State, C::Float64, op::ObsParams)::Float64

Sleep Bernoulli probability: `p = σ(k_C · C + k_A · A - c̃)` where
`σ` is the standard logistic. Mirrors `simulation._sleep_prob`
(line 244-246) and `Fsa.V5.sleepProb` in `Obs.lean`.
"""
@inline function sleep_prob(y::FSAv5State, C::Float64, op::ObsParams)::Float64
    z = op.k_C * C + op.k_A * y.A - op.c_tilde
    return 1.0 / (1.0 + exp(-z))
end

# ── Stress channel (wake-only) ─────────────────────────────────────────────

"""
    stress_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64

Stress observation mean: `μ_S = S_base + k_F · F - k_{A,S} · A + β_C^S · C`.

Mirrors `simulation.gen_obs_stress` (line 278) and `Fsa.V5.stressMean`
in `Obs.lean`.
"""
@inline stress_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64 =
    op.S_base + op.k_F * y.F - op.k_A_S * y.A + op.beta_C_S * C

# ── Steps channel (log-Gaussian) ───────────────────────────────────────────

"""
    steps_log_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64

Steps log-mean (log-Gaussian channel):
`μ_log = μ_step0 + β_B^st · B - β_F^st · F + β_A^st · A + β_C^st · C`.

Mirrors `simulation.gen_obs_steps` (line 292) and
`Fsa.V5.stepsLogMean` in `Obs.lean`.
"""
@inline steps_log_mean(y::FSAv5State, C::Float64, op::ObsParams)::Float64 =
    op.mu_step0 + op.beta_B_st * y.B - op.beta_F_st * y.F +
    op.beta_A_st * y.A + op.beta_C_st * C

# ── VolumeLoad channel (no circadian) ──────────────────────────────────────

"""
    volume_load_mean(y::FSAv5State, op::ObsParams)::Float64

VolumeLoad observation mean: `μ_VL = β_S^VL · S - β_F^VL · F`. No
circadian dependence. Mirrors `simulation.gen_obs_volumeload` (line
305) and `Fsa.V5.volumeLoadMean` in `Obs.lean`.
"""
@inline volume_load_mean(y::FSAv5State, op::ObsParams)::Float64 =
    op.beta_S_VL * y.S - op.beta_F_VL * y.F

export circadian
export hr_mean, sleep_prob, stress_mean, steps_log_mean, volume_load_mean
