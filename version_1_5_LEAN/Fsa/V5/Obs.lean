import Fsa.V5.Types

/-!
# FSA-v5 observation-channel deterministic means

Single source of truth for the deterministic per-channel mean (or
Bernoulli probability) functions. Transcribed line-by-line from
`models/fsa_high_res/simulation.py:240-319` (functions `_sleep_prob`,
`gen_obs_*`).

The Python `gen_obs_*` functions are stochastic (they add Gaussian
noise to the means and apply sleep-state masking). Here we expose the
**deterministic** core — the means and the sleep probability — as
pure functions. This is the right granularity for the LEAN4
differential test:

  - random sampling and masking are I/O-side concerns (RNG seeds,
    sleep labels), not model semantics; they don't need formal
    verification.
  - the means / probabilities ARE the model's claims about the
    observation distributions; differential testing them catches
    structural bugs in the channel formulae.

## Bug 1 prevention

`ObsParams` holds the obs-channel parameters separately from `Params`.
The historical Python `DEFAULT_PARAMS` dict had two `'sigma_S'` keys
(latent-S state noise vs stress-channel obs noise). The state-noise
lives in `Params.sigma_S`; the obs-channel noise lives in
`ObsParams.sigma_S_obs`. Two separate fields on two separate
structures — the dict-collision bug cannot be expressed.
-/

namespace Fsa.V5

/-- Observation-channel parameters. Distinct from `Params` (the
dynamics + state-noise scales). The historical `sigma_S` collision
class is structurally prevented: state-noise `Params.sigma_S` and
stress-obs noise `ObsParams.sigma_S_obs` are different fields on
different records. -/
structure ObsParams where
  /- ── HR channel ── -/
  HR_base    : Float
  kappa_B_HR : Float
  alpha_A_HR : Float
  beta_C_HR  : Float
  sigma_HR   : Float
  /- ── Sleep channel (Bernoulli probability via logistic) ── -/
  k_C        : Float
  k_A        : Float
  c_tilde    : Float
  /- ── Stress channel ── -/
  S_base      : Float
  k_F         : Float
  k_A_S       : Float
  beta_C_S    : Float
  /-- Stress observation noise. NOT the same as `Params.sigma_S`
      (latent-S Jacobi diffusion scale). -/
  sigma_S_obs : Float
  /- ── Steps channel (log-Gaussian) ── -/
  mu_step0  : Float
  beta_B_st : Float
  beta_F_st : Float
  beta_A_st : Float
  beta_C_st : Float
  sigma_st  : Float
  /- ── VolumeLoad channel ── -/
  beta_S_VL : Float
  beta_F_VL : Float
  sigma_VL  : Float
deriving Repr

/-! ## Channel means

Each function takes the current 6D state, the circadian forcing `C(t)`,
and the obs parameters; returns a deterministic scalar (mean or
probability). -/

/-- HR mean — sleep-active channel. Mirrors `simulation.py:265`:
    `mu_HR = HR_base - kappa_B_HR * B + alpha_A_HR * A + beta_C_HR * C`. -/
def hrMean (y : State6D) (C : Float) (op : ObsParams) : Float :=
  op.HR_base - op.kappa_B_HR * y.B + op.alpha_A_HR * y.A + op.beta_C_HR * C

/-- Sleep probability (Bernoulli p) — logistic of `k_C*C + k_A*A - c_tilde`.
    Mirrors `simulation.py:244-246`. -/
def sleepProb (y : State6D) (C : Float) (op : ObsParams) : Float :=
  let z := op.k_C * C + op.k_A * y.A - op.c_tilde
  1.0 / (1.0 + Float.exp (-z))

/-- Stress mean — wake-only channel. Mirrors `simulation.py:278`:
    `mu_S = S_base + k_F * F - k_A_S * A + beta_C_S * C`. -/
def stressMean (y : State6D) (C : Float) (op : ObsParams) : Float :=
  op.S_base + op.k_F * y.F - op.k_A_S * y.A + op.beta_C_S * C

/-- Steps log-mean (log-Gaussian channel). Mirrors `simulation.py:292`:
    `mu_log = mu_step0 + beta_B_st*B - beta_F_st*F + beta_A_st*A + beta_C_st*C`. -/
def stepsLogMean (y : State6D) (C : Float) (op : ObsParams) : Float :=
  op.mu_step0 + op.beta_B_st * y.B - op.beta_F_st * y.F
    + op.beta_A_st * y.A + op.beta_C_st * C

/-- VolumeLoad mean. Mirrors `simulation.py:305`:
    `mu_VL = beta_S_VL * S - beta_F_VL * F`. No circadian dependence. -/
def volumeLoadMean (y : State6D) (op : ObsParams) : Float :=
  op.beta_S_VL * y.S - op.beta_F_VL * y.F

end Fsa.V5
