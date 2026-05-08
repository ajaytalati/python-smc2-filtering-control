import Fsa.V15.Types

/-!
# FSA v1.5 estimation — obs log-weight + prior-application

Maps to `version_1_5_Julia/models/fsa_high_res/estimation.jl`.

★ **@match site #3** from writeup §7.1 (`apply_prior`): two prior
families (`:LogNormal` and `:Normal`) dispatched on `PriorKind`.

Pure functions only. The Julia `propagate(particles, …)` is just a
mapped EM step over an array; we don't re-implement the array layer
here — the diff-test exercises one particle at a time via
`Plant.plant_step`. The aggregate "propagate" is a thin Julia loop and
its correctness follows from `plant_step`'s correctness.
-/

namespace Fsa.V15

/-- Apply a prior to one unconstrained sample.

    ★ @match site #3:
    ```
    Julia (Match.jl):                        Lean4 (this file):
    @match kind begin                        match kind with
        :LogNormal => exp(clamp(u, -20, 20))  | .logNormal => Float.exp …
        :Normal    => mu + sigma * u          | .normal    => mu + sigma * u
    end
    ```

    `mu` and `sigma` are the prior mean/std in unconstrained space.
    For `.logNormal`, the result is `exp(clamp(u, -20, 20))` per
    `gpu_pf.jl:151-164`'s `_to_v15_constrained_nt`; the (mu, sigma)
    of the prior are NOT applied multiplicatively in the LogNormal
    arm — they parameterise the unconstrained sampling distribution
    upstream. -/
def apply_prior (kind : PriorKind) (mu sigma u : Float) : Float :=
  match kind with
  | .logNormal => Float.exp (min 20.0 (max (-20.0) u))
  | .normal    => mu + sigma * u

/-- Per-particle Gaussian log-weight given an observation and one
    particle's `(B, F, A)`. Maps to the inner body of
    `estimation.jl:91-120` (`obs_log_weight`).

    Returns
    `Σ_X [ −0.5·log(2π) − log σ_X − 0.5·((obs_X − X)/σ_X)² ]`
    over X ∈ {B, F, A}. Pure. -/
def obs_log_weight_one
    (B F A : Float) (o : Obs) (op : ObsNoiseParams) : Float :=
  let σB := op.sigma_B_obs
  let σF := op.sigma_F_obs
  let σA := op.sigma_A_obs
  let log2pi : Float := 1.8378770664093453   -- log (2 π)
  let log_norm := -1.5 * log2pi - Float.log σB - Float.log σF - Float.log σA
  let dB := o.obs_B - B
  let dF := o.obs_F - F
  let dA := o.obs_A - A
  let inv2sB2 := 0.5 / (σB * σB)
  let inv2sF2 := 0.5 / (σF * σF)
  let inv2sA2 := 0.5 / (σA * σA)
  log_norm - inv2sB2 * dB * dB - inv2sF2 * dF * dF - inv2sA2 * dA * dA

end Fsa.V15
