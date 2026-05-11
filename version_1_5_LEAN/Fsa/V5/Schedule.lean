import Fsa.V5.Types

/-!
# RBF schedule decoder for the FSA-v5 controller

Transcribed from `models/fsa_high_res/control.py:51-68` (function
`_make_schedule`) and the `smc2fc.control.RBFSchedule` design-matrix
construction at `python-smc2-filtering-control/smc2fc/control/rbf_schedules.py:41-49`.

Per the LEAN4-first charter (`LaTex_docs/lean4_first_charter.pdf`):
this file contributes to the structural prevention of Bug 5 (the
historical θ scalar/vector confusion). The controller's RBF
coefficients live in their own array — distinct from the SMC²
parameter posterior, which is a separate `Params` structure. Two
objects with the type signature `RBFCoeffs n_anchors` and `Params`
are not unifiable; an attempt to compare them as scalar intervals
would not type-check.
-/

namespace Fsa.V5

/-! ## Type aliases — distinct from `Params` and from regions in Φ-space

`RBFCoeffs` is the controller's decision variable, length `2 *
n_anchors` (first half are aerobic-channel basis weights, second half
strength-channel). The historical bug compared a 16-dim vector of
this type against a 2-D region in `(Φ_B, Φ_S)`-space as if both were
scalars; the type alias here is documentation that they are different
kinds of objects. -/
abbrev RBFCoeffs := Array Float

/-! Per-bin bimodal stimulus output of the schedule decoder. The
controller's HMC operates over `RBFCoeffs`; the cost evaluator and
plant consume `Schedule` (an array of `BimodalPhi`). Two distinct
roles, two distinct type names.
-/
abbrev Schedule := Array BimodalPhi

/-- Inverse-sigmoid bias for a default `Φ` mid-range. Mirrors
    `c_Phi = log(Phi_default / (Phi_max - Phi_default))` at
    `control.py:57-58`. With this bias, `sigmoid(c_Phi) * Phi_max
    = Phi_default` (i.e. `θ = 0` decodes to the default Φ). -/
def c_phi (phi_default phi_max : Float) : Float :=
  let p := phi_default / phi_max
  Float.log (p / (1.0 - p))

/-- Decode an `RBFCoeffs` vector `θ` (length `2 * n_anchors`) plus a
    pre-computed `Phi_design` matrix `(n_steps, n_anchors)` into a
    bimodal `Schedule` of length `n_steps`.

    Mirrors the body of `schedule_from_theta` at
    `control.py:60-68`. The two channels (`B` and `S`) share the
    same design matrix; only the coefficient slice differs. -/
def scheduleFromTheta
    (theta : RBFCoeffs)
    (phiDesign : Array (Array Float))   -- shape (n_steps, n_anchors)
    (cPhi : Float)
    (phiMax : Float)
    (n_anchors : Nat)
    : Schedule :=
  phiDesign.map (fun row =>
    let raw_B := Id.run do
      let mut acc : Float := cPhi
      for a in [0:n_anchors] do
        acc := acc + theta[a]! * row[a]!
      return acc
    let raw_S := Id.run do
      let mut acc : Float := cPhi
      for a in [0:n_anchors] do
        acc := acc + theta[n_anchors + a]! * row[a]!
      return acc
    { Phi_B := phiMax * sigmoid raw_B,
      Phi_S := phiMax * sigmoid raw_S })

/-! ## Design-matrix construction

`designMatrix n_steps dt n_anchors width_factor` builds the Gaussian
RBF design matrix mirroring `RBFSchedule.design_matrix` at
`smc2fc/control/rbf_schedules.py:41-49`.

Centres are `n_anchors` evenly-spaced points in `[0, T_total]` where
`T_total = n_steps * dt`. Width is `(T_total / max(n_anchors, 1))
* width_factor`. The basis is `Phi[t, a] = exp(-0.5 * ((t·dt -
centre_a) / width)^2)`. -/

private def linspace (a b : Float) (n : Nat) : Array Float :=
  if n = 0 then #[]
  else if n = 1 then #[a]
  else
    let step := (b - a) / (n - 1).toFloat
    Array.range n |>.map (fun i => a + i.toFloat * step)

/-- Build the Gaussian RBF design matrix. Returns shape
    `(n_steps, n_anchors)`. -/
def designMatrix (n_steps : Nat) (dt : Float)
    (n_anchors : Nat) (width_factor : Float) : Array (Array Float) :=
  let T_total := n_steps.toFloat * dt
  let centres := linspace 0.0 T_total n_anchors
  let denom : Float := if n_anchors = 0 then 1.0 else n_anchors.toFloat
  let width := (T_total / denom) * width_factor
  let t_grid := Array.range n_steps |>.map (fun i => i.toFloat * dt)
  t_grid.map (fun t =>
    centres.map (fun c =>
      let z := (t - c) / width
      Float.exp (-0.5 * z * z)))

end Fsa.V5
