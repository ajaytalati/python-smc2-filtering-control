import Fsa.V15.Types

/-!
# FSA v1.5 deterministic drift + state-dependent diffusion + EM step

Single source of truth for the v1.5 SDE drift, transcribed line-by-line
from `version_1_5_Julia/models/fsa_high_res/_dynamics.jl:51-119`. The
math:

  μ(B, F)  =  μ_0 + μ_B·B − μ_F·F − μ_FF·F²
  dB/dt    =  κ_B · (1 + ε_A·A) · Φ  −  B / τ_B
  dF/dt    =  κ_F · Φ                 −  (1 + λ_A·A) / τ_F · F
  dA/dt    =  μ · A − η · A³

  σ_B(B)   =  σ_B · √(B(1−B))    Jacobi
  σ_F(F)   =  σ_F · √F           CIR
  σ_A(A)   =  σ_A · √A           CIR

The implementation uses Lean4's built-in `Float` (IEEE-754 double),
mirroring Julia's `Float64` exactly. Boundary reflection enforces
B ∈ [0, 1] (reflect at both walls) and F, A ≥ 0 (`abs`).

These are pure functions of `(state, params, …)` — no allocation,
no RNG, no I/O. The JSON CLI bridge in `Main.lean` evaluates them
on stdin-supplied inputs and emits the result on stdout.
-/

namespace Fsa.V15

/-- Per-day drift `d[B, F, A]/dt` at state `y` under control `Φ_t`.
    Maps to `_dynamics.jl:58-67`. Returns the three derivatives as a
    triple to avoid allocating a Lean4 list per call. -/
def drift (y : PlantState) (p : Params_v1) (Φ_t : Float) : Float × Float × Float :=
  let B := y.B
  let F := y.F
  let A := y.A
  let mu := p.mu_0 + p.mu_B * B - p.mu_F * F - p.mu_FF * F * F
  let dB := p.kappa_B * (1.0 + p.epsilon_A * A) * Φ_t - B / p.tau_B
  let dF := p.kappa_F * Φ_t - (1.0 + p.lambda_A * A) / p.tau_F * F
  let dA := mu * A - p.eta * A * A * A
  (dB, dF, dA)

/-- State-dependent diagonal diffusion `σ(y) = [σ_B, σ_F, σ_A]`.
    Each component vanishes at its domain boundary so the SDE keeps
    each state in its physiological range without clipping.
    Maps to `_dynamics.jl:79-86`. -/
def diffusion_state_dep (y : PlantState) (p : Params_v1) : Float × Float × Float :=
  let B := y.B
  let F := y.F
  let A := y.A
  let σB := p.sigma_B * Float.sqrt (max (B * (1.0 - B)) 0.0)
  let σF := p.sigma_F * Float.sqrt (max F 0.0)
  let σA := p.sigma_A * Float.sqrt (max A 0.0)
  (σB, σF, σA)

/-- Reflect a real onto [0, 1] via two-wall reflection: x < 0 → −x;
    x > 1 → 2 − x; otherwise x. **★ @match site #2 from writeup §7.1**;
    Julia source `_plant.jl:57-61`. The Julia `@match x begin x, if x < 0
    end => -x; x, if x > 1 end => 2 - x; _ => x end` collapses cleanly
    to nested `if/else` — the @match dispatch and the if/else chain
    have the same evaluation semantics on Float. -/
def reflect_unit (x : Float) : Float :=
  if x < 0.0 then -x
  else if x > 1.0 then 2.0 - x
  else x

/-- Tail-recursive helper for `em_step_substepped`'s `n_substeps`
    deterministic Euler sub-steps. Returns the inner state after `n`
    sub-steps. -/
private partial def applySubsteps
    (y : PlantState) (p : Params_v1) (Φ_t : Float)
    (sub_dt : Float) (n : Nat) : PlantState :=
  match n with
  | 0     => y
  | n + 1 =>
    let (dB, dF, dA) := drift y p Φ_t
    let y' : PlantState := {
      B     := y.B + sub_dt * dB,
      F     := y.F + sub_dt * dF,
      A     := y.A + sub_dt * dA,
      t_bin := y.t_bin
    }
    applySubsteps y' p Φ_t sub_dt n

/-- Substepped Euler-Maruyama: `n_substeps` deterministic drift
    sub-steps then ONE Wiener increment of variance σ(y)²·dt at the
    outer boundary. Boundary reflection enforces B ∈ [0, 1] (two-wall
    reflect) and F, A ≥ 0 (abs).

    `noise = (ξ_B, ξ_F, ξ_A)` is a pre-drawn standard-normal triple
    (the diff-test driver provides it). Maps to `_dynamics.jl:100-119`.

    `t_bin` advances by 1 across one outer EM step. -/
def em_step_substepped
    (y : PlantState) (p : Params_v1)
    (noise : Float × Float × Float) (Φ_t : Float) (dt : Float)
    (n_substeps : Nat := 4) : PlantState :=
  let sub_dt := dt / n_substeps.toFloat
  let yi := applySubsteps y p Φ_t sub_dt n_substeps
  let (σB, σF, σA) := diffusion_state_dep yi p
  let (ξB, ξF, ξA) := noise
  let s_dt := Float.sqrt dt
  let B_pred := yi.B + σB * s_dt * ξB
  let F_pred := yi.F + σF * s_dt * ξF
  let A_pred := yi.A + σA * s_dt * ξA
  { B     := reflect_unit B_pred,
    F     := Float.abs F_pred,
    A     := Float.abs A_pred,
    t_bin := y.t_bin + 1 }

end Fsa.V15
