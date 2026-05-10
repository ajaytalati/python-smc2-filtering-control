import Fsa.V5.Types
import Fsa.V5.Drift

/-!
# FSA-v5 plant — deterministic Euler-Maruyama step

Single source of truth for the plant's per-bin Euler-Maruyama
integrator, transcribed from `models/fsa_high_res/_plant.py:84-150`
(function `_plant_em_step`). The transcription separates the
mathematical update from the JAX RNG: noise is taken as an explicit
input so the function is purely deterministic and amenable to
differential testing against the Python implementation.

Step formula (per bin):
  d_y     = drift(y, p, Φ)
  g(y)    = state-dependent diffusion vector (Jacobi for B,S; CIR for F,A,K_*)
  y_next  = y + dt · d_y + σ_diag ⊙ g(y) ⊙ √dt · noise
  y_next  := clip/floor to physical domain (B,S in [ε, 1-ε]; F,A,K ≥ 0)

Per the LEAN4-first charter (`LaTex_docs/lean4_first_charter.pdf`):
this is the executable reference. The Python differential test passes
the SAME noise vector to both implementations and asserts agreement
within the tolerance threshold.
-/

namespace Fsa.V5

/-- Clipping/floor epsilons mirror `_plant.py:104-106`. -/
private def EPS_B : Float := 1.0e-4
private def EPS_S : Float := 1.0e-4
private def EPS_A : Float := 1.0e-4

/-- Standard-Float clamp helper. -/
@[inline] private def clamp (lo hi x : Float) : Float :=
  if x < lo then lo else if x > hi then hi else x

/-- One Euler-Maruyama step at a single bin.

  - `y`         : current state (6D).
  - `phi`       : per-bin stimulus.
  - `p`         : truth parameters.
  - `sigmaDiag` : 6-vector of diffusion scales (typically frozen
                  to ``params.sigma_*``; passed as input here so the
                  caller can override for sensitivity tests).
  - `dt`        : bin width in days (e.g. ``DT_BIN_DAYS = 1/96``).
  - `noise`     : 6-vector of standard-normal samples drawn upstream.

Returns the next state. Mirrors `_plant.py:108-146`. -/
def emStep (y : State6D) (phi : BimodalPhi) (p : Params)
    (sigmaDiag : Array Float) (dt : Float) (noise : Array Float)
    : State6D :=
  let d_y := drift y p phi
  let sqrt_dt := Float.sqrt dt
  -- State-dependent diffusion magnitudes (Jacobi for B,S; CIR for F,A,K_*).
  let B_cl   := clamp EPS_B (1.0 - EPS_B) y.B
  let S_cl   := clamp EPS_S (1.0 - EPS_S) y.S
  let F_cl   := max y.F 0.0
  let A_cl   := max y.A 0.0
  let KFB_cl := max y.KFB 0.0
  let KFS_cl := max y.KFS 0.0
  let g_B   := Float.sqrt (B_cl * (1.0 - B_cl))
  let g_S   := Float.sqrt (S_cl * (1.0 - S_cl))
  let g_F   := Float.sqrt F_cl
  let g_A   := Float.sqrt (A_cl + EPS_A)
  let g_KFB := Float.sqrt KFB_cl
  let g_KFS := Float.sqrt KFS_cl
  -- y_next = y + dt * drift + sigma_diag * g(y) * sqrt(dt) * noise
  let yB_next   := y.B   + dt * d_y.B   + sigmaDiag[0]! * g_B   * sqrt_dt * noise[0]!
  let yS_next   := y.S   + dt * d_y.S   + sigmaDiag[1]! * g_S   * sqrt_dt * noise[1]!
  let yF_next   := y.F   + dt * d_y.F   + sigmaDiag[2]! * g_F   * sqrt_dt * noise[2]!
  let yA_next   := y.A   + dt * d_y.A   + sigmaDiag[3]! * g_A   * sqrt_dt * noise[3]!
  let yKFB_next := y.KFB + dt * d_y.KFB + sigmaDiag[4]! * g_KFB * sqrt_dt * noise[4]!
  let yKFS_next := y.KFS + dt * d_y.KFS + sigmaDiag[5]! * g_KFS * sqrt_dt * noise[5]!
  -- Boundary handling — clip/floor back to physical domain.
  { B   := clamp EPS_B (1.0 - EPS_B) yB_next,
    S   := clamp EPS_S (1.0 - EPS_S) yS_next,
    F   := max yF_next 0.0,
    A   := max yA_next 0.0,
    KFB := max yKFB_next 0.0,
    KFS := max yKFS_next 0.0 }

end Fsa.V5
