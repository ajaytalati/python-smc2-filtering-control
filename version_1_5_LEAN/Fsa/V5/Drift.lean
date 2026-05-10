import Fsa.V5.Types

/-!
# FSA-v5 deterministic drift

Single source of truth for the FSA-v5 SDE deterministic drift, transcribed
line-by-line from the canonical Python at
`models/fsa_high_res/_dynamics.py:177-258` (function `drift_jax`) and the
LaTeX equations at
`LaTex_docs/sections/14_v5_sedentary_collapse.tex:14-35` (eq `v5-mubar`).

Per the LEAN4-first charter (`LaTex_docs/lean4_first_charter.pdf` §5):
this file is both the formal specification AND the executable reference.
The Python `drift_jax` is differentially tested against the LEAN binary
produced from this file. Disagreement beyond `1e-6` is by construction
a Python bug.

The implementation uses Lean's built-in `Float` type (IEEE-754 double),
mirroring the Python's `float64` numerics exactly. Mathlib's `ℝ` is
available for theorem proving (deferred per charter §5 step 5).
-/

namespace Fsa.V5

/-- FSA-v5 deterministic drift `f : (state, params, Phi) → ẏ`.

Maps to `_dynamics.py:177-258`. The 6D state derivative is computed
from the bifurcation parameter `mu(B,S,F)` (with v5 Hill subtractions),
then plugged into the Stuart-Landau equation for `A`, the Banister
equations for `B` and `S`, the unified-fatigue equation for `F`, and
the Busso variable-dose K dynamics.

Setting `params.mu_dec_B = params.mu_dec_S = 0` recovers FSA-v4
exactly — the v5 Hill terms vanish identically. -/
def drift (y : State6D) (p : Params) (phi : BimodalPhi) : State6D :=
  let B   := y.B
  let S   := y.S
  let F   := y.F
  let A   := y.A
  let KFB := y.KFB
  let KFS := y.KFS
  let phi_B := phi.Phi_B
  let phi_S := phi.Phi_S

  -- ── Bifurcation parameter mu(B, S, F) — FSA-v5 ───────────────────────
  -- See LaTeX §10.2 eq:v5-mubar; Python `_dynamics.py:213-229`.
  --   mu_v5 = mu_v4(B,S,F)
  --         - mu_dec_B * B_dec^n / (B^n + B_dec^n)
  --         - mu_dec_S * S_dec^n / (S^n + S_dec^n)
  -- Each Hill term is in [0, mu_dec_i]: at x=0 saturates to mu_dec_i,
  -- at x=x_dec equals mu_dec_i/2, at x≫x_dec vanishes. C^∞ smooth.
  let F_dev := F - F_TYP
  let n     := p.n_dec
  let Bn    := Float.pow (max B 0.0) n
  let Sn    := Float.pow (max S 0.0) n
  let Bdn   := Float.pow p.B_dec n
  let Sdn   := Float.pow p.S_dec n
  let dec_B := p.mu_dec_B * Bdn / (Bn + Bdn)
  let dec_S := p.mu_dec_S * Sdn / (Sn + Sdn)

  let mu := p.mu_0
          + p.mu_B * B
          + p.mu_S * S
          - p.mu_F * F
          - p.mu_FF * F_dev * F_dev
          - dec_B
          - dec_S

  -- ── Aerobic capacity B (Banister; autonomic-modulated gain) ──────────
  -- `_dynamics.py:233-234`.
  let a_factor_B := (1.0 + p.epsilon_AB * A) / (1.0 + p.epsilon_AB * A_TYP)
  let dB         := p.kappa_B * a_factor_B * phi_B - B / p.tau_B

  -- ── Strength capacity S ──────────────────────────────────────────────
  -- `_dynamics.py:238-239`.
  let a_factor_S := (1.0 + p.epsilon_AS * A) / (1.0 + p.epsilon_AS * A_TYP)
  let dS         := p.kappa_S * a_factor_S * phi_S - S / p.tau_S

  -- ── Unified fatigue F (FSA-v4 dynamic gains) ─────────────────────────
  -- `_dynamics.py:244-245`.
  let a_factor_F := (1.0 + p.lambda_A * A) / (1.0 + p.lambda_A * A_TYP)
  let dF         := KFB * phi_B + KFS * phi_S - a_factor_F / p.tau_F * F

  -- ── Autonomic amplitude A (Stuart-Landau) ────────────────────────────
  -- `_dynamics.py:250`.
  let dA := mu * A - p.eta * A * A * A

  -- ── Busso variable-dose K dynamics (FSA-v4) ──────────────────────────
  -- `_dynamics.py:255-256`. Linear relaxation toward baseline K_{Fi}^0
  -- plus stimulus damage at rate mu_K. Slow-manifold equilibrium is
  -- K_{Fi}^* = K_{Fi}^0 + tau_K * mu_K * Phi_i.
  let dKFB := (p.KFB_0 - KFB) / p.tau_K + p.mu_K * phi_B
  let dKFS := (p.KFS_0 - KFS) / p.tau_K + p.mu_K * phi_S

  { B := dB, S := dS, F := dF, A := dA, KFB := dKFB, KFS := dKFS }

/-- Diffusion vector, state-dependent diagonal (6D).

Maps to `_dynamics.py:265-287` (`diffusion_state_dep`).

  - B, S use Jacobi-style `σ √(x(1-x))`, keeping x ∈ [0, 1].
  - F, A, K use CIR-style `σ √x`, keeping x ≥ 0.
  - K_FB and K_FS share the single scale `sigma_K` (per the canonical
    Python: empirically justified — K dynamics are slow). -/
def diffusion (y : State6D) (p : Params) : State6D :=
  let B   := y.B
  let S   := y.S
  let F   := y.F
  let A   := y.A
  let KFB := y.KFB
  let KFS := y.KFS
  { B   := p.sigma_B * Float.sqrt (max (B * (1.0 - B)) 0.0),
    S   := p.sigma_S * Float.sqrt (max (S * (1.0 - S)) 0.0),
    F   := p.sigma_F * Float.sqrt (max F 0.0),
    A   := p.sigma_A * Float.sqrt (max A 0.0),
    KFB := p.sigma_K * Float.sqrt (max KFB 0.0),
    KFS := p.sigma_K * Float.sqrt (max KFS 0.0) }

end Fsa.V5
