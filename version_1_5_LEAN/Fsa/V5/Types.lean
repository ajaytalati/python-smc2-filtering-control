/-!
# FSA-v5 typed declarations

Single source of truth for the FSA-v5 state, control, and parameter types.
Cross-references the LaTeX in `LaTex_docs/sections/14_v5_sedentary_collapse.tex`
and the canonical Python in `models/fsa_high_res/_dynamics.py` (lines 81-170).

Per the LEAN4-first charter (`LaTex_docs/lean4_first_charter.pdf`):
each field is a named record member, not a positional array index, and not
a string key into a dict. Two fields with the same name is a compile error
— the historical `sigma_S` dict-key collision (latent state-noise vs
stress-channel obs noise, silently kept the wrong value in Python) is
literally impossible to express in the type system below.
-/

namespace Fsa.V5

/-- 6D state vector ``y = [B, S, F, A, K_FB, K_FS]^T``.
    Maps to the Python tuple destructuring at ``_dynamics.py:201``. -/
structure State6D where
  /-- Aerobic fitness, Banister chronic, Jacobi diffusion in [0, 1]. -/
  B   : Float
  /-- Strength capacity, Banister chronic, Jacobi diffusion in [0, 1]. -/
  S   : Float
  /-- Unified fatigue pool, CIR diffusion in [0, ∞). -/
  F   : Float
  /-- Autonomic amplitude, Stuart-Landau, CIR diffusion in [0, ∞). -/
  A   : Float
  /-- Aerobic fatigue gain (Busso variable), CIR diffusion in [0, ∞). -/
  KFB : Float
  /-- Strength fatigue gain (Busso variable), CIR diffusion in [0, ∞). -/
  KFS : Float
deriving Repr

/-- 2D bimodal stimulus rate ``Φ = [Φ_B, Φ_S]``. Dimensionless.
    Matches the Python destructuring at ``_dynamics.py:202``. -/
structure BimodalPhi where
  /-- Aerobic stimulus rate. -/
  Phi_B : Float
  /-- Strength stimulus rate. -/
  Phi_S : Float
deriving Repr

/-- Full FSA-v5 parameter set with NAMED fields.

    The historical Python dict had two `'sigma_S'` keys; here the
    state-noise scale (`sigma_S`) and the stress-channel obs noise
    (`sigma_S_obs`) are distinct fields by construction — the bug
    cannot be written. Obs-channel parameters are kept on a separate
    record (see future `ObsParams` once the Obs module lands).

    All scalar parameters of the v5 SDE drift live here. Diffusion
    scales (`sigma_*`) are listed for completeness but are FROZEN in
    the production estimation pipeline — see `estimation.py`. -/

structure Params where
  /- ── Aerobic Fitness B (linear in Phi_B, decays with τ_B) ── -/
  tau_B      : Float
  kappa_B    : Float    -- κ_B^eff in G1 form
  epsilon_AB : Float    -- autonomic boost coefficient
  /- ── Strength Adaptation S ── -/
  tau_S      : Float
  kappa_S    : Float    -- κ_S^eff in G1 form
  epsilon_AS : Float
  /- ── Unified Fatigue F (driven by K_FB Phi_B + K_FS Phi_S) ── -/
  tau_F      : Float    -- τ_F^eff in G1 form
  lambda_A   : Float    -- autonomic-fatigue coupling
  /- ── Busso Variable-Dose Sensitivity K_FB, K_FS (FSA-v4) ── -/
  KFB_0  : Float        -- baseline aerobic fatigue gain
  KFS_0  : Float        -- baseline strength fatigue gain
  tau_K  : Float        -- ~3 weeks recovery timescale
  mu_K   : Float        -- 'damage' rate (Busso 2003)
  /- ── Stuart-Landau bifurcation parameter mu(B, S, F) ── -/
  mu_0   : Float        -- baseline autonomic drive
  mu_B   : Float        -- B → A coupling (positive)
  mu_S   : Float        -- S → A coupling (positive)
  mu_F   : Float        -- F → A coupling (negative)
  mu_FF  : Float        -- quadratic F penalty around F_TYP
  eta    : Float        -- cubic damping in A
  /- ── State-dependent diffusion scales (frozen in production) ── -/
  sigma_B : Float       -- Jacobi scale for B
  sigma_S : Float       -- Jacobi scale for S — the LATENT state-noise
                        -- (NOT the stress obs noise, which lives in ObsParams)
  sigma_F : Float       -- CIR scale for F
  sigma_A : Float       -- CIR scale for A
  sigma_K : Float       -- shared CIR scale for K_FB and K_FS
  /- ── FSA-v5 Hill deconditioning (silent when mu_dec_* = 0) ── -/
  B_dec    : Float      -- aerobic-fitness threshold
  S_dec    : Float      -- strength threshold
  mu_dec_B : Float      -- aerobic decond penalty (0 = v4 numerics)
  mu_dec_S : Float      -- strength decond penalty (0 = v4 numerics)
  n_dec    : Float      -- Hill exponent (steepness; default 4.0)
deriving Repr

/-! ## Operating-point reference constants

Maps to `_dynamics.py:81-83`. These are the "typical" values around
which the G1 reparametrisation is centred. -/

/-- Typical autonomic amplitude (dimensionless). -/
def A_TYP : Float := 0.10

/-- Typical unified fatigue (dimensionless). -/
def F_TYP : Float := 0.20

/-- Typical stimulus rate (dimensionless). -/
def PHI_TYP : Float := 1.0

/-- Standard logistic sigmoid `σ(x) = 1 / (1 + e^{-x})`. Mirrors
    `jax.nn.sigmoid`. -/
@[inline] def sigmoid (x : Float) : Float :=
  1.0 / (1.0 + Float.exp (-x))

/-! ## Canonical truth-parameter sets

Maps to `_dynamics.py:96-170`. `TRUTH_PARAMS` defaults to v4-recovering
numerics (`mu_dec_B = mu_dec_S = 0`). `TRUTH_PARAMS_V5` overrides with the
scanned values that produce the closed-island basin topology of §10.4. -/

/-- v4-recovering defaults (mu_dec_* = 0). Maps to `_dynamics.py:96-141`. -/
def TRUTH_PARAMS : Params := {
  -- Aerobic Fitness B
  tau_B      := 42.0,
  kappa_B    := 0.012 * (1.0 + 0.40 * A_TYP),
  epsilon_AB := 0.40,
  -- Strength Adaptation S
  tau_S      := 60.0,
  kappa_S    := 0.008 * (1.0 + 0.20 * A_TYP),
  epsilon_AS := 0.20,
  -- Unified Fatigue F
  tau_F      := 7.0 / (1.0 + 1.00 * A_TYP),
  lambda_A   := 1.00,
  -- Busso Variable-Dose K
  KFB_0  := 0.030,
  KFS_0  := 0.050,
  tau_K  := 21.0,
  mu_K   := 0.005,
  -- Stuart-Landau bifurcation parameter
  mu_0   := 0.02 + 0.40 * (F_TYP * F_TYP),
  mu_B   := 0.30,
  mu_S   := 0.15,
  mu_F   := 0.10 + 2.0 * F_TYP * 0.40,
  mu_FF  := 0.40,
  eta    := 0.20,
  -- Diffusion scales (frozen)
  sigma_B := 0.010,
  sigma_S := 0.008,
  sigma_F := 0.012,
  sigma_A := 0.020,
  sigma_K := 0.005,
  -- v5 Hill deconditioning (silent in v4)
  B_dec    := 0.10,
  S_dec    := 0.10,
  mu_dec_B := 0.0,
  mu_dec_S := 0.0,
  n_dec    := 4.0
}

/-- FSA-v5 truth (closed-island basin topology). Maps to
    `_dynamics.py:163-170`. -/
def TRUTH_PARAMS_V5 : Params := { TRUTH_PARAMS with
  B_dec    := 0.07,
  S_dec    := 0.07,
  mu_dec_B := 0.10,
  mu_dec_S := 0.10,
  n_dec    := 4.0
}

end Fsa.V5
