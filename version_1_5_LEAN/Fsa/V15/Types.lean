/-!
# FSA v1.5 typed declarations

Single source of truth for the FSA-v1.5 state, control, and parameter
types. Cross-references:

  - LaTeX writeup `version_1_5_Julia/docs/julia_fsa_v15_writeup.tex` §4.1
    (`_dynamics.jl` v1 verbatim) and §4.2 (`simulation.jl` NEW —
    parameter adapters in v1.5 basis).
  - Julia source `version_1_5_Julia/models/fsa_high_res/_dynamics.jl`
    (the verbatim v1 14-field NamedTuple `TRUTH_PARAMS`) and
    `simulation.jl` (the v1.5-basis 14-field `DEFAULT_PARAMS` Dict).

Two parameter parametrisations coexist:

  * **v1 form** (`Params_v1`): the original Banister/Stuart-Landau
    drift parameters with `kappa_B`, `kappa_F` as gain rates.
    Consumed by `Dynamics.drift` and `Dynamics.diffusion_state_dep`.
  * **v1.5 form** (`Params_v15`): re-parametrises (kappa_B, kappa_F) as
    (B_inf = kappa_B · tau_B, F_inf = kappa_F · tau_F) — the
    steady-state values at Φ=1. Consumed by the filter / FIM call
    sites. The adapter `Adapters.params_v15_to_v1` rotates the basis.

The two forms are kept as distinct Lean4 records so a function of
`Params_v15` cannot be silently passed `Params_v1` (or vice versa).
This is the structural fix for the FSA-v5 `theta` confusion documented
in the LEAN4-first charter §2.
-/

namespace Fsa.V15

/-- 3D plant state ``y = [B, F, A]^T``. Banister fitness, fatigue,
    and Stuart-Landau autonomic amplitude. Maps to
    `_plant.jl:33-36` (`struct PlantState`). -/
structure PlantState where
  /-- Aerobic fitness, Banister chronic, Jacobi diffusion in [0, 1]. -/
  B     : Float
  /-- Unified fatigue, Banister acute, CIR diffusion in [0, ∞). -/
  F     : Float
  /-- Autonomic amplitude, Stuart-Landau, CIR diffusion in [0, ∞). -/
  A     : Float
  /-- Global bin counter; advances by 1 per `plant_step`. -/
  t_bin : Nat
deriving Repr

/-- Per-bin observation tuple `(obs_B, obs_F, obs_A)`. Maps to the
    NamedTuple returned by `Plant.plant_step` and `Simulation.sample_obs_bfa`. -/
structure Obs where
  obs_B : Float
  obs_F : Float
  obs_A : Float
deriving Repr

/-- v1-form drift parameters (14 fields). Consumed by `Dynamics.drift`
    and `Dynamics.diffusion_state_dep`. Maps to the v1 `TRUTH_PARAMS`
    NamedTuple at `_dynamics.jl:26-48`. -/
structure Params_v1 where
  /- ── Banister timescales + gains ── -/
  tau_B    : Float
  tau_F    : Float
  kappa_B  : Float
  kappa_F  : Float
  /- ── A-coupling ── -/
  epsilon_A : Float
  lambda_A  : Float
  /- ── Stuart-Landau bifurcation parameter ── -/
  mu_0  : Float
  mu_B  : Float
  mu_F  : Float
  mu_FF : Float
  eta   : Float
  /- ── State-dependent diffusion scales ── -/
  sigma_B : Float
  sigma_F : Float
  sigma_A : Float
deriving Repr

/-- v1.5-form drift parameters (14 fields). Re-parametrises
    (kappa_B, kappa_F) as (B_inf, F_inf). Consumed by the filter /
    FIM-gate call sites. Maps to the v1.5 `DEFAULT_PARAMS` Dict at
    `simulation.jl:43-67` (dynamics part only — obs-noise lives
    separately on `ObsNoiseParams`). -/
structure Params_v15 where
  /- ── Banister timescales + steady-states ── -/
  tau_B  : Float
  tau_F  : Float
  B_inf  : Float
  F_inf  : Float
  /- ── A-coupling ── -/
  epsilon_A : Float
  lambda_A  : Float
  /- ── Stuart-Landau bifurcation parameter ── -/
  mu_0  : Float
  mu_B  : Float
  mu_F  : Float
  mu_FF : Float
  eta   : Float
  /- ── State-dependent diffusion scales ── -/
  sigma_B : Float
  sigma_F : Float
  sigma_A : Float
deriving Repr

/-- Pinned obs-noise parameters (3 fields). Maps to the obs-noise
    keys in `simulation.jl:64-66` (`sigma_B_obs`, `sigma_F_obs`,
    `sigma_A_obs`). All three pinned at 0.005 in v1.5. Kept on a
    SEPARATE record from `Params_v15` — the v1.5 codebase has no
    `sigma_B`/`sigma_B_obs` collision risk thanks to this split. -/
structure ObsNoiseParams where
  sigma_B_obs : Float
  sigma_F_obs : Float
  sigma_A_obs : Float
deriving Repr

/-- 10 estimated v1.5 params (the filter's decision variables).
    Maps to `estimation.jl:37-42` (`PARAM_NAMES`). The 4 pinned
    params (`tau_B`, `eta`, `epsilon_A`, `mu_FF`) are NOT here;
    they sit on `PinnedDynamics` and are inserted by `fill_pinned`. -/
structure EstimatedParams where
  tau_F    : Float
  B_inf    : Float
  F_inf    : Float
  lambda_A : Float
  mu_0     : Float
  mu_B     : Float
  mu_F     : Float
  sigma_B  : Float
  sigma_F  : Float
  sigma_A  : Float
deriving Repr

/-- The 4 pinned-at-truth dynamics parameters not estimated by the
    filter. Maps to `simulation.jl:80-85` (`PINNED_PARAMS`). -/
structure PinnedDynamics where
  tau_B     : Float
  eta       : Float
  epsilon_A : Float
  mu_FF     : Float
deriving Repr

/-! ## @match-site sum types

These inductive types make the three Match.jl `@match` sites in v1.5's
non-GPU code (writeup §7.1) into typed sum types. Each Lean4
`match … with` is the line-for-line translation of the corresponding
Julia `@match … begin … end`. -/

/-- @match site #1 (`simulation.jl:110-143`, `params_v15_to_v1_nt`).
    Discriminates the two parameter container shapes the Julia adapter
    accepts: a `Dict{Symbol, Float}` (truth values, plant) or a
    `NamedTuple` (filter / FIM call sites under ForwardDiff).

    The Julia `@match p begin ::Dict => …; ::NamedTuple => … end`
    becomes `match form with | .dictForm d => … | .ntForm nt => …`. -/
inductive ParamsForm where
  /-- Dict-form: a function lookup by symbol. Models the Julia
      `Dict{Symbol, Float64}`. The diff test exercises only the
      `ntForm` case (the production path); `dictForm` is preserved
      for symmetry with the Julia source but exercised lazily. -/
  | dictForm (lookup : String → Float)
  /-- NamedTuple-form: a fully-populated `Params_v15` record. -/
  | ntForm  (nt : Params_v15)

/-- @match site #3 (writeup §7.1, `apply_prior`). Discriminates the two
    prior families used in `PARAM_PRIOR_CONFIG`. -/
inductive PriorKind where
  /-- Lognormal prior: parameter = `exp(clamp(unconstrained, -20, 20))`. -/
  | logNormal
  /-- Normal prior: parameter = `mu + sigma · unconstrained`. -/
  | normal
deriving Repr

end Fsa.V15
