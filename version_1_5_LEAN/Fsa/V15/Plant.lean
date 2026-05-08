import Fsa.V15.Types
import Fsa.V15.Truth
import Fsa.V15.Dynamics
import Fsa.V15.Adapters

/-!
# FSA v1.5 plant — purely-functional Euler-Maruyama step + stride rollout

Maps to `version_1_5_Julia/models/fsa_high_res/_plant.jl`.

`reflect_unit` (★ @match site #2) lives in `Dynamics.lean` because
boundary reflection is conceptually part of the EM step. Re-exported
here for symmetry with the Julia module layout.

The Julia plant takes a UInt64 RNG `key` and derives sub-keys via
`hash((key, :obs))` and `hash((key0, :step, k))`. Lean4 has no clean
equivalent to Julia's `hash`, so the diff-test driver passes
*pre-drawn standard-normal noise* directly. Both implementations stay
deterministic and bit-identical for the same noise input.
-/

namespace Fsa.V15

/-- Build a fresh `PlantState` at the canonical init. Maps to
    `_plant.jl:43-50` (`init_plant_state`). -/
def init_plant_state : PlantState := INIT_STATE

/-- Sample one Gaussian obs per latent given pre-drawn noise.
    Maps to `simulation.jl:189-201` (`sample_obs_bfa`), but with the
    RNG key replaced by an explicit `(ξ_B, ξ_F, ξ_A)` standard-normal
    triple supplied by the caller (the diff-test driver). -/
def sample_obs_bfa
    (s : PlantState) (op : ObsNoiseParams)
    (noise : Float × Float × Float) : Obs :=
  let (ξB, ξF, ξA) := noise
  { obs_B := s.B + op.sigma_B_obs * ξB,
    obs_F := s.F + op.sigma_F_obs * ξF,
    obs_A := s.A + op.sigma_A_obs * ξA }

/-- One Euler-Maruyama step under control `Φ_t` from state `s`.
    Returns the new `PlantState` and the sampled `Obs`.
    Maps to `_plant.jl:75-107` (`plant_step`).

    The Julia source draws a 3-vector of standard normals via
    `randn(rng, 3)` for the SDE step and a separate 3-vector for the
    obs sample (with `hash((key, :obs))` as the seed). Here we take
    both as caller-supplied pre-drawn triples. -/
def plant_step
    (s : PlantState) (Φ_t : Float)
    (p : Params_v15) (op : ObsNoiseParams) (dt : Float)
    (sde_noise : Float × Float × Float)
    (obs_noise : Float × Float × Float) : PlantState × Obs :=
  let p_v1   := params_v15_to_v1_nt p
  let (dB, dF, dA) := drift s p_v1 Φ_t
  let (σB, σF, σA) := diffusion_state_dep s p_v1
  let (ξB, ξF, ξA) := sde_noise
  let s_dt := Float.sqrt dt
  let B_pred := s.B + dB * dt + σB * s_dt * ξB
  let F_pred := s.F + dF * dt + σF * s_dt * ξF
  let A_pred := s.A + dA * dt + σA * s_dt * ξA
  let s_next : PlantState := {
    B     := reflect_unit B_pred,
    F     := Float.abs F_pred,
    A     := Float.abs A_pred,
    t_bin := s.t_bin + 1
  }
  let obs := sample_obs_bfa s_next op obs_noise
  (s_next, obs)

end Fsa.V15
