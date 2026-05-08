import Fsa.V15.Types

/-!
# FSA v1.5 truth and default parameter sets

Maps to `_dynamics.jl:26-48` (`TRUTH_PARAMS` v1 form) and
`simulation.jl:43-85` (`DEFAULT_PARAMS` v1.5 form, `INIT_STATE`,
`PINNED_PARAMS`).

These constants are the diff-test seed values: the Julia harness
loads its `Simulation.DEFAULT_PARAMS`, the Lean4 binary uses
`Truth.DEFAULT_PARAMS_V15`, and they MUST be bit-identical at every
field. Because the v1.5 basis is a closed-form rotation of the v1
basis (`B_inf := kappa_B · tau_B`, `F_inf := kappa_F · tau_F`),
both the v1 and v1.5 forms are derivable from one source set.
-/

namespace Fsa.V15

/-- v1-form truth params, verbatim from `_dynamics.jl:26-48`. -/
def TRUTH_PARAMS_V1 : Params_v1 := {
  tau_B     := 42.0,
  tau_F     :=  7.0,
  kappa_B   := 0.012,
  kappa_F   := 0.030,
  epsilon_A := 0.40,
  lambda_A  := 1.00,
  mu_0      := 0.02,
  mu_B      := 0.30,
  mu_F      := 0.10,
  mu_FF     := 0.40,
  eta       := 0.20,
  sigma_B   := 0.010,
  sigma_F   := 0.012,
  sigma_A   := 0.020
}

/-- v1.5-form default params; basis-rotated from v1.
    Maps to `simulation.jl:43-67` dynamics fields. -/
def DEFAULT_PARAMS_V15 : Params_v15 := {
  tau_B     := TRUTH_PARAMS_V1.tau_B,
  tau_F     := TRUTH_PARAMS_V1.tau_F,
  B_inf     := TRUTH_PARAMS_V1.kappa_B * TRUTH_PARAMS_V1.tau_B,    -- 0.504
  F_inf     := TRUTH_PARAMS_V1.kappa_F * TRUTH_PARAMS_V1.tau_F,    -- 0.21
  epsilon_A := TRUTH_PARAMS_V1.epsilon_A,
  lambda_A  := TRUTH_PARAMS_V1.lambda_A,
  mu_0      := TRUTH_PARAMS_V1.mu_0,
  mu_B      := TRUTH_PARAMS_V1.mu_B,
  mu_F      := TRUTH_PARAMS_V1.mu_F,
  mu_FF     := TRUTH_PARAMS_V1.mu_FF,
  eta       := TRUTH_PARAMS_V1.eta,
  sigma_B   := TRUTH_PARAMS_V1.sigma_B,
  sigma_F   := TRUTH_PARAMS_V1.sigma_F,
  sigma_A   := TRUTH_PARAMS_V1.sigma_A
}

/-- Pinned obs-noise (all 0.005 in v1.5). Maps to
    `simulation.jl:64-66`. -/
def DEFAULT_OBS_NOISE : ObsNoiseParams := {
  sigma_B_obs := 0.005,
  sigma_F_obs := 0.005,
  sigma_A_obs := 0.005
}

/-- The 4 pinned dynamics params. Maps to `simulation.jl:80-85`. -/
def DEFAULT_PINNED : PinnedDynamics := {
  tau_B     := DEFAULT_PARAMS_V15.tau_B,
  eta       := DEFAULT_PARAMS_V15.eta,
  epsilon_A := DEFAULT_PARAMS_V15.epsilon_A,
  mu_FF     := DEFAULT_PARAMS_V15.mu_FF
}

/-- Canonical initial state, shared with v1 and v2. Maps to
    `simulation.jl:69` (`INIT_STATE = (B = 0.05, F = 0.30, A = 0.10)`). -/
def INIT_STATE : PlantState := {
  B := 0.05, F := 0.30, A := 0.10, t_bin := 0
}

end Fsa.V15
