import Fsa.V15
import Lean.Data.Json

/-!
# CLI bridge for differential testing

Reads one JSON object per line on stdin, writes one JSON object per
line on stdout. Mirrors `FSA_model_dev/lean/Main.lean` for FSA-v5.

Dispatch tags:

  `drift`        — (state, phi, params_v1) → deriv [dB, dF, dA]
  `diffusion`    — (state, params_v1)      → sigma [σ_B, σ_F, σ_A]
  `emStep`       — (state, params_v1, noise, phi, dt, n_substeps)
                                            → next_state [B, F, A]
  `paramsV15ToV1` — (params_v15)            → params_v1 (14 fields)
  `reflectUnit`  — (x : Float)              → x' : Float
  `plantStep`    — (state, phi, params_v15, obs_noise_params,
                    dt, sde_noise, obs_noise) → (next_state, obs)
  `obsLogWeight` — (B, F, A, obs, obs_noise_params) → log_weight
  `applyPrior`   — (kind ∈ {logNormal, normal}, mu, sigma, u) → x
-/

open Lean (Json)
open Fsa.V15

private def jsonToFloat? (j : Json) : Except String Float :=
  match j with
  | .num n => Except.ok n.toFloat
  | _      => Except.error s!"expected JSON number, got {j.compress}"

private def getFloat (obj : Json) (key : String) : Except String Float := do
  let v ← obj.getObjVal? key
  jsonToFloat? v

private def getFloatArr (j : Json) : Except String (Array Float) := do
  let arr ← j.getArr?
  arr.mapM jsonToFloat?

private def getStringField (j : Json) (key : String) : Except String String := do
  j.getObjValAs? String key

private def getNatField (j : Json) (key : String) : Except String Nat := do
  j.getObjValAs? Nat key

/-- 3-element JSON array → `PlantState` (with `t_bin` defaulted to 0,
    since the diff test seeds state from the Julia driver bin-by-bin
    and t_bin is only used as a counter in the rollout). -/
private def getPlantStateFromArr (j : Json) : Except String PlantState := do
  let arr ← j.getArr?
  if arr.size != 3 then
    Except.error s!"state array must have 3 elements, got {arr.size}"
  else
    pure {
      B := (← jsonToFloat? arr[0]!),
      F := (← jsonToFloat? arr[1]!),
      A := (← jsonToFloat? arr[2]!),
      t_bin := 0
    }

private def getTripleFromArr (j : Json) : Except String (Float × Float × Float) := do
  let arr ← j.getArr?
  if arr.size != 3 then
    Except.error s!"triple array must have 3 elements, got {arr.size}"
  else
    pure ((← jsonToFloat? arr[0]!),
          (← jsonToFloat? arr[1]!),
          (← jsonToFloat? arr[2]!))

private def getParamsV1 (j : Json) : Except String Params_v1 := do
  pure {
    tau_B     := (← getFloat j "tau_B"),
    tau_F     := (← getFloat j "tau_F"),
    kappa_B   := (← getFloat j "kappa_B"),
    kappa_F   := (← getFloat j "kappa_F"),
    epsilon_A := (← getFloat j "epsilon_A"),
    lambda_A  := (← getFloat j "lambda_A"),
    mu_0      := (← getFloat j "mu_0"),
    mu_B      := (← getFloat j "mu_B"),
    mu_F      := (← getFloat j "mu_F"),
    mu_FF     := (← getFloat j "mu_FF"),
    eta       := (← getFloat j "eta"),
    sigma_B   := (← getFloat j "sigma_B"),
    sigma_F   := (← getFloat j "sigma_F"),
    sigma_A   := (← getFloat j "sigma_A")
  }

private def getParamsV15 (j : Json) : Except String Params_v15 := do
  pure {
    tau_B     := (← getFloat j "tau_B"),
    tau_F     := (← getFloat j "tau_F"),
    B_inf     := (← getFloat j "B_inf"),
    F_inf     := (← getFloat j "F_inf"),
    epsilon_A := (← getFloat j "epsilon_A"),
    lambda_A  := (← getFloat j "lambda_A"),
    mu_0      := (← getFloat j "mu_0"),
    mu_B      := (← getFloat j "mu_B"),
    mu_F      := (← getFloat j "mu_F"),
    mu_FF     := (← getFloat j "mu_FF"),
    eta       := (← getFloat j "eta"),
    sigma_B   := (← getFloat j "sigma_B"),
    sigma_F   := (← getFloat j "sigma_F"),
    sigma_A   := (← getFloat j "sigma_A")
  }

private def getObsNoise (j : Json) : Except String ObsNoiseParams := do
  pure {
    sigma_B_obs := (← getFloat j "sigma_B_obs"),
    sigma_F_obs := (← getFloat j "sigma_F_obs"),
    sigma_A_obs := (← getFloat j "sigma_A_obs")
  }

private def getObs (j : Json) : Except String Obs := do
  pure {
    obs_B := (← getFloat j "obs_B"),
    obs_F := (← getFloat j "obs_F"),
    obs_A := (← getFloat j "obs_A")
  }

/-- Float → JSON-number string. Special-case ±inf/NaN to extended-JSON
    tokens (Julia's `JSON3` accepts these via `allow_inf=true`). -/
private def floatToJson (x : Float) : String :=
  if x.isNaN then "NaN"
  else if x.isInf then (if x < 0.0 then "-Infinity" else "Infinity")
  else toString x

private def fmt3 (a b c : Float) : String :=
  "[" ++ floatToJson a ++ "," ++ floatToJson b ++ "," ++ floatToJson c ++ "]"

private def fmtTriple (key : String) (t : Float × Float × Float) : String :=
  let (a, b, c) := t
  "{\"" ++ key ++ "\":" ++ fmt3 a b c ++ "}"

private def fmtScalar (key : String) (x : Float) : String :=
  "{\"" ++ key ++ "\":" ++ floatToJson x ++ "}"

private def fmtParamsV1 (p : Params_v1) : String :=
  "{\"params_v1\":{" ++
    "\"tau_B\":"     ++ floatToJson p.tau_B     ++ "," ++
    "\"tau_F\":"     ++ floatToJson p.tau_F     ++ "," ++
    "\"kappa_B\":"   ++ floatToJson p.kappa_B   ++ "," ++
    "\"kappa_F\":"   ++ floatToJson p.kappa_F   ++ "," ++
    "\"epsilon_A\":" ++ floatToJson p.epsilon_A ++ "," ++
    "\"lambda_A\":"  ++ floatToJson p.lambda_A  ++ "," ++
    "\"mu_0\":"      ++ floatToJson p.mu_0      ++ "," ++
    "\"mu_B\":"      ++ floatToJson p.mu_B      ++ "," ++
    "\"mu_F\":"      ++ floatToJson p.mu_F      ++ "," ++
    "\"mu_FF\":"     ++ floatToJson p.mu_FF     ++ "," ++
    "\"eta\":"       ++ floatToJson p.eta       ++ "," ++
    "\"sigma_B\":"   ++ floatToJson p.sigma_B   ++ "," ++
    "\"sigma_F\":"   ++ floatToJson p.sigma_F   ++ "," ++
    "\"sigma_A\":"   ++ floatToJson p.sigma_A   ++
  "}}"

private def fmtPlantStepResult (s : PlantState) (o : Obs) : String :=
  "{\"next_state\":[" ++ floatToJson s.B ++ "," ++ floatToJson s.F ++
    "," ++ floatToJson s.A ++ "]," ++
   "\"obs\":[" ++ floatToJson o.obs_B ++ "," ++ floatToJson o.obs_F ++
    "," ++ floatToJson o.obs_A ++ "]}"

private def fmtError (msg : String) : String :=
  "{\"error\":" ++ (Json.str msg).compress ++ "}"

/-- Process a single JSON request → response string (one line of JSON). -/
def handleRequest (input : Json) : Except String String := do
  let fn ← getStringField input "fn"
  match fn with
  | "drift" =>
    let st  ← (← input.getObjVal? "state")  |> getPlantStateFromArr
    let phi ← getFloat input "phi"
    let p   ← (← input.getObjVal? "params") |> getParamsV1
    pure (fmtTriple "deriv" (drift st p phi))
  | "diffusion" =>
    let st  ← (← input.getObjVal? "state")  |> getPlantStateFromArr
    let p   ← (← input.getObjVal? "params") |> getParamsV1
    pure (fmtTriple "sigma" (diffusion_state_dep st p))
  | "emStep" =>
    let st  ← (← input.getObjVal? "state")  |> getPlantStateFromArr
    let p   ← (← input.getObjVal? "params") |> getParamsV1
    let nz  ← (← input.getObjVal? "noise")  |> getTripleFromArr
    let phi ← getFloat input "phi"
    let dt  ← getFloat input "dt"
    let nSubs ← getNatField input "n_substeps"
    let s' := em_step_substepped st p nz phi dt nSubs
    pure ("{\"next_state\":" ++ fmt3 s'.B s'.F s'.A ++ "}")
  | "paramsV15ToV1" =>
    let p15 ← (← input.getObjVal? "params") |> getParamsV15
    pure (fmtParamsV1 (params_v15_to_v1_nt p15))
  | "reflectUnit" =>
    let x ← getFloat input "x"
    pure (fmtScalar "x" (reflect_unit x))
  | "plantStep" =>
    let st  ← (← input.getObjVal? "state")  |> getPlantStateFromArr
    let phi ← getFloat input "phi"
    let p15 ← (← input.getObjVal? "params") |> getParamsV15
    let op  ← (← input.getObjVal? "obs_params") |> getObsNoise
    let dt  ← getFloat input "dt"
    let sn  ← (← input.getObjVal? "sde_noise") |> getTripleFromArr
    let on  ← (← input.getObjVal? "obs_noise") |> getTripleFromArr
    let (s', o) := plant_step st phi p15 op dt sn on
    pure (fmtPlantStepResult s' o)
  | "obsLogWeight" =>
    let b ← getFloat input "B"
    let f ← getFloat input "F"
    let a ← getFloat input "A"
    let o ← (← input.getObjVal? "obs") |> getObs
    let op ← (← input.getObjVal? "obs_params") |> getObsNoise
    pure (fmtScalar "log_w" (obs_log_weight_one b f a o op))
  | "applyPrior" =>
    let kindStr ← getStringField input "kind"
    let mu    ← getFloat input "mu"
    let sigma ← getFloat input "sigma"
    let u     ← getFloat input "u"
    let kind : PriorKind := match kindStr with
      | "logNormal" => .logNormal
      | _           => .normal
    pure (fmtScalar "x" (apply_prior kind mu sigma u))
  | other => Except.error s!"unknown fn: {other}"

partial def loop (h : IO.FS.Stream) : IO Unit := do
  let line ← h.getLine
  if line.isEmpty then
    return ()
  else
    let trimmed := (line.trimAscii).toString
    if trimmed.isEmpty then
      loop h
    else
      match Json.parse trimmed with
      | .ok j =>
        match handleRequest j with
        | .ok out  => IO.println out
        | .error e => IO.println (fmtError e)
      | .error e => IO.println (fmtError s!"JSON parse error: {e}")
      (← IO.getStdout).flush
      loop h

def main : IO Unit := do
  let stdin ← IO.getStdin
  loop stdin
