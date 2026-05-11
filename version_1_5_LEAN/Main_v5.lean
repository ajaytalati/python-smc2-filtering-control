import Fsa.V5
import Lean.Data.Json

/-!
# CLI bridge for FSA v5 differential testing

Reads one JSON object per line on stdin, writes one JSON object per
line on stdout. Mirrors `Main.lean` (the v1.5 CLI) but exposes the
v5 surface from `Fsa.V5.*`.

Wire conventions:

  - State6D (B, S, F, A, K_FB, K_FS)  ↔  JSON array of 6 floats
  - BimodalPhi (Phi_B, Phi_S)         ↔  JSON object {"Phi_B", "Phi_S"}
  - Params (28 fields)                 ↔  JSON object, one key per field
  - ObsParams (22 fields)              ↔  JSON object, one key per field

Dispatch tags:

  `drift`            — (state, phi, params)               → deriv [6]
  `diffusion`        — (state, params)                     → sigma [6]
  `emStep`           — (state, phi, params, sigmaDiag, dt, noise)
                                                           → next_state [6]
  `hrMean`           — (state, C, obs_params)              → x : Float
  `sleepProb`        — (state, C, obs_params)              → x : Float
  `stressMean`       — (state, C, obs_params)              → x : Float
  `stepsLogMean`     — (state, C, obs_params)              → x : Float
  `volumeLoadMean`   — (state, obs_params)                 → x : Float
  `muBar`            — (A, phi, params)                    → x : Float
  `findASep`         — (phi, params)                       → x : Float (±Inf sentinels)
  `aSepGrid`         — (particles : Array Params, schedule : Array BimodalPhi)
                                                           → matrix : Array (Array Float)
  `scheduleFromTheta` — (theta, phiDesign, cPhi, phiMax, n_anchors)
                                                           → schedule : Array {Phi_B, Phi_S}
  `designMatrix`     — (n_steps, dt, n_anchors, width_factor)
                                                           → Array (Array Float)
  `cPhi`             — (phi_default, phi_max)              → x : Float
  `sigmoid`          — (x)                                  → x : Float
-/

open Lean (Json)
open Fsa.V5

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

/-- 6-element JSON array → `State6D`. Order: [B, S, F, A, K_FB, K_FS]. -/
private def getState6D (j : Json) : Except String State6D := do
  let arr ← j.getArr?
  if arr.size != 6 then
    Except.error s!"state array must have 6 elements, got {arr.size}"
  else
    pure {
      B   := (← jsonToFloat? arr[0]!),
      S   := (← jsonToFloat? arr[1]!),
      F   := (← jsonToFloat? arr[2]!),
      A   := (← jsonToFloat? arr[3]!),
      KFB := (← jsonToFloat? arr[4]!),
      KFS := (← jsonToFloat? arr[5]!)
    }

/-- JSON object {"Phi_B", "Phi_S"} → `BimodalPhi`. -/
private def getBimodalPhi (j : Json) : Except String BimodalPhi := do
  pure {
    Phi_B := (← getFloat j "Phi_B"),
    Phi_S := (← getFloat j "Phi_S")
  }

/-- JSON object with 28 keys → `Params`. -/
private def getParams (j : Json) : Except String Params := do
  pure {
    tau_B      := (← getFloat j "tau_B"),
    kappa_B    := (← getFloat j "kappa_B"),
    epsilon_AB := (← getFloat j "epsilon_AB"),
    tau_S      := (← getFloat j "tau_S"),
    kappa_S    := (← getFloat j "kappa_S"),
    epsilon_AS := (← getFloat j "epsilon_AS"),
    tau_F      := (← getFloat j "tau_F"),
    lambda_A   := (← getFloat j "lambda_A"),
    KFB_0      := (← getFloat j "KFB_0"),
    KFS_0      := (← getFloat j "KFS_0"),
    tau_K      := (← getFloat j "tau_K"),
    mu_K       := (← getFloat j "mu_K"),
    mu_0       := (← getFloat j "mu_0"),
    mu_B       := (← getFloat j "mu_B"),
    mu_S       := (← getFloat j "mu_S"),
    mu_F       := (← getFloat j "mu_F"),
    mu_FF      := (← getFloat j "mu_FF"),
    eta        := (← getFloat j "eta"),
    sigma_B    := (← getFloat j "sigma_B"),
    sigma_S    := (← getFloat j "sigma_S"),
    sigma_F    := (← getFloat j "sigma_F"),
    sigma_A    := (← getFloat j "sigma_A"),
    sigma_K    := (← getFloat j "sigma_K"),
    B_dec      := (← getFloat j "B_dec"),
    S_dec      := (← getFloat j "S_dec"),
    mu_dec_B   := (← getFloat j "mu_dec_B"),
    mu_dec_S   := (← getFloat j "mu_dec_S"),
    n_dec      := (← getFloat j "n_dec")
  }

/-- JSON object with 22 keys → `ObsParams`. -/
private def getObsParams (j : Json) : Except String ObsParams := do
  pure {
    HR_base     := (← getFloat j "HR_base"),
    kappa_B_HR  := (← getFloat j "kappa_B_HR"),
    alpha_A_HR  := (← getFloat j "alpha_A_HR"),
    beta_C_HR   := (← getFloat j "beta_C_HR"),
    sigma_HR    := (← getFloat j "sigma_HR"),
    k_C         := (← getFloat j "k_C"),
    k_A         := (← getFloat j "k_A"),
    c_tilde     := (← getFloat j "c_tilde"),
    S_base      := (← getFloat j "S_base"),
    k_F         := (← getFloat j "k_F"),
    k_A_S       := (← getFloat j "k_A_S"),
    beta_C_S    := (← getFloat j "beta_C_S"),
    sigma_S_obs := (← getFloat j "sigma_S_obs"),
    mu_step0    := (← getFloat j "mu_step0"),
    beta_B_st   := (← getFloat j "beta_B_st"),
    beta_F_st   := (← getFloat j "beta_F_st"),
    beta_A_st   := (← getFloat j "beta_A_st"),
    beta_C_st   := (← getFloat j "beta_C_st"),
    sigma_st    := (← getFloat j "sigma_st"),
    beta_S_VL   := (← getFloat j "beta_S_VL"),
    beta_F_VL   := (← getFloat j "beta_F_VL"),
    sigma_VL    := (← getFloat j "sigma_VL")
  }

/-! ## Output formatting

Float → JSON-number string. Special-case ±inf/NaN to extended-JSON
tokens (Julia's `JSON3` accepts these via `allow_inf=true`).
-/

private def floatToJson (x : Float) : String :=
  if x.isNaN then "NaN"
  else if x.isInf then (if x < 0.0 then "-Infinity" else "Infinity")
  else toString x

private def fmt6 (s : State6D) : String :=
  "[" ++ floatToJson s.B   ++ "," ++ floatToJson s.S   ++ "," ++
        floatToJson s.F   ++ "," ++ floatToJson s.A   ++ "," ++
        floatToJson s.KFB ++ "," ++ floatToJson s.KFS ++ "]"

private def fmtState6D (key : String) (s : State6D) : String :=
  "{\"" ++ key ++ "\":" ++ fmt6 s ++ "}"

private def fmtScalar (key : String) (x : Float) : String :=
  "{\"" ++ key ++ "\":" ++ floatToJson x ++ "}"

private def fmtFloatArr (xs : Array Float) : String :=
  let body := String.intercalate "," (xs.toList.map floatToJson)
  "[" ++ body ++ "]"

private def fmtFloatMatrix (m : Array (Array Float)) : String :=
  let rows := m.toList.map fmtFloatArr
  "[" ++ String.intercalate "," rows ++ "]"

private def fmtBimodalPhi (p : BimodalPhi) : String :=
  "{\"Phi_B\":" ++ floatToJson p.Phi_B ++
   ",\"Phi_S\":" ++ floatToJson p.Phi_S ++ "}"

private def fmtSchedule (s : Schedule) : String :=
  let rows := s.toList.map fmtBimodalPhi
  "[" ++ String.intercalate "," rows ++ "]"

private def fmtError (msg : String) : String :=
  "{\"error\":" ++ (Json.str msg).compress ++ "}"

/-- Pull `Array Params` from a JSON array of param objects. -/
private def getParticles (j : Json) : Except String (Array Params) := do
  let arr ← j.getArr?
  arr.mapM getParams

/-- Pull `Array BimodalPhi` from a JSON array of phi objects. -/
private def getScheduleArr (j : Json) : Except String (Array BimodalPhi) := do
  let arr ← j.getArr?
  arr.mapM getBimodalPhi

/-- Pull `Array (Array Float)` from a 2-D JSON array. -/
private def getFloatMatrix (j : Json) : Except String (Array (Array Float)) := do
  let rows ← j.getArr?
  rows.mapM getFloatArr

/-- Process a single JSON request → response string (one line of JSON). -/
def handleRequest (input : Json) : Except String String := do
  let fn ← getStringField input "fn"
  match fn with
  | "drift" =>
    let st  ← (← input.getObjVal? "state")  |> getState6D
    let phi ← (← input.getObjVal? "phi")    |> getBimodalPhi
    let p   ← (← input.getObjVal? "params") |> getParams
    pure (fmtState6D "deriv" (drift st p phi))
  | "diffusion" =>
    let st  ← (← input.getObjVal? "state")  |> getState6D
    let p   ← (← input.getObjVal? "params") |> getParams
    pure (fmtState6D "sigma" (diffusion st p))
  | "emStep" =>
    let st    ← (← input.getObjVal? "state")     |> getState6D
    let phi   ← (← input.getObjVal? "phi")       |> getBimodalPhi
    let p     ← (← input.getObjVal? "params")    |> getParams
    let sd    ← (← input.getObjVal? "sigmaDiag") |> getFloatArr
    let dt    ← getFloat input "dt"
    let nz    ← (← input.getObjVal? "noise")     |> getFloatArr
    let s' := emStep st phi p sd dt nz
    pure (fmtState6D "next_state" s')
  | "hrMean" =>
    let st ← (← input.getObjVal? "state") |> getState6D
    let C  ← getFloat input "C"
    let op ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (fmtScalar "x" (hrMean st C op))
  | "sleepProb" =>
    let st ← (← input.getObjVal? "state") |> getState6D
    let C  ← getFloat input "C"
    let op ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (fmtScalar "x" (sleepProb st C op))
  | "stressMean" =>
    let st ← (← input.getObjVal? "state") |> getState6D
    let C  ← getFloat input "C"
    let op ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (fmtScalar "x" (stressMean st C op))
  | "stepsLogMean" =>
    let st ← (← input.getObjVal? "state") |> getState6D
    let C  ← getFloat input "C"
    let op ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (fmtScalar "x" (stepsLogMean st C op))
  | "volumeLoadMean" =>
    let st ← (← input.getObjVal? "state") |> getState6D
    let op ← (← input.getObjVal? "obs_params") |> getObsParams
    pure (fmtScalar "x" (volumeLoadMean st op))
  | "muBar" =>
    let A   ← getFloat input "A"
    let phi ← (← input.getObjVal? "phi")    |> getBimodalPhi
    let p   ← (← input.getObjVal? "params") |> getParams
    pure (fmtScalar "x" (muBar A phi p))
  | "findASep" =>
    let phi ← (← input.getObjVal? "phi")    |> getBimodalPhi
    let p   ← (← input.getObjVal? "params") |> getParams
    pure (fmtScalar "x" (findASep phi p))
  | "aSepGrid" =>
    let particles ← (← input.getObjVal? "particles") |> getParticles
    let schedule  ← (← input.getObjVal? "schedule")  |> getScheduleArr
    pure ("{\"matrix\":" ++ fmtFloatMatrix (aSepGrid particles schedule) ++ "}")
  | "scheduleFromTheta" =>
    let theta     ← (← input.getObjVal? "theta")     |> getFloatArr
    let phiDesign ← (← input.getObjVal? "phiDesign") |> getFloatMatrix
    let cPhiVal   ← getFloat input "cPhi"
    let phiMax    ← getFloat input "phiMax"
    let nAnchors  ← getNatField input "n_anchors"
    let sched := scheduleFromTheta theta phiDesign cPhiVal phiMax nAnchors
    pure ("{\"schedule\":" ++ fmtSchedule sched ++ "}")
  | "designMatrix" =>
    let nSteps    ← getNatField input "n_steps"
    let dt        ← getFloat input "dt"
    let nAnchors  ← getNatField input "n_anchors"
    let widthFac  ← getFloat input "width_factor"
    pure ("{\"matrix\":" ++ fmtFloatMatrix (designMatrix nSteps dt nAnchors widthFac) ++ "}")
  | "cPhi" =>
    let phiDef ← getFloat input "phi_default"
    let phiMax ← getFloat input "phi_max"
    pure (fmtScalar "x" (c_phi phiDef phiMax))
  | "sigmoid" =>
    let x ← getFloat input "x"
    pure (fmtScalar "x" (sigmoid x))
  | "softChancePenalty" =>
    let val ← getFloat input "val"
    let thr ← getFloat input "thr"
    let beta ← getFloat input "beta"
    let scale ← getFloat input "scale"
    pure (fmtScalar "x" (softChancePenalty val thr beta scale))
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
