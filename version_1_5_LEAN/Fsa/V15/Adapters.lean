import Fsa.V15.Types

/-!
# FSA v1.5 parameter-basis adapters

★ **@match site #1** from the LaTeX writeup §7.1, Julia source
`simulation.jl:110-143` (`params_v15_to_v1_nt`).

Two adapters:

  * `params_v15_to_v1` — basis rotation from the v1.5 parametrisation
    `(B_inf, F_inf, …)` to the v1 drift parametrisation
    `(kappa_B, kappa_F, …)`. Conversion: `kappa_B = B_inf / tau_B`,
    `kappa_F = F_inf / tau_F`. Pure: same input → same output.
  * `fill_pinned` — merge a 10-field `EstimatedParams` with a
    4-field `PinnedDynamics` to produce the full 14-field
    `Params_v15`. Used by the filter's call site to insert pinned
    truth values for `tau_B`, `eta`, `epsilon_A`, `mu_FF`.

The Julia `@match` dispatch on `(::Dict, ::NamedTuple)` becomes a
Lean4 `match … with` over the inductive `ParamsForm` defined in
`Types.lean`. Both shapes converge on the same `Params_v1` output —
the dispatch is on the *input* representation only, mirroring the
Julia source's intent (Dict for truth-value lookups, NamedTuple for
ForwardDiff-friendly call sites).
-/

namespace Fsa.V15

/-- ★ @match site #1: v1.5-form params → v1-form params.

    Julia source `simulation.jl:110-143`:
    ```
    @match p begin
        ::Dict       => ( … kappa_B = p[:B_inf] / p[:tau_B], … )
        ::NamedTuple => ( … kappa_B = p.B_inf / p.tau_B,     … )
    end
    ```

    Lean4 transcription: `match form with | .dictForm d => … | .ntForm
    nt => …` over the `ParamsForm` sum type. Both arms produce the
    same v1-form `Params_v1` — only the input access pattern differs.

    For the `dictForm` arm we look up by the same string keys the
    Julia uses (`"tau_B"`, `"B_inf"`, …) since Lean4 has no
    `Symbol` type that maps cleanly. The diff test exercises only
    the `ntForm` case (the production path) at 1e-6; the
    `dictForm` arm is exercised in unit-test form at the Lean4 layer. -/
def params_v15_to_v1 (form : ParamsForm) : Params_v1 :=
  match form with
  | .dictForm d =>
      let tau_B    := d "tau_B"
      let tau_F    := d "tau_F"
      let B_inf    := d "B_inf"
      let F_inf    := d "F_inf"
      { tau_B     := tau_B,
        tau_F     := tau_F,
        kappa_B   := B_inf / tau_B,
        kappa_F   := F_inf / tau_F,
        epsilon_A := d "epsilon_A",
        lambda_A  := d "lambda_A",
        mu_0      := d "mu_0",
        mu_B      := d "mu_B",
        mu_F      := d "mu_F",
        mu_FF     := d "mu_FF",
        eta       := d "eta",
        sigma_B   := d "sigma_B",
        sigma_F   := d "sigma_F",
        sigma_A   := d "sigma_A" }
  | .ntForm p =>
      { tau_B     := p.tau_B,
        tau_F     := p.tau_F,
        kappa_B   := p.B_inf / p.tau_B,
        kappa_F   := p.F_inf / p.tau_F,
        epsilon_A := p.epsilon_A,
        lambda_A  := p.lambda_A,
        mu_0      := p.mu_0,
        mu_B      := p.mu_B,
        mu_F      := p.mu_F,
        mu_FF     := p.mu_FF,
        eta       := p.eta,
        sigma_B   := p.sigma_B,
        sigma_F   := p.sigma_F,
        sigma_A   := p.sigma_A }

/-- Convenience wrapper: a `Params_v15` value goes through the
    `ntForm` arm of `params_v15_to_v1`. -/
def params_v15_to_v1_nt (p : Params_v15) : Params_v1 :=
  params_v15_to_v1 (.ntForm p)

/-- Merge 10 estimated v1.5 params with the 4 pinned values to produce
    the full 14-field `Params_v15`. Maps to `simulation.jl:159-177`
    (`fill_pinned_nt`). Pure: same input → same output. -/
def fill_pinned (e : EstimatedParams) (pin : PinnedDynamics) : Params_v15 :=
  { tau_B     := pin.tau_B,
    tau_F     := e.tau_F,
    B_inf     := e.B_inf,
    F_inf     := e.F_inf,
    epsilon_A := pin.epsilon_A,
    lambda_A  := e.lambda_A,
    mu_0      := e.mu_0,
    mu_B      := e.mu_B,
    mu_F      := e.mu_F,
    mu_FF     := pin.mu_FF,
    eta       := pin.eta,
    sigma_B   := e.sigma_B,
    sigma_F   := e.sigma_F,
    sigma_A   := e.sigma_A }

end Fsa.V15
