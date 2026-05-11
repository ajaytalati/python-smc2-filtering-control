# Streamlined v2 controllability proofs document — LEAN4-ready

> Archived from plan mode: 2026-05-11 14:21.

## Context

The existing 42-page document `sedentary_basin_controllability.tex` is the *investigative writeup*: it spent the first ~20 pages building an impossibility certificate against the canonical model parametrisation, then pivoted across §13--§17 to find a *revised* parametrisation under which the FSA-v5 system has four qualitatively desirable controllability properties:

1. **Pathological sedentary basin** — passive Φ = (0.1, 0.1) from SEDENTARY_INIT drives A → 0.
2. **Constructive escape from SEDENTARY_INIT** — a constant Φ* = (0.86, 1.24) (FIM-selected) takes A from 0.10 to ≥ 1.16 in 100 d.
3. **Pathological over-training basin** — passive Φ = (2, 2) from TRAINED_ATHLETE_INIT_v2 drives A → 0.
4. **Constructive maintenance** — constant Φ = (1, 1) holds TRAINED_ATHLETE_INIT_v2 at the slow-manifold A* ≈ 1.24.

The investigative content (impossibility certificate, 5-parameter sweep, hand-designed bang-bang candidates, sensitivity sweep over pinned parameters) was essential exploration but is now noise relative to the cleaner constructive result. **The next step is a new document that presents only the constructive results, the v2 parametrisation, and everything a LEAN4 verifier needs to mechanise the four theorems.**

The new document must be self-contained: a verifier with ONLY this document should be able to (a) implement the dynamics in Julia/Python to cross-check numerics, and (b) write the four LEAN4 theorems and discharge their proofs. The existing investigative writeup will be cited as long-form background but is not a dependency.

## Approach

### File: `version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.tex`

New document, ~18--22 pages target. Sits in the same directory as the investigative writeup and reuses the same figures. Self-contained preamble (mirroring the existing tech-guide convention).

### Document outline

1. **Introduction** (~2 pp). Adapted from existing §1 and §10. Says: this document presents four constructive controllability theorems for the FSA-v5 deterministic dynamics under a *revised* parametrisation. The four behaviours are the qualitative cases the original empirical SMC²-MPC bench failed at — now provably achievable. References the long-form investigative writeup `sedentary_basin_controllability.tex` as background. States working assumptions (deterministic ODE, full observability) up front.

   **Two scope statements to include in §1 / the abstract** (per user clarification on plan approval):
   - **"v2 is a re-parametrisation, not a re-modelling"**: the mathematical structure of FSA-v5 (state vector, drift equations, observation model, slow-manifold cascade) is unchanged. Only the numerical values of 8 parameters differ from canonical TRUTH_PARAMS_V5. The existing `Fsa.V5.*` Lean specifications and Julia dynamics/observation modules need no structural edits. Numerical verification of the four theorems is fully self-contained in the App 2.2 reference Julia script (no project dependencies). The *only* code-change prerequisite for running the full SMC²-MPC bench under v2 is a small CLI-flag addition (~100 LOC) to expose the v2 parameter overrides; see §TODO. SMC² cost-function tuning is a separate controller-engineering concern, also flagged in §TODO.
   - **"FIM identifiability under v2 is open"**: the existing §15 tech-guide FIM analysis was performed under canonical parameters. v2 alters dynamics timescales, fatigue penalties, and deconditioning thresholds by large factors, all of which substantially change the FIM structure. A repeat analysis under v2 is a prerequisite for production deployment but is OUT OF SCOPE for this document; flagged in §TODO.

2. **Model recall (revised v2 parametrisation)** (~4 pp). Synthesised from existing §2 and §17.1. Self-contained:
   - 6D state vector with physical meaning + bounds (existing §2.6 table).
   - Drift equations (1)--(5) for B, S, F, A, K_FB, K_FS.
   - Bifurcation parameter μ(B, S, F) (Eq. 6) with Hill deconditioning function h_n.
   - Slow-manifold equilibria K*, B*, S*, F* in closed form (Eqs. 8--11).
   - **Full RECOMMENDED_V2 parameter table** showing canonical centred forms (μ_0 = 0.02 + 0.40·F_TYP², etc.) and the v2 overrides. Distinguishes canonical, v1-recommended (§15), and v2-recommended (§17) values clearly.
   - **Initial conditions**: SEDENTARY_INIT (canonical) and TRAINED_ATHLETE_INIT_v2 (computed as the slow-manifold state at the island center under v2). Both as plain numerical tables for verifier reproduction.
   - Inline reproducer snippet (paste-able pseudocode to verify μ_init = -0.115 under canonical and μ̄(0; (1,1)) = +0.119 under v2).

3. **Control system and controllability target** (~1 pp). Adapted from existing §3. Definitions of admissible control set 𝒰(T, U), reachable set R(T; y₀, U), controllability target {y : A ≥ A_target}.

4. **Cascade structure and slow-manifold** (~1 pp). Distilled from existing §4. Just the facts the verifier needs: cascade ordering K → (B, S) → F → A, closed-form slow-manifold expressions for each block, definition of μ̄(A; Φ) on the slow manifold. Brief — no upper-envelope analysis needed in the streamlined narrative.

5. **The healthy-island geometry under v2** (~2 pp). New unified section combining the relevant parts of §15.3 (recommended parametrisation) and §16 (island center). Shows the v2 island heatmap (`island_recommended_v2.png`), records numerically that island center is (1.06, 0.78), μ̄(0; (1,1)) = +0.119, μ̄(0; (0.1, 0.1)) = -0.144, μ̄(0; (2,2)) = -0.656. The full closed-form expressions for the slow-manifold A* solving μ̄(A; Φ) = η·A² at the center, yielding A* = 1.238.

6. **The four constructive controllability theorems** (~6 pp). Heart of the document. Each theorem is presented as: formal statement, parameter values and witness Φ, integration setup, numerical outcome, and LEAN-mechanisable proof shape.
   - **Theorem 6.1 (Pathological sedentary basin)**: ∀ trajectories y(t) under Φ(t) ≡ (0.1, 0.1), y(0) = SEDENTARY_INIT, recommended_v2 params: A(100) < ε for any ε > 10⁻⁴.
   - **Theorem 6.2 (Constructive escape from SEDENTARY_INIT)**: ∃ constant witness Φ* = (0.86, 1.24) such that y(0) = SEDENTARY_INIT, recommended_v2 params yields A(100) ≥ 1.0. Includes the figure `sedentary_basin_witness_v2.png` (left panel = Thm 6.1, right = Thm 6.2).
   - **Theorem 6.3 (Pathological over-training basin)**: ∀ trajectories y(t) under Φ(t) ≡ (2, 2), y(0) = TRAINED_ATHLETE_INIT_v2, recommended_v2 params: A(100) < 0.01.
   - **Theorem 6.4 (Constructive maintenance)**: ∀ trajectories y(t) under Φ(t) ≡ (1, 1), y(0) = TRAINED_ATHLETE_INIT_v2, recommended_v2 params: A(100) ∈ [A* - ε, A* + ε] where A* = 1.238 and ε is small.
   Includes the figure `overtraining_basin_witness.png` (left = Thm 6.3, right = Thm 6.4).
   
   Each theorem also names its "LEAN-mechanisable shape" — what kind of Lean machinery the proof needs (existence/uniqueness of ODE solution, scalar comparison theorem on A-equation, numerical evaluation of a finite Float inequality).

7. **FIM-duality witness selection** (~2 pp). From existing §17.2-§17.3. Explains the identification-↔-controllability duality and why the FIM-condition criterion is a principled way to select a constant Φ witness for Theorem 6.2. Includes `fim_grid_search.png`. States the empirical finding: FIM-low-κ region is a strict subset of the A-escape region (qualitative duality confirmed) but the optima differ by a few grid cells (quantitative duality loose). Both criteria give an escaping Φ; we pick the FIM optimum for principle.

8. **Implications for closed-loop control** (~1 pp). From existing §10, distilled for v2. Brief: under the revised parametrisation, a simple constant-Φ controller succeeds where the canonical model required impossible heroics. The bench failure observed on 2026-05-11 was a model issue, not a controller-tuning issue.

9. **Mechanisation roadmap toward LEAN4** (~2-3 pp). Adapted from existing §11, expanded for the four specific theorems. For each theorem:
   - Lean Mathlib dependencies (ODE existence/uniqueness, Gronwall, comparison theorem, scalar inequalities).
   - Expected formalisation effort in lines.
   - Open analytic gaps.
   - Suggested proof tactic outline (sorry-laden skeleton).

10. **TODO / Open questions and further work** (~1.5 pp). Three concrete actionable items at the top (per user request), then briefer items below.

    **§10.1 — CLI-flag exposure for v2 parameters in the Julia bench.** The current bench hardcodes `TRUTH_PARAMS_V5` in `simulation_v5.jl`; no CLI flags exist for the truth-side parameters (only controller-side coefficients have flags). To run closed-loop verification of the four theorems via standard launchers, add either: (a) ~8 individual CLI flags (`--truth-B-dec`, `--truth-S-dec`, `--truth-mu-F`, `--truth-mu-FF`, `--truth-tau-B`, `--truth-tau-S`, `--truth-kappa-B`, `--truth-kappa-S`) each overriding the corresponding entry in `TRUTH_PARAMS_V5`; or (b) a single `--truth-preset v2` selector backed by a new const `TRUTH_PARAMS_V5_RECOMMENDED_V2` in `simulation_v5.jl`. Estimated effort: <100 LOC. This is a pure engineering wiring change and does not touch any of the mathematical content of the present document.

    **§10.2 — FIM-based identifiability analysis under v2.** The existing tech-guide §15 FIM analysis was performed under canonical TRUTH_PARAMS_V5 and reported κ(F) ~ 10¹⁸ with 5 small eigenvalues corresponding to structural non-identifiabilities (e.g., $K_{FB}^0 \leftrightarrow K_{FS}^0$ aliasing, $\tau_K \mu_K$ joint slack). Under v2:
      - Dynamics are 2× faster ($\tau_B, \tau_S$ halved). Observations sample more rapidly relative to the latent timescales, potentially improving identifiability of slow parameters.
      - $\mu_F, \mu_{FF}$ are reduced 88--95\%, weakening the fatigue-mediated channels of $A \to$ observed sleep, HR, stress. May worsen identifiability of fatigue-coupling parameters.
      - $B_{\rm dec}, S_{\rm dec}$ are raised 3.6×, changing the regime in which the Hill deconditioning fires. Identifiability of $\mu_{B-}, \mu_{S-}$ from logged detraining episodes may shift.
      - A repeat FIM analysis under v2 is needed to (a) verify previously pinned parameters are still appropriately pinned, (b) identify new slack directions that emerge, (c) judge whether the reduced $\mu_F, \mu_{FF}$ introduce new identifiability problems in the autonomic-to-fatigue coupling. **This is a prerequisite for any production deployment of the v2 model on real data.**

    **§10.3 — SMC² cost-function tuning under v2.** The controller's effort / chance / island cost coefficients were tuned for the canonical parametrisation. Under v2, the natural scales of state variables change ($A$ ranges to $\sim 1.24$ vs canonical $\sim 0.55$; $F$ ranges to $\sim 0.8$ vs canonical $\sim 0.2$; $B$ ranges to $\sim 0.8$ vs canonical $\sim 0.5$). The cost coefficients $\lambda_{\rm Phi}, \lambda_F, \lambda_{\rm chance}, \lambda_{\rm island}$ may need re-tuning to give similar effective pressure. This is a controller-engineering issue distinct from the mathematical content of the present document; the four theorems hold regardless of how the SMC² controller's cost is shaped.

    **§10.4 — Stochastic extension.** Deterministic constructive controllability does not directly imply stochastic controllability under the FSA-v5 SDE. Freidlin–Wentzell large-deviations theory may give the analogous probabilistic statements. Out of scope.

    **§10.5 — Time-varying $\Phi(t)$.** All four witnesses in this document are constant Φ. Time-varying Φ might be more efficient (shorter horizons, faster recovery) but reintroduces the LEAN-mechanisability complexity that constant-Φ witnesses cleanly avoid.

### Appendices

- **Appendix 1 — Numerical anchors** (~2 pp). Adapted from existing App A, extended for the v2 parametrisation. Exhaustive table of every numerical claim in the document with its source: canonical centred-formula values from `simulation_v5.jl`, v2 overrides from §17.1, witness Φ* values from §17.3, A(T) values from `generate_figures.py` runs. Verifier should be able to reproduce every quantity in the document by following the source chain.

- **Appendix 2 — LEAN4 + Julia implementation kit** (~3-4 pp). The key deliverable for verifiers.
  - **App 2.1 — LEAN4 theorem stubs**. Formal Lean4 statements of Theorems 6.1--6.4. Each stub includes:
    - `def recommended_v2_params : Params := {...}` with all numerical values.
    - `def SEDENTARY_INIT, TRAINED_ATHLETE_INIT_v2 : State` definitions.
    - `def witness_phi_62 : ℝ × ℝ := (0.86, 1.24)` etc.
    - `theorem sedentary_basin_collapse : ... := by sorry` skeletons with proof obligations.
    - Notes on the Mathlib lemmas each stub depends on.
  - **App 2.2 — Julia reference numerics**. ~80-line self-contained Julia script: hard-codes the v2 parametrisation, defines `drift`, `mu_bar`, `slow_manifold_state`, `simulate` (RK4), runs each of the four cases, prints A(100). Verifier can run this to cross-check their Lean numerics against the reference. Compiles standalone with no project dependencies (no `using FSAv5`, etc.).

### Figures (re-used from existing directory)

- `island_recommended_v2.png` — §5, the v2 healthy island.
- `fim_grid_search.png` — §7, FIM-duality grid search.
- `sedentary_basin_witness_v2.png` — §6 (Thm 6.1 + Thm 6.2).
- `overtraining_basin_witness.png` — §6 (Thm 6.3 + Thm 6.4). [Note: under v2 params the trajectory is qualitatively identical to v1's because TRAINED_ATHLETE_INIT_v2 is the slow-manifold state in both cases. Document this in the figure caption.]

No new figures needed; all already generated by `generate_figures.py`.

## Critical files

**To create**:
- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.tex](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.tex) — the new streamlined document.
- [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/reference_numerics.jl](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/reference_numerics.jl) — App 2.2 standalone Julia reference script (about 80 lines).

**To read for source material**:
- [sedentary_basin_controllability.tex](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex) — existing 42-page investigative writeup; source for §1 intro, §2 model, §3 control, §4 cascade, §10 implications, §11 LEAN4 roadmap, §17 (FIM + v2 params), App A. Distil and condense.
- [generate_figures.py](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/generate_figures.py) — source for the reference Julia script (translate Python → Julia, keep it self-contained).
- [version_1_5_LEAN/Fsa/V5/Cost.lean](version_1_5_LEAN/Fsa/V5/Cost.lean) and [Drift.lean](version_1_5_LEAN/Fsa/V5/) — for the LEAN4 stub interface (must align with what's already mechanised).

## Verification

1. **pdflatex compiles clean**, twice (resolve ToC + refs). Target 18--22 pages.
2. **Every numerical claim in the body has a source in App 1** (every "0.1188", "1.06", "1.238", etc. traces to either the canonical centred formulas, the §17 v2 overrides, or `generate_figures.py` script output).
3. **App 2.1 Lean stubs are syntactically plausible** (compile if Lean is set up; otherwise read-checked by hand against the `Fsa.V5.*` module structure).
4. **App 2.2 Julia reference script runs standalone**: `julia reference_numerics.jl` should print four lines, one per case, matching the figures' A(100) values to 3 decimal places. The script must not `using FSAv5` or depend on any project package; only `Random`, `Printf` (and inline RK4) are allowed.
5. **Document is self-contained**: a reader who has never seen the investigative writeup should be able to read this document end-to-end and write a Lean proof for any of the four theorems. Verify by checking that every defined symbol is introduced before use and every external reference is given with a one-line gloss (not just a citation key).

## Out of scope (deferred / intentionally cut)

- The impossibility certificate against the canonical parametrisation (existing §5--§9). Not relevant to the constructive narrative.
- The 5-parameter sweep methodology (existing §15). The result is in App 1's "v2 recommended values" line; the search process is not narrated.
- Hand-designed bang-bang witnesses (existing §16.2 v1). The FIM-duality witness supersedes them.
- Sensitivity to pinned parameters (existing §13). Not relevant once the v2 parametrisation is chosen.
- The composite Lyapunov function (existing App B). Could be a future App 3 if the maintenance theorem (6.4) benefits from a Lyapunov-style proof, but not in this iteration.

## Plan archive

After ExitPlanMode and implementation, archive this plan to `claude_plans/Streamlined_v2_controllability_proofs_2026-05-11_HHMM.md` per the global rule.
