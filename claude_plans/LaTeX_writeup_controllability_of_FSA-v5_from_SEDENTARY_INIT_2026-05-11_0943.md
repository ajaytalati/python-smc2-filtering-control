# LaTeX writeup: controllability of FSA-v5 from SEDENTARY_INIT

> Archived from plan mode: 2026-05-11 09:43.

## Context

A 3-hour T=100d closed-loop SMC²-MPC bench from `SEDENTARY_INIT = (B=0.05, S=0.10, F=0.30, A=0.10, K_FB=0.030, K_FS=0.050)` failed to escape the sedentary basin (A → 0.003 by day 100), despite a heavily upgraded controller (A_t-tracked island-pull cost term, 256 SMC particles, 4 HMC moves/level, 60 tempering levels). The user's hypothesis: the failure has a **mathematical** cause, not a controller-tuning cause.

A back-of-envelope check supports the hypothesis. Using the canonical centred parameters from [simulation_v5.jl:100-103](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L100-L103) (`mu_0 = 0.036`, `mu_F = 0.26`, `mu_FF = 0.40`, etc.), the bifurcation parameter at SEDENTARY_INIT is **μ_init ≈ −0.115 — independent of Φ at t=0**, because Φ enters the dynamics only through the slow drift of (B, S, F, K). A's exponential-decay timescale at this μ is ≈ 8.7 days, while B and S need 24+ days to climb above their deconditioning thresholds. A positive-feedback loop (A↓ → a_F↓ → F↑ → μ↓ → A↓ faster) tightens the noose.

The user wants a comprehensive LaTeX document that:
1. Lays out the FSA-v5 deterministic dynamics (citing the existing tech guide rather than re-deriving)
2. Attempts a **constructive** escape proof (try ramp / bang-bang / early-burst Φ candidates)
3. Falls back to an **impossibility certificate** via comparison-theorem upper bound on μ(t) if constructive attempts fail
4. Characterises the **controllability region** in initial-condition space (the certified non-escape set N(T) as a 2-D figure over (B₀, S₀) at fixed A₀)
5. Does this for **both** admissible control spaces Φ ∈ [0, 3]² (kernel) and Φ ∈ [0, 1]² (realistic)
6. Is structured to mechanise into a Lean4 proof later (analytic where possible; numerical certificates as single Float inequalities deferred to `norm_num`)

The user has put a hard freeze on coding until the proof is in hand.

## Approach

### Document layout (target ≈ 30 pp; saved to `version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex`)

The doc lives in a **new subdirectory** `LaTex_docs/sedentary_basin_controllability_analysis/` (mkdir as a first step). This isolates the controllability writeup + any new figures it generates from the main tech-guide assets. Uses the **self-contained preamble** convention from `FSA_version_5_technical_guide.tex` (does NOT include the shared `preamble.tex` — see top of [the tech guide .tex](version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) for the standard `\documentclass[11pt,a4paper]{article}` + `amsmath, amssymb, amsthm, graphicx, booktabs, hyperref, geometry, bm` package list). Reuses existing assets in `LaTex_docs/figures/` (`v5_sedentary_collapse_island.png`, `v5_full_bifurcation.png`, `v5_trajectories_three_regimes.png`, `v4_mu_vs_A_three_regimes.png`) rather than regenerating them — paths in `\includegraphics` are relative (`../figures/v5_...`).

**Working assumptions** (stated explicitly in §1 of the document):
1. **Deterministic dynamics** — the proof is for the ODE, not the SDE. Stochastic extension is out of scope (deferred to §12).
2. **Full state observability** — the controller is assumed to know (B, S, F, A, K_FB, K_FS) exactly. The closed-loop bench's particle filter is removed from the picture; we ask the *open-loop* controllability question.

| § | Title | Purpose |
|---|---|---|
| 1 | Introduction & problem statement | Closed-loop failure observation; pose the deterministic controllability question over Φ ∈ U for U ∈ {[0,1]², [0,3]²}. |
| 2 | Model recall | One-page citation of the 6D SDE, drift, slow-manifold equilibria, and v5 Hill deconditioning. References §2-3 of the tech guide and `eq:v5-mubar`, `eq:hill` from [14_v5_sedentary_collapse.tex](version_1_5_LEAN/LaTex_docs/sections/14_v5_sedentary_collapse.tex). |
| 3 | The control system | Define admissible controls 𝒰 = {Φ:[0,T] → U measurable}, the reachable set R(t, y₀), and the controllability target {y : A ≥ A_target}. |
| 4 | Slow-manifold reduction & cascade upper envelopes | Closed-form upper bounds B_UB(t; Φ_max), S_UB(t; Φ_max), and (carefully signed) F bounds. Boundedness lemmas. |
| 5 | The forcing-rate bottleneck (central a-priori estimate) | Prove μ̄_max(t \| y₀) — a function of t and parameters only, NOT of Φ — that upper-bounds μ(t) over the reachable set. Drops the helpful negative terms (μ_F·F, μ_FF·F², η·A³) so the bound is conservative and an honest over-estimate. |
| 6 | Constructive attempts at escape (Theorem B candidates) | Three explicit Φ(t) candidates evaluated symbolically: (i) constant ramp Φ ≡ (φ̄, φ̄), (ii) bang-bang [Φ_max for [0, T_burst]; tracking schedule after], (iii) early-burst-and-taper. For each, show the slow-manifold trajectory and verify whether A(T) ≥ A_target. |
| 7 | Theorem A — impossibility certificate | If all candidates from §6 fail and `∫₀^T μ̄_max(s) ds < −ln(A_target / A₀)`, escape is impossible from y₀ over [0, T]. Symbolic statement; one numeric Float inequality as the certificate. |
| 8 | Resolution at SEDENTARY_INIT | Apply §6-7 with TRUTH_PARAMS_V5. Distinguish the U = [0,1]² and U = [0,3]² cases. Either name the constructive Φ that works (and prove it), or print the impossibility certificate (one line: `−0.115 · 100 < −ln(0.5/0.10) ≈ −1.61` etc., as appropriate). |
| 9 | Controllability region | Plot the level surface `A₀ = A_target · exp(−∫₀^T μ̄_max(s) ds)` in the (B₀, S₀) plane at fixed A₀ ∈ {0.10, 0.30, 0.45}. Both U cases. SEDENTARY_INIT marked as a black dot. |
| 10 | Implications for closed-loop control | Why no controller (SMC²-MPC included) can succeed when §7's bound is tight. What initial conditions the user should reset to instead. |
| 11 | Mechanisation roadmap toward Lean4 | Map every lemma to existing `Fsa.V5.muBar`, `Fsa.V5.findASep`, `Fsa.V5.drift`. Flag analytic gaps Lean4 will need. |
| 12 | Open questions | Stochastic extension (Freidlin-Wentzell may help escape); periodic-control averaging; richer admissible classes. |
| 13 | Sensitivity to pinned parameters | The model has parameters (μ_dec_B, μ_dec_S, B_dec, S_dec, n_dec) that are **pinned** rather than identified from data, because the observation model can't disambiguate them from the rest. The user's intuition: if these pinned values are off, the closed-island geometry is misshaped and the controllability conclusion may be an artifact of bad pinning. This section sweeps each pinned param ±50% and re-evaluates §7's certificate / §6's constructive attempts. If escape becomes possible under a small relaxation, it's strong evidence the **model itself needs modification** rather than the controller. References [feedback_swat_pinned_params](memory/feedback_swat_pinned_params.md) for the analogous SWAT pinning issue. |
| App A | Numerical-anchor table | μ̄(0; Φ) at the canonical 5 probe points, side-by-side with the existing §10.4 sanity table and the output of `plot_a_sep_landscape.jl`. |
| App B | Composite Lyapunov derivation | Self-contained for the Lean4 reader; adapts Theorem 11.1 of [11_stability_v4.tex](version_1_5_LEAN/LaTex_docs/sections/11_stability_v4.tex) from v4 to v5. |

### Proof strategy (recommended; one path, no alternatives)

**Lead with the comparison-theorem skeleton, then run constructive attempts as falsifiers.**

The central a-priori estimate (§5):
```
μ̄_max(t | y₀, U) ≤ μ_0 + μ_B · B_UB(t; U) + μ_S · S_UB(t; U)
                   − μ_dec_B · h_n(B_UB(t); B_dec) − μ_dec_S · h_n(S_UB(t); S_dec)
```
where `B_UB(t; U) = (1 - e^{-t/τ_B}) · τ_B · κ_B · a_B^max · Φ_B^max + B₀ · e^{-t/τ_B}` is the closed-form solution of the linear B-ODE under the a_B(A) multiplier replaced by its upper bound (conservative). S_UB analogous. The negative contributions (μ_F·F, μ_FF·F², η·A³) are *dropped* — they only make μ smaller, so omitting them gives an honest *upper* bound on μ, hence a *lower* bound on the rate of decay we can avoid.

The escape-impossibility theorem then reads:

> **Theorem (impossibility).** If `∫₀^T μ̄_max(s | y₀, U) ds < ln(A_target / A₀)`, then for every Φ ∈ 𝒰, the deterministic FSA-v5 trajectory satisfies `A(T) < A_target`.

Proof: scalar comparison theorem on dA/dt = μ(t)·A − η·A³ ≤ μ̄_max(t)·A ≤ μ̄_max(t)·A (drop η·A³, conservative); integrate. Standard Mathlib lemma.

Constructive attempts (§6) are then **falsifiers**: each candidate Φ produces a specific lower bound A_LB(T; Φ) — if any A_LB(T; Φ) > A_target, the impossibility theorem is wrong and we have a constructive escape. Three candidates to enumerate:

- **Ramp** Φ ≡ (φ̄, φ̄) constant, varying φ̄ ∈ {0.3, 0.5, 0.7, 1.0}. Closed-form on the slow manifold.
- **Bang-bang** [Φ_max for t ∈ [0, T_burst]; (0.3, 0.3) for t > T_burst], varying T_burst.
- **Early burst then taper**: Φ(t) = Φ_max · max(0, 1 − t/T_burst) + 0.3 · (1 − that).

For each: solve the slow-manifold trajectory analytically, evaluate A(T) via the scalar comparison from below (using μ̄_min along that trajectory), check whether A_LB(T) > A_target. Only one needs to succeed to refute impossibility.

### Mechanisation note

The proof is structured so that Lean4 needs to formalise:
- Existence/uniqueness for the scalar A-comparison (already in `Mathlib.Analysis.ODE.Gronwall`)
- Boundedness of B(t), S(t) under bounded Φ (linear ODE; trivial)
- Monotonicity of Hill function h_n (elementary)
- One **single Float inequality** as the numerical certificate (`norm_num` or external Float check)

The proof does NOT need Pontryagin or Lie-bracket controllability — those are the wrong tools for this question (§7 of the agent's design dismisses them as global-energy not local-manoeuvrability).

## Critical files

**Read for content (LaTeX/Lean/Julia)**:
- [version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex](version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) — top-level model reference (cite, don't re-derive)
- [version_1_5_LEAN/LaTex_docs/sections/14_v5_sedentary_collapse.tex](version_1_5_LEAN/LaTex_docs/sections/14_v5_sedentary_collapse.tex) — `eq:v5-mubar`, `eq:hill`, sanity table to quote verbatim
- [version_1_5_LEAN/LaTex_docs/sections/11_stability_v4.tex](version_1_5_LEAN/LaTex_docs/sections/11_stability_v4.tex) — composite Lyapunov for v4; adapt for App B
- [version_1_5_LEAN/LaTex_docs/sections/05_stability.tex](version_1_5_LEAN/LaTex_docs/sections/05_stability.tex) — earlier stability material
- [version_1_5_LEAN/LaTex_docs/sections/appendix_v5_parameters.tex](version_1_5_LEAN/LaTex_docs/sections/appendix_v5_parameters.tex) — canonical parameter list
- [version_1_5_LEAN/Fsa/V5/Cost.lean](version_1_5_LEAN/Fsa/V5/Cost.lean) — `muBar`, `findASep`, `aSepGrid` Lean reference
- [version_1_5_Julia/models/fsa_v5/simulation_v5.jl](version_1_5_Julia/models/fsa_v5/simulation_v5.jl) lines 80-145 — TRUTH_PARAMS_V5 with centred formulas (canonical numerical anchor)
- [version_1_5_Julia/models/fsa_v5/_dynamics_v5.jl](version_1_5_Julia/models/fsa_v5/_dynamics_v5.jl) — drift/diffusion implementation (verify against tech guide §2)
- [version_1_5_Julia/models/fsa_v5/cost_v5.jl](version_1_5_Julia/models/fsa_v5/cost_v5.jl) — `mu_bar`, `find_a_sep` Julia transcription

**Reference numerical anchors** (already verified by user):
- [version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_a_sep_landscape.jl](version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_a_sep_landscape.jl) — produces μ̄(Φ) values matching the tech guide §10.4 table to 4 dp
- [version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_mu_bar_landscape.jl](version_1_5_Julia/models/fsa_v5/model_notes_and_docs/plot_mu_bar_landscape.jl) — μ̄(0; Φ) heatmap visualising the closed island

**Failed bench output to cite**:
- [outputs/bench_runs/2026-05-11_053431_T100d_UltraFast/v5_traces.png](outputs/bench_runs/2026-05-11_053431_T100d_UltraFast/v5_traces.png) — A → 0.003 by day 100 despite controller upgrades

**Existing figure assets to reuse** (`version_1_5_LEAN/LaTex_docs/figures/`):
- `v5_sedentary_collapse_island.png` — closed-island geometry in (Φ_B, Φ_S)
- `v5_full_bifurcation.png` — basin diagram with transcritical and saddle-node curves
- `v5_trajectories_three_regimes.png` — time-domain A/B/S/F trajectories under the three regimes
- `v4_mu_vs_A_three_regimes.png` — μ vs A landscape illustrating bistability
- `v4_basin_diagram_phiB.png`, `v4_basin_diagram_phiS.png` — basin slices

**Existing tables to reuse** (`version_1_5_LEAN/LaTex_docs/tables/`): the v4 Lyapunov summary referenced from [11_stability_v4.tex](version_1_5_LEAN/LaTex_docs/sections/11_stability_v4.tex).

**New figures the doc may need to add** (deferred to writeup; user's coding freeze means they go to a TODO list, not generated now):
- (B₀, S₀) controllability-region phase plot at fixed A₀ (one figure per U ∈ {[0,1]², [0,3]²}, possibly a 2×3 grid)
- Slow-manifold A_LB(t) trajectories for each constructive Φ candidate in §6

**Outputs**:
- New directory `version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/`
- New file `version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/sedentary_basin_controllability.tex` (compiled to .pdf in the same folder)
- Plan archive copy at `claude_plans/Sedentary_basin_controllability_proof_<YYYY-MM-DD>_<HHMM>.md` with `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` line per the global archival rule in `~/.claude/CLAUDE.md`

## Open questions / risks (flagged in the document)

1. **Cubic damping −η·A³ is dropped from the upper bound** (helpful for impossibility — it would only drive A further down). Document explicitly.
2. **State clipping (A ≥ 0, B,S ∈ [ε, 1−ε]) is in the Julia plant but not in the proof's ODE.** Proof works on unclipped ODE; clipping makes the impossibility *easier* to see (A pinned to 0 once it touches it).
3. **Constructive attempts in §6 might unexpectedly succeed.** If any of the three candidate Φ's gives A_LB(T) > A_target, impossibility theorem is wrong and the document becomes a constructive controllability paper instead. Prior expectation: 70-30 in favour of impossibility, based on μ_init = −0.115 alone.
4. **Stochastic extension** is out of scope but noise might *help* escape (Freidlin-Wentzell large-deviation pathways). Noted in §12.
5. **The proof is for the deterministic ODE.** The bench failure was on the stochastic SDE under SMC²-MPC. The conclusions transfer in expectation but a formal stochastic statement is future work.

6. **Pinned parameters may be the real culprit.** The user's standing intuition (this conversation): the FSA-v5 model has unidentifiable parameters (μ_dec_B, μ_dec_S, B_dec, S_dec, n_dec — the v5 Hill-deconditioning block) that have to be pinned to fixed values because the data can't constrain them. If these pinned values are misspecified, the closed-island topology may be wrong and the impossibility result may be an artefact, not a true model property. §13 of the document addresses this directly via parameter sensitivity sweeps. The mitigation, if §13 confirms the suspicion: **modify the FSA-v5 model** (e.g., shrink μ_dec_B/μ_dec_S, raise B_dec/S_dec to widen the closed island, or use a softer Hill exponent n) rather than trying to make the controller smarter. This is a model-design decision the user takes after seeing the §13 evidence.

## Verification

End-to-end checks for the LaTeX writeup (no new code required):

1. **Anchor every quantitative claim to one of three sources**:
   - Tech guide §10.4 sanity table (5 reference μ̄ values)
   - `plot_a_sep_landscape.jl` console output (already user-confirmed to match)
   - Symbolic derivation traceable to TRUTH_PARAMS_V5 in [simulation_v5.jl:80-145](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L80-L145)

2. **Cross-check the impossibility certificate**: the constant `μ̄_max(t)` for the SEDENTARY_INIT case must be consistent with the empirical "A → 0.003 by day 100" from the bench trace. Specifically: the predicted A(100) from the upper bound should be ≥ 0.003 (since the upper bound on μ corresponds to a *lower* bound on the rate of decay we *cannot avoid*, hence an *upper* bound on A). If the certificate says A(100) ≤ 0.0001 but the bench saw 0.003, something is off and the bound is too loose to be useful.

3. **Sanity-check constructive attempts**: each Φ candidate's slow-manifold A(T) value should be re-derivable by a 5-line numerical script (deferred — user has frozen coding). The LaTeX should expose enough symbolic structure that the user can paste the parameters and re-evaluate.

4. **Lean4 readiness check**: every theorem in the document should have its dependencies traced to existing `Fsa.V5.*` symbols. If a theorem cites a function that doesn't exist in Lean yet, the §11 mechanisation roadmap lists it as "to be added".

5. **Visual verification** (when LaTeX compiles): the (B₀, S₀) controllability-region figure in §9 should show SEDENTARY_INIT (0.05, 0.10) firmly inside the certified non-escape region for U = [0, 1]², and possibly also for U = [0, 3]² depending on how the constructive attempts go.

The success criterion for the writeup: **the user can decide, on the basis of this document alone, whether to (a) abandon SEDENTARY_INIT as an escape target and pick a different starting point, or (b) modify the FSA-v5 model itself to make escape feasible** — without needing to run any further bench experiments.
