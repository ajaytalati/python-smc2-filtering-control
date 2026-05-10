# FSA v1.5 — LaTeX writeup + Match.jl evaluation

> Archived from plan mode: 2026-05-08 10:10.

> Plan only. The build artefacts go in a new `version_1_5_Julia/docs/` folder. Match.jl rewrite scope decided post-investigation.

## Context

The v1.5 build is complete and validated end-to-end:
- All 7 model files in `version_1_5_Julia/models/fsa_high_res/` work and pass smoke tests.
- The FIM gate cleared at rank 7/7, κ=2.95×10⁷ on the realistic 14d window after pinning {τ_B, η, ε_A, μ_FF}.
- Open-loop 28d bench reproduces v1's reference qualitative behaviour: A dips to 0.06 then climbs to 0.16 by day 28; final state matches v1 within MC noise (B=0.18, F=0.10, A=0.16 vs v1 ~0.18/0.11/0.20).
- Closed-loop 28d strong-controller bench produces the same A-dip-then-climb pattern, with A finishing at 0.154.

Two follow-on asks from the user:

**A.** A LaTeX writeup in a new `version_1_5_Julia/docs/` folder. Focus on the **functional semantics** of each file in `models/fsa_high_res/` so that humans and agents can semantically understand what every public function should do. Include a **data flow / computational graph** section. Final section: list **CLI flags** of `tools/bench_smc_full_mpc_fsa_gpu.jl`.

**B.** Project upgrade: rewrite all `models/fsa_high_res/` files using JuliaServices/Match.jl.

This plan covers both, with honest scoping for B based on a Phase-1 audit of the codebase.

---

## Part A — LaTeX writeup (clear scope)

### Files to add

```
version_1_5_Julia/docs/
├── julia_fsa_v15_writeup.tex   # main source
└── julia_fsa_v15_writeup.pdf   # pdflatex output
```

### Style — mirrors v2's writeup conventions

From the Phase-1 audit of [version_2_Julia/docs/julia_fsa_writeup.tex](version_2_Julia/docs/julia_fsa_writeup.tex):
- Document class: `article`, 11pt, a4paper.
- Packages: `geometry`, `amsmath, amssymb, bm`, `listings` (NOT `minted`), `hyperref`, `booktabs`, `enumitem`.
- Code presentation: `\lstset` with line numbers, frame=single, syntax-coloured.
- Build: standard `pdflatex` (no special toolchain).
- v2's writeup is ~1400 LOC / 23 pages. v1.5's should be **shorter** (~600–800 LOC) because v1.5 is simpler — no multi-channel obs, no burst envelope, no circadian C(t), no G1 reparametrisation cancellation issues to debug.

### Section structure

1. **Title + intro** (~50 LOC)
   - One-paragraph "what is v1.5" — functional bridge between v1 (open-loop only) and v2 (closed-loop with rich obs).
   - Status: FIM-gated, validated open- and closed-loop, reproduces v1 reference 28d behaviour qualitatively.

2. **Repository layout** (~30 LOC)
   - Tree of `version_1_5_Julia/` with one-line description per file.

3. **Functional design principles** (~80 LOC)
   - No mutable structs in the public API. Only `struct PlantState` with immutable `SVector{3}`.
   - No `!`-suffix mutating methods. Functions return NEW values.
   - Explicit `key::UInt64` RNG threading (JAX-style, via `StableRNGs.LehmerRNG`).
   - Pure facade over the inherently mutating GPU kernel — `gpu_log_density(target, U, grid_obs, key) → Vector{Float64}` allocates a fresh output, dispatches the kernel, returns. Caller never sees the CuArray mutation.
   - I/O isolated to the bench's top-level `main()` — JLD2 / PNG writes happen exactly once at the end.
   - Concrete table: v1/v2 imperative pattern → v1.5 functional equivalent.

4. **Per-file semantic specification** (~250 LOC) — the meat of the document
   For each file in `models/fsa_high_res/`, give:
   - **Purpose** — one-sentence summary.
   - **Public exports** — name + type signature + ~3-line description per export.
   - **Semantic contracts** — for each public function: inputs, outputs, invariants (purity, output shape), failure modes.
   - **Internal helpers** — brief description.

   Files (in module load order):
   - `_dynamics.jl` — G0 SDE: `drift`, `diffusion_state_dep`, `em_step_substepped`, `TRUTH_PARAMS`. Verbatim from v1; note this.
   - `simulation.jl` — owns BINS_PER_DAY (parsed from env at module load), DEFAULT_PARAMS (14 dynamics + 3 obs noise in v1.5 basis), INIT_STATE, PINNED_PARAMS (4 pinned at truth post-FIM-gate), `sample_obs_bfa`, `params_v15_to_v1_nt` (basis adapter), `fill_pinned_nt` (10-estimated → 14-field merger).
   - `_plant.jl` — `PlantState` (immutable record), `plant_step` (pure single-bin EM step + Gaussian obs), `plant_rollout` (pure stride rollout). Verbatim functional design from the FIM-gate plan.
   - `gpu_control.jl` — verbatim from v1; only export note: `FSAv1ControlGPUTarget`, `gpu_cost_log_density_batched`, `make_log_density_fn`. Note that the cost is Eq 37 (no λ_Φ·∫Φ² term, contrary to v1's CPU `control.jl`).
   - `estimation.jl` — `PARAM_NAMES` (10 estimated), `PARAM_PRIOR_CONFIG`, `propagate` (pure prior-predictive), `obs_log_weight` (3-channel diagonal Gaussian). Note the 4 dynamics params (τ_B, η, ε_A, μ_FF) are NOT in PARAM_NAMES per FIM-gate decision.
   - `gpu_pf.jl` — `FSAGPUTarget` (immutable), `gpu_log_density`, `gpu_grads`, `parallel_hmc_one_move`. All public functions take `key::UInt64` and return new outputs. Internal kernel `propagate_segment_kernel!` is the imperative core, hidden behind the pure facade.
   - `FSAHighRes.jl` — module aggregator, re-exports.

5. **Data flow & computational graph** (~120 LOC)
   - Step-by-step text walkthrough of one closed-loop stride iteration in `bench_smc_full_mpc_fsa_gpu.jl :: stride_step`:
     1. Slice next stride's Φ from the current plan.
     2. `plant_rollout` produces `(final_state, traj, obs_B, obs_F, obs_A)`.
     3. `window_grid_obs` carves the most recent window from accumulated history.
     4. `run_outer_smc` (tempered SMC² + parallel HMC) returns `(U_post, n_temp)`.
     5. On replan boundary: `posterior_mean_v15` → `fill_pinned_nt` → `params_v15_to_v1_nt` → `controller_plan` → new `Phi_plan`.
     6. New accumulator returned (immutable).
   - **Filter dataflow**: `U_unc` (M, 10) → `_to_v15_constrained_nt` → `fill_pinned_nt` → `params_v15_to_v1_nt` → `params_per_chain` (M, 14) CuArray → `propagate_segment_kernel!` → `log_w` → `run_segmented_smc_step!` (framework) → `log_lik_acc`.
   - **Controller dataflow**: posterior θ_dyn → params_v1 NamedTuple → `FSAv1ControlGPUTarget` constructor → `run_tempered_smc_gpu` (framework) → posterior θ_ctrl → RBF decode → per-bin Phi.
   - **Diagram** — text-based ASCII flowchart showing the `foldl` over strides with conditional replan.

6. **CLI flags reference** (~80 LOC)
   - `booktabs` table of all 23 flags from `bench_smc_full_mpc_fsa_gpu.jl`, grouped:
     - **Bench / filter** (15 flags): T-days, step-minutes, replan-K, N-smc, K-per-chain, num-mcmc, hmc-step-size, hmc-leapfrog, max-lambda-inc, target-ess-frac, max-temp-levels, seed, output-dir, open-loop.
     - **Controller** (9 flags, all newly CLI-exposed): ctrl-n-smc, ctrl-n-inner, ctrl-num-mcmc, ctrl-hmc-step, ctrl-hmc-leap, ctrl-chees-max, ctrl-max-levels, ctrl-target-nats, ctrl-sigma-prior.
   - Each row: flag name, default, type, one-line meaning.
   - Two example invocations: `--open-loop true` for diagnostic; default closed-loop.

7. **Lean4-port mapping appendix** (~120 LOC) — added because the Match.jl rewrite is motivated by future Lean4 port for formal verification.
   - **Why this section exists**: the rewrite from Julia + Match.jl into Lean4 + `match ... with` is structurally near-mechanical, but a few non-trivial decisions remain (record vs inductive, eltype-generic vs Float64, etc.). This section pre-documents those choices.
   - **Side-by-side table**: 8 entries, one per rewrite site, showing the Match.jl source and the proposed Lean4 equivalent. Format:
     ```
     site                          Julia (Match.jl)                 Lean4 (match ... with)
     ----------------------------- -------------------------------- --------------------------
     params_v15_to_v1_nt           @match p begin                   match p with
                                       ::Dict => ...;                 | .DictForm d => ...
                                       ::NamedTuple => ...;           | .NamedTupleForm nt => ...
                                   end
     _reflect_unit                 @match x begin                   match x with
                                       x, if x<0 end => -x;          | x => if x < 0 then -x
                                       x, if x>1 end => 2-x;         | x => if x > 1 then 2-x
                                       _ => x;                        | _ => x
                                   end
     ...                           ...                              ...
     ```
   - Notes on type-system gotchas (Lean4 needs explicit types; Float64 vs Float; SVector vs `Vector Float Nat`).
   - Out of scope: porting the GPU kernels themselves (KernelAbstractions has no Lean4 equivalent).

### Verification (Part A)

- `cd version_1_5_Julia/docs && pdflatex julia_fsa_v15_writeup.tex` builds without errors.
- Resulting PDF has the 6 sections, no overfull `\hbox` warnings, code listings render with line numbers.
- Section 4 covers all 7 files in `models/fsa_high_res/`.
- Section 6 covers all 23 CLI flags.

---

## Part B — Match.jl rewrite (Lean4-port preparation)

### Motivation (USER-CONFIRMED)

The user wants Match.jl applied **as a stepping stone to a future Lean4 port for formal verification**. This changes the value calculus entirely:

- For pure code-clarity: Match.jl is mostly "wash" on v1.5's already-functional code (per the audit below).
- For **Lean4 port preparation**: Match.jl is a **high-value win**, because Lean4's native pattern-matching syntax (`match x with | pattern => result`) is structurally near-identical to Match.jl's (`@match x begin pattern => result end`). Writing the Julia in Match.jl idioms makes the future Julia → Lean4 port nearly mechanical at every pattern site.

So the rewrite scope flips to **Option B-full** — apply `@match` at every site where the eventual Lean4 port will need pattern matching, regardless of whether it improves Julia readability today. The new criterion is:

> *"Will the Lean4 version of this site use `match ... with`?"*

If yes, write it in Match.jl now.

### Phase-1 audit findings (re-cast for Lean4 portability)

The same sites the original audit identified, now re-graded for Lean4 portability:

| Site | Current pattern | Lean4-port verdict |
|---|---|---|
| `simulation.jl :: params_v15_to_v1_nt` | Two methods (Dict / NamedTuple) | **PORT** — Lean4 will dispatch on a sum type (`inductive Params := DictForm | NamedTupleForm`). `@match p begin ::Dict => ...; ::NamedTuple => ...; end` maps 1:1. |
| `simulation.jl :: fill_pinned_nt` | Single method, named-field destructure | **PORT** — Lean4 destructures records via `match` patterns. `@match estimated begin (tau_F, B_inf, ...) => ... end` mirrors the Lean4 form. |
| `_plant.jl :: _reflect_unit` | Ternary `x < 0 ? -x : (x > 1 ? 2-x : x)` | **PORT** — Lean4: `match x with \| x if x < 0 => -x \| x if x > 1 => 2-x \| _ => x`. Match.jl with guards mirrors this exactly. |
| `_plant.jl :: plant_step` | Pure single-bin step; params-Dict → NamedTuple via adapter | **PORT** — Lean4 will pattern-match on the params record. The current Dict-key access `params[:sigma_B]` becomes `match params with \| { sigma_B, ... }`. Rewriting in Match.jl now sets the structure for the Lean4 port. |
| `gpu_pf.jl :: _to_v15_constrained_nt` | Vector→NamedTuple, 10 fields, exp on each | **PORT** — Lean4 destructures the vector via list pattern. `@match u begin [u1, u2, ..., u10] => ... end` is the natural Match.jl form, maps to Lean4's `match u with \| [u1, u2, ..., u10]`. |
| `gpu_pf.jl :: _pack_v1_row!` | NamedTuple→row write, 14 fields | **PORT** — same logic; pattern-match the v1 NamedTuple, write each field. Lean4 will write a similar `def pack_v1_row (p : V1Params) (row : ...) ...` with a destructured pattern. |
| `gpu_pf.jl :: gpu_log_density` last-segment branch | `is_last = ...; if is_last ... else ... end` | **PORT** — Lean4 will pattern-match on `(seg, n_segments)` to decide accumulator branch. `@match (seg, n_segments) begin (s, n) where s == n => ...; _ => ... end`. |
| `gpu_pf.jl :: parallel_hmc_one_move` accept/reject | Per-chain `if log(rand) < log_α[m]` loop | **PORT** — Lean4 will pattern-match on Bernoulli outcome (accept | reject) to update U. `@match outcome begin :accept => U_new[m,:]; :reject => U[m,:] end`. |
| `estimation.jl :: PARAM_PRIOR_CONFIG` consumers | Currently a flat list of tuples; consumers iterate | **PORT** — Lean4 will use a sum type for prior kinds (`inductive PriorKind := LogNormal \| Normal`). Refactor into a generic `apply_prior(kind, μ, σ, u)` that `@match`-dispatches on `kind ∈ {:LogNormal, :Normal}`. This is the Lean4-natural form. |
| `_dynamics.jl`, `gpu_control.jl` (v1 verbatim) | Pure math + KernelAbstractions kernels | **DON'T PORT** — these stay verbatim per v1.5's plan. Lean4 portability for kernel code is a separate, much harder problem (formal verification of fp32 GPU kernels). Out of scope for this round. |
| `_dynamics.jl :: drift / diffusion_state_dep` | Pure math (no branching) | **DON'T PORT** — straight arithmetic, no patterns. Lean4 will keep the same shape. |

**Net assessment for Lean4 portability**: 8 sites across 4 files (`simulation.jl`, `_plant.jl`, `gpu_pf.jl`, `estimation.jl`) become Match.jl idiomatic. ~150–200 LOC of rewrites. Each rewrite has a direct Lean4 analogue.

### Files affected by the rewrite (B-full)

| File | Rewritten functions | Approx LOC change |
|---|---|---|
| `simulation.jl` | `params_v15_to_v1_nt` (collapse 2 methods → 1 @match), `fill_pinned_nt` (record destructure), `sample_obs_bfa` (no change — no patterns) | +30 LOC |
| `_plant.jl` | `_reflect_unit` (ternary → @match), `plant_step` (params destructure → @match) | +25 LOC |
| `gpu_pf.jl` | `_to_v15_constrained_nt` (vector → @match), `_pack_v1_row!` (NamedTuple → @match), `gpu_log_density` (last-segment branch → @match), `parallel_hmc_one_move` (accept/reject → @match) | +80 LOC |
| `estimation.jl` | refactor PARAM_PRIOR_CONFIG consumers into a `apply_prior(kind, μ, σ, u)` function with @match on `kind`. Update `propagate` and `obs_log_weight` to use it where applicable. | +40 LOC |
| `_dynamics.jl`, `gpu_control.jl` | (unchanged — v1 verbatim, no patterns to port) | 0 |
| `FSAHighRes.jl` | (unchanged — module aggregator only) | 0 |

Total rewrite size: ~175 LOC of new pattern-matching code across 4 files.

### Implementation order

1. **Dependency**: add `Match` to `version_1_5_Julia/Project.toml`. UUID `7eb4fadd-790c-5f42-8a69-bfa0b872bfbf`. Resolve via `julia --project=. -e "using Pkg; Pkg.add(\"Match\")"`.
2. **Verify import**: `julia --project=. -e "using Match; @match 1 begin 1 => :ok; _ => :no end"` should print `:ok`.
3. **Rewrite simulation.jl** — start here because it has the simplest pattern (Dict|NamedTuple dispatch). Run `tools/fim_check.jl` after to confirm parameters still flow correctly.
4. **Rewrite _plant.jl** — `_reflect_unit` and `plant_step`. Run `tools/psim_sanity.jl` after to confirm trajectories haven't changed.
5. **Rewrite gpu_pf.jl** — the four sites identified. Run `tools/test_gpu_pf.jl` after to confirm all 6 smoke tests still pass.
6. **Rewrite estimation.jl** — `apply_prior` refactor + PARAM_PRIOR_CONFIG consumer updates. Re-run `tools/test_gpu_pf.jl`.
7. **End-to-end check** — run `tools/bench_smc_full_mpc_fsa_gpu.jl --T-days 14 --open-loop true --N-smc 8 --K-per-chain 100 --num-mcmc 1 --hmc-leapfrog 2 --max-lambda-inc 0.30` (~4 min). Confirm final state within MC noise of the pre-rewrite run.
8. **Lean4-port mapping doc** — add §7 to the writeup specifically for this. Side-by-side table: Julia Match.jl idiom → Lean4 `match ... with` equivalent. One example per rewrite site. This is the artefact that makes the future port mechanical.

### Verification (Part B)

- All 6 tests in `tools/test_gpu_pf.jl` pass after every file rewrite (`PASS ✓` for T1–T6).
- `tools/fim_check.jl` still reports rank 7/7, κ ≤ 10⁸ at the realistic 14d gate.
- `tools/psim_sanity.jl` produces a trajectory PNG within MC noise of the pre-rewrite version (visual check).
- `tools/bench_smc_full_mpc_fsa_gpu.jl` open-loop 14d run finishes in ≤ 5 min and produces a final state within ±5 % of the pre-rewrite run on each of (B, F, A).
- Writeup §7 has 8 side-by-side Julia/Lean4 mappings, one per rewrite site.

### Verification (Part B)

- `tools/test_gpu_pf.jl` passes all 6 tests after the rewrite.
- The open-loop 14d bench produces a final state within MC noise of the pre-rewrite run (B≈0.09, F≈0.13, A≈0.08 at the existing fast config).
- `pdflatex` builds the writeup successfully and §3 reflects the chosen Match.jl scope.

---

## Decisions locked in (user-confirmed)

- **Match.jl scope**: **B-full** — rewrite every applicable site in `simulation.jl`, `_plant.jl`, `gpu_pf.jl`, `estimation.jl`. Motivation: prepare the codebase for a future port to **Lean4 for formal verification**. Lean4's `match ... with` syntax is structurally near-identical to Match.jl's `@match`, so the rewrite makes the future port nearly mechanical.
- `_dynamics.jl` and `gpu_control.jl` (v1 verbatim) are NOT rewritten — they have no pattern-matching opportunities and are out of scope for Lean4 portability in this round (formal verification of fp32 GPU kernels is a separate, much harder problem).
- The writeup will include a §7 Lean4-mapping appendix with side-by-side Julia/Lean4 examples per rewrite site.

## Critical files (READ-ONLY for Part A)

- [version_2_Julia/docs/julia_fsa_writeup.tex](version_2_Julia/docs/julia_fsa_writeup.tex) — writeup style template.
- [version_1_5_Julia/models/fsa_high_res/_dynamics.jl](version_1_5_Julia/models/fsa_high_res/_dynamics.jl)
- [version_1_5_Julia/models/fsa_high_res/simulation.jl](version_1_5_Julia/models/fsa_high_res/simulation.jl)
- [version_1_5_Julia/models/fsa_high_res/_plant.jl](version_1_5_Julia/models/fsa_high_res/_plant.jl)
- [version_1_5_Julia/models/fsa_high_res/gpu_control.jl](version_1_5_Julia/models/fsa_high_res/gpu_control.jl)
- [version_1_5_Julia/models/fsa_high_res/estimation.jl](version_1_5_Julia/models/fsa_high_res/estimation.jl)
- [version_1_5_Julia/models/fsa_high_res/gpu_pf.jl](version_1_5_Julia/models/fsa_high_res/gpu_pf.jl)
- [version_1_5_Julia/models/fsa_high_res/FSAHighRes.jl](version_1_5_Julia/models/fsa_high_res/FSAHighRes.jl)
- [version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) — for the CLI flag table.

## Files to add

- `version_1_5_Julia/docs/julia_fsa_v15_writeup.tex` (~700 LOC).
- `version_1_5_Julia/docs/julia_fsa_v15_writeup.pdf` (built via pdflatex).

## Files to modify (Part B-full)

- `version_1_5_Julia/Project.toml` — add `Match` dependency.
- `version_1_5_Julia/models/fsa_high_res/simulation.jl` — `params_v15_to_v1_nt`, `fill_pinned_nt`.
- `version_1_5_Julia/models/fsa_high_res/_plant.jl` — `_reflect_unit`, `plant_step`.
- `version_1_5_Julia/models/fsa_high_res/gpu_pf.jl` — `_to_v15_constrained_nt`, `_pack_v1_row!`, `gpu_log_density` last-segment branch, `parallel_hmc_one_move` accept/reject.
- `version_1_5_Julia/models/fsa_high_res/estimation.jl` — `apply_prior` refactor + consumer updates.

NOT modified: `_dynamics.jl`, `gpu_control.jl`, `FSAHighRes.jl`.
