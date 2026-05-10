# LaTeX → Lean4 → Julia pipeline documentation (FSA v1.5)

> Archived from plan mode: 2026-05-09 16:08.

## Context

The pipeline that takes the FSA v1.5 mathematical specification (LaTeX equations) → Lean4 typed definitions → Julia model code → differential verification is currently undocumented and not reproducible by anyone except the author. The user wants a **practical, evidence-based, traceable** writeup that any future agent or human collaborator can follow end-to-end.

The new doc must reuse already-written prose where it exists (model description, SMC²-MPC pipeline, CLI, functional-Julia rationale) and add original content for the parts that aren't documented yet (the build commands, the JSON dispatch protocol, the diff-test harness, the verification gap on the GPU files).

It complements [`lean4_first_charter.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/lean4_first_charter.tex) (the policy doc); this new file is the *implementation* doc.

## Output

- **Path:** `/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/lean4_to_julia_pipeline.tex`
- **Reason for that location:** sits next to the existing `lean4_first_charter.tex` and reuses the same `preamble.tex` (already loads `tikz`, `listings`, `hyperref`, `amsmath`, `booktabs`, `bm`).
- **Build:** `cd version_1_5_LEAN/LaTex_docs && latexmk -pdf lean4_to_julia_pipeline.tex` — same pattern that already builds the charter cleanly.
- **Target length:** ~10–12 pages, 8 sections.

## Document structure

### Imported sections (lifted verbatim, light header touch-ups)

| §  | Title                                       | Source `.tex` (line range)                                                                                     | Self-contained? |
| -- | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | --------------- |
| 1  | The FSA v1.5 model                          | [`v15_horizon_sweep_writeup.tex:52–170`](/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/docs/v15_horizon_sweep_writeup.tex) | yes (has `tab:params`) |
| 2  | SMC²-MPC pipeline and cost functional       | `v15_horizon_sweep_writeup.tex:172–250`                                                                         | yes (has `eq:cost`) |
| 3  | Study configuration and CLI                 | `v15_horizon_sweep_writeup.tex:252–362`                                                                         | yes (has `tab:filter_cfg`, `tab:ctrl_cfg`); CLI example hardcodes `/home/ajay/...` at source line 276 — leave as-is |
| 4  | Functional programming paradigm in Julia    | [`julia_vs_python_v15_writeup.tex:186–260`](/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.tex) | yes |

`\cite{techguide}` and `\cite{writeup}` are referenced inside §1–3. Lift the two bib entries from `v15_horizon_sweep_writeup.tex:634–643` into a `\begin{thebibliography}` block at the end of the new doc (or replace the citations with footnoted file paths if cleaner — decide at write time).

### New sections (original content)

#### §5 — From LaTeX to Lean4: the formal specification layer

- The LaTeX math specification is **§1 of this document** — the doc is self-contained, no external `.tex` file lookups required. Reference §1 by `\ref{sec:fsa-model}` (or whatever label §1 ends up with).
- Lean4 module layout in [`version_1_5_LEAN/Fsa/V15/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V15/):
    - `Types.lean` — `PlantState`, `ParamsV15`, `ParamsV1`
    - `Truth.lean` — `DEFAULT_PARAMS`, `INIT_STATE`
    - `Adapters.lean` — `paramsV15ToV1` (basis rotation; the bridging function)
    - `Dynamics.lean` — `drift`, `diffusion_state_dep`, `em_step_substepped`, `reflect_unit`
    - `Plant.lean` — `plant_step`
    - `Estimation.lean` — `obs_log_weight_one`, `apply_prior`
- Build command: `cd version_1_5_LEAN && lake build` → `.lake/build/bin/fsa_v15_cli`.
- CLI dispatch protocol from [`Main.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Main.lean): one JSON object per line on stdin, one per line on stdout. Table of all 8 dispatch tags (`drift`, `diffusion`, `emStep`, `reflectUnit`, `plantStep`, `obsLogWeight`, `paramsV15ToV1`, `applyPrior`) with the file:line of each handler in `Main.lean:166–218`. Include one worked example: a `"drift"` JSON request and its JSON response.

#### §6 — From Lean4 to Julia: the executable port layer

- One-to-one map from each Lean4 function to its Julia twin in [`version_1_5_Julia/models/fsa_high_res/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/), with file:line:

    | Lean4 function                  | Julia function                                | Tested? |
    | ------------------------------- | --------------------------------------------- | :------: |
    | `Dynamics.drift`                | `_dynamics.jl:58–67`                           | ✓ |
    | `Dynamics.diffusion_state_dep`  | `_dynamics.jl:79–86`                           | ✓ |
    | `Dynamics.em_step_substepped`   | `_dynamics.jl:100–119`                         | ✓ |
    | `Dynamics.reflect_unit`         | `_plant.jl:57–61`                              | ✓ |
    | `Plant.plant_step`              | `_plant.jl:75–107`                             | ✓ |
    | `Estimation.obs_log_weight_one` | `estimation.jl:91–120` (per-particle slice)    | ✓ |
    | `Adapters.paramsV15ToV1`        | `simulation.jl:110–143` (`@match` site #1)     | ✓ |
    | `Estimation.apply_prior`        | `gpu_pf.jl:179–200` (`@match` site #3)         | ✓ |

- Three `@match` sites (the highest-risk LaTeX → Julia transitions; named in test:120–163 by `@match site #N`).
- Functions present in Julia with no Lean4 twin (array/wrapper layer; correctness inherits from per-particle core): `Plant.plant_rollout`, `Estimation.propagate`, `PARAM_NAMES`, `PARAM_PRIOR_CONFIG`.

#### §7 — The differential test harness — verifying parity

End-to-end walk-through of [`version_1_5_Julia/diff_test/test_lean_diff_v15.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v15.jl):

- Long-lived subprocess (`open_lean_client`, `test_lean_diff_v15.jl:42–63`) — one `lake build` binary, one process, JSON line-buffered.
- The `round_trip` helper sends one JSON, reads one JSON, asserts non-empty.
- **Pre-drawn noise convention** (`test_lean_diff_v15.jl:107–109`): both sides receive identical noise so RNG-divergence is excluded — the test is purely about deterministic numerical agreement.
- **Tolerances**: `1e-6` single-step (drift, diffusion, reflect, prior, obs_log_weight, plantStep, paramsV15ToV1); `1e-4` integrated (em_step_substepped only).
- Per-testset coverage table (8 testsets, ~5 random + 0–15 corner inputs each).
- Run command:
  ```
  cd version_1_5_LEAN && lake build && \
    cd ../version_1_5_Julia && julia --project=. diff_test/test_lean_diff_v15.jl
  ```
- **Flow diagram (TikZ)** — one `tikzpicture`. Boxes:

  ```
  LaTeX math spec
       │
       ▼
  Lean4 modules (Fsa/V15/{Types,Truth,Adapters,Dynamics,Plant,Estimation}.lean)
       │ lake build
       ▼
  fsa_v15_cli   ←─┐
       │          │ JSON line protocol
       ▼          │
  test_lean_diff_v15.jl   ←─── Julia models/fsa_high_res/{_dynamics,_plant,
       │                       estimation,simulation}.jl
       ▼ PASS
  SMC2FC_functional framework
       │
       ▼
  GPU layer (gpu_pf.jl, gpu_control.jl)   ⚠ NOT diff-tested
  ```

  Render in TikZ with `node[draw, rectangle]` boxes and arrows; the GPU layer in a dashed red box. Don't over-engineer the diagram — readability over polish.

#### §8 — Verification gap: what the diff test does not cover

Concrete table (built from the audit findings):

| Layer | File:line | What it does | In Lean4? |
| --- | --- | --- | :--: |
| Control cost rollout | `gpu_control.jl:fsa_v1_cost_kernel!` (line 54–147) | RBF-parameterised Φ(t), substepped EM, soft fatigue penalty | ✗ |
| Segmented PF loop | `gpu_pf.jl:propagate_segment_kernel!` (line 53–143) | per-particle SDE + obs log-weight, NaN guard, multi-bin loop | ✗ outer loop; ✓ inner SDE math (duplicated) |
| HMC leapfrog | `gpu_pf.jl:parallel_hmc_one_move` (line 555–610) | per-chain leapfrog with prior gradient | ✗ |
| FD gradient batcher | `gpu_pf.jl:gpu_grads` (line 506–543) | central-difference gradient on GPU | ✗ |
| NaN guard fallback | `gpu_pf.jl:120` | reset particle to init on NaN | ✗ (Lean4 `Float` ops don't NaN-fallback) |
| OT rescue, ESS | `gpu_pf.jl:270–298` | resampling-rescue heuristics | ✗ |

Two qualifying notes:
1. The per-particle SDE math inside the GPU kernels (drift, diffusion, reflect, obs_log_weight) duplicates the diff-tested Julia core line-by-line. So in principle the GPU kernels could be diff-tested by exposing a single-thread debug path. Today they aren't.
2. What the GPU code is currently trusted on: closed-loop bench parity with the JAX/Python reference (cite `julia_vs_python_v15_writeup`) — empirical end-to-end validation, not formal parity.

Mark this section explicitly as **future work**, not a bug.

## Files to read while writing

- Source 1, sections 1–3: [`v15_horizon_sweep_writeup.tex:52–362`](/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/docs/v15_horizon_sweep_writeup.tex) + bib at lines 634–643.
- Source 2, section 4: [`julia_vs_python_v15_writeup.tex:186–260`](/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.tex).
- [`Main.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Main.lean) — JSON dispatch protocol.
- [`Fsa/V15/*.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V15/) — module-level summaries.
- [`test_lean_diff_v15.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v15.jl) — harness walk-through.
- [`gpu_pf.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_pf.jl), [`gpu_control.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_control.jl) — gap section evidence.

## Existing utilities to reuse

- [`version_1_5_LEAN/LaTex_docs/preamble.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/preamble.tex) — already loads tikz / listings / hyperref / amsmath / booktabs / bm. The new doc does `\input{preamble.tex}`; no path tricks needed.
- TikZ `\usetikzlibrary{positioning, arrows.meta, shapes.geometric, fit}` already declared in the preamble — sufficient for the flow diagram.
- Same `latexmk -pdf` build pattern already proven on `lean4_first_charter.tex`.

## Verification

After writing, build:
```
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs
latexmk -C lean4_to_julia_pipeline.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error lean4_to_julia_pipeline.tex
```

Pass criteria:
- 0 LaTeX errors.
- 0 undefined references (`grep -E "Warning|Error|\?\?" lean4_to_julia_pipeline.log` returns 0 lines).
- Flow diagram renders cleanly (visual check by opening the PDF — every box has text, every arrow lands).
- Internal cross-references (`eq:cost`, `tab:params`, `tab:filter_cfg`, `tab:ctrl_cfg`, plus any new ones in §5–8) all resolve inside the new doc.
- Page count ~10–12.

Spot checks:
- `grep -nE "Python|JAX" lean4_to_julia_pipeline.tex` → only contextual mentions (e.g. "JAX/Python reference for end-to-end empirical validation"), no stale "Python is the implementation" framing.
- The §5 dispatch table and the §6 Lean4↔Julia map should agree on the 8 verified functions.

## Archive (per CLAUDE.md rule)

After exiting plan mode and starting execution, copy this plan file from `~/.claude/plans/i-want-you-to-cozy-hoare.md` into `claude_plans/LaTeX_to_Lean4_to_Julia_pipeline_documentation_<YYYY-MM-DD>_<HHMM>.md` with the `> Archived from plan mode:` line near the top. Keep the archive in sync if the plan evolves during execution.

## Out of scope (deliberate)

- Re-running any benches to refresh numbers in §1–3 — those were measured at the time of the source documents and are reused as-is.
- Implementing GPU diff-test paths to close the §8 gap — that's a separate code-level workstream, not a doc change.
- Changing the Lean4 module layout, the JSON protocol, or any Julia code — read-only documentation only.
