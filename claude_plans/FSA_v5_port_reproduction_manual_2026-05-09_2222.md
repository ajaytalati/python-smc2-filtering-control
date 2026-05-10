# FSA v5 port reproduction manual — LaTeX doc

> Archived from plan mode: 2026-05-09 22:22.

## Overview (what this plan is, in one paragraph)

This plan describes a LaTeX document — the **FSA v5 port reproduction
manual** — that walks a fresh human or AI agent through reproducing
or verifying the just-completed port of the FSA v5 model from
Lean4 into the Julia stack. The reader of the resulting manual
starts with two on-disk inputs (the v5 technical guide and the
Lean4 source files at `Fsa/V5/`) and ends with a working closed-loop
SMC²-MPC bench whose end-to-end run produces the same artefact set
as the verified reference run on 2026-05-09. The manual is
deliberately operational ("do X, verify Y, then do Z"), not a
narrative log of what happened.

## Motivation

Three things make a reproduction manual necessary at this point:

1. **The work to date is captured only in conversational history and
   in a chronological PORT_LOG.md.** Neither is a stable artefact a
   future reader can pick up and execute without reverse-engineering
   the order of operations and the rationale at each step.

2. **AI agents are lossy mediums for code.** The LEAN4-First Charter
   exists precisely because pattern-matching agents introduce
   transcription errors when porting math. The diff-test catches
   transcription errors at machine precision, but only IF a future
   port is structured so the diff-test gates apply. A manual that
   prescribes the structure, the gates, and the per-phase pass
   criteria preserves the verifiability of the work.

3. **Auditability.** The bench result that downstream researchers
   rely on (recovered θ posteriors, controller plans, A_acc /
   F-barrier traces) is only as trustworthy as the pipeline that
   produced it. The manual is the pipeline's contract: "if you
   reproduce these steps and hit these gates, you have the same
   pipeline I used."

The previous PORT_LOG.md remains as the chronological record of
WHAT was done; the manual is the reproducible HOW-TO. Together they
play the same WHAT/HOW pair that `FSA_version_5_technical_guide.tex`
and `lean4_to_julia_pipeline.tex` play at the architecture level.

## Flow-map: where this manual fits in the doc ecosystem

The four LaTeX docs sitting in `version_1_5_LEAN/LaTex_docs/` answer
four orthogonal questions:

```
                  ┌──────────────────────────────────┐
                  │ LEAN4-First Charter              │  WHY  (policy / rationale)
                  │ lean4_first_charter.tex          │
                  └──────────────────────────────────┘
                                  │ motivates
                                  ▼
                  ┌──────────────────────────────────┐
                  │ FSA v5 Technical Guide           │  WHAT (the model maths;
                  │ FSA_version_5_technical_guide.tex│        the input)
                  └──────────────────────────────────┘
                                  │ specifies
                ┌─────────────────┴─────────────────┐
                ▼                                   ▼
   ┌──────────────────────────┐     ┌────────────────────────────────┐
   │ LaTeX→Lean4→Julia        │     │ FSA v5 PORT REPRODUCTION       │
   │ pipeline doc             │     │ MANUAL  (THIS NEW DOC)         │
   │ lean4_to_julia_pipeline  │     │ fsa_v5_port_reproduction_      │
   │ .tex                     │     │ manual.tex                     │
   │                          │     │                                │
   │ HOW (architecturally) —  │     │ HOW (operationally) —          │
   │ what the components are  │     │ step-by-step recipe with       │
   │ and why they exist       │     │ per-phase verification gates   │
   └──────────────────────────┘     └────────────────────────────────┘
```

Inside the new manual, the seven port phases form a directed graph.
Phases on the same row run in parallel; arrows mean "must complete
before":

```
 Phase 0      Phase 1                Phase 3       Phase 4         Phase 5         Phase 6
 (inputs)     (Lean4 CLI)            (diff test)   (GPU port)      (bench driver)  (plotters +
 ┌─────┐      ┌───────────────┐      ┌──────────┐  ┌────────────┐  ┌────────────┐  launcher +
 │tech │ ──┬──│ Main_v5.lean  │──┐   │ ~15 test │  │ gpu_pf_v5 +│  │ tools_v5/  │  gating)
 │guide│   │  │ + lakefile    │  │   │ sets,    │  │ gpu_control│  │ + bench_   │  ┌────────┐
 │+    │   │  │ → fsa_v5_cli  │  ├──→│ 401 → 424│─→│ _v5,       │─→│ glue_v5 +  │─→│ 3 PNGs │
 │Lean4│   │  └───────────────┘  │   │ asserts  │  │ single-thr │  │ driver +   │  │ + log  │
 │code │   │                     │   │ green    │  │ debug      │  │ launchers  │  └────────┘
 └─────┘   │  Phase 2            │   └──────────┘  │ entries    │  └────────────┘
           │  (Julia CPU port)   │                 └────────────┘
           │  ┌───────────────┐  │
           └──│ 8 .jl files,  │──┘
              │ smoke-loaded  │
              └───────────────┘
```

Verification gates per phase are mechanical (single command, single
expected output line); they're listed in §10 of the manual.

## Context

The FSA v5 port from Lean4 to Julia is complete and verified. Phase
sequence (1–6 + 6b) produced: a Lean4 CLI binary, 8 CPU model files,
a 424-assertion differential test, two GPU files (PF + control), a
bench driver, three plotters, and a launcher. The first end-to-end
T=7 bench run finished cleanly in 362 s on 2026-05-09T22:01:01 with
all artefacts produced (`data.jld2`, three PNGs, per-stride CSV,
manifest, recovered final state).

The user now wants a **LaTeX instruction manual** that lets a future
human or AI agent — given only the FSA v5 technical guide PDF/TeX
and the Lean4 model files in `Fsa/V5/` — reproduce or verify the
port end-to-end. Critically: **not a log** ("I did X then Y") but
an **operational manual** ("do X, then verify Y, then do Z"). Heavy
on imperative steps, commands, and verification gates; light on
narrative prose.

The previous PORT_LOG.md (in `models/fsa_v5/`) is the chronological
record of WHAT was done. The manual is the reproducible HOW-TO.

## Output

- **Path:** `version_1_5_LEAN/LaTex_docs/fsa_v5_port_reproduction_manual.tex`
- **Sits alongside** `lean4_first_charter.tex` (policy doc),
  `lean4_to_julia_pipeline.tex` (architecture doc), and
  `FSA_version_5_technical_guide.tex` (model maths). The four LaTeX
  docs together cover: WHY (charter), WHAT (tech guide), HOW
  ARCHITECTURALLY (pipeline doc), HOW OPERATIONALLY (this manual).
- **Build:** `cd version_1_5_LEAN/LaTex_docs && latexmk -pdf fsa_v5_port_reproduction_manual.tex`
- **Reuses** the existing `preamble.tex` (already loads tikz,
  listings, hyperref, amsmath, booktabs, bm). Adds `\usepackage{enumitem}`
  inline for tighter imperative lists.
- **Target length**: ~25–30 pages.

## Document structure

### Front matter

- Title + author + date
- Abstract: one paragraph stating the input materials, the output
  artefacts, the verification gates, and citing the reference T=7
  run (362 s wall, 2026-05-09).

### §1 — Goal & overview

Two-paragraph orientation at the top of the manual:

- **One-sentence goal:** "Given the FSA v5 tech guide and
  `Fsa/V5/*.lean`, produce a verified Julia port that runs the
  closed-loop SMC²-MPC bench end-to-end."
- **One-paragraph motivation** (mirrors the *Motivation* section
  of this plan): why a structured manual exists at all — the prior
  work is captured only in conversational history + a chronological
  PORT_LOG; AI agents are a lossy medium; the diff-test gates
  preserve verifiability only if the port is structured to apply
  them.

Two TikZ figures, side by side:

- **Figure 1 — Doc ecosystem.** Where this manual sits relative to
  the LEAN4-First Charter (WHY), the FSA v5 Technical Guide (WHAT),
  and the LaTeX → Lean4 → Julia pipeline doc (HOW architecturally).
  Renders the same boxes/arrows shown in the *Flow-map* section of
  this plan, but in TikZ. Tells a fresh reader which sibling doc to
  consult for each kind of question.
- **Figure 2 — Seven-phase port flow.** Phase 0 inputs (tech guide
  + Lean4 code) on the left, Phase 6 plotters/launcher on the right,
  arrows for "must complete before" dependencies, with parallel
  paths for Phase 1 (Lean4 CLI) and Phase 2 (Julia CPU port) feeding
  Phase 3. Renders the same DAG sketched in the *Flow-map* section
  of this plan. Borrows the box/arrow style from
  `lean4_to_julia_pipeline.tex`'s Figure 1.

A short *How to read this manual* paragraph closes §1: read in
order, treat each phase's verification gate as a stop-and-check
point, only proceed when green.

### §2 — Prerequisites

- Hardware: NVIDIA GPU with CUDA support (project tuned for RTX 5090).
- OS: Linux (tested on Ubuntu).
- Toolchain: Julia ≥ 1.11, Lean4 with `lake`, CUDA driver, `comfyenv`
  (the project's conda env per `CLAUDE.md`).
- Repo state: clean checkout of `python-smc2-filtering-control`.
- Reading: §1–7 of the FSA v5 technical guide (skim only — math
  details are referenced as needed during the port).

### §3 — Inputs (Step 0)

The reader starts with these on-disk artefacts:

| Artefact | Path | Role |
| --- | --- | --- |
| FSA v5 technical guide | [`version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) | Math source-of-truth — every equation is here |
| Lean4 model | [`version_1_5_LEAN/Fsa/V5/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V5/) | Six files: `Types`, `Drift`, `Plant`, `Obs`, `Cost`, `Schedule`. Already transcribed from the tech guide; the Julia port mirrors these line-by-line |
| v1.5 Julia layout | [`version_1_5_Julia/models/fsa_high_res/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/) | The pattern to mirror: 7 `.jl` files + bench glue + GPU files. v5 maintains the same shape |

### §4 — Phase 1: Build the v5 Lean4 CLI

- **Goal**: produce `fsa_v5_cli` native binary that the diff test calls.
- **Inputs**: `Fsa/V5/` (six files; complete) + the existing
  `Main.lean` for v1.5 (pattern to mirror).
- **Files to create**: `Fsa/V5.lean` (umbrella, 6 lines), `Main_v5.lean`
  (~250 lines, JSON dispatcher); modify `lakefile.lean` (+5 lines for
  the second `lean_exe` target).
- **JSON wire conventions**: a small table — `State6D` ↔ array[6],
  `BimodalPhi` ↔ object{Phi_B, Phi_S}, `Params` ↔ object{28 keys},
  `ObsParams` ↔ object{22 keys}. Float printing uses Lean's default
  6-sig-digit `toString` (precision floor — see §11).
- **Dispatch tags exposed (15)**: `drift`, `diffusion`, `emStep`,
  `hrMean`, `sleepProb`, `stressMean`, `stepsLogMean`,
  `volumeLoadMean`, `muBar`, `findASep`, `aSepGrid`,
  `scheduleFromTheta`, `designMatrix`, `cPhi`, `sigmoid`. Document
  why `applyPrior` and `obsLogWeight` are deliberately out of scope.
- **Verification**:
  ```
  cd version_1_5_LEAN && lake build
  echo '{"fn":"sigmoid","x":0.0}' | .lake/build/bin/fsa_v5_cli
  # → {"x":0.500000}
  echo '{"fn":"cPhi","phi_default":1.0,"phi_max":3.0}' | .lake/build/bin/fsa_v5_cli
  # → {"x":-0.693147}    (= -log 2)
  ```

### §5 — Phase 2: Write the Julia CPU port (8 files)

Per-file table: each row is one `models/fsa_v5/<name>.jl` file with:
- the Lean source it transcribes (with line ranges)
- public functions / constants exposed
- a 1-line "what to look out for" gotcha

| File | Lean source | Notes |
| --- | --- | --- |
| `simulation_v5.jl` | `Types.lean` | constants, TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5, DEFAULT_INIT, TRAINED_ATHLETE_INIT, FROZEN_PARAMS_V5, PARAM_KEYS_V5, OBS_PARAM_KEYS_V5 |
| `_dynamics_v5.jl` | `Drift.lean` | `drift_v5`, `diffusion_v5`, `em_step_v5`. Hill exponent n_dec hardcoded to 4 (frozen per tech guide §7.2). |
| `_plant_v5.jl` | `Plant.lean` | `PlantState6D`, `plant_step_v5`, `plant_rollout_v5`. NB rollout return shape is bench-consumed (flat NamedTuple), not Lean-counterpart. |
| `obs_v5.jl` | `Obs.lean` | 5 deterministic channel-mean functions |
| `estimation_v5.jl` | `Obs.lean` + tech guide §3 | `obs_log_weight_v5` (5-channel sum, per-channel gates), `propagate_v5`, `PARAM_NAMES_V5`, `PARAM_PRIOR_CONFIG_V5` |
| `cost_v5.jl` | `Cost.lean` | `mu_bar`, `find_a_sep` (±Inf sentinels), `a_sep_grid` (Bug-2-prevention shape) |
| `schedule_v5.jl` | `Schedule.lean` | `sigmoid`, `c_phi`, `schedule_from_theta`, `design_matrix` |
| `FSAv5.jl` | aggregator | re-exports 39 (then 47 after Phase 4) public symbols |

**Lean → Julia idiom conversions** (~6 rules: `Id.run do` → `for`,
`Float.pow (max x 0.0) n` → `max(x, 0.0)^n`, `1.0 / 0.0` → `Inf`,
`Array (Array Float)` → `Matrix{Float64}`, state as `SVector{6, Float64}`,
params as `Dict{Symbol, Float64}` with NamedTuple converter for ForwardDiff).

**Boundary handling**: Phase 2 EM step uses **clamp/floor** per tech
guide §6.4, not v1.5-style reflect. Documented as a deliberate
deviation.

**Verification** (smoke-load only):
```
julia --project=. -e 'include("models/fsa_v5/FSAv5.jl"); using .FSAv5
sigmoid(0)             == 0.5
c_phi(1.0, 3.0)        == -log(2)
drift_v5(zeros(6), TRUTH_PARAMS_V5, (0,0))[5]
                        == TRUTH_PARAMS_V5[:KFB_0]/TRUTH_PARAMS_V5[:tau_K]
find_a_sep((0.30,0.30), TRUTH_PARAMS_V5) == -Inf
   (per tech guide §4.5: balanced moderate is healthy mono-stable)
'
```

### §6 — Phase 3: Differential test

- **File to create**: [`diff_test/test_lean_diff_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v5.jl) (~440 lines).
- Mirrors `test_lean_diff_v15.jl` structure: long-lived `fsa_v5_cli`
  subprocess, JSON line protocol, **pre-drawn noise** (so RNG-divergence
  is excluded from the comparison).
- **15 testsets** (full table with assert counts per testset).
- **Tolerance ladder**: 1e-6 for single-step, 1e-4 for `em_step_v5`,
  1e-5 for `schedule_from_theta` (Lean wire-format precision floor).
- **Run command + expected**:
  ```
  cd version_1_5_LEAN && lake build           # produces fsa_v15_cli AND fsa_v5_cli
  cd ../version_1_5_Julia
  julia --project=. diff_test/test_lean_diff_v15.jl   # 121/121 (no regression)
  julia --project=. diff_test/test_lean_diff_v5.jl    # 401/401 here, 424/424 after Phase 4
  ```

### §7 — Phase 4: GPU port (SOFT cost only)

- **Files to create**: `gpu_pf_v5.jl` (~430 lines), `gpu_control_v5.jl`
  (~330 lines). Update `FSAv5.jl` to include and re-export both.
- **Architectural rule** (cite the charter): copy v1.5's plumbing
  verbatim — kernel structure, RNG-per-thread layout, log-lik
  reduction, NaN guards, framework hooks. ONLY swap the math block
  with v5 versions transcribed from the diff-tested CPU functions.
  Do NOT rewrite the plumbing — that's the AI-fuckup-prone part.
- **Dimension bumps**: state 3 → 6, obs 3 → 5, params row 14 → (15
  estimated dyn + 22 estimated obs + 14 frozen), boundary
  reflect → clamp/floor.
- **SOFT cost only**: `J = λ_Φ·effort − A_acc + λ_F·F-barrier +
  λ_chance·∫σ(β·(A_thr − A)/scale) dt`. HARD chance-constraint
  (with per-bin A_sep) is deliberately out of scope; A_thr is a
  constant (default 0.05) instead.
- **Single-thread debug entry points**: `gpu_propagate_one!` and
  `gpu_cost_one!` allow ndrange=1 invocation so the diff test can
  verify per-thread math at fp32 precision (~1e-3 absolute).
- **HMC clarification** (cite PORT_LOG correction): both filter and
  controller HMC live in the framework (`SMC2FC_functional`); v5
  needs zero new HMC code, just exposes `gpu_log_density_v5` and
  `make_log_density_fn_v5` closures.
- **Extended diff test**: 401 → 424 asserts (23 new GPU parity
  testsets). Tolerance 1e-3 (fp32 vs fp64).

### §8 — Phase 5: Bench driver

- **Files to create**: `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`
  (top-level driver, ~250 lines), `tools_v5/bench/` (5 modules:
  `bench_args.jl` 270L copy + step-minutes default 60→15;
  `bench_filter.jl` 230L sed-renamed; `bench_controller.jl` 120L
  rewritten for bimodal Φ + 6D init + Dict params; `bench_loop.jl`
  210L rewritten for 5-channel obs + 6D state + plant_rollout_v5
  shape; `bench_postproc.jl` 280L sed-renamed + new data.jld2 schema;
  plus `models/fsa_v5/bench_glue_v5.jl` (~110 lines: posterior_mean_v5,
  window_grid_obs_v5).
- **Honest flag** about "model-agnostic" claim: bench/* flow is
  generic but symbols are NOT; copy-and-edit not verbatim copy.
- **Plant rollout shape change** in `_plant_v5.jl` (ripple edit):
  return shape changed to flat NamedTuple matching the bench's
  `accumulate_obs_history_v5` consumer. Not diff-tested (no Lean
  counterpart).
- **Smoke-test verification**:
  ```
  FSA_V5_DRY_RUN=1 julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl --T-days 1
  # → "[ Info: loading v5 model + framework (FSA_STEP_MINUTES=15)..."
  # → "[ Info: FSA_V5_DRY_RUN=1 — driver loaded but main() skipped"
  ```

### §9 — Phase 6: Plotters + production gating + launcher

Three plotters + one shell launcher + the production sleep/wake
gating logic.

| Item | File | Purpose |
| --- | --- | --- |
| State traces (8-panel) | `tools_v5/plot_state_traces_v5.jl` | 6 state components + 2 Φ-per-stride traces |
| Param traces (5×8 grid) | `tools_v5/plot_param_traces_v5.jl` | 37 estimated params, 5/95 quantile + median + truth line |
| Obs channels (6-panel) | `tools_v5/plot_obs_channels_v5.jl` | HR/Stress/Steps/VL/Sleep + circadian, with deterministic μ overlay and gate-aware colouring (the **"data-into-filter check"** plot) |
| Launcher (T=42 default) | `tools_v5/launchers/run_julia_v5_T42d.sh` | nvidia-smi 1 Hz sampler + bench + 3 plotters + manifest |
| Launcher (T=7 default) | `tools_v5/launchers/run_julia_v5_T7d.sh` | smoke-test variant |

**Production sleep/wake gating** (tech guide §3.2):
- HR sleep-only (22:00–06:00); Stress / Steps wake-only;
  VolumeLoad once per day at 18:00 (= bin 72 in 0-indexed bin
  with BINS_PER_DAY=96); Sleep label every bin.
- Implemented in `bench_loop.jl::accumulate_obs_history_v5` —
  per-bin gates computed from time-of-day, not all-ones.
- Verification arithmetic: 32 sleep bins, 64 wake bins, 1 VL bin,
  96 sleep-label bins (over a 96-bin day at 15-min grid).

**bench_postproc.jl additions**: save the 5 obs vectors, 5 gate
masks, and circadian C(t) to data.jld2 so the obs-channel plotter
has all required inputs.

### §10 — Verification gates (the final checklist)

A short numbered list a reader can mechanically check:

1. **Lean binaries build**: `cd version_1_5_LEAN && lake build` produces
   both `fsa_v15_cli` and `fsa_v5_cli` cleanly.
2. **v1.5 diff test** still passes: `julia --project=. diff_test/test_lean_diff_v15.jl`
   → 121/121.
3. **v5 diff test** passes: `julia --project=. diff_test/test_lean_diff_v5.jl`
   → 424/424 (8 model + 5 obs-channel + 5 cost/schedule + 2 GPU).
4. **Driver dry-run** loads cleanly:
   `FSA_V5_DRY_RUN=1 julia --project=. tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl --T-days 1`
   prints the load-banner and the dry-run skip, no errors.
5. **End-to-end T=7 smoke run** completes:
   `./tools_v5/launchers/run_julia_v5_T7d.sh 7 --N-smc 32 --K-per-chain 200 --ctrl-n-smc 256 --ctrl-num-mcmc 8 --ctrl-chees-max 256 --ctrl-n-anchors 8 --ctrl-max-levels 25`
   → ~6 min wall on RTX 5090, exit 0, output dir contains:
   `data.jld2`, `bench.log`, `manifest.json`, `experiment_run.md`,
   `per_stride.csv`, `controller_diagnostics.csv`, three PNGs.
6. **Reference run** (this manual's verified baseline):
   - Date: 2026-05-09T22:01:01
   - Wall: 362 s
   - Final state: B=0.0922, S=0.1098, F=0.3471, A=0.0409,
     K_FB=0.0488, K_FS=0.0652
   - 13 strides, 37 estimated params, 13 frozen
   - Output dir: `compare_v15_julia_vs_python/example_run/julia_v5_T7d_2026-05-09/T7d_seed42/`
   The reproducer's run should match this within stochastic
   variation (order of magnitude, qualitative trajectory shape).

### §11 — Troubleshooting & known gotchas

A reference list:

- **Lean's `Float.toString` floor at 6 sig digits.** Manifests as
  `schedule_from_theta` test failures at 1e-6 atol when output
  magnitude is ~3 (`phi_max · sigmoid`). Fix: use rtol=1e-5 for
  schedule outputs. Documented in `test_lean_diff_v5.jl`.
- **GPU plumbing reuse.** Don't refactor the kernel / framework
  glue when porting v5 — verbatim copy from v1.5 + math-block swap
  is the rule. Refactoring there is the AI-fuckup-prone part.
- **fp32 inner loops, fp64 outer state.** Project convention; do
  not annotate `Float64` inside SDE inner loops on GPU. See
  `CLAUDE.md` GPU-dtype section.
- **`FSA_STEP_MINUTES` env var must be set before model import.**
  The bench driver does this via `ENV["FSA_STEP_MINUTES"] = ...`
  before `include("FSAv5.jl")`. Reorder this and `BINS_PER_DAY`
  resolves to the wrong value.
- **`FSA_V5_DRY_RUN=1`** for dry-run smoke testing the driver
  (loads everything, skips `main()`).
- **`OT_MAX_WEIGHT=0.0`** default in the filter target. Setting
  this > 0 enables the framework's OT rescue with a documented
  ~12× wall-time penalty (per-chain Julia loop with PCIe
  round-trips).
- **HARD chance-constraint** is deliberately out of scope. Only
  SOFT (Lagrangian-relaxed with constant A_thr) is implemented.
- **Per-bin A_sep**-based chance constraint is deferred (would
  require a CPU-side precompute or separate GPU prep kernel
  given the 286M `mu_bar` evaluations per call at production
  sizes).
- **`@match` site numbering.** v1.5 has 3 `@match` sites
  (paramsV15ToV1, reflect_unit, apply_prior). v5 has none —
  no basis adapter, no reflection (uses clamp), no separate
  prior transform exposed at the JSON layer.

### Appendix A — Complete file tree (after the port)

A mechanical listing of every new / modified file with line counts.

### Appendix B — Command index

Every shell / Julia / lake command in the manual, in one place.

### Appendix C — Diff-test coverage table

The 15 testsets × tolerance × assert count matrix from PORT_LOG
Phase 3, plus the 23 GPU asserts from Phase 4. Total = 424.

### References

- FSA v5 technical guide (the model-maths input)
- LEAN4-First Charter (the policy doc that motivates this approach)
- LaTeX → Lean4 → Julia pipeline doc (the architecture-level companion)
- PORT_LOG.md (the chronological record this manual condenses)

## Files to read while writing

- [`models/fsa_v5/PORT_LOG.md`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/PORT_LOG.md) — every Phase entry; this is the source the manual condenses
- [`lean4_to_julia_pipeline.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/lean4_to_julia_pipeline.tex) — for the TikZ flow-diagram style + cross-references the manual will make
- [`lean4_first_charter.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/lean4_first_charter.tex) — to cite the LEAN4-first rationale
- [`FSA_version_5_technical_guide.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) — to cite specific sections (§2.2 drift, §3.2 gating, §6.4 boundary, §7 pinning)
- [`preamble.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/preamble.tex) — already loads tikz / listings / hyperref / amsmath / booktabs

## Existing utilities to reuse

- The TikZ flow diagram from `lean4_to_julia_pipeline.tex` Figure 1
  (LaTeX → Lean4 → fsa_v5_cli → diff test → SMC2FC_functional → GPU
  layer) — adapt for v5 with the seven phases as stages.
- The same `latexmk -pdf` build pattern proven on the other LaTeX
  docs in the directory.
- The reference T=7 run output dir as visual evidence (file
  listings, the experiment_run.md summary excerpt, the three PNGs
  embedded as `\includegraphics`).

## Verification

After writing, build:
```
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs
latexmk -C fsa_v5_port_reproduction_manual.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error fsa_v5_port_reproduction_manual.tex
```

Pass criteria:
- 0 LaTeX errors.
- 0 undefined references / citations (`grep -E "Warning|Error|\?\?" *.log` returns 0 lines).
- TikZ flow diagram renders.
- All `\path{...}` and `\href{...}` references resolve.
- Page count ~25–30.
- Manual self-test: pick any verification gate from §10; the
  command prints what the gate prescribes.

## Out of scope (deliberate)

- Re-running the bench at production T=42 (this manual is the
  reproduction guide; running the bench is the reader's task once
  they've followed it).
- Documenting v1.5 (covered by `lean4_to_julia_pipeline.tex`).
- Designing alternative obs-gating schemas, alternative cost
  formulations (HARD vs SOFT chance constraint), or any
  forward-looking design discussion. The manual is descriptive of
  the AS-IS port, not prescriptive of future variants.

## Archive (per CLAUDE.md rule)

After exiting plan mode and starting execution, copy this plan from
`~/.claude/plans/i-want-you-to-cozy-hoare.md` into:
`claude_plans/FSA_v5_port_reproduction_manual_2026-05-09_<HHMM>.md`
with the `> Archived from plan mode:` line near the top. Update the
archive in lock-step as the plan evolves.
