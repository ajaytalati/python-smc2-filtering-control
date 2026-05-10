# FSA v5 port: Lean4 → Julia, with differential test

> Archived from plan mode: 2026-05-09 17:23.

## Context

The FSA v5 model is already coded in Lean4 at [`version_1_5_LEAN/Fsa/V5/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V5/) (six files: `Types`, `Drift`, `Plant`, `Obs`, `Cost`, `Schedule`). The technical guide at [`version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) is the maths source-of-truth, and the Lean files transcribe it line-by-line with no gaps. The painful step is the Julia port: it has to mirror the v1.5 Julia layout at [`version_1_5_Julia/models/fsa_high_res/`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/) and be backed by a differential test that catches transcription errors at machine precision.

The user requires the port to be **methodological and reproducible**: every file's port is logged with what was lifted, what was new, and what was tested. Diff-testing every public function against the Lean reference is the safety net.

## Strategy in one paragraph

Port file-by-file in dependency order, CPU first, GPU last and gated. Every Julia function gets a matching Lean dispatch tag in a new `Main_v5.lean`, a matching testset in a new `test_lean_diff_v5.jl`, and a one-line entry in a methodology log. The CPU port is mechanical Lean → Julia transcription (line-for-line; trivial Float arithmetic) and is the deterministic, low-risk part. The GPU port is gated on the CPU port being green; its math block is a copy of the diff-tested CPU functions, its plumbing is copy-with-dim-bumps from the v1.5 GPU kernels.

## Output artefacts

| # | Artefact                                                              | Location                                                                  |
| - | --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1 | New Julia model directory (parallel to v1.5)                          | `version_1_5_Julia/models/fsa_v5/`                                |
| 2 | New Lean CLI entry + binary                                           | `version_1_5_LEAN/Main_v5.lean` → `.lake/build/bin/fsa_v5_cli`             |
| 3 | New differential test                                                 | `version_1_5_Julia/diff_test/test_lean_diff_v5.jl`                         |
| 4 | Methodology log (one entry per phase, with file:line citations)        | `version_1_5_Julia/models/fsa_v5/PORT_LOG.md`                     |

GPU files (`gpu_pf_v5.jl`, `gpu_control_v5.jl`) live in the same Julia directory but are deferred behind a Phase-4 gate (see below).

## Five execution phases

### Phase 1 — Lean CLI for v5 (the JSON bridge)

Today, `version_1_5_LEAN/Main.lean` only dispatches v1.5 functions. The v5 diff test needs its own binary.

- **Add `Main_v5.lean`** alongside the existing `Main.lean`, mirroring its structure ([`Main.lean:166-218`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Main.lean)).
- Dispatch tags (one per public function in `Fsa/V5/`):
    `drift`, `diffusion`, `emStep`, `plantStep`, `hrMean`, `sleepProb`, `stressMean`, `stepsLogMean`, `volumeLoadMean`, `obsLogWeight`, `muBar`, `findASep`, `aSepGrid`, `scheduleFromTheta`, `designMatrix`, `cPhi`, `sigmoid`, `applyPrior` (if v5 reuses the v1.5 prior transform).
- JSON schema: write a `State6D ↔ JSON array[6]`, `BimodalPhi ↔ JSON object{Phi_B, Phi_S}`, `Params ↔ JSON object{41 fields}`, `ObsParams ↔ JSON object{16 fields}` convention; document in the dispatch table.
- **Update lakefile**: register `Main_v5.lean` as a second `lean_exe` target named `fsa_v5_cli`. Verify both binaries build with `lake build`.
- **PORT_LOG entry**: tag list, JSON schema, build command verified.

### Phase 2 — Julia CPU port (file-by-file in dependency order)

Each file ports the corresponding Lean module. **Mirror v1.5 layout naming exactly**; only the math content changes.

| File (new)                                  | Lean source                                  | What it provides                                                                                              | Mirrors v1.5                          |
| ------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `simulation_v5.jl`                          | `Fsa/V5/Types.lean` (TRUTH_PARAMS, A_TYP, F_TYP, PHI_TYP) + technical guide §8 (15-min bins) | `BINS_PER_DAY = 96`, `DT_BIN_DAYS = 1/96`, `DEFAULT_PARAMS_V5`, `OBS_DEFAULT_PARAMS_V5`, `INIT_STATE_V5`, scenario presets if motivated by tech guide §9 | `simulation.jl`                       |
| `_dynamics_v5.jl`                           | `Fsa/V5/Drift.lean`                          | `drift_v5(y::SVector{6}, p, phi)`, `diffusion_v5(y, p)`, `em_step_v5(y, p, phi, σ_diag, dt, noise)`           | `_dynamics.jl`                        |
| `_plant_v5.jl`                              | `Fsa/V5/Plant.lean`                          | `PlantState6D`, `plant_step_v5`, `plant_rollout_v5`, `init_plant_state_v5`                                    | `_plant.jl`                           |
| `obs_v5.jl` (NEW pattern)                   | `Fsa/V5/Obs.lean`                            | `hr_mean`, `sleep_prob`, `stress_mean`, `steps_log_mean`, `volume_load_mean`                                  | (no v1.5 analogue — v1.5's 3 channels are inlined in `simulation.jl`'s `sample_obs_bfa`) |
| `estimation_v5.jl`                          | `Fsa/V5/Obs.lean` + tech guide §5 (likelihood) | `obs_log_weight_v5` (5-channel sum, sleep-gated where appropriate), `propagate_v5`, `PARAM_NAMES_V5`, `PARAM_PRIOR_CONFIG_V5` | `estimation.jl`                       |
| `cost_v5.jl` (NEW; no v1.5 CPU analogue)    | `Fsa/V5/Cost.lean`                           | `mu_bar(A, phi, p)`, `find_a_sep(phi, p) → Float64` (returns `±Inf` sentinels), `a_sep_grid(particles, schedule) → Matrix{Float64}` (n_particles × n_steps) | (none; v1.5 cost is GPU-only)         |
| `schedule_v5.jl` (NEW; v1.5 uses smc2fc/control) | `Fsa/V5/Schedule.lean`                       | `sigmoid`, `c_phi`, `schedule_from_theta`, `design_matrix`                                                    | (analogous to `smc2fc/control/rbf_schedules.py` patterns) |
| `FSAHighResV5.jl`                           | (aggregator)                                 | top-level module; re-exports the seven submodules                                                              | `FSAHighRes.jl`                       |

**Implementation rules per file**:
1. **Lean comments lift verbatim** as Julia docstrings (preserves the technical-guide-line and Python-line citations).
2. **Lean variable names map 1:1** to Julia (`B → B`, `KFB → K_FB` if Julia naming convention demands underscore; otherwise verbatim).
3. **Lean's `Id.run do` mutable loops** in `Cost.lean` (`firstSignChangeIdx`, `bisect`) → ordinary Julia `for` loops with mutable accumulators.
4. **Lean's `Float.pow (max x 0.0) n`** → Julia `max(x, 0.0)^n`.
5. **Lean's `1.0 / 0.0` for `+∞`** → Julia `Inf`; same for `-Inf`.
6. **Lean's `Array (Array Float)`** → Julia `Matrix{Float64}` (or `Vector{Vector{Float64}}` if irregular).
7. **State and params**: use `StaticArrays.SVector{6,Float64}` for state, named `NamedTuple` for `Params` and `ObsParams` (matches v1.5's `params_v15_to_v1_nt` style, keeps zero-allocation calls hot).
8. **Module structure**: each file declares `module DynamicsV5`, `module SimulationV5`, etc., re-exported by `FSAHighResV5.jl`. Mirrors the v1.5 pattern.

**Per-file PORT_LOG entries** (each entry, ~10 lines):
- File created
- Lean source(s) it transcribes (with line ranges)
- Functions ported (count + names)
- Notable conversions (Lean idiom → Julia idiom)
- Diff-test status (which testsets cover this file)

### Phase 3 — Differential test

New file [`version_1_5_Julia/diff_test/test_lean_diff_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v5.jl), structurally a copy of [`test_lean_diff_v15.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v15.jl) with v5-specific testsets. Same conventions as v1.5: long-lived `fsa_v5_cli` subprocess, JSON line protocol, pre-drawn noise, tolerance ladder (`1e-6` single-step, `1e-4` integrated).

**Testsets (~15)**:

| Testset                | Tests                                       | Inputs per session    | Tolerance |
| ---------------------- | ------------------------------------------- | --------------------- | --------- |
| `sigmoid`              | `Schedule.sigmoid`                          | 5 random + 4 corner   | $10^{-6}$ |
| `c_phi`                | `Schedule.c_phi`                            | 4 corner              | $10^{-6}$ |
| `apply_prior`          | (inherit from v1.5 if reused; else add)     | 7 LogNormal + 5 normal | $10^{-6}$ |
| `drift_v5`             | `Drift.drift`                               | 5 random              | $10^{-6}$ |
| `diffusion_v5`         | `Drift.diffusion`                           | 5 random              | $10^{-6}$ |
| `em_step_v5`           | `Plant.emStep`                              | 5 random with noise   | $10^{-4}$ |
| `plant_step_v5`        | (one-bin wrapper around `emStep`)           | 5 random              | $10^{-4}$ |
| `hr_mean` etc. (×5)    | `Obs.{hrMean, sleepProb, stressMean, stepsLogMean, volumeLoadMean}` | 5 random each | $10^{-6}$ |
| `obs_log_weight_v5`    | combined 5-channel log-likelihood           | 5 random + 1 sleep-gated corner | $10^{-6}$ |
| `mu_bar`               | `Cost.muBar`                                | 5 random              | $10^{-6}$ |
| `find_a_sep`           | three-way return: -Inf / +Inf / finite      | 1 each + 3 random in bistable regime | $10^{-6}$ (and `isinf` / sign checks) |
| `a_sep_grid`           | shape check (n_particles, n_steps) + values | 1 random with 3 particles × 4 schedule bins | $10^{-6}$ |
| `schedule_from_theta`  | `Schedule.scheduleFromTheta`                | 3 random θ            | $10^{-6}$ |
| `design_matrix`        | `Schedule.designMatrix`                     | 1 fixed (n_steps=24, n_anchors=4, dt=1/96) | $10^{-6}$ |

**Pre-drawn noise convention**: copy verbatim from `test_lean_diff_v15.jl:107-109`. Both sides receive the same 6-vector for `em_step` testing.

**JSON schemas** (documented inline in the diff test):
- `State6D` ↔ `[B, S, F, A, KFB, KFS]` (length-6 array)
- `BimodalPhi` ↔ `{"Phi_B": ..., "Phi_S": ...}`
- `Params` ↔ flat object with all 26 dynamics+diffusion+Hill fields
- `ObsParams` ↔ flat object with all 16 obs-channel fields

**PORT_LOG entry**: testset count, total assertions per session, run command, expected output.

### Phase 4 — GPU port (gated; ONLY if Phase 1-3 are green AND user explicitly approves)

The GPU is the high-risk part. The plan structure to keep the risk bounded:

- **`gpu_pf_v5.jl`** — copy [`gpu_pf.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_pf.jl) verbatim, then bump:
    - Particle state dim 3 → 6 (allocations, indexing, packing)
    - Obs dim 3 → 5 (channel-mean accumulation block)
    - Per-particle math block ([`gpu_pf.jl:99-134`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_pf.jl#L99-L134)) replaced with the body of `_dynamics_v5.jl`'s `drift_v5` / `diffusion_v5` and `estimation_v5.jl`'s `obs_log_weight_v5` (which are diff-tested at $10^{-6}$).
    - **Plumbing left untouched**: thread/block geometry, RNG-per-thread, NaN guards, log-weight reductions, resampling, ESS rescue, HMC.
- **`gpu_control_v5.jl`** — copy [`gpu_control.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_control.jl) verbatim, then:
    - Bump state dim, schedule type → bimodal `(Phi_B, Phi_S)`.
    - Replace cost-accumulation block with the v5 chance-constrained form using `find_a_sep` (Cost.lean §6.3 / §7.3). Critical: this is genuinely new logic, not a copy from v1.5.
- **GPU diff-test extension** (closes the §8 gap from the pipeline doc): add a single-thread debug entry point to each kernel that emits the same JSON the v5 diff test understands. Then add testsets to `test_lean_diff_v5.jl` that round-trip through the GPU kernel debug path. Catches per-thread math bugs at $10^{-6}$.
- **Smoke test**: run `bench_smc_full_mpc_fsa_v5_gpu.jl` at T=1 day before T=14d.

### Phase 5 — Methodology log

[`PORT_LOG.md`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/PORT_LOG.md), one entry per phase, every entry with:
- Date / commit hash
- Files added / modified (with line counts)
- Lean sources transcribed (file + line range)
- Diff-test status at end of phase (pass count, fail count, regression sentinels)
- Notable Lean → Julia idiom conversions
- Known-not-tested items (deliberate gaps; e.g. GPU plumbing reused unchanged from v1.5)

This is what makes the port reproducible: a future agent or human reading PORT_LOG.md can see every step and re-execute it.

## Files to read while executing

- [`version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex) — maths source of truth (haven't read end-to-end yet; do this before Phase 2 starts)
- [`version_1_5_LEAN/Fsa/V5/*.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V5/) — six files, all read; transcription targets
- [`version_1_5_LEAN/Main.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Main.lean) — JSON dispatch pattern for `Main_v5.lean`
- [`version_1_5_LEAN/lakefile.lean`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/lakefile.lean) — register `fsa_v5_cli` target
- [`version_1_5_Julia/models/fsa_high_res/*.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/) — the v1.5 layout to mirror
- [`version_1_5_Julia/diff_test/test_lean_diff_v15.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/diff_test/test_lean_diff_v15.jl) — diff-test pattern
- [`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) — bench driver to mirror for v5 (Phase 4 only)

## Existing utilities to reuse

- **JSON dispatch protocol** — `Main.lean:166-218` is a one-handler-per-tag style; copy that style for `Main_v5.lean`.
- **`open_lean_client` / `round_trip` helpers** in `test_lean_diff_v15.jl:42-63` — copy verbatim into the v5 diff test (only the binary path changes, `fsa_v15_cli` → `fsa_v5_cli`).
- **Tolerance constants** (`SINGLE_STEP_TOL = 1e-6`, `INTEGRATED_TOL = 1e-4`) — keep identical.
- **`StaticArrays.SVector`** — already imported in v1.5's `_dynamics.jl`; same pattern for v5.
- **`Test.@testset`** — same `@testset "name" begin ... end` blocks.

## Verification (end-to-end)

```
# Build both Lean binaries
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN
lake build                                         # produces fsa_v15_cli AND fsa_v5_cli

# Run both diff tests
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
julia --project=. diff_test/test_lean_diff_v15.jl  # must still pass (no regression)
julia --project=. diff_test/test_lean_diff_v5.jl   # must pass at end of Phase 3

# (Phase 4 only, gated) smoke test the GPU bench
julia --project=. tools/bench_smc_full_mpc_fsa_v5_gpu.jl --T-days 1 --seed 42 --output-dir /tmp/v5_smoke
```

**Pass criteria**:
- `lake build` produces both `fsa_v15_cli` and `fsa_v5_cli` cleanly.
- `test_lean_diff_v15.jl` still green (zero regression on the existing v1.5 surface).
- `test_lean_diff_v5.jl` green: ~15 testsets, ~80 assertions, all within tolerance.
- Phase 4 only: T=1 smoke completes without NaN, with a posterior cloud of finite log-weights.

## Out of scope (deliberate)

- **Closing the §8 GPU gap on the v1.5 side** (the existing `gpu_pf.jl` / `gpu_control.jl`). The Phase 4 work introduces single-thread debug paths for v5; bringing the same coverage to v1.5 is a separate task.
- **Production T=14 / T=84 sweep on v5**. Smoke test only at T=1; full sweeps are a separate workstream.
- **Bench driver beyond the smoke test** (`bench_smc_full_mpc_fsa_v5_gpu.jl`). Phase 4 will sketch it but not tune any of the saturated config flags.
- **Refactoring v1.5** to a shared common-base with v5. Coexistence first; refactor later only if motivated.

## Risks and mitigations

| Risk                                                            | Mitigation                                                                                                       |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Hand-port introduces sign or factor errors in the v5 math       | Diff test at $10^{-6}$ catches any divergence; corner cases for `find_a_sep` (mono-stable / bistable) explicitly tested |
| GPU kernel rewrite introduces a bug not reachable on CPU        | Phase 4 single-thread debug path round-trips through the diff-test JSON protocol — the same testset that pins the CPU side pins the GPU per-thread math |
| `Params` JSON schema (41 fields) is verbose and easy to typo    | Generate the schema from the Lean `Types.lean` source via a one-off helper, document the field order in the diff test |
| Lean's `Id.run do` constructs (`bisect`, `firstSignChangeIdx`) translate subtly wrongly | Each is a separate testset; bisection is tested with hand-computed reference roots |
| `find_a_sep` ±∞ sentinels round-trip through JSON awkwardly      | Both sides accept `Infinity` / `-Infinity` JSON literals (already enabled in v1.5 with `allow_inf=true`)         |

## Archive (per CLAUDE.md rule)

Immediately after exiting plan mode, copy this plan from `~/.claude/plans/i-want-you-to-cozy-hoare.md` into:
`claude_plans/FSA_v5_port_Lean4_to_Julia_with_diff_test_2026-05-09_<HHMM>.md`
with the `> Archived from plan mode:` line near the top. Update the archive in lock-step as the plan evolves during execution.
