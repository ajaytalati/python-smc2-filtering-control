# Handover (post-compact): v5 GPU-starvation diagnosis and fix

> Paste this as a single message after compacting to bring me back up
> to speed. Self-contained — does not assume the prior conversation
> history.

## TL;DR — what you're picking up

The FSA v5 port from Lean4 to Julia is **complete and verified through
6 phases + a docs phase**. The differential test is green at 424/424
and the bench runs end-to-end. The currently-open task is a **GPU
throughput investigation**: on `nvtop` the bench shows two phases per
stride — **~90% saturated** (the kernels) and **~20% starved** — and
the user wants the starved phase eliminated. The plan for fixing it
is at `/home/ajay/.claude/plans/i-want-you-to-cozy-hoare.md` (and was
saved by the user at `claude_plans/old_plans/GPU-starved-phase
diagnosis and fix for the v5 bench.md`). **Plan was approved; work
hasn't started yet.**

## Current state of the v5 port (snapshot as of 2026-05-09 23:30)

### Phase 1 — Lean4 CLI for v5: DONE

- Files: `version_1_5_LEAN/Fsa/V5.lean` (umbrella),
  `version_1_5_LEAN/Main_v5.lean` (~250-line JSON dispatcher),
  `version_1_5_LEAN/lakefile.lean` (`lean_exe fsa_v5_cli` target).
- 15 dispatch tags exposed:
  `drift`, `diffusion`, `emStep`, `hrMean`, `sleepProb`,
  `stressMean`, `stepsLogMean`, `volumeLoadMean`, `muBar`,
  `findASep`, `aSepGrid`, `scheduleFromTheta`, `designMatrix`,
  `cPhi`, `sigmoid`.
- Verification: `lake build` produces both `fsa_v15_cli` and
  `fsa_v5_cli`; `echo '{"fn":"sigmoid","x":0.0}' | fsa_v5_cli`
  returns `{"x":0.500000}`.

### Phase 2 — Julia CPU port: DONE

8 files under `version_1_5_Julia/models/fsa_v5/`:

- `simulation_v5.jl` — `BINS_PER_DAY=96`, `DT_BIN_DAYS=1/96`,
  `TRUTH_PARAMS_V5`, `DEFAULT_OBS_PARAMS_V5`, `DEFAULT_INIT`,
  `TRAINED_ATHLETE_INIT`, `FROZEN_PARAMS_V5`, `PARAM_KEYS_V5`,
  `OBS_PARAM_KEYS_V5`.
- `_dynamics_v5.jl` — `drift_v5`, `diffusion_v5`, `em_step_v5`.
  Hill exponent `n_dec` hardcoded to 4.
- `_plant_v5.jl` — `PlantState6D`, `plant_step_v5`,
  `plant_rollout_v5`. **Important:** rollout return shape is a
  flat NamedTuple (`final_state`, `trajectory`, `obs_HR`, `obs_S`,
  `obs_steps`, `obs_VL`, `obs_sleep`, `Phi_B`, `Phi_S`, `C`) —
  designed for the bench to consume directly.
- `obs_v5.jl` — `hr_mean`, `sleep_prob`, `stress_mean`,
  `steps_log_mean`, `volume_load_mean`.
- `estimation_v5.jl` — `obs_log_weight_v5` (5-channel sum,
  per-channel gates), `propagate_v5`, `PARAM_NAMES_V5` (37
  estimated), `PARAM_PRIOR_CONFIG_V5`.
- `cost_v5.jl` — `mu_bar`, `find_a_sep` (`±Inf` sentinels),
  `a_sep_grid` (per-particle × per-bin matrix).
- `schedule_v5.jl` — `sigmoid`, `c_phi`, `schedule_from_theta`,
  `design_matrix`.
- `FSAv5.jl` — aggregator; re-exports 47 public symbols total
  (after Phase 4 added the GPU files).

Boundary handling: **clamp/floor** (B,S clamped to [ε, 1-ε];
F,A,K floored at 0) per tech guide §6.4. NOT v1.5-style reflect.

### Phase 3 — Differential test: DONE (401/401 here)

`version_1_5_Julia/diff_test/test_lean_diff_v5.jl`. Mirrors
v1.5's structure: long-lived `fsa_v5_cli` subprocess, JSON line
protocol, **pre-drawn noise** for `em_step_v5`. Tolerance ladder:
- `1e-6` single-step (drift, diffusion, channel means, etc.)
- `1e-4` integrated EM step
- `1e-5` schedule_from_theta (Lean's 6-sig-digit `Float.toString`
  is the wire-format precision floor)

### Phase 4 — GPU port (SOFT cost only): DONE (extends diff test 401→424)

- `version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl` (~430 lines).
  Plumbing copied verbatim from v1.5's `gpu_pf.jl`; only math block
  swapped. fp32 inner loops, fp64 outer state.
- `version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl` (~330 lines).
  SOFT chance-constrained cost (HARD deliberately out of scope; uses
  constant `A_thr=0.05` instead of per-bin `A_sep`).
- Single-thread debug entry points (`gpu_propagate_one!`,
  `gpu_cost_one!`) wired into the diff test for fp32 vs fp64
  parity check at `1e-3` tolerance. **23 new GPU asserts → 424
  total.**
- HMC: BOTH filter and controller HMC live in the framework
  (`SMC2FC_functional`'s `parallel_hmc_one_move_generic!` for
  filter, `run_tempered_smc_gpu` for controller). v5 model files
  expose closures (`gpu_log_density_v5`, `make_log_density_fn_v5`)
  but no HMC code.

### Phase 5 — Bench driver: DONE

- `version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl`
  (top-level driver, ~250 lines).
- `version_1_5_Julia/tools_v5/bench/` (5 modules):
  `bench_args.jl`, `bench_filter.jl`, `bench_controller.jl`,
  `bench_loop.jl`, `bench_postproc.jl`.
- `version_1_5_Julia/models/fsa_v5/bench_glue_v5.jl`
  (`posterior_mean_v5`, `window_grid_obs_v5`).
- Driver guard: `FSA_V5_DRY_RUN=1` skips `main()` for smoke-loading.

### Phase 6 — Plotters + production gating + launchers: DONE

- `tools_v5/plot_state_traces_v5.jl` — 8-panel state plot.
- `tools_v5/plot_param_traces_v5.jl` — 5×8 grid for 37 params.
- `tools_v5/plot_obs_channels_v5.jl` — 6-panel obs plot
  (HR/Stress/Steps/VL/Sleep + circadian) with `μ` overlay and
  gate-aware colouring. The "data-into-filter check" diagnostic.
- `tools_v5/launchers/run_julia_v5_T42d.sh` (T=42 default).
- `tools_v5/launchers/run_julia_v5_T7d.sh` (T=7 smoke variant).
- Production sleep/wake gating in
  `tools_v5/bench/bench_loop.jl::accumulate_obs_history_v5`:
  HR sleep-only (22:00–06:00); Stress/Steps wake-only;
  VolumeLoad once per day at 18:00 (= bin 72); Sleep label
  every bin. Verified arithmetic: 32 sleep bins, 64 wake bins,
  1 VL bin, 96 sleep-label bins per day.

### Docs phase — DONE

Four LaTeX docs in `version_1_5_LEAN/LaTex_docs/`:

- `lean4_first_charter.tex` — WHY (policy)
- `FSA_version_5_technical_guide.tex` — WHAT (model maths)
- `lean4_to_julia_pipeline.tex` — HOW architecturally
- `fsa_v5_port_reproduction_manual.tex` — HOW operationally
  (17 pages with two TikZ flow diagrams; built clean)

### PORT_LOG

`version_1_5_Julia/models/fsa_v5/PORT_LOG.md` — chronological
record of every phase. Reads top-to-bottom for the full history.

## Verified reference run

T=7 closed-loop bench at small config completed cleanly on
2026-05-09 22:01 in 362 s wall on RTX 5090. Output dir:
`compare_v15_julia_vs_python/example_run/julia_v5_T7d_2026-05-09/T7d_seed42/`
contains `data.jld2`, `bench.log`, `nvidia_smi.csv`,
`per_stride.csv`, `controller_diagnostics.csv`, `manifest.json`,
`experiment_run.md`, three PNGs. Final state recovered:
`B=0.0922, S=0.1098, F=0.3471, A=0.0409, K_FB=0.0488, K_FS=0.0652`.
13 strides, 37 estimated params, 13 frozen.

## Currently-running background task

**Task ID `b3ya3k85a`** — T=42 saturated-controller / small-filter
bench (`./tools_v5/launchers/run_julia_v5_T42d.sh 42
--filter-N-smc 32 --filter-K-per-chain 200 --ctrl-n-smc 256
--ctrl-num-mcmc 8 --ctrl-chees-max 256 --ctrl-n-anchors 8
--ctrl-max-levels 25`). Output to
`compare_v15_julia_vs_python/example_run/julia_v5_T42d_2026-05-09/T42d_seed42/`.
Projected wall ~30–40 min based on T=7 timing × 6× strides.
Check status with `tail bench.log` in that dir; the system
will notify when the task completes.

## What's next: GPU-starvation plan

The plan to execute next is at:
- `/home/ajay/.claude/plans/i-want-you-to-cozy-hoare.md` (live)
- `claude_plans/old_plans/GPU-starved-phase diagnosis and fix for the v5 bench.md`
  (user's snapshot)

**Approved scope**: instrument `bench_loop.jl::run_one_stride_v5`
with per-phase wall timers, run a T=2 diagnostic at the small
config, correlate timestamps with `nvidia_smi.csv` to identify
which phase is the ~20% GPU-starved one, then fix that phase.

**Hypothesis ranking** (don't act on this; diagnose first):
1. `plant_rollout_v5` — 48-bin Julia loop, all CPU, fp64
2. Filter inter-λ-level CPU bookkeeping (smooth_resample / Liu-West)
3. Controller-target reconstruction per replan (CPU-side `randn` of
   the CRN noise grid)

**Phases of the plan**:
- A: instrument with `time()` timers around 8 stride sub-phases;
     log to `bench.log` and per_stride.csv.
- B: short diagnostic run at T=2 (small config, 3 strides, ~1 min).
- C: correlate per-phase timings with `nvidia_smi.csv`. Maybe write
     `tools_v5/diagnose_gpu_starvation.jl` (~30 lines) for the
     correlation plot.
- D: apply the fix conditional on the diagnosis (plant rollout
     speedup OR move smooth_resample to GPU OR pre-allocate
     controller target across replans).

Verification: re-run T=2; per-stride wall drops; nvidia_smi mean
util rises; diff test still 424/424.

## Conventions and gotchas to respect

- **`FSA_STEP_MINUTES` env var must be set BEFORE `include
  ("FSAv5.jl")`** — `simulation_v5.jl` reads it at module load.
  The bench driver does this correctly; if you see it elsewhere,
  preserve the order.
- **`FSA_V5_DRY_RUN=1`** smoke-loads the driver without running
  `main()`. Useful for parse / link checks.
- **`OT_MAX_WEIGHT=0.0`** is the right default. Setting > 0
  enables the framework's OT rescue with a documented ~12×
  slowdown.
- **Filter flag aliasing fix landed earlier today**: in
  `tools_v5/bench/bench_args.jl`, the alias lookup is now
  case-insensitive so both `--filter-N-smc` and `--filter-n-smc`
  work. The legacy unprefixed names (`--N-smc`, `--K-per-chain`)
  also still work.
- **fp32 inner loops, fp64 outer state.** Project convention; do
  not annotate `Float64` inside SDE GPU kernels.
- **GPU plumbing reuse rule.** When porting a v5 GPU change, copy
  v1.5's plumbing verbatim and only swap the math block. Don't
  refactor the kernel structure.
- **Plant rollout shape is bench-consumed.** `_plant_v5.jl::
  plant_rollout_v5` returns a flat NamedTuple. NOT diff-tested
  against Lean (no Lean counterpart for the rollout wrapper).
- **HARD chance-constraint deliberately out of scope.** Only SOFT
  is implemented in `gpu_control_v5.jl`.

## Behavioural memories that apply

- Junior-engineer stance under user (senior). Ask, don't declare.
  Hedge confidence ("I think", "I'd want to verify", "from what
  I can see"). Avoid "definitely" / "obviously" / "clearly wrong".
- Verify before asserting. Don't speculate about which phase is
  starved; diagnose with timing instrumentation FIRST.
- Plain language, no GitHub / dev-ops jargon unless the user
  used it. Avoid "spike", "vendor the deps", "PyPI", etc. If a
  technical term is unavoidable, say what it means in plain words.
- Don't refactor uninvited. The plan's Phase D is conditional
  on the diagnosis; pick ONE branch, do it, then stop.
- Bench writeups: separate measured from speculated. Never claim
  a perf number without a measurement; never project a wall time
  without showing the linear extrapolation.
- Bistable systems need basin analysis, not single-trajectory
  probes. Doesn't apply to this task but a recurring user rule.

## Out of scope for the GPU-starvation task

- Re-tuning the saturated config in response to better
  utilisation.
- Rewriting `SMC2FC_functional`'s `run_tempered_smc_gpu` even
  if Phase C points at framework-level CPU bookkeeping. Document
  it as a framework-backlog item instead.
- Any change to the model maths, the obs schema, or the diff test.
- Touching the currently-running `b3ya3k85a` T=42 bench.
  Instrumentation work goes into the same files; the next bench
  run after instrumentation will be the diagnostic T=2.

## Files to read on resume (in order)

1. The plan: `/home/ajay/.claude/plans/i-want-you-to-cozy-hoare.md`
2. The instrumentation site:
   `version_1_5_Julia/tools_v5/bench/bench_loop.jl::run_one_stride_v5`
3. The CSV writer to extend:
   `version_1_5_Julia/tools_v5/bench/bench_postproc.jl::save_bench_outputs`
4. The launcher (already captures `nvidia_smi.csv`):
   `version_1_5_Julia/tools_v5/launchers/run_julia_v5_T7d.sh`
5. The PORT_LOG (general context):
   `version_1_5_Julia/models/fsa_v5/PORT_LOG.md`

## What to do first when I resume

1. Read the plan file end-to-end.
2. Read this handover doc to confirm context match.
3. Check the status of background task `b3ya3k85a` (the T=42
   bench) — if still running, leave it alone; the diagnostic
   work is independent. If it completed, glance at its
   `bench.log` and `nvidia_smi.csv` for any signal about the
   starvation phase.
4. Start Phase A of the plan: instrument
   `bench_loop.jl::run_one_stride_v5` with the 8 per-phase
   timers listed in the plan.
5. Then Phase B: run the T=2 diagnostic.
6. Then Phase C: correlate.
7. Then Phase D (conditional fix).

That's the work.
