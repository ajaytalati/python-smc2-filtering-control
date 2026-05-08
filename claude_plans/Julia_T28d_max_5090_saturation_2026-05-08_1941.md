# Julia T=28d max-saturation: filter + controller, RTX 5090
# (Julia-only; Python+JAX comparison deferred)

> Archived from plan mode: 2026-05-08 19:41.
> Updated: 2026-05-08 19:57 — added controller-saturation track (Phases B–E extended). User flagged the controller as the priority lever; Phase C now sweeps controller knobs (`ctrl-n-smc`, `ctrl-num-mcmc`, `ctrl-chees-max`, newly-exposed `--ctrl-n-anchors`) alongside the filter (N_smc, K_pf) sweep.
> Updated: 2026-05-08 20:04 — de-scoped the Python+JAX cross-stack comparison per user direction. Plan is now Julia-only: 4 phases (A baseline, B nsys profile, C dual sweep, D closed-loop at saturated config), plus Phase E reduced from cross-stack diff to a Julia-side findings write-up. Title updated and Phase A no longer uses the paired launcher.

## Context

Today's perf work (commits `da847d9`, `099a14a`, `4a5c007`) collapsed the
v1.5 Julia GPU filter cost from **81 ms / call → 0.9 ms / call**.
The closed-loop bench has only been re-measured at the *toy* config
(T=2d, N=16, K=100) where wall time dropped 13×. The full T=28d
closed-loop has not been re-run yet, and Julia is running at the
config that was originally chosen to be tractable for the *slow* code
path. With ~5× filter compute headroom and ~30 GB free VRAM on the
RTX 5090, the right next move is to scale up.

**The user has now explicitly de-scoped the cross-stack comparison
with Python+JAX.** This plan is **Julia-only**: profile, sweep, pick a
saturated config, run T=28d at that config, record the result. The
Python comparison harness is parked; it will pick up later from the
Julia-saturated artefacts.

The user has also explicitly flagged the **controller** as the priority
lever, not just the filter. The controller cost / call is currently
0.4 ms; with that headroom, the controller can explore more candidate
schedules (more outer SMC² particles), mix better in θ_ctrl space
(more ChEES-HMC moves, longer trajectories, bigger ChEES candidate
list), and use a richer schedule basis (more RBF anchors). The
controller is a "pure outer SMC²" — it has no inner PF — so the only
levers are particle-count, MCMC-mixing, and schedule-basis dimension.
All three are in scope.

## Goal in one sentence

**Find the largest (filter_config, controller_config) pair that the
RTX 5090 can run on the v1.5 Julia closed-loop bench at T=28d in under
30 minutes wall time, and record the Julia-side numbers (wall time,
GPU utilisation, VRAM, posterior MSE, schedule shape) at that
saturated config.**

## Scope

**In scope (Julia-only):**
- Profiling Julia under `nsys profile --trace=cuda` to attribute the
  current 0.9 ms / call (filter) and 0.4 ms / call (controller) to
  specific kernels.
- Microbench sweep over **filter** (N_smc, K_pf) until either the
  compute saturates (effective TFLOPS ≥ 50% of 104.8 peak) or VRAM
  fills (≥ 24 GB used).
- Microbench sweep over **controller** levers — `ctrl-n-smc`,
  `ctrl-num-mcmc`, `ctrl-chees-max`, and a newly-exposed `n_anchors`
  (currently hard-coded at 8 in the bench, line 459) — to find the
  largest config that does not blow VRAM and stays under a wall-time
  budget per replan.
- Picking the joint best filter + controller config from the sweeps.
- Running the Julia closed-loop bench at T=28d at that joint config,
  seed=42.
- Recording Julia-side telemetry: wall time, mean GPU utilisation,
  VRAM peak, posterior MSE to truth, F-violation rate, controller
  schedule shape (mean Φ across replans, daily-Φ variance,
  final-stride controller `n_temp`).

**Out of scope:**
- **Cross-stack comparison vs Python+JAX** — explicitly deferred by
  the user. The launcher
  `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh`
  and the comparison script
  `version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py` are
  not invoked in this plan. The Python+JAX side is left at its
  current tuned config; we'll re-run the harness against the
  Julia-saturated artefacts in a later plan.
- Rewriting `gpu_ot_blend_chain!` as a single batched-across-chains
  GPU kernel (framework work; sketched in writeup §6.7's "how to
  make it cheap" note).
- Re-introducing OT rescue: still off by default until the framework
  rewrite lands.
- Multi-seed robustness — single-seed (42) at T=28d for this plan.
  Multi-seed is item 5 of writeup §8 and a separate plan.
- H3 cost-function variants — also a separate plan.
- Switching the controller to fp64 to match Python's controller
  precision.

## Approach (4 phases, Julia-only)

### Phase A — Re-measure the Julia baseline at the new fast path

**Why:** The writeup's §7 numbers were captured before today's fixes
and are stale. We need a clean Julia baseline at the *current* default
config (filter: N=32, K=200; controller: ctrl-n-smc=256,
ctrl-num-mcmc=8, ctrl-chees-max=256, n_anchors=8) at T=28d to compare
against the saturation-tier result.

**Action:** Run the Julia closed-loop bench directly. We do NOT use
the paired launcher (it also runs Python+JAX, which is out of scope).
Capture the GPU-telemetry CSV separately via a backgrounded
`nvidia-smi` process.

```bash
cd version_1_5_Julia
mkdir -p /tmp/v15_T28d_baseline_julia
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv -l 1 \
           > /tmp/v15_T28d_baseline_julia/gpu_telemetry.csv &
NSMI_PID=$!
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
     --T-days 28 --replan-K 2 --seed 42 \
     --N-smc 32 --K-per-chain 200 \
     --num-mcmc 3 --max-temp-levels 30 \
     --output-dir /tmp/v15_T28d_baseline_julia \
     2>&1 | tee /tmp/v15_T28d_baseline_julia/bench.log
kill $NSMI_PID 2>/dev/null
```

Also capture the microbench at the same config so we have a per-call
reference for both filter and controller:

```bash
julia --project=. tools/profile_gpu.jl --both \
     > /tmp/v15_T28d_baseline_julia/profile_gpu.log
```

**Pass criteria for Phase A:**
- Bench completes without errors.
- Julia wall time at T=28d < 30 min (was 37.2 min at T=7d before
  today's fixes; expect ~10× reduction).
- `data.jld2`, `manifest.json`, `per_stride.csv`,
  `gpu_telemetry.csv`, `profile_gpu.log`, `bench.log` all present in
  the output dir.
- Captured baseline numbers: total wall time; mean GPU util; peak
  VRAM; posterior MSE to truth (final window); F-violation rate;
  per-replan controller `n_temp`; mean Φ across replans;
  filter `gpu_log_density` median ms; controller
  `gpu_cost_log_density_batched` median ms.

### Phase B — `nsys` profile, both filter AND controller

**Why:** Find what's left after the easy wins, on both sides.

**Action:**
```bash
cd version_1_5_Julia
nsys profile --trace=cuda --output=/tmp/v15_T28d_nsys \
     julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
     --T-days 2 --replan-K 1 --N-smc 32 --K-per-chain 200 \
     --num-mcmc 2 --max-temp-levels 8 \
     --output-dir /tmp/v15_T28d_nsys_run
```

Open the resulting `.qdrep` in NVIDIA Nsight Systems. Two analyses:

1. **Filter (`gpu_log_density`)** — top 5 kernels, host-gap %,
   memory-bandwidth utilisation.
2. **Controller (`gpu_cost_log_density_batched`)** — same metrics.
   Note: the controller calls `parallel_hmc_one_move_generic!` and
   `gpu_grads_parallel_chains_fd` from
   `julia/SMC2FC_functional/src/Control/GPUControlSMC.jl`, which in
   turn launches the cost kernel. Each tempering level fires the
   kernel `ctrl-num-mcmc + log₂(chees-max/16)` times for HMC + ChEES
   selection. The bench's per-replan cost is dominated by these.

**Pass criteria for Phase B:**
- A short text note in `claude_plans/` summarising both sides:
  - Top 5 kernels by cumulative time in one filter call.
  - Top 5 kernels by cumulative time in one controller replan.
  - GPU active vs idle ratio per side.
  - One concrete recommendation per side for the saturation-tier
    config in Phase C (e.g. "controller is memory-bound on n_anchors;
    bump n_smc instead").

### Phase C — Dual saturation sweep (filter + controller)

This phase has three parallel tracks; each writes a small results
table to `claude_plans/`.

#### C.1 — Filter sweep (existing plan)

**Action:** Add a `--sweep` mode to `tools/profile_gpu.jl` (or a
small new script `tools/profile_gpu_sweep.jl`) that runs the existing
filter-PF microbench over the grid:

| N_smc       | K_per_chain  |
|-------------|--------------|
| 32 (current) | 200 (current) |
| 64           | 400           |
| 128          | 400           |
| 256          | 400           |
| 256          | 800           |
| 512          | 800           |

For each cell, record: median time per call (ms); effective TFLOPS /
104.8 peak %; VRAM after target-build (GB) and after timing (GB);
threads in flight.

#### C.2 — Controller sweep (new)

**Action:** The same sweep harness runs the controller microbench
(`gpu_cost_log_density_batched`) at the controller-side grid. Per
the explore agent's ranked levers, sweep:

| ctrl-n-smc | ctrl-num-mcmc | ctrl-chees-max | n_anchors |
|------------|---------------|----------------|-----------|
| 256 (current) | 8 (current) | 256 (current) | 8 (current) |
| 512        | 8             | 256            | 8         |
| 1024       | 8             | 256            | 8         |
| 1024       | 16            | 256            | 8         |
| 1024       | 16            | 512            | 8         |
| 1024       | 16            | 512            | 12        |
| 1024       | 24            | 512            | 12        |

The controller microbench in `tools/profile_gpu.jl` already exposes
all but `n_anchors` (currently hardcoded). Either:
- (a) Make `tools/profile_gpu.jl` accept an `n_anchors` kwarg and
  pass it into `FSAv1ControlGPUTarget` directly. Bench-side
  `ctrl_n_anchors` stays hardcoded for now — only the microbench
  needs the knob.
- (b) Or also expose `--ctrl-n-anchors` in the bench. (Recommended
  if the sweep finds n_anchors > 8 useful — see C.3.)

For each cell, record: median time per call (ms); effective TFLOPS;
VRAM; M_max = ctrl_n_smc × (1 + 2 × n_anchors) — flag any config
that overshoots reasonable VRAM headroom (e.g. ctrl_n_smc=1024,
n_anchors=16 → M_max = 33,792 chains; verify the
`cost_per_thread`, `theta_per_chain`, and FD-batch buffers fit).

#### C.3 — Expose `n_anchors` in the bench CLI

**Why:** Currently hardcoded at 8 in
`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl:459`. The
explore agent ranks bumping this 8 → 12–16 as the second-most-
effective controller lever (after ctrl-n-smc). Exposing it lets
Phase D actually try bumped values without editing the bench
each time.

**Action:** Add `"ctrl-n-anchors" => 8` to `_parse_args` defaults
with a heavy comment block (mirroring `--ot-max-weight` style)
explaining: linear-in-n_anchors per-thread cost, M_max scales as
(1 + 2 × n_anchors), and that the RBF basis cardinality is
qualitatively different from particle count. Plumb the value through
to the `FSAv1ControlGPUTarget` constructor at line ~459 and into the
local `ctrl_n_anchors` reference (which is also read at the M_max
calculation a few lines later).

#### C.4 — Pick the joint config

Pick the largest (N\*, K\*; ctrl-n-smc\*, ctrl-num-mcmc\*,
ctrl-chees-max\*, n_anchors\*) where:

- Filter VRAM after timing ≤ 24 GB.
- Filter median time per call × ~5000 calls ≤ 25 min.
- Controller VRAM ≤ 8 GB additional (controller buffers are smaller).
- Controller median time per call × ~5000 calls ≤ 5 min total
  controller cost across the whole T=28d run.
- Combined wall-time budget ≤ 30 min total.

**Pass criteria for Phase C:**
- Three markdown tables in `claude_plans/` (filter sweep, controller
  sweep, joint pick).
- One picked config marked, with reasoning ("ctrl-n-smc=1024 is the
  ceiling because M_max × n_inner × Float32 just fits in 8 GB; n_anchors
  beyond 12 is bandwidth-bound by the per-thread RBF decode loop";
  etc.).
- The `--ctrl-n-anchors` CLI flag exists and is heavily commented.

### Phase D — Closed-loop bench at the picked filter + controller config

**Why:** Confirm the microbench saturation generalises to the closed
loop. The controller's behaviour is the headline scientific output —
does the bumped controller produce a richer / more discriminating
schedule than the default at T=28d?

**Action:**
```bash
cd version_1_5_Julia
mkdir -p /tmp/v15_T28d_saturated_julia
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used \
           --format=csv -l 1 \
           > /tmp/v15_T28d_saturated_julia/gpu_telemetry.csv &
NSMI_PID=$!
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
     --T-days 28 --replan-K 2 --seed 42 \
     --N-smc <N*>      --K-per-chain <K*> \
     --num-mcmc 3      --max-temp-levels 30 \
     --ctrl-n-smc      <ctrl_n_smc*> \
     --ctrl-num-mcmc   <ctrl_num_mcmc*> \
     --ctrl-chees-max  <ctrl_chees_max*> \
     --ctrl-n-anchors  <n_anchors*> \
     --output-dir /tmp/v15_T28d_saturated_julia \
     2>&1 | tee /tmp/v15_T28d_saturated_julia/bench.log
kill $NSMI_PID 2>/dev/null
```

**Pass criteria for Phase D (Julia-only — no Python comparison):**

*Saturation (RTX 5090 actually used):*
- Total wall time ≤ 30 min.
- Mean GPU utilisation ≥ 60% (was 24.8% before today's fixes).
- VRAM during run ≥ 50% of 32 GB (≈ 16+ GB used).

*Correctness (algorithm still works at the bigger config):*
- Posterior MSE to truth (final window) ≤ 0.05 (was 0.0868 at the
  small config; bigger N → tighter posterior expected).
- F-violation rate = 0.

*Controller quality (the user's primary interest):*
- Mean Φ over the T=28d run differs from the Phase-A baseline by
  ≥ 0.05 (i.e. richer compute changes the schedule shape).
- The 14 daily Φ values from the saturated run have visibly more
  structure than the baseline (variance across days higher; schedule
  looks less flat / less constant-rest).
- Final-stride controller `n_temp` ≥ 10 (more compute is being used
  to converge a richer posterior).

### Phase E — Record findings (no cross-stack work)

**Why:** With the cross-stack comparison de-scoped, this phase just
captures the Julia-saturation result so the writeup tells the story.

**Action:**
- Drop a small `FINDINGS_T28d_julia_saturated.md` note into
  `compare_v15_julia_vs_python/example_run/` (or a new
  `julia_saturation/` subdir) with: the Phase A baseline numbers,
  the Phase C joint pick + reasoning, the Phase D headline numbers,
  and the controller-quality observations (mean Φ, daily-Φ shape,
  controller `n_temp`).
- Snapshot the bench's PNGs (`v15_T28d_traces.png`,
  `v15_T28d_param_traces.png`) into the same directory.
- Append a short paragraph to `julia_vs_python_v15_writeup.tex` §7
  ("Where we are now") noting: Julia saturation tier picked,
  closed-loop wall time at T=28d, controller-quality observation
  vs the default config. Defer the cross-stack comparison entry
  until the Python+JAX run is done in a future plan.

**Pass criteria for Phase E:**
- Findings note + plots committed.
- Writeup §7 updated with the saturation result.
- A clear statement of "what's next for the cross-stack
  comparison" left for the next plan.

## Critical files

**Filter side:**
- `version_1_5_Julia/tools/profile_gpu.jl` — extend with `--sweep`
  mode (or write `profile_gpu_sweep.jl` next to it).
- `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` — already
  has `--ot-max-weight`, `--liu-west-a`, `--N-smc`, `--K-per-chain`,
  `--num-mcmc`, `--max-temp-levels`. Add `--ctrl-n-anchors` here
  (Phase C.3), heavily commented.
- `version_1_5_Julia/models/fsa_high_res/gpu_pf.jl` — `FSAGPUTarget`
  constructor + the segmented PF kernels.

**Controller side:**
- `version_1_5_Julia/models/fsa_high_res/gpu_control.jl` —
  `FSAv1ControlGPUTarget` (constructor that needs `n_anchors` plumbed
  through), `gpu_cost_log_density_batched`, `make_log_density_fn`.
- `julia/SMC2FC_functional/src/Control/GPUControlSMC.jl` — the
  framework's `run_tempered_smc_gpu` and the ChEES picker. Read-only
  in this plan; the framework is unchanged.
- `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl:459` —
  current hardcoded `ctrl_n_anchors = 8`; this is the line to change
  for Phase C.3.

**Writeup (Julia-only update):**
- `compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.tex`
  — needs a §7 paragraph after Phase E with the Julia saturation
  numbers. The cross-stack table stays at its current values until
  the comparison plan picks them up later.

**Out of this plan:**
- `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh` —
  the paired launcher (also runs Python+JAX). Not invoked here.
- `version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py` —
  the cross-stack diff. Not invoked here.

## Existing utilities to reuse (do NOT re-implement)

- `FSAGPUTarget` and the segmented PF kernels live in
  `version_1_5_Julia/models/fsa_high_res/gpu_pf.jl`. The constructor
  takes `M_max` directly; bumping N just means passing a larger N to
  the bench's CLI.
- `gpu_log_density`, `gpu_grads`, `parallel_hmc_one_move!` are the
  public surface — model-side hot path, do not edit.
- `FSAv1ControlGPUTarget` and `gpu_cost_log_density_batched` in
  `gpu_control.jl` — the public surface for the controller. The
  constructor already takes `n_anchors` as a kwarg; the bench just
  hardcodes 8. Phase C.3 only changes the bench, not the model.
- `run_outer_smc` in the bench file is the hand-rolled tempered SMC²
  loop for the FILTER side; do not reach into framework-side SMC²
  for the filter. The controller side already calls
  `SMC2FC_functional.run_tempered_smc_gpu` — leave that wiring alone.
- `nvidia-smi` 1 Hz CSV sampling pattern is already in the launcher.

## Verification (pulled together)

A single end-to-end sanity check after all phases:

```bash
# Sanity: existing GPU PF tests still green
cd version_1_5_Julia && julia --project=. tools/test_gpu_pf.jl

# Sanity: model+lean diff test still green (proves model unchanged)
julia --project=. diff_test/test_lean_diff_v15.jl
```

Both should pass.

## Out-of-scope items recorded for future plans

- **Multi-seed robustness at saturation tier** (writeup §8 item 5).
- **Cost-function variants if T=28d still picks rest** (writeup §8
  item 4).
- **Batched-across-chains OT-rescue GPU kernel** (writeup §6.8).
- **Re-introducing fp64 on the Julia controller** to match Python's
  controller precision and explain the controller cost / call gap
  (Julia 0.4 ms vs Python 23.4 ms — likely fp32 vs fp64).
