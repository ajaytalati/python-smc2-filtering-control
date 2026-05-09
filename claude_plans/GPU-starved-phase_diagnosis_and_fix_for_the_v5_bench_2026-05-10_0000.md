# GPU-starved-phase diagnosis and fix for the v5 bench

> Archived from plan mode: 2026-05-10 00:00.
> Updated: 2026-05-10 00:15 — user corrected nvtop observation from ~1–2 s dip to ~40 s up + 40 s down on ~80 s cycle; this matches replan-stride cadence and the BOTE for ~2,600 controller-HMC log-density calls per replan, so candidate ranking collapses to controller cost-density transfer + sync as the only viable suspect; plant rollout and filter bookkeeping moved out of scope; Phase D's plant-rollout branch removed.
> Updated: 2026-05-10 00:46 — Phase B finished. Controller cost-density TRANSFER hypothesis FALSIFIED. Fine instrumentation inside `gpu_cost_log_density_batched_v5` (T=4 steady-state, strides 4 + 6) shows: kernel + sync = 73% of replan wall, input transfer = 5%, output transfer + CPU sum = 10%, framework CPU "other" = 12%. The v1.5-`da847d9` fix template would save ~10-15% of replan wall, not transformative. nvidia_smi peak util 74% matches the 73% kernel-busy fraction — the GPU is NOT starved, it's running at ~70-75% util because of small inter-call CPU overheads. The dominant cost is the kernel itself (1.75 ms/call). Phase D's controller-cost-density GPU reduction NOT justified by data. Awaiting user direction on next step (kernel optimisation vs accept current perf).
> Updated: 2026-05-10 00:57 — Phase D shipped per user direction. Implementation: `gpu_cost_accumulate_kernel_v5!` + `cost_acc_gpu :: CuArray{Float64}` field on `FSAv5ControlGPUTarget`; refactored `gpu_cost_log_density_batched_v5` to do n_inner reduction on GPU and read back only M Float64 (was M·n_inner Float32, ~32× less PCIe data); dropped explicit `synchronize` after cost kernel (the final readback syncs once). Diff test 424/424 unchanged. T=4 measurement: replan wall 7.6s → 6.9s (-9.3%), peak GPU util 74% → 85% (+11pp), total bench wall 189.4s → 186.4s (-1.6% — small because filter dominates this config's wall and filter is unchanged). Projected T=42 saved-wall: ~5-6 min on a 73-min run (controller-only fix). Phase D outcome: real but modest improvement, matched the hedged prediction.

## Overview

User observed on `nvtop` that a v5 bench stride has two GPU-utilisation
regimes: a **~90% saturated** phase (the kernels) and a **~20% starved**
phase. The starvation isn't fp64-on-GPU — `gpu_pf_v5.jl` and
`gpu_control_v5.jl` are fp32 throughout
([gpu_pf_v5.jl:54-89](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl) verified). So the starvation is a CPU-bound
code path between kernel launches. Goal: identify which one, then move
it to GPU or otherwise eliminate the stall.

## Motivation

Per-stride wall is currently dominated by both phases. If the starved
~20% phase is, say, 30% of stride wall, eliminating it could cut bench
total wall by close to a third without touching the math. Worth doing
properly: diagnose first, then fix the right thing.

## Strategy in one paragraph

Don't speculate. **Instrument** `bench_loop.jl::run_one_stride_v5`
with per-phase wall timers, run a short diagnostic bench (T=2 days, 3
strides), then correlate the per-phase timing log with the existing
`nvidia_smi.csv` 1 Hz GPU-util sampler that the launcher already
captures. Whichever phase's wall window aligns with the dip in GPU
utilisation is the culprit. Then fix THAT phase — either migrate it
to GPU, pre-allocate its buffers, or remove unnecessary CPU work.

## User's prior + my reading of v1.5 prior art (2026-05-09)

User asked me to first read v1.5's `models/fsa_high_res/` because that
code "has been optimised PREVIOUSLY to remove this bug" — the same
GPU-starvation symptom on the v1.5 bench was fixed earlier this week.

What I found in v1.5's git history (commits `c4a9e86`, `da847d9`,
`099a14a`, `9bab259`):

| v1.5 fix | Commit | v5 status |
| --- | --- | --- |
| GPU-resident `log_lik_acc_gpu` buffer in `gpu_log_density`; killed 12 per-segment `Array(view(...))` PCIe transfers + 6 redundant `synchronize` per call | `da847d9` | ✓ inherited at [gpu_pf_v5.jl:284, 521, 571, 579](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl#L284) — `gpu_log_lik_accumulate_kernel_v5!` + `target.log_lik_acc_gpu` |
| OT rescue OFF by default (was 116× slower at M=32 because framework's `gpu_ot_blend_chain!` does CPU softmax per chain) | `099a14a` | ✓ inherited — `ot_max_weight = 0.0` default at [gpu_pf_v5.jl:357](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl#L357), `OT_MAX_WEIGHT=0.0` is the bench env default |
| Mutating `parallel_hmc_one_move!` reuses persistent buffers | `c4a9e86` | **superseded by framework migration** — both v1.5 and v5 use framework ChEES HMC: filter side calls `parallel_hmc_one_move_generic!` from `SMC2FC_functional` (see [v5 bench_filter.jl:161-193](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_filter.jl#L161-L193) "Switched from the model-specific `parallel_hmc_one_move!` … to the framework's `parallel_hmc_one_move_generic!` + `chees_pick_L_generic`"); controller side calls `run_tempered_smc_gpu` from the same framework (which internally uses the same ChEES machinery). The model-side `parallel_hmc_one_move` / `parallel_hmc_one_move!` in `gpu_pf.jl` and `gpu_pf_v5.jl` are **dead code** in both versions, kept only as historical reference under the "NOT USED NOW — CHEESE HMC IS THE DEFAULT" annotation. |

So **all three v1.5 starvation fixes are in place on the v5 FILTER
side** (two via inheritance, one via the framework). The handover-doc
claim "plumbing copied verbatim from v1.5; only math block swapped"
holds up under inspection.

**Important consequence**: because both filter and controller HMC live
in `SMC2FC_functional`, the only place a v1.5-`da847d9`-style starvation
pattern can still be alive in model-side code is on a non-HMC GPU
surface. That narrows the search down to `gpu_log_density_v5` (which
already has the fix) and `gpu_cost_log_density_batched_v5` (which
doesn't — see the next section).

## What v1.5 did NOT fix — same pattern still unfixed in BOTH versions on the CONTROLLER side

The pattern that `da847d9` killed on the filter-side `gpu_log_density`
is **still present** on the controller-side `gpu_cost_log_density_batched`
in BOTH v1.5 AND v5:

[gpu_control_v5.jl:381-396](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl#L381-L396)
(verbatim copy of v1.5's [gpu_control.jl:257-268](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_control.jl#L257-L268)):

```julia
KernelAbstractions.synchronize(CUDABackend())                # forced GPU stall
cost_cpu = Array(view(target.cost_per_thread, 1:Ntot))       # M·n_inner Float32 → CPU
cost_mat = reshape(cost_cpu, target.n_inner, M)
@inbounds for m in 1:M
    s = 0.0
    for t in 1:target.n_inner
        s += Float64(cost_mat[t, m])
    end
    out[m] = -s / target.n_inner
end
```

This runs **every controller-HMC log-density call** — once for the
M-chain log-density and once for the `M·(1+2·n_anchors)`-chain
FD-gradient batch. At the bench's small-filter / saturated-controller
config (`ctrl_n_smc=256`, `ctrl_n_anchors=8`, `n_inner=64`):

- log-density call: `Ntot = 256 × 64 = 16,384` Float32 = 64 KiB → CPU
- gradient batch:   `Ntot = 4,352 × 64 = 278K` Float32 = ~1.1 MiB → CPU

per call, with a hard `synchronize` forcing the GPU to drain.
Repeated across every HMC step in every tempering level in every
replan. **This mirrors the v1.5 PF starvation bug 1:1, just on a
different GPU surface — and unlike the PF version, neither v1.5
nor v5 has fixed it yet.**

## User's corrected nvtop observation (2026-05-10)

User flagged that the starved phase isn't 1–2 s as the original plan
said — it's **about 40 s up + 40 s down on a ~80 s cycle**. Cross-
referenced against the live bench log of the in-progress T=42 run
(`compare_v15_julia_vs_python/example_run/julia_v5_T42d_2026-05-09/T42d_seed42/bench.log`):

- non-replan stride: ~25 s wall (filter only)
- replan stride: 26 s filter + 58 s replan = **~84 s wall**
- Replan cadence: every K=2 strides ⇒ matches the 80 s starved/
  saturated period 1:1.

**Conclusion: the 40 s of starvation lives inside the 58 s replan
phase**, not in the filter or plant. This categorically rules out:

- **Plant rollout** (BOTE: ≈ 1 ms / stride; three orders of
  magnitude off the observed dip)
- **Filter inter-λ-level CPU bookkeeping** (a few hundred ms; runs
  every stride, but the dip is ONLY on replan strides per the
  per-stride wall pattern)

Only the controller-side suspect has the right time budget. BOTE for
the controller HMC call count per replan, at saturated config
(`ctrl_n_smc=256`, `ctrl_n_anchors=8`, `n_inner=64`,
`ctrl_num_mcmc=8`, ChEES HMC with `L=64` typical from the bench log,
5–6 tempering levels per replan):

- 1 energy eval + L gradient evals per HMC move ≈ 65 calls / move
- 5–6 levels × 8 moves / level ≈ 40–48 moves per replan
- **≈ 2,600 calls to `gpu_cost_log_density_batched_v5` per replan**
- Each call: forced `synchronize` + `Array(view(cost_per_thread, 1:Ntot))`
  of M·n_inner Float32. At gradient-batch size `Ntot = 4352·64 = 278K`
  Float32 ≈ 1.1 MiB per call.
- At ~10–15 ms per call (a forced PCIe sync plus the transfer) →
  **26–39 s of stalls per replan**.

That fits the user's 40 s starvation observation almost exactly. The
controller cost-density transfer + sync is the only candidate
consistent with the measured time-budget.

## Plant rollout (deprioritised — categorically ruled out)

Reading [_plant_v5.jl:75-107](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/_plant_v5.jl#L75-L107)
the visible CPU smell is per-bin `StableRNG` allocation (96 fresh
RNG inits per stride). BOTE puts the rollout at ≈ 1 ms / stride —
**~40,000× too small** to be the 40 s starvation phase. v1.5's
`_plant.jl` uses the same pattern and was never fixed for
performance reasons. Plant rollout is not the bottleneck on this
bench at this config and is not in scope for Phase D.

## Suspect inventory (what could be starving the GPU)

The per-stride flow in
[bench_loop.jl::run_one_stride_v5:151-252](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl) is:

| Phase | Implementation | GPU? | Likely cost |
| --- | --- | --- | --- |
| 1. Slice plan into per-stride Φ | [bench_loop.jl:158-160](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl#L158-L160) | CPU | tiny |
| 2. Plant rollout | [_plant_v5.jl::plant_rollout_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/_plant_v5.jl) | **CPU only** — Julia loop over 48 bins per stride | could be 1–2 s per stride; **likely starvation suspect #1** |
| 3. Accumulate obs/traj history | [bench_loop.jl::accumulate_obs_history_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl#L65) | CPU | `vcat` calls; small but allocation-heavy |
| 4. Build window grid_obs | [bench_glue_v5.jl::window_grid_obs_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/bench_glue_v5.jl) | CPU | array slices + 13-field NamedTuple build |
| 5. Filter outer SMC² (`run_outer_smc`) | [bench_filter.jl](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_filter.jl) | mostly GPU | per-λ-level: `gpu_log_density_v5` (GPU, big), then CPU bookkeeping (ESS bisection, systematic resample, Liu-West / smooth_resample), then HMC moves (GPU) |
| 6. `extract_xhat` | [bench_filter.jl:225](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_filter.jl) | GPU memcpy + CPU `mean` | tiny |
| 7. Controller plan (every K=2 strides) | [bench_controller.jl::controller_plan_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_controller.jl) | GPU + CPU | builds `FSAv5ControlGPUTarget` (CPU + memcpy), runs framework `run_tempered_smc_gpu` (GPU + CPU bookkeeping per level), decodes Φ schedule (CPU) |
| 8. Per-stride log row | [bench_loop.jl::compose_per_stride_log_row_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl#L117) | CPU | tiny |

**Candidate ranking after the corrected 40 s observation** —
collapses to essentially one candidate:

1. **Controller cost-density `Array(view(cost_per_thread))` + sync
   on every HMC log-density call** (~2,600 calls per replan, at
   ~10–15 ms each ⇒ 26–39 s of stalls per replan, vs the observed
   40 s starvation). Mirrors the v1.5 `da847d9` PF bug 1:1, just on
   the controller surface. Fix template: GPU-resident accumulator +
   reduction kernel; return only M Float64 to CPU.
2. **Controller-target reconstruction at the start of each replan**
   — fresh `FSAv5ControlGPUTarget` allocation + CPU `randn` of the
   CRN noise grid + memcpy. ~1 s per replan plausible. **Too small
   to be the dominant 40 s phase on its own**, but worth checking
   that it isn't an additional ~1 s of starvation at the start of
   each replan that adds to (1).
3. **Filter inter-λ-level CPU bookkeeping** — runs every stride;
   the per-stride wall pattern (25 s flat vs 84 s on replan) shows
   filter strides are NOT starved, so this is ruled out as the
   dominant phase. Keep on the watch list only as a secondary check.

Diagnostic still runs — but the BOTE math + per-stride-wall pattern
+ v1.5 prior art all point to suspect #1 with very high confidence.
Phase B's purpose is now confirmation rather than disambiguation.

## Phases of the plan

### Phase A — Instrument (cheap, no behavioural change)

Edit [tools_v5/bench/bench_loop.jl::run_one_stride_v5](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl) to capture
per-phase wall in seconds:

- `t_slice_phi`     (line ~160)
- `t_plant_rollout` (line ~163)
- `t_accumulate`    (lines ~166–167; obs + traj history append)
- `t_window_grid`   (line ~180; `window_grid_obs_v5`)
- `t_filter`        (lines ~181–185; `run_outer_smc`)
- `t_xhat`          (line ~186; `extract_xhat`)
- `t_replan`        (lines ~196–215; `controller_plan_v5` if it fires)
- `t_postlog`       (line ~232; the log-row + accumulator updates)

Add an `@info` line after each stride logging the breakdown:

```
[stride X] phase walls (s): plant=1.20  acc=0.05  grid=0.01
                              filter=9.50  xhat=0.02  replan=4.30
                              postlog=0.04   total=15.12
```

Also extend `compose_per_stride_log_row_v5` to add the same fields to
each row, and update `bench_postproc.jl::save_bench_outputs`'s
`per_stride.csv` writer to include them. That gives both a
human-readable log and a machine-readable CSV the user can grep.

**No code paths change.** Pure addition of `time()` deltas around
existing calls.

### Phase B — Diagnostic run

Run a short bench:

```
./tools_v5/launchers/run_julia_v5_T7d.sh 2 \
    --filter-N-smc 32 --filter-K-per-chain 200 \
    --ctrl-n-smc 256 --ctrl-num-mcmc 8 --ctrl-chees-max 256 \
    --ctrl-n-anchors 8 --ctrl-max-levels 25
```

T=2 days at the small config produces 3 strides (~1 min wall) and
both an `nvidia_smi.csv` and a `bench.log` with per-phase timings.

### Phase C — Correlate

The launcher already captures `nvidia_smi.csv` at 1 Hz with absolute
timestamps. The new per-stride timings give relative phase walls that
can be aligned with the absolute timestamps in the bench log. From
that:

Expected pattern (matches the user's nvtop observation):

- `t_plant_rollout`, `t_accumulate`, `t_window_grid`, `t_xhat`,
  `t_postlog` all together < 0.5 s per stride — none of these are
  the source.
- `t_filter` ≈ 25 s on every stride, with the GPU **saturated**
  throughout (no dip during filter).
- `t_replan` ≈ 58 s **only on every K=2 strides**, with a **~40 s
  starvation segment inside it**. This is the target.
- Within the replan window, the dip is broken into many small stalls
  (~10–15 ms each, ~2,600 of them per replan) caused by the per-call
  `synchronize` + `Array(view(cost_per_thread, ...))` round-trip. The
  1 Hz `nvidia_smi.csv` sampler will see the AGGREGATE as a low-util
  band, not each individual stall.

If Phase B contradicts any of the above (e.g. `t_filter` shows a
dip too, or `t_replan` is fully saturated), pause and re-diagnose.
Otherwise proceed straight to Phase D's controller-cost-density branch.

A small Julia script (~30 lines) that joins the two timestamp series
can produce a definitive correlation plot. Put it at
`tools_v5/diagnose_gpu_starvation.jl`.

### Phase D — Fix the identified phase

Diagnosis-conditional. Four branches, ordered by the updated prior.

**If controller cost-density transfer + sync is the culprit (mirror
v1.5 `da847d9`):**

This is the highest-prior branch given the v1.5 pattern match. The
fix mirrors `da847d9` 1:1, just on the controller surface:

1. Add a GPU-resident reduction kernel
   `gpu_cost_accumulate_kernel_v5!` next to
   [`gpu_cost_kernel_v5!`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl)
   in `gpu_control_v5.jl`, mirroring `gpu_log_lik_accumulate_kernel_v5!`'s
   structure: ndrange = M, each thread sums its `n_inner` slice of
   `cost_per_thread` into a length-M output buffer (Float64 to
   preserve precision).
2. Add a `cost_acc_gpu :: CuArray{Float64, 1}` field to
   `FSAv5ControlGPUTarget`, allocated once in the constructor at
   `M_max` length. Mirror `target.log_lik_acc_gpu` exactly.
3. Edit `gpu_cost_log_density_batched_v5` to (a) launch the
   reduction kernel after `target.kernel(...)` instead of
   `Array(view(cost_per_thread, ...))`, (b) drop the per-call
   `KernelAbstractions.synchronize` (the final `Array(view(cost_acc_gpu, 1:M))`
   syncs once), (c) read back only the M-element accumulator. Final
   CPU step is just `out[m] = -cost_acc_gpu[m] / n_inner` per chain.
4. Keep `make_log_density_fn_v5` and the v5 diff-test entry
   `gpu_cost_one!` mathematically identical — the diff-test asserts
   v5/Lean parity at `1e-3` and the change is a pure transfer-pattern
   fix, not a maths change.
5. **Mirror the v1.5 commit message and verification protocol**:
   T=2d closed-loop, before/after wall comparison, diff test
   424/424, mean A within stochastic variation.

Expected wall delta: at gradient-batch (`Ntot ≈ 278K` Float32), the
PCIe transfer is ~1.1 MiB → ~17 KiB per call (M=4352 Float64). That
is ~64× less data per call AND removes the forced sync. v1.5's
filter-side equivalent gave 12.77× total wall reduction at small N
(per `da847d9`'s commit message); the controller is hit per HMC
step rather than per filter call so the per-call frequency is even
higher, but how much of the 20% starvation dip comes from THIS
versus the framework's per-tempering-level CPU bookkeeping is what
Phase B's timer settles.

**If controller-target reconstruction shows up as a ~1 s additional
stall at the start of each replan** (subordinate to #1):
- Pre-allocate ONE `FSAv5ControlGPUTarget` at bench start and reuse
  across replans. Each replan currently allocates a fresh
  `noise_grid` on CPU and memcpys to GPU; the GPU buffers can be
  reused with a `randn!` refresh in place.
- Move `randn` for the CRN noise grid to a GPU-side `randn!` (CUDA.jl
  supports this) so no host→device copy is needed.

**Plant rollout and filter inter-λ-level bookkeeping**: ruled out by
the corrected 40 s observation + the per-stride wall pattern.
Out of scope for Phase D.

## Files to read while executing

- [`tools_v5/bench/bench_loop.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl) — instrumentation site (Phase A)
- [`models/fsa_v5/gpu_control_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl) — top-prior fix site (lines 381–396); also study lines 70, 259 (`cost_per_thread` field)
- [`models/fsa_high_res/gpu_pf.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_pf.jl) — v1.5 fix template (`da847d9`); read `gpu_log_lik_accumulate_kernel!` and how `log_lik_acc_gpu` is wired through `FSAGPUTarget`
- [`models/fsa_v5/gpu_pf_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl) — v5 already has the same template; copy idiom and rename for the controller
- [`tools_v5/bench/bench_filter.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_filter.jl) — for inter-λ-level CPU code if that's the culprit
- [`tools_v5/bench/bench_controller.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_controller.jl) — for replan reconstruction code
- [`tools_v5/bench/bench_postproc.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_postproc.jl) — `per_stride.csv` writer needs new columns
- [`models/fsa_v5/_plant_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/_plant_v5.jl) — only if Phase B contradicts the controller-cost prior

## Existing utilities to reuse

- `nvidia_smi.csv` 1 Hz sampler in
  [`tools_v5/launchers/run_julia_v5_T7d.sh`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/launchers/run_julia_v5_T7d.sh) — already in place
- Julia's `time()` for cheap monotonic wall measurements
- `CUDA.@profile` for kernel-level timing if Phase C points at a
  GPU-side issue (not the working hypothesis)
- Framework's `run_tempered_smc_gpu` already has optional
  `collect_diagnostics=true` mode that emits per-tempering-level
  walls to `controller_diagnostics.csv` — reuse those rows when the
  diagnosis points at the controller path

## Verification

End-to-end test of the fix (after Phase D):

1. **Re-run the same T=2 diagnostic** with the fix applied.
2. **Per-stride wall** should drop by the predicted amount (whatever
   the diagnosed phase contributed).
3. **`nvidia_smi.csv` mean GPU util** should rise; the ~20% dip
   should shrink or disappear.
4. **Diff test still green** — `julia diff_test/test_lean_diff_v5.jl`
   reports 424/424. Any optimisation that breaks math correctness
   has to fail this gate.
5. **Final state at T=2** matches the pre-fix run within stochastic
   variation (same seed → same trajectory, modulo any RNG state
   change introduced by the fix).

## Out of scope (deliberate)

- **Re-tuning the saturated config** in response to better
  utilisation. That's a follow-up benchmarking task; this plan only
  fixes the starvation, not retunes around it.
- **Rewriting the framework's `run_tempered_smc_gpu`** even if the
  diagnosis points at framework-level CPU bookkeeping. The framework
  is shared with v1.5 and changing it has a wider blast radius;
  document the issue and put it on the framework backlog instead.
- **Moving the differential test to GPU.** The diff test is CPU-only
  by design (it's the verification gate, not the production path).
- **Adding new model surfaces.** Strictly throughput-fixing; the
  model maths and the obs schema are unchanged.
- **The currently-running T=42 bench in task `b3ya3k85a`.** It can
  finish first; instrumentation work happens in parallel and the
  fix only lands after the user has approved it.
