# GPU-starved-phase diagnosis and fix for the v5 bench

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

**Top three candidates for the ~20% phase**, in priority order:

1. **Plant rollout** — fully CPU; runs once per stride; Julia loop with
   per-bin RNG draws and SDE updates in fp64. At 48 bins × 6 state
   dims + 6 noise draws + 5 obs draws per bin, this could be 1–2 s
   per stride that the GPU sees as idle.
2. **Filter inter-λ-level CPU bookkeeping** — between each
   `gpu_log_density_v5` call inside `run_outer_smc`, the bench does
   ESS bisection, systematic resample, then either Liu-West (cheap
   per-dim) or smooth_resample (Silverman-bandwidth KDE on M×M
   matrix). The smooth_resample at M=512 is M² = 262K floats — small,
   but the kernel-matrix and re-blending step is CPU and could be a
   few hundred ms.
3. **Controller-target reconstruction** — every replan, the bench
   builds a fresh `FSAv5ControlGPUTarget`, which allocates CRN noise
   (CPU `randn` of `n_inner × n_steps × 6` Float32) and copies it to
   GPU. At T=42, n_inner=64, n_steps=4032: the noise grid is ~6 MiB
   generated via Julia's RNG. Could be ~1 s per replan.

Diagnosis pins down which.

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

- If `t_plant_rollout` (~1–2 s) coincides with a 1–2 s dip in
  `nvidia_smi.csv:gpu_util_percent`, plant rollout is the culprit.
- If the dip is only at the *end* of a filter cycle (post-`gpu_log_density_v5`,
  during the resample / Liu-West / HMC-prep window), the
  filter inter-λ-level bookkeeping is the culprit.
- If the dip aligns with `controller_plan_v5` setup time, the
  controller-target reconstruction is the culprit.

A small Julia script (~30 lines) that joins the two timestamp series
can produce a definitive correlation plot. Put it at
`tools_v5/diagnose_gpu_starvation.jl`.

### Phase D — Fix the identified phase

Diagnosis-conditional. Three branches:

**If plant rollout is the culprit:**
- Plant rollout is sequential per bin (each bin depends on the
  previous), so it's NOT embarrassingly parallel — bad GPU fit.
- Better: stay on CPU, kill the per-bin Julia overhead. Concretely:
  use `StaticArrays.SVector{6, Float64}` end-to-end (already used in
  `_plant_v5.jl`), avoid `vcat` in the inner loop, hoist allocations.
  Likely 5-10× speedup.
- If after that the per-stride plant wall is still > 200 ms,
  consider running the plant on GPU as a SINGLE KA kernel with
  ndrange=1 (one thread, just to keep the GPU warm and avoid the
  CPU↔GPU context switch cost).

**If filter inter-λ-level CPU bookkeeping is the culprit:**
- Move the M×M `log_kernel_matrix` and `silverman_bandwidth`
  computations to GPU using `KernelAbstractions`. Same algorithm,
  just GPU-resident.
- `smooth_resample` becomes a single kernel call instead of CPU
  matrix arithmetic.

**If controller-target reconstruction is the culprit:**
- Pre-allocate ONE `FSAv5ControlGPUTarget` at bench start and reuse
  across replans. Today each replan allocates a fresh `noise_grid`
  on CPU and copies to GPU; the GPU buffers should be reusable.
- Move `randn` for the CRN noise grid to a GPU-side `randn!` (CUDA.jl
  supports this) so no host→device copy is needed.

## Files to read while executing

- [`tools_v5/bench/bench_loop.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_loop.jl) — instrumentation site
- [`tools_v5/bench/bench_filter.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_filter.jl) — for inter-λ-level CPU code if that's the culprit
- [`tools_v5/bench/bench_controller.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_controller.jl) — for replan reconstruction code
- [`tools_v5/bench/bench_postproc.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools_v5/bench/bench_postproc.jl) — `per_stride.csv` writer needs new columns
- [`models/fsa_v5/_plant_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/_plant_v5.jl) — `plant_rollout_v5` if it's the culprit
- [`models/fsa_v5/gpu_pf_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_pf_v5.jl), [`gpu_control_v5.jl`](/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl) — confirmed fp32 inner loops; not the bottleneck

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
