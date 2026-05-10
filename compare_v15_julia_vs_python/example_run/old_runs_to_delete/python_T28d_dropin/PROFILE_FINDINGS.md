# Why the Python bench's GPU sits idle most of the time

Date: 2026-05-08

## TL;DR

**Of 88 s of `nvidia-smi` telemetry sampled at 1 Hz during the failed
T=28d run, 77 % was GPU < 10 % util, 22 % was GPU 90-100 %.** Almost
all of the idle time is host-side: device→host copies on the
critical path, per-particle GPU launches, and (the dominant cost)
**a fresh JIT compile of the controller's `cost_fn` closure on every
replan stride**.

This is the controller-side analogue of the filter-side anti-pattern
that CLAUDE.md explicitly forbids:

> ### **MANDATORY: use the JAX-native compile-once SMC kernel path. NOT BlackJAX.**

The filter side already has `make_gk_dpf_v3_lite_log_density_compileonce`.
The controller side does **not** — `build_control_spec(...)` is called
inside `_replan_one_stride` on every replan, freshly building the
`schedule_from_theta`, `em_step`, `cost_fn`, and `traj_sample_fn`
JIT'd closures. Each rebuild has different captured-constant values
(posterior-mean params, smoothed end-of-window state) so JAX's trace
cache misses and XLA recompiles from scratch — tens of seconds of
host-side compile time per replan, during which the GPU is idle.

The functional rewrite I committed earlier today preserved
bit-identical numerics from the imperative source but inherited
this perf antipattern (and added a few new device→host syncs in the
new accumulator structure). **The agent that wrote this bench
(me, in the previous session) messed up — the user is right.**

## Telemetry summary

```
util %   buckets       count    fraction
0-10%    (idle)        68       77 %
10-50%                 1        1 %
50-90%                 0        0 %
90-100%  (saturated)   19       22 %
```

Pattern (each digit is one second of GPU util %):
```
1 1 1 1 2 1 1 1 1 1 2 1 1 1   ← 14s startup + first plant rollout (idle)
100 100 100 100 100 100 100 100 100   ← 9s filter tempering level (busy)
4 1 1 1 2 1 0 1 1 2 1   ← 11s host-side between filter levels (idle)
100   ← 1s burst
1 1 1 1 1 1 1 1 1 0 1 1   ← 12s idle
100 100 100 100 100 100 100 100 100   ← 9s tempering level (busy)
5 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 2 1 1   ← 19s idle (controller compile begins)
3 3 10 6 7 5 9 9 9 9 9 6   ← 12s low util (HLO compile / first HMC trace)
                          ← then OOM
```

## Root causes, ranked by impact

### 1. (DOMINANT) `build_control_spec` rebuilds the JIT'd controller every replan

**File:** `version_1_5_Python_JAX_functional/tools/bench_smc_full_mpc_fsa_v15.py:707`
```python
ctrl_spec = build_control_spec(
    T_total=float(cfg.T_days), dt_days=env.DT_BIN_DAYS,
    params_v15=params_v15, init_state=init_state_named,
    n_inner=cfg.ctrl_n_inner,
    n_anchors=cfg.ctrl_n_anchors,
)
```

`build_control_spec` (in
`version_1_5_Python_JAX_functional/models/fsa_high_res/control.py`)
allocates fresh JIT'd closures every call:

* `_make_schedule(...)` returns a new `@jax.jit schedule_from_theta`
* `_make_em_step_fn(params_v15, dt, n_substeps)` bakes the v1-rotated
  parameters as JAX scalars into the closure of a new `@jax.jit em_step`
* `_build_cost_and_traj_fns(...)` returns a new `@jax.jit cost_fn` and
  `@jax.jit traj_sample_fn`, both closing over the new `em_step`
* `build_crn_noise_grids(...)` allocates a fresh `(n_inner, n_steps, 3)`
  noise tensor

The `cost_fn` is then handed to `run_tempered_smc_loop_native`, which
wraps it in `jax.tree_util.Partial` and passes it to the module-
level-JIT'd `_run_tempered_chain_jit`. **Because the underlying
PjitFunction object is new on every replan, the Partial pytree's
identity changes, JAX's trace cache misses, and XLA recompiles the
entire HMC + tempered-SMC kernel HLO.**

At T=28d the controller does ~27 replans. Each compile takes
on the order of 10-30 s (cf. CLAUDE.md "~15 s of HLO recompile per
stride" on the filter side that the compile-once path closed). That
is potentially 5-15 min of pure host-side compile time over the
run, all of which the GPU sits at 0 %.

**The filter side has already solved this**:
`smc2fc.filtering.gk_dpf_v3_lite.make_gk_dpf_v3_lite_log_density_compileonce`
(used at line 158 of the bench) builds the log-density factory
ONCE; per-stride per-replan dynamic data is bound via
`jax.tree_util.Partial`. The bench then uses
`smc2fc.core.jax_native_smc.run_smc_window_native` which is
module-level JIT'd. **The controller has no equivalent
compile-once factory.**

### 2. Per-particle GPU launches in posterior-cloud construction

**File:** `bench_smc_full_mpc_fsa_v15.py:600-606` (filter path)
```python
particles_np = np.asarray(particles)
samp_constrained = np.array([
    np.asarray(unconstrained_to_constrained(jnp.asarray(p), env.T_arr))
    for p in particles_np
])
```

**File:** `bench_smc_full_mpc_fsa_v15.py:690-697` (replan path —
identical anti-pattern)

For `--n-smc 512`, this issues **512 sequential tiny GPU launches**
per stride. Each launch has ≈1 ms of CPU launch latency
dominating the µs of actual GPU work, and forces a host-side
gather to materialise `samp_constrained`. Total ≈0.5 s of host-
bottlenecked tiny launches per stride × 56 strides ≈ 30 s.

Should be a single `jax.vmap(unconstrained_to_constrained, ...)` call
operating on the whole (N, n_dim) particle array, then one numpy
materialisation at the end.

### 3. `np.asarray(rollout.X)` on every plant-rollout output

**File:** `bench_smc_full_mpc_fsa_v15.py:_advance_plant_one_stride` lines 493-497
```python
new_acc = acc._replace(
    traj_chunks=acc.traj_chunks + (np.asarray(rollout.trajectory),),
    obs_B_chunks=acc.obs_B_chunks + (np.asarray(rollout.obs_B),),
    obs_F_chunks=acc.obs_F_chunks + (np.asarray(rollout.obs_F),),
    obs_A_chunks=acc.obs_A_chunks + (np.asarray(rollout.obs_A),),
    ...
)
```

Each `np.asarray(jnp_array)` is a **synchronous device→host
transfer**. Four of them per stride. The plant rollout itself is
small (12 bins × 3 states) so the transfer is fast — but it forces
the GPU to drain and the CPU to wait, breaking pipelining between
plant rollout and the subsequent filter call.

The original imperative source kept obs as Python lists of floats
(also host) so the issue was already present there — the
functional rewrite did not introduce it; it just preserved the
sync.

### 4. `np.concatenate` on every window assembly

**File:** `bench_smc_full_mpc_fsa_v15.py:_accumulated_obs_for_window` lines 521-535
```python
obs_B = np.concatenate(acc.obs_B_chunks) if acc.obs_B_chunks else np.zeros(0)
if obs_B.shape[0] < window_bins:
    return None
obs_F = np.concatenate(acc.obs_F_chunks)
obs_A = np.concatenate(acc.obs_A_chunks)
Phi = np.concatenate(acc.Phi_chunks)
return {
    'obs_B': obs_B[-window_bins:].astype(np.float64),
    ...
}
```

Pure host work. Then the result is fed into `align_obs_fn` which
issues `jnp.asarray(...)` (host→device transfer, blocks on copy).

For T=28d (56 strides, 12 bins each), the chunks tuple grows to 56
elements; concatenation work is small (672 floats per channel) but
adds up because it runs every stride. Should accumulate as a
device-resident jnp array once and slice on device.

### 5. `_stride_telemetry` re-concatenates the entire history every stride

**File:** `bench_smc_full_mpc_fsa_v15.py:_stride_telemetry` ~line 781
```python
if acc.traj_chunks:
    full_so_far = np.concatenate(acc.traj_chunks, axis=0)
    A_mean_so_far = float(np.mean(full_so_far[:, 2]))
```

This is O(n²) over the run — at stride 56 it concatenates 56 chunks
of 12 rows each. Small in absolute terms (672 rows) but
unnecessary. Should track a running sum/count.

### 6. JIT first-call compile is unmonitored

The first filter call (cold-start path) takes ≈10 s to compile. The
first filter-bridge call (separate trace) is another ≈10 s. The
first controller call is tens of seconds. CLAUDE.md says the bench
should set `JAX_COMPILATION_CACHE_DIR=~/.jax_compilation_cache` so
the second-and-onwards run reuses the on-disk cache; the bench
script does not set this. (Cause #1 above is a separate issue —
it's per-replan recompile, which the cache CAN'T fix because every
replan has different cache key.)

## What is already on the GPU continuously

Within a single tempering level of the filter or controller, the
inner SMC kernel + the inner-PF / cost_fn rollout do run at 90-100 %
util (the 22 % of the telemetry showing 100 %). The kernels
themselves are fine — `gk_dpf_v3_lite` uses `jax.lax.scan`,
`jax.checkpoint`, the controller's `cost_fn` uses
`jax.lax.scan` over `jax.vmap(trial)`. The problem is what happens
*around* the kernel calls.

## Proposed fixes (await user direction before applying)

In priority order:

1. **Add a `build_control_spec_compileonce` factory** to the v1.5
   functional model, mirroring `make_gk_dpf_v3_lite_log_density_compileonce`.
   * Takes static config (n_steps, n_substeps, n_inner, n_anchors,
     dt) at build time.
   * Returns a callable `cost_fn(theta, params_v15_arr, init_arr)`
     where `params_v15_arr` and `init_arr` are JAX-array runtime
     arguments (not closure-captured).
   * Build it ONCE in `_build_env`; bind per-replan dynamic data
     via `jax.tree_util.Partial` inside `_replan_one_stride`.
   * Estimated saving: ≈10-30 s per replan × 27 replans = 5-15 min
     of compile time eliminated. Plus the GPU stays continuously
     warm between replans instead of cycling cold.

2. **Vectorise the per-particle posterior-cloud construction** in
   `_filter_one_stride` and `_replan_one_stride`:
   ```python
   samp_constrained = np.asarray(
       jax.vmap(lambda p: unconstrained_to_constrained(p, env.T_arr))(particles)
   )
   ```
   One GPU call instead of 512. Trivial change (≈4 lines).

3. **Keep accumulators device-resident.** Replace `np.asarray` on
   the rollout outputs with no-op (jnp arrays go directly into the
   tuple). Replace `np.concatenate` in
   `_accumulated_obs_for_window` with `jnp.concatenate` + `.at[]`
   slicing on device.

4. **Set `JAX_COMPILATION_CACHE_DIR`** in the bench script header
   so first-run compile is one-time per `(machine, JAX-version)`
   pair, not one-time per process.

5. **Drop `_stride_telemetry`'s O(n²) concat.** Maintain a running
   `(A_sum, A_count, last_traj_chunk)` tuple in the accumulators
   and read from that.

Fix 1 alone should change the operating point from "≈22 % mean GPU
util, 77 % idle" to something closer to Julia's quoted 49.5 % at the
same config. Fixes 2-5 are smaller but cumulative.

## Separate issue: the OOM at the first replan

The bench OOM'd allocating 27.96 GiB during the first replan call
(`run_tempered_smc_loop_native`'s `final_particles.block_until_ready()`).
At the matched-Julia config (`--ctrl-n-smc 2048 --ctrl-num-mcmc 16
--ctrl-n-anchors 12 --ctrl-n-inner 64` over `n_steps = 28×24 = 672`
bins) the HMC reverse-mode AD tape per leapfrog step dominates:
roughly N_smc × n_anchors × leap × n_inner × n_steps × bytes ≈
2048 × 12 × 16 × 64 × 672 × 8 B = ~135 GiB tape (before
checkpointing kicks in to bring it down to ≈28 GiB, which is what we
saw). The Julia bench survives this because Julia's ChEES adapts the
leapfrog count and the same kernel is in CUDA.jl + KernelAbstractions.jl
(less Python-overhead per AD step).

Likely fixes:
* `--ctrl-num-mcmc 8` (Julia uses 16; halve it). Cuts AD tape.
* `--ctrl-n-inner 32` (Julia uses 64; halve it). Cuts AD tape.
* Add `jax.checkpoint` to the inner cost-rollout `scan` body —
  trades compute for memory.

These are separate from the GPU-utilisation issue. The user was
right to stop the run; we should not re-launch T=28d until both
issues are addressed.
