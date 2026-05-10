# Practical methods to get SMC²/MPC working on the RTX 5090

This is a **living document**. It records the things we had to do — and
the things we wish we'd known up-front — to take the FSA-v2 closed-loop
SMC² + MPC framework from a 124-min T=14 wall on the original h=15min
BlackJAX code path to a 27-min T=14 wall (with 8× more outer particles
and 4× more inner particles), and to make T=28 closed-loop pass all
acceptance gates after the pre-stage-M version had failed
catastrophically (ratio 0.569, F-viol 75 %, id-cov 4 / 54 windows).

The motivation: every one of these levers cost engineering effort to
discover. If we ever start from scratch — or port to Julia + DiffEq.jl
— we want this list, with the *why*, so that the wall-clock and
identifiability fights don't have to be re-fought.

## A. Hardware / environment caveats

### A.1 Consumer Blackwell FP64 penalty

The RTX 5090 is consumer-Blackwell. Its FP64 throughput is roughly
**1/64** of FP32. Naïvely keeping the whole pipeline in `float64` (the
JAX default with `JAX_ENABLE_X64=True`) leaves most of the silicon idle
during the SDE inner loops.

**What we do**:
- SDE plant integration (`_plant.py:_plant_em_step`) runs in FP32:
  particles, drifts, diffusions, noise increments are all `float32`.
- Log-domain reductions (SMC log-weights, log-likelihood evaluations,
  ESS) **stay in FP64** to avoid catastrophic cancellation when many
  small log-likelihoods are summed.
- Parameter posteriors stay in FP64 — the dynamic range across 30
  parameters spans many orders of magnitude (e.g. `tau_B` ~ 40 vs
  `epsilon_A` ~ 1e-4).
- Mass-matrix Cholesky stays in FP64; the FP32 path is numerically
  unstable for nearly-degenerate covariances.

### A.2 JAX preallocation off

Without these env vars, Wayland + nvidia-590.48.01 + RTX 5090 + JAX
caused **overnight desktop-compositor crashes** and black-screen
freezes (the JAX process grabs all VRAM eagerly, the compositor
starves, the kernel reaps the wrong process):

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.60
```

Lazy allocation + a 60 % cap leaves the Wayland/Xorg compositor
reliably ~13 GB of headroom.

### A.3 Process isolation from VS Code

VS Code's IDE state contributed to the same crash mode. Stable
overnight runs require:

```bash
tmux new -s sweep -d "$HOME/bench_logs/run_h1h_sweep.sh"
tmux ls; tmux attach -t sweep   # from a plain terminal, NOT VS Code
```

Run benches in a tmux session whose ancestor is `bash`, **not**
`code`. Reattach from a plain terminal, never from VS Code's
integrated terminal. The bench can then survive VS Code being closed
or crashing.

### A.4 Parallel-horizon multi-process (one GPU, multiple horizons)

A single bench run uses ~8 GB of VRAM on the 5090's 32 GB and pulls
~340 W of the ~575 W TDP cap during kernels — leaving **~24 GB and
~50 % of TDP** unused. The simplest way to put that headroom to work
is multi-process: launch independent bench processes (different `T`,
or different seeds, or different model variants) in separate tmux
sessions. Each grabs GPU memory lazily; the CUDA driver time-slices
between their kernel launches.

Three launchers ship with the repo at
`version_2/tools/launchers/`:

| script | purpose |
|---|---|
| `run_h1h_sweep.sh` | Sequential T=14 → T=28 → T=42/56/84 with the T=28 hard gate. |
| `run_horizon.sh <T_days>` | Single-horizon launcher, parameterised by T. Designed to be run multiple times in parallel under different tmux sessions. |
| `run_t42_only.sh` | Convenience wrapper for the T=42 single run. |

Pattern for running 3 horizons in parallel from a clean GPU:

```bash
tmux new -s t42 -d "$REPO/version_2/tools/launchers/run_horizon.sh 42"
tmux new -s t56 -d "$REPO/version_2/tools/launchers/run_horizon.sh 56"
tmux new -s t84 -d "$REPO/version_2/tools/launchers/run_horizon.sh 84"
tmux ls
```

Each tmux session writes its own log under `~/bench_logs/g4_t<T>_h1h_<stamp>.log`
and its own checkpoint under
`outputs/fsa_high_res/g4_runs/T<T>d_replanK2_h60min_no_infoaware/`.

**Memory budget**: at current settings (n_smc=1024, n_pf=800), each
process peaks at ~8 GB. Three processes ≈ 24 GB of 32 GB total, leaving
~8 GB compositor headroom (verified empirically). Per-process cap is
set to `XLA_PYTHON_CLIENT_MEM_FRACTION=0.40` in `run_horizon.sh` so
even if one process tries to grow past 8 GB, it can't starve the
others.

**Empirical speedup result (2026-04-30 measurement)**: the
parallelism does **not** pay off for this workload.

What we expected:
- GPU utilization is already 97-99 % *inside* a kernel during a
  single-process run.
- The parallelism would fill the inter-stride idle gaps (100-200 W,
  ~30-40 % of wall) → ~1.5-2× aggregate speedup, individual runs
  ~30 % slower.

What we actually measured running T=42 + T=56 + T=84 in parallel:

| | solo | 3-way parallel |
|---|---|---|
| T=42 stride wall | 43 s | **122 s** (2.84× slower) |
| T=56/T=84 cold compile | ~110 s | **311 s** (2.83× slower) |
| GPU memory used | 8 GB | **25.7 GB / 32 GB** |
| GPU power | 340 W | 290-340 W (no aggregate gain) |
| aggregate throughput | 1.0× (1 process) | 3 / 2.84 = **1.05×** (3 processes) |

In other words, every process gets slowed by 2.8× — almost exactly
matching N processes contending for a single GPU. There is **no net
wall-clock speedup**. Total wall-to-all-3-done is identical to
sequential.

Why the prediction failed:
- **Host-side XLA compiler thread contention**: cold compile is
  CPU-bound, not GPU-bound. 3 processes hit cold compile near
  simultaneously and each one takes 2.8× longer.
- **CUDA driver kernel-launch serialization**: time-slicing of kernels
  on a single context is not as efficient as I assumed. The 5090's
  in-kernel utilization was already near-saturating, so there's no
  spare SM time to fill — only spare host time, which is what the
  XLA compile uses.

Memory was also a real concern: 25.7 GB / 32 GB total ran the
compositor uncomfortably close to OOM during the 3-way parallel.

**Bottom line**: for *this* SMC² workload on *this* GPU, run horizons
sequentially.

### A.4.1 The same applies to cross-seed runs

A natural follow-up: "fine, multi-tmux is dead for different horizons,
but surely 5 seeds at the *same* T benefit because the JAX compile
cache hits 100 % after the first one?" — that's only the *compile*
contention solved; the *kernel* contention remains.

Once all 5 processes are in steady state, each is launching the same
SMC² kernel onto the same SMs. The CUDA driver time-slices them at
roughly 1/N speed each. Same arithmetic as 3-way:

| processes | per-process slowdown | aggregate throughput |
|---|---|---|
| 1 | 1.0× (baseline) | 1.0× |
| 2 | ~1.5× | 1.33× |
| 3 | 2.84× *(measured)* | 1.05× |
| 5 | likely ~5× | ~1.0× |

Cross-seed multi-tmux therefore wastes effort. Five seeds in five tmux
sessions take the same wall-clock as five seeds run sequentially in
one session.

### A.4.2 The generalization (the actual lesson)

> **On a single GPU, multi-process parallelism only beats sequential
> when the per-process workload does not already saturate the GPU.**

Our SMC² at n_smc=1024 / n_pf=800 *does* saturate the 5090's SMs
(99 % util inside kernels). There is no in-kernel slack for extra
processes to fill — only host-side slack, which is what the XLA
compiler eats during cold compile. Adding a second process steals SM
time-slices from the first; aggregate throughput is conserved.

Multi-tmux **is** the right answer when any of these hold:
- The per-process workload is CPU-bound or has SM idle time
  (e.g. small models, sparse kernels, host-heavy orchestration).
- Processes are *asynchronous* in compile timing (started hours
  apart) so cold compiles don't pile up on the host.
- You genuinely need a few cross-seed runs and prefer sequential
  per-process wall to be slightly slower so you don't have to
  refactor; multi-tmux is then a "no code change" lazy answer.

For our workload, none of those apply.

### A.4.3 The only path to genuine parallelism on this hardware

`jax.vmap` over a leading parallel-axis (seed, model variant,
hyperparameter) is **not** the same mechanism as multi-tmux. vmap
compiles **one bigger kernel** with the parallel dimension fused in:

```
multi-tmux 5-way:  [kernel_seed0] [kernel_seed1] ... [kernel_seed4]   ← serialised on SMs
vmap 5-way:        [kernel(5×N_smc, N_pf, ...)]                       ← single launch
```

The vmap version still has 5× the work, but it runs as ONE launch
with no driver-level context switching, no host-side compile
contention, and XLA can choose tile sizes that fill SM lanes denser
than they fill at single-seed batch shape. Memory-bandwidth-limited
sections gain a real (~1.5-2×) speedup; compute-bound sections gain
nothing but lose nothing either.

For the FSA-v2 closed-loop bench, going to vmap-per-seed would
require ~1 day of refactor to make the bench's per-stride orchestration
shape-tolerant of a leading seed axis. The result would be
~30-40 min wall for 5 seeds at T=14 (vs 2.25 h sequential) — and
**this is the only path** to actual parallelism on this single GPU
for this workload.

For the FSA-v2 closed-loop bench at n_smc=1024 / n_pf=800, current
guidance is therefore:

- **Run a single horizon → sequential.** T=42 → T=56 → T=84.
- **Run cross-seed → sequential.** Or refactor to vmap-per-seed if
  you'll do it more than a few times.
- **Run anything multi-tmux → only when per-process workload is small
  enough that GPU isn't already saturated.**

**Cold-compile sharing**: `JAX_COMPILATION_CACHE_DIR` is set in
`run_horizon.sh`, so the 3 processes share a single on-disk HLO cache.
The first process to compile a given binary shape pays the ~110 s
cold cost; the others read from cache (sub-10 s). Extends naturally
to running cross-seed sweeps on the same T (each seed → its own tmux
session; the seed only affects rng_key, not binary shape, so cache
hits 100 %).

**Watch the GPU** while running 3 parallel:

```bash
nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv -l 2
```

Healthy pattern: power draws into 380-450 W (vs 340 W solo), util
sustained at 99 %, memory ~24 GB. If memory > 28 GB, kill the
weakest-priority horizon (`tmux kill-session -t t84`).

## B. JAX compilation discipline

### B.1 The BlackJAX closure problem (the dominant wall-clock cost)

`blackjax.smc.tempered.build_kernel` returns a fresh Python closure
per call:

```python
def build_kernel(logprior_fn, loglikelihood_fn, ...):
    def kernel(rng_key, state, num_mcmc_steps, ...):
        def log_weights_fn(position):
            return delta * loglikelihood_fn(position)        # closes over ll_fn
        def tempered_logposterior_fn(position):
            return logprior_fn(position) + state.tempering_param * loglikelihood_fn(position)
        ...
    return kernel
```

When the bench builds a fresh `loglikelihood_fn` per stride (as it
must, because the Schrödinger-Föllmer bridge prior changes every
window), BlackJAX returns a new `kernel` closure with new Python
identity. The downstream `smc_kernel_jit = jax.jit(kernel, ...)` cache
is keyed on Python identity — so it misses every stride. Result:
**~15 s of HLO recompile per stride**, ~50 % of total wall.

### B.2 Persistent XLA HLO disk cache helps the cold start, not the bridge

```bash
export JAX_COMPILATION_CACHE_DIR=$HOME/.jax_compilation_cache
```

This caches HLO across runs. Helps the *cold filter* compile (one-time
~110 s on the first stride) drop to ~30 s on a re-run. **Does
nothing** for the BlackJAX bridge variants because the closure
identity changes every call within a single run, so each recompile is
a fresh JAX cache miss anyway.

### B.3 `jax.tree_util.Partial`-wrapped loglikelihood

`jax.tree_util.Partial` is a pytree-stable callable: different bound
arg values with the same pytree structure (same shape/dtype) hash to
the same trace cache slot.

**Pattern**:

```python
ld = jax.tree_util.Partial(
    log_density_factory,
    grid_obs=grid_obs,            # device array, fixed shape
    fixed_init_state=fixed_init_state,
    w_start=w_start_arr,
    key0=key0_stride,
)
```

That `ld` is then passed as a runtime argument into a module-level
`@jax.jit`'d kernel. The trace cache key matches across strides;
**one compile per process** instead of one per stride.

### B.4 Module-level `@jax.jit`'d primitives only

Don't wrap an inner function whose Python identity drifts across
calls. Top-level `@jax.jit` decorations are stable; inner
`jax.jit(some_inner_thing)` is a recompile trap.

## C. Replacing BlackJAX's tempered SMC kernel

We wrote `smc2fc/core/jax_native_smc.py` (~480 lines). Key shape:

```python
@jax.jit
def _run_tempered_chain_jit(initial_state, loglikelihood_fn, logprior_fn,
                             rng_key, target_ess_frac, max_lambda_inc,
                             num_mcmc_steps, hmc_step_size, hmc_num_leapfrog):
    def cond_fn(carry):
        state, _, _ = carry
        return state.tempering_param < 1.0 - 1e-6

    def body_fn(carry):
        state, key, step_idx = carry
        next_lam = _solve_delta_for_ess(loglikelihood_fn, state.particles,
                                         target_ess_frac, max_lambda_inc,
                                         state.tempering_param)
        new_state = _tempered_step(state, loglikelihood_fn, ..., next_lam, ...)
        return (new_state, key, step_idx + 1)

    final_state, _, n_steps = lax.while_loop(cond_fn, body_fn,
                                              (initial_state, rng_key, 0))
    return final_state, n_steps
```

Key design choices:

- The whole adaptive λ chain is one `lax.while_loop`'d region. No host
  sync between iterations, no GPU↔CPU bounce.
- The adaptive ESS solver (`_solve_delta_for_ess`) is a 30-step
  bisection inlined as `lax.scan` — fully on-device.
- We **reuse** `blackjax.mcmc.hmc.build_kernel()` and
  `blackjax.smc.resampling.systematic` because those are module-level
  and not the source of recompiles. Only the tempered SMC wrapper is
  rewritten.
- `loglikelihood_fn` and `logprior_fn` are arguments, not closures.
- Equivalence test (`tests/test_jax_native_smc.py`) on a 2-D Gaussian
  target matched native vs BlackJAX posterior mean/cov to ~1e-3.

## D. Plant SDE on GPU

The plant Euler-Maruyama loop was originally in NumPy on the host.
The bench launched a JAX kernel for the filter, then the host stepped
the plant forward, then handed back to JAX — a CPU↔GPU bounce per
stride. Moving the plant into `lax.scan`:

```python
@jax.jit
def _plant_em_step(initial_state, Phi_subdaily, p_jax, sigma_diag, dt, rng_key):
    sqrt_dt = jnp.sqrt(dt)
    stride_bins = Phi_subdaily.shape[0]
    def step(carry, k):
        y, key = carry
        key, sub = jax.random.split(key)
        d_y = _drift_jax_v2(y, p_jax, Phi_subdaily[k])
        # sqrt-Itô diffusion + Jacobi/CIR boundary clipping
        ...
    (final_state, _), traj = lax.scan(step, (initial_state, rng_key),
                                       jnp.arange(stride_bins))
    return final_state, traj
```

Boundary handling that we cannot omit:
- **Jacobi B**: clip `y[0]` to `[0, 1]` after each Euler step.
- **CIR F, A**: ensure `y[1], y[2] >= 0` after each step (sqrt-CIR
  diffusion takes the absolute value before the sqrt).
- These clips are subtle: omit them and FP32 noise can drive states
  negative, the next sqrt-CIR diffusion is `nan`, and the SMC
  posterior collapses silently.

Modules that were already pure-jnp (don't refactor):
- `smc2fc/core/mass_matrix.py:estimate_mass_matrix`
- `smc2fc/core/sf_bridge.py:fit_sf_base` (returns float64 device arrays
  for `m`, `L_chol`, `L_inv`, `log_det` — use as-is)
- `smc2fc/filtering/gk_dpf_v3_lite.py:log_density`

## E. Controller mirroring the filter

The MPC controller's tempered SMC inner loop has the **same**
closure-baking problem as the filter. We mirrored the fix in
`smc2fc/control/tempered_smc_loop.py:run_tempered_smc_loop_native`.

Pitfalls discovered along the way:
- Wrap the cost function with `jax.tree_util.Partial(cost_fn)` —
  bare `PjitFunction` is **not** a pytree.
- If `prior_mean` is a 0-dim scalar, broadcast it to the full
  `theta_dim` before entering the chain:
  `prior_mean = jnp.broadcast_to(pm_arr, (spec.theta_dim,))`
- The Partial pattern requires bound args **first**, runtime input
  **last**: `def f(lp_fn, ll_fn, lam_val, u): return lp_fn(u) + lam_val * ll_fn(u)`.

Replanning at K=2 (daily) became cheap (~5 s vs 30-60 s pre-M); this
is what unlocked daily-replan-everywhere as a default.

## F. Particle batch sizing

After Stage M closed the per-stride compile cost, GPU was still ~50 %
idle inside kernels — small batch sizes weren't filling SM lanes. Two
levers:

| | pre-M | post-M (J5) |
|---|---|---|
| n_smc (outer) | 128 | **1024** (8×) |
| n_pf (inner) | 200 | **800** (4×) |

Per-stride wall went from ~31 s to ~43 s — net **6× useful work / sec**.

**Hard ceiling on n_pf**: tried 1600 first; XLA stayed in compile for
>10 min, no progress. The Triton GEMM fusion config printed in the
log:

```
Fusion: gemm_fusion_dot_general.205 = f32[1024,3,3]{2,0,1} fusion(...)
backend_config={"kind":"__triton_nested_gemm_fusion",
                "block_level_fusion_config":{"num_warps":"4",...}}
```

…showed XLA building a single fused kernel covering the full
(1024, 1600, 3) particle tensor. K=800 is the practical ceiling we've
verified empirically.

The combination — wider N + K=2 daily replan — fixed the SF-bridge
drift that caused the pre-M T=28 catastrophic failure (see §H).

## G. Time-grid choice

Original code path used h=15min (96 bins/day). We switched to **h=1h**
(24 bins/day) at all horizons. 4× fewer obs per window → 4× less
inner-loop work, no measurable accuracy loss for FSA-v2 latents. The
24-bin daily resolution is also the cadence at which the user's real
HRV/HR/RPE data is ingested.

The general principle (see `feedback_grid_resolution` memory): pick the
grid resolution the *model* needs, not the resolution some other model
in the codebase used. OU has dt=0.1h; FSA does not need it.

## H. Replan cadence

**Pre-M**: K=14 (weekly replan) at long horizons. The Schrödinger-
Föllmer bridge connects window k's posterior to window k+1's prior via
a Gaussian q0 fit on the previous window's SMC particles. With K=14,
the bridge is re-fit only every ~4 weekly replans across T=28. Bias
in q0 (rank-deficient covariance estimates from n_smc=128) compounds
across handoffs. After 4 replans the posterior is junk: id-cov 4/54
windows pre-M.

**Post-M**: K=2 (daily replan) at all horizons, made cheap by the
Stage M-controller. Refreshes the bridge handoff every day so q0 fits
never compound bias across multi-week handoffs. id-cov 54/54 windows
post-M.

The two fixes (K=2 + N=1024) are **complementary**: K=2 alone with
n_smc=128 might still drift because each q0 is rank-deficient;
N=1024 alone with K=14 might still compound bias over weeks. Together
they kill the failure mode.

## I. Diagnostic recipes (so we don't debug blind again)

### I.1 Count compile events

```bash
JAX_LOG_COMPILES=True python -u tools/bench_smc_full_mpc_fsa.py 14 \
    --step-minutes 60 2>&1 | tee log_with_compiles.log
grep -c "Compiling" log_with_compiles.log
```

Pre-M: ~80 compiles per T=14 run. Post-M: ≤5.

### I.2 GPU power / util while a sweep runs

```bash
watch -n 1 'nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

Healthy reads: 285-352 W during kernels (consumer 5090 TDP cap is
around 575 W; we don't expect to saturate it in non-batched workloads).
100-200 W between strides. <50 W means the bench is stuck in compile
or in host-side work.

### I.3 CPU-side numerical A/B

```bash
JAX_PLATFORMS=cpu python -u tools/bench_smc_full_mpc_fsa.py 3 --step-minutes 60
```

For short T=2 or T=3 runs, the CPU output should match GPU output up
to floating-point noise. Use this to check that an FP32 SDE change
hasn't introduced a numerical regression vs the old FP64 NumPy path.

### I.4 Tail the live log + GPU together

```bash
tail -F ~/bench_logs/g4_t28_h1h_*.log &
nvidia-smi -l 1
```

## J. What would transfer to a Julia + DiffEq.jl rewrite

### Architectural lessons that transfer
- Filter native (no library closures baking in dynamic data).
- Controller native (same pattern, mirrored).
- K=2 daily replan (cadence is a problem-level decision, not a
  language one).
- Schrödinger-Föllmer Path B q0 fit pattern (mathematics carries
  across).

### Numerical lessons that transfer
- FP32 SDE / FP64 reductions split.
- Boundary clipping on Jacobi B, CIR F/A.
- Mass-matrix in FP64 always.

### Process lessons that transfer
- tmux process isolation, GPU mem caps, persistent compile cache
  (Julia has a similar cache as `~/.julia/compiled`).

### Doesn't transfer (Python/JAX-specific)
- `jax.tree_util.Partial` — Julia's multiple dispatch sidesteps the
  closure-baking issue at the language level, so this trick has no
  analogue.
- `XLA_PYTHON_CLIENT_PREALLOCATE` — CUDA.jl has different memory
  management defaults; check `CUDA.memory_status()` instead.
- The BlackJAX `tempered.build_kernel` closure pattern itself.

### Cleaner in Julia
- DiffEq.jl SDE solvers are first-class GPU-accelerated; no
  scan-fusion gymnastics needed.
- Float64 path on CUDA.jl can use cuBLAS without the consumer-Blackwell
  64× penalty (or honest-Float32 if you're on consumer cards — the
  language doesn't pretend the GPU is fast at FP64).
- Compile time is per-method dispatch, not per-Python-closure. A bench
  with 27 strides should pay one compile, not 80.

### Risk in Julia
- TTFX (time-to-first-execution) — first run of any method dispatches
  through a JIT compile. Use `PrecompileTools.jl` and run a tiny T=2
  warm-up before the real T=14 run, or you'll pay tens of seconds at
  bench start.
- The BlackJAX closure problem reincarnates as **type instability** if
  you're sloppy with cost / loglikelihood return types. Use
  `@code_warntype` aggressively.
- DiffEq.jl + GPU still has rough edges around `EnsembleProblem` +
  `EnsembleGPUKernel`. Read the GPU-DiffEq.jl docs end-to-end before
  committing.

## K. Cumulative speedup table (T=14 wall, h=1h)

| stage | n_smc | n_pf | wall | speedup vs original |
|---|---|---|---|---|
| Original (h=15min, BlackJAX) | 128 | 200 | 124 min | 1.0× |
| Stage J + K + L | 128 | 200 | 50 min | 2.5× |
| + Stage M (filter native) | 128 | 200 | 25 min | 5.0× |
| + Stage N (plant on GPU) | 128 | 200 | 22 min | 5.6× |
| + Stage J5 (n_smc 1024, n_pf 800) | **1024** | **800** | 27 min | **4.6× wall, 30× useful work / sec** |

The Stage J5 wall is *worse* than Stage M+N at the smaller batch sizes
— but that's the price of the 8× / 4× particle bump. It's what bought
the T=28 redemption (ratio 0.569 → 1.60, F-viol 75 % → 0 %, id-cov
4 / 54 → 54 / 54).

## L. Open / deferred

- **MoG SF-bridge** for multi-week handoff. Single-Gaussian q0 was
  always going to be inadequate for K=14 weekly replans. K=2 daily
  replan obviates it for now. If we ever need multi-week handoffs
  again (e.g. for forecasting beyond T=84 in one shot), this is the
  next bridge upgrade.
- **Info-aware controller** (`sf_info_aware`). Orthogonal to
  performance, future feature.
- **n_pf = 1600 ceiling** investigation. XLA compile blocker
  empirically at this batch size. Likely fixable by tuning Triton
  fusion config or batch-splitting the inner PF, but we haven't done
  the work. If you need bigger inner-PF, this is where to start.
- **Full T=56 / T=84 sweep**. Pending T=42 result (the T=28 hard gate
  is satisfied; T=42 is currently the next horizon under study).

## Maintenance

When new lessons land, append a section. When a section becomes false
(e.g. n_pf=1600 unblocked, or a future driver makes FP64 fast on
consumer Blackwell), strike through the old text and add a dated
update — don't delete history, since the *why* of an old lesson is
often more valuable than the lesson itself.
