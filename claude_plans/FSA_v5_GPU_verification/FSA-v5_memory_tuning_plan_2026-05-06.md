# Memory tuning plan for Stage 3 at full particle counts (FSA-v5)

**Goal:** run Stage 3 closed-loop SMC²-MPC at the same particle counts
that succeed in Stage 2 (`n_smc=256`) plus the full filter particle
count (`n_pf=800`) and full HMC kernel (`mcmc=10, leapfrog=16`),
without OOM. The previous overnight launcher backed off to
`n_smc=128, n_pf=200` because full particles OOM'd at ~30.85 GB on the
32 GB device — that backed-off config produced the bad run-19
trajectory you flagged.

**Hardware:** RTX 5090, 32607 MiB total VRAM. Currently 28244 MiB free
(verified just now via `nvidia-smi`).

## What I will do, in order

### 1. Baseline measurement (no GPU yet)

Set up monitoring and inspect prior runs' reported VRAM peaks:

- Confirm `XLA_PYTHON_CLIENT_PREALLOCATE=false` (already in benches at
  the env-var preamble).
- Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`. Reserves ~5 GB for the
  Wayland compositor (per `version_3/GPU_TUNING_RTX5090.md` §A.2 — the
  driver freezes the desktop if JAX grabs everything).
- In a side `tmux` window: `nvidia-smi --loop=1 --query-gpu=memory.used,utilization.gpu --format=csv`
  while runs proceed. So I can see the actual VRAM peak as JIT compiles
  and then as the run cycles strides.

### 2. Smoke test at T=2 d, target config

Goal: confirm the target config fits *before* paying for the full T=14d
wall.

```
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v5.py \
    --cost soft --scenario healthy --T-days 2 --replan-K 2 \
    --n-smc 256 --n-pf 800 --n-inner 64 \
    --run-tag stage3_smoke_healthy_T2_full_particles
```

T=2d → 4 strides → 1 replan. JIT compiles for the worst-case shape
once; subsequent strides reuse the compiled graphs. So peak VRAM at
T=2d is a tight upper bound on T=14d peak (memory doesn't scale with T
at fixed particle counts; only the trajectory tensor itself grows
linearly — small).

**Pass:** T=2d completes without OOM, peak VRAM ≤ ~28 GB.

**Fail modes:**

| Symptom | Probable cause | Next step |
|---|---|---|
| OOM during first JIT compile | Compile-time tensor grew too large | Reduce `n_pf` first (filter particle counts dominate) |
| OOM mid-run (after stride 1) | Bridge proposal Jacobian | Reduce `n_smc` instead |
| Runs but VRAM at 30+ GB | Compositor in danger zone | Reduce `MEM_FRACTION` further or drop one of the counts |

### 3. Backoff schedule (only if step 2 fails)

I'd ratchet down in this order (keeping HMC kernel intact per your "full
HMC only" rule):

| Step | n_smc | n_pf | n_inner | Notes |
|---|---|---|---|---|
| 0 (target) | 256 | 800 | 64 | If this works, lock in. |
| 1 | 256 | 600 | 64 | -25% filter particles |
| 2 | 256 | 400 | 64 | -50% filter |
| 3 | 256 | 400 | 32 | drop n_inner halfway |
| 4 | 192 | 400 | 32 | -25% controller |
| 5 | 192 | 200 | 32 | matches old run-19 except n_smc |

I'll stop at the first config that fits and lock it in. Each step takes
~1-2 min wall (just T=2d smoke). I'll log each attempt as a row in
`experiments_log.md` with the VRAM peak and pass/fail.

### 4. Why this order

Memory cost rough order (Stage 3, per-stride, fp32 SDE, fp64 log-domain):

- **Filter inner**: `n_pf × WINDOW_BINS × n_states × 4 bytes` plus
  bridge gradients. Dominant for Stage 3.
- **Controller HMC**: `n_smc × n_inner × leapfrog × theta_dim × 8 bytes`
  plus per-particle SDE rollout `n_smc × plan_n_steps × n_states × 4`.
  Second-largest.
- **Plant trajectory**: `n_bins × n_states × 4` — small.

`n_pf` is the most expensive lever per unit memory and the easiest to
back off without breaking any contract — the SMC² filter posterior
quality degrades smoothly with `n_pf`, no thresholds. `n_smc` matters
more for controller exploration so I'd preserve it longer.

### 5. What I will NOT do (per your rules)

- **No HMC trimming.** `mcmc=10, leapfrog=16` stays. Per your 2026-05-06
  rule this is the "only validated config."
- **No `soft_fast`.** Cost variant is `soft` (full HMC inside).
- **No smc2fc framework edits.** If memory pressure is in
  framework-side bridge code, escalate.

### 6. Decision rule

**Success = T=2d smoke at target config completes, peak VRAM ≤ 28 GB.**
Then I run T=14d at that config, then expand to other scenarios.

**Failure = even step 5 in the backoff table OOMs.** Then I stop and
escalate — that would suggest a memory leak or wasteful tensor
allocation that needs framework investigation, which is outside my
edit scope.

## Run plan once memory-tuning succeeds

Numbering starts fresh at run 1 (the existing 19 prior runs sit in
`outputs/fsa_v5/experiments/old_experiments/`; the atomic
`_run_dir.py` allocator only counts the live `experiments/` subdir, so
new runs auto-number from 01).

| Run | Stage | Scenario | T_days | Cost | Notes |
|---|---|---|---|---|---|
| 01 | 3 | healthy | 14 | soft | Smoke + first full-particle Stage 3 |
| 02 | 3 | sedentary | 14 | soft | |
| 03 | 3 | overtrained | 14 | soft | **Direct comparison vs run-19** (the bad-trajectory run) |
| 04 | 3 | healthy | 42 | soft | Banister-overload long-horizon |
| 05 | 3 | overtrained | 42 | soft | If 03 trajectory looks clean |

Each completed run writes a row to
`outputs/fsa_v5/experiments_log.md` with measured wall-clock, VRAM
peak, post-hoc weighted_violation_rate (now Option-A correct),
mean_A_integral, and Pass/Fail vs the bench's gates.

I'll pause for your input after run 03 to compare its basin overlay
against run-19's directly — that's the question you wanted answered.

## What I need from you before starting

1. **Confirm the run plan above** (especially: do you want me to run
   05 right after 04, or wait for your sign-off on 04 first?).
2. **GPU exclusivity**: should I assume nothing else is using the
   GPU? If you're working in VS Code or running anything else, I need
   to either back off MEM_FRACTION or wait.
3. **Wall-clock budget**: at full particles, run 01 might take 2–4 h
   based on the v2 baseline. Are you comfortable with that, or do you
   want a max-time cap?
