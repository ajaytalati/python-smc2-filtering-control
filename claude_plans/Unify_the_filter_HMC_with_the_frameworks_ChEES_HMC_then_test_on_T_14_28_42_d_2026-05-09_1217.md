# Unify the filter HMC with the framework's ChEES HMC, then test on T = 14, 28, 42 d

> Archived from plan mode: 2026-05-09 12:17.

## Context

Two prior runs in this session established hard evidence about the
v1.5 filter HMC.

**Evidence A — the bench's filter HMC defaults (ε=0.05, L=4, h_fd=1e-3)
sit in the 0 % accept regime**, while the standalone HMC sweep
identifies a viable operating point. Source:
[`compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv`](compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/diag_hmc_accept_sweep.csv).

| ε       | L  | h_fd  | accept % |
|---------|---:|------:|---------:|
| 0.001   | 1  | 1e-2  | 73.4     |
| 0.001   | 4  | 1e-2  | 74.2     |
| 0.005   | 1  | 1e-2  | 58.6     |
| 0.005   | 4  | 1e-2  | 36.7     |
| 0.01    | 1  | 1e-2  | 50.0     |
| **0.05**| 4  | 1e-3  | **0.0**  |

Sweet spot: ε ≈ 0.001–0.005, h_fd = 1e-2.

**Evidence B — the bench's filter HMC is a *different* code path
from the controller HMC**, and it is the broken one.

- Filter: [`version_1_5_Julia/models/fsa_high_res/gpu_pf.jl::parallel_hmc_one_move!`](version_1_5_Julia/models/fsa_high_res/gpu_pf.jl#L635) — model-specific, FIXED leapfrog count (no ChEES), no β-tempering inside `tempered_grads` (the closure at lines 657-662 returns the full-posterior log-density / gradient regardless of the level).
- Controller: [`julia/SMC2FC_functional/src/Control/GPUControlSMC.jl::parallel_hmc_one_move_generic!`](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl#L102) — generic, β-tempering built in (line 127: `vals_prior + beta * vals_data`), used together with [`chees_pick_L_generic`](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl#L167) for ChEES-adaptive L.

The user's call: **the filter should use the same HMC code as the
controller**. Doing this fixes both the missing-λ bug and the
fixed-L design choice in one shot, removes a redundant model-specific
code path, and makes filter and controller hyperparameters truly
analogous.

## What changes

### Change 1 — expose `--h-fd` as a CLI flag (don't hardcode)

[`bench_smc_full_mpc_fsa_gpu.jl:47-48,593`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L47):

- `--hmc-step-size` and `--hmc-leapfrog` are already CLI flags. ✓ No
  source default change; the sweep launcher passes the desired
  values explicitly.
- Add `--h-fd` (Float64, default `1e-2`) — currently hardcoded at
  line 593 inside `filter_cfg = (... h_fd = 1e-3, ...)`. Replace with
  `h_fd = args["h-fd"]` and add the entry to the `defaults` dict.
- Optionally add `--filter-chees-max` and `--filter-chees-min` (Int,
  defaults 64 and 4) so the ChEES candidate list on the filter side
  is CLI-controllable, mirroring the existing `--ctrl-chees-max` for
  the controller.

The sweep launcher passes these explicitly so the run record is
self-documenting.

### Change 2 — switch the filter HMC to the framework's generic ChEES path

[`bench_smc_full_mpc_fsa_gpu.jl:370-373`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L370)
inside `run_outer_smc`'s tempering loop. Today:

```julia
n_acc_total += parallel_hmc_one_move!(U_curr, target, grid_obs,
                                        cfg.hmc_step, cfg.hmc_leap,
                                        prior_means, prior_sigmas, sub_key;
                                        h_fd = cfg.h_fd)
```

After the change:

```julia
# Wrap gpu_log_density into the Matrix{Float64} -> Vector{Float64}
# signature the framework's generic HMC expects. The bench's
# `target` (FSAGPUTarget) holds preallocated GPU buffers; build a
# closure that calls gpu_log_density with the bench's RNG key.
log_density_fn = U_in -> gpu_log_density(target, U_in, grid_obs,
                                            hash((sub_key, :ll)))

# ChEES picks L on a small subset (matches the controller pattern in
# Control/GPUControlSMC.jl:301-308).
L_used, _ = chees_pick_L_generic(U_curr[1:min(8, n_smc), :],
                                    log_density_fn, target.M_max,
                                    cfg.hmc_step, cfg.chees_L_candidates,
                                    prior_means, prior_sigmas, rng;
                                    beta = next_λ, h_fd = cfg.h_fd)

# Each HMC move uses the same generic kernel the controller uses.
n_acc_total += parallel_hmc_one_move_generic!(U_curr, log_density_fn,
                                                target.M_max,
                                                cfg.hmc_step, L_used,
                                                prior_means, prior_sigmas, rng;
                                                beta = next_λ,
                                                h_fd = cfg.h_fd)
```

Notes:
- `target.M_max` is the existing buffer size (set at filter-target
  build time to `n_smc · (1 + 2·d) = 512 · 21 = 10752`). The
  framework's `gpu_grads_parallel_chains_fd` checks
  `M · (1+2d) ≤ M_max`, which holds.
- `beta = next_λ` is the natural fix for the missing-λ bug — the
  generic HMC bakes it into the gradient at line 127-128 of
  `Control/GPUControlSMC.jl`.
- The model-specific `parallel_hmc_one_move!` and
  `parallel_hmc_one_move` in `gpu_pf.jl` become unused. Leave them
  in place for now (a separate cleanup) — the bench just stops
  calling them.

[`bench_smc_full_mpc_fsa_gpu.jl:217-225`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L217)
imports — add `parallel_hmc_one_move_generic!`, `chees_pick_L_generic`
to the existing `using SMC2FC_functional: ...` block.

[`bench_smc_full_mpc_fsa_gpu.jl:580-600`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L580)
`filter_cfg` — add `chees_L_candidates` (built from
`--filter-chees-min`/`--filter-chees-max` like the controller's at
[lines 532-535](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L532)),
and `h_fd = args["h-fd"]`.

The bench's [`extract_xhat`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L479)
needs the `target.bufs.mu_per_chain` populated by the LAST
`gpu_log_density` call. With the unified HMC, the last call is from
`gpu_grads_parallel_chains_fd` on the FD-batch (size
M·(1+2d), not M). The bench already handles this:
[lines 380-384](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L380)
do an explicit `_ = gpu_log_density(target, U, grid_obs, hash((key,
:final)))` after the tempering loop to refresh `mu_per_chain` to the
M=n_smc posterior particles. This call still works. ✓

## Test sweep — T ∈ {14, 28, 42} d, reduced controller, fixes applied

```bash
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days <T> --seed 42 \
    --num-mcmc 3 \
    --hmc-step-size 0.005 \
    --hmc-leapfrog 4 \
    --h-fd 0.01 \
    --filter-chees-max 64 \
    --liu-west-a 0.97 \
    --smooth-resample-bw 1.0 \
    --gaussian-bridge true \
    --N-smc 512 --K-per-chain 1000 \
    --ctrl-n-smc 1024 --ctrl-num-mcmc 8 \
    --collect-ctrl-diagnostics true \
    --output-dir <SWEEP_ROOT>/T<T>d_seed42
```

Notes:
- `--hmc-leapfrog 4` is the ChEES-picker initial fallback; the
  picker overrides it from the candidate list `[4, 8, 16, 32, 64]`
  per tempering level. The CLI value matters only when the
  candidate list is overridden externally.
- All filter-side rejuvenation flags (LW 0.97, smooth-resample 1.0,
  Gaussian bridge ON) match the no-filter-HMC baseline so the
  comparison is apples-to-apples on everything except the HMC.
- Controller config (`--ctrl-n-smc 1024 --ctrl-num-mcmc 8`,
  reduced from saturated 2048/16) matches the no-filter-HMC
  baseline.

**Sweep root** (sibling of the no-filter-HMC sweep so nothing is
overwritten):
`compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_filter_hmc_unified/`.

## Steps

1. **Apply Change 1** to
   [`bench_smc_full_mpc_fsa_gpu.jl`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl)
   — add `--h-fd`, `--filter-chees-max`, `--filter-chees-min` flags
   to `_parse_args` defaults; replace hardcoded `h_fd = 1e-3` in
   `filter_cfg` with `args["h-fd"]`; build `chees_L_candidates` from
   the min/max pair (mirrors the controller at lines 532-535).
2. **Apply Change 2** — extend the bench's `using SMC2FC_functional:
   ...` import; replace the call site at lines 370-373 of
   `run_outer_smc` with the `chees_pick_L_generic` +
   `parallel_hmc_one_move_generic!` pair, passing
   `beta = next_λ` and `h_fd = cfg.h_fd`. Update the manifest writer
   to record the new flags.
3. **Sanity test** (≈ 30 s wall) — quick T = 2 d small-config run
   to confirm the unified path compiles and the bench log shows
   non-zero per-stride accept rates:
   ```bash
   julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
       --T-days 2 --seed 42 \
       --num-mcmc 3 --hmc-step-size 0.005 --h-fd 0.01 \
       --filter-chees-max 64 \
       --smooth-resample-bw 1.0 \
       --N-smc 64 --K-per-chain 200 \
       --ctrl-n-smc 256 --ctrl-num-mcmc 4 \
       --output-dir /tmp/v15_filter_hmc_unified_T2d
   ```
4. **Three-horizon test sweep** (≈ 22–35 min wall, see budget). New
   launcher
   `version_1_5_Julia/tools/launchers/run_julia_horizon_sweep_filter_hmc_unified.sh`
   (sibling of `run_julia_horizon_sweep_no_filter_hmc.sh`, with the
   new flag set inlined and `T_LIST=(14 28 42)` default).
5. **Diagnostics** on the new sweep root:
   - `compare_v15_julia_vs_python/src/diag_cloud_collapse.jl --sweep-dir`
     → cloud-std-vs-time PNG.
   - `compare_v15_julia_vs_python/src/diag_controller_hmc.jl --sweep-dir`
     → controller-HMC plots + summary.
   - **NEW** small script
     `compare_v15_julia_vs_python/src/diag_filter_hmc_accept.jl`
     that parses `bench.log` for the per-tempering-level
     `accept=X%` lines and emits a CSV of per-stride filter HMC
     accept rates. (The filter HMC doesn't yet have a
     `controller_diagnostics.csv`-style structured CSV; we could add
     one in a later change but for this test the log parse is
     enough.)
6. **Numerical posterior comparison** — load
   `data.jld2::posterior_particles` for each of T=14, 28, 42 from
   both this sweep AND the no-filter-HMC sibling sweep
   (`julia_horizon_sweep_2026-05-09_no_filter_hmc/`); compute the
   posterior median per parameter at the final stride; write a CSV
   `posterior_median_compare_TX.csv` per horizon with columns
   `(param, truth, median_no_HMC, median_HMC_unified, |Δ|_no_HMC,
   |Δ|_HMC_unified)`. Tag every cell with its source.

## Critical files

- [`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl)
  — defaults at lines 39-155, imports at lines 217-225, HMC call
  site at lines 370-373, filter_cfg at lines 587-600, manifest
  writer at lines 944-950.
- [`julia/SMC2FC_functional/src/Control/GPUControlSMC.jl`](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl)
  — `parallel_hmc_one_move_generic!` (line 102),
  `chees_pick_L_generic` (line 167),
  `gpu_grads_parallel_chains_fd` (line 48). Already exported by
  `SMC2FC_functional` (lines 32-33). **No edits to the framework.**
- [`version_1_5_Julia/models/fsa_high_res/gpu_pf.jl`](version_1_5_Julia/models/fsa_high_res/gpu_pf.jl)
  — `gpu_log_density` (the callable wrapped into `log_density_fn`
  for the framework HMC) and the now-unused `parallel_hmc_one_move!`
  / `parallel_hmc_one_move` (lines 555, 635). **No edits — leave the
  unused functions in place; clean up in a later pass.**
- [`compare_v15_julia_vs_python/src/diag_cloud_collapse.jl`](compare_v15_julia_vs_python/src/diag_cloud_collapse.jl),
  [`diag_controller_hmc.jl`](compare_v15_julia_vs_python/src/diag_controller_hmc.jl)
  — already exist; reused on the new sweep root.
- [`compare_v15_julia_vs_python/src/diag_filter_hmc_accept.jl`](compare_v15_julia_vs_python/src/diag_filter_hmc_accept.jl)
  *(NEW)* — parses `bench.log` for filter HMC accept lines.

## Verification (acceptance criteria for the unification)

The test sweep is judged on three measurable signals.

1. **Filter HMC accept rate on the rolling-window run is non-zero.**
   Today: 0 % across all logged levels. After the change, the
   standalone sweep predicts ≈ 30–50 % at ε = 0.005, h_fd = 1e-2.
   **Pass:** mean accept rate across all logged tempering levels
   ≥ 20 %. **Fail:** still near 0 % — investigate before continuing.
2. **At least one parameter's posterior median is closer to truth
   than the no-filter-HMC baseline.** Numerical comparison via
   `posterior_median_compare_T<N>.csv` (Step 6). The cells are
   measurements; per-cell source-tagged.
3. **Closed-loop A trajectory and F-violation count match the
   no-filter-HMC baseline within numerical noise.** Same controller
   config; the controller shouldn't notice the filter posterior
   tightening. If it diverges, flag.

## Wall-time budget — **guess, not measurement**

The unified HMC adds, per tempering level: `chees_pick_L_generic`
(one HMC move at each of, say, 4 candidates on a small 8-chain
subset) + `num_mcmc` HMC moves. Each HMC move calls
`gpu_grads_parallel_chains_fd` once (which calls `log_density_fn`
once on a `M·(1+2d)` matrix) plus `L` leapfrog steps each with
another grad call.

Rough per-stride HMC overhead (n_smc=512, d=10, M_max=10752):
- ChEES picker: 4 candidates × 1 HMC move × ~100 ms per grad call
  ≈ 0.4 s.
- Per HMC move at L=4: 5 grad calls × ~100 ms ≈ 0.5 s.
- num_mcmc=3 moves per level × 5 levels: 7.5 s.
- Per stride total: ≈ 8 s.
- T=14, 27 strides: ≈ 220 s extra; total wall ≈ 350 s (vs no-HMC's
  133 s).

Linear-extrapolated test-sweep wall:

| T_days | n_strides | guess wall (s) | guess wall (min) |
|-------:|----------:|---------------:|-----------------:|
| 14     | 27        | ~350           | ~6               |
| 28     | 55        | ~720           | ~12              |
| 42     | 83        | ~1080          | ~18              |
| **Σ**  |           | **~2150 s**    | **~36 min**      |

These are guesses extrapolated from a single per-stride number, not
measurements. Real wall could differ ±50 % either way. Tagged as
guess.

## Will using ChEES on the filter break the tempering schedule?

**Short answer: no — not the schedule itself. The risk is wall-time,
not correctness.** Walked through:

1. **The schedule's ESS bisection at
   [bench:244-263](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L244)
   is independent of the MCMC kernel.** It computes Δλ from
   `ll_data = gpu_log_density(target, U, grid_obs, …)` — the per-θ
   inner-PF marginal log-likelihood. Whatever MCMC runs after the
   resample doesn't enter this calculation. Same input, same Δλ.
2. **The framework's HMC with `beta = next_λ` leaves π_λ invariant
   by construction.**
   [Control/GPUControlSMC.jl:120-128](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl#L120)
   computes `grads_prior + beta * grads_data`; leapfrog + MH thus
   target `log p(θ) + next_λ · log p(y|θ)`. That's exactly π_{next_λ}.
3. **`chees_pick_L_generic` works on a *copy* of an 8-chain subset**
   ([line 179-180](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl#L179)
   — `U_try = copy(U_subset)`) — read-only on the production cloud.
4. **ChEES uses a fresh RNG**
   ([line 307](julia/SMC2FC_functional/src/Control/GPUControlSMC.jl#L307)
   — `MersenneTwister(rand(rng, UInt32))`), so its draws don't shift
   the main RNG state.

**Wall-time risks (real, not correctness):**

- ChEES adds ~one extra HMC move per candidate L per tempering
  level. With 4–5 candidates plus `num_mcmc=3` production moves,
  per-level HMC calls go from 3 to ~7–8. Inner-PF cost dominates
  each call (M·(1+2d) ≈ 10 752 chains). Per-stride HMC wall could
  roughly double vs the previous plan's guess.
- ChEES may pick very different `L`s at different levels. Schedule
  still correct; per-level wall just becomes uneven.

**Mitigation if wall is a problem:** keep the candidate list
short (`[4, 8, 16]` instead of `[4, 8, 16, 32, 64]`) — exposed as
`--filter-chees-max 16` in the test launcher.

## What this plan does NOT do

- Does not delete the now-unused
  `parallel_hmc_one_move!` / `parallel_hmc_one_move` in
  `gpu_pf.jl`. A separate cleanup pass.
- Does not edit the framework
  (`SMC2FC_functional/src/Control/GPUControlSMC.jl`). Filter and
  controller now both call the same un-modified framework code.
- Does not run T = 56, 70, 84. If the test sweep shows the unification
  works, the user can authorise the longer-horizon sweep separately.
- Does not extend `--collect-ctrl-diagnostics` to the filter HMC.
  For this test sweep we read filter accept rates from `bench.log`
  via the new `diag_filter_hmc_accept.jl` script. Adding a
  structured `filter_diagnostics.csv` is a follow-up.
