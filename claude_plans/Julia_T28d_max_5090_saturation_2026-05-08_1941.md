# Julia T=28d max-saturation profiling and configuration plan

> Archived from plan mode: 2026-05-08 19:41.

## Context

Today's perf work (commits `da847d9`, `099a14a`, `4a5c007`) collapsed the
v1.5 Julia GPU filter cost from **81 ms / call → 0.9 ms / call**, taking
Julia from 17× slower to ~5× **faster** than Python+JAX on the headline
microbench. The closed-loop bench has only been re-measured at the
*toy* config (T=2d, N=16, K=100) where wall time dropped 13×. The full
T=28d closed-loop has not been re-run yet, and Julia is running at the
config that was originally chosen to be tractable for the *slow* code
path. With ~5× compute headroom and ~30 GB free VRAM on the RTX 5090,
the right next move is:

1. measure where the remaining ms goes,
2. find the (N, K, T) point that saturates the GPU,
3. re-run the long-horizon closed-loop comparison at that point.

This is item 2 of §8 in
`compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf`,
now reframed because items 1 (framework migration) and the per-call
perf bottleneck have already been resolved.

## Goal in one sentence

**Find the largest (N_smc, K_pf) config the RTX 5090 can run at
T=28d in under 30 minutes wall time on Julia, and compare its closed-
loop posteriors and control behaviour against Python+JAX at the same
horizon.**

## Scope

**In scope:**
- Profiling Julia under `nsys profile --trace=cuda` to attribute the
  current 0.9 ms / call to specific kernels.
- Microbench sweep over (N_smc, K_pf) on Julia until either the
  compute saturates (effective TFLOPS ≥ 50% of 104.8 peak) or VRAM
  fills (≥ 24 GB used).
- Picking the best (N, K) from the sweep and running:
  - Julia closed-loop bench at T=28d, that config, seed=42.
  - The paired launcher to also run Python+JAX at T=28d at its current
    tuned config.
  - The cross-stack comparison script.
- Recording the new headline numbers in `compare_v15_julia_vs_python`.

**Out of scope:**
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

## Approach (5 phases)

### Phase A — Re-measure the baseline at the new fast path

**Why:** The writeup's §7 numbers were captured before today's fixes
and are stale. We need a clean baseline at the *current* default
config (N=32, K=200) at T=28d to compare against any saturation-tier
result.

**Action:**
```bash
bash version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh \
     28 42
```

This runs both Julia and Python+JAX at T=28d, seed=42, matched config
(or the launcher's current matched-defaults — verify before kicking
off). Captures `julia_profile.log`, `python_profile.log`,
`gpu_telemetry.csv` per stack, `data.jld2` / `trajectory.npz`,
per-stride CSVs.

Then:
```bash
JAX_ENABLE_X64=True PYTHONPATH=.:.. \
  python version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py \
  outputs/v15_julia_vs_python_T28d_seed42
```

**Pass criteria for Phase A:**
- Both stacks complete without errors.
- Julia wall time at T=28d < 30 min (was 37.2 min at T=7d before
  today's fixes; expect ~10× reduction = ~6 min at T=7d, ~25 min at
  T=28d).
- The `summary.txt` table lands; the 6-panel `comparison.png` renders.

### Phase B — `nsys` profile of one Julia bench window

**Why:** Find what's left after the easy wins. We don't yet know if
the remaining 0.9 ms / call is dominated by kernel launches, memory
bandwidth, or per-call host-side bookkeeping.

**Action:** Run a short `nsys` capture (1–2 strides is plenty):

```bash
cd version_1_5_Julia
nsys profile --trace=cuda --output=/tmp/v15_julia_T28d_nsys \
     julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
     --T-days 2 --replan-K 1 --N-smc 32 --K-per-chain 200 \
     --num-mcmc 2 --max-temp-levels 8 \
     --output-dir /tmp/v15_T28d_nsys_run
```

Open the resulting `.qdrep` in NVIDIA Nsight Systems. Note the top 5
kernels by cumulative time, the host gap percentage, and any
synchronisation hot-spots.

**Pass criteria for Phase B:**
- A short text note (~½ page) in `claude_plans/` summarising:
  - Top 5 kernels by cumulative time in one `gpu_log_density` call.
  - GPU active vs idle ratio.
  - Memory-bandwidth utilisation if Nsight surfaces it.
  - One concrete recommendation for the saturation-tier config in
    Phase C (e.g. "K=400 is memory-bound; bumping N is cheaper").

### Phase C — Saturation sweep on the Julia microbench

**Why:** Find the (N, K) corner of the configuration space that maxes
out either the 104.8 TFLOPS fp32 peak or the 32 GB VRAM. The
microbench reports both, so this is mechanical.

**Action:** Add a `--sweep` mode to `tools/profile_gpu.jl` (or a small
new script `tools/profile_gpu_sweep.jl`) that runs the existing
filter-PF and controller-cost microbenches over a grid:

| N_smc       | K_per_chain  |
|-------------|--------------|
| 32 (current) | 200 (current) |
| 64           | 400           |
| 128          | 400           |
| 256          | 400           |
| 256          | 800           |
| 512          | 800           |

For each cell, record:
- Median time per call (ms).
- Effective TFLOPS / 104.8 peak %.
- VRAM after target-build (GB) and after timing (GB).
- Threads in flight.

Plot or tabulate the sweep. Pick the largest (N, K) where:
- VRAM after timing ≤ 24 GB (leaving 8 GB headroom).
- Effective TFLOPS ≥ 50% of peak OR memory-bandwidth-bound (whichever
  is the actual ceiling).
- Median time per call × ~5000 (rough closed-loop call count for
  T=28d) ≤ 25 minutes wall-clock.

**Pass criteria for Phase C:**
- A markdown table in `claude_plans/` with the sweep results.
- One picked config (N\*, K\*) marked.
- Argued: why this point is the saturation tier, not just a faster one.

### Phase D — Closed-loop bench at the picked config, T=28d

**Why:** Confirm the microbench saturation generalises to the closed
loop. Compare against the Phase A baseline.

**Action:**
```bash
cd version_1_5_Julia
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
     --T-days 28 --replan-K 2 \
     --N-smc <N*> --K-per-chain <K*> \
     --num-mcmc 3 --max-temp-levels 30 \
     --output-dir outputs/v15_julia_T28d_seed42_saturated
```

**Pass criteria for Phase D:**
- Total wall time on Julia at saturation ≤ 30 min.
- Mean GPU utilisation ≥ 60% (was 24.8% before today's fixes).
- VRAM during run ≥ 50% of 32 GB (≈ 16+ GB used).
- Posterior MSE to truth (final window) ≤ 0.05 (was 0.0868 at the
  small config; we expect more particles → tighter posterior).
- F-violation rate = 0 (correctness, not perf).

### Phase E — Cross-stack comparison at T=28d

**Why:** This is the actual scientific question. Does Julia at its
saturated config produce control behaviour that matches Python+JAX's
tuned config?

**Action:**
- Run the paired launcher with the new flags so the Julia leg uses
  N\*, K\* and Python+JAX uses its existing tuned config.
- Run the comparison script.
- Snapshot the `summary.txt` table and `comparison.png` 6-panel plot
  into `compare_v15_julia_vs_python/example_run/T28d_seed42_saturated/`.

**Pass criteria for Phase E:**
- 6-panel diagnostic plot renders cleanly.
- `summary.txt` table populated.
- Mean A on Julia is within ± 5% of Python+JAX's mean A at T=28d.
  (At T=7d, both lost ~13% to baseline; at T=28d, v2's reference
  shows +60%. We expect both stacks to either both ramp up or both
  pick rest, not split.)
- Posterior medians within IQR overlap on at least 8 of 10
  parameters.

## Critical files

- `version_1_5_Julia/tools/profile_gpu.jl` — extend with `--sweep`
  mode (or write `profile_gpu_sweep.jl` next to it).
- `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` — already
  has `--ot-max-weight`, `--liu-west-a`, `--N-smc`, `--K-per-chain`,
  `--num-mcmc`, `--max-temp-levels` CLI flags.
- `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh` —
  the paired launcher; confirm it forwards N/K to Julia.
- `version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py` —
  the cross-stack diff. Confirm it reads from `outputs/v15_julia_vs_python_T28d_seed42`.
- `compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.tex`
  — the writeup. Will need a §7 update with the new T=28d numbers
  after Phase E.

## Existing utilities to reuse (do NOT re-implement)

- `FSAGPUTarget` and the segmented PF kernels live in
  `version_1_5_Julia/models/fsa_high_res/gpu_pf.jl`. The constructor
  takes `M_max` directly; bumping N just means passing a larger N to
  the bench's CLI.
- `gpu_log_density`, `gpu_grads`, `parallel_hmc_one_move!` are the
  public surface — model-side hot path, do not edit.
- `run_outer_smc` in the bench file is the hand-rolled tempered SMC²
  loop. Do not reach into `SMC2FC_functional.run_tempered_smc_gpu` for
  the filter side; keep that for the controller side.
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
