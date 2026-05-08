# RTX 5090 saturation sweep — N × K grid at Python-matched bench config + closed-loop accuracy at the saturated cell

> Archived from plan mode: 2026-05-07 23:56.

> Plan only. Read-only on existing files; the only file edits are scratch sweep artefacts (sweep script + nvidia-smi CSVs + final report + closed-loop comparison) under `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/`.

> **Two phases.** Phase 1 finds the (N, K) cell that saturates the GPU best. Phase 2 then runs the closed-loop bench at that cell and tests how accurately the controller's schedule matches Python's reference. The user's stated objective is closed-loop accuracy at the highest-saturation operating point, so Phase 2 is the actual deliverable; Phase 1 is the prerequisite that picks the operating point.

# Phase 1 — saturation sweep (N × K grid)

## Context

**Why this sweep.** The user wants empirical guidance on what `--N-smc` × `--K-per-chain` setting maximises both GPU core utilisation and memory occupancy on the RTX 5090. The Python equivalent of this codebase saturates the GPU; the Julia bench currently does not. §2.14 of [version_2_Julia/docs/julia_fsa_writeup.pdf](version_2_Julia/docs/julia_fsa_writeup.pdf) reports that at `n_smc=1024` the bench had not finished one stride after ~140 s of warmup with util ≈ 51% and only ~7 GB / 32 GB used. §2.12 traces this to per-chain host loops in `run_outer_smc_gpu`, `gpu_log_density_batched`, `run_segmented_smc_step!`, and `parallel_hmc_one_move!` that scale with `n_smc` despite the work being batchable on-device. So the operating-point question is: where does util plateau (= host-loop ceiling), and how much memory does that leave on the table?

**Sweep target.** A 4 × 3 grid:

- `--N-smc` ∈ {32, 64, 128, 256}
- `--K-per-chain` ∈ {400, 800, 1600}

Twelve cells. For each cell collect: median GPU util %, peak memory used, median per-stride wall (from bench's own log).

**Fixed flags (Python-matched config from §5.3 of the writeup):**

```
--step-minutes 60   --replan-K 2   --T-days 14   --seed 42
```

**Time budget per cell.** ~5–10 min wall. Long enough for Julia JIT (~60–90 s) plus 4–5 strides. Hard cap 8 min — cells that can't complete 4 strides in that budget contribute the equally-useful data point "saturated to the point of impractical wall time".

## Approach

### Why bench, not `tools/profile_gpu.jl`

The user pointed at `tools/profile_gpu.jl` as methodology inspiration. Reading that file:

- It is a microbenchmark of one kernel call. Hardcoded `BINS_PER_DAY=96` (15-min step, not the user's 60-min), hardcoded `N_smc=32`, `K_per_chain=400` ([profile_gpu.jl:70-73](version_2_Julia/tools/profile_gpu.jl#L70-L73)).
- "Per-stride wall" is not a profiler concept — the profiler only times one kernel call.
- The full pipeline's host-loop overhead (§2.14's diagnosis) does not appear in microbenchmarks.

The bench `tools/bench_smc_full_mpc_fsa_gpu.jl` already exposes the four flags I need (`--N-smc`, `--K-per-chain`, `--step-minutes`, `--replan-K`), produces per-stride wall in its `@info` log lines, and stresses exactly the orchestration the user is trying to optimise. So: run the bench at each cell, sample `nvidia-smi` in parallel (using the same query pattern as [profile_gpu.jl:51](version_2_Julia/tools/profile_gpu.jl#L51) — `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`), kill after 4–5 strides, parse metrics.

### Per-cell execution recipe

For each `(N, K)` cell:

1. **Output dir**: `outputs/fsa_high_res/gpu_saturation_sweep/<sweep_timestamp>/N<N>_K<K>/`. Bench artefacts (`data.jld2`, plots) go in the bench's own `--output-dir`; sweep artefacts (`gpu_log.csv`, `bench.log`) go alongside.

2. **Launch `nvidia-smi` poller in background**, sampling every 2 s, writing `wall_seconds, util%, mem_MB` to `gpu_log.csv`. Sample lines: `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits`.

3. **Launch bench in background**, redirecting stderr to `bench.log`. Command:
   ```bash
   julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
       --step-minutes 60 --replan-K 2 --T-days 14 \
       --N-smc <N> --K-per-chain <K> --seed 42 \
       --output-dir <out_dir>
   ```

4. **Watch `bench.log`** (tail / grep loop). The bench logs `[stride NN/NN] X.Xs N filter levels Φ=Y.YYY x̂=(...)` per stride ([bench_smc_full_mpc_fsa_gpu.jl:691](version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L691)). Count `[stride ` matches.

5. **Kill conditions** (whichever fires first):
   - 5 stride lines have appeared → kill bench + poller.
   - 8 min hard timeout → kill, mark cell as `did_not_complete_4_strides`.

6. **Parse and record**:
   - **Per-stride wall**: regex `\[stride\s+\d+/\d+\]\s+([\d.]+)s` → median of stride times excluding stride 1 (which contains JIT for the inner-PF kernel — the bench's first stride is bigger because it advances `WINDOW_BINS` instead of `STRIDE_BINS`).
   - **Median GPU util %**: from `gpu_log.csv`, drop the first 60 s (Julia JIT phase, GPU idle), median over the rest.
   - **Peak memory MB**: max over the entire `gpu_log.csv` (warmup memory is also a real allocation, worth reporting).
   - **Strides completed**: count of stride log lines.

7. **Append one row** to `RESULTS.csv`.

### Sweep script

One bash script `run_sweep.sh` under the sweep output dir. Runs cells **sequentially** (NOT in parallel — KernelAbstractions JIT cache is per-process, parallel cells would each pay the JIT cost and contend for GPU memory). Sweep order: small cells first (N=32, all K), then climb. That way if a later cell saturates the budget without producing 4 strides, the earlier cells have already given clean data.

```bash
# pseudo-structure
SWEEP_DIR=outputs/fsa_high_res/gpu_saturation_sweep/$(date +%Y-%m-%d_%H%M)
mkdir -p $SWEEP_DIR
for N in 32 64 128 256; do
  for K in 400 800 1600; do
    CELL_DIR=$SWEEP_DIR/N${N}_K${K}
    mkdir -p $CELL_DIR
    # 1. start nvidia-smi poller (background)
    # 2. start bench (background, writing to $CELL_DIR/bench.log)
    # 3. poll bench.log for stride count or 8min timeout
    # 4. kill both
    # 5. parse metrics, append row to RESULTS.csv
  done
done
# 6. emit RESULTS.md from RESULTS.csv
```

### Total time

12 cells × up to 8 min = ~96 min worst case. Probably faster — N=32 cells finish in ~3–5 min, only the upper-right cells (N=128, 256 × K=800, 1600) will hit the 8-min timeout.

## Expected results pattern

Predictions worth noting up front so the data has something to falsify:

- **N=32 row** — util plateaus low (under-occupied even for the kernel), memory minimal (~5 GB), per-stride wall short (~10–20 s). This is the existing default.
- **Climbing N at K=400** — util rises; per-stride wall rises roughly linearly with N (§2.14's observation). Saturation wall in the §2.14 ballpark of ~51 % util when host-loops dominate.
- **Climbing K at fixed N** — util should rise WITHOUT proportional wall blow-up, because §2.14 says K widens per-launch threads without amplifying the per-chain host loops. So `(N=64, K=1600)` might give better util than `(N=128, K=400)` at similar wall.
- **Memory** — climbs ~linearly with `M_max × K_per_chain`. M_max = N × (1 + 2d) = 61 N filter chains. Particle cloud is `M_max × K × ~16 bytes` = ~1 GB at (N=32, K=400), ~16 GB at (N=128, K=1600). Peak memory should saturate around (N=128, K=1600) or (N=256, K=800).

If both util AND memory saturate at the same cell, that's the operating point. If util plateaus before memory does, §2.12's host-loop ceiling is real and the recommendation is "biggest K you can afford within wall budget, modest N". If memory caps before util does, the answer is "biggest N you can afford".

## Deliverable

Single markdown report at `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/RESULTS.md` containing:

1. **One-line summary**: which cell to use, why.
2. **The 4 × 3 table**:

   | N | K | M_max | util_median % | util_max % | peak_mem_GB | per_stride_wall_s | strides | status |
   |---|---|---|---|---|---|---|---|---|
   | 32 | 400 | 1952 | … | … | … | … | … | OK |
   | 32 | 800 | 1952 | … | … | … | … | … | OK |
   | … | … | … | … | … | … | … | … | … |

3. **Saturation diagnosis**: where util plateaus, where memory plateaus, evidence for/against §2.12's host-loop ceiling.
4. **Recommended flag combination** for `--N-smc` and `--K-per-chain`, with the trade-off (e.g. "N=64 K=1600 saturates util at ~75% and uses 18 GB / 32 GB; per-stride wall is X s for a 14-day bench wall of ~Y min, vs. N=32 K=400 default at ~30% util / 5 GB / Z min").
5. **What's left on the table**: if the headline util number is still well below 90%, point at §2.12's host-loop fix as the next move.

Raw artefacts in the same directory: `gpu_log_<cell>.csv` and `bench_<cell>.log` per cell, plus `RESULTS.csv`.

## Critical files (READ-ONLY)

- [version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl](version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) — bench entry point (CLI parsing :16-60, stride log :691).
- [version_2_Julia/tools/profile_gpu.jl](version_2_Julia/tools/profile_gpu.jl) — methodology reference for nvidia-smi querying (:51).
- [version_2_Julia/docs/julia_fsa_writeup.pdf](version_2_Julia/docs/julia_fsa_writeup.pdf) §2.12, §2.14, §5.3 — context, prior measurements, matched-Python config.

## Files I will create

- `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/run_sweep.sh` — sweep orchestrator (bash).
- `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/N<N>_K<K>/{bench.log, gpu_log.csv, data.jld2 (bench output)}` — per-cell raw artefacts.
- `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/RESULTS.csv` and `RESULTS.md` — sweep summary.

I will not edit `bench_smc_full_mpc_fsa_gpu.jl` or `profile_gpu.jl`.

## Verification

After all 12 cells run:

1. **Sanity row**: `(N=32, K=400)` should match the §2.14 baseline ballpark (per-stride wall ~7–10 s at h=60min K=2; util well under 50%; ~5 GB used).
2. **Monotonicity check**: per-stride wall should be monotone non-decreasing as N grows at fixed K, and as K grows at fixed N. Any violation indicates a noisy sample (one extra HMC bisection level on that stride) — cross-check by inspecting the stride log's `N filter levels` field.
3. **Memory check**: `peak_mem_GB ≈ 0.5 + 0.001 × M_max × K × 12 bytes / 1e9` (rough formula for the inner-PF particle cloud + state buffers). Wildly off → some cell hit OOM or memory growth was masked.
4. **Reproducibility spot-check**: re-run the cell I recommend in the report once more (different sweep timestamp) to confirm the numbers hold.

## Open question for the user

Default: I will run Phase 1 autonomously, then proceed straight into Phase 2 at the chosen cell, then produce a single final report covering both. If the user wants a checkpoint between phases (e.g. to inspect the saturation table before committing 30+ min to the closed-loop accuracy run), say so and I'll pause for confirmation after Phase 1.

---

# Phase 2 — does saturation help the controller MAXIMISE ∫A dt (area under the A curve)?

## Context

**The cost is `J(Φ) = -∫A(t) dt + λ_F·∫max(F-F_max, 0)² dt`. The controller's job is to maximise the integral of A over the bench horizon — i.e. the area under the A trajectory — subject to the soft F-cap.** Schedule shape (U-shape, recovery → overload, peak day, oscillation) is downstream of how well the optimiser is solving that — diagnostic, not objective.

The single number that judges the controller is therefore:

> **∫A_MPC dt**, the area under the MPC plant's A trajectory, and the relative gain `(∫A_MPC − ∫A_baseline) / ∫A_baseline` over the Φ = 1 baseline plant.

Computed numerically as `sum(traj_mpc[:, 3]) * dt_days` over the 14-day bench (units: A·days). Equivalent to `T_total_days × mean(A_MPC)` since `dt_days` is constant.

Reference points (from §2.15 of the writeup, which reports the same quantity in the equivalent "mean A" form — multiply by T_total = 14 d to get ∫A dt):

- **Python (target):** ∫A_MPC dt ≈ 1.71 A·d (= 0.122 × 14) vs ∫A_baseline dt ≈ 1.13 A·d (= 0.081 × 14) → **+51 % gain in area**.
- **Julia closed-loop (current):** ∫A_MPC dt ≈ 1.22 A·d vs ∫A_baseline dt ≈ 1.19 A·d → **+2.3 % gain**.
- **Julia open-loop ground truth (current):** ∫A_MPC dt ≈ 1.22 A·d — same area as closed-loop. Smooth recovery → ramp Φ peaking at ~0.78 by day 13.

So the closed-loop is currently delivering ~5 % of Python's area gain. §2.15 row 10/11 says this magnitude gap survived §2.9 and §2.10 fixes; the open question is whether the gap is also invariant under GPU saturation, or whether it shrinks once the filter has more particles to lock onto a tighter posterior.

Phase 2's question, in one line:

> **Does running the bench at the saturated `(N*, K*)` from Phase 1 increase ∫A_MPC dt toward Python's ≈ 1.71 A·d, or does it stay near 1.22 A·d regardless?**

If saturation moves the area up, the recommendation is "use these flags". If saturation does not move the area, the recommendation is "saturate AND fix the source-level bug §2.15 row 10/11 is tracking — flags alone won't close the gap".

## Approach

Three runs at the saturated cell `(N*, K*)` chosen from Phase 1, all with `--step-minutes 60 --replan-K 2 --T-days 14 --seed 42`:

1. **A — closed-loop @ saturated cell**: default mode (no `--open-loop`). This is the headline test — what the production bench will produce after the recommendation is adopted.
2. **B — open-loop @ saturated cell**: `--open-loop true`. Same kernel, same horizon, same seed; differs from A only by replan-vs-fixed-plan. B's Ā_MPC is the **upper bound on what the saturated controller alone can deliver** (no replan-oscillation noise muddying the cost optimisation).
3. **C — closed-loop @ default `(N=32, K=400)`**: the un-saturated baseline. C's Ā_MPC is the **lower bound under current production defaults**.

Phase 1's recommended cell is reused for A and B. C is fixed.

For each of A / B / C the bench writes `data.jld2` containing `trajectory_mpc`, `trajectory_baseline`, `Phi_per_bin_mpc`, `Phi_per_bin_baseline`, `daily_phi_per_stride`, `posterior_particles`, `param_names`, `truth_params` (per §5.7 of the writeup).

### Headline metric (the controller's job: maximise area under A)

For A / B / C:

- **∫A_MPC dt** = `sum(traj_mpc[:, 3]) * dt_days` (area under the MPC plant's A curve, A·days).
- **∫A_baseline dt** = `sum(traj_base[:, 3]) * dt_days` (area under the Φ=1 baseline plant's A curve).
- **Gain** = `(∫A_MPC − ∫A_baseline) / ∫A_baseline`, in percent.

This is exactly what the cost function `J = -∫A dt + λ_F·barrier` is optimising (modulo the soft F-cap). Python's target: gain ≈ +51 %. Current Julia: gain ≈ +2.3 %.

### Where Phase 2 lands relative to the targets

I will read the gain triple `(C_gain, A_gain, B_gain)` (closed-loop default, closed-loop saturated, open-loop saturated) and place it on the `[+2.3 %, +51 %]` interval:

- **A_gain ≫ C_gain** → saturation is helping the closed-loop optimiser deliver more area, magnitude gap shrinking. Recommend the saturated flags as the production operating point.
- **A_gain ≈ C_gain ≈ +2.3 %, B_gain also ≈ +2.3 %** → the cost surface is delivering the wrong optimum (low area) at every saturation level. Area gap is in the kernel / cost surface, not in the sampling noise. Saturation does not help; the next move is the source-level bug §2.15 row 10/11 is tracking.
- **A_gain ≈ C_gain but B_gain ≫ A_gain** → saturation can find a higher-area schedule (B), but the closed-loop replan loop is throwing it away. Area gap is the closed-loop oscillation bug §2.15 row 11. Recommend `--open-loop true` for production until that's fixed.

This is the entire diagnostic. Everything else below is supporting evidence for which of the three buckets the data lands in.

### Secondary diagnostics (HOW the controller is or isn't maximising ∫A dt)

These don't decide the recommendation; they explain it.

- **F-violation %** — fraction of bins where `traj_mpc[:, 2] > F_max = 0.40`. If a high ∫A_MPC came with F-violation ≫ 1 %, the controller cheated against the soft cap and the apparent area gain is unsafe. Python's target: ≤ 1 %.
- **Daily Φ shape** — `daily_phi_per_stride` over 14 days. Two schedules with the same ∫A dt can look very different: pinning Φ → Φ_max throughout vs. running the recovery → overload basin. The writeup expects the recovery → overload shape; if Phase 2 produces the same area via a different shape, flag it.
- **Φ peak day and value** — `argmax`, `max` of `daily_phi_per_stride`.
- **Oscillation amplitude** — std-dev of `daily_phi_per_stride − moving_average(daily_phi_per_stride, 3)` as a % of `mean(daily_phi_per_stride)`. Closed-loop A is expected to oscillate around the smooth open-loop B (§2.15 row 11). Quantifying the oscillation tells me whether the closed-loop replan loop is stable enough to lock onto B's high-area optimum or is bouncing between basins, leaking area in the process.
- **Cumulative ∫A(t) over time** — running integral, plotted alongside the baseline cumulative. Lets me see when the controller is actually accumulating area vs the baseline (e.g. all in the late-bench overload phase, or evenly across the bench).

### Plots

The bench's auto-plot already writes `E5_full_mpc_T14d_traces.png` per run. I'll additionally produce ONE comparison figure overlaying A / B / C's daily Φ on the same axes (4-panel: B, F, A, Φ — same panels as `plot_state_traces.jl` but with three line series per panel). Output PNG `closed_loop_accuracy_comparison.png` in the Phase 2 dir.

### Time budget Phase 2

- A: full 14 d closed-loop at saturated `(N*, K*)`. Per-stride wall from Phase 1 × 27 strides (with K=2 replan there are ~13 replans, 27 filter strides). At `(N=64, K=800)` per-stride ~30–40 s would give ~14–18 min total. At `(N=128, K=1600)` could be 30–60 min.
- B: full 14 d open-loop at same `(N*, K*)`. ~260 s baseline at `(N=32, K=400)` per §5.4; expect 5–20 min at saturated cell because filter still runs even though replans don't (cost is dominated by filter, not controller).
- C: full 14 d closed-loop at `(N=32, K=400)`. ~200 s per §5.3.

Total Phase 2: ~25–80 min, dominated by A.

## Deliverable

The same `RESULTS.md` from Phase 1 grows a Phase 2 section. Headline content:

- **Three-row area-gain table** (A, B, C) with columns: ∫A_MPC dt (A·d), ∫A_baseline dt (A·d), **gain %** (the headline number), F-violation %, Φ peak (day, value), Φ-mean, oscillation amplitude %, total wall-time, output dir.
- **Side-by-side comparison plot** `area_under_A_comparison.png` showing: A trajectories overlaid (A vs B vs C vs baseline), with the cumulative integral ∫₀ᵗ A(s) ds drawn alongside so the area gain is visually obvious. Daily Φ overlay underneath as supporting evidence.
- **Diagnosis paragraph** — which of the three diagnostic buckets the data landed in (saturation helps / saturation invariant / closed-loop loop is leaking the area B finds).
- **Recommended operating-point flags + caveats** for the production bench:
  - "Use `--N-smc <N*> --K-per-chain <K*> --step-minutes 60 --replan-K 2`. Closed-loop ∫A_MPC dt = X A·d, gain = Y % (vs Python +51 %), util = Z %, peak mem = W GB. Residual area gap traces to §2.15 row 10/11; not a flag question." OR
  - "Saturated flags don't move the area; recommendation is to fix §2.15 row 10/11 first and saturate second."

## Verification (Phase 2)

- **B reproduces the §2.15 open-loop number** (∫A_MPC dt ≈ 1.22 A·d, smooth Φ peaking ~0.78 by day 13). Known-good cross-check: if B disagrees materially, my pipeline (or my run) is broken before the area comparison even starts.
- **C reproduces the §5.3 closed-loop number** (the existing `T14d_replanK2_h60min_no_infoaware/` reference). Same cross-check at default config.
- **A's posterior-mean params** (from `posterior_particles[end, :, :]`) sit reasonably close to `truth_params` — i.e. the filter at the saturated config is at least competent. If A's posterior collapsed onto a wrong basin, that confounds the area reading and I'll flag it before declaring a winner.

## Files I will create (Phase 2)

- `outputs/fsa_high_res/gpu_saturation_sweep/<timestamp>/closed_loop_at_saturation/run_A_closed_<N*>_<K*>/` — bench output for A.
- `…/closed_loop_at_saturation/run_B_open_<N*>_<K*>/` — bench output for B.
- `…/closed_loop_at_saturation/run_C_closed_default/` — bench output for C.
- `…/closed_loop_at_saturation/closed_loop_accuracy_comparison.png` — overlay plot.
- `…/closed_loop_at_saturation/extract_metrics.jl` — small read-data.jld2 + compute-metrics + emit-CSV script. One file, no edits to existing model code.
- `…/closed_loop_at_saturation/PHASE2_METRICS.csv` — three rows (A, B, C) with all the columns above.

`RESULTS.md` is the same single report covering both phases.
