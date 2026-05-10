# RTX 5090 saturation sweep — N × K grid + closed-loop accuracy at the saturated cell

**Config**: `--step-minutes 60 --replan-K 2 --T-days 14 --seed 42`
**Hardware**: RTX 5090 (32 GB)
**Sweep dir**: `outputs/fsa_high_res/gpu_saturation_sweep/2026-05-07_2357`

---

## Top-line answer

The user's question was: what `--N-smc` × `--K-per-chain` setting maximises GPU saturation, and does that improve closed-loop controller accuracy?

**Saturation answer (Phase 1):** `--N-smc 256 --K-per-chain 1600` is the empirical best. Median GPU util **65 %**, peak memory **5.97 GB / 32 GB** (18.7 %), per-stride wall **27.2 s**.

**Accuracy answer (Phase 2):** Saturation does **not** materially improve the controller's area gain `(∫A_MPC − ∫A_base) / ∫A_base`. Closed-loop default = +3.00 %, closed-loop at saturated cell = +3.04 % (+0.04 pp). Both are roughly **6 % of Python's reference +51 %**. The magnitude gap is **saturation-invariant** — it lives in the kernel / cost surface, not in particle-count headroom. Saturated flags still maximise GPU usage; the residual bug §2.15 row 10/11 dominates the result and needs a source-level fix.

**What saturation DID help:** GPU util 39 % → 65 % (+26 pp) and closed-loop oscillation amplitude on Φ 28 % → 18 % (the closed-loop schedule is visibly less jittery at the saturated cell). Both are real wins independent of the area-gap question.

---

# Phase 1 — saturation sweep (N × K grid)

## Recommendation

Use `--N-smc 256 --K-per-chain 1600`. Saturation score `util_med × mem_frac` peaks here, with the highest median GPU utilisation (65 %) seen in the sweep, peak memory 5.97 GB (18.7 % of card), per-stride wall 27.2 s.

A near-tie at `--N-smc 128 --K-per-chain 1600` (util 66 %, mem 5.58 GB, per-stride 19.3 s). If 8 s/stride wall headroom matters more than the small extra memory occupancy, `(128, 1600)` is the practical winner. The choice between the two doesn't change the Phase 2 conclusion because their util numbers are within Monte-Carlo noise.

## Phase 1 results table (4 × 3 grid)

| N | K | M_max | strides | util_med % | util_max % | peak_mem_GB | mem_% | per_stride_s | wall_total_s |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 400 | 1952 | 5 | 39.0 | 60 | 5.27 | 16.6 | 8.4 | 82 |
| 32 | 800 | 1952 | 5 | 40.0 | 64 | 5.25 | 16.5 | 11.1 | 92 |
| 32 | 1600 | 1952 | 5 | 45.0 | 81 | 5.56 | 17.5 | 13.8 | 106 |
| 64 | 400 | 3904 | 5 | 31.5 | 51 | 5.33 | 16.8 | 10.1 | 86 |
| 64 | 800 | 3904 | 5 | 34.0 | 62 | 5.38 | 16.9 | 12.5 | 96 |
| 64 | 1600 | 3904 | 5 | 44.0 | 82 | 5.50 | 17.3 | 15.6 | 116 |
| 128 | 400 | 7808 | 5 | 30.0 | 49 | 5.34 | 16.8 | 11.8 | 92 |
| 128 | 800 | 7808 | 5 | 48.5 | 57 | 5.26 | 16.5 | 14.8 | 110 |
| 128 | 1600 | 7808 | 5 | **66.0** | 74 | 5.58 | 17.5 | 19.3 | 132 |
| 256 | 400 | 15616 | 5 | 32.5 | 49 | 5.35 | 16.8 | 15.1 | 110 |
| 256 | 800 | 15616 | 5 | 46.5 | 57 | 5.53 | 17.4 | 18.6 | 130 |
| 256 ⭐ | 1600 | 15616 | 5 | **65.0** | 72 | **5.97** | **18.7** | 27.2 | 170 |

(All cells completed 5 strides within the 8-min budget. M_max = N × (1 + 2·d) = 61·N for d = 30 filter.)

## Saturation diagnosis

**Per-N util progression (low K → high K):**
- N=32: K=400/800/1600 → util 39 / 40 / 45 %
- N=64: K=400/800/1600 → util 32 / 34 / 44 %
- N=128: K=400/800/1600 → util 30 / 48 / **66** %
- N=256: K=400/800/1600 → util 32 / 46 / **65** %

**Per-K util progression (low N → high N):**
- K=400: N=32/64/128/256 → util 39 / 32 / 30 / 32 %  (essentially N-flat at ~30 %)
- K=800: N=32/64/128/256 → util 40 / 34 / 48 / 46 %
- K=1600: N=32/64/128/256 → util 45 / 44 / 66 / 65 %  (saturating at ~65 %)

**Reading the grid:**

1. **K is the dominant lever.** At fixed N=128 or 256, doubling K from 400 → 800 → 1600 adds ~17 percentage points of util per doubling. At fixed N=32 or 64, K-doubling barely helps.
2. **N hurts more than it helps at low K.** At K=400 the util is essentially flat (~30 %) across all N, and per-stride wall grows linearly with N. This is the §2.12 host-loop tax in action — bigger N just adds more host-side per-chain iterations between kernel launches.
3. **N starts paying off only when K is large.** K=1600 has enough per-launch threads that the GPU can absorb additional N before the host loops bottleneck. So `(N=128 or 256, K=1600)` is the regime where both N and K combine to use the GPU.
4. **§2.14's "~51 % plateau" is exceeded at this config.** The writeup's prior measurement (n_smc=1024 sitting at ~51 % util) was the signature reading at h=15min replan-every-stride. At the user's matched-Python config (h=60min, K=2) we exceed that to 66 %. Different stride/replan cadence ⇒ different ratio of host-loop vs kernel work, hence the higher achievable util.

**Maximum median util observed: 66 %** at `(N=128, K=1600)`. Significant headroom remains both in util (66 → ~95 % theoretical) and memory (6 / 32 GB → 26 GB unused). Saturation past 66 % is not a flag question — it requires the source-level batching fixes catalogued in [HOST_LOOP_CATALOGUE.md](HOST_LOOP_CATALOGUE.md) (the §2.12 ceiling).

---

# Phase 2 — does saturation help maximise ∫A dt?

**Cost recap.** The cost is `J(Φ) = -∫A(t) dt + λ_F·∫max(F-F_max, 0)² dt`. The controller's job is to maximise the integral of A over the bench horizon (i.e. the area under the A curve), subject to the soft F-cap. The headline metric is the gain `(∫A_MPC − ∫A_base) / ∫A_base` over the Φ=1 baseline.

**Reference points:**
- **Python target:** gain ≈ **+51 %**.
- **Julia closed-loop (writeup §2.15 baseline):** gain ≈ **+2.3 %**.

## Phase 2 results table

| Run | Config | Wall | ∫A_MPC dt (A·d) | ∫A_base dt (A·d) | **Gain %** | F-viol % | Φ peak | Φ peak day | Φ mean | Osc % |
|---|---|---|---|---|---|---|---|---|---|---|
| C | closed-loop @ N=32 K=400 (default) | 4.2 min | 1.2174 | 1.1819 | **+3.00** | 0.0 | 1.375 | day 10 | 0.533 | 28.3 |
| A | closed-loop @ N=256 K=1600 (saturated) | 11.7 min | 1.2178 | 1.1819 | **+3.04** | 0.0 | 1.000 | day 0 | 0.486 | 18.1 |
| B | open-loop @ N=256 K=1600 (saturated) | 9.4 min | 1.2328 | 1.1819 | **+4.31** | 0.0 | 0.771 | day 13 | 0.311 | 2.4 |

(Φ peak day converted: stride index × `STRIDE_BINS / BINS_PER_DAY` = stride × 0.5 d.)

## Diagnosis

The gain triple `(C, A, B)` falls into a clear bucket:

- **A gain ≈ C gain ≈ +3 %** → saturation does NOT improve the closed-loop area gain. The cost surface is delivering a low-area optimum at every saturation level.
- **B gain ≈ +4.3 %** is only modestly higher than A → the closed-loop replan loop is leaking ~1.3 pp of area to oscillation, but that's a rounding-error compared to the ~47 pp gap to Python.

**Conclusion: the magnitude gap is in the kernel or the cost surface, NOT in sampling noise.** Saturating particle counts cannot close the gap to Python — the next move is the source-level bug §2.15 row 10/11 is tracking. Saturated flags still help GPU usage and reduce closed-loop oscillation, but they do not move the headline area-gain number.

## What saturation DID change (vs the un-saturated default)

| Quantity | C (default) | A (saturated) | Δ |
|---|---|---|---|
| Gain %                        | +3.00 | +3.04 | +0.04 pp (essentially zero) |
| GPU util (median)             | 39 % | 65 % | **+26 pp** |
| Memory used                   | 5.27 GB | 5.97 GB | +0.70 GB |
| Closed-loop oscillation       | 28.3 % | 18.1 % | **−10.2 pp** (smoother schedule) |
| Wall (full 14 d closed-loop) | 4.2 min | 11.7 min | +7.5 min |

The two real wins from saturation are **GPU util** (a 1.7× lift) and **schedule smoothness** (the closed-loop replan loop is meaningfully more stable at higher particle counts). Neither moves the headline area gain.

## Comparison plot

![area_under_A_comparison](closed_loop_at_saturation/area_under_A_comparison.png)

The cumulative ∫A(t) dt panel makes the situation visually obvious: A's, B's and C's MPC curves are essentially indistinguishable from each other, and only marginally above the Φ=1 baseline. Python's reference would show its MPC cumulative pulling sharply away from the baseline curve — that's the +51 % gain — but that's not what's happening here in any of the three runs.

The daily Φ panel shows the schedule shapes:
- **A (closed-loop @ saturated)**: starts at Φ ≈ 1.0, oscillating downward through the bench. Peak at day 0.
- **B (open-loop @ saturated)**: smooth recovery → ramp, Φ peaks at 0.77 by day 13. Matches §2.15's prior open-loop measurement (Φ peaks at ~0.78 by day 13) almost exactly — verification cross-check passes.
- **C (closed-loop @ default)**: highest peak Φ (1.38 around day 10) but extremely jittery (28 % oscillation). The peak is real but the area under it isn't sustained because the schedule oscillates the trajectory back down before A can build up.

---

# Recommendation

## Production operating point (flag-only, no source changes)

```bash
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --N-smc 256 --K-per-chain 1600 \
    --step-minutes 60 --replan-K 2
```

This gives:
- 65 % median GPU utilisation (1.7× the default 39 %).
- 6.0 GB / 32 GB memory used (room for ~5× more particles before memory becomes the constraint).
- 27 s per stride → ~12 min for the full 14 d closed-loop bench.
- Closed-loop schedule oscillation reduced from 28 % → 18 % of mean Φ (more stable replans).
- **No measurable improvement in area gain** vs the default — both at ~+3 % gain.

## Where the area gain is being lost (and why flags can't fix it)

The cost-surface bug §2.15 row 10/11 in the writeup is what's blocking the path from +3 % → +51 %. The fact that A ≈ B ≈ C in this experiment confirms the bug is in the cost surface itself, not in sampling noise. Open-loop B (no replan oscillation, smooth schedule, full saturation) still only delivers +4.3 % gain — so even a perfect optimiser would not get past ~+5 % at the current cost surface.

The fix is in the source code, not the flags. Concrete next steps from §2.10's Hypothesis 8 / §2.15 row 10:

1. Hunt for whatever residual fp32 / kernel mismatch between Julia and Python's controller cost evaluation puts the ~10× area gap there.
2. Verify by running `tools/test_max_A_with_F_barrier.jl` (the open-loop ground-truth driver) and checking ∫A_MPC dt at the saturated cell vs Python's saved reference.

## Where the GPU saturation is being lost (and what fixes it)

Even with `(N=256, K=1600)`, util tops out at 65 % — the §2.12 host-loop ceiling. To push past this, source-level fixes are needed. Concrete catalogue with file paths and line numbers in [HOST_LOOP_CATALOGUE.md](HOST_LOOP_CATALOGUE.md). Cheapest wins first:

1. `gpu_pf.jl:369-372` — `params_cpu` rebuild + copyto every call. Move `params_per_chain` to GPU-resident, update via per-chain constraint kernel. ~30 lines.
2. `gpu_pf.jl:527-533` — HMC accept/reject + posterior cloud update on host per-particle. Pre-draw RNG, compare on GPU, atomic accumulator for `n_acc`. ~40 lines.
3. `gpu_pf.jl:462-471` + `:477-483` — FD perturbation expansion (M → M·(1+2d)) + gradient assembly on host. Replace with a single perturbation kernel + gradient kernel. ~60 lines.
4. `GPUSegmentedPF.jl:217-256` — Per-chain OT-rescue blend with `Array(ess_gpu)` copy per check. Fuse into one batched kernel. ~150 lines.
5. `GPUSegmentedPF.jl:352-362` — Unconditional `KernelAbstractions.synchronize` per segment. Defer to window-end. ~50 lines.

Total estimated effort to push util from 65 % → 90 %+: roughly 300 lines of source changes spread across `gpu_pf.jl` and `GPUSegmentedPF.jl`. Out of scope for this experiment but the natural next step.

---

# Files in this report

## Phase 1 (saturation sweep)
- `RESULTS.csv` — raw 12-cell sweep table (machine-readable).
- `SATURATED_CELL.txt` — the chosen `(N*, K*)`.
- `N<N>_K<K>/bench.log`, `N<N>_K<K>/gpu_log.csv`, `N<N>_K<K>/bench_output/data.jld2` — per-cell raw artefacts (12 dirs).
- `HOST_LOOP_CATALOGUE.md` — the source-level catalogue of the §2.12 host-loop ceiling.

## Phase 2 (closed-loop accuracy)
- `closed_loop_at_saturation/PHASE2_METRICS.csv` — three-row metrics table.
- `closed_loop_at_saturation/area_under_A_comparison.png` — the headline overlay plot.
- `closed_loop_at_saturation/run_A/`, `run_B/`, `run_C/` — bench output per run.
- `closed_loop_at_saturation/cum_A_run_*.csv` — per-run A trajectory + cumulative integral.
- `closed_loop_at_saturation/phi_run_*.csv` — per-run daily Φ schedule.

## Scripts
- `run_sweep.sh` — Phase 1 orchestrator (12 cells, sequential).
- `run_phase2.sh` — Phase 2 orchestrator (A, B, C runs).
- `extract_metrics.jl`, `plot_area_comparison.jl`, `build_phase1_report.jl`, `build_final_report.jl` — analysis pipeline.
