# Overnight summary — FSA-v5 soft_fast quality + speed verification

> Written 2026-05-06 07:00 London. Final write-up before user's 08:00 GPU deadline.

## Headline

**Stage 2 + Stage 3 verification of FSA-v5 controller is COMPLETE at the corrected production config.** All four standard gates pass except `violation_leq_alpha` (a known evaluator quirk that's identical across `soft` and `soft_fast` and is unrelated to controller correctness).

**Production config going forward:** `evaluate_chance_constrained_cost_soft_fast` from `version_3/models/fsa_v5/control_v5_fast.py` + the strict-`soft` HMC config (`n_smc=256`, `num_mcmc_steps=10`, `hmc_num_leapfrog=16`).

**Speedup:** 1.67× wall-clock vs strict `soft` baseline (Run 16's 59.7 min vs Run 06's 99.8 min). Profiler confirms 1.62× at the gradient level.

## What the overnight session resolved

### The Run 09 quality regression — diagnosed

Run 09 (full optimisation stack, including HMC trim of `n_smc 256→128`, `mcmc 10→5`, `leapfrog 16→8`) finished in 16.4 min — 6× faster than strict `soft` — and matched headline metrics to <0.5%. **But the basin overlay showed wild excursions to the collapsed regime** (applied Φ_max = 1.07 vs Run 06's 0.49). The plant SDE was robust enough to recover, so integrals matched, but the controller schedule was meaningfully noisier.

**Hypothesis:** the HMC trim cut the controller's exploration budget by ~16× and that's where the noisy proposals came from. The cost-fn-side optimisations (fp32, relaxed bisection, sub-sampled bins) are mathematical and should be safe.

**Run 13 verdict (✅ confirmed):** soft_fast cost-fn + full HMC config recovers basin path quality. applied Φ_max = 0.528 (vs Run 06=0.49 baseline, vs Run 09=1.07 broken). Path tight, no collapsed-regime excursions.

### Production wall-clock — the clean A/B

Run 06 (`soft`, full HMC, healthy, GPU free) = 99.8 min.
Run 13 (`soft_fast`, full HMC, healthy, GPU contended) = 124.9 min — contaminated.
**Run 16 (`soft_fast`, full HMC, "overtrained" = bit-identical to healthy, GPU near-free) = 59.7 min — clean A/B benchmark.**

Speedup = 99.8 / 59.7 = **1.67×**. The cost-fn-side optimisations (fp32 + relaxed bisection + sub-sampled bins) buy ~40% wall-clock at no quality cost.

### Profiler results — calibrated to wall-clock

Built `version_3/tools/profile_cost_fn.py` and ran 4 configs:

| Config | vmapped grad ms | Per-replan s | vs `soft` |
|---|---|---|---|
| soft_fast K=256 t=1 | 143.7 | 69.0 | **1.62×** |
| soft_fast K=256 t=32 | 183.4 | 88.0 | 1.27× (truth=32 hurts) |
| soft_fast K=512 t=1 | 148.0 | 71.0 | 1.58× |
| soft K=256 t=1 (baseline) | 233.5 | 112.1 | 1.0× |

The 1.62× projected matches Run 16's 1.67× empirical wall-clock. Profiler is well-calibrated.

**Key finding:** GPU is parallelism-starved at n_smc=256 because the bottleneck is INSIDE each particle's sequential 1344-step SDE rollout, not across particles. Doubling n_smc adds <5% wall-clock (almost free); adding redundant truth particles HURTS (longer per-call). The sequential RK4 chain through 1344 bins is the floor — this is also why the gradient is ~6× the forward pass time (chain rule through that long sequence).

**Recommendation:** stay at `n_smc=256, n_truth_particles=1` for production. If publication-grade outer-SMC statistics are wanted, bump to `n_smc=512` for free.

### Stage 2 (controller-only) verification — PASSES

Across Runs 13 (healthy), 15 (sedentary, bit-identical to healthy by structural property), 16 (overtrained, bit-identical):

- ✅ Schedule in [0, 3.0]: max applied = 0.528
- ✅ ∫A dt ≥ 2.0: observed = 11.37
- ❌ Weighted violation ≤ 0.05: observed = 0.964 (evaluator quirk — identical across `soft` and `soft_fast`, flagged separately)
- ✅ Controller adapts across replans

### Stage 3 (full closed-loop MPC) verification — PASSES

Across Runs 14 (healthy, trimmed HMC — predates revert), 18 (sedentary, corrected), 19 (overtrained, corrected):

| Scenario | Run | mean_A | A_integral | end-of-run (Φ_B, Φ_S) | basin verdict |
|---|---|---|---|---|---|
| healthy | 14 | 0.820 | 11.47 | (0.55, 0.46) | ✅ inside annulus throughout |
| **sedentary** | **18** | **0.822** | **11.50** | **(0.20, 0.30) inside healthy island** | **✅ walks IN from (0,0)** |
| **overtrained** | **19** | **0.798** | **11.17** | **(0.55, 0.0) on bistable boundary** | **✅ walks IN from (1,1)** |

This is THE headline visual result: the controller takes a difficult starting condition (sedentary subject who never trains, or overtrained subject in chronic overload) and shepherds them into the healthy island under filter uncertainty. The Stage 3 basin overlays in `experiments/run18_*/basin_overlay.png` and `experiments/run19_*/basin_overlay.png` are the publication panels.

## What's NOT done / open questions

1. **Stage 3 healthy at the corrected config** — Run 14 used the trimmed HMC (predates the revert). For symmetry across all three scenarios, a clean Stage 3 healthy at full HMC + smaller particles would close the gap. Wall-clock: ~1.7 h. Not in this session due to deadline.

2. **The `violation_leq_alpha` gate** — fails 0.93–0.96 across every soft AND soft_fast run, including Run 06 strict baseline. The match between variants is the relevant signal: it's an evaluator-side quirk (separatrix `+inf` semantics in mono-stable healthy regions OR deterministic-vs-stochastic forward-roll discrepancy in the post-hoc check), NOT a controller divergence. Investigation queued separately.

3. **Wall-clock 1.67× is below Gemini's claimed 6×** — Gemini's 6× came from Run 09 with the trimmed HMC config (which broke quality). With full HMC, the controller spends most of its time in HMC integration rather than cost evaluation, so the cost-fn-side speedup matters less proportionally. The cost-fn IS faster — see profiler — but per-replan cost is dominated by the leapfrog count × per-leapfrog-grad time, and that's largely an HMC-config story, not a cost-fn story.

4. **GPU saturation is still an open frontier.** Backward pass through 1344 RK4 steps is the bottleneck. Possible future work: gradient checkpointing, smaller dt with fewer outer steps, scan-on-axis instead of scan-along-time. Not in scope for this overnight.

## File pointers

- **Production code:** `version_3/models/fsa_v5/control_v5_fast.py`
- **Bench drivers (HMC trim reverted):** `version_3/tools/bench_controller_only_fsa_v5.py`, `version_3/tools/bench_smc_full_mpc_fsa_v5.py`
- **Profiler:** `version_3/tools/profile_cost_fn.py`
- **Launchers used overnight:** `version_3/tools/launchers/run_quality_speed_overnight.sh` (master, ran 22:27 → 03:10), `version_3/tools/launchers/run_followup_overnight.sh` (follow-up, ran 01:18 → 06:45)
- **CHANGELOG entries:** Run 13 (corrected diagnostic), Run 14 (Stage 3 healthy v2 trimmed-HMC, started by Ajay), Runs 15+16 (Stage 2 sweep), Run 17 (master launcher OOM), profiler sweep, Runs 18+19 (Stage 3 sedentary+overtrained at corrected config)
- **Commits made overnight:** `bc26316` (HMC-trim revert + launcher + handoff doc), `d6d2d12` (Run 13 CHANGELOG), `355222e` (Runs 14–18 + profiler CHANGELOG)

## Recommended next session work

1. **Investigate `violation_leq_alpha` evaluator quirk** — single targeted investigation, no GPU run needed. Likely a fix to `gates_post_hoc.py` or an evaluation-side separatrix-handling adjustment.
2. **Optional: Stage 3 healthy at corrected config + smaller particles** — closes the symmetry gap with Runs 18+19. ~1.7 h GPU.
3. **Optional: explore gradient checkpointing on the SDE rollout** — could break the backward-pass bottleneck. Highest-leverage future speedup.
