# Localising residual fp32 bias in the v2 GPU cost kernel

> Archived from plan mode: 2026-05-07 19:56.


## Context

After §2.8 (the Horner-form rewrite of μ + precomputed `a_typ_inv_*`), the production fp32 kernel matches Python fp64 bit-for-bit at the two probe θ values (θ=0 and θ_RO, max rel diff 3e-6). The light-config open-loop test (`test_one_plan_fp64.jl`, n_inner=16) finds the recovery→overload optimum end-to-end. **But the heavy-config closed-loop bench (n_inner=128, 27 replans) still settles to the flat-low schedule** — verified empirically in the run that just finished. Each per-replan plan looks recovery→overload-shaped (Φ̄=0.30–0.39), but cumulative applied Φ stays in [0.10, 0.30].

The plausible mechanism: **the §2.8 rewrite removed the worst (cancellation-prone) fp32 ops, but smaller systematic biases survive in the still-fp32 parts of the kernel. At light config (n_inner=16) Monte-Carlo noise on each cost evaluation is roughly 8× larger than at heavy (n_inner=128), so the surface shape is dominated by CRN variance and the global recovery→overload basin still wins. At heavy config the MC noise is averaged away; whatever residual systematic fp32 bias is left becomes the dominant distortion of the cost surface, displacing the optimum to flat-low.** This is consistent with everything observed: bit-for-bit matches Python where the surface is well-conditioned (probe θ), but optima diverge at production scale where MC noise no longer masks the bias.

The fully-promoted fp64 kernel (currently sitting in `gpu_control.jl` from the last edit) confirms the diagnosis at the cost of ~30× wall-time on consumer Blackwell. That isn't shippable. The goal of this plan is to **find the smallest fp64 promotion that fixes the bug** and ship that.

## Three candidate locations for residual fp32 bias

Inside the kernel body in [version_2_Julia/models/fsa_high_res/gpu_control.jl](version_2_Julia/models/fsa_high_res/gpu_control.jl) (the current full-fp64 version is the upper bound; we're hunting for the minimal subset):

1. **Cost accumulators**, lines 76-77 + 90-91 of the §2.8 fp32 form:
   `A_acc = 0f0; barrier_acc = 0f0` then per-bin `A_acc = A_acc + A*dt`. Sum of 1344 small terms accumulating to ~1.4. By the tail of the sum the relative add ratio `|A·dt|/|A_acc|` is ~1e-3, well into the regime where fp32 mantissa bits start dropping increments. The bias is **plan-dependent** (plans that drive A high accumulate more, lose more bits) → distorts the cost *surface*, not just the level. **Cheapest to fix** — two scalar promotions per thread, no per-substep cost.

2. **State accumulation**, lines 113-115 of the §2.8 fp32 form:
   `B = B + sub_dt * drift_B` × 5376 substeps. Same compounding-rounding pattern, but on the state itself. If state drifts off, downstream A diverges.

3. **`mu_bif` Horner sum**, line 107: `mu_bif = p_mu_const + p_mu_B*B + p_mu_lin_F*F - p_mu_FF*F*F`. Four signed terms of similar magnitude added in fp32. Less likely to be plan-dependent at the magnitudes involved (all four terms are O(0.01–0.1)) but worth checking if (1) and (2) don't suffice.

## Methodology — promote one location at a time, measure each step

### Step 0: Restore the §2.8 production baseline

Revert [gpu_control.jl](version_2_Julia/models/fsa_high_res/gpu_control.jl) from the current full-fp64 state to the §2.8 fp32+Horner state. That's the actual production kernel under investigation. Re-confirm:

- **Bit-for-bit at θ=0 / θ_RO** via [test_cost_at_theta0.jl](version_2_Julia/tools/test_cost_at_theta0.jl) → max rel diff vs Python fp64 should be ≤ 3e-6.
- **Light open-loop** via [test_one_plan_fp64.jl](version_2_Julia/tools/test_one_plan_fp64.jl) → posterior-mean Φ ramps 0.10→0.98 over 14 d (recovery→overload).
- **Heavy closed-loop** via [bench_smc_full_mpc_fsa_gpu.jl](version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) → applied Φ stays in [0.10, 0.30] across 27 strides (the broken state we are trying to fix).

This step is the control: nothing else can be claimed until we have these three baseline outcomes back.

### Step 1: Promote cost accumulators to fp64 — Hypothesis 1

The minimal kernel diff:

```julia
# inside the @kernel body
A_acc = 0.0          # was 0f0
barrier_acc = 0.0    # was 0f0
# accumulator increments — Julia auto-promotes Float32 * Float32 to Float64 at the +=:
A_acc       = A_acc + Float64(A * dt)
barrier_acc = barrier_acc + Float64(max(F - F_max, 0f0)^2 * dt)
# final:
cost_per_thread[i] = Float32(-A_acc + lam_F * barrier_acc)   # store as fp32 if cost_per_thread is Float32
```

Everything else (state, drift, mu_bif, params, noise) stays fp32. Per-thread cost overhead: two Float64 adds per outer bin × 1344 bins = 2688 extra Float64 ops vs fp32. Negligible vs the ~5376 substep multiplies that dominate.

Verifications, in order:

a. **Bit-for-bit at θ=0 / θ_RO** — should *improve* slightly vs Step 0 (less rounding in the per-trial cost), still match Python to ≤ 3e-6.
b. **Light open-loop** — must still find recovery→overload. Regression test: if Φ shape collapses to flat-low at light config, the change broke something orthogonal. Stop, debug.
c. **Heavy closed-loop bench** — the diagnostic. Two outcomes:
   - **applied Φ shifts to recovery→overload (Φ̄ rises toward 0.5–1.0 across strides, F-violation gate tightens, mean ∫A/T jumps from 0.085 to ~0.12)**: Hypothesis 1 confirmed. Ship.
   - **still flat-low**: Hypothesis 1 ruled out; move to Step 2 with the accumulators kept in fp64 (don't revert — they're a free correctness win even if not sufficient).

### Step 2: Add state (B, F, A) to fp64 — Hypothesis 2

Cumulative on top of Step 1: also promote the state vector.

```julia
B = Float64(init_state[1])    # init_state may stay Float32 array
F = Float64(init_state[2])
A = Float64(init_state[3])
# substep loop: B, F, A are now Float64, all drift/Phi multiplies in mixed precision
# Julia auto-promotes; the assignment `B = B + sub_dt * drift_B` keeps B Float64.
# After the diffusion + reflection block, B, F, A are still Float64 (no demotion).
```

Per-thread cost overhead: state lives in Float64 registers throughout — modest GPU register pressure, no DRAM traffic change. Should be nearly free on Blackwell.

Same verification suite (a/b/c). If (c) shifts: Hypothesis 2 confirmed (need accumulators + state). Otherwise move to Step 3.

### Step 3: Add the drift computation to fp64 — Hypothesis 3

Cumulative: promote `mu_bif`, `a_factor_B`, `a_factor_F`, `drift_B/F/A` and the params they read to fp64. This is the spot the failed full-fp64 conversion already occupied — it's the upper bound. If even this doesn't fix the heavy closed-loop, the bug isn't fp32 precision and the hypothesis from §2.8 was wrong. (Open-loop with full-fp64 already shipped recovery→overload, so this end-state should at least match the open-loop result.)

### Step 4: Pick the minimum sufficient promotion and lock it in

From the three measurements, pick the cheapest set that produces recovery→overload in the heavy closed-loop. Document in `gpu_control.jl` with a comment block citing the empirical comparison. Update the writeup `julia_fsa_writeup.pdf` §2.8 to reflect what actually fixed it.

## Files touched

Single file each step: [version_2_Julia/models/fsa_high_res/gpu_control.jl](version_2_Julia/models/fsa_high_res/gpu_control.jl). The struct (`FSAControlGPUTarget`), the constructor, and `gpu_cost_log_density_batched` may need to follow if `cost_per_thread` or `init_state` change dtype. No bench script changes needed — same call signature.

## Existing utilities to reuse

- [test_cost_at_theta0.jl](version_2_Julia/tools/test_cost_at_theta0.jl) — bit-for-bit Julia ↔ Python comparison at fixed θ on saved CRN noise. Run after every kernel edit.
- [test_one_plan_fp64.jl](version_2_Julia/tools/test_one_plan_fp64.jl) — light-config open-loop, ~15–60 s wall. Quick regression: every step must keep recovery→overload here.
- [bench_smc_full_mpc_fsa_gpu.jl](version_2_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl) — production heavy closed-loop, ~12 min. The actual diagnostic.

## Wall-time budget for the whole investigation

| Step | bit-for-bit | open-loop light | closed-loop heavy | total |
|---|---:|---:|---:|---:|
| 0 (baseline) | 10 s | 15 s | ~13 min | 13 min |
| 1 (acc fp64) | 10 s | 15 s | ~13 min | 13 min |
| 2 (+ state fp64) | 10 s | 15 s | ~13–15 min | 15 min |
| 3 (+ drift fp64) | 10 s | 15 s | ~30–60 min | 60 min |

Best case (Hypothesis 1 wins): ~26 min wall. Worst case (need everything): ~100 min.

## Stop conditions

- **Stop early if Step 1 already shifts the heavy bench to recovery→overload.** That's the cheapest fix; ship it.
- **Stop and reassess if any step regresses the open-loop test** (recovery→overload → flat-low at light config). Means the kernel diff broke something orthogonal to the precision question. Don't proceed to the heavy bench until the open-loop is back to recovery→overload.
- **Stop after Step 3 regardless.** If even the full-fp64 kernel doesn't fix the heavy closed-loop, the §2.8 framing of "fp32 cancellation displaces the optimum" was wrong, and we're back to looking at orchestration / strategy bugs (the closed-loop wiring, the bursty plant ↔ smooth-cost mismatch, the seed). The full-fp64 single-plan test already showed the open-loop *does* find recovery→overload, so the open-loop ↔ closed-loop divergence would point firmly at orchestration in that case.

## Verification — what counts as "fixed"

The Julia heavy closed-loop bench produces the trace plot `E5_full_mpc_T14d_traces.png` (auto-generated via the bench's plotting hook). The fix is judged against the **Python reference plot** at:

`/home/ajay/Repos/python-smc2-filtering-control/version_2/outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/E5_full_mpc_T14d_traces.png`

That reference shows the result the v2 closed-loop is *supposed* to produce. Header reads: "mean A 0.122 vs baseline 0.081, F-viol 0.0%, 27 min". Specifically:

- **Bottom-right panel (applied daily Φ across 27 strides)**: starts at Φ=1.0 (day 0), drops sharply to Φ≈0.2 (days 1–3, recovery), holds low through day 5, then climbs in a bumpy ramp to Φ≈1.2 by day 13 (overload). Distinctive U-then-ramp shape, not flat. Current Julia bench: flat band in [0.10, 0.30] across all strides.
- **Top-left (B trajectory)**: MPC rises gradually from 0.05 to ~0.10 by day 13. Current Julia: stuck at 0.05–0.06.
- **Top-right (F)**: MPC decays from 0.30 down to ~0.11 by day 10, then rises slightly to 0.14 by day 13 (overload phase). F_max=0.40 never crossed (F-viol = 0%). Current Julia: monotone decay to 0.07, no late uptick.
- **Bottom-left (A trajectory)**: MPC rises from 0.10 → ~0.15 by day 8, peaks at ~0.17 around day 11, settles ~0.15 at end. Mean MPC = **0.122** vs baseline = **0.081** (+51%). Current Julia: 0.085 vs 0.082 (+3.7%).

The Julia plot is "fixed" when its `E5_full_mpc_T14d_traces.png` qualitatively matches all four panels of the Python reference. Numerical match within MC noise is sufficient (different RNGs, no expectation of pixel match):

- Mean A under MPC ≥ ~0.115 (allow ~5% MC slop on Python's 0.122)
- F-violation fraction ≤ 1%
- Applied Φ: max value over the 27 strides ≥ ~1.0 (i.e. the late-overload phase actually reaches baseline-level training); min value over the first 4 days ≤ ~0.30 (recovery actually happens).
- Φ schedule visibly U-shaped (early high → mid low → late high) rather than flat in any band.
