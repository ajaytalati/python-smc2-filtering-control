# SWAT Benchmark Changelog

## [Run 23] - 2026-05-05
**A/B comparator: Run 22 with λ_E=0 (pure ∫T objective)**
- **Auto-launched** by watchdog at 01:09 once Run 22 passed clean.
- **Result:** mean T (MPC) = **0.1835** vs baseline 0.0884 → **2.08× improvement**. T-floor violation 29.02%. Wall-clock 230.5 min.
- **A/B verdict:** vs Run 22 (λ_E=1) the difference is mean T 0.1847 → 0.1835, i.e. −0.65% — well within MC noise. **The entrainment-shaping cost term (`λ_E · ∫E_dyn`) makes no measurable difference at T=14 d** under the pathological scenario. Same conclusion as the T=2 A/B (Run 16 vs Run 17 were identical to 4 decimals). H3 (bifurcation-cliff argument) was correctly demoted; the pure ∫T objective is sufficient.
- **Plots:** `version_2/outputs/swat/experiments/run23_PRODUCTION_T14_full_closedloop_lambda0/`.

## [Run 22] - 2026-05-04 → 2026-05-05
**Production T=14 closed-loop (full filter + controller, all qualitative fixes)**
- **Changes:** Reverted D2 (controller `n_substeps` 10 → 4) and D5 (controller `max_lambda_inc` 0.1 → 0.5) after the first attempt projected ~7 h wall-clock vs the user's ~3 h expectation. D2 and D5 had shown no measurable mean-T benefit at T=2; their cost was the slowdown. Kept D1 (V_c bound), D3 (logit bias), D4 (V_c σ scale), D6 (λ_E=1). Restored particles to Run 09 retry levels (256/200/256/32) since memory fits at n_substeps=4. Truth τ_T=48 h, scenario `pathological`, λ_E=1.
- **Result:** **mean T (MPC) = 0.1847 vs baseline 0.0884 → 2.09× improvement.** ✓ Gate 1 (mean T ≥ 0.95× baseline). Comparable to Run 08's fixed-test-controls reference (mean T 0.204). T-floor violation 28.4% (Run 08: 25.7%). id-cov gate fail. Wall-clock 224.6 min (3.74 h) — under the 4 h gate. Closed-loop SMC²-MPC reaches ~91% of the optimal-fixed-controls performance while reacting to noisy obs in real time.
- **Plots:** `version_2/outputs/swat/experiments/run22_PRODUCTION_T14_full_closedloop_lambda1/`.

## [Run 21] - 2026-05-04
**Controller-only, T=7, tau_T quartered to 12h — qualitative win**
- **Changes:** Run 19's setup + diagnostic τ_T override (`SWAT_TAU_T_OVERRIDE_HOURS=12`, applied identically to plant and controller). Quartering τ_T from 48 h to 12 h speeds up the Stuart-Landau exp() growth so the bifurcation crossing fits inside 7 days. Truth value untouched in source.
- **Result:** **mean T (MPC) = 0.2564 vs baseline 0.0314 → 8.2× improvement**. T-floor violation 21.43% (vs Run 19's 56.70% at truth τ_T; comparable to Run 08's 25.7% on the analogous T=14 truth-τ_T fixed-control reference). Per-replan: V_h̄ 1.19–2.92, V_n̄ 0.01–0.14, V_c̄ −0.69 to +0.72. Compute 141 min. Visible T climb across days 2–7 mirrors Run 08's days 8–14 climb at truth τ_T.
- **Plots:** `version_2/outputs/swat/experiments/run21_controller_only_T7_pathological_lambda1_tauT12h/`.

## [Run 20] - 2026-05-04 (cancelled — superseded)
**Controller-only T=14 (cancelled)** — first attempt OOMed at stride 1 (256/32 particles → 41 GiB working set vs 32 GB cap). Lowered `_particle_counts_for_horizon` to 128/16 for T≥14. Retry was running but cancelled when Ajay suggested the τ_T override approach (Run 21).

## [Run 19] - 2026-05-04
**Controller-only, T=7, all fixes, truth params**
- **Changes:** First use of new `tools/bench_controller_only_swat.py` (skips the SMC² filter; uses truth params + actual plant state at every replan, per `claude_plans/controller_only_test_methodology.md`). All 6 controller fixes (D1 V_c bound, D2 n_substeps=10, D3 logit bias, D4 V_c σ scale, D5 max_lambda_inc=0.1, D6 λ_E=1).
- **Result:** **mean T (MPC) = 0.0471 vs baseline 0.0446 → 1.06× improvement** (first MPC > baseline at any horizon, but at the noise floor — at truth τ_T the Stuart-Landau exp() growth needs >τ_T = 2 d to start climbing, so 7 d is too short to show the dramatic separation). T-floor violation 56.7%. V_n̄ 0.05–0.24 (correctly low). V_c̄ −0.38 to +0.70 (well-behaved). Compute 134 min.
- **Speedup honesty:** measured ~14% faster than full closed-loop bench at T=7, not the 3–4× I'd estimated. Controller's tempered-SMC at 10 levels × T=7 horizon × 10 sub-steps × 256 particles dominates compute now; filter was a smaller share than estimated. Methodology doc updated.
- **Plots:** `version_2/outputs/swat/experiments/run19_controller_only_T7_pathological_lambda1/`.

## [Run 18] - 2026-05-04 (cancelled — superseded by controller-only)
**ALL FIXES T=7 (full closed-loop, cancelled)** — running cleanly at stride 30/49 in 78 min when Ajay redirected to the new controller-only test strategy. Final config (D1+D2+D3+D4+D5+D6) was promoted to Run 19's controller-only bench.

## [Run 17] - 2026-05-04
**D1+D2+D3+D4+D5+D6 — λ_E=1 (T=2)**
- **Changes:** Layered D6 on D1+D2+D3+D4+D5. D6 = `SWAT_LAMBDA_E=1.0` (env var only; restores bench default cost shaping that Run 09 had ablated).
- **Result:** Identical to D5 to 4 decimals. λ_E=0 vs λ_E=1 makes no measurable difference at T=2 — confirms H3 demotion. The structural fixes (D1, D3) drove the visible improvement.
    - mean T (MPC) = 0.0372 vs baseline 0.0384 → 0.969×. **Gate 1 ✓.**
    - V_h̄ 2.15–2.78, V_n̄ 0.06–0.12, V_c̄ −0.14 to 0.15 across 4 replans — flat, committed, healthy operating point.
    - id-cov 9/9, wall-clock 26.3 min.
- **Plots:** `version_2/outputs/swat/experiments/run17_D1D2D3D4D5D6_T2_lambda1/`.

## [Run 16] - 2026-05-04
**D1+D2+D3+D4+D5 — controller `max_lambda_inc` 0.5→0.1 (T=2)**
- **Changes:** Layered D5 on D1+D2+D3+D4. D5 = `bench_smc_full_mpc_swat.py:367` controller `max_lambda_inc` lowered from 0.5 to 0.1 to force the tempering ladder to ≥10 steps (Run 09 / Run 10 stuck at 5 levels).
- **Result:** Tempering now 10 levels per replan (was 5). Controller's posterior tighter and more committed.
    - mean T (MPC) = 0.0372 vs baseline 0.0384 → 0.969×. **Gate 1 ✓.**
    - V_h̄ now 2.35 / 2.45 / 2.70 / 2.38 (very tight; was 2.16–3.36 with D3).
    - V_n̄ 0.05 / 0.08 / 0.09 / 0.19 (low, rest/recovery).
    - V_c̄ −0.16 / 0.28 / 0.11 / −0.10 (well-behaved near 0).
    - Wall-clock 26.3 min (~16% slower from doubled tempering work).
- **Plots:** `version_2/outputs/swat/experiments/run16_D1D2D3D4D5_T2_max_lambda_inc_lambda0/`.

## [Run 15] - 2026-05-04
**D1+D2+D3+D4 — per-variate σ on V_c (T=2)**
- **Changes:** Layered D4 on D1+D2+D3. D4 = scale `raw_c` by 1/3 inside `schedule_from_theta`, equivalent to an effective σ_prior=0.5 on the V_c block (engine's σ_prior is scalar; true per-variate would need engine change → out of scope per principle).
- **Result:** Marginal further improvement.
    - mean T (MPC) = 0.0372 vs baseline 0.0384 → 0.969× ≥ 0.95×. **Gate 1 ✓.**
    - V_c̄ tightened further (last replan V_c=0.08); applied schedule shows V_c hovering very close to 0 throughout. Confirms the tighter effective prior keeps V_c near the unbiased centre when the data doesn't pull it.
    - V_h̄, V_n̄ behaviour unchanged from D3 (D4 only touches V_c).
    - id-cov 9/9, wall-clock 22.6 min.
- **Plots:** `version_2/outputs/swat/experiments/run15_D1D2D3D4_T2_per_variate_sigma_lambda0/`.

## [Run 14] - 2026-05-04
**D1+D2+D3 — logit bias on V_h, V_n (T=2)**
- **Changes:** Layered D3 on D1+D2. D3 = logit-biased sigmoid in `schedule_from_theta` so θ=0 → V_h=1.0, V_n=0.2 (the bench's `set_A` healthy operating point), instead of the sigmoid mid-range default V_h=2.0, V_n=2.5.
- **Result:** **First gate-1 PASS.**
    - mean T (MPC) = 0.0370 vs baseline 0.0384 → 0.964× ≥ 0.95× threshold. **Gate 1 ✓.**
    - **V_n̄ now 0.13 / 0.18 / 0.28 / 0.38 / 0.22 / 0.03** across replans (was 0.85–2.51 in D1+D2). Controller is now picking low chronic load — physiologically correct rest/recovery for a pathological cold-start.
    - V_h̄ tightened to 2.16–3.36 (was 0.78–3.55).
    - V_c̄ tightened to −0.93 to +0.33 (was −0.33 to +1.07).
    - T-floor violation 66.15% — horizon artifact at T=2 (baseline 64.6%).
    - id-cov 9/9, wall-clock 23.1 min.
- **Plots:** `version_2/outputs/swat/experiments/run14_D1D2D3_T2_logit_bias_lambda0/`.

## [Run 13] - 2026-05-04 (cancelled — superseded)
**D1 + D2 — T=7 promotion test**
- **Changes:** D1 + D2 (V_c bound fix + controller n_substeps 4→10) at T=7 d, scenario `pathological`, λ_E=0.
- **Hypothesis:** D2's cumulative-integration-accuracy gain shows up over longer horizons; T=7 is where Stuart-Landau drift on T (τ_T=2 d) has time to lift T off zero, so any controller-quality difference becomes visible.
- **First attempt OOMed at stride 2** (controller-side, 41.3 GiB allocation request vs 32 GB ceiling) — D2's 2.5× substep work pushed memory past the limit at T=7's MPC rollout. Lowered the particle-halving threshold in `_particle_counts_for_horizon` from T≥14 to T≥7 (the post-D2 memory-load curve forces halving sooner). Re-launched with n_smc=256, n_pf=200, ctrl_n_smc=256, ctrl_n_inner=32.
- **Plots:** `version_2/outputs/swat/experiments/run13_D1D2_T7_v_c_bound_and_substeps_lambda0/` (when complete).

## [Run 12] - 2026-05-04
**D1 + D2 — V_c bound fix + controller n_substeps match (T=2 smoke test)**
- **Changes:** Layered D2 on D1. D2 = bench's `build_control_spec(...)` call now passes `n_substeps=10` (was 4) to match `_plant.py:78` and `estimation.py:194`. Controller's cost rollout now uses the same ODE discretisation as the plant.
- **Result:** **No measurable difference vs D1 alone at T=2.** mean T (MPC) = 0.0363, V_c̄ = 1.08 / 0.17 / −0.33 across replans — essentially identical to Run 10. T-floor violation 66.7% (horizon artifact, baseline 64.6%). Wall-clock 22.9 min (~8% slower than D1 alone, as expected from 2.5× substep work). Confirms D2 is the right structural change but its accuracy gain is below the noise floor at T=2 — promote to T=7 to see it.
- **Plots:** `version_2/outputs/swat/experiments/run12_D1D2_T2_v_c_bound_and_substeps_lambda0/`.

## [Run 10] - 2026-05-04
**D1 — V_c bound fix (T=2 smoke test)**
- **Changes:**
    - Single change: `control.py` now uses `V_c_max = V_C_MAX_HOURS = 3` (imported from `_dynamics`) instead of `V_C_BOUNDS[1] = 12` (from `_v_schedule`). Aligns the controller with `_dynamics.entrainment_quality()`, `estimation.FROZEN_PARAMS`, and `simulation.py`, all of which already used 3.
    - Motivation: Run 09 had V_c̄ swinging −11.57 to +7.22 across replans; for any |V_c| > 3 the dynamics' phase term is zero, so the controller had no gradient signal in 75% of its V_c parameter space and any stride with |V_c|>3 actively drove T → 0 via μ = −0.5.
    - Scenario `pathological`, λ_E=0, T=2 d, h=15min, n_substeps=4 (controller still mismatches plant; D2 next).
- **Result:** **V_c oscillation eliminated.**
    - Per-replan V_c̄ values: 1.07 / 0.15 / −0.33 — all well within ±3.
    - mean T (MPC) = 0.0363 vs baseline 0.0384 (within MC noise; 0.945× — gate is 0.95×).
    - T-floor violation: MPC 66.7%, **baseline also 64.6%** at T=2 cold-start (T₀=0, τ_T=2 d → T can't physically climb past T_floor=0.05 in 2 days). T-floor gate is a horizon artifact at T=2.
    - id-cov: 9/9 windows pass.
    - Wall-clock: 21.2 min on RTX 5090.
- **Plots:** `version_2/outputs/swat/experiments/run10_D1_T2_v_c_bound_fix_lambda0/` — `E5_full_mpc_swat_T2d_traces.png` (T trajectory + applied V_h/V_n/V_c), `E5_full_mpc_T2d_param_traces.png` (param posteriors), plus the standard latent/obs/entrainment panels.

## [Run 08] - 2026-05-03
**14-Day Convergence Validation (Synchronized Model)**
- **Changes:**
    - Run length extended to 14 days (105 strides).
    - Verified Plant/Filter synchronization (10x sub-stepped EM with independent noise).
    - Hardcoded controls: V_h=1, V_n=0, V_c=0.
- **Result:** **EXACT PARAMETER RECOVERY.**
    - The longer horizon allowed the filter to fully cross the Stuart-Landau bifurcation.
    - Parameter traces show high-fidelity convergence to truth values as the system settles in the healthy regime.
- **Plots:** `version_2/outputs/swat/experiments/run08_14d_synchronized/E5_full_mpc_T14d_param_traces.png`

## [Run 07] - 2026-05-03
**Phase I: Identification Fix (7-Day Synchronized Run)**
- **Changes:**
    - Synchronized all `FROZEN_PARAMS` (Plant vs Filter) across the codebase.
    - Fixed `tau_Z` (0.25h) and `tau_a` (10h) timescales in `_dynamics.py`.
    - Fixed `E_crit` schema bug (removed triplicate entries).
    - Aligned initial condition ($T_0=0$) for the filter.
- **Result:** **SUCCESSFUL IDENTIFICATION.**
    - Identified subset coverage reached **9/11** during the run.
    - Parameter traces now trend accurately toward truth values.
    - Confirmed that "Ideal Recovery" physics allows both latent state and parameter recovery.
- **Plots:** `version_2/outputs/swat/experiments/run07_7d_fix_alignment/E5_full_mpc_T7d_param_traces.png`

## [Run 06] - 2026-05-03
**Phase I: Physics Isolation (2-Day Ideal Recovery Test)**
- **Changes:**
    - Forced Ideal Recovery controls (V_h=1, V_n=0, V_c=0) for all time steps.
    - Scenario: Pathological (T=0, V_h=0, V_n=4, V_c=12).
    - Horizon: 2 days (test).
- **Result:** Successfully ran on GPU using `comfyenv`. Generated parameter trace plots. Mean T (MPC) = 0.0372 after 2 days (recovering from 0).

## [Run 04] - 2026-05-03
**G1-SWAT Reparametrization & Stabilization**
- **Changes:**
    - Implemented G1-reparametrization (replaced `mu_0` with `E_crit`).
    - Added observation offsets (`delta_HR`, `delta_s`).
    - Implemented 10x sub-stepping for numerical stability.
    - Fixed sub-step noise generation bug.
    - Set `lambda_E = 1.0` and MPC horizon = 3 days.
- **Result:** Mathematically stable, identifiable, but slow testosterone recovery.

## [Run 03] - 2026-05-02
**2-Day G1 Diagnostic**
- **Changes:** First test of G1-reparam and sub-stepping on short horizon.
- **Result:** Proved numerical stability (no filter crashes).

## [Run 02] - 2026-05-01
**7-Day LambdaE1 with Initial Condition Fix**
- **Source:** Pre-Gemini archive.
- **Result:** Used `lambda_E=1.0` and fixed initial T.

## [Run 01] - 2026-05-01
**7-Day LambdaE1 Baseline**
- **Source:** Pre-Gemini archive.
- **Result:** Standard `lambda_E=1.0` rollout.

## [Run 00] - 2026-05-01
**Original Pathological Baseline**
- **Source:** Pre-Gemini archive.
- **Result:** Pathological start, standard physics.
