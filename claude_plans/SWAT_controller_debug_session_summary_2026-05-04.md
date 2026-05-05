# SWAT controller debug session — summary and followup

> Followup to `SWAT_controller_debug_plan_2026-05-04_0753.md`. Written 2026-05-04 ~20:25, after the autonomous debugging session ended.

## What we started with

Run 09 (T=14, λ_E=0, pathological cold-start, full closed-loop SMC²-MPC) had failed acceptance gates: mean T 0.076 vs baseline 0.088 (0.86×, gate is 0.95×), T-floor violation 40%. Critically, the controller was picking V_c values that swung between −11.57 and +7.22 hours across replans — physically nonsensical schedules. Run 08 (same horizon, fixed test controls V_h=1, V_n=0, V_c=0) had passed cleanly with mean T 0.204 (2.3× baseline), so filter and plant were already validated. The problem was on the controller side.

## Static audit (Phase A1) — two concrete code-level mismatches

Reading `version_2/models/swat/control.py` end-to-end against the senior files (`_dynamics.py`, `_plant.py`, `estimation.py`, `simulation.py`, `_v_schedule.py`) surfaced two mismatches between the controller and the rest of the SWAT model.

### Finding 1 — V_c bound mismatch (the smoking gun)

- `_dynamics.py:81` — `V_C_MAX_HOURS = 3.0` ("clinical pathology threshold"). Used in `entrainment_quality()` (line 187) which clamps `|V_c|` to 3 and passes that into `cos(π · |V_c| / (2 · V_C_MAX_HOURS))` so the phase term hits exactly zero at |V_c|=3.
- `estimation.py:99` — uses `V_c_max = V_C_MAX_HOURS` (3 h).
- `simulation.py:31` — imports `V_C_MAX_HOURS`.
- **`control.py:91` (before fix)** — `V_c_max = V_C_BOUNDS[1] = 12` (from `_v_schedule`). Tanh transform produced V_c ∈ [−12, +12] hours.

**Mathematical consequence.** For any planned |V_c| > 3, `phase = cos(π/2) = 0` → `E_dyn = damp · amp_W · amp_Z · 0 = 0` → `μ = μ_E · (E − E_crit) = −0.5` → `dT/dt = −0.5 · T − 0.5 · T³` → **T actively collapses toward zero**. So the controller had a 75% V_c subspace where the cost was V_c-independent (zero gradient) AND any plan with sustained |V_c| > 3 actively destroyed T. That matches Run 09's V_c̄ flailing −11.57 ↔ +7.22 and the sub-baseline mean T.

### Finding 2 — n_substeps mismatch

- `_plant.py:78` — `n_substeps = 10` (sub-stepped Euler-Maruyama).
- `estimation.py:194` — `n_substeps = 10`.
- `bench_smc_full_mpc_swat.py:185` (before fix) — `build_control_spec(..., n_substeps=4, ...)`.

The controller was optimising under a different ODE discretisation than the plant executed. With outer `dt = 15 min ≈ τ_Z`, the controller's effective sub-step was 0.25·τ_Z (forward-Euler stable but coarse for accuracy on long horizons); the plant's was 0.1·τ_Z. Both stable, but the controller's "optimal" plan rolled out under a different forward model than the plant, so the optimum doesn't transfer cleanly.

### Hypothesis ranking that came out of the audit

- **H_VC** (V_c bound) — top priority. Concrete mismatch, mathematically explains the V_c flailing.
- **H_DT** (n_substeps) — second. Real discrepancy; smaller delta than H_VC.
- **H1** (no logit bias on V_h, V_n) — controller's prior centre at θ=0 was V_h=2, V_n=2.5 (mid chronic load), strictly worse than every scenario baseline. Controller had to fight uphill.
- **H2** (single shared σ_prior) — engine accepts only a scalar σ; per-variate would need engine change. Workaround: scale `raw_c` inside `schedule_from_theta`.
- **H4** (5 tempering levels for 24-dim θ) — `max_lambda_inc=0.5` was letting the engine satisfy ESS/cap in 5 steps. FSA used 10–20.
- **H3** (λ_E=0 disables shaping) — DEMOTED. Earlier Run 00 worked at similar settings; the gradient-cliff argument was speculative.

## Six fixes applied, each tested at T=2

Per the guiding principle (estimator + plant are the senior decision; controller is the new hire), all changes were scoped to `version_2/models/swat/control.py` and `version_2/tools/bench_smc_full_mpc_swat.py`. The senior files (`_dynamics.py`, `_plant.py`, `estimation.py`, `simulation.py`, `_v_schedule.py`) were not modified.

| Fix | Change | Where | Tested at T=2 |
|---|---|---|---|
| **D1** | `V_c_max` 12 → 3 (= `V_C_MAX_HOURS`) | `control.py:91` + import line | Run 10 |
| **D2** | controller `n_substeps` 4 → 10 | `bench_smc_full_mpc_swat.py:185` | Run 12 |
| **D3** | logit bias on V_h, V_n so θ=0 → set_A operating point (V_h=1, V_n=0.2, V_c=0) | `control.py:_make_three_schedules` | Run 14 |
| **D4** | scale `raw_c` by 1/3 inside `schedule_from_theta` (effective σ_eff=0.5 on V_c block) | `control.py:_make_three_schedules` | Run 15 |
| **D5** | controller `max_lambda_inc` 0.5 → 0.1 (force ≥10 tempering levels) | `bench_smc_full_mpc_swat.py:367` | Run 16 |
| **D6** | `SWAT_LAMBDA_E=1` (env var only — restores bench default cost shaping) | env | Run 17 |

### T=2 progression (mean T, baseline 0.0384)

| Run | Config | Mean T | V_n̄ range | V_c̄ range | Gate-1 |
|---|---|---|---|---|---|
| 09 (T=14) | none | 0.0761 | wild | −11.6 to +7.2 | ✗ (controller broken) |
| 10 | D1 | 0.0363 | 0.12–2.51 | −0.33 to +1.07 | 0.95× ✗ marginal |
| 12 | +D2 | 0.0363 | 0.11–2.53 | similar | 0.95× ✗ marginal |
| 14 | +D3 | 0.0370 | **0.03–0.38** | −0.93 to +0.33 | **0.96× ✓ first pass** |
| 15 | +D4 | 0.0372 | 0.02–0.38 | tighter | 0.97× ✓ |
| 16 | +D5 | 0.0372 | 0.05–0.19 | −0.16 to +0.28 | 0.97× ✓ (tempering now 10) |
| 17 | +D6 | 0.0372 | 0.04–0.12 | −0.14 to +0.15 | 0.97× ✓ |

T-floor violation stayed at 66.7% across all T=2 runs. Verified that this is a horizon artifact, not a controller issue: the constant-control baseline itself violated T-floor 64.6% at T=2 cold-start because T₀=0 and τ_T=2 d means T literally cannot climb above 0.05 in 2 days regardless of controls.

**Order of contribution:**
- **D1** was the structural fix. Without it the controller cannot be coherent at any horizon.
- **D3** was the qualitative win. Centring the prior cloud on the operating point made V_n̄ drop from 0.85–2.51 to 0.03–0.38 — the controller now consistently prescribes rest/recovery, the right thing for a pathological cold-start.
- **D2, D4, D5, D6** are each correct per the senior-files principle but their incremental T=2 gains are below the noise floor. Their real test is at longer horizons.

## Methodology pivot — controller-only test bench

Mid-session, Ajay pointed out that running the full closed-loop bench (filter + controller + plant) on every T=2 candidate was wasted compute when the bug was on the controller side. The controller's two inputs (`init_state` and `params`) are known in any synthetic test — truth params from `simulation.DEFAULT_PARAMS`, current state from `plant.state` after each `plant.advance(...)`. There's no need to run the SMC² filter just to recover them.

Built `version_2/tools/bench_controller_only_swat.py` (~280 LoC) — same controller invocation, same plant, same plot output, but skips the entire filter side. Methodology rationale in `claude_plans/controller_only_test_methodology.md`.

**Speedup honesty check.** Estimated 3–4× speedup; measured ~14% at T=7 (134 min controller-only vs ~155 min full closed-loop projection). The controller's tempered-SMC at 10 tempering levels × T=7 horizon × 10 sub-steps × 256 particles dominates the compute now; the filter is a smaller share than I'd estimated. The principle (controller-only is the cleaner test) holds, but the wall-clock saving is modest. Documented in the methodology doc.

## T=7 results (controller-only, all 6 fixes)

### Run 19 — truth τ_T = 48 h

mean T (MPC) = 0.0471 vs baseline 0.0446 → **1.06× improvement** (first MPC > baseline at any horizon, but at the noise floor). T-floor violation 56.7%. V_h̄ 1.22–2.75, V_n̄ 0.05–0.24, V_c̄ −0.38 to +0.70.

The 1.06× number is unimpressive because at truth τ_T=48 h, T's exponential growth time-constant is τ_T / μ ≈ 4 d. With T₀=0 and a 7-d horizon, T just barely starts climbing in the last day — most of the trajectory stays near the floor. The reference Run 08 plot shows a similar pattern: T sits near 0 for days 0–7 and only climbs past 0.5 around day 12. **At T=7 with truth τ_T, you can't see the controller's effect on T's climb, because T hasn't started climbing.**

### Run 21 — diagnostic τ_T override (12 h, applied to plant + controller identically)

Quartering τ_T from 48 h to 12 h shrinks the growth time-constant from ~4 d to ~1 d, so a 7-d run shows the same qualitative climb as Run 08's 14-d run did at truth τ_T.

**Result: mean T (MPC) = 0.2564 vs baseline 0.0314 → 8.2× improvement.** T-floor violation 21.4% (vs Run 08's 25.7% on the truth-τ_T fixed-control reference). V_h̄ 1.19–2.92, V_n̄ 0.01–0.14, V_c̄ −0.69 to +0.72. The controller is producing physiologically sensible plans across all 28 replans and T climbs visibly across days 2–7.

The override is recorded in the manifest as `tau_T_override_hours: 12` and printed with a `⚠️` warning at runtime. Truth value untouched in `_dynamics.TRUTH_PARAMS` and `simulation.DEFAULT_PARAMS`.

## What's now in the production code

Every code change was scoped to `control.py` and the bench, per the guiding principle. None of the senior model files were modified.

- `version_2/models/swat/control.py` — V_c_max imports `V_C_MAX_HOURS` from `_dynamics` (3 h, was 12); logit bias on V_h, V_n centred on set_A operating point; `raw_c` scaled by 1/3 (effective σ_eff for V_c block).
- `version_2/tools/bench_smc_full_mpc_swat.py` — controller `n_substeps=10` (was 4); controller `max_lambda_inc=0.1` (was 0.5); `_particle_counts_for_horizon` halves at T≥7 (was T≥14).
- `version_2/tools/bench_controller_only_swat.py` — new file, controller-only test bench.

## Lessons captured for the next session

- **Read the senior files first.** Two concrete mismatches surfaced from a careful 30-min static review: file:line evidence > pattern-matching > "I think it's wrong".
- **Filter is not needed to debug the controller.** Truth params + actual plant state make the closed-loop test ~unnecessary; promoted to `CLAUDE.md` and the methodology doc. Speedup is real but ~14%, not 3–4×.
- **τ_T override is a legitimate diagnostic move.** When the timescale of the system makes the result invisible at a feasible horizon, a temporary parameter scaling (applied identically on both sides of the test) gives a faster qualitative readout. Recorded in the manifest, warning at runtime, truth values untouched. Diagnostic only — do not commit results from override runs as production.
- **D1 was the structural fix, D3 was the qualitative one.** Most of the visible improvement came from those two. D2, D4, D5 are each correct per the principle but their gains were below noise at T=2.
- **Verify the gate is satisfiable.** T-floor < 5% at T=2 cold-start is unsatisfiable for *any* controller (baseline itself violates 64.6%). Don't argue from a horizon-artifact gate.

## Path to production

The qualitative validation under the τ_T override gives high confidence that the controller is now picking physiologically correct plans. The next step is the production sign-off run:

- **Full closed-loop SMC²-MPC bench** (`bench_smc_full_mpc_swat.py`, NOT the controller-only one).
- **Truth τ_T = 48 h** (no override; truth values from `_dynamics.TRUTH_PARAMS`).
- **All 6 controller fixes still applied** (in code; no rollback).
- **Pathological cold-start** (T₀=0, V_h=0, V_n=4, V_c=12).
- **λ_E = 1.0** (bench default; matches Run 17's "all fixes" config).
- **Horizon T=14 d** to allow T to fully cross the Stuart-Landau bifurcation, the way Run 08's reference does.

Acceptance gates revert to the standard set: mean T ≥ 0.95× baseline, T-floor ≤ 5%, id-cov ≥ subset / windows-with-obs, compute ≤ 4 h. The id-cov gate now applies again because the filter is back in the loop.

Memory: with controller `n_substeps=10` (D2) and T=14, the controller's MPC rollout at the previous halved counts (256/200/256/32) OOMed in the controller-only bench at stride 1. Need to either keep halving the controller particles or quarter both filter and controller at T≥14. To verify before launching the production run.
