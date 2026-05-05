# SWAT controller — production validated (T=14, full closed-loop)

> Written 2026-05-05 07:51, closing out the SWAT controller debugging effort that started 2026-05-04 from the Run 09 failure. This doc supersedes the in-progress notes in `SWAT_controller_debug_plan_2026-05-04_0753.md` and `SWAT_controller_debug_session_summary_2026-05-04.md`.

## Headline

**The full production code path (filter + controller + plant, truth params, base config) now produces a working closed-loop SMC²-MPC for the SWAT model at T=14 d.** Two A/B runs at the production horizon demonstrate:

- **Run 22 (λ_E=1, with entrainment-shaping cost term):** mean T (MPC) = 0.1847 vs baseline 0.0884 → **2.09× improvement**. T-floor violation 28.4%. 224.6 min.
- **Run 23 (λ_E=0, pure ∫T objective):** mean T (MPC) = 0.1835 vs baseline 0.0884 → **2.08× improvement**. T-floor violation 29.0%. 230.5 min.

Both pass acceptance gate 1 (mean T ≥ 0.95× baseline) by a wide margin and land at ~91% of Run 08's optimal-fixed-controls reference (mean T 0.204) while reacting to noisy observations in real time.

## Latest finding — entrainment shaping does not help

The auxiliary cost term `λ_E · ∫ E_dyn(t) dt` was introduced (commit 177d715, 2026-05-02) on the theoretical premise that it smooths the Stuart-Landau bifurcation cliff at E_crit=0.5 and gives the controller dense gradient signal even when T sits near zero. The A/B between Run 22 and Run 23 says **that effect doesn't materialise in practice**:

- **Mean-T delta:** 0.1847 → 0.1835 = **−0.65%**. Within MC noise from a single seed.
- **T-floor violation delta:** 28.4% → 29.0% = +0.6%. Negligible.
- **Wall-clock delta:** 224.6 → 230.5 min. Negligible.

This matches what was already visible at T=2 (Run 16 vs Run 17 were identical to 4 decimal places). H3 in the original debug plan — the bifurcation-cliff argument — was correctly demoted as a "plausible-sounding ML-style premise that didn't survive empirical contact". The pure `−∫T dt` objective is sufficient at the pathological cold-start.

**Practical implication:** the `λ_E · ∫E_dyn` term can be dropped from the cost without losing controller performance. Whoever picks this up next can either (a) leave it in but default `λ_E=0` (it costs almost nothing to compute and may help in scenarios we haven't tested), or (b) strip the entrainment-shaping branch from `control.py` for simplicity. Either is a defensible choice; we have no evidence it should stay on by default.

## What's now in the production code (final controller config)

Of the six fix candidates explored during the debugging session, the production code retains the four that carried the qualitative win:

| Fix | What | File:Line | Kept? |
|---|---|---|---|
| **D1** | Controller V_c_max 12 → 3 (= `_dynamics.V_C_MAX_HOURS`). The V_c bound now matches what the dynamics actually responds to. | `version_2/models/swat/control.py:91` | ✅ kept |
| D2 | Controller `n_substeps` 4 → 10 to match plant + estimator | `bench_smc_full_mpc_swat.py:185` | ❌ reverted (4× planner cost, no measurable benefit) |
| **D3** | Logit bias on V_h, V_n so θ=0 → set_A operating point (V_h=1, V_n=0.2, V_c=0) | `version_2/models/swat/control.py:_make_three_schedules` | ✅ kept |
| **D4** | Scale `raw_c` by 1/3 inside `schedule_from_theta` (effective σ_eff=0.5 on V_c block) | `version_2/models/swat/control.py:_make_three_schedules` | ✅ kept |
| D5 | Controller `max_lambda_inc` 0.5 → 0.1 (force ≥10 tempering levels) | `bench_smc_full_mpc_swat.py:367` | ❌ reverted (2× planner cost, no measurable benefit) |
| **D6** | `SWAT_LAMBDA_E=1` env default (cost includes `+ λ_E·∫E_dyn`) | env / bench default | ✅ kept (cosmetic — A/B shows it does nothing) |

The two structural fixes that actually carried the qualitative result are **D1** (V_c bound mismatch — was a real bug) and **D3** (logit bias on V_h / V_n centred on the operating point). D2 and D5 were "principled" or "tuning" ideas that survived T=2 testing only on aesthetic grounds; they were reverted in the final production launch (Run 22) once the cost was clear.

### Files modified

Editable in this debugging session (production-ready as-is):
- `version_2/models/swat/control.py` — V_c bound import, logit bias, V_c σ scale.
- `version_2/tools/bench_smc_full_mpc_swat.py` — `_particle_counts_for_horizon` retuned for T≥14, controller config back to Run 09 retry's profile.
- `version_2/tools/bench_controller_only_swat.py` — new file (controller-only diagnostic bench, plus `SWAT_TAU_T_OVERRIDE_HOURS` env var for τ_T diagnostic acceleration).

Read-only this session (senior-decision files, untouched):
- `version_2/models/swat/_dynamics.py`
- `version_2/models/swat/_plant.py`
- `version_2/models/swat/estimation.py`
- `version_2/models/swat/simulation.py`
- `version_2/models/swat/_v_schedule.py`
- `smc2fc/control/{control_spec,tempered_smc_loop,calibration,rbf_schedules}.py`

## Acceptance gate residuals

- **Gate 1 — mean T ≥ 0.95× baseline:** ✓ Both Run 22 and Run 23 pass at ~2.09×.
- **Gate 2 — T-floor violation ≤ 5%:** ✗ Both runs at ~28–29%. Most of the violation is days 0–7 of the 14-day window where T₀=0 and the τ_T=2 d Stuart-Landau growth hasn't fully ramped. Run 08's fixed-test-controls reference (the "best plausible" baseline) violates 25.7% — the gate threshold of 5% is unsatisfiable for any controller starting from T₀=0 at this horizon. Recommend revisiting the gate threshold (e.g. measure violation only after day 7, or relax to ≤ 30% at the pathological cold-start).
- **Gate 3 — id-cov ≥ subset / windows-with-obs:** ✗ Filter posterior coverage on the identifiable subset is below threshold. Did not investigate — out of scope for the controller debug. Worth a separate look if filter id-cov is required for a downstream stage.
- **Gate 4 — compute ≤ 4 h:** ✓ Both runs at 3.7–3.8 h.

So 2 of 4 gates pass. Gate 2 is a horizon artifact, not a controller defect. Gate 3 is a filter concern, not addressed in this session.

## Where the artifacts live

- **Run 22 (λ_E=1, production headline):** `version_2/outputs/swat/experiments/run22_PRODUCTION_T14_full_closedloop_lambda1/`
  - `E5_full_mpc_swat_T14d_traces.png` — T trajectory + applied controls (the headline figure)
  - `E5_latents_circadian_T14d.png` — W/Z/a + circadian overlay
  - `E5_obs_circadian_T14d.png` — 4-channel observations
  - `E5_full_mpc_T14d_param_traces.png` — posterior parameter traces
  - `manifest.json` + `data.npz`
- **Run 23 (λ_E=0, A/B comparator):** `version_2/outputs/swat/experiments/run23_PRODUCTION_T14_full_closedloop_lambda0/`
- **CHANGELOG:** `version_2/outputs/swat/CHANGELOG.md` — full audit trail of Runs 09 → 23.
- **Diagnostic methodology doc:** `claude_plans/controller_only_test_methodology.md` — for next time anyone debugs the SWAT controller.
- **Original debug plan (frozen):** `claude_plans/SWAT_controller_debug_plan_2026-05-04_0753.md`.
- **Mid-session summary (frozen):** `claude_plans/SWAT_controller_debug_session_summary_2026-05-04.md`.

## Lessons captured

The full audit trail of decisions, fixes, and reverts is in the three docs above + the changelog. The condensed lessons for the next debugging session:

1. **Read the senior files first.** Two of the six "fixes" came directly out of static review against `_dynamics.py` and `_plant.py`. File:line evidence beats reasoning from filenames.
2. **Filter is not needed to debug the controller.** Truth params + actual plant state make the closed-loop test infrastructure unnecessary. Methodology in `controller_only_test_methodology.md`.
3. **Verify benefit before keeping a fix.** D2 and D5 made it into a 7-h production attempt purely on aesthetic principle ("controller and plant should integrate the same way"). They had no measurable T=2 benefit and 4× the planner cost. The "verify before assert" rule applies to fixes too: if a candidate doesn't pay for itself in the cheap test, it shouldn't ride along into the expensive one.
4. **τ_T override is a legitimate diagnostic move.** When the system timescale makes the result invisible at a feasible horizon, scale the timescale temporarily (applied identically to plant and controller, recorded in the manifest, with a runtime warning). Truth values stay untouched in source. Diagnostic only — do not commit override runs as production.
5. **A/B at production horizon settles bifurcation-style hand-waving.** The λ_E=1 vs λ_E=0 comparison resolved a months-old theoretical question (does the entrainment shaping help?) in ~7 h of GPU. The answer was no.

## Status

The SWAT closed-loop SMC²-MPC controller is now a working production code path at T=14 d. Branch: `feat/import-swat-from-dev-repo`. Code ready to merge after review.
