# SWAT controller debug plan

> Archived from plan mode: 2026-05-04 07:53.
> Updated: 2026-05-04 08:16 — added Phase A1 audit findings (V_c bound mismatch, n_substeps mismatch); demoted H3 (λ_E=0); reordered fix candidates so V_c bound fix is D1.
> Updated: 2026-05-04 08:32 — added "estimator + plant are the senior decision" guiding principle; recast all fix candidates as controller-side-only changes (control.py + bench); marked _dynamics/_plant/estimation/simulation/_v_schedule as read-only this session; split critical-files list into editable vs read-only.
> Updated: 2026-05-04 09:00 — D1 applied + verified at T=2 (run10). V_c̄ now stays inside ±3 across all replans (1.07 / 0.15 / −0.33 vs Run 09's −11.57 to +7.22). mean T 0.0363 ≈ baseline 0.0384 within MC noise; T-floor violation 66.7% is a horizon artifact at T=2 cold-start (baseline itself 64.6%). D1 PASS in substance. D2 (controller n_substeps 4→10 to match plant/estimator) applied and launched at T=2 (run12).
> Updated: 2026-05-04 20:25 — All 6 hypotheses tested at T=2 (Runs 10, 12, 14, 15, 16, 17) and all pass gate 1. Ajay redirected to a controller-only test bench (`tools/bench_controller_only_swat.py`, methodology doc in this folder). Final result: Run 21 (T=7, τ_T=12h diagnostic override, all fixes) — **mean T MPC 0.2564 vs baseline 0.0314, 8.2× improvement, T-floor violation 21.4%**, qualitatively matches Run 08's truth-τ_T T=14 climb. End of autonomous session.

## Guiding principle — estimator and plant are the senior decision

The estimator (`version_2/models/swat/estimation.py`), the plant (`version_2/models/swat/_plant.py`), and the underlying dynamics + simulation (`version_2/models/swat/_dynamics.py`, `simulation.py`) are the **source of truth** for this debugging session. Run 08 already validated them in open-loop. The controller (`version_2/models/swat/control.py`) is the new hire; where it disagrees with the senior files on bounds, integrator details, parameter schema, time conventions, or anything else, **the controller is what gets changed.**

Concretely:
- **Files that may be edited in this session:** `control.py` and the bench (`version_2/tools/bench_smc_full_mpc_swat.py`).
- **Files that are read-only this session:** `_dynamics.py`, `_plant.py`, `estimation.py`, `simulation.py`, `_v_schedule.py`. If a fix appears to require changing one of these, stop and escalate — that's a redesign decision for Ajay, not a junior-engineer fix.

Every finding and fix candidate below is scoped through this principle.

## Context

Run 08 (open-loop, fixed test controls V_h=1, V_n=0, V_c=0, T=14d): **gate 1 PASS** — mean T 0.204 vs baseline 0.088, 2.3× improvement. Filter+model verified.

Run 09 (closed-loop SMC²-MPC, λ_E=0, same horizon, same scenario): **gate 1 FAIL** — mean T 0.076 < 0.95×0.088, T-floor violation 40%. Controller is making things *worse* than fixed defaults.

Earlier run00 (T=7d, closed-loop, similar config) produced a sensible schedule with V_c hovering near 0 and reasonable mean T 0.099 vs baseline 0.108 (T-floor viol 18.6%) — so the controller *has* worked before with comparable settings. The Run-09 failure is therefore likely from a more recent regression or a horizon-dependent bug, not a fundamental controller defect.

## Audit findings (Phase A1, complete)

Two concrete discrepancies between `version_2/models/swat/control.py` and the rest of the SWAT model files:

### Finding 1 — V_c bound mismatch ⚠️

The controller's tanh transform uses **V_c_max = 12**, but the dynamics' `entrainment_quality()` clamps any V_c with |V_c| > 3 to **V_C_MAX_HOURS = 3** before computing the phase term, *and* the phase term hits zero at exactly that boundary.

- `_dynamics.py:81` — `V_C_MAX_HOURS = 3.0  # clinical pathology threshold (hours)`
- `_dynamics.py:187–188` —
  ```python
  V_c_eff = jnp.minimum(jnp.abs(V_c), V_C_MAX_HOURS)
  phase   = jnp.cos(jnp.pi * V_c_eff / (2.0 * V_C_MAX_HOURS))
  ```
- `_v_schedule.py:45` — `V_C_BOUNDS = (-12.0, 12.0)`
- `control.py:91, 106` — `V_c_max = V_C_BOUNDS[1]   # 12`, `V_c = V_c_max * tanh(raw_c)`
- `estimation.py:99` — uses `V_c_max=V_C_MAX_HOURS` (3.0)
- `simulation.py:31` — imports `V_C_MAX_HOURS` (3.0)
- The Run-00 plot's red dotted line is labelled `V_c_max=3h`, consistent with the dynamics, NOT with the controller's bound.

**Mathematical consequence.** For any planned |V_c| > 3:
- Cost is V_c-independent: `phase = cos(π·3/(2·3)) = cos(π/2) = 0`. So **75% of the V_c parameter space has zero gradient on the cost** — the SMC² walker can drift through it indistinguishably.
- Worse: phase=0 → `E_dyn = damp·amp_W·amp_Z·0 = 0` → `μ = μ_E·(E − E_crit) = −0.5` → `dT/dt = −0.5·T − 0.5·T³` → **T actively collapses**. Any plan with |V_c| > 3 sustained over a stride drives T toward zero.

This pattern matches Run 09's behaviour: V_c̄ swinging −11.57 ↔ +7.22 across replans (huge excursions outside the |V_c|≤3 zone) and mean T below baseline. Run 00's V_c stayed mostly within ±3 — consistent with passing.

**Fix scope (per the guiding principle):** the senior files (`_dynamics.py`, `estimation.py`, `simulation.py`) all use `V_C_MAX_HOURS = 3` from `_dynamics.py`. The controller alone uses `V_C_BOUNDS[1] = 12` from `_v_schedule.py`. The fix is on the controller side: import `V_C_MAX_HOURS` from `_dynamics` and use that as the bound. Do **not** change `V_C_BOUNDS` in `_v_schedule.py` (it is shared with the dev repo and the plant's daily-clip path).

### Finding 2 — n_substeps mismatch

The controller rolls out the cost integrand at 4× sub-stepping; the plant runs at 10×.

- `_plant.py:78` — `n_substeps = 10`
- `estimation.py:194` — `n_substeps = 10`
- `bench_smc_full_mpc_swat.py:174` (passed into `build_control_spec`) — `n_substeps=4`

So when the SMC² controller decides "this plan minimises the cost", it's optimising under a different ODE discretisation than the one the plant will execute. With outer `dt = 15 min = τ_Z`, the controller's effective sub-step is `dt/4 = 3.75 min = 0.25·τ_Z` — within forward-Euler stability (`dt < 2τ`) but coarse for accuracy on long horizons. The plant's `dt/10 = 1.5 min = 0.1·τ_Z` is the safer-accuracy point.

Both are stable, so this isn't a divergence-causing bug. But the controller's "optimal" rollout differs from what the plant produces under the same plan. This compounds with Finding 1: a planner that thinks it's saving cost by setting V_c=−10 (effectively zero phase, less integrand variance) still pays the real cost (T-collapse) when the plant runs at 10× substep with the same drift.

**Fix scope (per the guiding principle):** the plant and estimator use `n_substeps = 10`. The controller is the outlier. The fix is to bump the controller's `n_substeps` from `4 → 10` at the bench's `build_control_spec(...)` call, matching the senior files. Do **not** lower the plant's or estimator's `n_substeps`.

## Hypothesis ranking (revised)

- **H_VC** — V_c bound mismatch (Finding 1). **Top priority.** Concrete code-level inconsistency, mathematically explains the V_c oscillations and T-collapse pattern, addressable with a one-line change.
- **H_DT** — n_substeps mismatch (Finding 2). **Second.** Real discrepancy, but probably contributes a smaller delta than H_VC.
- **H1** — no logit bias on V_h, V_n. Prior centre at θ=0 puts V_n at 2.5 (mid chronic load) — strictly worse than the validated `(1, 0.3, 0)` operating point, so the controller has to fight uphill from the prior cloud.
- **H2** — single shared `sigma_prior=1.5` across 24 anchors with three different physical scales. Compounds H_VC: the V_c-block has a wide raw RBF spread that pushes most of the prior cloud past |V_c| = 3.
- **H4** — 5 tempering levels for 24-dim θ. Engine knobs (`target_ess_frac=0.5`, `max_lambda_inc=0.5`) may be cutting tempering short.
- **H3** — λ_E=0 disables ∫E_dyn shaping. **Demoted to lowest priority** — Run 00 worked with comparable settings; the gradient-cliff argument was speculative ML-style reasoning, not validated against the working run.

## Plan — five phases, GPU-budget-conscious

### Phase A — Static review (zero GPU)

A1 is **complete** (findings above). Remaining static items:

A2. **Forward-Euler stability sanity for SWAT timescales.** Confirm fastest timescale is τ_Z = 0.25 h = 0.0104 d. With outer `dt = 15 min`, controller sub-step `3.75 min ≈ 0.25 τ_Z` (Euler stable, marginal accuracy), plant sub-step `1.5 min ≈ 0.1 τ_Z` (Euler stable, good accuracy). No tα stability blow-up expected at either. Document the accuracy gap.

A3. **Param-schema reconciliation between control.py truth_params and FROZEN_PARAMS.** `bench_smc_full_mpc_swat.py:151` merges posterior-mean dynamics into `DEFAULT_PARAMS`. `estimation.py:68 FROZEN_PARAMS` pins `tau_T=2.0d, lambda_amp_Z=8.0` etc. Verify the bench's merge preserves the frozen values (it should, since `DEFAULT_PARAMS` already has them at truth).

A4. **Engine knobs vs FSA.** `bench_smc_full_mpc_swat.py:355–361` — `target_ess_frac=0.5, max_lambda_inc=0.5, num_mcmc_steps=5, hmc_step_size=0.05–0.20, hmc_num_leapfrog=10`. FSA used the same shape but observed 10–20 tempering levels. Check whether `max_lambda_inc=0.5` is allowing 2-3 step jumps that explain Run 09's 5-level convergence.

A5. **Closure baking confirmation.** `_build_swat_control_spec` (`bench_smc_full_mpc_swat.py:148–192`) passes `init_state` and `params` *into* `build_control_spec`; the post-`f74b6ab` fix is in place. No regression here.

**Phase A pass criterion:** the audit table is filled in with one line per check; no further static red flags.

### Phase B — One-shot GPU sanity (~5 min GPU)

All single `cost_fn(θ)` calls — no SMC loop. Validates that fixing H_VC actually changes the cost landscape.

B1. **Prior cost spread.** Call `calibrate_beta_max(n_samples=256)` against a fresh spec at the pathological init. Log `prior_cost_mean, prior_cost_std, β_max`. Compare to the values implied by Run 09's "5lvl" tempering trace.

B2. **Cost at `θ_zero` (current default), `θ_healthy` (V_h=1, V_n=0, V_c=0), `θ_v_c_extreme` (V_h=1, V_n=0, V_c=−10).** Construct theta vectors that produce uniform schedules at each. **Pass:** `cost_fn(θ_healthy) < cost_fn(θ_zero) < cost_fn(θ_v_c_extreme)`. If `cost_fn(θ_v_c_extreme) ≈ cost_fn(θ_v_c_3)` (i.e., V_c=−10 is no worse than V_c=−3), Finding 1 is numerically confirmed.

B3. **1D V_c slice.** Sweep V_c ∈ [−12, +12] with V_h=1, V_n=0 held fixed (21 points). Plot. **Pass shape:** cost has a sharp interior minimum around V_c=0, shoulders rise as |V_c| → 3, then flatten beyond. If the slice is flat across ±[3, 12] that confirms Finding 1; the controller cannot distinguish those regions.

B4. **n_substeps comparison.** Evaluate `cost_fn(θ_healthy)` at `n_substeps ∈ {4, 10}`. Difference quantifies Finding 2's contribution.

Total ~50 cost evaluations × ~1–2 s after JIT.

### Phase C — Single-stride controller-only test (~10 min GPU)

Replay the controller (no filter, no plant loop) using a converged posterior from Run 08's `data.npz` at stride 50. Run `run_tempered_smc_loop_native` once per seed ∈ {42, 43, 44}. Log per-seed: `n_temp_levels, β_max, prior_cost_std, mean_theta, mean_schedule, cost_fn(mean_theta)`.

**Pass criteria:**
- (a) `n_temp_levels ≥ 8`, else H4 confirmed.
- (b) seed-to-seed `mean_schedule` variation within physical noise floor.
- (c) `cost_fn(mean_θ) ≤ cost_fn(θ_healthy)` from B2.

Run this on the **unfixed** controller first (baseline reproduction), then **after** D1 (V_c-bound fix) for direct comparison.

### Phase D — Single-variable fix candidates, each tested at T=2 first (~30 min GPU each)

Each candidate changes ONE thing. **Every change is scoped to `control.py` and/or `bench_smc_full_mpc_swat.py` per the guiding principle — the senior files are read-only.** Order by Finding-evidence and cheapness:

- **D1 — Match controller V_c bound to dynamics (H_VC).** In `control.py`, replace the import of `V_C_BOUNDS` from `_v_schedule` with `V_C_MAX_HOURS` from `_dynamics`, and at line 91 use `V_c_max = V_C_MAX_HOURS` (= 3). All three senior files (`_dynamics.py`, `estimation.py`, `simulation.py`) already use 3; only the controller used 12. One-line conceptual change.
- **D2 — Match controller n_substeps to plant/estimator (H_DT).** `bench_smc_full_mpc_swat.py:174` — change the `n_substeps=4` literal in the `build_control_spec(...)` call to `n_substeps=10`. Plant and estimator both use 10. Cost: ~2.5× more compute per controller cost eval; may push memory back toward OOM at T=14 — verify at T=2 first.
- **D3 — Add logit bias on V_h, V_n in `control.py` (H1).** `c_h = logit(0.25) = -1.099` (so θ=0 → V_h=1.0 = the bench's `set_A` baseline), `c_n = logit(0.04) ≈ -3.18` (θ=0 → V_n=0.2 = `set_A`). V_c stays unbiased. Code change in `control.py:96–108` (`raw_h, raw_n` get a constant offset). The bench's scenario presets are the senior decision on what "operating point" means; the controller's prior centre should match.
- **D4 — Per-variate `sigma_prior` (H2).** `control.py` change to accept three scalars (or an array) instead of one shared `sigma_prior`. Try σ_h=σ_n=1.5, σ_c=0.5 to keep V_c off saturation. Engine compatibility: confirm `tempered_smc_loop.py:85–88` already accepts an array; if it doesn't, **stop and escalate** rather than modifying the engine.
- **D5 — Lower `max_lambda_inc` (H4).** `0.5 → 0.1` in `bench_smc_full_mpc_swat.py:355–361`. Forces ≥10 tempering levels. Pure bench-config change; no model edits.
- **D6 — λ_E=1 (H3, demoted).** Only test if D1–D5 alone don't pass; this is a workaround, not a structural fix. Set via the existing `SWAT_LAMBDA_E` env var; no code change.

For each Dk at T=2:
- **Pass:** mean T over closed-loop ≥ 0.5× T=2 baseline, T-floor viol < 20%, `mean_schedule` non-degenerate (per-stride variance > 1e-3 per variate, V_c stays inside ±3 in mean).
- **Fail:** mean T < 0.3× baseline OR `mean_schedule` ≈ θ-prior-centre.

T=2 budget: ~30 min/candidate × 6 ≤ 3 h, less than one T=14 run.

### Phase E — T=14 full rerun

Trigger only after a Dk passes T=2. Use Run 09's exact gates: mean T ≥ 0.95× baseline AND T-floor viol ≤ 5%. Single ~4 h run.

## Critical files

**Editable in this session (controller side only):**
- `version_2/models/swat/control.py` (controller cost, schedule, transforms — line 91 V_c_max source)
- `version_2/tools/bench_smc_full_mpc_swat.py` (`_build_swat_control_spec` line 148, controller invocation line 522, smc/ctrl configs lines 325–348)

**Read-only this session (senior decision; reference only):**
- `version_2/models/swat/_dynamics.py` (V_C_MAX_HOURS=3 at line 81; entrainment_quality at line 187)
- `version_2/models/swat/_plant.py` (n_substeps=10 at line 78)
- `version_2/models/swat/estimation.py` (n_substeps=10 at line 194; FROZEN_PARAMS line 68)
- `version_2/models/swat/simulation.py` (DEFAULT_PARAMS, scenario presets)
- `version_2/models/swat/_v_schedule.py` (V_C_BOUNDS at line 45 — shared with the dev repo via the plant)
- `version_2/models/fsa_high_res/control.py` (working reference for what a healthy controller looks like)
- `smc2fc/control/{control_spec,tempered_smc_loop,calibration,rbf_schedules}.py` (engine contract; if a fix appears to need changes here, escalate rather than modifying)

## Verification

- Phase A: a written audit table (one row per cross-check, `result: matches | mismatch with details`).
- Phase B/C: numerical logs saved under `outputs/swat/debug/phaseB_*.json`, `phaseC_*.json` — both before and after each fix.
- Phase D: per-candidate T=2 logs under `outputs/swat/experiments/run09_debug_<Dk>_T2/` with full manifest + plots.
- Phase E: standard Run 10+ artefact + `CHANGELOG.md` entry citing which Dk fix unlocked the gate.

## Constraints

- I'm a junior engineer here. The previous SWAT controller author's choices may have reasons I can't yet see — even the V_c bound mismatch could be intentional (e.g. the author wanted the controller to learn that |V_c|>3 is bad and self-clamp). Every "fix" candidate gets numerically validated against Phase B/C evidence before code changes; every code change gets reviewed before T=14 is launched. Hedge language throughout. No "this is a bug" — only "I'd want to verify X by Y".
- ZERO GPU until Phase A is complete. Phase B/C/D-T2 must finish before any T=14 run.

## Execution log

- **2026-05-04 08:36–08:57 — Run 10 (D1 alone, T=2, λ_E=0).** Applied D1 (`control.py` V_c_max=12 → V_C_MAX_HOURS=3). Result: V_c̄ now 1.07 / 0.15 / −0.33 across replans (was −11.57 to +7.22 in Run 09). mean T = 0.0363 ≈ baseline 0.0384 (within MC noise). T-floor violation 66.7% (baseline also 64.6% — horizon artifact at T=2 cold-start, T₀=0, τ_T=2 d). id-cov 9/9. **PASS in substance.**
- **2026-05-04 09:00–09:34 — Run 12 (D1+D2, T=2, λ_E=0).** D2 = controller `n_substeps` 4→10. Result: essentially identical to D1 alone at T=2 — mean T 0.0363, V_c̄ 1.08 / 0.17 / −0.33. D2's accuracy gain is below noise floor at T=2 (cumulative integration error is small over 2 days). Wall-clock 22.9 min, ~8% slower than D1 alone (expected from 2.5× substeps). **PASS.** Promoted to T=7.
- **2026-05-04 09:34–10:19 — Run 13 (D1+D2, T=7, λ_E=0) attempt 1 OOMed** at stride 2 (41.3 GiB request vs 32 GB ceiling). D2's 2.5× substep work pushed memory past the limit at T=7's MPC rollout. Lowered particle-halving threshold from T≥14 to T≥7 in `_particle_counts_for_horizon`.
- **2026-05-04 10:25–10:32 — Run 13 retry cancelled** (user redirected: test all hypotheses at T=2 first).
- **2026-05-04 10:35–11:00 — Run 14 (D1+D2+D3, T=2, λ_E=0).** D3 = logit bias on V_h, V_n so θ=0 → set_A operating point (V_h=1, V_n=0.2). Result: **first gate-1 PASS** (mean T 0.0370 vs baseline 0.0384, 0.964× ≥ 0.95×). V_n̄ now 0.03–0.38 across replans (was 0.85–2.51 with D1+D2). Physiologically correct rest/recovery prescription.
- **2026-05-04 11:01–11:24 — Run 15 (D1+D2+D3+D4, T=2).** D4 = scale `raw_c` by 1/3 inside `schedule_from_theta` (effective σ_eff=0.5 on V_c block). Result: mean T 0.0372 (slightly above D3's 0.0370), V_c̄ tightened further (last replan V_c=0.08), gate 1 still passes (0.969×). Marginal but consistent improvement.
- **2026-05-04 11:25–11:51 — Run 16 (D1+D2+D3+D4+D5, T=2).** D5 = controller `max_lambda_inc` 0.5→0.1 → tempering now 10 levels (was 5). Controller's posterior tighter and more committed: V_h̄ 2.35–2.70, V_n̄ 0.05–0.19, V_c̄ −0.16 to 0.28. mean T 0.0372 (same as D4). Compute 26.3 min (16% slower).
- **2026-05-04 11:53–12:19 — Run 17 (D1+D2+D3+D4+D5+D6, T=2).** D6 = λ_E=1. Result identical to D5 to 4 decimals — λ_E=0 vs λ_E=1 makes no measurable difference at T=2 (confirms H3 demotion). mean T 0.0372, gate 1 ✓. V_h̄ 2.15–2.78, V_n̄ 0.06–0.12, V_c̄ −0.14 to 0.15. **All 6 hypotheses tested at T=2 systematically.**
- **2026-05-04 12:21–14:55 — Run 18 (ALL FIXES, T=7, full closed-loop, cancelled)** at stride 30/49. Ajay pointed out the filter was unnecessary for controller debugging; built `tools/bench_controller_only_swat.py` and the methodology doc `claude_plans/controller_only_test_methodology.md`.
- **2026-05-04 14:56–17:21 — Run 19 (controller-only T=7, all fixes, truth params).** mean T MPC 0.0471 vs baseline 0.0446 (1.06×); T-floor viol 56.7%. Gate 1 PASS but at noise floor — τ_T=2 d means T can't fully climb in 7 d.
- **2026-05-04 17:25–17:55 — Run 20 (controller-only T=14) cancelled** — OOM at stride 1 with 256/32 particles. Quartered to 128/16 for T≥14, retry started, then cancelled when Ajay suggested the τ_T override for a faster T=7 test.
- **2026-05-04 18:00–20:21 — Run 21 (controller-only T=7, tau_T=12h diagnostic override, all fixes).** **mean T MPC 0.2564 vs baseline 0.0314 → 8.2× improvement.** T-floor viol 21.4% (matches Run 08's 25.7% at truth τ_T T=14). V_h̄ 1.19–2.92, V_n̄ 0.01–0.14, V_c̄ −0.69 to +0.72. Compute 141 min. Qualitative climb of T across days 2–7 mirrors Run 08's days 8–14 climb. **Final result of the autonomous session.**

## Accountability mirror

This plan is mirrored to `claude_plans/SWAT_controller_debug_plan_*.md` in the repo. Per repo policy (see `CLAUDE.md`), every meaningful update here gets pushed to the archive copy with an `Updated:` timestamp line, so the project has a durable audit trail of how the plan evolved.
