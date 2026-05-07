# FSA v5 — Verification & Debugging Plan

## Context

Ajay imported the FSA v5 model (6D bimodal variable-dose extension of FSA v2's
3D state) into the smc2fc framework on branch `importing_FSA_version_5`. The
code lives at [version_3/models/fsa_v5/](version_3/models/fsa_v5/). It is **not
mature** and is expected to contain many bugs.

A previous agent ran overnight, made many claims (Stage 2 PASS, Stage 3 PASS,
1.67× speedup of a `soft_fast` variant), and produced a handoff Ajay considers
unreliable. The **only consistent finding** that survives is: full HMC
(`n_smc=256, num_mcmc_steps=10, hmc_num_leapfrog=16`) outperforms trimmed HMC.
One concrete bug was fixed (warm-up off-by-one at
[bench_controller_only_fsa_v5.py:458](version_3/tools/bench_controller_only_fsa_v5.py#L458),
commit `6241a9d`).

The intended outcome of this work: independently verify whether FSA v5
produces realistic closed-loop results, identify and fix bugs **only in
[version_3/](version_3/)**, and put a tabular experiment log in place so Ajay
has scannable oversight.

## Hard constraints (immutable)

1. **No edits outside [version_3/](version_3/).** [smc2fc/](smc2fc/) and
   [version_2/](version_2/) are off-limits. If a bug appears framework-side,
   escalate to Ajay rather than patch.
2. **No deletions in [claude_plans/](claude_plans/).** Audit trail.
3. **Validate in the dev repo before paying for SMC²-MPC in smc2fc.** The
   external sandbox at `/home/ajay/Repos/FSA_model_dev` (branch
   `claude/dev-sandbox-v4`) has dedicated open-loop simulation, stability,
   FIM and static optimisation tools that exercise the model in isolation.
   Use them to triage hypotheses cheaply — running smc2fc closed-loop just
   to see whether the plant is sane is wasteful and is what got the
   previous agent into trouble.
4. **Stage 2 (sim-est consistency) is already done.** Do not re-run from
   scratch. Only revisit if Stage 3 / Stage 4 surface filter-side bugs.
5. **Full HMC only.** `control_v5_fast.py` (the trimmed soft_fast variant) is
   quarantined. Do not run, do not measure, do not ship.
6. **No wall-clock estimates without measurement.** No "this will take 2h"
   without a timed basis from a real run.
7. **Long horizons (≥30 d) are the point**, not a problem to optimise away.
   Banister-overload mechanics only manifest at length.

---

## Confirmed code-level findings (verified by direct file reading)

### Bug 1 — `sigma_S` dict-literal collision (HIGH PRIORITY)

In [version_3/models/fsa_v5/simulation.py](version_3/models/fsa_v5/simulation.py),
the `DEFAULT_PARAMS` dict at lines 356–397 declares `'sigma_S'` twice:

- [line 377](version_3/models/fsa_v5/simulation.py#L377): `'sigma_S': 0.008` — the **state-noise** for Strength
- [line 394](version_3/models/fsa_v5/simulation.py#L394): `'sigma_S': 4.0` — the **stress-channel obs noise**

Python's dict-literal semantics silently keep the second. `DEFAULT_PARAMS_V5`
inherits from `DEFAULT_PARAMS` ([line 403](version_3/models/fsa_v5/simulation.py#L403)),
so any code that calls `diffusion_state_dep(y, params=DEFAULT_PARAMS_V5)` uses
**`sigma_S = 4.0`** — 500× the intended state-noise.

**`TRUTH_PARAMS_V5`** in
[_dynamics.py:127](version_3/models/fsa_v5/_dynamics.py#L127) is clean
(no obs-channel keys). So blast radius depends on which dict each tool uses.
Stage 0 must trace this.

**Fix:** rename one. The state-noise is the SDE-level parameter; the
stress-channel one is observational. Suggest the obs key becomes `sigma_S_obs`
(matching the `_obs` suffix convention already used elsewhere). Update the
stress-channel sampler in `simulation.py` (`gen_obs_stress`) to read
`sigma_S_obs`. **Audit all callers** of these dicts before renaming.

### Bug 2 — Particle-0 separatrix template (POSSIBLY ROOT-CAUSE OF 96% VIOLATION RATE)

In [control_v5.py:576](version_3/models/fsa_v5/control_v5.py#L576),
`_compute_cost_internals` collapses the SMC² particle ensemble to particle 0
when computing the separatrix `A_sep_per_bin`:

```
template = jax.tree_util.tree_map(lambda x: x[0], theta_stacked)
A_sep_per_bin = jax.vmap(find_sep_at)(Phi_schedule)   # shape (n_steps,)
```

This single (n_steps,) array is then broadcast to all particles via
`A_sep[None, :]` at lines [631](version_3/models/fsa_v5/control_v5.py#L631)
(hard) and [646](version_3/models/fsa_v5/control_v5.py#L646) (soft). The
separatrix depends on `mu_0, mu_B, mu_S, mu_F, mu_FF, eta, B_dec, S_dec,
mu_dec_B, mu_dec_S` — all of which vary across the SMC² particle population.

**Diagnose first, fix second.** The persistent 0.93–0.96 weighted_violation_rate
may or may not stem from this. Stage 3 will surface the actual mechanism by
logging the per-bin `A_sep` distribution from a real run.

### Bug 3 — Run-numbering race

[bench_controller_only_fsa_v5.py:137-139](version_3/tools/bench_controller_only_fsa_v5.py#L137-L139):

```
n = _next_run_number(exp_dir)
out_dir = exp_dir / f"run{n:02d}_{run_tag}"
out_dir.mkdir(exist_ok=True)
```

Two processes both read max+1, both call `mkdir(exist_ok=True)`, second
silently writes into first's directory. `bench_smc_filter_only_fsa_v5.py`
and `bench_smc_full_mpc_fsa_v5.py` likely share the helper (Stage 0 confirms).

**Fix:** atomic `mkdir(exist_ok=False)` retry loop with bounded attempts.
Apply the fix once in a shared helper (suggest
`version_3/tools/_run_dir.py`) and replace the duplicates in all three
benches.

### Already-fixed (verified)

Warm-up off-by-one at
[bench_controller_only_fsa_v5.py:458](version_3/tools/bench_controller_only_fsa_v5.py#L458)
is correctly `if k > 0 and k % replan_K == 0:`. Trace shows
`applied_phi_per_stride[0] = baseline_phi` is set correctly because
`current_schedule = np.array([baseline_phi])` is initialized at
[line 450](version_3/tools/bench_controller_only_fsa_v5.py#L450) and the
`else` branch at line 504 falls through to set `phi_this_stride =
current_schedule[0:1]` for k=0. **Basin-overlay start dot is therefore
correct after the fix.** No further action.

---

## Stage-by-stage execution

### Stage 0 — Static sanity (cheap, no GPU)

**Goal:** establish a clean baseline of what passes today.

**Actions:**
1. Run all four existing tests:
   ```
   PYTHONPATH=.:.. pytest version_3/tests/test_fsa_v5_smoke.py \
       version_3/tests/test_fsa_v5_bench_smoke.py \
       version_3/tests/test_obs_consistency_v5.py \
       version_3/tests/test_reconciliation_v5.py -v
   ```
2. Reproduce the `sigma_S` collision in a Python REPL:
   `from version_3.models.fsa_v5 import DEFAULT_PARAMS_V5; print(DEFAULT_PARAMS_V5['sigma_S'])`.
   Expect `4.0`.
3. Trace blast radius: grep `version_3/` for `DEFAULT_PARAMS_V5`,
   `DEFAULT_PARAMS`, `TRUTH_PARAMS_V5`. List which tools read which dict.
   Crucially: does the existing
   [bench_smc_filter_only_fsa_v5.py](version_3/tools/bench_smc_filter_only_fsa_v5.py)
   pass `DEFAULT_PARAMS_V5` to the plant for synthetic-data generation, or
   does it pass `TRUTH_PARAMS_V5`?
4. Apply Bug 1 fix: rename obs key to `sigma_S_obs`; update
   `gen_obs_stress` and any callers.
5. Apply Bug 3 fix: shared atomic-mkdir helper at
   `version_3/tools/_run_dir.py`; replace `_allocate_run_dir` in all three
   benches.
6. Re-run tests after the fixes; everything that previously passed must
   still pass.
7. Add a regression test
   `version_3/tests/test_fsa_v5_param_dict.py` asserting
   `DEFAULT_PARAMS_V5['sigma_S'] == 0.008` (state-noise) and
   `DEFAULT_PARAMS_V5['sigma_S_obs'] == 4.0` (obs-channel).

**Pass criteria:**
- 4/4 existing tests pass before and after fixes.
- New regression test passes.
- `DEFAULT_PARAMS_V5['sigma_S']` is 0.008 after the rename.

**Effort:** small.

### Stage 1 — Open-loop sanity in the FSA_model_dev sandbox (cheap, off-smc2fc)

**Goal:** confirm the FSA v5 plant produces qualitatively realistic
trajectories under known-truth parameters across the three regimes,
**without paying any smc2fc-side compute or risking smc2fc-side bugs
contaminating the diagnosis**. This replaces what previous notes called
"psim" — the dedicated psim repo is deprecated and the validation tooling
now lives in the FSA_model_dev sandbox.

**Sandbox repo:** `/home/ajay/Repos/FSA_model_dev` on branch
`claude/dev-sandbox-v4`. Per its README, the source files in
`models/fsa_high_res/` were the basis for the smc2fc port at
[version_3/models/fsa_v5/](version_3/models/fsa_v5/). The dev repo carries
its own simulator (`simulator/sde_solver_diffrax.py`) and validation
tools.

**Tools to use (in priority order):**

1. `examples/run_fsa_simulation.py` — open-loop SDE simulation with the
   v4/v5 model. Run at three scenarios (healthy / sedentary / overtrained).
   Confirm qualitative realism per the gates below.
2. `tools/stability_basins_v4.py` — bifurcation / basin diagram. Confirms
   the closed-island topology (only balanced (Phi_B≈Phi_S≈0.30) is healthy)
   that v5 was designed to encode.
3. `tools/static_optimization_scan_v4.py` — scan static schedules to check
   which (Phi_B, Phi_S) maximise A under the truth model. Confirms the
   model has the basin Stage 3 expects to find.
4. `tools/fim_analysis_v5.py` — Fisher-information identifiability check.
   Confirms which params are identifiable under realistic obs schedules
   (informs failure diagnosis for any future Stage 2 re-visit).

**Bug-1 (`sigma_S` collision) verification in the sandbox:**
- Inspect `models/fsa_high_res/simulation.py` in the dev repo for the same
  duplicate-key collision. If the bug is present in the source, escalate to
  Ajay — but **do not edit the dev repo**, only flag.
- Per hard constraint #1, fix only the smc2fc-side copy at
  [version_3/models/fsa_v5/simulation.py](version_3/models/fsa_v5/simulation.py).

**Three required scenarios** (matching
[bench_controller_only_fsa_v5.py:88–101](version_3/tools/bench_controller_only_fsa_v5.py#L88-L101)):

| Scenario     | Phi          | init                                  | T_days |
|--------------|--------------|---------------------------------------|--------|
| healthy      | (0.30, 0.30) | [0.5, 0.45, 0.2, 0.45, 0.06, 0.07]    | 14     |
| sedentary    | (0.0, 0.0)   | same                                  | 14     |
| overtrained  | (1.0, 1.0)   | same                                  | 14     |

**Pass criteria** (qualitative realism — eyeball + bounds check):
- Trajectories finite (no NaN); state bounds respected (B,S ∈ [0,1]; F,A,K ≥ 0).
- Healthy: B, S grow toward mid-island; A ≥ 0.5 mean; F bounded.
- Sedentary: A → 0 within 14d (Hill bites); B, S decay.
- Overtrained: F rises sharply; A collapses (-mu_F·F suppresses bifurcation).
- Static-scan and stability-basin tools confirm the closed-island topology.

**Logging:** even though this stage runs outside smc2fc, log each run as a
row in `version_3/outputs/fsa_v5/experiments_log.md` with
`Stage = 1`, `Artifact dir = (FSA_model_dev:<path>)`, and key
metric / pass-fail. Ajay's oversight requires one index regardless of
where the artifacts physically live.

**Stop on first failure.** If healthy fails in the sandbox, the bug is in
the canonical v5 dynamics shared between the two repos — escalate to Ajay
with a precise file:line pointer, do not paper over in version_3 alone.

**Effort:** small (read-only on the dev repo for the most part; a few
script invocations).

### Stage 2 — Sim-est consistency (already done — skip)

**Status: COMPLETE** per Ajay (2026-05-03). The synthetic-data → fit →
recover-params verification has already been performed and is not part
of this plan's scope.

**Re-visit only** if Stage 3 or Stage 4 surface a filter-side bug whose
root cause cannot be diagnosed from the closed-loop run alone (e.g.
identifiability collapse at long horizons that wasn't seen at the original
T tested). In that case:

- Re-run [bench_smc_filter_only_fsa_v5.py](version_3/tools/bench_smc_filter_only_fsa_v5.py)
  at the specific (T, scenario, regime) where Stage 3/4 failed.
- Diagnose using the prior-distribution / likelihood lens from
  [estimation.py](version_3/models/fsa_v5/estimation.py).
- If the failure looks framework-side (smc2fc), escalate to Ajay per
  hard constraint #1.

### Stage 3 — Controller-only (open-loop posterior, no filter loop) — **PRIMARY START**

**Goal:** confirm the chance-constrained controller plans schedules that
move state toward the healthy basin under each scenario.

**Pre-condition:** Stage 1 sandbox sanity passes (sedentary / healthy /
overtrained look right). Stage 2 is already verified.

**Cross-check option:** the dev repo's
`tools/optimize_dynamic_plan_v4.py` solves a related dynamic schedule
optimisation problem (no SMC² inner loop). If Stage 3 surfaces a wildly
unexpected applied schedule, run the dev-repo optimiser at the same truth
parameters as a sanity reference. **Read-only** in the dev repo.

**Config:** `--cost soft` only; full HMC (`256/10/16`); never `soft_fast`.

**Actions:**
```
python version_3/tools/bench_controller_only_fsa_v5.py --cost soft \
    --scenario healthy --T-days 14 --replan-K 2 \
    --run-tag stage3_ctrl_soft_healthy_T14_full_hmc
```
Then sedentary, then overtrained — each only after the previous passes.

**Pass criteria** (drawn from bench gates around lines 552–564):
- `applied_phi ∈ [0, 3]`.
- `∫A dt ≥ A_target` (default 2.0).
- `weighted_violation_rate ≤ alpha = 0.05`.
- Controller's `mean_theta` differs across consecutive replans (real adaptation).

**Mandatory diagnosis of the 0.93–0.96 violation rate (regardless of pass):**

The post-hoc evaluator at
[bench_controller_only_fsa_v5.py:542](version_3/tools/bench_controller_only_fsa_v5.py#L542)
calls `evaluate_chance_constrained_cost_hard` with a single particle holding
truth params. Inside, with healthy truth and small Phi, `_jax_find_A_sep`
should return `-inf` for most bins (mono-stable healthy → indicator False →
violation rate ≈ 0). Reaching 0.96 means either A_sep is finite/`+inf` for
~96% of bins, or there's a sign / index bug.

The bench already returns `posthoc['A_sep_per_bin']` ([control_v5.py:620](version_3/models/fsa_v5/control_v5.py#L620)).
Build a small standalone diagnostic script
`version_3/tools/diagnose_violation_rate.py` that loads a completed run's
`manifest.json + trajectory.npz`, reconstructs the inputs, and reports:
- Fraction of bins with `A_sep == -inf` (healthy, mono-stable)
- Fraction with finite (bistable annulus)
- Fraction with `+inf` (collapsed)

**Two fix paths depending on diagnosis:**

a) If most bins are `+inf` under truth+applied: the **internal** controller
   cost (using particle-0 template) saw mono-stable healthy and chose a
   schedule that the **post-hoc** truth check finds collapsed. This is Bug 2
   biting. Fix: vmap `_jax_find_A_sep` across the particle axis so
   `A_sep_per_bin` becomes shape `(n_particles, n_steps)`, then drop the
   `[None, :]` broadcast at lines 631 and 646 of
   [control_v5.py](version_3/models/fsa_v5/control_v5.py). Mathematically
   sound — separatrix is per-particle.

b) If most bins are `-inf` but indicator still 0.96: off-by-one or sign
   flip in the indicator computation. Read lines 631 and 646 carefully.

**Do not modify** [smc2fc/](smc2fc/). All controller code lives in
[version_3/](version_3/).

**Effort:** large (controller wall-clock dominant; multiple HMC runs).

### Stage 4 — Full closed loop (SMC² + MPC)

**Goal:** integrated filter+controller behaves like v2 at long horizons.

**Pre-condition:** Stage 3 healthy passes (or violation rate is fully
explained as evaluator-side, not controller-side).

**Actions:**
1. T=14d healthy smoke at full HMC config.
2. T=42d to exercise Banister-overload regime — match v2's
   [bench_smc_full_mpc_fsa.py](version_2/tools/bench_smc_full_mpc_fsa.py)
   known-good behaviour.
3. Sedentary and overtrained at T=14d, T=42d if budget allows.

**Pass criteria:**
- Long-horizon `mean_A_integral` rises across windows under healthy.
- F bounded; no runaway.
- Posterior coverage ≥ 30/37 across all windows.
- Filter `mean_theta` shows real movement, not flatline.
- Violation gate behaviour consistent with Stage 3 conclusion.

**Effort:** large.

---

## Experiment-logging methodology

**Single index file:** `version_3/outputs/fsa_v5/experiments_log.md` (Markdown
table, append-only). Per-run dirs at
`version_3/outputs/fsa_v5/experiments/runNN_<stage>_<scenario>_<tag>/`
keep `manifest.json`, `trajectory.npz`, plots — unchanged from existing
convention. The Markdown table is the human-scannable index over all of
those.

**Tabular columns:**

| Run ID | Date | Stage | Scenario | T_days | Cost | HMC cfg | Wall-clock (s) | Key metric | Pass/Fail | Notes | Artifact dir |

- `Run ID` — matches `_next_run_number`.
- `Stage` — 0 / 1 / 2 / 3 / 4.
- `Pass/Fail` — PASS / FAIL / INFO (Stage 0 is INFO) / PARTIAL.
- `HMC cfg` — `n_smc/num_mcmc_steps/hmc_num_leapfrog`, e.g. `256/10/16`.
- `Wall-clock` — measured. Never estimated.
- `Key metric` — stage-appropriate one-liner (e.g. Stage 3 → `viol=0.94,
  ∫A=11.4`).
- `Artifact dir` — relative path to per-run dir.

**Worked example row:**

```
| 24 | 2026-05-05 | 3 | healthy | 14 | soft | 256/10/16 | 4180 | viol=0.94, ∫A=11.4, target met | FAIL | viol gate fails; particle-0 template suspected | experiments/run24_ctrl_soft_healthy_T14_fullhmc/ |
```

**Append helper:** `version_3/tools/_append_experiment_log.py` reads
`manifest.json` from a run dir, appends a row to `experiments_log.md` under
`fcntl.flock`. Each bench calls it at end-of-run; manifest remains the
source of truth.

---

## Critical files (modify in [version_3/](version_3/) only)

- [version_3/models/fsa_v5/simulation.py](version_3/models/fsa_v5/simulation.py)
  — Bug 1 (sigma_S rename), gen_obs_stress callsite update.
- [version_3/models/fsa_v5/control_v5.py](version_3/models/fsa_v5/control_v5.py)
  — Bug 2 (vmap separatrix), pending Stage 3 diagnosis.
- [version_3/tools/bench_controller_only_fsa_v5.py](version_3/tools/bench_controller_only_fsa_v5.py)
  — Bug 3 fix via shared helper.
- [version_3/tools/bench_smc_filter_only_fsa_v5.py](version_3/tools/bench_smc_filter_only_fsa_v5.py)
  — Bug 3 fix via shared helper.
- [version_3/tools/bench_smc_full_mpc_fsa_v5.py](version_3/tools/bench_smc_full_mpc_fsa_v5.py)
  — Bug 3 fix via shared helper.

## New files to create (all under [version_3/](version_3/))

- `version_3/tools/_run_dir.py` — atomic-mkdir helper (Bug 3 fix).
- `version_3/tools/_append_experiment_log.py` — log-row append helper.
- `version_3/tools/diagnose_violation_rate.py` — Stage 3 diagnostic.
- `version_3/tests/test_fsa_v5_param_dict.py` — Stage 0 regression test.

(No `psim_fsa_v5.py`. Stage 1 uses the FSA_model_dev sandbox at
`/home/ajay/Repos/FSA_model_dev` instead of building a parallel tool in
smc2fc.)

## Existing functions / utilities to reuse

- `StepwisePlant.advance` from
  [version_3/models/fsa_v5/_plant.py](version_3/models/fsa_v5/_plant.py)
  — sub-daily binned plant rollout. Reuse in psim_fsa_v5.py.
- `verify_physics` at
  [simulation.py:339](version_3/models/fsa_v5/simulation.py#L339) — bounds
  checking. Reuse in psim Stage 1 pass criterion.
- `evaluate_chance_constrained_cost_hard` and
  `_jax_find_A_sep` at
  [control_v5.py:520](version_3/models/fsa_v5/control_v5.py#L520) — reuse
  for Stage 3 diagnostic script.
- The four existing tests in [version_3/tests/](version_3/tests/) — re-run
  them after every Stage 0 change.
- The v2 bench at
  [version_2/tools/bench_smc_full_mpc_fsa.py](version_2/tools/bench_smc_full_mpc_fsa.py)
  as **read-only** Stage 4 cross-reference.

## Recommended FIRST action

**Stage 0, action 1+2: run the four existing tests, then verify the
`sigma_S` collision in a REPL.**

Why first:
- Read-only, <5 minutes, no GPU.
- Disambiguates more than any other single action — if the collision is
  real and `DEFAULT_PARAMS_V5` is on the data-generation path, every prior
  Stage 3 / Stage 4 run consumed wrong-noise-model state trajectories.
  The 96% violation rate, the controller weirdness all become potentially
  explained.
- It is a precondition for trusting any downstream measurement.

**Followed immediately by Stage 1 in the FSA_model_dev sandbox** —
cheap, off-smc2fc, gives an independent sanity check on the v5 dynamics
before the heavy Stage 3 controller runs.

## Risks per stage

- **Stage 0:** existing tests pass but `sigma_S` bug wasn't caught — they
  don't exercise S state-noise. Add the regression test (in plan above).
- **Stage 1:** psim looks fine because it uses `TRUTH_PARAMS_V5` (clean),
  masking Bug 1 from the test path. The `--param-source` flag in
  `psim_fsa_v5.py` is specifically designed to expose this divergence.
- **Stage 2:** identifiability collapse on Strength-channel params under
  buggy `sigma_S`. After Bug 1 fix, regenerate synthetic data and re-run.
- **Stage 3:** violation rate stays at 0.96 after Bug 1 fix → Bug 2
  (particle-0 template) is the suspect. Diagnostic script clarifies which
  fix path applies.
- **Stage 4:** filter posterior collapses past day ~21 — would suggest
  framework-side issue (the v2 known-good has no VolumeLoad channel). If
  diagnosis points framework-side, **escalate to Ajay**. Do not patch.

## Effort summary (no wall-clock estimates)

| Stage | Effort | Notes                                              |
|-------|--------|----------------------------------------------------|
| 0     | small  | Static fixes + regression test                     |
| 1     | small  | FSA_model_dev sandbox runs, no smc2fc compute      |
| 2     | —      | **Skipped — already done; revisit only if needed** |
| 3     | large  | Primary controller-only verification + 0.96 fix    |
| 4     | large  | Long-horizon closed loop + Banister regime         |

## Verification (end-to-end)

The plan succeeds when:

1. Stage 0 — `version_3/tests/` 5/5 pass (4 existing + 1 new regression);
   `DEFAULT_PARAMS_V5['sigma_S'] == 0.008`; run-numbering race fixed.
2. Stage 1 — sandbox runs at `/home/ajay/Repos/FSA_model_dev` for
   healthy / sedentary / overtrained at T=14d hit the qualitative-realism
   gates; stability-basin and static-scan tools confirm closed-island
   topology. (No smc2fc-side compute spent on this stage.)
3. Stage 2 — **skipped** (already done); only re-visited if Stage 3/4
   surface a filter-side bug.
4. Stage 3 — `bench_controller_only_fsa_v5.py --cost soft` healthy passes
   `weighted_violation_rate ≤ 0.05` AND `∫A ≥ A_target`. The 0.96 mystery
   is resolved with a written cause (Bug 2 fix or evaluator fix).
5. Stage 4 — `bench_smc_full_mpc_fsa_v5.py` healthy at T=14d smokes, then
   T=42d shows long-horizon mean_A_integral growth and bounded F.
6. `version_3/outputs/fsa_v5/experiments_log.md` exists, has one row per
   run completed across all stages (including Stage 1 sandbox runs), and
   is human-scannable at a glance.
7. No edits made outside [version_3/](version_3/), including no edits to
   the FSA_model_dev sandbox.
