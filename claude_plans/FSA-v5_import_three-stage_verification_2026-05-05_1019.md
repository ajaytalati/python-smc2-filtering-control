# FSA-v5 import + three-stage verification

> Archived from plan mode: 2026-05-05 10:19.
> Updated: 2026-05-05 10:35 — Phase A done (8 model files + 3 test files copied, paths fixed, package imports clean, 12/12 tests pass). Plan annotated with import-time findings: __init__.py slim resolved to no-slim, test count 12 not 13, fp64 anti-pattern flagged in CHANGELOG. Ready to write bench drivers.
> Updated: 2026-05-05 10:50 — Cost-architecture finding (#6): `evaluate_chance_constrained_cost` is NOT JAX-jittable (uses scipy.brentq + Python loops). Resolution mirrors SWAT precedent: use `build_control_spec_v5` for the jittable controller cost; call `evaluate_chance_constrained_cost` post-hoc as a verification gate. Cadence + scenario decisions also recorded.
> Updated: 2026-05-05 11:05 — Ajay overrode TWO decisions after reading [`Advice_on_smc2fc_Port_Cost_Function_Architecture_2026-05-05_1055.md`](Advice_on_smc2fc_Port_Cost_Function_Architecture_2026-05-05_1055.md): (a) issue 7 (fp64 anti-pattern) — apply fp32 dtype optimization in version_3/models/fsa_v5/ mirroring FSA-v2's choices. Senior-files-immutability rule dropped for this case. (b) issue 6 (cost-architecture) — gradient-OT path is rejected as production cost. Use the upcoming JIT-friendly chance-constrained cost variants from the upstream rewrite, and TEST BOTH (`_soft` for HMC = Option B, `_hard` for pure-SMC² = Option C) in Stages 2 and 3. Open question on who does the upstream rewrite is in the plan.

## Context

Ajay is bringing the FSA-v5 model from the dev-sandbox repo (`github.com/ajaytalati/FSA_model_dev`, branch `claude/dev-sandbox-v4`, pinned SHA `d8f20c6`) into the `python-smc2-filtering-control` framework. The model has already been written, validated, and tested in the dev sandbox (13/13 tests green at d8f20c6). My job is **not to design or modify the model** — it's to wire the existing files into the framework's bench infrastructure so the closed-loop SMC²-MPC bench can drive it, then verify in three stages.

This is the same pattern already used for FSA-v2 (`fsa_high_res`) and SWAT, hardened in the SWAT debugging session (2026-05-04 → 2026-05-05).

Two senior planning docs guide this work:
- [`Importing_FSA_version_5_model_2026-05-05_0859.md`](Importing_FSA_version_5_model_2026-05-05_0859.md) — broad process plan
- [`Importing_FSA_version_5_model_specific_notes_2026-05-05_0935.md`](Importing_FSA_version_5_model_specific_notes_2026-05-05_0935.md) — model-specific notes from the FSA-v5 dev-sandbox author

I have read both end-to-end. My plan below consolidates them into an executable sequence, restates the three-stage verification per Ajay's explicit ordering, and flags two small discrepancies for Ajay's attention.

---

## Discrepancies to flag to Ajay before starting

These are NOT contradictions in the model design — they are small mismatches between the planning docs and what's actually on disk. None blocks the work; I want a yes/no from Ajay before proceeding.

1. **`bench_controller_only_swat.py` does not exist.** The broad plan says the FSA-v5 controller-only bench should "mirror `version_2/tools/bench_controller_only_swat.py` exactly". I verified — that file is **not** in [`../version_2/tools/`](../version_2/tools/). The actual SWAT benches present are `bench_smc_full_mpc_swat.py` only. The methodology doc [`controller_only_test_methodology.md`](controller_only_test_methodology.md) gives the recipe (~150 LoC lifted from the full bench), but no concrete SWAT controller-only bench exists yet to copy from. **Plan:** I'll build the FSA-v5 controller-only bench from the methodology recipe, using the FSA-v2 [`bench_smc_full_mpc_fsa.py`](../version_2/tools/bench_smc_full_mpc_fsa.py) `_build_phase2_control_spec` helper as the spec-construction template and stripping the filter loop.

2. **Model file list — `sim_plots.py` does not exist at d8f20c6, but `_phi_burst.py` does and is not mentioned in either planning doc.** The broad plan's layout includes `sim_plots.py # ← copied (if it exists in dev-sandbox-v4)`. Verified: it doesn't exist. What does exist (and is not in either plan's file list) is `_phi_burst.py` (sub-daily Φ expansion helper). **Plan:** I'll copy `_phi_burst.py` along with the rest. I'll skip `sim_plots.py`. Plotting will live inline in the bench drivers, matching the FSA-v2 pattern (FSA-v2 also has no `sim_plots.py`).

3. **`__init__.py` slim recommendation — resolved during execution.** Companion notes §6 says drop the v4 back-compat aliases; broad plan says do not modify imported files after path fixes. Resolved in favor of broad plan + junior-engineer stance: do not slim. See the path-fix section below.

4. **Test count — 12 not 13.** Both planning docs cite "13 tests" (4 + 6 + 2). Actual collected count from pytest is 12 (4 + 6 + 2 = 12). Arithmetic error in the docs; substance is correct. All 12 pass.

5. **fp64 anti-pattern found in copied files.** Per the broad plan's "Inspection step before writing benches", I grepped the copied files. Found explicit `dtype=jnp.float64` / `jnp.float64(...)` casts inside `lax.scan` step bodies and per-trial cost rollouts in `_plant.py`, `control.py`, `control_v5.py`, `_phi_burst.py`, and `simulation.py` (full list in [`../version_3/outputs/fsa_v5/CHANGELOG.md`](../version_3/outputs/fsa_v5/CHANGELOG.md) Run 00). Same pattern that costs SWAT 2-5× wall-clock. Per the broad plan, fix is upstream in `FSA_model_dev`; do NOT modify `version_3/`. Recorded in CHANGELOG, flagged to Ajay, will not block verification stages.

6. **Chance-constrained cost is NOT JAX-jittable at SHA `d8f20c6`** — and the upstream FSA_model_dev author has committed to a rewrite (see [`Advice_on_smc2fc_Port_Cost_Function_Architecture_2026-05-05_1055.md`](Advice_on_smc2fc_Port_Cost_Function_Architecture_2026-05-05_1055.md)). At d8f20c6: `evaluate_chance_constrained_cost` uses `scipy.optimize.brentq` in a Python `for k in range(n_steps)` loop and another Python `for i in range(n_particles)` loop. Cannot live inside HMC's leapfrog. The advice doc commits to a rewrite that exposes TWO JIT-friendly variants:
   - `evaluate_chance_constrained_cost_hard` — true indicator. For pure-SMC² importance weighting (no gradients).
   - `evaluate_chance_constrained_cost_soft` — sigmoid surrogate with `beta` knob. For HMC + temperature annealing (smooth gradients).

   **Ajay's call (2026-05-05 ~11:00):** test BOTH variants in Stages 2 and 3. Gradient-OT (`build_control_spec_v5`) is back-compat fallback only, NOT the production cost.

7. **fp64 anti-pattern fix — Ajay's override (2026-05-05 ~11:00):** drop the senior-files-immutability rule for the dtype anti-pattern. Apply fp32-inner / fp64-outer optimization to the imported model files in `version_3/models/fsa_v5/`, mirroring FSA-v2's working pattern at `version_2/models/fsa_high_res/_plant.py:_plant_em_step`. Specific lines listed in CHANGELOG.md "Run 00b — fp32 optimization applied" (to be added).

---

## Critical files

**Source (read-only, copied unchanged):**
- `/home/ajay/Repos/FSA_model_dev/` at SHA `d8f20c6` on branch `claude/dev-sandbox-v4`
- `models/fsa_high_res/` (8 files) → renamed to `version_3/models/fsa_v5/`
- `tests/test_fsa_v5_smoke.py`, `test_obs_consistency_v5.py`, `test_reconciliation_v5.py` (13 tests total) → `version_3/tests/`
- `LaTex_docs/FSA_version_5_technical_guide.tex` (read for cadence, identifiable subset, gates)

**Templates to copy/adapt (read, do not modify):**
- [`../version_2/tools/bench_smc_full_mpc_fsa.py`](../version_2/tools/bench_smc_full_mpc_fsa.py) — closer template for full MPC (FSA family)
- [`../version_2/tools/bench_smc_full_mpc_swat.py`](../version_2/tools/bench_smc_full_mpc_swat.py) — newer patterns (`_pop_scenario_from_argv`, param-trace auto-plot)
- [`../version_2/tools/bench_smc_rolling_window_fsa.py`](../version_2/tools/bench_smc_rolling_window_fsa.py) — Stage 1 (filter-only) template
- [`../version_2/models/fsa_high_res/_plant.py`](../version_2/models/fsa_high_res/_plant.py) — fp32-inner reference for plant integration
- [`../smc2fc/filtering/gk_dpf_v3_lite.py:132-160`](../smc2fc/filtering/gk_dpf_v3_lite.py#L132-L160) — fp32-cast-once reference (filter side; framework already does this for free)

**To be created:**
- `version_3/__init__.py`, `version_3/models/__init__.py`, `version_3/models/fsa_v5/` (8 model files copied + path-fixed)
- `version_3/tests/test_fsa_v5_smoke.py` + `test_obs_consistency_v5.py` + `test_reconciliation_v5.py` (copied verbatim, paths fixed) + `test_fsa_v5_bench_smoke.py` (new, bench-level)
- `version_3/tools/bench_smc_filter_only_fsa_v5.py` (Stage 1)
- `version_3/tools/bench_controller_only_fsa_v5.py` (Stage 2)
- `version_3/tools/bench_smc_full_mpc_fsa_v5.py` (Stage 3)
- `version_3/outputs/fsa_v5/.gitkeep` and `CHANGELOG.md`

---

## Phase A — Import (file copy + path fixes)

**Setup:**
```bash
cd ~/Repos/python-smc2-filtering-control
git checkout master && git pull
git checkout -b importing_FSA_version_5
conda activate comfyenv
```
(Branch off master, NOT off `feat/import-swat-from-dev-repo` — would muddy the diff.)

**Pin to SHA `d8f20c6`:**
```bash
mkdir -p /tmp/fsa_v5_import && cd /tmp/fsa_v5_import
git clone --depth 1 -b claude/dev-sandbox-v4 https://github.com/ajaytalati/FSA_model_dev
cd FSA_model_dev && git rev-parse HEAD     # confirm d8f20c6
```
Or use existing local clone at `/home/ajay/Repos/FSA_model_dev/` (verified clean, on `claude/dev-sandbox-v4`, SHA confirmed). Record the SHA in `version_3/outputs/fsa_v5/CHANGELOG.md` first entry.

**Copy 8 model files** from `models/fsa_high_res/` → `version_3/models/fsa_v5/`:
- `__init__.py`
- `_dynamics.py`
- `_phi_burst.py` ← not in planning docs, but present and load-bearing
- `_plant.py`
- `control.py` (legacy gradient-OT controller, kept as comparison baseline)
- `control_v5.py` ← v5 main novelty: `evaluate_chance_constrained_cost`, `find_A_sep_v5`
- `estimation.py`
- `simulation.py`

**Path fixes after copy** (grep `version_3/models/fsa_v5/` for these):
- `from models.fsa_high_res.<x>` → `from version_3.models.fsa_v5.<x>` (~12 occurrences across the files)
- Verify no straggling `from simulator.sde_model` lines (verified zero at d8f20c6, but check after copy in case of accidental edits)

**`__init__.py` re-export surface — RESOLVED to minimum-change.** Companion notes §6 recommends slimming to drop the `HIGH_RES_FSA_V4_*` aliases; broad plan says "do not modify any file in `version_3/models/fsa_v5/` after the verbatim copy + path fix". Resolved during execution in favor of the broad plan + junior-engineer stance: leave `__init__.py` at its full senior surface, do not slim. The benches don't need slimming — they import from specific modules anyway.

**Watch out for the three §9 footguns** while writing benches:
- `params['sigma_S']` returns the stress-OBS noise (~4.0), NOT the latent-S Jacobi diffusion scale (~0.008). Read diffusion scales from `_dynamics.SIGMA_*_FROZEN` directly.
- Plant `truth_params` must be `DEFAULT_PARAMS_V5`, not `TRUTH_PARAMS_V5` (the latter lacks the obs coefficient keys). Pattern: `StepwisePlant(truth_params=dict(DEFAULT_PARAMS_V5), ...)`.
- Direct `drift_jax(...)` calls (e.g. for a counterfactual baseline) need the Hill keys merged in: `p_full = {**_FROZEN_V5_DYNAMICS, **estimated_dict}`. The estimator's `propagate_fn_v5` and `evaluate_chance_constrained_cost` already do this internally.

---

## Phase B — Tests (copy verbatim + bench-level smoke)

**Copy three test files verbatim from FSA_model_dev tests/ → version_3/tests/**, changing only import paths (`from models.fsa_high_res.*` → `from version_3.models.fsa_v5.*`):
- `test_fsa_v5_smoke.py` (4 tests) — imports clean, plant pipeline, propagate_fn, chance-constrained cost
- `test_obs_consistency_v5.py` (6 tests) — pin each obs channel formula bit-equivalently sim ↔ estimator (catches D1/D2-class drift)
- `test_reconciliation_v5.py` (2 tests) — plant ↔ estimator drift parity (< 1e-10) + plant 1-bin smoke

If `tests/conftest.py` is copied across, drop or trim the `collect_ignore` list (it excludes legacy v2/v3 tests that won't exist in `version_3/tests/`).

**Write a thin bench-level smoke test** `version_3/tests/test_fsa_v5_bench_smoke.py`:
- imports the three new bench modules without running them
- verifies CLI parsing works for `--scenario`, `--step-minutes`
- verifies the bench can build a control spec from `DEFAULT_PARAMS_V5 + DEFAULT_INIT` without crashing
- asserts the manifest schema matches what `tools/plot_param_traces.py` expects

**Run all together:**
```bash
cd version_3 && PYTHONPATH=.:.. pytest tests/ -v
```
Target: 13 + small-handful = ~17 tests green before any bench run.

---

## Phase C — Bench drivers (three of them, one per verification stage)

All three drivers share the same JAX setup boilerplate from existing v2 benches:
```python
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(Path.home() / ".jax_compilation_cache"))
```
And all three parse `--step-minutes` via `_pop_step_minutes_from_argv()` BEFORE model imports (the `FSA_STEP_MINUTES` env var is read at module-import time to set `BINS_PER_DAY`).

### C1 — `version_3/tools/bench_smc_filter_only_fsa_v5.py` (Stage 1)
**Template:** [`../version_2/tools/bench_smc_rolling_window_fsa.py`](../version_2/tools/bench_smc_rolling_window_fsa.py).
**What it does:** drives `StepwisePlant.advance(STRIDE_BINS, fixed_Phi_daily)` for the full horizon under a healthy-operating-point fixed control schedule (no controller). Accumulates per-channel obs, runs the rolling-window SMC² filter with bridge handoff, writes posterior particles + summary. **Replaces v2 scalar Φ with v5 bimodal Φ = (Φ_B, Φ_S)** — fixed-control schedule is `(n_days, 2)`.

### C-pre — Two prerequisite work items before benches

#### Pre-1. Apply fp32 dtype optimization to `version_3/models/fsa_v5/`

**Ajay's override (2026-05-05 ~11:00):** drop the senior-files-immutability rule for the dtype anti-pattern. Optimize the dtype usage in the imported model files, mirroring FSA-v2's choices (FSA-v2 behaves well, so its fp32-inner / fp64-outer split is the proven pattern).

Lines to change (target: fp32 inside `lax.scan` step bodies and per-trial cost rollouts; keep fp64 for accumulators, log-weights, posteriors, host-side IO):
- `_plant.py:135` — `noise = jax.random.normal(sub, (6,), dtype=jnp.float32)` (was fp64) inside the EM step body
- `_plant.py:276-285` — cast `p_jax`, `sigma_jax`, `Phi_jax`, `y0_jax`, and `dt` to fp32 BEFORE the scan; the scan body runs fp32. Final state cast back to fp64 on output (line 289 already does this)
- `control.py:150,192` — controller cost-rollout noise to fp32
- `control.py:200,206` — `Phi_const`, `Phi_zero` to fp32
- `_phi_burst.py:56,89` — `h` arange and `daily_phi` cast inputs to fp32
- `simulation.py:229` — `p_jax = {k: jnp.float32(v) for k, v in params.items()}` IF this dict feeds an inner SDE loop (verify by reading context)

LEAVE fp64:
- `control.py:141` — accumulator `init_carry` (these are accumulators, MUST stay fp64)
- `_plant.py:104-106` — `EPS_*` (one-time, doesn't matter)
- All log-likelihood / log-weight / posterior accumulation in `estimation.py`

Reference template: [`../version_2/models/fsa_high_res/_plant.py`](../version_2/models/fsa_high_res/_plant.py) `_plant_em_step` (FSA-v2's working fp32-inner pattern).

The plan still records the original fp64-anti-pattern finding in CHANGELOG; it just adds an entry "Run 00b — fp32 optimization applied" so the dtype change is auditable.

#### Pre-2. Upstream rewrite of `control_v5.py` in `FSA_model_dev`

Per the advice doc, the rewrite produces:
- `_jax_bisection` helper (~25 lines, pure-JAX `lax.while_loop`)
- `_jax_find_A_sep` helper (drop-in JAX replacement for the scipy `find_A_sep_v5`)
- `evaluate_chance_constrained_cost_hard` (`@jax.jit`, indicator variant)
- `evaluate_chance_constrained_cost_soft` (`@jax.jit`, sigmoid variant with `beta`, `scale` args)
- Back-compat alias `evaluate_chance_constrained_cost` → `_hard` so existing imports keep working
- 4 new tests in `tests/test_fsa_v5_smoke.py`: `_jits`, `_jits` (soft), `_grad_finite`, `_soft_to_hard_limit`

**Open question for Ajay:** who does the upstream rewrite? Three options (asking before doing):
- **Option α:** I do it now in `~/Repos/FSA_model_dev/` on `claude/dev-sandbox-v4`, push, re-pin to the new SHA, re-copy `control_v5.py` and the new tests into `version_3/`. Adds ~1-2 hours to Phase A.
- **Option β:** the FSA-author / a separate dev-sandbox-Claude does it. I wait. Re-pin once it lands.
- **Option γ:** I do a "shim" version IN-PLACE in `version_3/models/fsa_v5/control_v5.py` (violates senior-files but pragmatic), get Stage 2/3 running, and the upstream rewrite is reconciled later when it lands.

I'll wait for Ajay's choice before doing either. My recommendation: **α**, because the user said I do "all the coding and technical work and github administration" and α keeps the audit trail clean.

### C2 — `version_3/tools/bench_controller_only_fsa_v5.py` (Stage 2)
**Template:** the methodology doc recipe (no concrete SWAT controller-only bench exists to copy). Lift the `_build_phase2_control_spec` helper structure from [`../version_2/tools/bench_smc_full_mpc_fsa.py`](../version_2/tools/bench_smc_full_mpc_fsa.py) and the controller-only loop pattern from [`controller_only_test_methodology.md`](controller_only_test_methodology.md):
```python
plant = StepwisePlant(truth_params=dict(DEFAULT_PARAMS_V5), ...)
for k in range(n_strides):
    plant.advance(STRIDE_BINS, applied_Phi)
    if k % replan_K == 0:
        spec = build_control_spec(init_state=plant.state.copy(),
                                  params=DEFAULT_PARAMS_V5,
                                  cost_fn=evaluate_chance_constrained_cost, ...)
        result = run_tempered_smc_loop_native(spec, ctrl_cfg, key)
        applied_Phi = decode_schedule(result.mean_schedule)  # shape (n_days, 2)
```
No filter, no obs sampling. Truth params come from `DEFAULT_PARAMS_V5`; current state from `plant.state.copy()` after each advance.

### C3 — `version_3/tools/bench_smc_full_mpc_fsa_v5.py` (Stage 3)
**Templates:** [`../version_2/tools/bench_smc_full_mpc_fsa.py`](../version_2/tools/bench_smc_full_mpc_fsa.py) (closer family) and [`../version_2/tools/bench_smc_full_mpc_swat.py`](../version_2/tools/bench_smc_full_mpc_swat.py) (newer patterns: `_pop_scenario_from_argv`, `_particle_counts_for_horizon`, param-trace auto-plot).

**TWO-VARIANT CONTROLLER COST (per Ajay's "test BOTH" instruction).** Stages 2 and 3 each run TWICE per scenario:
- **Variant B (soft / HMC).** `--cost soft --beta 50.0`: uses `evaluate_chance_constrained_cost_soft` as the smc2fc tempered-SMC cost_fn. Smooth, JAX-jittable, HMC inside leapfrog works. The Lagrangian wrapper combines the dict into a scalar: `cost = λ_Φ · mean_effort − mean_A_integral + λ_chance · weighted_violation_rate_soft`.
- **Variant C (hard / pure SMC²).** `--cost hard`: uses `evaluate_chance_constrained_cost_hard` for the outer-loop importance weighting; HMC inside is skipped or replaced with random-walk MH to avoid taking gradients through the indicator. **NOTE:** wiring this into smc2fc may require either `num_mcmc_steps=0` in `SMCControlConfig` or a non-HMC mutation kernel selector. I'll verify when wiring; if it requires touching `smc2fc/`, I will flag and ask before modifying framework code.

The bench saves both runs side-by-side under `version_3/outputs/fsa_v5/runNN_<scenario>_<variant>/`. Comparison plots (B-vs-C schedule overlays, B-vs-C A-trajectory overlays, gate-pass tables) go in a sibling `runNN_<scenario>_compare/` folder.

The CLI also accepts `--cost gradient_ot` for back-compat sanity checks against `build_control_spec_v5`, but it is NOT the default and is NOT shipped as production.
**Structural changes from FSA-v2** (companion notes §4):
- Latent state: 3D → 6D `[B, S, F, A, K_FB, K_FS]`
- Control: scalar Φ → bimodal `Φ = (Φ_B, Φ_S)`, RBF schedule output dim 2
- Obs channels: 4 → 5 (added VolumeLoad)
- `Phi_daily` shape: `(n_days,)` → `(n_days, 2)` — do NOT rely on the plant's auto-promote
- Cost functional: `evaluate_chance_constrained_cost(...)` returns a dict, not a scalar — combine into a single SMC² scalar score per technical guide §4 (chance-constraint re-weighting on `weighted_violation_rate`, score from `mean_A_integral`) or §4.5 (Lagrangian relaxation if gradients needed)
- Posterior-mean init for next window: 6 entries
- Diagnostic plots: 6 latent + 5 obs panels

**Cadence (stride / window / replan):** decide from FSA-v5 LaTeX technical guide's timescale analysis. **Do not blindly copy SWAT's 3-h stride / 1-d window / 6-h replan or FSA-v2's 1-d window / 12-h stride / 24-h replan** — read the guide, match the fastest dynamic the controller needs to react to, write the choice into `CHANGELOG.md`.

**Identifiable subset / acceptance gates:** read from the v5 technical guide's identifiability and acceptance-gate sections. Do NOT copy SWAT's gates (`mean_T ≥ 0.95×baseline`, `T-floor ≤ 5%`) — those are SWAT-specific.

### fp32-inner / fp64-outer convention (REQUIRED for bench-side hot loops)

Per [`../CLAUDE.md`](../CLAUDE.md) "GPU dtype convention":
- Cast to fp32 inside hot inner loops: particles, drifts, diffusions, noise, parameter dict, time scalars, schedule arrays during SDE rollout. Cast back to caller dtype on output.
- Never cast down: SMC log-weights, ESS, log-likelihood sums, posterior particles, mass-matrix Cholesky, parameter posteriors.

The framework's filter inner loop already does this ([`../smc2fc/filtering/gk_dpf_v3_lite.py:132-160`](../smc2fc/filtering/gk_dpf_v3_lite.py#L132-L160) and [`:253-265`](../smc2fc/filtering/gk_dpf_v3_lite.py#L253-L265)). My bench-side cost rollout (the `cost_fn`'s `vmap`'d trial inside `lax.scan`) must do the same — pattern in CLAUDE.md.

**Inspection step before writing benches:** grep the just-copied `version_3/models/fsa_v5/{_plant,_dynamics,control,control_v5}.py` for `jnp.float64` / `dtype=jnp.float64` inside inner loops. If found, **flag it to Ajay** — fix is upstream in `FSA_model_dev`, not in the imported copy (senior-files principle). My bench code does the cast-once-to-fp32 regardless.

---

## Three-stage verification (per Ajay's explicit instruction, in order)

These run AFTER all three benches are written and the smoke test (`version_3/tests/`) passes. Each stage has a clear success criterion. Do not skip stages or run out of order.

### Stage 1 — Filter + plant only (controller NOT run)
**Goal:** recover ground-truth model parameters from synthetic obs.
**Driver:** `version_3/tools/bench_smc_filter_only_fsa_v5.py`.
**Procedure:** plant runs under fixed healthy-operating-point bimodal `Φ = (Φ_B, Φ_S)` schedule for the horizon → emits 5-channel obs → rolling-window SMC² filter ingests obs → posterior over (params, latent state).
**Success:** posterior 90% CI covers truth on the **identifiable subset** (read from v5 technical guide; typically obs-side params + slow dynamics params identifiable at the configured horizon).
**Why first:** if the filter can't recover truth from clean synthetic data under fixed controls, no controller work matters. Bugs in `_dynamics`, `_plant`, `estimation`, or obs samplers all surface here. Cheaper than debugging inside a closed-loop run.
**Compute:** ~30-60 min on RTX 5090 at FSA-v2-equivalent T=14 d (likely longer for FSA-v5 due to 6D state and 5 channels — I'll hedge wall-clock estimates rather than predict).

### Stage 2 — Controller only (filter NOT run) — RUN TWICE per scenario (Variant B and Variant C)
**Goal:** sensible qualitative controller behavior under truth params + actual plant state, AND a side-by-side comparison of the two v5-correct cost variants (soft vs hard).
**Driver:** `version_3/tools/bench_controller_only_fsa_v5.py`.
**Procedure:** truth params from `DEFAULT_PARAMS_V5`; latent state from `plant.state.copy()` after each `plant.advance()`; controller plans, plant advances, repeat. No filter, no obs sampling. Run once with `--cost soft` (Variant B), once with `--cost hard` (Variant C). Save both to disk; run a comparison report.
**Success:** all of —
- applied schedule stays inside physical bounds (no controller flailing — see SWAT Run 09 failure mode for the antipattern)
- mean of cost-relevant state at end-of-horizon ≥ constant-baseline reference under truth params
- controller adapts to plant state changes (replans differ meaningfully across windows)
- exact thresholds set from v5 technical guide, **not copied from SWAT**

**Why second:** isolates intrinsic controller bugs (RBF transforms, cost composition, prior structure, integrator settings, tempering schedule) without paying the filter's per-stride cost. ~3-4× faster iteration than Stage 3 full closed-loop. Reuse for any controller-side debugging.
**Compute:** roughly Stage 3 minus the filter (~30% saving).

### Stage 3 — Full closed-loop SMC²-MPC — RUN TWICE per scenario (Variant B and Variant C)
**Goal:** end-to-end pipeline verification AND side-by-side comparison of the two v5-correct cost variants under filter-uncertainty conditions (which is where they may diverge most).
**Driver:** `version_3/tools/bench_smc_full_mpc_fsa_v5.py`.
**Procedure:** filter posterior every window → controller plans from posterior-mean every K windows → plant advances under applied controls → next window's filter. Run once with `--cost soft` (Variant B), once with `--cost hard` (Variant C). Compare gates side-by-side.
**Success:** all four standard gates —
1. mean of cost-relevant state ≥ α × baseline (α from v5 spec)
2. constraint-violation gate (FSA-v5 analog of T-floor / F-cap; the chance-constrained cost evaluator returns `weighted_violation_rate` directly — check it's < α)
3. filter id-cov: posterior 90% CI covers truth on the identifiable subset for at least N of M windows
4. compute ≤ 4 h on RTX 5090

**Failure-mode triage:**
- Stage 3 fails, Stages 1+2 pass → integration bug. Check posterior → controller wiring (`_build_fsa_v5_control_spec`), filter posterior summary extraction, bridge handoff config.
- Stage 3 fails AND Stage 1 or 2 fails → fix the upstream stage first.

**Compute:** ~2-4 h on RTX 5090 at canonical horizon. Hedged.

---

## Pre-stage smoke
Before any of the three stages: `pytest version_3/tests/test_fsa_v5_bench_smoke.py -v`. Catches typo-level mistakes in bench code before they waste GPU time.

---

## Things I will NOT do (senior-files principle)

- Modify any file in `version_3/models/fsa_v5/` after the verbatim copy + path fix
- Modify `smc2fc/` framework code (escalate to Ajay if I find a real framework bug)
- Modify `version_2/` (separate work; FSA-v5 is a fresh subtree)
- Delete or rewrite docs in `claude_plans/` (audit trail)
- Commit `version_3/outputs/fsa_v5/*.png` / `*.npz` / `*.json` (excluded by `.gitignore`; only `CHANGELOG.md` is tracked)
- Bundle unrelated framework changes into the FSA-v5 PR (the SWAT-session anti-pattern explicitly called out in the broad plan's "respect" section)
- Keep compute-heavy controller knobs in production after they show no measurable benefit at cheap test (the SWAT D2/D5 fiasco)
- State confident wall-clock estimates without verification

---

## When done

1. Three bench drivers in `version_3/tools/` produce sensible plots and pass their stage gates
2. Test suite green (~17 tests)
3. `version_3/outputs/fsa_v5/CHANGELOG.md` has at least one production-horizon Stage 3 run logged with the SHA pin recorded
4. Session-summary doc written into `claude_plans/` (similar shape to `SWAT_controller_production_validated_2026-05-05_0751.md`) covering: what was imported, what bench was written, what gates pass, any open issues
5. No read-only files modified (model files, smc2fc/, version_2/)
6. This plan archived into `claude_plans/` with the `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` line and updates kept in sync per CLAUDE.md plan-archive policy
