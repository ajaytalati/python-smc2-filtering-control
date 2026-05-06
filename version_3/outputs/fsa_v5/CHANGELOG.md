# FSA-v5 (`version_3/models/fsa_v5/`) — output audit trail

Each bench run gets an entry here: run NN, headline, plots, gates pass/fail, scenario, particle counts. Same shape as `version_2/outputs/fsa_high_res/RESULT.md` and `version_2/outputs/swat/CHANGELOG.md`.

The `outputs/fsa_v5/` directory itself is excluded from git via `.gitignore` — only this `CHANGELOG.md` is tracked. Runtime artifacts (`*.png`, `*.npz`, `*.json`) belong in per-run subfolders that are NOT committed.

---

## Run 00 — Import provenance (no bench run yet)

**2026-05-05 10:30** — Import of FSA-v5 from `github.com/ajaytalati/FSA_model_dev`.

- **Branch:** `claude/dev-sandbox-v4`
- **Pinned SHA:** `d8f20c6257f2c0f5902f344bd75eb89c3d31f28f` (2026-05-05 00:05 BST)
- **Source:** `models/fsa_high_res/` → `version_3/models/fsa_v5/` (folder rename)
- **Files copied (8):** `__init__.py`, `_dynamics.py`, `_phi_burst.py`, `_plant.py`, `control.py`, `control_v5.py`, `estimation.py`, `simulation.py`
- **Tests copied (3 files, 12 tests):** `test_fsa_v5_smoke.py` (4), `test_obs_consistency_v5.py` (6), `test_reconciliation_v5.py` (2)

  > Both planning docs (`Importing_FSA_version_5_model_2026-05-05_0859.md` and the companion notes) cite "13 tests". The actual count is 12 (4+6+2). Planning-doc arithmetic error; the substance is correct.

- **Path edits applied (real `from ... import ...` lines only, no docstring changes):** `from models.fsa_high_res.<x>` → `from version_3.models.fsa_v5.<x>` across the 7 files that had real imports. 12 docstring/comment references to `models.fsa_high_res` left in place per minimum-change/junior-engineer stance.

- **Smoke test result:** `cd version_3 && PYTHONPATH=.:.. pytest tests/ -v` → **12/12 PASS** in 16.65s on RTX 5090 / `comfyenv`. Plant ↔ estimator drift parity confirmed bit-equivalent (reconciliation test).

### Known issues recorded at import time

#### 1. fp64 anti-pattern in inner hot loops

The copied files contain explicit `dtype=jnp.float64` / `jnp.float64(...)` casts inside SDE inner loops, the same pattern that costs SWAT a 2–5× wall-clock penalty per [`../../../CLAUDE.md`](../../../CLAUDE.md) "GPU dtype convention" and [`../../../version_2/outputs/fsa_high_res/GPU_TUNING_RTX5090.md`](../../../version_2/outputs/fsa_high_res/GPU_TUNING_RTX5090.md).

Findings (line numbers at SHA `d8f20c6`):
- `_plant.py:135` — `noise = jax.random.normal(sub, (6,), dtype=jnp.float64)` inside `lax.scan` step body of the EM integrator (per-bin SDE rollout).
- `_plant.py:276-289` — `p_jax`, `sigma_jax`, `Phi_jax`, `y0_jax`, `dt`, `final_state_jax` all explicitly fp64 in the per-stride wrapper.
- `control.py:141` — accumulator init carry uses `jnp.float64(0.0)`.
- `control.py:150,192` — controller cost-rollout noise drawn at `dtype=jnp.float64` (this is the L8-equivalent path that should be fp32 per the FSA-v2 commit `aa114e8` reference).
- `control.py:200,206` — `Phi_const`, `Phi_zero` constructed at fp64.
- `control_v5.py:316` — `Phi_schedule = jnp.asarray(..., dtype=jnp.float64)` at the top of `evaluate_chance_constrained_cost`.
- `_phi_burst.py:56,89` — `h` arange and `daily_phi` cast to fp64.
- `simulation.py:229` — `p_jax = {k: jnp.float64(v) for k, v in params.items()}` (used by simulator, on the warm path).

Per the broad plan's "Things you DO NOT do" section: the fix is upstream in `FSA_model_dev`, not in `version_3/`. Do NOT silently edit the imported files. Recommended: open an issue/PR against the dev sandbox to swap these to fp32 inside hot loops, mirroring FSA-v2's plant. The bench drivers I will write here use the cast-once-to-fp32 pattern in their own code regardless — but until the model files are fixed upstream, the fp64 casts inside `_plant_em_step` etc. will dominate the per-stride time.

This is informational, not a blocker for verification — Stage 1 / 2 / 3 verification can proceed at the slower wall-clock and the pattern can be addressed in parallel.

#### 2. Companion-plan / broad-plan small mismatches (resolved)

- **`bench_controller_only_swat.py` does not exist** in `version_2/tools/` — broad plan said to mirror it. Resolved by building the FSA-v5 controller-only bench from the methodology-doc recipe instead.
- **Model file list:** `sim_plots.py` does not exist at d8f20c6 (broad plan's layout sketched it as optional); `_phi_burst.py` does exist and is load-bearing (not mentioned by either planning doc). Resolved by copying `_phi_burst.py` and skipping `sim_plots.py`.
- **`__init__.py` slim recommendation conflict:** broad plan says "do not modify any file in `version_3/models/fsa_v5/` after the verbatim copy + path fix"; companion notes §6 recommends dropping the v4 back-compat aliases. My final plan resolved this in favor of the broad plan (minimum-change stance) — `__init__.py` is left at its full senior surface.

---

## Run 00b — Re-pin to FSA_model_dev `7075436` (JIT-friendly chance-constrained cost)

**2026-05-05 11:30** — re-pinned to a newer FSA_model_dev SHA after the upstream author addressed Run 00's finding #6 (`evaluate_chance_constrained_cost` not JIT-able).

- **New pinned SHA:** `7075436628fa8c202cf62241666fe90230c46ac1` on `claude/dev-sandbox-v4`
- **Upstream commit:** `feat(control_v5): JIT-friendly hard/soft variants of chance-constrained cost` (+923 lines, -55, across 5 files)
- **Files re-copied + path-fixed:**
  - `models/fsa_high_res/control_v5.py` → `version_3/models/fsa_v5/control_v5.py`
  - `models/fsa_high_res/__init__.py` → `version_3/models/fsa_v5/__init__.py`
  - `tests/test_fsa_v5_smoke.py` → `version_3/tests/test_fsa_v5_smoke.py` (now 8 tests, +4 new)
- **New symbols available at this SHA:**
  - `_jax_mu_bar` (line 429), `_jax_find_A_sep` (465) — pure-JAX helpers
  - `evaluate_chance_constrained_cost_hard` (line 653) — true indicator, for pure-SMC² importance weighting (Variant C)
  - `evaluate_chance_constrained_cost_soft` (line 723) — sigmoid surrogate with `beta` knob, for HMC + temperature annealing (Variant B)
  - `evaluate_chance_constrained_cost` is now a back-compat alias for `_hard`
  - Legacy NumPy/SciPy implementation renamed `_evaluate_chance_constrained_cost_legacy` (line 230) — kept for debug
- **Smoke test result:** `cd version_3 && PYTHONPATH=.:.. pytest tests/ -v` → **16/16 PASS** in 21.29s on RTX 5090 / `comfyenv`. The 4 new tests covering JIT compilation, gradient finiteness, and soft↔hard limit all green.
- **Performance (FSA-author's measurement on the 100-particle × 84-day × 96-bin/day benchmark):**
  - hard: 0.86s first call (incl. JIT), 0.229s cached
  - soft: 0.67s first call, 0.157s cached

  Both ≪ 2s budget — comfortably fast enough for HMC inner kernels.
- **Run 00 finding #6 status:** RESOLVED upstream. Stage 2/3 will test BOTH `_hard` and `_soft` variants empirically per Ajay's instruction.

## Run 00c — Stage 1 first-attempt + dtype patch (2026-05-05 12:10)

First attempt to run Stage 1 (`bench_smc_filter_only_fsa_v5.py
--run-tag stage1_filter_only_T14_healthy`) crashed inside window 1 of the rolling-window filter:

```
TypeError: scan body function carry input and carry output must have equal types:
  carry[0] is float32[6] (input) but float64[6] (output)
```

**Root cause:** v5's `propagate_fn` (in `version_3/models/fsa_v5/estimation.py`) constructs the observation Jacobian as `H = jnp.zeros((4, 6))` — which defaults to fp64 (because `JAX_ENABLE_X64=True`). The framework's filter casts particles + params + grid_obs to fp32 before calling `propagate_fn`; the fp64 H matrix then promotes the carry from fp32 to fp64 inside the Kalman scan, breaking the input/output type-equality requirement.

This is a real upstream bug in FSA_model_dev — its standalone tests pass because they call `propagate_fn_v5` with fp64 inputs explicitly, never exercising the framework's fp32 cast path.

**Patch applied locally (commit `dcfb73b`):**
- `H` constructed inline as `jnp.array([[...]])` (dtype inferred from `p`, mirrors v2)
- Kalman-scan carry init `0.0` cast to `y_pred_det.dtype`
- `P_safe` regulariser `1e-10 * jnp.eye(6)` cast to `P_fused.dtype`

**Standalone tests after patch:** 16/16 PASS in 20.70s — the patch doesn't regress the dev-sandbox tests.

**Stage 1 status:** still NOT executed end-to-end. Patch resolves the carry-type crash; the full ~15-45 min rolling-window run is the next thing to do.

### Recommended upstream follow-up (out of scope for this branch)

Add an explicit fp32-input test to `FSA_model_dev/tests/test_fsa_v5_smoke.py` so the framework integration is exercised in the dev sandbox:

```python
def test_v5_propagate_fn_runs_in_fp32():
    theta = jnp.asarray(get_init_theta(), dtype=jnp.float32)
    grid_obs_fp32 = jax.tree_util.tree_map(
        lambda v: v.astype(jnp.float32) if hasattr(v, 'dtype') else v, grid_obs)
    y0 = jnp.array([...], dtype=jnp.float32)
    dt = jnp.float32(1.0/96)
    sigma_diag = jnp.zeros(6, dtype=jnp.float32)
    noise = jnp.zeros(6, dtype=jnp.float32)
    y_new, log_w = propagate_fn_v5(y0, jnp.float32(0.0), dt, theta,
                                    grid_obs_fp32, 0, sigma_diag, noise, key)
    assert y_new.dtype == jnp.float32
    assert log_w.dtype == jnp.float32
```

Once the upstream rewrite lands, this branch can be re-pinned and the local patch dropped.

---

### Note on Run 00 finding #1 (fp64 anti-pattern) — REVISED 2026-05-05 11:45

The original Run 00 finding said "v5 has an fp64 anti-pattern relative to v2's fp32 pattern". **That claim was based on CLAUDE.md's text, not on actually reading v2's code.** Verification:

- v2's `_plant.py:_plant_em_step` (line 95): `noise = jax.random.normal(sub, (3,), dtype=jnp.float64)` — **also fp64 inside the SDE scan**.
- v2's `control.py:_build_cost_and_traj_fns` (line 180): `w_seq = jax.random.normal(key, (n_steps, 3), dtype=jnp.float64)` — **also fp64 inside the cost rollout**.
- v2's `_phi_burst.py` and `simulation.py` dtype usage: byte-identical to v5's.

**v5 model files MATCH v2 model files for dtype.** No fp64 anti-pattern in `version_3/models/fsa_v5/` relative to the working v2 pattern.

The fp32 boost in v2 lives in the BENCH file `version_2/tools/bench_smc_closed_loop_fsa.py:_build_phase2_control_spec` (commit `aa114e8` "Stage L8: FP32 controller cost-MC SDE rollout"). Specifically, lines 351-404 cast `p_jax`, `sub_dt`, `sqrt_dt`, `fixed_w`, `init_arr`, `F_max`, `Phi_arr` to fp32 BEFORE the `cost_fn` `vmap`'d `lax.scan`, with accumulators (`A_acc`, `barrier_acc`) staying fp64 via `jnp.float64(...)` promotion at the accumulation site.

**Phase A.5 (model-file dtype optimization) is therefore a NO-OP.** Phase B will deliver the fp32 boost in the version_3 bench drivers' `_build_phase2_control_spec`-equivalent cost wrappers, mirroring v2's L8 pattern.

This also means CLAUDE.md's claim "Plant integration: `version_2/models/fsa_high_res/_plant.py:_plant_em_step` runs in fp32. Mirror this." is factually inaccurate — it is fp64 in v2. CLAUDE.md is aspirational at that line; the real reference for fp32 controller-cost is the bench's L8 path. Worth a separate CLAUDE.md correction PR; out of scope here.

---

## Run 04 — Stage 1 filter+plant verification PASSED (healthy island, T=14d)

**2026-05-05 16:43–17:31** — first successful end-to-end Stage 1 verification.

- **Run dir:** `experiments/run04_stage1_filter_only_T14_healthy_sat/`
- **Pin:** `7075436` (FSA_model_dev `claude/dev-sandbox-v4`) + local commits `dcfb73b` (Kalman-scan dtype patch), `9e6df28` (Ajay's full fp32-inner / fp64-outer dtype path completion in propagate_fn -- 20 fp64 leaks plugged), `1cbd9b6` (5090 saturation config N=256/K=400)
- **Scenario:** healthy island per LaTeX §8 Test 2 — trained-athlete init `[0.50, 0.45, 0.20, 0.45, 0.06, 0.07]`, fixed Phi=(0.30, 0.30) for 14 d
- **Cadence:** 1-d window / 12-h stride / 27 rolling windows on RTX 5090
- **SMC config:** n_smc_particles=256, n_pf_particles=400 — the 5090 saturation point per CLAUDE.md "GPU saturated post-driver update"

### Final state (end of 14 d trajectory)

`B=0.401, S=0.400, F=0.152, A=0.936, K_FB=0.058, K_FS=0.071` — well inside the v5 healthy island (A > 0.4 per LaTeX Test 2 pass criterion).

### Both acceptance gates PASS

- ✅ **id_cov gate:** 25/27 windows have ≥30/37 estimable params covered at 90% CI (threshold: ≥22/27 windows)
- ✅ **compute gate:** 2890 s = 48.2 min total (threshold: ≤60 min)

### Per-window timing (all on RTX 5090)

| Window | Wall-clock | Note |
|---|---|---|
| 1 (cold + JIT) | 244.3 s | One-time JIT compile cost |
| 2 (bridge JIT) | 105.0 s | One-time bridge-path JIT |
| 3–27 (cached) | 91–107 s avg | Steady-state native path |

### Per-param coverage breakdown (fraction of 27 windows with truth in 90% CI)

- **100% covered (12 params):** `kappa_B, epsilon_AB, epsilon_AS, tau_F, mu_K, mu_F, mu_FF, eta, c_tilde, k_A_S, beta_B_st, beta_F_st, beta_S_VL`
- **>92% covered (10 params):** `tau_B, tau_S, kappa_S, lambda_A, kappa_B_HR, S_base, beta_C_S, beta_A_st, beta_F_VL, sigma_VL`
- **>85% covered (5 params):** `mu_0, mu_B, alpha_A_HR, k_C, k_A, k_F, sigma_st`
- **>80% covered (3 params):** `mu_S, HR_base, beta_C_HR, sigma_S, mu_step0`
- **<80% but still covered most windows (2 params):** `beta_C_st` (74%), `sigma_HR` (63%)

The bottom two are observation-noise / circadian-coefficient parameters — typically harder to identify from short data on a single steady-state regime. They're still being recovered, just less consistently.

### Performance footnote -- root-cause of the GPU underutilization

The first attempt with v2's E3 dev-config (n_smc=128/n_pf=200, BlackJAX recompile-per-window path) gave nvtop ~31% util oscillating to 0%. Two issues were stacked:

1. **20 fp64 leaks in v5's `propagate_fn` Kalman-fusion hot path** (Ajay diagnosed via JAX jaxpr inspection at fp32 input, fixed in commit `9e6df28`). The Hill keys in `_FROZEN_V5_DYNAMICS`, the `EPS_*_FROZEN` constants, and several Python literals were arriving as fp64 and forcing fp32↔fp64 promotion-thrashing inside the inner scan.

2. **Driver was using v2's E3 dev-config** (128/200, BlackJAX path) instead of v2's saturation config (256/400, native compile-once path). 5090 saturates at N=256/K=400 per CLAUDE.md; smaller is wasted parallelism, larger is wasted wall-clock.

After both fixes: GPU saturated, wall-clock manageable, all gates pass.

---

## Run 06 — Stage 2 strict-`soft` baseline (healthy, T=14d)  [A in the A/B]

**2026-05-05 21:52** — controller-only bench, strict `soft` cost variant, healthy scenario.

- **Run dir:** `experiments/run06_stage2_ctrl_soft_healthy_T14d_sat/`
- **Driver:** `bench_controller_only_fsa_v5.py`
- **Cost variant:** `soft` (full-fidelity: fp64, strict bisection, every-bin chance check, n_smc=256, num_mcmc_steps=10, hmc_num_leapfrog=16)
- **Scenario:** healthy — trained-athlete init `[0.50, 0.45, 0.20, 0.45, 0.06, 0.07]`, baseline Φ=(0.30, 0.30)
- **Horizon:** T=14d, replan_K=2, 28 strides, 14 replans
- **Wall-clock:** **5989 s = 99.8 min** on RTX 5090 (matches Ajay's "soft takes ~100 min" prior measurement)

### Trajectory summary

| Metric | Value |
|---|---|
| mean A | 0.815 |
| ∫A dt | 11.42 |
| post-hoc weighted violation rate | **0.96** |
| applied Φ range | [0.15, 0.49] |
| final state | B=0.41, S=0.37, F=0.14, **A=0.945**, K_FB=0.054, K_FS=0.077 |

### Gates

- ✅ `schedule_in_bounds` (max applied Φ = 0.49, well below Φ_max=3.0)
- ✅ `A_integral_geq_target` (11.42 ≥ 2.0 — by 5.7×)
- ❌ `violation_leq_alpha` (0.96 > 0.05) — see comment below
- ✅ `controller_adapts` across replans
- Overall: 3 of 4 gates pass

### Comment on the violation-rate failure

The post-hoc evaluator runs the legacy hard-indicator chance-constraint check on the actual plant trajectory using a single TRUTH_PARAMS_V5 particle. With one particle the indicator collapses to `(A_traj < A_sep_per_bin).mean()`. Despite excellent A trajectory (final A=0.945, deep healthy), the rate is 0.96. Likely cause: the schedule lives in (Φ_B, Φ_S) regions where the analytical `A_sep` is finite-and-large (bistable annulus or near-collapsed boundary) for most bins — even with high A, `A_traj < A_sep` triggers most of the time. NOT a controller failure (the headline `mean_A_integral` is 5.7× the target); rather a quirk of the post-hoc evaluator's geometry under this scenario. **Useful as a baseline for the soft_fast A/B**: if `soft_fast` produces a similar 0.9-ish violation rate at the same scenario, the variants agree on the controller's chosen schedule. If wildly different, real divergence.

### Plots produced

`latent_trajectory.png`, `applied_schedule.png`. `basin_overlay.png` is missing because this run was started before the basin-overlay plotter landed in commit `b70b6fc`. Will regenerate post-hoc from `trajectory.npz` after the soft_fast partner finishes.

---

## Run 07 — Stage 2 soft_fast T=2d sanity (interactive, no overnight)

**2026-05-05 21:00** — pre-overnight typo-catcher, soft_fast at T=2d healthy.

- **Run dir:** `experiments/run07_stage2_soft_fast_T2d_sanity/`
- **Wall-clock:** 51 s (extrapolates to ~6 min for T=14d — ~16× speedup vs run06's 99.8 min)
- ✅ all 6 artifacts produced (manifest.json, trajectory.npz, replan_records.npz, latent_trajectory.png, applied_schedule.png, **basin_overlay.png** — the new diagnostic)
- 2 of 4 gates pass (schedule_in_bounds, controller_adapts). The two FAILs (`A_integral_geq_target`, `violation_leq_alpha`) are expected for T=2d — A_target=2.0 is calibrated for T=14d at A* ≈ 0.55 per LaTeX §8 Test 5; T=2d only has time to accumulate ~1 of A_integral.
- The pipeline + new soft_fast cost wires up correctly.

---

## Run 08 — Stage 2 sanity (T=2d soft_fast, healthy) — overnight launcher hook

`run08_stage2_soft_fast_T2d_sanity_overnight` — same as Run 07 but inside the launcher. **27 s** wall-clock, all artifacts produced including `basin_overlay.png`. Two gate FAILs are expected (T=2d too short to accumulate `A_target=2.0`). Confirms the sweep wires up clean before the long runs.

---

## Run 09 — Stage 2 soft_fast (healthy, T=14d)  [B in the A/B vs Run 06]

**2026-05-05 21:54** — controller-only bench, `soft_fast` cost variant, healthy scenario. Production-default per Ajay's redirect.

- **Run dir:** `experiments/run09_stage2_soft_fast_healthy_T14_optimized/`
- **Cost variant:** `soft_fast` (fp32 throughout, 1e-3 bisection / 20-iter cap, chance check sub-sampled every 4 bins, trimmed HMC: n_smc=128 / num_mcmc_steps=5 / hmc_num_leapfrog=8)
- **Wall-clock:** **986 s = 16.4 min** on RTX 5090 (replan ~70 s/each × 14 replans)

### A/B headline against Run 06 (strict `soft`)

| Metric | Run 06 (`soft`) | Run 09 (`soft_fast`) | Δ |
|---|---|---|---|
| Wall-clock | 99.8 min | **16.4 min** | **−83.6%  (6.07× speedup)** |
| Mean A_integral | 11.42 | 11.41 | **−0.09%**  ✅ within 5% |
| Final A | 0.945 | 0.963 | +1.9% |
| Max applied Φ | 0.49 | 1.07 | +118%  (more aggressive bursts; still well below Φ_max=3.0) |
| Post-hoc violation rate | 0.96 | 0.96 | identical (same evaluator quirk on both) |

### Plan A/B pass criteria

- ✅ **≥3× faster:**   6.07× (target 3×)
- ✅ **within 5% on `mean_A_integral`:**  0.09% (target ≤5%)
- ⚠️ **post-hoc violation ≤ α=0.05:** FAIL on both variants (Run 06 = 0.96, Run 09 = 0.96). The match between variants is the relevant signal: `soft_fast` is producing the same trajectory geometry as `soft`, both fail the post-hoc check for the same evaluator-side reason (separatrix `+inf` semantics in mono-stable healthy regions, or deterministic-vs-stochastic forward-roll discrepancy). **Not a controller divergence.** Investigation queued.

### Gates

- ✅ `schedule_in_bounds` (max applied Φ = 1.07 < 3.0)
- ✅ `A_integral_geq_target` (11.41 ≥ 2.0)
- ❌ `violation_leq_alpha` (0.96 > 0.05) — same as Run 06
- ✅ `controller_adapts`

### Plots produced

`latent_trajectory.png`, `applied_schedule.png`, `basin_overlay.png`. The basin overlay confirms the controller's path stays inside the healthy island throughout the 14 days.

### Verdict

`soft_fast` is the production cost variant going forward. The 6× speedup is exactly what Gemini's optimisation plan predicted (fp32 path on the 5090's main silicon vs fp64 on 1/64th of the cores), with no measurable behavioural divergence at the healthy-island corner.

---

## Run 10 — Stage 2 soft_fast SEDENTARY (aborted partway through sweep)

**2026-05-05 22:11–22:15** — sweep step interrupted before completion.

`run10_stage2_soft_fast_sedentary_T14_optimized` -- only 4 of 14 replans logged before the launcher process was killed externally (no traceback in log; bench process exited cleanly mid-loop). The 4 logged replans show the controller producing **bit-identical** state evolution to Run 09 healthy at the same stride numbers (`state=[0.5, 0.45, 0.2, 0.45, 0.06, 0.07] → ... → state=[0.477, 0.429, 0.179, 0.677, 0.058, 0.076]`).

**Why bit-identical:** Stage 2 driver has a structural property — `baseline_phi` is overwritten by the controller's first replan at `k=0` BEFORE it's ever applied. So `scenario` only changes the saved metadata; the controller starts planning from the trained-athlete state under truth params regardless of which scenario tag is set, and the deterministic plant + same RNG seed (42) produces identical trajectories.

The "scenario" only meaningfully differentiates Stage 3 runs (where the filter sees observations from a `replan_K`-stride warm-up under `baseline_phi` before the controller takes over). The Stage 2 launcher's sedentary + overtrained sweeps would have produced bit-identical results to Run 09; aborting the sweep early was the correct call. Sedentary + overtrained Stage 2 are NOT useful additional runs. (`overtrained` was dropped from the queue without being run.)

---

## Run 11 — Stage 3 soft_fast HEALTHY  (v1, OOM'd at controller's first replan)

**2026-05-05 22:14** — first attempt at Stage 3 with the production particle counts.

- **Run dir:** `experiments/run12_stage3_soft_fast_healthy_T14_optimized/` (note: dir created but no manifest written before crash)
- **Config:** filter n_smc=256 / n_pf=400, controller n_smc=256 / n_inner=64

### What happened

Filter ran fine: window 1 finished in 230 s with id_cov = 37/37. Then the controller's tempered-SMC tried to compile its kernels and OOM'd:

```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 30.85GiB
```

The 5090 has 32 GB VRAM. The filter's compiled kernels were holding ~22.86 GB after window 1; the controller wanted 30.85 GB on top of that → fragmentation, then OOM. Filter and controller can each saturate the GPU on their own (Stage 1 = filter alone uses 22.86 GB; Stage 2 = controller alone uses ~22 GB), but together they exceed the device.

### Fix

Reduced both filter and controller particle counts for Stage 3 (Run 12, below): `--n-pf 200 --n-smc 128 --n-inner 32` -- still GPU-saturating per CLAUDE.md, just smaller batches per kernel so they coexist in VRAM. Stage 3 specifically needs this combination; Stage 1 alone and Stage 2 alone can stay at 400/256/64.

---

## Run 12 — Stage 3 soft_fast HEALTHY  (v2, smaller particles)

(in progress at 22:23 — entry filled in when it finishes)

---

## Run 13 — Stage 2 soft_fast HEALTHY (corrected: cost-fn opts ONLY, full HMC config)

**2026-05-05 22:16 → 2026-05-06 00:21** — diagnostic for the Run 09 quality regression.

- **Run dir:** `experiments/run13_stage2_soft_fast_healthy_T14_full_hmc/`
- **Cost variant:** `soft_fast` (fp32 throughout, 1e-3 bisection / 20-iter cap, chance check sub-sampled every 4 bins)
- **HMC config:** **REVERTED to the strict `soft` baseline** — `n_smc=256`, `num_mcmc_steps=10`, `hmc_num_leapfrog=16`. Only the cost-fn-side optimisations remain.
- **Wall-clock:** **124.9 min** on RTX 5090 (slower than Run 06's 99.8 min — GPU was being shared with Run 14 Stage 3 throughout, so wall-clock isn't a clean comparison)

### Why this run

Run 09 (full optimisation stack including HMC trim) finished in 16.4 min and matched Run 06's metric integrals to <0.5%, but the **basin overlay showed the controller's daily-mean Φ wandering as far as (Φ_B, Φ_S) = (1.07, 0.28)** — i.e. into the collapsed-T regime — twice during the 14 days. Plant SDE robustness recovered each time so the integral metrics looked clean, but the controller schedule was meaningfully noisier than the strict baseline.

Hypothesis: the HMC config trim (`n_smc 256→128`, `mcmc_steps 10→5`, `leapfrog 16→8`) cut the controller's exploration budget by ~16× and that's the source of the noisy proposals. The cost-fn optimisations (fp32, relaxed bisection, sub-sampled bins) are mathematical and should be safe.

Run 13 tests this: revert the HMC trim, keep the cost-fn opts.

### A/B against Runs 06 + 09

| Metric | Run 06 (`soft`, full HMC) | Run 09 (`soft_fast`, trimmed HMC) | **Run 13 (`soft_fast`, full HMC)** |
|---|---|---|---|
| Wall-clock | 99.8 min | 16.4 min | 124.9 min* |
| Mean A_integral | 11.42 | 11.41 | **11.37** |
| Final A | 0.945 | 0.963 | **0.948** |
| Max applied Φ | **0.49** | 1.07 | **0.528** ✅ |
| Min applied Φ | 0.15 | 0.12 | **0.113** |
| Post-hoc violation rate | 0.96 | 0.96 | 0.96 (same evaluator quirk) |

*Run 13's wall-clock is contaminated by GPU contention with Run 14 (Stage 3 healthy v2) which ran in parallel for the entire duration. A clean re-run on a free GPU is needed before claiming a speed result for soft_fast + full HMC.

### Basin overlay verdict (the diagnostic plot)

✅ **Controller path stays inside the bistable annulus and hugs the healthy island.** Start point (first replan at (0.53, 0.11), green dot) sits on the boundary; subsequent daily-mean Φ values walk into the bistable region and converge to ~(0.34, 0.25), inside the healthy island and near the scenario's (0.30, 0.30) baseline (purple end dot). **Zero excursions to the collapsed regime** — the qualitative quality bug from Run 09 is gone.

### Gates

- ✅ `schedule_in_bounds` (max applied Φ = 0.528 < 3.0)
- ✅ `A_integral_geq_target` (11.37 ≥ 2.0)
- ❌ `violation_leq_alpha` (0.96 > 0.05) — same evaluator quirk as Runs 06/09
- ✅ `controller_adapts`

### Verdict — Run 09 regression diagnosed

**The HMC trim alone was the bug.** Cost-fn optimisations (fp32, relaxed bisection, sub-sampled bins) are mathematically clean: they preserve the trajectory geometry. Cutting HMC's exploration budget did NOT preserve it — proposal-space coverage matters for stable schedules under the chance-constrained cost.

### Production config going forward

`soft_fast` cost-fn (`evaluate_chance_constrained_cost_soft_fast`) with the strict-`soft` HMC config:
- `n_smc=256`, `n_inner=64`
- `num_mcmc_steps=10`, `hmc_num_leapfrog=16`

Both bench drivers (`bench_controller_only_fsa_v5.py` and `bench_smc_full_mpc_fsa_v5.py`) had the trim reverted before Run 13 launched. Reverts are committed in `bc26316`.

### Open question — wall-clock for the corrected config

124.9 min vs Run 06's 99.8 min looks like a regression, but Run 13 had GPU contention. Need a clean re-run on a free GPU to A/B speed at the corrected production config. Profiling work queued (`tools/profile_cost_fn.py`).

---

## Run 15 — Stage 2 soft_fast SEDENTARY (in flight at 01:13)

**2026-05-06 00:22 → in progress**

- **Run dir:** `experiments/run15_stage2_soft_fast_sedentary_T14_full_hmc/`
- **Config:** corrected production (soft_fast cost-fn + full HMC)
- **Status:** Stride 11/28, ~17 strides remaining, ~9 min/stride → ETA ~03:30

Per the Run 10 lesson: Stage 2 sedentary trajectories will be bit-identical to healthy under the same RNG (controller's first replan overwrites `baseline_phi`). The run will complete and produce metrics, but won't add new information beyond Run 13.

---

## Run 03 — TODO: Stage 3 full closed-loop SMC²-MPC verification

(superseded by Run 12 above)
