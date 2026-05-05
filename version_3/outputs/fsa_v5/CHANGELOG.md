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

## Run 01 — TODO: Stage 1 filter+plant verification

(empty until first run)

## Run 02 — TODO: Stage 2 controller-only verification

(empty until first run)

## Run 03 — TODO: Stage 3 full closed-loop SMC²-MPC verification

(empty until first run)
