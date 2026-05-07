# FSA-v5 verification — session handoff

**Session:** 2026-05-06 09:22 → 09:46 (~25 min)
**Working branch:** `feat/import-swat-from-dev-repo` (hot — see end notes)
**Status:** Stage 0 complete; Stage 1 sandbox sanity passed; **GPU not used** per instruction.

---

## Headline findings

### 1. The 96% violation rate is a **formulation artifact, not a controller bug**

The persistent `weighted_violation_rate ≈ 0.96` the previous agent observed
across all soft / soft_fast Stage 2/3 runs is mechanically explained by the
combination of:
- Bursty Phi-burst expansion (`_phi_burst.py`) → median per-bin Phi = 0.0 in both channels for typical schedules
- At Phi=(0,0), the v5 closed-island topology gives `A_sep = +inf` (mono-stable collapsed)
- Indicator `(A_traj < +inf)` always fires → 96% of bins "violate" by construction
- Run 16 (overtrained, full HMC, soft_fast) has exactly **1296/1344 = 0.9643** bins where A_sep is `+inf`, matching the reported rate to 4 decimals.

**Reproducer:** `version_3/tools/diagnose_violation_rate.py <run_dir>`. Three formulations:
1. **Per-bin** (production): 0.9643 (run 16) — vacuous
2. **Per-day mean Phi**: 0.2143 (3/14 days outside healthy island) — sensible
3. **Active-bin only** (Phi>0.05): 0.91 — still high because peak Phi during morning bursts also lies outside the narrow healthy island

The controller was actually performing **reasonably** on run 16: PhiB=0.52 over-aggressive on day 1, settled into the (0.30, 0.30) neighborhood from day 7 onwards, hit healthy mono-stable on 8/14 days.

### 2. Bug 2 (particle-0 separator template) is **real but demoted**

The Plan agent's hypothesis about [control_v5.py:576](version_3/models/fsa_v5/control_v5.py#L576) (separatrix computed only from particle-0 then broadcast across all particles) is **confirmed by code reading** but is **NOT the cause of the 96% rate** — that's fully explained by Finding 1 above. Bug 2 may still bias the *internal* HMC cost when the SMC² posterior cloud has variance in `mu_0/mu_B/mu_S/mu_F`, but that's a separate and lower-priority question. Confirming would need a multi-particle GPU run.

### 3. Bug 1 (`sigma_S` dict-key collision) is **real but inert**

`DEFAULT_PARAMS_V5['sigma_S'] = 4.0` (the obs-channel value silently overwriting the state-noise 0.008). **However**, both [_plant.py:266-273](version_3/models/fsa_v5/_plant.py#L266-L273) and [estimation.py SIGMA_*_FROZEN](version_3/models/fsa_v5/estimation.py#L58-L62) **already work around it** by hard-coding the correct state-noise scales. Obs samplers correctly read `sigma_S=4.0` for stress noise. The bug is real in source-form but doesn't corrupt any current behavior.

**Decision: skip the rename** (high blast radius — estimation prior config + obs samplers + tests). Added 5 guardrail tests instead at `version_3/tests/test_fsa_v5_param_dict.py` so any future "fix" that breaks the workaround is caught immediately. The same collision exists in the upstream FSA_model_dev repo; flagged but not patched per hard constraint.

### 4. Bug 3 (run-numbering TOCTOU race) **fixed**

Replaced three identical buggy `_allocate_run_dir` helpers with a shared atomic-mkdir helper at `version_3/tools/_run_dir.py`. Five race tests at `version_3/tests/test_run_dir_atomic.py`, including 8-process concurrent-launch tests with both same-tag and distinct-tag scenarios.

---

## Decision needed from Ajay (revised — much smaller scope than initially thought)

**The cost function is CORRECT. The bug is in the bench's post-hoc evaluator wiring.** Two independent reads of the Phi schedule:

| Caller | Phi source | Shape | Result |
|---|---|---|---|
| Controller's internal cost (during HMC) | `schedule_from_theta(theta_ctrl)` — smooth RBF | (n_steps, 2) — non-bursty | Indicator gives a meaningful violation estimate |
| Bench's post-hoc evaluator (gate check) | `plant.history['Phi_value']` — **burst-expanded** | (1344, 2) at T=14d, median=0 | Most bins have Phi=0 → A_sep=+inf → 96% vacuous |

The chance-constraint spec at [FSA_version_5_technical_guide.tex eq:chance-constraint](`/home/ajay/Repos/FSA_model_dev/LaTex_docs/FSA_version_5_technical_guide.tex`) says `P[A_t < A_sep(Phi_t)] ≤ alpha ∀ t ∈ [0, T]`. The math is faithful for smooth Phi. It breaks only because the bench's post-hoc evaluator passes the burst-expanded form.

**The fix is small and local to the bench**, not the cost function. Two clean options:

a) **Pass the smooth RBF schedule to post-hoc**, by recomposing the per-replan `mean_theta`s into a single (n_steps_full, 2) array via `schedule_from_theta`. Then the post-hoc sees what the controller actually optimised against.
b) **Pass a daily-mean of full_phi**: aggregate `full_phi` to `(n_days, 2)` and evaluate A_sep at the daily-mean. Loses some temporal resolution but captures the chronic-stimulus intent.

**No code change made yet — your call which to apply** (if either; some teams treat the discrepancy as informative). It's `~5 lines` in [bench_controller_only_fsa_v5.py:517-531](version_3/tools/bench_controller_only_fsa_v5.py#L517-L531) plus the analogous fix in `bench_smc_full_mpc_fsa_v5.py`.

The cost function in [control_v5.py](version_3/models/fsa_v5/control_v5.py) does NOT need changing.

---

## Test status

- **Before session:** 30/30 tests passing across 4 files
- **After session:** 40/40 tests passing across 6 files (added 10 tests in 2 new files)

```
version_3/tests/test_fsa_v5_bench_smoke.py        14 tests
version_3/tests/test_fsa_v5_param_dict.py          5 tests   ← NEW (guardrails for Bug 1 workaround)
version_3/tests/test_fsa_v5_smoke.py               8 tests
version_3/tests/test_obs_consistency_v5.py         6 tests
version_3/tests/test_reconciliation_v5.py          2 tests
version_3/tests/test_run_dir_atomic.py             5 tests   ← NEW (race tests for Bug 3 fix)
                                            ───────────────
                                                  40 passed in ~40s
```

---

## Files touched

### New files (all under [version_3/](version_3/))

- [version_3/tools/_run_dir.py](version_3/tools/_run_dir.py) — atomic-mkdir helper (Bug 3 fix)
- [version_3/tools/diagnose_violation_rate.py](version_3/tools/diagnose_violation_rate.py) — read-only forensics tool for the 96% rate
- [version_3/tests/test_fsa_v5_param_dict.py](version_3/tests/test_fsa_v5_param_dict.py) — sigma_S guardrail tests
- [version_3/tests/test_run_dir_atomic.py](version_3/tests/test_run_dir_atomic.py) — race tests
- [version_3/outputs/fsa_v5/experiments_log.md](version_3/outputs/fsa_v5/experiments_log.md) — new tabular methodology
- [claude_plans/FSA-v5_verification_debug_plan_2026-05-06_0922.md](claude_plans/FSA-v5_verification_debug_plan_2026-05-06_0922.md) — the approved plan

### Modified files

- [version_3/tools/bench_controller_only_fsa_v5.py](version_3/tools/bench_controller_only_fsa_v5.py) — `_allocate_run_dir` delegates to shared helper (was buggy)
- [version_3/tools/bench_smc_filter_only_fsa_v5.py](version_3/tools/bench_smc_filter_only_fsa_v5.py) — same delegation
- [version_3/tools/bench_smc_full_mpc_fsa_v5.py](version_3/tools/bench_smc_full_mpc_fsa_v5.py) — same delegation

### Memory updates

- `feedback_psim_first_always.md` — updated to point at `/home/ajay/Repos/FSA_model_dev` for FSA validation (legacy `Python-Model-Scenario-Simulation/` is deprecated for FSA per Ajay 2026-05-06; SWAT psim path remains valid).

### Files NOT touched

- Anything outside [version_3/](version_3/) (per hard constraint).
- The `sigma_S` rename (per design decision — see Bug 1 above).
- The chance-constraint cost function (per Bug 4 — needs Ajay's input).
- The FSA_model_dev sandbox (read-only per hard constraint).

---

## Stage 1 sandbox sanity (FSA_model_dev) — PASSED qualitatively

Open-loop SDE sims under TRUTH_PARAMS_V5 with TRAINED initial conditions
matching the bench's healthy scenario:

| Scenario     | Phi          | T_days | A end → mean       | F end | Verdict                                       |
|--------------|--------------|--------|--------------------|-------|-----------------------------------------------|
| healthy      | (0.30, 0.30) | 14     | A=0.877, mean=0.78 | 0.18  | Strong autonomic; basin-stable                |
| sedentary    | (0.0, 0.0)   | 14     | A=0.904, mean=0.81 | 0.01  | Hill not yet biting at T=14                   |
| overtrained  | (1.0, 1.0)   | 14     | A=0.195, mean=0.50 | 0.98  | A collapses; F near runaway                   |
| sedentary    | (0.0, 0.0)   | 42     | B 0.50→0.19; A holds then declines | 0   | Hill kicks in around day 28 ✓ matches memory |
| healthy      | (0.30, 0.30) | 42     | A 0.45→0.85 plateau; B/S decline 0.50→0.32 | 0.17 | Sustainable but mild capacity detrain at Phi=0.3 |

Model qualitatively correct. Closed-island topology confirmed: healthy
disc ≈ Phi_B ∈ [0.20, 0.40] × Phi_S ∈ [0.20, 0.35]; everything outside is mono-stable collapsed.

---

## What I did NOT do

- **No GPU experiments.** Per Ajay's instruction. Stage 3/4 (controller-only and full closed-loop) untouched.
- **No git commit.** Branch `feat/import-swat-from-dev-repo` is hot with unrelated SWAT work; let Ajay decide whether to commit on this branch or split.
- **No FSA_model_dev edits.** Sigma_S collision flagged in the upstream too; not patched.
- **No cost-function edits.** Bug 4 needs Ajay's input.
- **Stage 2 (sim-est consistency)** as Ajay said it was already done. Did not re-run.

---

## Recommended next steps (in order)

1. **Decide on Bug 4** — keep per-bin formulation, switch to per-day, or active-only. This is the gating decision before any Stage 3 GPU runs.
2. **Stage 3 healthy** at full HMC (`256/10/16`) once Bug 4 is decided. If keeping per-bin, expect 96%-ish violation rate (will fail viol gate but other gates may pass).
3. **Investigate Bug 2 with a real run** — the diagnostic tool can be extended to per-particle A_sep distribution by reading `replan_records.npz` and comparing particle-0 vs particle-mean separator.
4. **Stage 4** at full HMC, T=14 then T=42.

---

## Critical files for continued work

- [version_3/tools/bench_controller_only_fsa_v5.py](version_3/tools/bench_controller_only_fsa_v5.py) — post-hoc wiring (Bug 4 lives here, NOT in the cost function)
- [version_3/tools/bench_smc_full_mpc_fsa_v5.py](version_3/tools/bench_smc_full_mpc_fsa_v5.py) — same Bug 4 wiring
- [version_3/models/fsa_v5/control_v5.py](version_3/models/fsa_v5/control_v5.py) — cost function (correct as-is — leave alone)
- [version_3/tools/diagnose_violation_rate.py](version_3/tools/diagnose_violation_rate.py) — forensics
- [version_3/outputs/fsa_v5/experiments_log.md](version_3/outputs/fsa_v5/experiments_log.md) — Ajay's index
- [claude_plans/FSA-v5_verification_debug_plan_2026-05-06_0922.md](claude_plans/FSA-v5_verification_debug_plan_2026-05-06_0922.md) — original approved plan

---

## Draft fix for Bug 4 (NOT applied — for Ajay's review)

### Option A: post-hoc on smooth RBF schedule (recommended)

Modify [bench_controller_only_fsa_v5.py:513-531](version_3/tools/bench_controller_only_fsa_v5.py#L513-L531). Replace the `full_phi` source for the post-hoc with the controller's own decoded RBF schedule joined across replans:

```python
# Re-compose the smooth schedule the controller actually optimised
# against, instead of using the burst-expanded plant input.
smooth_schedule_chunks = []
for r in replan_records:
    smooth_schedule_chunks.append(np.asarray(r['mean_schedule']))
# replan_records covers the n_replans points where the controller fired;
# strides between replans applied the same daily slot. Build a (n_steps_full, 2)
# array by concatenating smooth chunks and slicing.
full_schedule_smooth = np.concatenate(smooth_schedule_chunks, axis=0)
n_steps_full = min(full_schedule_smooth.shape[0], full_traj.shape[0])

posthoc = posthoc_eval(
    [dict(truth)], np.array([1.0]),
    full_schedule_smooth[:n_steps_full],   # ← was full_phi[:n_steps_full]
    dt=DT, alpha=alpha, A_target=A_target,
    truth_params_template=truth,
    initial_state=init_state,
)
```

**Pros:** post-hoc evaluates the EXACT schedule the controller's HMC was optimising, so it's an honest verdict on controller quality. **Cons:** if the smooth schedule ≠ what the plant actually saw (because of burst-expansion-then-back), there's a mismatch we should think about.

### Option B: post-hoc on daily-mean burst Phi

Cheaper and aligns with the chronic-stimulus interpretation:

```python
# Use daily-mean Phi instead of per-bin to side-step the burst issue.
BPD = BINS_PER_DAY
n_days = full_phi.shape[0] // BPD
phi_daily = full_phi[:n_days*BPD].reshape(n_days, BPD, 2).mean(axis=1)
posthoc = posthoc_eval(
    [dict(truth)], np.array([1.0]),
    phi_daily,
    dt=DT * BPD,                            # daily timestep
    alpha=alpha, A_target=A_target,
    truth_params_template=truth,
    initial_state=init_state,
)
```

**Pros:** trivial change, evaluator sees the chronic stimulus level. **Cons:** loses sub-daily resolution; A_traj also needs daily averaging or this is comparing apples to oranges (the post-hoc internally rolls forward A_traj at the dt provided).

### Option C: keep both — log per-bin AND daily-mean

Lowest-risk: report both numbers. Per-bin remains the official gate (96%); daily-mean is a "diagnostic" line. Lets Ajay compare across runs without committing to a semantic change.

**My recommendation:** Option A. It directly evaluates what the controller was optimising. Closely tied to the LaTeX guide §8 spec: `P[A_t < A_sep(Phi_t)]` — `Phi_t` should be the schedule, not the burst expansion (which is a plant implementation detail).

Apply this only after Ajay confirms the option.
