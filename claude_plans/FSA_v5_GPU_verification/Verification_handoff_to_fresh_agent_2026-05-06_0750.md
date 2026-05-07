# Verification handoff — re-verify FSA-v5 Stage 2 + Stage 3 from scratch

> Written 2026-05-06 07:50 London by the previous agent (who is the source of the bugs you are about to find). Read this end-to-end before touching anything.

## Read this first — explicit warning

**All Stage 2 and Stage 3 results produced overnight (2026-05-05 → 2026-05-06) by the previous agent are suspect.** The previous agent found ONE bug already (Stage 2 baseline_phi warm-up — see below) and wrote up the overnight session with conclusions like "1.67× speedup", "Stage 2 verification PASSES", "Stage 3 verification PASSES across three scenarios". **Treat all of these conclusions as unverified.** A fresh agent (you) is being brought in specifically because the user no longer trusts the previous agent's audit.

Specifically, do NOT do any of the following without independent verification:
- Do NOT cite "Run 16's 59.7 min wall-clock" as a clean A/B benchmark.
- Do NOT cite "Run 18 / Run 19 basin overlays" as Stage 3 verification panels.
- Do NOT cite "soft_fast cost-fn + full HMC = production config" as confirmed.
- Do NOT trust the CHANGELOG entries written by the previous agent — they may be wrong in ways the previous agent did not catch.

The user's **only** consistent observation across the overnight runs is: **"full HMC is always better than trimmed HMC"**. That qualitative pattern stands. Everything else is suspect.

## Your job

1. **Read the code in `version_3/tools/bench_controller_only_fsa_v5.py` and `version_3/tools/bench_smc_full_mpc_fsa_v5.py` end-to-end yourself.** Do not skim. The previous agent missed a bug at line 449 of the controller-only driver. There may be more.
2. **Independently verify the code is correct** — especially around the warm-up logic, the `current_schedule` indexing, the `applied_phi_per_stride` recording, and the basin-overlay plotter.
3. **Re-run Stage 2 sedentary + healthy + overtrained** at the corrected driver. Confirm by inspecting the basin overlays that the start point matches the scenario's `baseline_phi`.
4. **Re-run Stage 3 sedentary + healthy + overtrained** at the corrected production config. Independently judge whether the previous agent's "verification passes" claim holds.
5. **Form your own audit trail** — do not just append to the existing CHANGELOG. Start fresh entries with explicit "verified by [you]" provenance.

## What the previous agent claims is the production config (verify this)

- Cost variant: `evaluate_chance_constrained_cost_soft_fast` from `version_3/models/fsa_v5/control_v5_fast.py` (a NEW file written by the previous agent on 2026-05-05; verify its math against the upstream `evaluate_chance_constrained_cost_soft` in `control_v5.py`).
- HMC config: `n_smc=256`, `num_mcmc_steps=10`, `hmc_num_leapfrog=16` (matches strict baseline, NOT trimmed).
- Filter (Stage 3): `n_smc=128, n_pf=200, n_inner=32` (smaller particles to fit in 32 GB VRAM alongside the controller; full particles OOM on RTX 5090).

## The bug the previous agent found (and admitted to)

`tools/bench_controller_only_fsa_v5.py` (the Stage 2 driver) had `if k % replan_K == 0:` at line 449 which fires at k=0 and replans BEFORE `baseline_phi` is ever applied. So every Stage 2 run started from `init_state = TRAINED_ATHLETE_STATE` regardless of the `scenario` tag. Sedentary, healthy, and overtrained Stage 2 trajectories collapsed to identical metrics.

The previous agent originally framed this bit-identicality as a "structural property" of Stage 2 — that was wrong. The user (Ajay) caught it.

**Fix already applied** in commit `6241a9d` — the previous agent changed the condition to `if k > 0 and k % replan_K == 0:` so baseline_phi gets applied for the first replan_K strides.

**You must verify this fix is correct.** Specifically:
- Run a smoke test (T=2d Stage 2 sedentary on CPU or GPU) and inspect the basin overlay's start point. It should sit at (0.0, 0.0).
- Read the diff of `6241a9d` in detail and decide whether the warm-up logic now mirrors Stage 3's design correctly.

## Bugs the previous agent did NOT find (suspected — find them yourself)

The previous agent's track record this session was: assert "structural property", get caught, retract. There may be more failures of the same kind. Plausible places to look:

- **The basin-overlay plotter** (`version_3/tools/plot_basin_overlay.py`). Does it correctly read the SCENARIO baseline as the start dot, or does it use the first applied Φ (which differs from baseline pre-fix)? The Stage 3 overlays for Run 18/19 happened to look correct because Stage 3 actually applied baseline_phi during warm-up — but the plotter's source-of-truth needs verifying.
- **The `weighted_violation_rate` evaluator** — fails 0.93–0.96 across every soft AND soft_fast run. The previous agent waved this away as a "known evaluator quirk". It might actually be a real bug. Investigate.
- **`evaluate_chance_constrained_cost_soft_fast`** — new file, written by the previous agent on 2026-05-05. The four "optimisations" (fp32 throughout, relaxed 1e-3 bisection, sub-sampled bins every 4, fp32 cost-fn body) were all hand-rolled. Run the smoke tests `version_3/tests/test_fsa_v5_bench_smoke.py::test_soft_fast_*` and verify they actually exercise the corner cases.
- **Run-numbering race.** The bench drivers auto-number runs by counting existing dirs. Multiple parallel runs can race and produce out-of-order numbering (the previous agent's "Run 11" diagnostic actually landed in `run13_*`). Check that any run you launch lands where you expect.

## Files the previous agent created or modified overnight

Treat all of these as subject to your independent verification:

**Code:**
- `version_3/models/fsa_v5/control_v5_fast.py` — NEW file, ~200 lines. Hand-written soft_fast cost-fn.
- `version_3/tools/bench_controller_only_fsa_v5.py` — modified: HMC trim revert (commit `bc26316`), warm-up bug fix (commit `6241a9d`).
- `version_3/tools/bench_smc_full_mpc_fsa_v5.py` — modified: HMC trim revert (commit `bc26316`).
- `version_3/tools/profile_cost_fn.py` — NEW file, the JAX profiler.
- `version_3/tools/plot_basin_overlay.py` — written earlier this session.
- `version_3/tools/plot_stage2_theta_traces.py` — NEW file (commit `ebea1d5`).
- `version_3/tools/launchers/run_quality_speed_overnight.sh` — overnight master launcher.
- `version_3/tools/launchers/run_followup_overnight.sh` — overnight follow-up launcher.

**Docs (treat as a record of claims, not facts):**
- `version_3/outputs/fsa_v5/CHANGELOG.md` — Runs 06–19 entries written by the previous agent. Read with skepticism.
- `claude_plans/Overnight_summary_quality_speed_2026-05-06_0700.md` — the previous agent's overnight summary. Read with skepticism.
- `claude_plans/Session_handoff_quality_speed_overnight_2026-05-05_2230.md` — the previous agent's handoff doc from earlier in the night.

## How to actually run the benches (verified working invocations)

The runs ARE able to execute end-to-end on the user's RTX 5090. The verification question is whether the results are MEANINGFUL, not whether the bench scripts run.

```bash
cd /home/ajay/Repos/python-smc2-filtering-control
conda activate comfyenv
cd version_3

# Stage 2 controller-only, T=14 days, soft_fast cost-fn
PYTHONPATH=.:.. python tools/bench_controller_only_fsa_v5.py \
    --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
    --run-tag stage2_verifier_sedentary

# Stage 3 full closed-loop, T=14 days, soft_fast cost-fn, smaller particles
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v5.py \
    --cost soft_fast --scenario sedentary --T-days 14 --replan-K 2 \
    --n-smc 128 --n-pf 200 --n-inner 32 \
    --run-tag stage3_verifier_sedentary

# Generate basin overlay on CPU after a run finishes
PYTHONPATH=.:.. JAX_PLATFORMS=cpu python tools/plot_basin_overlay.py \
    outputs/fsa_v5/experiments/<run_dir>/

# Cost-fn profiler
PYTHONPATH=.:.. python tools/profile_cost_fn.py \
    --cost soft_fast --n-smc 256 --n-truth-particles 1 --n-iters 20
```

**Wall-clock budgets observed empirically (the previous agent's measurements; verify these too):**
- Stage 2 controller-only T=14d: ~60–125 min on RTX 5090 (60 min on free GPU, 125 min under contention).
- Stage 3 full closed-loop T=14d at smaller particles: ~100–110 min on free GPU.
- Profile sweep (4 configs): ~5–10 min total.

## Concrete verification protocol I suggest (override if you have a better one)

### Phase 1 — Code audit (no GPU needed)

1. Read both bench drivers end-to-end. Trace the `current_schedule` / `applied_phi_per_stride` / `plant.advance` flow on paper for the first 4 strides of a sedentary run. Confirm `applied_phi_per_stride[0:replan_K] == baseline_phi`.
2. Read `control_v5_fast.py` end-to-end. Compare the 4 optimisations to the strict `control_v5.py.evaluate_chance_constrained_cost_soft`. Verify they are mathematical (not behaviour-changing).
3. Read `plot_basin_overlay.py`. Verify the start-dot uses scenario.baseline_phi, not the first applied Φ.
4. Run the smoke test suite: `cd version_3 && PYTHONPATH=.:.. pytest tests/ -v`. Should be 16/16 pass at the current SHA.

### Phase 2 — Smoke test on GPU (~10–15 min)

Run a T=2d Stage 2 sedentary at the corrected driver:

```bash
cd version_3
PYTHONPATH=.:.. python tools/bench_controller_only_fsa_v5.py \
    --cost soft_fast --scenario sedentary --T-days 2 --replan-K 2 \
    --run-tag verifier_smoke_T2_sedentary
```

Inspect the basin overlay's start dot. **It must be at (0.0, 0.0).** If it's not, the warm-up fix is incomplete.

### Phase 3 — Full Stage 2 + Stage 3 sweep (~6–8 h GPU)

Only after Phase 1 + 2 pass:

- Stage 2 healthy, sedentary, overtrained at corrected driver — 3 × ~60 min.
- Stage 3 healthy, sedentary, overtrained at corrected production config + smaller particles — 3 × ~110 min.

Each at the same RNG seed (42). Inspect every basin overlay for correct start point + path direction. Check whether the user's "full HMC is always better" pattern still holds.

### Phase 4 — Independent write-up

Do NOT append to `version_3/outputs/fsa_v5/CHANGELOG.md` — the existing entries are tangled with the previous agent's claims. Write your own audit doc in `claude_plans/Verification_results_<your_date>.md` with explicit findings: which prior claims hold, which don't, what the correct production config is.

## What the user (Ajay) cares about

- Plain everyday language. No GitHub / packaging jargon.
- Verify before you assert. Don't call code "wrong" without checking. Don't call results "verified" without re-deriving them yourself.
- Junior engineer stance. The previous agent (and you) are junior under Ajay. Existing code from upstream `FSA_model_dev` (pinned at SHA `7075436`) is the senior decision — don't rewrite it without being asked.
- The end goal of FSA-v5 is real-data closed-loop control with rolling-window sequential estimation over years of training-load data. Every claim about "the controller works" needs to be solid because it'll be cited downstream in publications.

Good luck. Don't trust the previous agent's claims. Re-derive.
