# FSA-v5 Experiments Log

Append-only tabular index over every experiment run since the
verification campaign started on 2026-05-06. Each row points at a
self-contained artifact dir under `version_3/outputs/fsa_v5/experiments/`
(or, for Stage-1 sandbox runs, under `/home/ajay/Repos/FSA_model_dev/`).

The per-run `manifest.json + trajectory.npz + plots` remain the source
of truth; this file is the human-scannable index.

## Conventions

- **Run ID** — sequential within the smc2fc experiments dir (allocated
  atomically by `version_3/tools/_run_dir.py`). Sandbox runs (Stage 1)
  use a `dev-` prefix instead since they live outside the smc2fc
  numbering namespace.
- **Pass/Fail** — one of: `PASS`, `FAIL`, `INFO` (Stage 0 audits),
  `PARTIAL` (some gates met, some not).
- **HMC cfg** — compact `n_smc/num_mcmc_steps/hmc_num_leapfrog`.
- **Wall-clock (s)** — measured, not estimated. `n/a` for static stages.
- **Key metric** — stage-appropriate one-liner.
- **Cost** — `soft` / `hard` / `gradient_ot` / `n/a`. Note that
  `soft_fast` is QUARANTINED per the 2026-05-06 plan and must not
  appear in this log without an explicit override note.

## Log

| Run ID  | Date       | Stage | Scenario | T_days | Cost | HMC cfg     | Wall-clock (s) | Key metric                                                          | Pass/Fail | Notes                                                                               | Artifact dir                                  |
|---------|------------|-------|----------|--------|------|-------------|----------------|---------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------|-----------------------------------------------|
| 0a-001  | 2026-05-06 | 0     | n/a      | n/a    | n/a  | n/a         | 41             | 30/30 existing tests pass at start                                  | INFO      | baseline established before any code changes                                        | n/a (Stage 0 inline)                          |
| 0a-002  | 2026-05-06 | 0     | n/a      | n/a    | n/a  | n/a         | 2              | DEFAULT_PARAMS_V5['sigma_S']==4.0 (collision confirmed)             | INFO      | Bug 1 confirmed; functionally INERT — plant + estimation use SIGMA_*_FROZEN constants | n/a (REPL inspection)                         |
| 0a-003  | 2026-05-06 | 0     | n/a      | n/a    | n/a  | n/a         | n/a            | Added 5 guardrail tests (test_fsa_v5_param_dict.py)                  | PASS      | Locks in current workaround so future refactor can't silently break it              | version_3/tests/test_fsa_v5_param_dict.py     |
| 0a-004  | 2026-05-06 | 0     | n/a      | n/a    | n/a  | n/a         | n/a            | Bug 3 fixed (atomic mkdir helper); 5 race tests pass                | PASS      | Shared at version_3/tools/_run_dir.py; replaces 3 in-bench TOCTOU helpers           | version_3/tools/_run_dir.py                   |
| 0a-005  | 2026-05-06 | 0     | n/a      | n/a    | n/a  | n/a         | 41             | 40/40 tests pass after Stage 0 fixes                                | PASS      | Net add: +10 tests (5 sigma_S guardrails, 5 race tests)                             | version_3/tests/                              |
| 0b-001  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | 5              | A_sep map: healthy island ≈ Phi_B∈[0.20,0.40] × Phi_S∈[0.20,0.35]    | INFO      | Truth-params island is narrow; everything outside is mono-stable collapsed (+inf)   | n/a (REPL)                                    |
| 0b-002  | 2026-05-06 | 0+    | overtrn  | 14     | n/a  | n/a         | n/a            | run16 forensics: per-bin viol=0.9643 explained by burst (Phi=0 most bins → A_sep=+inf) | INFO  | Bug 2 (particle-0) NOT the cause; conceptual issue with per-bin chance constraint   | (analysis of old_experiments/run16/)           |
| 0b-003  | 2026-05-06 | 0+    | overtrn  | 14     | n/a  | n/a         | n/a            | Per-day reformulation of run16: viol drops to 0.2143 (3/14 days)     | INFO      | Daily-mean Phi gives sensible verdict: controller settled by day 4                  | diagnose_violation_rate.py                     |
| 0b-004  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | Built diagnose_violation_rate.py — read-only, three-formulation report | PASS    | Reproducer for the violation-rate forensics; works on any past or future run        | version_3/tools/diagnose_violation_rate.py    |
| dev-1a  | 2026-05-06 | 1     | n/a      | n/a    | n/a  | n/a         | 2              | Confirmed sigma_S collision exists in dev repo too (upstream)         | INFO      | Same root; flagged but not patched per hard constraint #1                            | /home/ajay/Repos/FSA_model_dev/                |
| dev-1b  | 2026-05-06 | 1     | healthy  | 14     | n/a  | n/a         | 4              | A end=0.877, mean=0.781; B/S/F bounded; no NaN                       | PASS      | Trained-init Phi=(0.30,0.30); sandbox via diffrax solver                            | (REPL inspection)                              |
| dev-1c  | 2026-05-06 | 1     | sedent.  | 14     | n/a  | n/a         | 1              | A end=0.904 (still high at T=14d — Hill not yet biting)              | PASS      | Trained-init Phi=0; F→0.01, B/S declining gently                                    | (REPL inspection)                              |
| dev-1d  | 2026-05-06 | 1     | overtrn  | 14     | n/a  | n/a         | 1              | F end=0.983 (near runaway), A end=0.195 (collapsing)                  | PASS      | Trained-init Phi=(1.0,1.0); fatigue dominates by day 14 — expected                  | (REPL inspection)                              |
| dev-1e  | 2026-05-06 | 1     | sedent.  | 42     | n/a  | n/a         | 1              | B 0.50→0.19, S 0.45→0.23 (deconditioning); A holds then declines    | PASS      | Confirms Banister-overload timescale memory: A inflects at ~day 28                  | (REPL inspection)                              |
| dev-1f  | 2026-05-06 | 1     | healthy  | 42     | n/a  | n/a         | 1              | A 0.45→0.85 plateau; B/S decline 0.50→0.32 (mild detrain at Phi=0.3) | PASS      | Long-horizon healthy is sustainable; capacities slowly decline                       | (REPL inspection)                              |
| 0c-001  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | Trajectory-quality forensics: run19 vs run16 differ in **particle counts**, NOT HMC kernel | INFO | run16 (Stage 2): n_smc=256, n_pf=n/a. run19 (Stage 3): n_smc=128, n_pf=200. HMC mcmc/leapfrog is 10/16 in BOTH. The "smaller" in the run name is particle count, not HMC config. | (manifest + launcher inspection) |
| 0c-002  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | Bug 4 fix (Option A): post-hoc evaluator now uses smooth RBF schedule, not burst-expanded plant input | PASS | Applied to bench_controller_only_fsa_v5.py and bench_smc_full_mpc_fsa_v5.py. Stage 3 bench also stores `mean_schedule` in replan_records now. | version_3/tools/bench_*.py |
| 0c-003  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | sigma_S rename complete in smc2fc (no more workarounds): obs key→`sigma_S_obs`, plant reads from truth_params, 7 guardrail tests rewritten | PASS | DEFAULT_PARAMS_V5['sigma_S']==0.008 (state), ['sigma_S_obs']==4.0 (obs). Plant `_em_step` reads sigmas from `self.truth_params`. estimation.py PARAM_PRIOR_CONFIG / R_diag / likelihood all use new key. 42/42 tests pass. | version_3/models/fsa_v5/{simulation,estimation,_plant}.py |
| 0c-004  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | FSA_model_dev sigma_S rename: BLOCKED — needs explicit user authorization | BLOCKED | Permission system blocked cross-repo edits even though user said "fix everywhere". Files needing the same edits: models/fsa_high_res/{simulation,estimation,_plant}.py + tools/fim_analysis_v5.py + tests/test_obs_consistency_v5.py | /home/ajay/Repos/FSA_model_dev/ |
| 0c-005  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | FSA_model_dev sigma_S rename: COMPLETED after explicit user authorization | PASS | All 5 files renamed (simulation, estimation, _plant, tools/fim_analysis_v5, tests/test_obs_consistency_v5). 17/17 dev-repo tests pass. | /home/ajay/Repos/FSA_model_dev/ |
| 0c-006  | 2026-05-06 | 0+    | n/a      | n/a    | n/a  | n/a         | n/a            | Bug 2 (particle-0 separator template) FIXED in both repos | PASS | Per-particle vmap over A_sep computation; indicator no longer broadcasts a single template. 4 new regression tests use a 2-particle ensemble (healthy + collapsed) to prove the fix. 46/46 smc2fc tests + 17/17 dev-repo tests pass. | both repos: control_v5.py |

## Bugs (status)

| Bug ID | Description                                                          | Status              | Notes                                                                                       |
|--------|----------------------------------------------------------------------|---------------------|---------------------------------------------------------------------------------------------|
| 1      | `sigma_S` dict-literal collision in `simulation.py::DEFAULT_PARAMS`  | **FIXED** (smc2fc only) | Renamed obs key to `sigma_S_obs`. Plant now reads sigmas from `truth_params` (workaround removed). estimation.py R_diag + likelihood + prior use new key. 7 guardrail tests rewritten. **FSA_model_dev rename blocked by permission system — needs explicit authorization.** |
| 2      | Particle-0 separator template in `control_v5.py::_compute_cost_internals` | **FIXED** (both repos) | Per-particle separator now computed via outer-vmap over the SMC² particle axis; A_sep shape changed from (n_steps,) → (n_particles, n_steps). Indicator dropped the `[None, :]` broadcast in both hard and soft variants. 4 new regression tests at `version_3/tests/test_per_particle_separator.py` use a 2-particle ensemble where particle-0 is healthy (-inf A_sep) and particle-1 is collapsed (+inf), proving each particle's indicator now uses its own separator. |
| 3      | TOCTOU race in `_allocate_run_dir` (3 benches)                       | FIXED               | Shared atomic helper at `version_3/tools/_run_dir.py`. 5 race tests added.                  |
| 4      | **Bench post-hoc evaluator wiring**: passes burst-expanded `plant.history['Phi_value']` (median=0) to a per-bin chance-constraint evaluator — produces ~96% vacuous "violations" during rest bins. | **FIXED** (Option A) | Post-hoc now receives the smooth RBF schedule the controller actually optimised against, recomposed from per-replan `mean_schedule` arrays + baseline_phi during warm-up. Stage 3 bench's replan_records now also stores `mean_schedule`. Both `bench_controller_only_fsa_v5.py` and `bench_smc_full_mpc_fsa_v5.py` updated. |
| 5      | ~~Controller theta prior wider than healthy-island width~~ **RETRACTED** | NOT A BUG | Original entry incoherent — conflated θ as a vector of RBF schedule coefficients (∈ ℝ^(2·n_anchors)) with the healthy island as a 2D region in (Φ_B, Φ_S) control space. There's no scalar comparison to make. The wide θ prior is correct: it lets the RBF schedule warp over time. Retracted on Ajay's correction 2026-05-06. |

## Quarantined / Deferred

- `control_v5_fast.py` (the soft_fast trimmed-HMC variant) is **not** to
  be used pending empirical evidence it preserves correctness. Per Ajay
  2026-05-06: full HMC (256/10/16) is the only validated config.
- **Trajectory-quality investigation** (run19 vs run16 basin overlays):
  the difference is **particle counts**, not HMC kernel config (both
  have mcmc=10, leapfrog=16). run19 used n_smc=128 + n_pf=200 to fit
  GPU memory; run16 used n_smc=256 (Stage 2, no filter). To re-test
  this hypothesis, run Stage 3 with full particle counts (n_smc=256,
  n_pf=800) — the launcher comments say this previously OOM'd at
  ~30.85 GB on the 32 GB device, so memory tuning may be needed first.

## Retracted observations

- **Original "Bug 5" (theta prior width vs healthy-island width)**:
  retracted 2026-05-06. The original entry conflated θ (RBF schedule
  coefficients, a vector) with the healthy island (a 2D region in
  control space). Not a coherent comparison. The wide controller-θ
  prior is correct — it lets the RBF schedule shape vary over time.
