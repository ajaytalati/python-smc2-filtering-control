# Plan: Port SWAT model into the SMC²-MPC framework

## Status update (2026-04-30)

**Phase 0 (sync Repos B and C with Repo A) was DROPPED** after a
direct empirical consistency check. The two presentations of SWAT
(Repo A's 7-state estimation form vs Repo C's 4-state control form)
agree to sub-picosecond precision on T(t) trajectories under
matching (V_h, V_n, V_c) values across all three canonical
scenarios. The "inversion" the audit flagged was a deliberate dual
API presentation, not a model-level drift. psim auto-tracks Repo A
via lazy imports.

See [`swat_consistency_check/README.md`](swat_consistency_check/README.md)
for the verification script and numerics.

The simplified path forward is: `git pull` Repo A → Phase 1
(create `version_2/models/swat/`) → Phase 2 (duplicate bench) →
Phase 3 (smoke + scientific validation). All three feasible without
upstream-repo sync work.

---

## First actions on resume

1. **/loop check on T=56** — handle the wakeup that fires when T=56
   finishes; launch T=84 sequentially.
2. **Pull Repo A** to current main (`9b948bf` or later) — the post
   V_h-inversion-fix state.
3. Phase 1 — create `version_2/models/swat/`.

## Context

The user is bringing the SWAT model (7-state sleep/wake/adenosine/
testosterone with Stuart-Landau pulsatility) into the closed-loop
SMC²-MPC framework. Two motivating insights:

1. **SWAT's entrainment threshold is a learnable F_max analog.** In
   FSA-v2, F_max is a hard-coded constraint. In SWAT, the bifurcation
   point E_crit = −μ_0/μ_E is a true mathematical threshold whose
   parameters (μ_0, μ_E) are formally identifiable from the obs
   likelihood when the trajectory straddles the threshold.

2. **SWAT has THREE control variates** (V_h, V_n, V_c) — vitality
   reserve, chronic load, phase shift. Currently constants in the
   model; the user wants them as **time-varying plant inputs** so
   the controller has a 3-D action space (vs FSA-v2's 1-D Φ).

In SWAT, testosterone pulsatility T follows:

```
dT/dt = (1/τ_T) [ μ(E)·T − η·T³ ] + noise
μ(E)  = μ_0 + μ_E · E
E     = amp_W(V_h, V_n) · amp_Z(V_h, V_n) · phase(V_c)
```

E_crit = −μ_0/μ_E = 0.5 currently (frozen μ_0=−0.5, μ_E=1.0). The
port makes both estimable.

## Pre-existing assets — trust hierarchy + sync requirement

The user has built three repos that bear on this port. **They have
different trust levels:**

```
Repo A (canonical model)  →  TRUSTED, port from here
Repo B (psim scenarios)   →  AUDIT, then SYNC to A if stale
Repo C (validation gate)  →  AUDIT, then SYNC to A if stale
```

The user explicitly stated:

> "If repos B or C are stale relative to A — then FIX / UPDATE THEM
> so that all three are consistent. They are my repos so you have
> full access. You have to do work to ensure the model is CORRECT
> in all three BEFORE you can port it into this repo."

This means: **all three repos must agree on the SWAT model** before
the SMC²-MPC port begins. Repo A is the source of truth; Repos B
and C must be updated to match Repo A. We can't just *skip* a stale
repo — we have to *fix* it. Otherwise downstream consumers of the
ecosystem (other people, future work, CI pipelines that vendor by
tag) would be broken.

Scope expansion implied: Phase 0 is no longer "audit + maybe gate"
— it's now "audit + fix-if-stale + commit + push to all three
repos". This is a real engineering effort (~3-5 days for stale
repos) before any porting starts.

### Repo A — `Python-Model-Development-Simulation` — **TRUSTED, source of truth**

Locally at `/home/ajay/Repos/Python-Model-Development-Simulation`.
The user-confirmed canonical SWAT model. Everything we port is
sourced from here. Contains:
- `version_1/models/swat/simulation.py` — 7-state drift + 4-channel
  obs (HR Gaussian, Sleep ordinal 3-level, Steps Poisson, Stress
  Gaussian). Updated 2026-04-26.
- `version_1/models/swat/estimation.py` — JAX `propagate_fn`,
  `align_obs_fn` (2026-04-27).
- `version_1/models/swat/_dynamics.py` — drift re-derived for
  est-parity checks.
- `version_1/models/swat/TESTING.md` — reference scenarios (52 KB).

Port the math from here. (Currently NumPy + JAX hybrid; we'll
JAX-ify any remaining NumPy hot loops, mirroring FSA-v2 Stage N.)

### Repo B — `Python-Model-Scenario-Simulation` (psim) — HIGH VALUE but **needs refreshing**

Locally at `/home/ajay/Repos/Python-Model-Scenario-Simulation`. The
user has previously built a forward-simulation + scenario-packaging
layer for SWAT. **No SMC² inference, no MPC** — but several
artefacts directly reusable AFTER A REFRESH PASS (see below).

**CRITICAL CAVEAT from user**: the four scenario presets stored in
`outputs/swat/set_{A,B,C,D}/` were validated against an **older**
SWAT model. The current safekeeping-repo SWAT has different param
semantics — specifically, the user believes **V_n is now 0** in
scenarios A, C, D (was 0.3 in the old presets), and there may be
other drifts. The stored `outputs/swat/*/trajectory.npz` and
`obs/*.npz` files are **stale**. We cannot consume them as
ground truth for the SMC²-MPC port without verifying.

What's in psim that IS reusable as-is (model-independent):

1. **The four scenario STRUCTURES** (the *what*, not the *truth values*):
   - **Set A — Healthy**: nominal V_h, V_n, V_c → T near
     bifurcation peak.
   - **Set B — Amplitude collapse**: depleted V_h, elevated V_n,
     V_c=0 → E falls below E_crit via amplitude failure → T
     collapses.
   - **Set C — Recovery**: nominal V_h, V_n, V_c=0 but T_0 starts
     near zero → T should rise to peak. **Natural cold-start
     scenario** for closed-loop MPC.
   - **Set D — Phase-shift flatline**: nominal V_h, V_n, V_c=6h
     → E falls below E_crit via phase failure → T collapses.

   The structural distinctions (amplitude vs phase failure modes)
   are still valid for the current model. Only the numerical truth
   values for V_h, V_n need refreshing.

2. **The scenario preset PYTHON FILES** in `psim/scenarios/presets/`
   — re-run them against the current safekeeping SWAT to check if
   V_n etc. need updating. If yes, fix the presets in psim (one of
   the OUT-OF-SCOPE-but-easy follow-ups), and regenerate the
   outputs.

3. **The test patterns** are model-independent:
   - `tests/test_scenario_swat.py` — shape checks.
   - `tests/test_consistency_swat.py` — drift parity, obs-prediction
     parity per channel.
   - `tests/test_round_trip_swat.py` — synthesise → propagate
     zero-noise → recover state.

2. **Three test patterns to mirror** in our SWAT port:
   - `tests/test_scenario_swat.py` — shape checks, all 35 estimable
     params filled exactly.
   - `tests/test_consistency_swat.py` — **drift parity** (simulator
     vs est-side), **obs-prediction parity** per channel.
   - `tests/test_round_trip_swat.py` — synthesise → propagate
     zero-noise → recover state, tolerance 3.0 abs (loose because
     SWAT uses Euler-Maruyama 4-substep vs IMEX 1-step).

3. **Orchestrator pattern** in `examples/swat/_common.py:33–117` —
   sys.path injection, scenario synthesis, validation, packaging.
   Reuse the `synthesise_scenario(...)` call as ground-truth
   generator for our closed-loop tests.

### Repo C — `Python-Model-Validation` — **AUDIT, SYNC IF STALE**

GitHub-only at `https://github.com/ajaytalati/Python-Model-Validation`.
Per the user, this repo's validation code MAY have been written
against an older SWAT version. **If stale, we must update Repo C
in place** so its validation code targets the current Repo A SWAT.

The audit + sync workflow:

1. `git clone https://github.com/ajaytalati/Python-Model-Validation.git
   ~/Repos/Python-Model-Validation` on a fresh feature branch
   `feat/sync_swat_to_current_repo_A`.
2. Read its SWAT-related files:
   - `tests/swat/` — pytest suite
   - `identifiability/swat/compute_fim.py` — Fisher-info rank check
   - `stability/swat/corner_case_sweep.py` — Lyapunov sweep
   - `snapshots/manifest.json` — vendored SWAT commit registry
3. Compare against current Repo A SWAT:
   - Does each test import the correct API surface?
   - Does each test reference current parameter names + truth values?
   - Does the snapshot manifest list a recent SWAT commit?
4. **If stale** — fix:
   - Update test imports, parameter references, expected values to
     match current Repo A SWAT.
   - Update snapshot manifest to point at the latest Repo A commit.
   - Run the suite, confirm green.
   - Commit on the `feat/sync_swat_to_current_repo_A` branch.
   - Open PR for user review.
   - Merge when approved; the auto-tagger publishes a new
     `swat-validated-<date>-<sha>` tag.
5. **If current** — just run the suite + tag.

After this step, Repo C's tests become the formal gate for the
SMC²-MPC port: if they pass, the underlying SWAT model is
structurally sound and we're free to port.

### What we still need to build new

- `version_2/models/swat/` directory in
  python-smc2-filtering-control (does not exist yet).
- SWAT-side `_plant.py` (StepwisePlant) — psim does not have a
  closed-loop stateful simulator; mirror the FSA-v2 pattern.
- SWAT-side `control.py` (3-D control variates).
- SWAT-side bench tool by duplicating
  `bench_smc_full_mpc_fsa.py`.

The `smc2fc/` framework itself is model-agnostic (verified earlier
in this session) — no changes needed there.

## The three control variates (per user's framing)

Currently in SWAT (per the safekeeping repo's Phase-1 model
assumption), V_h, V_n, V_c are estimable but **constant in time**.
The user's proposal is to make all three **time-varying plant
inputs** that the MPC manipulates:

| variate | physiological role | control-knob analog |
|---|---|---|
| V_h (vitality reserve) | underlying arousal capacity | training, recovery, nutrition — controller raises V_h |
| V_n (chronic load) | tonic stress / inflammation | stress mgmt, CBT, anti-inflammatory — controller lowers V_n |
| V_c (phase shift, h) | circadian misalignment | light therapy, sleep scheduling — controller pushes V_c → 0 |

This makes the controller's action space **3-D**, mirroring
clinical multi-modal intervention. psim's Set B and Set D show the
two failure modes (amplitude vs phase) the controller must learn
to address with the right variate.

### Coupling-to-FSA-v2 mapping (future work, out of scope here)

The user's directional analogy:
- **V_h (SWAT) ~ B (FSA-v2)** — both slow accumulating "good state".
- **V_n (SWAT) ~ F (FSA-v2)** — both pathological / fatigue-like.
- **|V_c| (SWAT) ~ |F| (FSA-v2)** — phase misalignment is also
  fatigue-like.

Hierarchical coupling: FSA-v2 (training → B, F) drives SWAT's
potentials (V_h ← B, V_n ← F). The MPC then jointly controls Φ
(FSA input) and V_c (SWAT input). This is the bigger scientific
aim, prerequisite to which is SWAT-standalone closed-loop working.
**Out of scope for this port.**

## Scope: Phase 1-3 only

Per user: "I think just getting up to Phase 3 of the plan is a
massive achievement". Phase 4 (multi-subject E_crit ablation) and
all FSA-v2 ↔ SWAT coupling deferred.

### Phase 0 — Sync Repos B and C with Repo A (3-5 days)

Before any port work in Repo D, ensure all three upstream repos
agree on the current SWAT model. The user has full ownership; we
have authority to push fixes.

#### Phase 0a — Sync Repo C (Python-Model-Validation)

1. Clone Repo C locally on branch
   `feat/sync_swat_to_current_repo_A`:
   ```bash
   cd ~/Repos
   git clone https://github.com/ajaytalati/Python-Model-Validation.git
   cd Python-Model-Validation
   git checkout -b feat/sync_swat_to_current_repo_A
   ```
2. Audit each SWAT file:
   - `tests/swat/test_*.py`
   - `identifiability/swat/compute_fim.py`
   - `stability/swat/corner_case_sweep.py`
   - `snapshots/manifest.json`
   - `.github/workflows/swat_validation.yml`
3. For each file with stale references to old SWAT (param names,
   truth values, API signatures):
   - Update imports to current Repo A API.
   - Update expected values to match current
     `Python-Model-Development-Simulation/version_1/models/swat/simulation.py:DEFAULT_PARAMS`.
   - Re-run tests locally; iterate until green.
4. Update `snapshots/manifest.json` to vendor the latest Repo A
   SWAT commit (the canonical SHA from
   `~/Repos/Python-Model-Development-Simulation/.git/refs/heads/main`
   or wherever the SWAT lives).
5. Run the full SWAT validation suite end-to-end:
   ```bash
   pytest tests/swat/ -v
   python identifiability/swat/compute_fim.py --report
   python stability/swat/corner_case_sweep.py
   ```
   All three must pass.
6. Commit + push the branch. Open a PR using `gh pr create`.
7. Wait for user merge approval (or ask explicitly), then merge.
8. Post-merge, confirm the CI auto-tag fires and a new
   `swat-validated-<date>-<sha>` tag exists. Note the tag — Repo D
   port will pin to this validated version.

If Repo C is already current → steps 3-4 are no-ops; just run + tag.

#### Phase 0b — Sync Repo B (psim) scenarios + tests

1. In `~/Repos/Python-Model-Scenario-Simulation`, create branch
   `feat/sync_swat_to_current_repo_A`.
2. Read all SWAT-touching files:
   - `psim/scenarios/presets/swat_set_{A,B,C,D}_*.py`
   - `tests/test_scenario_swat.py`
   - `tests/test_consistency_swat.py`
   - `tests/test_round_trip_swat.py`
   - `examples/swat/14d_set_*.py`
   - `examples/swat/_common.py`
3. Compare truth-param tuples in the preset files vs. current
   Repo A `DEFAULT_PARAMS`. The user expects V_n = 0 in A/C/D
   (was 0.3 before). Verify and fix.
4. Update test files for any API drift.
5. Run consistency tests; iterate until green:
   ```bash
   pytest tests/test_consistency_swat.py -v
   pytest tests/test_round_trip_swat.py -v
   pytest tests/test_scenario_swat.py -v
   ```
6. Re-run the four scenarios to regenerate outputs:
   ```bash
   python examples/swat/14d_set_A_healthy.py
   python examples/swat/14d_set_B_amplitude.py
   python examples/swat/14d_set_C_recovery.py
   python examples/swat/14d_set_D_phase_shift.py
   ```
   Verify each scenario produces the qualitative outcome it claims
   (healthy / amplitude collapse / recovery / phase-shift flatline).
7. Commit + push. Open PR.
8. Wait for user merge approval, then merge.

If Repo B is already current → steps 3-4 are no-ops; just regenerate
outputs from clean current SWAT.

#### Phase 0c — Verify ecosystem consistency

After both sync PRs are merged:
- Repo A unchanged (was already canonical).
- Repo B: scenarios + tests up to date with Repo A.
- Repo C: validation gate green against Repo A; latest tag exists.

Now and only now, the SMC²-MPC port begins.

### Phase 1 — Create `version_2/models/swat/` (5-7 days)

Mirror FSA-v2's file structure but with SWAT dynamics + V_h, V_n,
V_c as time-varying inputs.

| file | content |
|---|---|
| `__init__.py` | empty |
| `_dynamics.py` | JAX `drift_jax(y, params, V_h_t, V_n_t, V_c_t)` — three time-varying inputs. Diffusion. `TRUTH_PARAMS`. State `y = [W, Zt, a, T]`. Adapted from safekeeping `_dynamics.py` + JAX-ifying any remaining NumPy. |
| `_v_schedule.py` | (analog of FSA's `_phi_burst.py`) — sub-daily expansion of three daily schedules onto the bin grid. |
| `simulation.py` | Top-level forward sim; circadian forcing C(t); 4 obs samplers (HR Gaussian, Sleep ordinal, Steps Poisson, Stress Gaussian). |
| `_plant.py` | `StepwisePlant.advance(stride_bins, V_h_daily, V_n_daily, V_c_daily)` returning per-stride obs dict. |
| `estimation.py` | `EstimationModel` instance `SWAT_ESTIMATION` with sequential-scalar Kalman fusion for Gaussian channels (HR, Stress) + particle-based likelihood for Poisson (Steps) and ordinal (Sleep). **Make μ_0 and μ_E estimable** — adds 2 dims (35 → 37 estimable). |
| `control.py` | `ControlSpec` with `cost_fn(theta) = -∫T dt + λ_h·(V_h cost) + λ_n·(V_n cost) + λ_c·(V_c cost)`. `schedule_from_theta` decodes 3 RBF anchor sets. `theta_dim = 3 × n_anchors_per_variate`. |

### Phase 2 — Duplicate the bench tool (1 day)

Copy `version_2/tools/bench_smc_full_mpc_fsa.py` →
`bench_smc_full_mpc_swat.py`. Changes:
- Imports point at `models.swat`.
- `identifiable_subset` switches to SWAT-specific params (per
  identifiability extension: α_HR, σ_HR, τ_W, τ_Z, β_Z, τ_a,
  λ_step, W*, λ_base, μ_0, μ_E).
- Acceptance gates:
  - `mean_T_mpc ≥ 0.95 × mean_T_no_intervention`
  - `id-cov ≥ 30/N` (37 estimable, 6/6 thresholds analog)
  - `|V_c|_max stays within tolerance`
  - "F-violation" analog: stay above E_crit (no T collapse)
- Output dirs: `swat_T<N>d_...`.

Future refactor (out of scope): `--model {fsa_v2, swat}` CLI flag.

### Phase 3 — Smoke + scientific validation (3-5 days)

Reuse psim's test patterns + scenario presets:

1. **Drift / obs parity**: mirror `psim/tests/test_consistency_swat.py`
   between our new `simulation.py` and `estimation.py`.
2. **Round-trip**: zero-noise propagate, recover state to within
   psim's 3.0-abs tolerance for Zt.
3. **Forward-sim smoke**: T=2 day on Set C (recovery). Verify
   trajectory matches psim's Set C output to MC noise.
4. **Filter smoke**: T=14 cold-start on Set C synthetic data
   (psim has it pre-generated in `outputs/swat/set_C_recovery/`).
   Verify all 37 params get reasonable posteriors; id-cov ≥ 30/37.
5. **Closed-loop smoke**: T=14 MPC run on **Set D** (phase-shift
   flatline) initial conditions, **using whichever ground truth
   source survived Phase 0c** (refreshed psim outputs OR fresh
   in-port scenario presets). Verify the controller:
   - Drives V_c trajectory → 0 (re-entrainment).
   - Respects V_h and V_n constraints.
   - Recovers T pulsatility past E_crit by end of horizon.
6. **F_max-from-data, single subject** (T=14): make μ_0 truly
   estimable on a Set B (amplitude collapse) initial condition.
   Verify SMC² posterior on μ_0 recovers truth (single subject —
   Phase 4 multi-subject is deferred).

This Phase 3 is the "massive achievement" the user named.

## Future work — principled identifiability reparametrization (NOT current Phase 3)

The current SWAT estimation.py pins `tau_T = 2.0 days` and
`lambda_amp_Z = 8.0` in FROZEN_PARAMS to break two structural
identifiability degeneracies surfaced by Repo C's FIM analysis
(rank 24/27 → 25/25 after pinning). **Pinning is a lazy fix.** The
principled fix is to reparametrize the model so the redundant
directions don't exist.

### Reparametrization plan (Repo A model-level change)

**1. Stuart-Landau time-rate absorption**

Current drift:
```
dT/dt = (mu_0 + mu_E * E) * T / tau_T  -  eta * T**3 / tau_T
```

Reparametrized:
```
dT/dt = mu_0_tilde * T  +  mu_E_tilde * E * T  -  eta_tilde * T**3
```
with `mu_0_tilde := mu_0 / tau_T`, etc. Three parameters instead
of four; tau_T disappears entirely as a meaningful parameter.
Mirrors the FSA-v2 G1 reparametrization pattern (kappa_B^eff,
mu_F^eff, etc.).

**2. Entrainment-amplitude V_h response collapse**

Current:
```
A_W = lambda_amp_W * V_h    (independent gains)
A_Z = lambda_amp_Z * V_h
E ~ amp_W * amp_Z           (only the product is identifiable)
```

Reparametrized (assume symmetric V_h response):
```
A_W = lambda_amp * V_h
A_Z = lambda_amp * V_h      (single gain)
```

One parameter instead of two. Defensible biologically — V_h is
"general vitality reserve" and should boost both flip-flop sides
equally.

### Implementation sequence

This is a **Repo A model change**, not just an estimation-side fix.
Steps:

1. Update `Python-Model-Development-Simulation/version_1/models/swat/_dynamics.py`
   drift formula and `simulation.py:PARAM_SET_*` truth values.
2. Re-run Repo C `identifiability/swat/compute_fim.py` to confirm
   the reparametrized model is full rank (no pins needed).
3. Re-run Repo B (psim) consistency tests + regenerate Sets A/B/C/D
   outputs.
4. Update the bundled tag in
   `Python-Model-Validation/snapshots/manifest.json`.
5. Re-pull SWAT into our port at the new validated tag; remove
   `tau_T` and `lambda_amp_Z` from FROZEN_PARAMS, no replacement
   priors needed (the reparametrization eliminated them).

### Why this is deferred

Doing the reparametrization properly affects every consumer of
SWAT — Repo A, Repo B (psim), Repo C (validation), our SMC²-MPC
port, and the OT-Control engine. It needs careful staged rollout.

For Phase 3 validation in the SMC²-MPC framework, the **pinning
approach is sufficient** because:
- The FIM is full rank (25/25) under pinning.
- mu_0 and mu_E are individually identifiable, which is what the
  F_max-from-data experiment requires.
- All downstream consumers see the same pinned values
  (they're already in Repo A's PARAM_SET_A truths).

The pinning gets us across the line for Phase 3. The reparametrization
is what we'd ship before publication.

## Out of scope

- **Phase 4 multi-subject E_crit validation** (recover E_crit
  across subjects with different μ_0).
- **FSA-v2 ↔ SWAT coupling** (V_h ← B, V_n ← F joint control).
- **Bench tool refactor for `--model` flag**.
- **Real-data integration** — synthetic SWAT trajectories only.
- **Updates to Repo A (`Python-Model-Development-Simulation`).**
  Repo A is treated as canonical; we do not edit it in this plan.
  If Phase 0 surfaces a genuine model bug in Repo A, that's a
  separate fix-and-publish cycle (Repo A fix → ripple through B
  and C → resume here).
- **Updates to Repos B and C beyond what's needed for Repo A
  parity.** Phase 0a/b is *minimum-viable sync*: bring B and C in
  line with current A, no broader feature work. New scenario
  presets, new validation tests etc. → separate plan.
- **Adding new SWAT model variants or extensions.** This plan only
  ports the existing Repo A SWAT. Future model upgrades (e.g.
  V_h/V_n made dynamic in Repo A itself, real-data integration)
  → separate plan.

## Risks + mitigations

| risk | mitigation |
|---|---|
| Poisson + ordinal obs likelihoods are new (FSA-v2 was Gaussian + log-Gaussian only). | Hybrid: sequential-scalar Kalman fusion for Gaussian channels + particle-based likelihood for Poisson (Steps) and ordinal (Sleep). Pattern already in safekeeping `estimation.py`. |
| Longer timescale (τ_T = 48 h) needs longer windows for identifiability. | Run T=14 minimum. Per identifiability extension, ≥ 8 days needed. |
| (V_h, V_n) rank-deficient pair without stress channel. | Stress channel included from the start. |
| (μ_0, μ_E) only identifiable when E spans regimes — if controller keeps subject solidly healthy, threshold is unobservable. | Use Set B (amplitude collapse) and Set D (phase shift) presets as initial conditions — these naturally start the trajectory ON the wrong side of E_crit, forcing the data to span both regimes. |
| 3-D control surface (V_h, V_n, V_c) is bigger than FSA-v2's 1-D. Tempering convergence may slow. | Start K=2 daily replan + n_smc=1024. Bump n_smc → 2048 only if posterior over schedule is wobbly. |
| Validation gate (Repo C) could fail on the safekeeping SWAT version — wasted port effort. | Run validation FIRST (Phase 0). Fix in safekeeping repo before porting if it fails. |

## Critical files

**To create**:
- `project_upgrade_plans/Port_SWAT_into_SMC2_MPC_framework.md`
  (this plan, exactly)
- `version_2/models/swat/__init__.py`
- `version_2/models/swat/_dynamics.py`
- `version_2/models/swat/_v_schedule.py`
- `version_2/models/swat/simulation.py`
- `version_2/models/swat/_plant.py`
- `version_2/models/swat/estimation.py`
- `version_2/models/swat/control.py`
- `version_2/tools/bench_smc_full_mpc_swat.py`
- `version_2/tools/launchers/run_swat_horizon.sh`
- `version_2/tests/test_swat_smoke.py` — drift/obs parity +
  round-trip, mirroring psim patterns.

**To clone before Phase 0**:
- `Python-Model-Validation` →
  `/home/ajay/Repos/Python-Model-Validation/`.

**To read for reference (do not modify)**:
- `Python-Model-Development-Simulation/version_1/models/swat/*.py`
- `Python-Model-Scenario-Simulation/examples/swat/_common.py`
- `Python-Model-Scenario-Simulation/tests/test_*_swat.py` —
  test patterns to mirror.
- `Python-Model-Scenario-Simulation/outputs/swat/set_{A,B,C,D}/` —
  ground-truth scenarios for closed-loop validation.
- `Python-Model-Development-Simulation/version_1/model_documentation/swat/SWAT_Identifiability_Extension.md`
  — rank-deficient pair documentation.

## Verification

End-to-end sequence:

```bash
# Phase 0: validation gate
cd ~/Repos && git clone https://github.com/ajaytalati/Python-Model-Validation.git
cd Python-Model-Validation && pytest tests/swat/ -v
python identifiability/swat/compute_fim.py --report
python stability/swat/corner_case_sweep.py
# all must pass before Phase 1.

# Phase 1+2: smoke
cd ~/Repos/python-smc2-filtering-control/version_2
PYTHONPATH=.:.. python -c "
from models.swat._plant import StepwisePlant
from models.swat.simulation import DEFAULT_PARAMS, DEFAULT_INIT, BINS_PER_DAY
import numpy as np
plant = StepwisePlant(seed_offset=42)
result = plant.advance(2 * BINS_PER_DAY, np.zeros(2), np.zeros(2), np.zeros(2))
print('OK', result['trajectory'].shape)
"

# Phase 3: drift parity / round-trip / closed-loop
pytest version_2/tests/test_swat_smoke.py -v
$HOME/bench_logs/run_swat_horizon.sh 14
# expect: id-cov ≥ 30/37, V_c → 0 by end (Set D init),
#         T recovers past E_crit, μ_0 posterior covers truth.
```

## Concurrent task

T=56 GPU sweep continues. The /loop wakeup at 17:09 fires
independently. SWAT port work is CPU + git, no GPU competition.
