# Importing FSA-v5 model — model-specific notes

> Companion document to `Importing_FSA_version_5_model_2026-05-05_0859.md`. **Read both together** — the other file is the broad process plan; this file is the model-specific notes from the dev-sandbox author.

I'm the prior-session Claude that built the FSA-v5 dev sandbox at `github.com/ajaytalati/FSA_model_dev` on the `claude/dev-sandbox-v4` branch (commits `a1f0638` → `d8f20c6`, 2026-05-04 → 2026-05-05). Below are six model-specific things that surfaced during that work and are NOT covered by the broad plan. None of these contradict it; they're all additions.

---

## 1. File-copy mechanism — the rename + the import-path footgun

The broad plan says "copied from FSA_model_dev / dev-sandbox-v4" but doesn't spell out the mechanics. Three things to know:

### 1.1 Folder rename

The dev sandbox has the model files in `models/fsa_high_res/`. The broad plan's `version_3/models/fsa_v5/` is a **rename**, not a 1:1 copy. After copying, every internal import line of the form

```
from models.fsa_high_res.<x> import ...
```

inside the copied files becomes

```
from version_3.models.fsa_v5.<x> import ...
```

There are about a dozen of these across `_dynamics.py`, `_plant.py`, `simulation.py`, `estimation.py`, `control.py`, `control_v5.py`, and `__init__.py`. Grep for `models.fsa_high_res` after copying.

### 1.2 Two control files, not one

The dev sandbox has BOTH:

- `control.py` — legacy gradient-OT controller, kept as a comparison baseline.
- `control_v5.py` — the v5 main novelty: the chance-constrained cost evaluator (`evaluate_chance_constrained_cost`, `find_A_sep_v5`).

**Both need to be copied.** The bench should import its production cost evaluator from `control_v5.py`, not `control.py`.

### 1.3 The `simulator/` import-path footgun

Inside `simulation.py`, the canonical import line on `claude/dev-sandbox-v4` reads:

```
from smc2fc.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec, DIFFUSION_DIAGONAL_STATE)
```

This was fixed during the dev-sandbox-v4 work — earlier snapshots had `from simulator.sde_model import ...` (a relative import to a now-deleted bundled `simulator/` folder). If for any reason an older copy is picked up, the agent will see `ModuleNotFoundError: No module named 'simulator'`. Fix: change `simulator.sde_model` to `smc2fc.simulator.sde_model`.

This same swap also applies to any other file that does `from simulator.sde_solver_diffrax import solve_sde_jax` (a few of the dev-sandbox tests/examples did historically). Grep `from simulator` after copying.

---

## 2. Pin the import to a specific dev-sandbox-v4 commit

The `claude/dev-sandbox-v4` branch is unmerged into the dev repo's main, and may continue to evolve. A fresh agent six weeks from now might pick up unrelated later edits.

**Pin to `d8f20c6` (2026-05-05 00:05 BST)** for this import. That commit was the state that produced 13/13 passing tests — the recommended snapshot to copy from.

Suggested workflow:

```bash
mkdir -p /tmp/fsa_v5_import
cd /tmp/fsa_v5_import
git clone --depth 1 -b claude/dev-sandbox-v4 \
    https://github.com/ajaytalati/FSA_model_dev
cd FSA_model_dev
git rev-parse HEAD             # confirm SHA matches d8f20c6
                               #   (or document whichever SHA you take)
# … then copy models/fsa_high_res/* → version_3/models/fsa_v5/
#   and tests/test_*_v5*.py    → version_3/tests/
```

Record the SHA you actually copied from in the new `version_3/outputs/fsa_v5/CHANGELOG.md`'s first entry, so future audits can trace back to the exact source.

---

## 3. FSA-v5 footguns from the technical guide §9

The v5 technical guide (`LaTex_docs/FSA_version_5_technical_guide.tex` on `claude/dev-sandbox-v4`) has a §9 "Common bugs and gotchas" section listing real bugs that took non-trivial effort to surface. Three of them are likely to bite a fresh agent writing benches for the first time:

### 3.1 The `sigma_S` name collision (guide §9.1)

`params['sigma_S']` returns the **stress-observation noise** (~4.0), NOT the latent-S Jacobi diffusion scale (~0.008). The latent-S diffusion is hard-coded as `_dynamics.SIGMA_S_FROZEN` (= 0.008) and read from there directly by the plant + estimator. Any new bench code that reads diffusion scales from the params dict will silently get the wrong number — the strength state's noise will be ~500× too large.

Pattern to follow: read diffusion scales from `_dynamics.SIGMA_*_FROZEN` constants, never from `params`.

### 3.2 Plant `truth_params` must be `DEFAULT_PARAMS_V5`, not `TRUTH_PARAMS_V5` (guide §9.5)

The dev sandbox exposes both:

- `TRUTH_PARAMS_V5` (in `_dynamics.py`) — dynamics + diffusion keys only.
- `DEFAULT_PARAMS_V5` (in `simulation.py`) — dynamics + diffusion + observation coefficients.

The plant calls the obs samplers (`gen_obs_hr`, etc.), which read obs coefficients (`HR_base`, `kappa_B_HR`, `k_C`, etc.) from `self.truth_params`. If you initialise the plant with `truth_params=TRUTH_PARAMS_V5`, the obs samplers `KeyError` on the missing observation keys.

Pattern to follow:

```python
plant = StepwisePlant(truth_params=dict(DEFAULT_PARAMS_V5), ...)
# or, if you want to override dynamics from TRUTH_PARAMS_V5 explicitly:
plant = StepwisePlant(
    truth_params={**DEFAULT_PARAMS_V5, **TRUTH_PARAMS_V5}, ...)
```

### 3.3 Particle parameter dicts must include the v5 Hill keys (guide §9.8)

The v5 deconditioning extension adds five Hill keys to the canonical drift: `B_dec`, `S_dec`, `mu_dec_B`, `mu_dec_S`, `n_dec`. These live in the estimator's `_FROZEN_V5_DYNAMICS` dict, NOT in `PARAM_PRIOR_CONFIG` (which only has the 37 estimable parameters).

If the SMC² engine carries θ-particles whose dicts contain only the estimated keys, then any direct call to `drift_jax` from the bench (e.g. for a counterfactual baseline rollout, or a pre-bench sanity check) will `KeyError` on the missing Hill keys.

Pattern to follow — always merge the frozen dict in before calling `drift_jax`:

```python
from version_3.models.fsa_v5.estimation import _FROZEN_V5_DYNAMICS

p_full = {**_FROZEN_V5_DYNAMICS, **estimated_dict}
y_next = y + dt * drift_jax(y, p_full, Phi_t)
```

The estimator's `propagate_fn_v5` and the chance-constrained cost evaluator in `control_v5.py` already do this internally; the issue only bites code paths that call `drift_jax` directly.

---

## 4. Structural changes from FSA-v2 → FSA-v5 the bench will need to handle

The broad plan says "mirror `version_2/tools/bench_smc_full_mpc_fsa.py`" but doesn't list what's structurally different in v5. A copy-paste from v2 will not run; the table below shows what changes:

| | FSA-v2 (template) | FSA-v5 (target) |
|---|---|---|
| Latent state | 3D `[B, F, A]` | **6D** `[B, S, F, A, K_FB, K_FS]` |
| Control | scalar Φ | **bimodal** `Φ = (Φ_B, Φ_S)` |
| Obs channels | 4 | **5** (added VolumeLoad) |
| Plant `Phi_daily` arg | shape `(n_days,)` | shape `(n_days, 2)` |
| Cost functional | `−E[∫T dt]` | `evaluate_chance_constrained_cost(...)` returns a **dict**, not a scalar |
| Posterior-mean init for next window | 3 entries | **6 entries** |
| Diagnostic plot panels | 3 latent + 4 obs | **6 latent + 5 obs** |

Two specific bench-side consequences of these structural changes:

### 4.1 Controller schedule shape

The controller's `schedule_from_theta(theta)` callable must now produce an output of shape `(n_steps, 2)`, not `(n_steps,)`. The plant's `advance(stride_bins, Phi_daily)` accepts both shapes (auto-promotes 1D to 2D by zero-padding `Phi_S`), but you generally do **not** want the auto-promote — you want both `Φ_B` and `Φ_S` to be controllable. Make sure your RBF schedule has output dim 2.

### 4.2 The chance-constrained cost return-value contract

`evaluate_chance_constrained_cost(theta_particles, weights, Phi_schedule, ...)` returns a dict with these fields:

```
{'mean_effort':                    float,           # ∫ ‖Φ‖² dt
 'mean_A_integral':                float,           # weighted-mean ∫ A_t dt
 'violation_rate_per_particle':    (n_particles,),
 'weighted_violation_rate':        float,
 'satisfies_chance_constraint':    bool,
 'satisfies_target':               bool,
 'A_sep_per_bin':                  (n_steps,)}
```

The SMC² outer loop must combine these into a single scalar score for the tempered re-weighting. Two clean options:

- **The natural pairing** (per guide §4): use `weighted_violation_rate` to re-weight or reject `θ`-particles whose simulated trajectories violate the chance constraint at rate > α; use `mean_A_integral` for the score.
- **The Lagrangian relaxation** (guide §4.5): if you need a smooth differentiable scalar (e.g. for an SMC² warmstart), use

  ```
  cost = λ_Φ · mean_effort − mean_A_integral
       + λ_chance · max(0, weighted_violation_rate − α)²
  ```

The guide gives both. Decide which one based on whether your outer loop needs gradients.

---

## 5. Reuse the existing dev-sandbox-v4 tests instead of rewriting them

`claude/dev-sandbox-v4` (commit `d8f20c6`) already has 13 passing tests covering the v5 model:

| File | Tests | Catches |
|---|---|---|
| `tests/test_fsa_v5_smoke.py` | 4 | imports clean, plant forward pipeline, propagate_fn runs, chance-constrained cost runs |
| `tests/test_obs_consistency_v5.py` | 6 | one per obs channel (HR / Stress / Steps / VolumeLoad / Sleep) plus a `HR_base` regression sweep — pins each formula bit-equivalently between simulator and estimator. Catches D1/D2-class drift bugs. |
| `tests/test_reconciliation_v5.py` | 2 | plant ↔ estimator drift parity (bit-equivalent Euler step, < 1e-10) + plant 1-bin smoke |

**Recommendation:** copy these three files verbatim into `version_3/tests/`, changing only the import paths (`from models.fsa_high_res.*` → `from version_3.models.fsa_v5.*`). Then add your own bench-level smoke test (the broad plan's "test_fsa_v5_smoke.py" sketch) on top.

This saves ~500 lines of test code that would otherwise have to be rewritten by hand. The reconciliation test, in particular, is the single most valuable regression net for keeping the closed-loop bench correct under future framework changes — copying it is essentially free.

If you copy the dev-sandbox `tests/conftest.py` along with the test files, drop it (or trim it). Its `collect_ignore` list excludes legacy v2/v3 test files that won't exist in `version_3/tests/` anyway.

---

## 6. Spell out the minimum smc2fc-facing surface for the bench's `__init__.py`

The dev-sandbox `models/fsa_high_res/__init__.py` re-exports ~25 names including the v4 back-compat aliases. A copy-paste will pull in v4 names the version_3 bench doesn't need, plus v3 / v4 estimation models that aren't imported anywhere downstream.

The minimum surface the bench actually needs from `version_3.models.fsa_v5`:

```python
# Forward simulator + grid constants + circadian
HIGH_RES_FSA_V5_MODEL, DEFAULT_PARAMS_V5, DEFAULT_INIT,
BINS_PER_DAY, DT_BIN_DAYS, circadian, circadian_jax,

# Canonical dynamics (used by plant + estimator + bench's
# counterfactual baseline, if any)
TRUTH_PARAMS_V5, drift_jax, diffusion_state_dep,

# Inference
HIGH_RES_FSA_V5_ESTIMATION,

# Closed-loop plant
StepwisePlant,

# v5 cost evaluator + separatrix root-finder
evaluate_chance_constrained_cost, find_A_sep_v5,
```

Drop the `HIGH_RES_FSA_V4_*` and `HIGH_RES_FSA_V4_ESTIMATION` aliases from the re-export list unless v4 is also being imported into `version_3` (which the broad plan does not call for). Keeping unused aliases on the import surface costs nothing technically but adds noise to the public API for a downstream reader trying to figure out what's really exported.

---

That's all I have. Items 7+ that I'd also drafted (the conftest.py note, the JAX_ENABLE_X64 environment-variable note) were dropped per the user's explicit direction — the latter was too technical for inclusion here, and the former is covered indirectly by the "drop or trim conftest.py" line in §5.
