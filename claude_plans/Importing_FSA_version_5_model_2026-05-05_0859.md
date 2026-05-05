# Importing FSA version 5 model into version_3

> Plan written 2026-05-05 08:59 by the SWAT-debugging-session Claude in response to Ajay's note `Ajays Notes 1- Importing the FSA version 5 model.md`. Hand this to a fresh Claude session that will do the actual import.

## Context for the incoming agent

Ajay (the user) is bringing a new, more advanced FSA-v5 model into the
SMC² filtering + control framework. The model itself has already been
written, validated, and documented in a dev-sandbox repo. Your job is
**not to design or modify the model** — it's to wire the existing model
files into the framework's bench infrastructure so the closed-loop
SMC²-MPC bench can drive it.

This mirrors what was already done for `fsa_high_res` (the v2 FSA
model) and `swat` (the in-development SWAT model). Follow those as
templates. The methodology was hardened in the SWAT debugging session
(2026-05-04 to 2026-05-05) and is captured in the docs in
`claude_plans/`.

## Read these first, in this order

1. **[`CLAUDE.md`](../CLAUDE.md)** at the repo root — junior-engineer
   stance, "verify before assert", senior-files principle, plans-archive
   policy, conda env (`comfyenv`), `PYTHONPATH=.:..` invocation pattern,
   per-model gotchas. **This is mandatory reading before you touch any
   code.** Don't skip.
2. **`claude_plans/controller_only_test_methodology.md`** — debug the
   controller without the filter. ~3-4× faster iteration. Use this for
   any controller debugging once the model is wired up.
3. **`claude_plans/SWAT_controller_production_validated_2026-05-05_0751.md`**
   — final outcome of the SWAT debugging session. Demonstrates the
   senior-files principle in action and the typical fix-candidate flow
   (D1–D6 reasoning).
4. **The FSA-v5 LaTeX technical guide** (in the dev-sandbox repo, link
   below) — read end-to-end before you start writing benches. The bench
   needs to know the obs channels, control variates, default params,
   scenario presets.

## Sources

- **Model code** lives at
  [`https://github.com/ajaytalati/FSA_model_dev/tree/claude/dev-sandbox-v4`](https://github.com/ajaytalati/FSA_model_dev/tree/claude/dev-sandbox-v4)
  on the `claude/dev-sandbox-v4` branch. **You do not modify these
  files.** They are the senior decision (validated against
  identifiability, stiffness, plant-vs-estimator reconciliation,
  likelihood, and controller checks in the dev sandbox before being
  released for import).
- **Technical guide** at
  [`LaTex_docs/FSA_version_5_technical_guide.tex`](https://github.com/ajaytalati/FSA_model_dev/blob/claude/dev-sandbox-v4/LaTex_docs/FSA_version_5_technical_guide.tex)
  — the canonical model documentation. Read it before writing any code.

## Setup steps

```bash
cd ~/Repos/python-smc2-filtering-control
git fetch origin
git checkout master
git pull
git checkout -b importing_FSA_version_5
conda activate comfyenv
```

Do NOT branch off `feat/import-swat-from-dev-repo` — that branch carries
the SWAT-specific work and would muddy your FSA-v5 PR diff. Branch off
`master` (which now has the framework-level `gk_dpf_v3_lite.py`
per-particle PRNGKey extension and `CLAUDE.md`).

## Layout — make a fresh `version_3/` subtree

This is Ajay's explicit instruction: FSA-v5 lives in `version_3/`, not
in `version_2/`. Mirror `version_2/`'s layout:

```
version_3/
├── __init__.py
├── models/
│   ├── __init__.py
│   └── fsa_v5/
│       ├── __init__.py
│       ├── simulation.py        # ← copied from FSA_model_dev / dev-sandbox-v4
│       ├── _dynamics.py         # ← copied
│       ├── _plant.py            # ← copied
│       ├── estimation.py        # ← copied
│       ├── control.py           # ← copied
│       └── sim_plots.py         # ← copied (if it exists in dev-sandbox-v4)
├── tools/
│   ├── bench_smc_full_mpc_fsa_v5.py        # ← YOU write this
│   └── bench_controller_only_fsa_v5.py     # ← YOU write this
├── tests/
│   └── test_fsa_v5_smoke.py                # ← YOU write a smoke test
└── outputs/
    └── fsa_v5/
        ├── .gitkeep
        └── CHANGELOG.md                    # ← start an empty audit trail
```

## Your job — model-specific files NOT in `models/`

The model files (`simulation`, `_dynamics`, `_plant`, `estimation`,
`control`, `sim_plots`) come from FSA_model_dev unchanged. You write:

### 1. Bench drivers (`version_3/tools/`)

Two benches per model is the established pattern:

- **`bench_smc_full_mpc_fsa_v5.py`** — full closed-loop SMC²-MPC.
  Plant + filter + controller. Mirrors `version_2/tools/bench_smc_full_mpc_fsa.py`
  (the FSA-v2 bench, the closer template since FSA-v5 is the same family)
  and `version_2/tools/bench_smc_full_mpc_swat.py` (the SWAT bench, more
  recent patterns: `_particle_counts_for_horizon`, `--scenario` CLI flag,
  param-trace auto-plot hook, `SWAT_LAMBDA_E` style env var if relevant).
- **`bench_controller_only_fsa_v5.py`** — skips the filter, uses truth
  params + actual plant state. Mirrors `version_2/tools/bench_controller_only_swat.py`
  exactly. **This is the bench you should use for any controller
  debugging** — it's ~3–4× faster than the full closed-loop bench, per
  the methodology doc.

Things to copy from the SWAT benches (recent patterns worth keeping):

- `_pop_step_minutes_from_argv()` and `FSA_STEP_MINUTES` env var (set
  before model imports).
- `_pop_scenario_from_argv()` with explicit scenario presets (use the
  scenario set from FSA_model_dev — read its `simulation.py` for them).
- `_particle_counts_for_horizon()` returning a tuple matching the
  bench's compute envelope at multiple horizons.
- `SWAT_TAU_T_OVERRIDE_HOURS`-style env var if you want a way to make
  T grow faster for diagnostic short-horizon runs (only if FSA-v5 has a
  similar slow-timescale parameter where this would help).
- Auto-generated param-trace plot at end of run — `try: from tools.plot_param_traces import main as _plot_param_traces; _plot_param_traces(run_dir); except ...`.
- Manifest layout (`bench`, `T_total_days`, `step_minutes`, `BINS_PER_DAY`,
  `STRIDE_BINS`, `WINDOW_BINS`, `DT`, `n_strides`, `replan_K`,
  `truth_params`, `summary {mean_X_mpc, mean_X_baseline, gates, ...}`).

### 2. Model-specific bench-side adjustments

Some things are not in `models/fsa_v5/` itself but the bench needs to
know them:

- **Control variates and bounds.** The bench's plant.advance call,
  scenario presets, and counterfactual baseline all depend on what
  variates the model exposes. Read FSA-v5's `_v_schedule.py` (or
  whatever the equivalent is — check the dev-sandbox file list).
- **Stride / window / replan cadence.** SWAT used 3-h stride / 1-d
  window / 6-h replan because of its fast subsystem. FSA-v2 used 1-d
  window / 12-h stride / 24-h replan. **Decide FSA-v5's cadence from
  the LaTeX technical guide's timescale analysis** — match it to the
  fastest dynamic the controller needs to react to.
- **Identifiable subset.** Pick the params you expect the filter to
  recover at the configured horizon. Used by the id-cov gate and the
  posterior-mean construction.
- **Acceptance gates.** Set thresholds appropriate to FSA-v5's cost
  functional — likely `mean_X_geq_alpha_x_baseline`, an analog of T_floor
  if the model has a slow-growth state, id-cov, compute-budget. Don't
  copy SWAT's gates blindly; they are SWAT-specific.

### 3. Smoke test (`version_3/tests/test_fsa_v5_smoke.py`)

A quick test the bench can be loaded and stepped:

```python
"""Smoke test: model imports, plant advances, controller plans."""
import os
os.environ.setdefault('FSA_STEP_MINUTES', '15')
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import numpy as np
from version_3.models.fsa_v5._plant import StepwisePlant
from version_3.models.fsa_v5.simulation import DEFAULT_INIT, DEFAULT_PARAMS
from version_3.models.fsa_v5.control import build_control_spec

def test_plant_advance_one_stride():
    plant = StepwisePlant(state=DEFAULT_INIT.copy())
    obs = plant.advance(stride_bins=12, ...)   # bench-equivalent call
    assert obs['trajectory'].shape == (12, <n_states>)

def test_controller_one_plan():
    spec = build_control_spec(n_steps=96, dt=1.0/96, ..., params=dict(DEFAULT_PARAMS))
    # Run a tiny tempered-SMC pass on the controller; just verify it
    # produces a finite cost and a non-degenerate schedule.
```

Run with `cd version_3 && PYTHONPATH=.:.. pytest tests/ -v`.

### 4. `version_3/outputs/fsa_v5/CHANGELOG.md`

Empty audit trail to start. Each bench run gets an entry — same
convention as `version_2/outputs/swat/CHANGELOG.md` and
`version_2/outputs/fsa_high_res/RESULT.md`. Run NN, headline, plots,
gates pass/fail.

## Things you DO NOT do

- **Do not modify any file in `version_3/models/fsa_v5/`** — those came
  from FSA_model_dev/claude/dev-sandbox-v4 and are the senior decision.
- **Do not modify `smc2fc/` framework code.** If you find a real
  framework bug or genuinely-needed capability extension, escalate to
  Ajay first. The senior-files principle (CLAUDE.md) makes this
  explicit.
- **Do not modify `version_2/`.** That's separate work. FSA-v5 is a
  fresh subtree.
- **Do not delete or rewrite documentation in `claude_plans/`** — that's
  the audit trail of prior work. You can add new docs.
- **Do not commit `version_3/outputs/fsa_v5/*.png` / `*.npz` / `*.json`** —
  the project's `.gitignore` excludes those. Only the CHANGELOG.md
  should be tracked from outputs.

## Test sequence (recommended order)

1. **Smoke** — run `pytest version_3/tests/test_fsa_v5_smoke.py`. Should
   pass before any GPU work.
2. **Controller-only at small horizon** — `bench_controller_only_fsa_v5.py`
   at the smallest horizon that exercises the dynamics meaningfully. Use
   truth params (no filter). Verify the controller produces a
   non-degenerate schedule and the integrand cost is finite. ~10 min GPU.
3. **Full closed-loop at small horizon** — `bench_smc_full_mpc_fsa_v5.py`
   at the same horizon. Verify the filter runs to completion and the
   posterior-mean states are sane. ~30 min GPU.
4. **Promote to production horizon** — full closed-loop at the canonical
   horizon (whatever the LaTeX technical guide says is "production" for
   FSA-v5). Acceptance-gate readout.

If anything fails, **debug on the controller-only bench first**, not the
full closed-loop bench. The methodology doc explains why.

## Plan-archive policy

When you finalise a plan in plan mode, archive it from
`~/.claude/plans/` to `claude_plans/` in this repo with a human-readable
filename and a `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` line.
Update the archive in step with the master plan; accumulate `> Updated:`
lines. **CLAUDE.md has the full policy.**

## When you're done

1. The bench drivers in `version_3/tools/` produce sensible plots
   (T or analogous state climbs as expected, applied controls are in
   physical range, mean state vs baseline ratio passes the gate you
   set).
2. The smoke test passes.
3. The CHANGELOG.md has at least one production-horizon run logged.
4. You've written a session-summary doc in `claude_plans/`
   (similar to `SWAT_controller_production_validated_2026-05-05_0751.md`)
   covering: what was imported, what bench was written, what gates pass,
   any open issues.
5. You've **not** modified any read-only files (model files, smc2fc/,
   version_2/).

## A note on respect

The previous SWAT-session Claude (me) made several mistakes that cost
GPU time and Ajay's patience:

- Bundling unrelated framework changes into a SWAT commit without
  asking.
- Keeping compute-heavy controller knobs in production after they had
  shown no measurable benefit at the cheap test (the D2 / D5 fiasco).
- Starting from over-confident estimates for run wall-clock and
  refusing to admit when extrapolation didn't reconcile.

Don't repeat those. Hedge your wall-clock estimates. Ask before
committing files you didn't write. If a "principled" knob doesn't pay
off in the cheap test, do not let it ride into the expensive test. Read
the "Verify before you assert" section of CLAUDE.md before every
commit.

Good luck.
