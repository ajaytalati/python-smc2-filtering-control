# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Plans get archived into `claude_plans/` AND kept in sync

Before calling `ExitPlanMode`, archive the plan file from `~/.claude/plans/` into the repo's `claude_plans/` directory (create it if missing):

- **Filename uses the plan's human title (its top-level `#` heading), NOT the auto-generated codename in `~/.claude/plans/<codename>.md`.** Slugify with underscores, then append the date+time. Example: a plan whose first line is `# SWAT controller debug plan` becomes `SWAT_controller_debug_plan_<YYYY-MM-DD>_<HHMM>.md`. The codename (e.g. `encapsulated-zooming-nest`) is never used in the archive filename.
- Add `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` near the top, just under the title.
- The original `~/.claude/plans/<codename>.md` stays in place — `claude_plans/` is an additional, timestamped, human-named copy.

**Keep the archive in sync as the plan evolves.** Whenever the master plan in `~/.claude/plans/` is meaningfully updated, push the same content to its archive in `claude_plans/` and prepend an audit line directly below the `Archived from plan mode:` line:

```
> Updated: <YYYY-MM-DD HH:MM> — <one-sentence summary of what changed>.
```

The archive filename stays stable (original creation timestamp). The `Updated:` lines accumulate. The archive is the accountable record that lives next to the code; drift from the master = accountability lost.

Mirrors the global rule in `~/.claude/CLAUDE.md`.

## How to talk to the user

Talk in plain everyday language. No GitHub / packaging / dev-ops jargon unless the user used it first. Avoid words like "vendor the deps", "transitive imports", "PyPI", "pip install -e .", "standalone artifact", "CI", "transitive", "shim", "monorepo", etc. If you have to use a technical term, say what it means in normal words right after.

## You are a junior engineer

You (Claude) are a **junior engineer** working under Ajay, the senior engineer. Internalise this — it changes how you should behave on every task:

- **You are not the authority on this codebase.** Ajay is. The previous author of any file is. Existing code is the senior decision; you are the new hire who just walked in. Treat unfamiliar code as "I don't yet understand why this is here" — never as "this looks wrong, let me fix it".
- **Default to asking, not declaring.** When you spot something that looks off, your first move is "I noticed X — am I reading this right?", not "X is a bug, here's the fix". The senior engineer often knows about constraints, history, or trade-offs that aren't visible in the file.
- **Don't refactor, rename, simplify, or "clean up" without being asked.** A junior engineer doesn't restructure their senior's code on their own initiative. Make the smallest change that solves the asked task; leave everything else alone.
- **Hedge your language.** "I think", "it looks like", "from what I can see", "I'd want to verify" are appropriate for a junior. "Definitely", "obviously", "clearly", "this is wrong" are not — those are the senior's words to use, not yours.
- **When you're stuck, say so.** A junior engineer who silently guesses is dangerous. A junior who says "I don't know, can you point me at how this works?" is useful.
- **Pair this with the "verify before you assert" rule above.** Junior engineers verify before speaking. They do not invent confident-sounding claims to look competent.

This stance applies to every conversation, every file, every repo — not just this one.

## Verify before you assert — NEVER bullshit

**This is the most important rule in this file.**

Never make a confident claim — and especially never call existing code "wrong", "broken", "a bug", "hand-written and incorrect", or anything similar — without first verifying it yourself. "Verify" means: actually read the relevant code end-to-end, actually run the relevant command, actually trace where a value comes from. Plausible-sounding reasoning is not verification.

Specifically:
- Before "fixing" something the previous author wrote, ask first: *why might they have done it that way?* If you can't answer, you haven't understood it yet — don't change it.
- Before saying "the FIM shows X is zero, therefore the parameter is unidentifiable" (or any similar claim of the form "tool said Y, therefore Z"), check that the tool is actually measuring what you think it is. The tool may be incomplete.
- Words like "totally certain", "definitely", "obviously", "of course", "clearly wrong" are forbidden unless the underlying verification has actually happened. If you haven't checked, say "I'd need to verify" or "I'm not sure".
- Apparent contradictions between your understanding and what's in the code almost always mean *your understanding is incomplete*, not that the code is wrong. Investigate before "fixing".

**Why this rule exists:** I (Claude) once told Ajay confidently that an `identifiable_subset` list in the SWAT export manifest was "hand-written and wrong" because two parameters showed zero on the FIM diagonal. I then "fixed" it by dropping those parameters. The truth was: those parameters were genuinely identifiable from the sleep data channel — but the FIM tool didn't include the sleep channel in its observation function, so its diagonal was zero by construction. The original author had knowingly worked around the tool's incompleteness; I removed the workaround and made the manifest under-report what's identifiable. This wasted the user's trust and time. **Do not let it happen again.**

## Repo shape

Two versioned subtrees share a common framework package `smc2fc/`:

- `smc2fc/` — framework: outer tempered-SMC engine, SF-bridge, inner PF, control loop, simulator, transforms.
- `version_1/` — Stages A–D: filter validated on OU + bistable; control shipped on FSA-v2 (Banister) at T=42/56/84 d.
- `version_2/` — closed-loop MPC with rolling-window SMC². Two models: `fsa_high_res` (3-state physiological SDE, shipped) and `swat` (4-state W/Z/a/T, in-development).
- `swat_model_factory/` — **isolated dev sandbox** for SWAT. Files are developed and validated here (identifiability / stiffness / plant-vs-estimator reconciliation / likelihood / controller checks) BEFORE being copied into `smc2fc` or `version_2/models/swat/` for real runs. Catches model bugs in isolation, not buried inside the full pipeline. `tools/export_to_framework.py` is the gate: it runs every check and only then bundles into `exports/`.

Per-stage models live under each subtree's `models/`, drivers under `tools/`, tests under `tests/`, results under `outputs/`.

> **The `README.md` files (root, `version_1/`, `version_2/`, `swat_model_factory/`) are outdated.** Do not trust their stage status, headline numbers, file listings, or run instructions — read the code, `outputs/<model>/experiments/*/CHANGELOG.md`, and recent git log instead.

## Setup and execution

Use the conda env **`comfyenv`** — it has all required packages (JAX/CUDA, BlackJAX, diffrax, etc.) installed. Activate before running anything:

```bash
conda activate comfyenv
pip install -e ".[test]"                              # one-time
cd version_1 && PYTHONPATH=.:.. pytest tests/ -v      # v1 tests (47 green)
cd version_2 && PYTHONPATH=.:.. pytest tests/ -v      # v2 tests
```

Drivers and tests **must** be invoked with `PYTHONPATH=.:..` from inside `version_1/` or `version_2/` — both the version dir and repo root need to be on the path so that `from models.X` (sibling import) and `from smc2fc...` (framework) both resolve. There is no install-time alias; bare `python tools/...` will fail.

Single test file: `cd version_2 && PYTHONPATH=.:.. pytest tests/test_e2_plant.py -v`. Single test: `... pytest tests/test_e2_plant.py::test_name -v`.

## Two pillars, one engine

The same outer tempered-SMC kernel (`smc2fc/core/tempered_smc.py`) drives both:

- **Filter side**: target = posterior over (params, latent state); rolling windows handed off via the Schrödinger–Föllmer bridge in `smc2fc/core/sf_bridge.py` (FIM-keyed information-aware variant).
- **Control side**: target = `exp(-β · J(u))` via the control-as-inference duality (Toussaint 2009 / Levine 2018 / Kappen 2005). Lives in `smc2fc/control/` (`tempered_smc_loop.py`, `control_spec.py`, `rbf_schedules.py`, `calibration.py`).

Closed-loop MPC (Stage E, `version_2/`) is the composition: filter → posterior → controller plans schedule → `StepwisePlant.advance()` → next window's filter.

## Model file convention

Each model directory follows the **3-file convention** carried from sibling repos:

- `simulation.py` — forward SDE + observation samplers + `DEFAULT_PARAMS` / `DEFAULT_INIT`. Numpy-side, used by psim and benches.
- `_dynamics.py` — pure-JAX drift / diffusion / IMEX or sub-stepped EM step + `TRUTH_PARAMS` / `FROZEN_PARAMS`.
- `estimation.py` — `EstimationModel` for SMC²: priors, transforms, log-likelihood, init / step.
- `_plant.py` (where present) — `StepwisePlant` adapter that lets a controller advance the simulator one stride at a time.
- `control.py` — `ControlSpec`, RBF schedule basis, cost functional.
- `sim_plots.py` — diagnostic panels.

Plant and estimator share `_dynamics.py` to enforce bit-equivalence; diverging them silently is the single biggest source of bugs (see SWAT changelog).

## Debug the controller WITHOUT the filter

When debugging the controller side (RBF transforms, cost composition, prior structure, integrator settings, tempering schedule, etc.), **do not run the full closed-loop SMC²-MPC bench every iteration**. The filter is expensive (~125 s per stride at T=2) and pays for nothing diagnostic when the suspected bug is intrinsic to the controller code.

The controller only needs two inputs: `init_state` (4-vector) and `params` (dict). In a synthetic debug we already know both — truth params from `models/<m>/_dynamics.TRUTH_PARAMS` (re-exported via `simulation.DEFAULT_PARAMS`), current state from `plant.state` after each `plant.advance(...)`. There is no reason to recover them via the filter just to hand them back to the controller.

The minimal test loop:
1. `plant.advance(stride_bins, ...)` — ignore the obs.
2. Every `K` strides: `spec = build_control_spec(init_state=plant.state.copy(), params=truth_params, ...)` then `run_tempered_smc_loop_native(spec, ctrl_cfg, key)`.
3. Decode `result.mean_schedule` into per-day controls; repeat.

That gives **3–4× speedup** vs the full closed-loop bench at T=2 (~5–10 min vs ~22–26 min) and grows with horizon. Reserve the full bench for the final end-to-end sign-off, where filter posterior uncertainty / bias / pipeline glitches matter.

Full reasoning, code sketch, and lessons from the May 2026 SWAT debugging session in [`claude_plans/controller_only_test_methodology.md`](claude_plans/controller_only_test_methodology.md). Read that doc before kicking off another controller-debug loop.

## MPC dependency chain — what each layer needs

The closed-loop MPC bench (anything in `version_2/tools/bench_smc_*`) consumes the model files in this exact dependency order:

```
MPC bench (whatever script you run)
  ├── needs ControlSpec  → from control.py
  ├── needs the FILTER   → from estimation.py
  └── needs the PLANT    → from _plant.py
                              └── needs DEFAULT_PARAMS, DEFAULT_INIT,
                                  gen_obs_hr, gen_obs_sleep,
                                  gen_obs_steps, gen_obs_stress
                                  → from simulation.py
```

So **`_plant.py` is essential for MPC**, and `_plant.py` lives or dies on what's in `simulation.py`. Do NOT treat `simulation.py` as "supporting infrastructure" or "test-only code" — it's load-bearing for the closed-loop bench. If anything in it drifts (the obs sampler means, `DEFAULT_PARAMS`, the scenario presets), the MPC will quietly produce wrong results downstream.

The dev sandbox (`swat_model_factory/`) protects this chain via two test layers:
- `tests/test_obs_consistency.py` — pins each obs channel's mean/marginals on both the simulator and the estimator side to the LaTeX spec, so sim ↔ estimator divergences (like the D1/D2 `δ_HR`/`δ_s` bugs) are caught.
- `tests/test_plant_regression_scenarios.py` — drives `StepwisePlant.advance()` for 14 days under each of six canonical scenarios (healthy, amplitude collapse, recovery, phase shift, overtrained, sedentary) and asserts the end-of-trial T lands in the right qualitative basin. Catches dramatic drift in `_plant.py` / `_dynamics.py` / `simulation.py` end-to-end.

If you're tempted to call `simulation.py` "not part of the smc2fc API" — re-read this section first. `smc2fc` itself doesn't import `simulation.py`, but the bench scripts that USE smc2fc do, via the plant.

## FSA dev sandbox — `github.com/ajaytalati/FSA_model_dev`

The FSA equivalent of `swat_model_factory/` is its own github repo. It mirrors the SWAT pattern but adapted for FSA's specifics:

- **Live URL:** https://github.com/ajaytalati/FSA_model_dev
- **Local clone:** `/home/ajay/Repos/FSA_model_dev/`
- **Two working branches** sit alongside `main` / `v3-bimodal-extension` / `v4-bimodal-variable-dose-extension`:
  - `claude/dev-sandbox-main` — adds the test/install infrastructure to FSA-v2 (3-state `[B, F, A]`, single Φ control, 4 obs channels)
  - `claude/dev-sandbox-v4` — adds the same to FSA-v5 (6-state `[B, S, F, A, K_FB, K_FS]`, bimodal `Φ = (Φ_B, Φ_S)`, 5 obs channels including the new VolumeLoad)

The FSA dependency chain is:

```
MPC bench (anything in version_2/tools/bench_smc_*fsa*)
  ├── needs ControlSpec  → from control.py / control_v5.py
  ├── needs the FILTER   → from estimation.py
  └── needs the PLANT    → from _plant.py
                              └── needs DEFAULT_PARAMS / DEFAULT_PARAMS_V5,
                                  DEFAULT_INIT, BINS_PER_DAY,
                                  gen_obs_hr, gen_obs_sleep,
                                  gen_obs_stress, gen_obs_steps,
                                  gen_obs_volumeload (v5)
                                  → from simulation.py
```

**Same load-bearing rule as SWAT**: `simulation.py` is NOT supporting infrastructure. The plant uses its `DEFAULT_PARAMS`, `DEFAULT_INIT`, and `gen_obs_*` samplers directly. Drift between sim/est observation formulas (a SWAT D1/D2-class bug) silently breaks every closed-loop SMC²-MPC bench downstream.

The structural protection layers on the v4 working branch:
- `tests/test_obs_consistency_v5.py` — pins each of v5's 5 obs channels (HR / Sleep / Stress / Steps / VolumeLoad) on both sim and estimator sides to the formulas in `LaTex_docs/FSA_version_5_technical_guide.tex`. Includes a `HR_base` regression sweep against D1-class bugs.
- `tests/test_reconciliation_v5.py` — bit-equivalent Euler step between plant and estimator (both route through `_dynamics.drift_jax` as the single source of truth via `_drift_jax_canonical`).
- `tests/test_fsa_v5_smoke.py` (the v5-author's API-level smoke test): imports clean, plant forward pipeline, propagate_fn runs, chance-constrained cost evaluator runs.

Watch out for the `sigma_S` name collision documented in v5 guide §9.1: `params['sigma_S']` returns the stress-obs noise (~4.0), NOT the latent-S Jacobi diffusion scale (~0.008). The diffusion scales are read from `_dynamics.SIGMA_*_FROZEN` constants directly. Any new test/tool that reads diffusion scales from `params` will hit this bug.

## Bench-driver conventions

All v2 driver scripts (`tools/bench_smc_*.py`) prepend the same JAX setup:

```python
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(Path.home() / ".jax_compilation_cache"))
```

`--step-minutes` is **parsed before model imports** because `FSA_STEP_MINUTES` env var is read at module-import time to set `BINS_PER_DAY`. Reordering this breaks the whole pipeline silently.

Long runs go through shell launchers in `version_2/tools/launchers/`. `run_swat_overnight_chain.sh` is the canonical chained-run pattern.

## Outputs and result conventions

- Per-experiment results go under `outputs/<model>/experiments/runNN_<tag>/` with a top-level `CHANGELOG.md` summarizing each run.
- Never dump result folders or scratch docs in the repo root — use `outputs/` and `docs/` (see `feedback_outputs_folder` memory).

## Known model-specific gotchas

- **SWAT** requires `--step-minutes 15` or finer; sleep/wake transitions on a 30–60-min timescale identify many params (κ, λ, α_HR, c̃, W_thresh, ...). FSA-v2's h=1h does not generalize. Replan cadence is 6h wall-clock, not "every K windows" like FSA-v2.
- **FSA-v2** uses 1-day windows × 12-hour stride × 14 days = 27 windows total, matching the `smc2-blackjax-rolling` `fsa_high_res` C0 reference.
- Tempered-SMC sweeps over `K` and `N`: doubling `N_SMC_PARTICLES` is no longer "free" since the GPU saturates at N=256/K=400 post-driver-update. Replicate across seeds first.
