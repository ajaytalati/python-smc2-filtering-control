# New `LaTex_docs/` for the `smc2fc` Repository — Plan

> Archived from plan mode: 2026-05-06 08:00.

## Context

The framework documentation in `python-smc2-filtering-control/LaTex_docs_outdated/`
is FSA-v2-specific in many concrete places, treats the Schrödinger-Föllmer
bridge as a black box, doesn't cover the OT rescue or the JAX-native
backend at all, and is scattered in scope (mixes framework + FSA).

The user wants a NEW `LaTex_docs/` folder in the same repository,
containing **pure framework** documentation: filtering, SMC², control,
plant, MPC. Model-independent (no FSA content). Both pedagogic and
technical, but **down-to-earth and need-to-know-basis** — not a
graduate textbook. Document what the codebase actually does today;
document one concrete future-work direction (controller speed/quality);
do not speculate about alternatives the codebase doesn't support.

This plan has been refined through two rounds of user feedback:

* **Round 1:** add a dedicated section on the JAX-native compile-once
  architecture (the framework's key practical contribution); document
  the failed NUTS experiment with empirical numbers; document the
  empirical finding that the soft-HMC chance-constrained variant works
  and the hard-IS variant doesn't; **delete §10 (upgrade paths) and all
  theoretical appendices except Appendix D (code-API reference)**.
* **Round 2:** shorten §7 (the Gaussian-bridge warm-start) substantially
  — the SF/BW geometry adds little practical value beyond the Gaussian
  bridge itself, so keep it brief. Add a NEW §11 (Future Work) covering
  the controller speed-vs-quality trade-off (HMC step count, leapfrog
  length tuning) — this is the only concrete bottleneck the user has
  hit and is worth documenting explicitly.

## Output location and structure

A new folder at
`/home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/`:

```
LaTex_docs/
├── main.tex
├── preamble.tex
├── README.md
├── sections/
│   ├── 01_overview.tex
│   ├── 02_setting.tex
│   ├── 03_particle_filter.tex
│   ├── 04_ot_rescue.tex
│   ├── 05_smc2.tex
│   ├── 06_jax_native.tex                  ← NEW dedicated section
│   ├── 07_gaussian_bridge.tex             ← SHORT
│   ├── 08_control_inference.tex
│   ├── 09_mpc.tex
│   ├── 10_config.tex
│   ├── 11_future_work.tex                 ← NEW
│   └── appendix_D_code_api.tex            ← only surviving appendix
└── figures/
```

The existing `LaTex_docs_outdated/` folder is left untouched as
historical reference.

## Section-by-section content (final, post-revision)

### §1 Overview + reader's guide + architecture diagram
*New.* One-page introduction. Scope statement (framework only, no model
content). Block diagram of the four layers: inner PF, outer SMC²,
controller, MPC driver. Reading-order suggestions for mathematician vs
engineer audience.

### §2 Generic SDE-with-observations setting
*New.* The HMM setup (X_n hidden, Y_n observed, U_n exogenous control).
The two abstractions the framework requires from the user's model:
`SDEModel` (forward simulation) and `EstimationModel` (inference).
Code-interface specs only — no FSA content.

### §3 Bootstrap particle filter
*Lifted from old §6 with notation generalised.*
- §3.1 Bayesian filtering recursion (predict / update).
- §3.2 Sequential importance resampling, systematic resampling.
- §3.3 Liu–Vest shrinkage (currently mentioned-in-passing in old doc;
  promote to a proper subsection — production-critical).
- §3.4 Unbiased marginal likelihood (Del Moral 2004) — the theoretical
  hook for SMC².
- Implementation pointer: `filtering/gk_dpf_v3_lite.py`,
  `filtering/_gk_kernel.py`.

### §4 OT rescue inside the filter
*Net new.* Three subsections, ~3 pages.
- §4.1 When SIR + Liu–West isn't enough: degeneracy detection via the
  ESS sigmoid blend (`gk_dpf_v3_lite.py:25–27`).
- §4.2 Low-rank Nyström kernel approximation
  (`filtering/transport_kernel.py`).
- §4.3 Sinkhorn iterations + barycentric projection
  (`filtering/sinkhorn.py`, `filtering/project.py`,
  `filtering/resample.py:ot_resample_lr`).
- §4.4 The blended output (sigmoid interpolation between systematic+LW
  and OT).
- Implementation pointer.

### §5 SMC² over parameters: tempered SMC + HMC
*Lifted from old §7, with HMC-mass-matrix subsection promoted.*
- §5.1 The tempered posterior (lifted, generalised).
- §5.2 Adaptive ESS-based tempering schedule (lifted).
- §5.3 The HMC move step.
- §5.4 Adaptive diagonal mass matrix (`core/mass_matrix.py`) —
  re-estimated per tempering level. Justify: full mass matrix collapses
  HMC acceptance to zero by λ ≈ 0.3 on this kind of problem.
- Implementation pointer.

### §6 JAX-native SMC: once-per-run compilation *(NEW DEDICATED SECTION)*
*Net new — flagged by user as the framework's key practical contribution.*
- §6.1 The compile-cost problem with the BlackJAX-wrapped backend
  — per-window JIT recompilation dominates total cost.
- §6.2 The fix: module-level HMC kernel (`jax_native_smc.py:53`) +
  on-device tempering loop via `lax.while_loop` (`jax_native_smc.py:308`).
  Adaptive δλ replaced by on-device bisection
  (`_solve_delta_for_ess`, `jax_native_smc.py:206`).
  One trace, one compile, reused across all subsequent windows.
- §6.3 Empirical speedup (HMC JIT + first run = 1.6s; cached run = 1.1s
  per window).
- §6.4 **Failed experiment: NUTS as a drop-in replacement.**
  File: `core/jax_native_smc_nuts.py`. Empirical numbers (preserved
  from user's benchmark):
    - HMC JIT + first run: 1.63s vs NUTS: 102.03s (~60× worse)
    - HMC cached run:      1.09s vs NUTS:  94.23s (~85× worse)
  Why NUTS fails inside this architecture: dynamic tree-building
  (U-turn check) requires an internal `while_loop` that JAX cannot
  efficiently compile when nested inside `vmap` over 256 particles +
  outer `lax.while_loop` for tempering. "5 NUTS steps" can mean
  hundreds–thousands of gradient evaluations per particle per
  tempering level vs HMC's predictable 5 × 8 = 40.
  *Conclusion: do not replace HMC with NUTS.* The file is preserved
  as a documented dead-end; future agents should consult §6.4 before
  re-attempting.
- §6.5 Implementation pointers (file map, how to switch backends in
  user code).

### §7 Gaussian bridge for warm-starts *(SHORT — revised per round 2)*
*Brief — was going to be the full SF/BW geometry treatment; now ~1 page total.*
- §7.1 Why warm-start? Cold-start tempering cost dominates if every
  window restarts from the prior.
- §7.2 The Gaussian bridge in practice: importance-weighted moment
  match of the previous-window posterior to a Gaussian, take a blended
  initial condition (default ρ=0.7 toward new evidence). Implementation
  in `core/sf_bridge.py`.
- §7.3 The Schrödinger–Föllmer/Bures–Wasserstein machinery available
  in `sf_bridge.py` is documented as an option, but **empirically does
  not add measurable value beyond the simple Gaussian bridge** — the
  closed-form BW geodesic and the simple linear interpolation between
  Gaussians give comparable initialisation quality on tested problems.
  Mention info-aware blending as a knob for users who want it; do not
  develop the geometry.
- Implementation pointer.

### §8 Control as inference (with empirical hard-vs-soft finding)
*Lifted from old §8, generalised, plus new empirical content.*
- §8.1 The cost functional, abstract setting.
- §8.2 Schedule decoder (RBF as one example; framework-agnostic).
- §8.3 Tempered control posterior (control-as-inference, posterior
  mean as L² minimiser).
- §8.4 **Hard-indicator vs soft-sigmoid chance-constrained variants.**
  Both forms targeting Pr[A_t < A_sep(Φ_t)] ≤ α.
  - Hard: indicator 1[A_t < A_sep]. Designed for pure-SMC²
    importance-weighting controllers.
  - Soft: sigmoid(β · (A_sep - A_t) / scale). Differentiable,
    designed for HMC.
  - **Empirical finding (preserved from user's run):** the soft
    variant works correctly with HMC; the hard variant does not
    converge to the correct chance-constrained optimum in practice.
  - Recommendation: use the soft variant only. The hard variant is
    preserved in `control_v5.py` for documentation / regression
    purposes.
  - One sentence on the failure mode of the hard variant — to be
    filled in at draft time by asking the user for the specific
    symptom they observed (particle collapse? wrong mode? slow
    convergence?). I'll prompt for this then.

### §9 Receding-horizon MPC
*Lifted from old §9 with FSA-specific content stripped.*
- §9.1 Finite-horizon stochastic optimal control (lifted).
- §9.2 Receding-horizon principle (lifted).
- §9.3 Stochastic MPC variants: certainty-equivalent / expected-cost /
  output-feedback (lifted).
- §9.4 The smc2fc closed-loop pipeline — generic algorithm, generic
  plant interface (`StepwisePlant.advance(stride_bins, U_schedule)`),
  generic bench-driver pattern. Replace all FSA-v2 specifics in old
  §9.4–9.5 with framework-level descriptions.
- §9.5 Recursive feasibility and soft-barrier discussion (lifted from
  old §9.6, generalised — F-barrier example replaced with a generic
  state-bound penalty).
- §9.6 Flat-cost-surface robustness argument (lifted from old §9.7).

### §10 Configuration and knobs
*New.* Walk through `SMCConfig` (`core/config.py`) field-by-field.
Four hyperparameter groups:
- Outer SMC: n_smc_particles, num_mcmc_steps, hmc_step_size,
  hmc_num_leapfrog, target_ess_frac, max_lambda_inc.
- Inner PF: n_pf_particles, bandwidth_scale.
- OT rescue: ot_ess_frac, ot_temperature, ot_max_weight, ot_rank,
  ot_n_iter, ot_epsilon.
- Bridge: bridge_type, sf_blend, sf_entropy_reg, sf_q1_mode,
  sf_info_aware, sf_info_lambda_thresh_quantile,
  sf_info_blend_temperature.
For each: what it does mathematically, sensible default ranges,
diagnostic signals that say "increase / decrease this".

### §11 Future work *(NEW)*
*Net new — added per round 2 user feedback.*
**Scope:** narrowly about **optimising the soft-HMC controller**. The
hard-IS variant is settled in §8.4 (it doesn't work; not used) and is
not part of this section. Everything below is about making the
soft-HMC controller faster without losing control quality.

- §11.1 The trade-off as observed today. Running the soft-HMC
  controller's tempered SMC² with high-quality settings (more leapfrog
  steps, more MCMC steps per tempering level, finer tempering schedule)
  produces better-quality control plans but costs more wall-clock time.
  This is the framework's main practical bottleneck: planning latency
  scales linearly with HMC work per tempering level.
- §11.2 The tunable knobs and what each one costs.
    - `hmc_num_leapfrog` — leapfrog steps per HMC trajectory. Higher =
      better mixing per move, more compute per move.
    - `num_mcmc_steps` — number of HMC moves per tempering level.
      Higher = more rejuvenation, more compute per level.
    - `target_ess_frac` — ESS threshold for tempering bisection.
      Lower threshold = larger λ jumps = fewer tempering steps but
      more aggressive reweighting.
    - `max_lambda_inc` — cap on λ jump per tempering step.
    For each knob: which direction improves quality, which improves
    speed, and where the diminishing-return knee tends to be.
- §11.3 Open directions worth investigating to improve the soft-HMC
  controller (each one a short paragraph; presented as candidates the
  user/team could explore, not recommendations from me):
    - Adaptive HMC step-size schedule during tempering (e.g.\ smaller
      step at low λ where the target is wide, larger at high λ where
      it's tight).
    - Per-tempering-level adaptive leapfrog count rather than the
      current fixed `hmc_num_leapfrog`.
    - Reusing the inference posterior as the controller's prior
      (currently a fresh prior is used; reuse may shorten the
      tempering schedule substantially).
    - Per-bridge HMC step-count tuning — currently 3 for bridge,
      5 for cold-start; bridge count may be reducible if §7's Gaussian
      bridge already does enough work.
    - Profiling / instrumentation: where does the wall-clock actually
      go? `jax.profiler` snippet to find the hot loop before changing
      anything else.
    - Mixed precision (float32 inside HMC inner loop, float64 outside)
      — potential 2× speedup on GPU at modest accuracy cost.

### Appendix D — Code-API quick reference
*Net new (only surviving appendix).* One-line description of every
public function in `core/` and `filtering/`, alphabetised. The "if you
can't remember which file it's in" lookup. Generated by walking the
namespace at draft time.

### What's deleted compared to my earlier plan

- Appendix A (Bures–Wasserstein primer) — gone (§7 no longer develops
  the geometry; primer not needed).
- Appendix B (HMC + adaptive mass matrix recap) — gone (covered briefly
  in §5.4; standalone primer not needed).
- Appendix C (Sinkhorn + low-rank kernels) — gone (covered in §4;
  primer not needed).
- Appendix E (LQG baseline) — gone (FSA-specific anyway, was a stretch
  for a framework doc).

## Length target

Roughly 25 pages (down from previous 30-page estimate after slimming
§7 and adding §11). 11 main sections + 1 appendix.

## Critical files

- New: every file under `LaTex_docs/`.
- Read-only references during drafting:
  - `LaTex_docs_outdated/sections/06_particle_filter.tex` — lift §3 from.
  - `LaTex_docs_outdated/sections/07_smc2.tex` — lift §5 from.
  - `LaTex_docs_outdated/sections/08_control_inference.tex` — lift §8.1–8.3 from.
  - `LaTex_docs_outdated/sections/09_mpc.tex` — lift §9 from.
  - `core/jax_native_smc.py`, `core/jax_native_smc_nuts.py` — for §6.
  - `core/sf_bridge.py` — for §7 (briefly).
  - `core/config.py` — for §10.
  - `filtering/gk_dpf_v3_lite.py`, `_gk_kernel.py`,
    `transport_kernel.py`, `sinkhorn.py`, `project.py`, `resample.py` — for §3 and §4.

## Verification

1. `cd LaTex_docs && latexmk -pdf main.tex` — clean compile, ~25 pages.
2. `pdftotext main.pdf - | grep -iE 'fsa|banister|autonomic|\(B,\s*F,\s*A\)'` — should return zero matches in body text (model-independent check; "appendix D" reference list may have file paths that mention FSA-related model files, that's OK).
3. Every cited file:line reference checked against current line numbers
   via Bash before commit.
4. Both empirical findings (NUTS failure §6.4, hard-vs-soft §8.4) quote
   the user's actual numbers verbatim.

## Open questions for the user (to ask before drafting)

1. **One sentence on the hard-IS failure mode** for §8.4 — what specific
   symptom did the user observe when running the hard variant? Without
   this, §8.4 will say something neutral like "fails to converge to the
   constrained optimum in practice"; with it, the documentation is
   empirically grounded.
2. *(Notation, low priority)* Keep `θ_dyn` / `θ_ctrl` distinction from
   the old doc, or generalise to plain `θ` with context? My instinct is
   to keep it; happy to be redirected.
