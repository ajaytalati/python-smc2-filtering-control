# Plan: LEAN4-first development for FSA models (per charter)

> Archived from plan mode: 2026-05-06 (re-entry — replaces the prior FSA-v5
> debug plan from earlier this session).
> Updated: 2026-05-06 ~15:00 — scaffolding green. Drift + Cost (muBar, findASep, aSepGrid) transcribed; mathlib v4.30.0-rc2 installed via cache-get; Python bridge subprocess-based; **10/10 differential tests pass** (3 sanity + 200 hypothesis-random for drift; 2 LaTeX §10.4 anchor + 200 random for muBar; 2 closed-island gates + 100 random for findASep). Bugs structurally prevented: 1 (sigma_S, by named fields), 2 (particle-0 template, by aSepGrid signature). Remaining per charter table: Plant (~1h), Schedule (~30m), Obs (~1h).
> Updated: 2026-05-06 ~15:30 — **full charter table done.** Plant.lean, Schedule.lean, Obs.lean transcribed and wired into the CLI + Python bridge. **16/16 differential tests pass** end-to-end at `1e-6` tolerance: drift (3 anchor + 200 random), muBar (2 + 200), findASep (2 + 100), schedule (1 + 50), emStep (1 + 100), all 5 obs channels (1 HR anchor + 50 random over all channels). ObsParams structure (separate from Params) makes Bug 1's sigma_S collision class structurally impossible on the obs side too. Charter §5 step-1-through-7 fully demonstrated for FSA-v5 drift+cost+plant+schedule+obs.
>
> Aligned to: `/home/ajay/Repos/FSA_model_dev/LaTex_docs/lean4_first_charter.pdf`
> (May 6 2026, Ajay Talati + Claude). The charter is the authoritative
> standing rule; this plan is the implementation of that charter for FSA-v5.

## Context

In this session, three semantic/type-confusion bugs were found in
`version_3/models/fsa_v5/` + the upstream `FSA_model_dev` repo: a
`sigma_S` dict-key collision (named-field collision silently kept
the wrong value), a particle-0 separator template (the legacy
implementation collapsed an SMC² ensemble to a single template
before computing `A_sep`), and a post-hoc evaluator wiring mismatch
(smooth-schedule cost during HMC vs burst-expanded post-hoc) that
produced a vacuous 96% violation rate. A fourth claim — about a
"theta prior wider than the healthy island" — was incoherent and
retracted, because the agent treated a 16-dim vector as a scalar.

The retracted claim is the symptom of a deeper failure mode: an LLM
agent reads code by surface pattern (variable names, syntactic shape)
rather than by *type* and *semantic role*. Pattern-matching can't be
fully eliminated by adding rules to a CLAUDE.md (we tried), because
the failure happens silently inside the agent's reasoning.

Per the charter: invert the trust direction. **Lean4 is the formal
source of truth AND the executable reference implementation.** Python
is a secondary, GPU-targeted port that is differentially tested against
the Lean4 native binary on every PR. Disagreement beyond a tolerance
threshold is *by construction* a Python bug.

## The architecture: 3 layers, with the Lean4 binary as oracle

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — LaTeX equations (DESIGN, as today)                   │
│  • Drift, diffusion, cost, plant, schedule, observation         │
│  • Lives in LaTex_docs/sections/                                │
│  • Authored by humans                                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ line-by-line transcription
                           │ (human-reviewed; this is the one place
                           │  reviewer attention has to land)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2 — LEAN4 spec + executable reference (CANONICAL)        │
│  • lean/Fsa/V5/*.lean — typed defs + runnable impls             │
│  • Dimensions encoded in the type (e.g. RBFCoeffs 16,           │
│      ParameterVector 37, ControlOutput) — collisions become     │
│      compile errors                                             │
│  • lake build → native binary (compiled via C)                  │
│  • lean/python_bridge/ — thin ctypes wrappers expose each       │
│      verified function as a Python callable                     │
│  • Same file is both the spec and the executable reference      │
│      — no spec/code drift possible                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ differential testing
                           │ (pytest + hypothesis; LEAN binary on CPU
                           │  vs Python on JAX/GPU, random inputs in
                           │  the physiological-bound box)
                           │ tolerance: 1e-6 single-step drift,
                           │            1e-4 integrated trajectory
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Python/JAX implementation (agent-written, fast path) │
│  • models/fsa_high_res/*.py — JAX, JIT, GPU                     │
│  • The agent IS permitted to pattern-match here                 │
│  • Differential test failure = Python bug (always)              │
│  • smc2fc port is a thin shim from the verified LEAN4 reference │
│    into EstimationModel/SDEModel; also differentially tested    │
└─────────────────────────────────────────────────────────────────┘
```

The trust contract: **the LEAN4 binary is right about types and
about the math; Python is right when it agrees with LEAN4 within
tolerance.** No "spec drifted from code" failure mode is possible
because the LEAN4 file IS the executable reference. Property tests
encoding theorems are NOT the bridge — the bridge is direct
input/output comparison of two implementations of the same function.

## How each found bug would have been caught (or prevented)

This is the load-bearing argument — does the architecture actually
help, or is it just ceremony?

### Bug 1 — `sigma_S` dict-literal collision

**In Python**: `DEFAULT_PARAMS = { …, 'sigma_S': 0.008, …, 'sigma_S': 4.0 }`
silently kept the second value (4.0). The latent-S state-noise was
silently overwritten by the stress-channel obs noise. Caught only by
manual REPL probe.

**In Lean4**: `Params` is a structure with named fields. Two fields
with the same name is a compile error. The bug is *literally
impossible to write*.

```lean
structure Params where
  -- state-noise scales (frozen, never estimated)
  sigma_B   : Real
  sigma_S   : Real        -- one field
  sigma_F   : Real
  sigma_A   : Real
  sigma_K   : Real
  -- stress-channel obs noise (estimated)
  sigma_S_obs : Real      -- different field, different name
  …
```

**Prevention strength: total.** The duplicate-field error fires at
`lake build` time, before any Python runs. The Python port either uses
`sigma_S_obs` as a distinct key (correct), or its values diverge from
the LEAN4 reference under the differential test.

### Bug 2 — Particle-0 separator template

**In Python**: `_compute_cost_internals` did
`template = jax.tree_util.tree_map(lambda x: x[0], theta_stacked)`,
collapsing the SMC² ensemble to particle 0 before computing `A_sep`. The
broadcast `A_sep[None, :]` then applied that single template's
separator to all particles' trajectories. Mathematically wrong for
multi-particle runs.

**In Lean4**: dependent types make the shape part of the function
signature. Two distinct candidate signatures:

```lean
-- The buggy version: collapses to a single template
def compute_internals_buggy
  (particles : Vec n Params) (sched : Vec m Phi)
  : Vec n (Vec m Real) × Vec m Real := …
  --                                    ^^^^^^^^^^^^^
  --                                    A_sep is (Vec m), one per bin

-- The correct version: separator is per particle, per bin
def compute_internals
  (particles : Vec n Params) (sched : Vec m Phi)
  : Vec n (Vec m Real) × Vec n (Vec m Real) := …
  --                       ^^^^^^^^^^^^^^^^^
  --                       A_sep is (Vec n × Vec m), per particle
```

The cost-function spec would have a theorem:

```lean
theorem A_sep_depends_on_bifurcation_params
  (p₁ p₂ : Params) (Φ : Phi) :
  p₁.bifurcation ≠ p₂.bifurcation →
  A_sep Φ p₁ ≠ A_sep Φ p₂
```

That theorem is unprovable about `compute_internals_buggy` because the
buggy version collapses two distinct inputs to one. The proof
*obligation itself* tells you the buggy version is wrong.

**Prevention strength: high.** The LEAN4 reference's
`compute_internals` returns the per-particle-per-bin shape. The
differential test calls both implementations on a 2-particle ensemble
where the bifurcation parameters diverge across particles; the LEAN4
binary returns distinct rows, the buggy Python returns identical rows,
the diff fires.

### Bug 4 — Post-hoc evaluator wiring (smooth vs burst-expanded Phi)

**In Python**: The bench's post-hoc evaluator received
`plant.history['Phi_value']` (burst-expanded, mostly zeros) while the
controller's HMC cost saw the smooth RBF schedule. The chance-constraint
indicator vacuously fired during rest bins (Phi=0 → A_sep=+inf →
indicator=1). The agent (me) called this "a bug" but the more honest
framing was "an HMC-vs-spec tension with three legitimate resolutions
(A: post-hoc on smooth; B: HMC on burst-expanded; C: reformulate the
spec to per-window)".

**In Lean4**: the chance constraint is written *once*, formally, with
its argument explicitly typed:

```lean
-- Two distinct types so they can't be confused
def SmoothSchedule  := Time → Phi
def BurstSchedule   := Time → Phi   -- post-burst-expansion

def chance_constraint
  (Φ : SmoothSchedule)            -- ← spec choice made HERE
  (A : Trajectory)
  (α : ℝ) : Prop :=
  ∀ t ∈ [0, T], P (A t < A_sep (Φ t)) ≤ α
```

The author of the spec is *forced* to commit to one of (smooth /
burst) at the type level. The controller's HMC and the bench's
post-hoc both consume the same type → the wiring mismatch is a *type
error*, not a runtime mystery.

**Prevention strength: high.** The spec choice still has to be made
(that's a research question, not something Lean can decide), but it's
made *visibly*, *once*, *explicitly* — not silently across two
disconnected files.

### Retracted "Bug 5" — theta scalar/vector confusion

**In Python**: I conflated SMC² parameter posterior θ (37-dim) with
controller RBF coefficients θ (16-dim) and treated the latter as if it
were a scalar comparable to a region in 2D Phi-space.

**In Lean4**: distinct types prevent the conflation outright.

```lean
def ParamPosterior  := Vec 37 Real     -- SMC² inferred parameters
def CtrlCoefficients := Vec (2*nAnchors) Real  -- RBF anchor weights
def HealthyIsland    := Set Phi         -- 2D region in Phi-space
```

These three types are not unifiable. Any "claim" that compares
`CtrlCoefficients` to `HealthyIsland` would not type-check. The bug
literally cannot be expressed.

**Prevention strength: total.** This is the cleanest case for the
type-driven approach.

### Summary table

| Bug | Python-only catch difficulty | Lean4 prevention strength |
|---|---|---|
| 1 — sigma_S collision | hard (silent dict semantics) | total (named fields) |
| 2 — particle-0 separator | medium (need multi-particle test) | high (dependent shapes + theorem) |
| 4 — post-hoc wiring | high (research-level disambiguation) | high (typed schedule kind) |
| 5 — theta confusion (retracted) | high (semantic, not syntactic) | total (distinct types) |

**All four semantic/type-confusion cases are dramatically easier
under a Lean4-first discipline.** The one historical bug not listed
here (a generic concurrency race in the run-dir allocator) is a
different bug class entirely — Lean4 doesn't help much, and the
existing Python tooling catches it well.

## The 7-step per-function workflow (from the charter)

For every function in scope, the work runs in this order, no skipping:

1. **LaTeX equations** (already exist in `LaTex_docs/sections/`).
2. **LEAN4 transcription** in `lean/Fsa/V5/*.lean`. Typed defs +
   runnable impls. Line-by-line, side-by-side with the LaTeX.
   Translation, not design.
3. **`lake build`** → native binary; `lean/python_bridge/` ctypes
   wrappers expose each function as a Python callable.
4. **Python implementation** (agent-written, JAX/GPU, fast path).
   The agent is permitted to pattern-match here; step 6 catches the
   consequences.
5. **Optional property proofs** — Lipschitz / basin-of-attraction /
   monotonicity. Defer unless a specific incident motivates one. The
   charter does not gate on these.
6. **Differential testing** — `pytest tests/test_lean4_diff.py` with
   `hypothesis` random-input generation in the physiological-bound
   box. Both LEAN4 binary AND Python implementation are evaluated;
   any disagreement beyond tolerance (`1e-6` single-step drift,
   `1e-4` integrated trajectory) is by construction a Python bug.
7. **smc2fc port** — thin shim from the verified LEAN4 reference into
   `EstimationModel` / `SDEModel`; also differentially tested before
   merge.

## Per-function effort estimate (from the charter)

This is **translation work, not research**. The LaTeX spec already
exists; LEAN4 has the linear-algebra and ODE infrastructure;
`mathlib` supplies the lemmas. The full FSA-v5 model is **~one
day's work for a competent human developer**:

| Function                  | File                                          | LEAN4 transcription |
|---------------------------|-----------------------------------------------|---------------------|
| FSA-v5 drift              | `models/fsa_high_res/_dynamics.py`            | ~2 hours            |
| Cost (chance-constrained) | `models/fsa_high_res/control_v5.py`           | ~1.5 hours          |
| StepwisePlant             | `models/fsa_high_res/_plant.py`               | ~1 hour             |
| Schedule decoder (RBF)    | `models/fsa_high_res/control.py`              | ~30 min             |
| Observation model         | (in `estimation.py`)                          | ~1 hour             |
| FFI wrapper + diff tests  | `lean/python_bridge/`                         | ~2 hours            |
| **Total**                 |                                               | **~8 hours**        |

The LEAN4 source is roughly the same length as the LaTeX it
transcribes. Most of the time goes into wiring the FFI bridge and
writing the differential-test harness; the math itself is direct.

The cost/value frame: a single 8-hour LEAN4 transcription pass
amortises across every future FSA version, every future model
extension, and every future agent attempting a port to `smc2fc`.

## Standing rule (the charter, verbatim)

> For every new mathematical model, every model extension, and every
> import into `smc2fc`, the LEAN4 specification is written **before
> any Python implementation**. The LEAN4 native binary is the
> authoritative reference. Python is differentially tested against
> it. Disagreement beyond a tolerance threshold is by construction
> a Python bug.

This rule is permanent. It applies to FSA-v5, FSA-v6, FSA-v7, every
future model extension, and every agent who touches an FSA model in
any session. The charter PDF
(`LaTex_docs/lean4_first_charter.pdf`) is the authoritative document.

## Audit checklist for future agents

Every agent runs this checklist at session start, before touching any
FSA-related Python file. If any box is unchecked, the agent does not
proceed — the session is paused, the user is notified, the missing
artefact is produced before any code is changed.

- [ ] Is there a LEAN4 reference for the function I am about to
      touch? If no, the charter says I am **not allowed** to write
      the Python until the LEAN4 reference exists. Stop and flag.
- [ ] Does `lake build` succeed on the current commit (the LEAN4
      reference type-checks)?
- [ ] Does the Python differential test pass against the LEAN4
      binary on the current commit (`pytest tests/test_lean4_diff.py`
      is green)?
- [ ] Have I written down — in plain English — the type, role, and
      pipeline position of every variable I am about to reason about?
      (Per the PRIMARY RULE in
      `~/.claude/projects/.../memory/feedback_no_fabricated_bugs.md`.)
- [ ] If I am about to claim a bug: do I have a reproducer (input →
      both implementations → disagreement beyond the tolerance
      threshold)? If no: it is not a bug, it is an open question;
      record it as such and move on.

## What LEAN4 will NOT fix (charter §4.4 verbatim)

The charter explicitly bounds the scope. Three things are out:

1. **Floating-point semantics.** The gap between LEAN4's notion of ℝ
   and IEEE-754 double-precision arithmetic is real and not addressed
   by this charter; it is bounded by the differential-test tolerance
   threshold (`1e-6` single-step, `1e-4` integrated).

2. **GPU correctness.** The differential test runs Python on whatever
   backend it normally uses (JAX/XLA, GPU); the LEAN4 binary is
   CPU-only. Disagreements driven by GPU-specific behaviour are out
   of scope and trusted as before.

3. **The transcription itself.** If the LaTeX → LEAN4 transcription
   is wrong, both the LEAN4 reference and (likely) the agent-written
   Python will be wrong in the same way. This is why the
   LaTeX→LEAN4 step is **human-reviewed line-by-line** — the one
   place reviewer attention has to land.

Other items remain out of scope but are trusted upstream as today:
SMC²/PF framework code in `smc2fc/core/` and `smc2fc/filtering/`,
HMC/NUTS internals, the JAX-native compile-once architecture,
biological calibration of truth params.

## Risks

- **LLM writing LEAN4.** LLMs writing LEAN4 is an active research
  area; an agent transcribing LaTeX → LEAN4 may itself introduce
  errors. Mitigation: the human reviews the LEAN4 transcription
  line-by-line. That's the one place reviewer attention has to land
  (charter §4.4 and §7).
- **Spec choice is still hard.** LEAN4 forces a spec choice to be
  visible, but doesn't tell you which choice is right. The
  smooth-vs-burst chance-constraint question is still a research
  question; LEAN4 just makes it explicit at the type level instead
  of hidden in Python wiring.
- **`mathlib` gaps for SDE/SMC² theory.** We avoid these by
  formalising signatures + executable definitions only — not
  convergence theorems.

## Decision: adopt the charter; execute the per-function table

Per the charter and Ajay's direction, this is the standing development
process for FSA models from this point forward. No tactical
fallbacks. The single ~8-hour transcription pass produces the LEAN4
reference, the FFI bridge, and the differential-test harness; from
there the existing Python is differentially tested against it, and any
divergence is a Python bug.

The discipline pay-off compounds: every future model variant (v6,
v7, …) gets the same type-checked LEAN4 contract for free, instead
of each variant re-inheriting the dict-collision / signature-shape /
schedule-type-confusion bug class.

## Critical files this plan creates / modifies

In `FSA_model_dev/` (the home of the canonical model, per the charter):

- New: `lean/lakefile.lean`, `lean/lean-toolchain` — `lake` project
  config + pinned LEAN4 toolchain version.
- New: `lean/Fsa/V5/Types.lean` — `State6D`, `BimodalPhi`,
  `Params45`, `Obs5`, `RBFCoeffs`, `ControlOutput`,
  `SmoothSchedule`, `AppliedSchedule`. Dimensions encoded in types.
- New: `lean/Fsa/V5/Drift.lean` — drift function (typed defs +
  runnable impls).
- New: `lean/Fsa/V5/Diffusion.lean` — diffusion structure;
  `sigma_S` and `sigma_S_obs` are distinct fields.
- New: `lean/Fsa/V5/Plant.lean` — `StepwisePlant` reference.
- New: `lean/Fsa/V5/Schedule.lean` — RBF schedule decoder.
- New: `lean/Fsa/V5/Cost.lean` — chance-constrained cost; type
  signature forces per-particle `A_sep` (kills the particle-0
  template class).
- New: `lean/Fsa/V5/Obs.lean` — 5-channel observation model.
- New: `lean/python_bridge/` — ctypes wrappers exposing each LEAN4
  function as a Python callable.
- New: `tests/test_lean4_diff.py` — pytest + hypothesis differential
  test harness; samples from physiological-bound box; tolerance
  `1e-6` / `1e-4`.

Modified: existing FSA-v5 Python implementation files
(`models/fsa_high_res/_dynamics.py`, `_plant.py`, `control.py`,
`control_v5.py`, `estimation.py`) keep their public signatures
aligned to the LEAN4 spec. The earlier `sigma_S → sigma_S_obs`
rename and the Bug 2 per-particle separator fix already match
the LEAN4 spec; no further changes needed.

Modified: CI config to run the differential test on every PR.

## Existing infrastructure to reuse

- The 12 tests added in this session
  (`test_per_particle_separator.py`, `test_fsa_v5_param_dict.py`,
  `test_run_dir_atomic.py`) are already property-test-style. They
  remain valuable as Python-only regression tests, complementing the
  LEAN4 differential test rather than being replaced by it.
- The `diagnose_violation_rate.py` tool's three-formulation analysis
  is exactly the kind of explicit-spec-choice exercise LEAN4 forces at
  the type level — it's the prototype showing why explicit
  `SmoothSchedule` vs `AppliedSchedule` types matter.
- The existing `experiments_log.md` tabular format remains the
  empirical-bench status pane; LEAN4 governs *correctness*, the bench
  governs *biological/closed-loop quality*.

## Verification — how to know this is working

The plan succeeds when, on a fresh PR introducing a regression of any
of the three semantic/type-confusion historical bugs (`sigma_S`
collision reintroduced, particle-0 template reintroduced, post-hoc
wiring mismatch reintroduced):

1. The differential test (`pytest tests/test_lean4_diff.py`) produces
   a failing case on that PR before it can be merged. The failure
   reports a concrete input on which LEAN4 and Python disagree by
   more than the tolerance threshold.
2. `lake build` either still succeeds (if the regression is
   Python-only) or fails with a type error (if the regression touches
   the LEAN4 reference).
3. Diagnosis surfaces in CI within the same minute as `pytest`
   would normally finish — no GPU runs, no overnight launchers.
4. A *new* type of semantic/type-confusion bug — not previously
   identified — is also caught on at least one trial PR (else the
   framework is just memorising known failures, not finding new ones).

If any of these fail, the framework hasn't earned its keep.
