# Julia Port Charter for `smc2fc` — LaTeX Document Plan

  

## Context

  

This document is the second post-mortem-and-charter in the same

series as the LEAN4-first charter for the FSA model side

([`lean4_first_charter.pdf`](https://github.com/ajaytalati/FSA_model_dev/blob/claude/dev-sandbox-v4/LaTex_docs/lean4_first_charter.pdf)).

That charter addressed the **model side** (FSA-v5 mathematics, written

once in LEAN4 as the source of truth, agent-written Python

differentially tested). This new document addresses the

**framework side**: the `smc2fc` Bayesian filtering and control

codebase, whose ~5,600 lines of agent-touched Python pose the *same*

class of pattern-matching risks the LEAN4 charter just spent eight

pages diagnosing.

  

The argument the new document needs to make:

  

1. **Python + LLM is a perfect storm for the kind of math `smc2fc`

does.** No shape-aware types; `jnp.ndarray` covers everything from

a 1-D weight vector to a 4-D batched-particle-trajectory tensor.

No multiple dispatch. Mutation and pure-functional code coexist

in the same file (some functions are JIT-traced, some are not).

The closure-of-data trap that triggered the per-window recompile

bug documented in §6 of the framework doc was a Python-and-JAX

misstep, not a mathematical one. Agents pattern-matching on

`theta`, `particles`, `weights`, `log_weights`, `lp` propagate

shape and role confusion silently.

2. **Julia is structurally a much narrower surface for that

class of error.** Multiple dispatch + parametric types catch

shape and role mismatches at the function-signature level; a

function declared `drift(x::State{6}, θ::Params{37})` cannot

accept a 37-D state vector by accident. Functional style is the

default; mutation is opt-in (`!` suffix convention). Static

analysis via `JET.jl` is one command. The two-language problem

is moot — Julia is the high-level *and* fast layer.

3. **Julia and LEAN4 are aligned.** Both have parametric types,

abstract type hierarchies, and dispatch-based polymorphism.

Code reads more like the LaTeX it transcribes. The LEAN4

reference for the FSA model and a Julia implementation of the

`smc2fc` framework that consumes it form a single coherent

typed pipeline; Python sits awkwardly in the middle today.

4. **Most of the framework is already in the Julia ecosystem.**

`AdvancedHMC.jl` (the original HMC implementation BlackJAX is

modelled on), `OptimalTransport.jl` (Sinkhorn iterations,

entropic and unbalanced variants), `DifferentialEquations.jl` /

`StochasticDiffEq.jl` (the gold-standard SDE infrastructure

`diffrax` aspires to), `Distributions.jl`, `Bijectors.jl`

(unconstrained-space transforms), `Tullio.jl` (einsum). A

substantial fraction of `smc2fc/` is wiring these libraries

together in a domain-specific shape; the port re-uses them

directly.

5. **This is translation work between the two most common

scientific computing languages.** The user has explicitly

rejected inflated effort estimates. For a developer competent in

both languages, with the existing Julia scaffolding from §4

doing the heavy lifting, the full port is **approximately seven

hours** — most of which is wiring library calls together and

running differential tests, not writing new algorithmic code.

  

The output of this plan is **one LaTeX document** (the

"constitution") that a future coding agent reads at session start

before doing any port work, plus a per-module port-mapping table

that the same agent uses as a checklist.

  

## Output location and structure

  

A new document, intentionally placed in the **smc2fc repo** (since it

is about that repo), reusing the same `LaTex_docs/` directory pattern

as the framework doc:

  

```

/home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/

├── julia_port_charter.tex

└── julia_port_charter.pdf (compiled deliverable)

```

  

**The document is two-part and self-contained.** A future agent

reading only this one PDF must come away with the full constitution

for the joint workflow — both the LEAN4-first model-side process

*and* the Julia framework-side process. We therefore include the

LEAN4 charter verbatim as Part I, and the Julia port material as

Part II. Do not assume the agent has read

`lean4_first_charter.pdf`; assume they have not.

  

```

Part I — LEAN4-first charter for FSA model development (~7 pages,

verbatim from lean4_first_charter.tex in FSA_model_dev)

Part II — Julia port charter for the smc2fc framework (~10 pages,

new content)

Total — ~17 pages, single self-contained .tex file

```

  

Reuses the existing `preamble.tex` for math macros to keep notation

consistent with the framework doc.

  

## Section-by-section content (~17 pages total — Part I + Part II)

  

### Part I — LEAN4-first charter for FSA model development (~7 pages)

  

**Verbatim from

`/home/ajay/Repos/FSA_model_dev/LaTex_docs/lean4_first_charter.tex`.**

Copied into the Julia port `.tex` file as its own numbered part so

that the document is fully self-contained. The Part I content covers:

  

- §I.1 The problem in one paragraph (model side).

- §I.2 Concrete evidence of agent failures: variable-name collisions

(the 16-D vs 37-D `theta`), the verbatim Bug 5 incident, plan

churn (7:1 abandoned:executed ratio in `claude_plans/`), commit

`d886f36` errata-after-deploy.

- §I.3 Root-cause analysis: LLMs are surface pattern-matchers,

Python provides no formal scaffolding, "be more careful" doesn't

work.

- §I.4 The structural fix: LEAN4 as the source of truth.

- §I.5 The new model-side development process: LaTeX → LEAN4

transcription → `lake build` → Python (agent-written) → diff

test against LEAN4 binary.

- §I.6 Scope and timing: ~8 hours of focused translation work.

- §I.7 In/out scope.

- §I.8 Audit checklist for the model-side agent.

  

The transition from Part I to Part II is a single paragraph linking

the two: *the LEAN4 charter solves the model side; what about the

framework `smc2fc` itself, also Python, also agent-touched, also

exposed to the same pattern-matching risks?* That paragraph is the

hinge between the two halves.

  

### Part II — Julia port charter for the `smc2fc` framework (~10 pages)

  

#### §II.1 The problem in one paragraph

*~½ page.* The `smc2fc` codebase is doing the right mathematics; the

failure mode is the same as the FSA-v5 model side: agent

pattern-matching across a Python codebase with no type-level

enforcement of dimensions or roles. Part I solves half of this —

the model side — by making LEAN4 the formal source of truth for

the math `smc2fc` consumes. Part II solves the other half by

porting the framework itself to a language where the type system

catches the pattern-matching errors at compile time. The audience

is the same two populations — the user (accountability) and future

LLM agents (constitution loaded at session start).

  

#### §II.2 The Python + LLM perfect storm

*~2½ pages — concrete failure modes inside `smc2fc/` specifically.*

  

- **§2.1 Everything is `jnp.ndarray`.** Cite specific signatures:

- `jax_native_smc.py:67` — `_ess_at_delta(loglikelihood_fn,

particles, delta)` where `particles` is `(n_smc, d)` but the

type is just `jnp.ndarray`; an agent passing a single particle

`(d,)` gets a runtime broadcast surprise.

- `_gk_kernel.py:185` — `smooth_resample_basic(particles,

log_weights)`, where `log_weights` could be a 1-D vector of

length `n_pf` or a 2-D `(n_smc, n_pf)` block depending on

whether the caller already vmapped; nothing in the signature

says which.

- `tempered_smc.py` — closure capture of `data` per window that

caused the BlackJAX-wrapped backend to recompile per window

(the bug §6 of the framework doc documents); a Python+JAX

foible an agent has no way to predict from the signature.

- **§2.2 Mutability and pure-functional code coexist invisibly.** In

`smc2fc/control/diagnostics.py` the matplotlib helpers are

imperative side-effectful Python; in

`smc2fc/core/jax_native_smc.py` the SMC loop is pure-functional

for JIT compatibility. An agent editing one file reaches for the

pattern of the other and breaks the JIT contract. The LEAN4

charter's PRIMARY RULE — "build a semantic model BEFORE any

claim" — has to fire on every cross-file edit; empirically it

does not.

- **§2.3 Closures over data are a recompile landmine.** A

long-running session that thinks it is reusing a JIT-compiled

function silently re-traces because a dataclass's `__hash__`

changed. Documented in framework-doc §6.1; an agent cannot infer

this from any signature in the file.

- **§2.4 The vmap/while-loop nesting cliff.** §6.4 of the framework

doc records that NUTS-inside-vmap-inside-`lax.while_loop` is

60–85× slower than HMC because of dynamic-tree-building inside

nested control flow. This is a Python+JAX architectural

consequence, not a mathematical one — and exactly the kind of

trap an agent walks into by pattern-matching "NUTS is a strict

upgrade from HMC" from training data.

  

#### §II.3 Why Julia narrows the failure surface

*~2 pages.*

  

- **§3.1 Multiple dispatch + parametric types.** A Julia function

declared

```julia

function ess_at_delta(loglik::F, particles::Matrix{Float64},

delta::Float64) where {F<:Function}

...

end

```

refuses anything else at compile-time. `Vector{Float64}` and

`Matrix{Float64}` are different types; passing one where the other

is expected does not silently broadcast.

- **§3.2 Functional style is the default.** Mutation is opt-in via

the `!` suffix convention (`resample!(particles, weights)` mutates

in-place; `resample(particles, weights)` returns a fresh array).

An agent reading any function signature can tell instantly which

mode it is in.

- **§3.3 Code reads like the LaTeX.** A Julia drift function with

parametric types reads almost identically to the equation it

transcribes:

```julia

function drift(x::FSAState{6}, θ::DynParams)::FSATangent{6}

...

end

```

This is much closer to the LEAN4 spec than the equivalent

Python.

- **§3.4 No two-language problem.** Python+JAX exists because Python

is too slow for the inner loop. Julia is fast natively. The

JIT-traced / non-traced split that creates §2.2's invisible

mutability landmine is a Python artefact; in Julia, the inner

and outer loops are the same kind of code.

- **§3.5 Static analysis catches the rest.** `JET.jl` runs a static

type analysis on the entire package and flags every type

uncertainty. This is a single command and has no analogue in

Python (`mypy` doesn't reach into JAX trees).

- **§3.6 LEAN4 alignment.** A LEAN4 specification for the FSA

model (per the LEAN4-first charter) translates to Julia almost

one-to-one because both languages are dispatch-based and

type-driven. The LEAN4 → Julia transcription is shorter than

LEAN4 → Python by roughly the ratio of the language verbosities.

  

#### §II.4 Existing Julia scaffolding — what we re-use

*~1½ pages — explicit package list with what each replaces.*

  

The bulk of `smc2fc/` is wiring well-known algorithms together. Julia

already has each of those algorithms as a maintained library; the

port is mostly *replacing custom code with library calls*.

  

| Python `smc2fc` module | Julia replacement | Notes |

|---|---|---|

| `core/jax_native_smc.py` HMC kernel | `AdvancedHMC.jl` | The original HMC implementation BlackJAX modelled on. Direct drop-in. |

| `core/tempered_smc.py` | Custom (built on `AbstractMCMC.jl`) | No exact analogue but the abstractions are there; ~150 lines of Julia. |

| `core/sf_bridge.py` BW geodesic | `LinearAlgebra` (stdlib) + `Distributions.jl` | Matrix square root, multivariate Gaussian. ~50 lines. |

| `core/mass_matrix.py` | `Statistics.var` (stdlib) | Literal one-liner. |

| `filtering/sinkhorn.py` + `transport_kernel.py` + `resample.py:ot_resample_lr` | `OptimalTransport.jl` | Maintained, supports entropic + low-rank. |

| `filtering/_gk_kernel.py` Liu–West | Custom (~80 lines) | Standard moment-match + Gaussian smoother; trivial in Julia. |

| `filtering/gk_dpf_v3_lite.py` | Custom (~150 lines) | Bootstrap filter loop; reuses `Distributions.jl` for log-density. |

| `simulator/sde_solver_diffrax.py` | `StochasticDiffEq.jl` (part of `DifferentialEquations.jl`) | Gold standard. Replaces diffrax wholesale. |

| `transforms/unconstrained.py` | `Bijectors.jl` | The Julia ecosystem's standard bijection library. |

| Einsum-style ops (Liu–West kernels, RBF designs) | `Tullio.jl` or built-in broadcasting | Often broadcasting alone suffices. |

| Autodiff for HMC gradients | `Enzyme.jl` (preferred) or `Zygote.jl` | `Enzyme.jl` is more efficient for the kind of code here. |

| Static analysis | `JET.jl` | The "Julia answer to the LLM pattern-match risk". |

| Testing harness | `Test.jl` (stdlib) | Standard. |

| GPU arrays | `CUDA.jl` | NVIDIA backend; `CuArray` parametric type plays naturally with the generic `AbstractArray{T,N}` dispatch the rest of the package uses. |

| Cross-device kernels | `KernelAbstractions.jl` | Write a kernel once, runs on CPU (`CPU()`) or GPU (`CUDABackend()`). Used where broadcasting alone won't suffice. |

  

The custom code in the port — the SMC$^2$ tempering loop, the

bootstrap filter wrapper, Liu–West shrinkage, the cost-as-likelihood

controller loop, the bridge-warm-start glue — is **roughly 1,000–1,200

lines of Julia**. Compare to the ~5,600 lines of Python the port

replaces. The reduction is mostly the disappearance of JAX-specific

boilerplate (closures, `Partial` wrappers, `lax.while_loop`

scaffolding, manual `jit` placement).

  

#### §II.5 The port plan — file-by-file mapping

*~2 pages — the operational section.*

  

The Julia package is named `SMC2FC.jl` and lives at

`/home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC/` (sister

directory to the Python `smc2fc/`). The port proceeds bottom-up so

that lower layers are tested before upper layers depend on them.

  

**§II.5.0 Strategic principle: the port is optimised for Julia, not

a Python re-skin.** The most common mistake an agent will make on

this port is translating Python line-by-line. The result is a

wooden, Python-shaped Julia package that realises none of Julia's

actual advantages — and is therefore not worth the porting effort

at all. The point of doing this work is to land in a codebase that

is *idiomatic Julia*: shorter, faster, more type-safe, and better

matched to the LEAN4 spec language than the Python original. A

``Python in Julia syntax'' port fails the charter even if it

type-checks and passes the differential test, because it does not

realise any of the §II.3 benefits.

  

The translation discipline is therefore to port the *algorithm* and

let Julia's native features (loops, multiple dispatch, broadcasting,

`AbstractArray` dispatch) express it the Julia way:

  

- **Replace JAX control flow primitives with native Julia loops.**

`jax.lax.scan`, `jax.lax.fori_loop`, `jax.lax.while_loop` exist

in JAX because Python's interpreter overhead makes a plain

`for` loop unusably slow. **Julia has no such overhead.** A

standard `for k in 1:n` loop in Julia is fast under both

ordinary execution and `Enzyme.jl` / `Zygote.jl` autodiff. Do

not port `lax.scan` to a `Tullio.jl` reduction or to a

`KernelAbstractions.jl` kernel out of habit — port it to a

`for` loop. The control-flow scaffolding that bloats the JAX

source files (§II.4's "~5,600 LOC vs ~1,200 LOC" reduction

comes mostly from this) collapses into ordinary Julia.

- **Replace conditional logic with multiple dispatch.** Where

Python branches on flags (e.g., the `bridge_type` switch in

`core/sf_bridge.py`, the soft-vs-hard chance-constrained

switch in `control_v5.py`), Julia dispatches on a small set of

marker types (`struct GaussianBridge end`, `struct

SchrodingerFollmerBridge end`, `struct SoftSurrogate end`,

`struct HardIndicator end`). The conditional logic disappears

into method definitions; each behaviour lives in its own

method, the dispatcher chooses at compile time, and the agent

cannot accidentally reach the wrong branch.

- **Flatten nesting.** Python's `vmap`-over-particles +

`lax.while_loop`-over-tempering + `lax.scan`-over-bins nests

three primitives because each layer needs its own

JAX-recognised iterator. In Julia: nested `for` loops, end of

story. Broadcasting (`f.(particles)`) handles the `vmap`

layer; ordinary loops handle the rest.

- **Use `AbstractArray{T,N}` in signatures, not concrete `Array`

or `CuArray`.** This is the GPU-portability discipline from

§II.4 stated as a translation rule: write the function once

against the abstract type, run it on whichever backend the

caller supplied. The agent should never need to write two

versions of any function.

  

These principles apply pervasively across the phases that follow.

The phase descriptions list *what* to port; this subsection is

*how* to port it.

  

**Phase enumeration begins below.**

  

**Phase 1 — Foundations (~1 hour):**

  

1. `Project.toml` + `Manifest.toml` with the dependency graph from

§4 (including `CUDA.jl` and `KernelAbstractions.jl`).

2. `src/Types.jl` — the parametric types that encode the

dimensional distinctions Python erases. `State{N}`,

`Particles{N,D}`, `Weights{N}`, `LogWeights{N}`, `RBFCoeffs{D}`,

`DynParams`, `CtrlParams` etc. **All array fields use the

abstract `AbstractArray{T,N}` so `Array` (CPU) and `CuArray`

(GPU) are accepted by the same code.** This is the

LEAN4-charter-spirit applied to the framework.

3. `src/Config.jl` — translation of `core/config.py` SMCConfig as a

`@kwdef` struct, plus a `backend::Symbol = :cpu` field

(`:cpu` or `:cuda`) that selects the array constructor.

4. `src/Transforms.jl` — wraps `Bijectors.jl` for the unconstrained-

space mapping.

  

**Phase 2 — Filtering (~2 hours):**

  

5. `src/Filtering/Kernels.jl` — Liu–West shrinkage moment match,

Gaussian smoothing kernels, ESS computation.

6. `src/Filtering/Bootstrap.jl` — the inner bootstrap filter,

producing $\widehat{L}_N(\theta_{\rm dyn})$.

7. `src/Filtering/OT.jl` — wraps `OptimalTransport.jl` for the OT

rescue; implements the sigmoid blend with the systematic-resample

output.

8. Tests for Phase 2: differential against Python on a battery of

random SDE+observation pairs. Tolerance $10^{-6}$ on per-step

weights, $10^{-4}$ on integrated likelihood. **Tests run twice:

once with `CPU()` arrays, once with `CUDA.CuArray`; both must

pass within the same tolerance.**

  

**Phase 3 — Outer SMC$^2$ (~1.5 hours):**

  

9. `src/SMC2/Tempering.jl` — adaptive $\delta\lambda$ via

bisection. Direct port of `_solve_delta_for_ess`.

10. `src/SMC2/HMC.jl` — wraps `AdvancedHMC.jl` for the rejuvenation

moves; re-estimates the diagonal mass matrix per tempering level.

11. `src/SMC2/TemperedSMC.jl` — the outer loop, calling Phase 2 for

likelihood, Phase 3.1–3.2 for tempering and HMC.

12. `src/SMC2/Bridge.jl` — Gaussian + BW-geodesic warm-starts.

13. Tests for Phase 3: differential against Python on the same

benchmark `tests/test_obs_consistency_v5.py` already uses.

  

**Phase 4 — Control (~1 hour):**

  

14. `src/Control/Spec.jl` — `ControlSpec` struct (cost callable,

decoder, horizon, initial state).

15. `src/Control/Calibration.jl` — $\beta_{\max}$ auto-calibration.

16. `src/Control/RBFSchedule.jl` — RBF schedule decoder.

17. `src/Control/TemperedSMC.jl` — re-uses Phase 3.3 with the

cost-as-likelihood substitution.

18. Tests for Phase 4: differential against Python on the

chance-constrained controller benchmark.

  

**Phase 5 — Plant + Simulator (~½ hour):**

  

19. `src/Simulator/SDEModel.jl` — wraps `StochasticDiffEq.jl`.

20. `src/Simulator/Observations.jl` — per-channel observation

sampling.

  

**Phase 6 — End-to-end + GPU benchmark (~1.5 hours):**

  

21. A full-pipeline test: run a complete inference + control loop

on the FSA-v5 model (consuming the LEAN4 reference from the

other charter), compare end-to-end against the Python pipeline

on the same inputs. Run on **both backends** (`backend = :cpu`

and `backend = :cuda`), confirm numerical agreement to within

the same tolerances as Phase 2.

22. JET.jl static analysis pass; fix any flagged type uncertainties.

23. Benchmark per-window cost on CPU and GPU against the JAX-native

Python baseline of ~1 second/window. Record the numbers in the

package README; this is the artefact that justifies the port to

anyone reading it later.

  

#### §II.6 Scope and timing

*~½ page.*

  

**This is translation between two scientific computing languages.**

The user has explicitly rejected inflated estimates. With the

existing Julia scaffolding from §4 doing the heavy lifting, the

port is:

  

| Phase | Estimate |

|---|---|

| Phase 1 — Foundations | 1 hour |

| Phase 2 — Filtering (CPU + GPU) | 2 hours |

| Phase 3 — Outer SMC² (CPU + GPU) | 1.5 hours |

| Phase 4 — Control (CPU + GPU) | 1 hour |

| Phase 5 — Plant + Simulator (CPU + GPU) | 0.5 hour |

| Phase 6 — End-to-end + GPU benchmark | 1.5 hours |

| **Total** | **~7.5 hours of focused work** |

  

**The CPU/GPU dual-target adds essentially nothing** to the

per-phase budgets above because Julia's generic dispatch on

`AbstractArray{T,N}` means the same code runs on both `Array`

(CPU) and `CuArray` (GPU); the library packages

(`AdvancedHMC.jl`, `StochasticDiffEq.jl`, `OptimalTransport.jl`,

`Distributions.jl`, `Bijectors.jl`) already have GPU support

upstream. The marginal GPU cost is in (a) running the existing

test suite a second time on `CuArray`, and (b) the Phase 6

benchmark — both already accounted for in the table.

  

For a developer competent in both languages, this is a single

focused day's work. Most of it is wiring library calls together

and running differential tests; the algorithmic content is not

re-derived but transcribed. The resulting Julia is markedly

shorter (~1,200 LOC vs ~5,600 LOC) because the boilerplate that

Python+JAX requires (closures, `Partial` wrappers, `lax.while_loop`

scaffolding, manual `jit` placement) collapses in Julia.

  

**The 7.5-hour estimate assumes an idiomatic Julia port, not a

line-by-line Python re-skin.** A wooden translation that ports

each `lax.scan` to a Tullio reduction and keeps Python-shaped

control flow would technically compile in less time, but it would

not realise any of the §II.3 benefits and would fail the §II.5.0

discipline. The estimate above is for code written the Julia way:

plain `for` loops, multiple dispatch on marker types,

`AbstractArray{T,N}` signatures, broadcasting where it is the

cleanest expression. If the agent finds itself ported the same

algorithm faster than the table predicts, the result should still

be checked against §II.5.0 — speed is not the objective; an

idiomatic, type-safe, GPU-portable Julia codebase is.

  

The cost/value frame: a single 7-hour port amortises across

**every future model the framework runs** (the JAX-native vs

BlackJAX backend split disappears; the per-window recompile

landmine disappears; the NUTS-inside-vmap cliff disappears; the

agent pattern-match risk on framework internals disappears). The

current trajectory — paying repeatedly for agent rounds of

pattern-match-then-debug-Python+JAX-internals — is what this

charter exists to stop.

  

#### §II.7 What is and isn't covered

  

**In scope.** A Julia package `SMC2FC.jl` providing

filtering (bootstrap PF + OT rescue), outer SMC$^2$ (tempered + HMC +

adaptive mass matrix + bridges), control-as-inference, an

SDE simulator, and the unconstrained-space transforms.

**Both CPU and GPU (CUDA) backends are in scope from day one** —

selectable via the `backend::Symbol` field on `SMCConfig`. Every

test in the suite runs on both. Differential tests against the

Python implementation on the existing benchmarks for both

backends.

  

**Out of scope (Python stays in service).** The Python `smc2fc/`

package continues to exist during and after the port. The Julia

port does *not* replace it; it sits alongside as a type-checked

alternative implementation. Existing user code that imports the

Python package is unaffected.

  

**Out of scope (settled in framework doc).** The hard-vs-soft

chance-constrained finding (use soft only), the NUTS-as-HMC-

replacement dead-end, the SF/BW-vs-Gaussian-bridge equivalence

finding. These results carry over from Python; the Julia port

replicates the soft-HMC + Gaussian-bridge defaults.

  

#### §II.8 Audit checklist for the porting agent

  

A future agent runs this at session start, before touching any

Julia file:

  

- [ ] Is there a Python reference for the function I am about to

port?

- [ ] Does the LEAN4-first charter's process (Part I) apply

(i.e., am I porting model code that should consume a LEAN4

spec)? If yes, does the spec exist?

- [ ] Have I declared parametric types for every input and output,

capturing the dimensional information Python erases?

- [ ] Have I left array fields as `AbstractArray{T,N}` so the same

code accepts `Array` (CPU) and `CuArray` (GPU)?

- [ ] Did I port the *algorithm*, not the *code*? Specifically:

no `jax.lax.scan` / `lax.fori_loop` / `lax.while_loop`

analogues survived the translation — they became plain

Julia `for` loops. (Per §II.5.0.)

- [ ] Did I express type-dispatched behaviours via multiple

dispatch (one method per marker struct) rather than as

flag-driven conditionals carried over from Python?

(Per §II.5.0.)

- [ ] Does `julia --project=. -e 'using Pkg; Pkg.test()'` pass on

**both** the CPU and the CUDA test paths?

- [ ] Does `julia --project=. -e 'using JET; report_package(SMC2FC)'`

report no type uncertainties?

- [ ] Does the differential test against the Python implementation

pass within the tolerance threshold for the function I just

ported, on both backends?

- [ ] Have I updated the §II.5 port-mapping table to mark this

file as ported?

  

If any box is unchecked, the agent does not proceed.

  

## Length target

  

~17 pages total: ~7 pages Part I (LEAN4 charter, verbatim) + ~10

pages Part II (Julia port). Single `.tex` file. One table per

section where it helps; no figures.

  

## Critical files

  

**To create:**

- `/home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/julia_port_charter.tex`

  

**To read during drafting (read-only, for citation and inclusion):**

- `/home/ajay/Repos/FSA_model_dev/LaTex_docs/lean4_first_charter.tex`

— **the source of Part I, included verbatim** (modulo

light-touch numbering changes for the I.1, I.2, ... scheme).

- `/home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/main.pdf`

— the framework doc, especially §6 (JAX-native + failed NUTS) and

§3–§5 (filter, OT rescue, SMC²) for the architecture being ported.

- `smc2fc/core/jax_native_smc.py` — concrete signatures cited in §II.2.

- `smc2fc/filtering/_gk_kernel.py` — concrete signatures cited in §II.2.

- `smc2fc/core/tempered_smc.py` — closure-recompile bug for §II.2.

- `LaTex_docs/preamble.tex` — reuse for notation consistency.

  

## Verification

  

1. `cd LaTex_docs && pdflatex julia_port_charter.tex && pdflatex

julia_port_charter.tex` — clean compile, ~17 pages.

2. Part I content matches `lean4_first_charter.tex` byte-for-byte

(modulo the numbering scheme change). Spot-check via `diff`.

3. Every cited file:line reference checked against current state via

Bash before commit.

4. The §II.6 timing table totals to ~7.5 **hours** (not days, not

weeks) and contains no per-phase estimate larger than 2 hours.

**No inflated estimates.** Verification step is to grep the

compiled PDF text for "weeks" and "days" and reject any usage

that refers to the port itself.

5. Every Julia package in §II.4 is a real, currently-maintained

package, including the GPU stack (`CUDA.jl`,

`KernelAbstractions.jl`); cross-checked at draft time.

6. The CPU/GPU dual-target requirement is reflected in:

(a) §II.4's table (CUDA.jl + KernelAbstractions.jl rows present),

(b) §II.5 phases 1–6 (every test runs on both backends),

(c) §II.7's in-scope statement (no longer "deferred"),

(d) §II.8's audit checklist (both-backends row present).

7. The "translate algorithms not code" discipline (§II.5.0) is

reflected in §II.8's audit checklist (the `lax.scan`/dispatch

rows) and is referenced from the phase descriptions where

relevant.

7. The document compiles to a single, self-contained PDF the user

(and future agents) can read on the GitHub blob URL.