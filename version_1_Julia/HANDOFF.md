# `version_1_Julia/` handoff — read first

Durable architecture note for future agents (and post-compact me).
Written 2026-05-06 right before starting the FSA-v2 (`fsa_high_res`)
port. Branch: `Julia-port-verison-1`. Origin:
`https://github.com/ajaytalati/python-smc2-filtering-control`.

## TL;DR for a fresh agent

We are porting the three test models from `version_1/models/` to Julia
on top of the `SMC2FC.jl` framework at `julia/SMC2FC/`. **Two of three
models are done, one is not started yet.**

| stage | model                | status     |
|-------|----------------------|------------|
| A     | scalar_ou_lqg        | ✓ done     |
| B     | bistable_controlled  | ✓ done — also has the GPU parallel-chains HMC reference impl |
| D     | fsa_high_res (v2 Banister) | **NOT YET PORTED — your task** |

**Recommended sampler stack for FSA-v2**: GPU parallel-chains HMC + ChEES
adaptation. **Reference implementation lives at**
[`models/bistable_controlled/gpu_pf.jl`](models/bistable_controlled/gpu_pf.jl) —
your job for FSA-v2 is to write a model-specific equivalent there, then
reuse the rest of the bench scaffolding.

## What's on the branch (after the GPU work)

Recent commits, newest first:

```
532fc5d  GPU parallel-chains B3   (153× per HMC move, 6× B3 e2e)
74fb468  GPU log-density + FD gradient + batched kernel building blocks
5d85576  AutoMALA sampler
8d60bab  MALA + thread-safety fix (MersenneTwister race in @threads)
c901d3e  Enzyme + NUTS samplers + ad_backend / sampler config
4e70f96  B3 v3 with bigger HMC budget
e6dd89b  B3 v1 (HMC, low budget)
937659e  Phase 6 follow-up #2.5: fused KernelAbstractions kernel for forward-only PF saturation
329eaae  RESULT.md per model with Python ↔ Julia comparison
b5de78e  PNG diagnostic plots (force-added past the **/outputs/**/*.png .gitignore)
5fc00ba  Panel-matched plotting (utils/diagnostics.jl)
1a0b860  version_1_Julia v1: scalar_ou_lqg + bistable_controlled
7a03983  SMC2FC.jl Phase 6 follow-ups #1 + #2 (AD-compatible PF + GPU PF)
3bab302  SMC2FC.jl v1 (charter Part II)
```

`392d85b` (between `7a03983` and `1a0b860`) is **another agent's** FSA-v5
work in `version_3_julia/` — leave it alone.

## The architecture we converged on

The Phase 1 inference problem (8-D θ, T = 144 obs, bistable double-well)
broke our framework's CPU samplers in interesting ways. After 8 attempts
(HMC at varying budgets, Enzyme reverse-mode AD, NUTS, MALA, AutoMALA,
ChEES on CPU, GPU sequential, GPU parallel-chains) the WINNING stack is:

```
For each tempering level of outer SMC² (CPU bookkeeping):
  for each leapfrog step in HMC:
    ONE batched kernel launch over (n_smc × (1 + 2·d_θ)) sub-chains
    → returns n_smc gradients in parallel
    → MH accept per chain on CPU (cheap)
```

### Per-call performance (RTX 5090, K_per_chain = 2_000):

| op | time | notes |
|---|---|---|
| primal log-density at 1 θ (K=200k fp32) | 2 ms | the "bench_gpu_bistable_fused" baseline |
| FD gradient at 1 θ (16 PF evals at K=200k) | 22 ms | sequential — what we DON'T do |
| **batched FD gradient at 1 θ (K=2k × 17 chains in 1 launch)** | **12 ms** | 3× faster, equal total particle work |
| **64-chain HMC move (8 leapfrog × 1 batched grad each)** | **134 ms** | **153× faster than serialised GPU HMC** (was 20.5 s) |
| Phase 1 filter (11 tempering levels) | **23 s** | was 146 s on CPU AutoMALA — 6× e2e |

### B3 closed-loop result with the GPU stack

| metric | best CPU result | **GPU parallel-chains** |
|---|---|---|
| α rel.err vs truth | 5.9 % (HMC v3) | **2.4 %** ← best of any sampler tried |
| cost / oracle ratio | 0.888× (AutoMALA) | **0.740×** ← best |
| transition rate (gate ≥ 80 %) | 100 % | 100 % |

Plot: [`outputs/bistable_controlled/B3_gpu_parallel_diagnostic.png`](outputs/bistable_controlled/B3_gpu_parallel_diagnostic.png).
Both gates pass; schedule shape now visibly matches the oracle.

### Caveat — known structural property of the GPU stack

The GPU PF is pure bootstrap (no per-step Liu-West shrinkage / no
systematic resampling between obs). For **identified** parameters like
α this gives the best recovery of any sampler we tried. For
**partially-identified** parameters (σ_u, γ, σ_obs in the bistable
Phase 1 case where x is observed but u is not), it explores a different
mode of the joint posterior than the CPU `gk_dpf_v3_lite` path does.
Both are valid estimators of the same posterior; they put the
finite-sample variance in different places.

For FSA-v2 this means: the GPU stack will work great on parameters the
HR/RPE channels strongly identify, and may need the CPU framework as a
sanity check on parameters they don't.

## Key files for the FSA-v2 port

You should mirror this structure for `models/fsa_high_res/` (v2 = Banister,
3-state):

```
version_1_Julia/
├── HANDOFF.md                                      ← you are here
├── models/bistable_controlled/                     ← REFERENCE — copy this pattern
│   ├── _dynamics.jl                                ← drift + diffusion (CPU)
│   ├── simulation.jl                               ← simulator + obs
│   ├── estimation.jl                               ← framework EstimationModel
│   ├── control.jl                                  ← cost fn + RBF schedule
│   ├── gpu_pf.jl                                   ← ★ GPU kernels — the model-specific GPU work
│   └── BistableControlled.jl                       ← umbrella module
└── tools/
    ├── bench_smc_filter_bistable.jl                ← B1: filter (CPU, framework path)
    ├── bench_smc_control_bistable.jl               ← B2: control (CPU, framework path)
    ├── bench_smc_closed_loop_bistable.jl           ← B3: filter+plan (CPU AutoMALA — slow)
    └── bench_b3_gpu_parallel.jl                    ← ★ B3 with GPU parallel-chains HMC
```

Items marked ★ are the GPU work that's the deliverable. The CPU benches
are kept as fallbacks + sanity checks.

### `gpu_pf.jl` — the GPU kernel + parallel-chains primitives

[`models/bistable_controlled/gpu_pf.jl`](models/bistable_controlled/gpu_pf.jl)
is the reference implementation. It contains:

1. `bootstrap_pf_kernel!` (single chain, K particles) — the original
   Phase 6 #2.5 kernel
2. `BistableGPUTarget` + `gpu_log_density(target, u)` + `gpu_log_density_with_grad(target, u)`
   — the single-chain wrapper
3. `bootstrap_pf_kernel_batched!` (M chains, K particles each, **shared
   noise grid for CRN**) — the parallel-chains kernel
4. `BistableGPUTargetBatched` + `gpu_log_density_batched(target, U)` +
   `gpu_fd_gradient_batched(target, u)` — the batched wrappers
5. `gpu_grads_parallel_chains(target, U)` + `parallel_hmc_one_move!(U, target, ε, L, ...)`
   — the n_smc-chain parallel HMC primitive

For FSA-v2 you write the equivalent of (1) and (3) with FSA-v2 dynamics
(3-state Banister: `dB`, `dF`, `dA` SDEs with square-root Itô diffusion);
the rest of the abstractions reuse cleanly because the kernels just
return per-particle log-weights into a flat CuArray.

### The CRN noise critical-detail

The batched kernel layout that broke and then worked:

WRONG (ate 16× more memory, broke FD gradients):
```
noise_x : (M_max · K_per_chain, T+1)    # per-chain noise
threads index noise_x[global_thread_id, k]
```

RIGHT (CRN restored, ~M× memory saving):
```
noise_x : (K_per_chain, T+1)            # SHARED across chains
threads index noise_x[((global_thread_id - 1) % K_per_chain) + 1, k]
```

With CRN, chain m at θ and chain m' at θ + h·e_i see the same noise
sample at every (particle, t). The FD gradient signal isn't drowned by
per-chain noise differences. **This is the single most important
implementation detail of the GPU work.**

## SMC2FC.jl framework state

[`julia/SMC2FC/`](../julia/SMC2FC/) — the model-agnostic framework. Has
been stable through all this work. Currently has 4 CPU samplers
(`:HMC`, `:NUTS`, `:MALA`, `:AutoMALA`) + 2 AD backends (`:ForwardDiff`,
`:Enzyme`). 304 tests pass + JET clean.

The GPU parallel-chains pattern is **NOT YET in the framework as a
generic option** — it's currently a model-specific bench in
`version_1_Julia/models/bistable_controlled/gpu_pf.jl` +
`tools/bench_b3_gpu_parallel.jl`. Generalising it (so the framework's
`_tempered_step!` can dispatch to GPU parallel chains when the model
provides a GPU target) is **future work**. For FSA-v2 the simplest path
is to copy the bistable pattern.

## What you should do for FSA-v2

1. Read `version_1/models/fsa_high_res/_dynamics.py` and `control.py` —
   that's the Python reference (drift = Banister 3-state; cost involves
   integral of A(t)).
2. Mirror the bistable file structure under `models/fsa_high_res/`:
   - `_dynamics.jl` — drift + square-root Itô diffusion (Jacobi for B,
     CIR for F and A — keep states in their physiological domains).
   - `simulation.jl` — `PARAM_SET_A`, `INIT_STATE_A`, `EXOGENOUS_A`,
     `simulate_em` + `simulate_diffeq`. Use the Python's exact param
     values verbatim.
   - `estimation.jl` — framework `EstimationModel` with the FSA-v2
     priors from the Python.
   - `control.jl` — cost functional from the Python's `control.py`
     (it's the integral-of-A cost with a soft constraint on F).
   - `gpu_pf.jl` — model-specific GPU PF (single + batched kernels).
     Copy bistable's structure; replace the per-particle drift inside
     the kernel with FSA-v2's 3-state Banister update. Keep the CRN
     noise layout exact.
   - `FSAHighRes.jl` — umbrella module.
3. Create the benches:
   - `tools/bench_fsa_filter.jl` (CPU) — sanity check on framework path
   - `tools/bench_fsa_gpu_parallel.jl` (GPU) — the full GPU SMC² for
     filter + control. Reuse `bench_b3_gpu_parallel.jl` structure.
4. Match the Python's plots: there are 4 of them (T = 28, 42, 56, 84 d
   horizons). Mirror the layout. Save to
   `outputs/fsa_high_res/D_v2_T<H>_diagnostic_julia.png`.
5. Write `outputs/fsa_high_res/RESULT.md` with Python ↔ Julia
   side-by-side, mirroring the existing two RESULT.md files.

## How to run + verify

```bash
conda activate comfyenv
cd version_1_Julia/

# CPU sanity
julia --threads auto --project=. tools/bench_smc_filter_bistable.jl

# GPU end-to-end (this is what the user actually wants)
julia --threads auto --project=. tools/bench_b3_gpu_parallel.jl
```

Threading via `--threads auto` (default 1 thread = serialised → bad).
GPU is RTX 5090 with `nvidia-smi` driver 595.58.03+ (post the saturation
fix the framework needs).

## Compaction-survival meta-notes

If a new agent reads this and wants to rebuild the situation:

1. **The GPU parallel-chains HMC pattern is the production path** for
   FSA-v2 and any future model with a GPU-friendly drift function.
2. **The CPU framework path** (`bootstrap_log_likelihood` + `run_smc_window`
   with `:AutoMALA` or `:MALA`) is the **fallback** for when you don't
   want to write a GPU kernel.
3. **The framework's existing CPU benches** for bistable + scalar_ou_lqg
   already work and pass their gates — those are the reference for "what
   a working port looks like".
4. **Both gates pass on every B3 attempt** — what we've been chasing is
   the visible schedule-shape match in panel [0,1] of the plot, which
   only the GPU stack achieves cleanly.
5. **Don't redo the AD-backend work**: ForwardDiff with d ≤ 8 is faster
   than Enzyme on this code (engineering finding documented in
   `models/bistable_controlled/loglik_enzyme.jl`).
6. **Don't try to make Pigeons.jl work** — its newer versions
   resolver-conflict with our deps; we rolled our own AutoMALA + ChEES.
