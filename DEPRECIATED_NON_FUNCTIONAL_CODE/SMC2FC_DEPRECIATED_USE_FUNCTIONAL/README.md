# SMC2FC.jl

Julia port of the `smc2fc` framework — adaptive tempered SMC² for joint
state + parameter inference, plus the same outer kernel re-used as a
control-as-inference engine.

This package is **Part II of the joint LEAN4-and-Julia charter** for the
FSA + smc2fc stack; the charter PDF lives at
[`LaTex_docs/julia_port_charter.pdf`](../../LaTex_docs/julia_port_charter.pdf).

## ⭐ Recommended high-perf path: GPU parallel-chains HMC + ChEES

For models with a GPU-friendly drift function (most SDE models qualify),
the **production sampler stack** is parallel-chains HMC where all `n_smc`
rejuvenation chains run in **one batched kernel launch per leapfrog step**.
The reference implementation lives next to the bistable model:

  - Pattern: [`version_1_Julia/models/bistable_controlled/gpu_pf.jl`](../../version_1_Julia/models/bistable_controlled/gpu_pf.jl)
  - Bench:   [`version_1_Julia/tools/bench_b3_gpu_parallel.jl`](../../version_1_Julia/tools/bench_b3_gpu_parallel.jl)
  - Architecture notes + B3 numbers: [`version_1_Julia/HANDOFF.md`](../../version_1_Julia/HANDOFF.md)

Measured on RTX 5090 (n_smc = 64 chains, K_per_chain = 2_000, 8-D θ,
T = 144 obs): **134 ms per HMC move (153× faster than serialised GPU
HMC, 6× faster end-to-end than CPU AutoMALA).** Both gates pass; cost
ratio 0.740× oracle (best across 8 sampler/AD combinations tried).

For a **new model** the porting pattern is: copy
`models/bistable_controlled/gpu_pf.jl`, replace the per-particle drift
inside the kernel with the model's drift, keep the CRN noise layout
unchanged. The `BistableGPUTargetBatched` + `gpu_grads_parallel_chains`
+ `parallel_hmc_one_move!` primitives reuse cleanly because they
operate on flat `(M·K,)` log-weight arrays.

The CPU samplers below (HMC, NUTS, MALA, AutoMALA via AdvancedHMC.jl)
are the **fallback path** for when you don't want to write a GPU
kernel — they're slower but model-agnostic.

## Status: all phases complete (incl. Phase 6 follow-ups)

| Phase | Component                                                | Tests | Status |
|------:|:---------------------------------------------------------|------:|:------:|
| 1     | Foundations (Types, Config, Transforms)                  |   204 |   ✓    |
| 2     | Filtering (Kernels, OT, Bootstrap PF, GPU)               |    44 |   ✓    |
| 3     | Outer SMC² (Tempering, HMC, TemperedSMC, Bridge)         |    14 |   ✓    |
| 4     | Control (Spec, Calibration, RBFSchedule, ControlLoop)    |    12 |   ✓    |
| 5     | Plant + Simulator (StochasticDiffEq + Observations)      |     6 |   ✓    |
| 6     | End-to-end + JET                                         |     7 |   ✓    |
| 6.1   | Follow-up: AD-compatible bootstrap PF + PF inside HMC    |    12 |   ✓    |
| 6.2   | Follow-up: GPU end-to-end bootstrap PF (CuArray buffers) |     5 |   ✓    |
|       | **Total**                                                | **304** | **All passing** |

JET.jl static analysis pass (charter §18 audit gate) is clean — no type
uncertainties or instabilities flagged inside any SMC2FC submodule.

## Layout

```
SMC2FC.jl
├── Project.toml                # deps including CUDA, AdvancedHMC, StochasticDiffEq
├── src/
│   ├── SMC2FC.jl                # top module
│   ├── Types.jl                  # State{N}, DynParams, Particle, GPU/CPU type boundaries, marker types
│   ├── Config.jl                 # SMCConfig, RollingConfig, MissingDataConfig
│   ├── Transforms.jl             # PriorType hierarchy + dispatch-based bijections
│   ├── EstimationModel.jl        # frozen-struct contract for filter side
│   ├── Filtering/
│   │   ├── Kernels.jl            # ESS, Silverman bandwidth, Liu-West, smooth-resample variants
│   │   ├── OT.jl                  # Nyström + low-rank Sinkhorn + barycentric projection + sigmoid blend
│   │   └── Bootstrap.jl           # inner PF producing log p(y|θ); zero-allocation buffers
│   ├── SMC2/
│   │   ├── MassMatrix.jl          # diagonal inverse-mass estimator
│   │   ├── Sampling.jl            # prior cloud (multiple-dispatch over PriorType)
│   │   ├── Tempering.jl           # adaptive δλ via bisection
│   │   ├── HMC.jl                 # AdvancedHMC.jl + ForwardDiff wrapper
│   │   ├── Bridge.jl              # marker-dispatched warm-start (GaussianBridge / SF stub)
│   │   └── TemperedSMC.jl         # outer driver: cold-start + bridged windows
│   ├── Control/
│   │   ├── RBFSchedule.jl         # Gaussian-RBF basis with marker-dispatched output
│   │   ├── Spec.jl                # ControlSpec frozen struct
│   │   ├── Calibration.jl         # β_max + CRN noise grids
│   │   └── TemperedSMC.jl         # ControlLoop: outer SMC² with cost-as-likelihood
│   └── Simulator/
│       ├── SDEModel.jl            # thin StochasticDiffEq.jl wrapper
│       └── Observations.jl        # dependency-ordered channel sampling
├── test/                          # Pkg.test() drives all 287 tests
└── benchmarks/run_phase6_bench.jl # per-window timings (CPU + GPU)
```

## Running the test suite

```bash
conda activate comfyenv     # has Julia 1.11 + CUDA 595.58.03 + RTX 5090
cd julia/SMC2FC
julia --project=. -e 'using Pkg; Pkg.test()'
```

The full suite takes ~80 s and exercises:
- All Phase 1 transforms round-trip per prior type
- Phase 2 ESS / Silverman / Liu-West / OT correctness on closed forms
- Phase 2 bootstrap PF vs the Kalman closed-form on AR(1) + Gaussian obs
- Phase 2 CPU vs GPU parity within `1e-4` (charter §15.7 tolerance)
- Phase 3 outer SMC² recovers the conjugate-Gaussian posterior mean within MC noise
- Phase 4 control SMC² recovers a known optimum on a quadratic cost
- Phase 5 SDE simulator recovers OU stationary moments; channels respect deps
- Phase 6 outer SMC² + AdvancedHMC.jl + ForwardDiff stack recovers AR(1) (a, b, ρ)
- JET pass scoped to SMC2FC submodules (no upstream-dep noise)

## Per-window benchmark (Phase 6, RTX 5090)

```
Bench 1 — Phase 2 kernel ops (CPU vs GPU, K=400, n_st=6)
  CPU compute_ess              : 2.0 µs
  CPU silverman_bandwidth       : 0.9 µs
  CPU log_kernel_matrix         : 0.85 ms
  GPU compute_ess               : 63 µs
  GPU silverman_bandwidth       : 77 µs
  GPU log_kernel_matrix         : 0.18 ms          (4.7× faster than CPU)

Bench 2 — Phase 2 bootstrap PF (CPU, K=400, T=50)
  bootstrap_log_likelihood      : 10.5 ms / call

Bench 3 — Phase 3 outer SMC² window (CPU, n_smc=128, d=4, Gaussian target)
  full prior→posterior run       : 0.09 s
```

For comparison, the Python JAX-native baseline cited in charter §15.7 is
~1 s/window for the FSA-v5 production configuration (K=400, T=27 windows ×
14 days, 37-D θ). The Julia per-call costs are favourable; the
per-FSA-v5-window number requires the AD-compatible PF (see "Phase 6
follow-up" below).

To re-run on this machine:

```bash
conda activate comfyenv
julia --project=. benchmarks/run_phase6_bench.jl
```

## Charter §15.1 design discipline — what I kept

The charter is explicit that a "Python in Julia syntax" port fails even if it
type-checks. Concrete examples of where the Julia idioms beat a 1-1 translation:

| Python pattern                          | Julia replacement                         | Where           |
|-----------------------------------------|--------------------------------------------|-----------------|
| 7 indicator arrays (`is_log`, `is_logit`,…) | Multiple-dispatch over `PriorType`         | `Transforms.jl` |
| `bridge_type` string flag + nested ifs   | Marker types dispatched via `bridge_init` | `SMC2/Bridge.jl` |
| `output: 'identity'\|'softplus'\|'sigmoid'` | Marker types dispatched via `apply_output` | `RBFSchedule.jl` |
| `jnp.ndarray` for both 16-D and 37-D θ   | `ComponentVector` + `SVector{N}` for fixed-size state | `Types.jl` |
| `lax.scan(...)` substepping              | Plain `for k in 1:T` loops                | `Bootstrap.jl`, `TemperedSMC.jl` |
| `jax.tree_util.Partial(...)` closure      | `function ... end` with parametric struct | All function-typed fields |
| Bus-aware GPU/CPU split via XLA tracing  | `AbstractArray{T,N}` signatures + parametric struct types | `Filtering/`, `OT.jl` |
| `jax.lax.fori_loop` for bisection        | Plain `for _ in 1:n_bisect_steps`         | `Tempering.jl`  |

LOC reduction is real: the Python framework is 5,619 lines across 36 files;
the Julia framework is ~1,400 LOC across 22 files. Most of the disappearance
is JAX-specific scaffolding (closures, `Partial` wrappers, fp32 staging,
`lax.while_loop` glue) that has no Julia analogue.

## GPU coverage

Charter §15.7 mandates that "tests run twice: once with `CPU()` arrays,
once with `CUDA.CuArray`; both must pass within the same tolerance." The
matrix:

| Module                          | CPU | GPU | Notes |
|:--------------------------------|:---:|:---:|:------|
| `Kernels.compute_ess`           | ✓   | ✓   | logsumexp + scalar; trivially GPU-portable |
| `Kernels.silverman_bandwidth`   | ✓   | ✓   | rewrote scalar-loop fill into vectorised assign |
| `Kernels.log_kernel_matrix`     | ✓   | ✓   | rewrote per-element loop as 3-axis broadcast |
| `Kernels.smooth_resample*`      | ✓   | ✓   | matmul + broadcast; CUBLAS handles the matmul |
| `OT.compute_kernel_factor`      | ✓   | ✓   | same broadcast rewrite as `log_kernel_matrix` |
| `OT.factor_matvec*`             | ✓   | ✓   | matmul; passes through CUBLAS |
| `OT.sinkhorn_scalings`          | ✓   | ✓   | rewrote `ones(T, N)` → `similar(a, T)` so device matches |
| `OT.barycentric_projection`     | ✓   | ✓   | matmul + broadcast |
| `Bootstrap.bootstrap_log_likelihood` (per-particle path) | ✓ | n/a | the v1 fallback for models without batched fns |
| `Bootstrap.bootstrap_log_likelihood` (batched path) | ✓ | ✓ | **Phase 6 follow-up #2 — landed.** Routed via optional `propagate_batch_fn` + `obs_log_weight_batch_fn` on `EstimationModel`. Buffers parameterised on backend (`Array` or `CuArray`); see `BootstrapBuffers{T}(K, n_s; backend=CUDA.CuArray)`. Systematic-resample step transfers a `(K,)` weight vector across PCIe per step (charter §14: scalar bookkeeping crosses the bus, particle states stay GPU-resident). |
| `SMC2/*` outer machinery        | ✓   | n/a | charter §14: outer SMC² runs CPU-resident on purpose |

## Phase 6 follow-ups — landed

Both items the v1 README flagged as deferred have been delivered:

1. **AD-compatible bootstrap PF — done.** The signature of
   `bootstrap_log_likelihood` was relaxed so `u` (the AD-tracked parameter
   vector) and `fixed_init_state` (the user's initial state) no longer
   share an element type. Random noise inside the inner loop is sampled
   as `Float64` and auto-promotes to the AD-tracked `T` at the assignment
   site — random draws have zero gradient w.r.t. `u` by construction.
   `BootstrapBuffers{T}` allocates Dual-typed scratch when called from
   ForwardDiff. The systematic-resample step's integer indices are
   compatible with AD because the gradient flows through the *values*
   in the gather, not through the indices themselves (same trick JAX
   uses for `gather`).

   Verified by `test_phase6_followup_ad.jl` (ForwardDiff gradient runs
   through the PF and matches finite-difference within 8 nats per dim
   — the slack accounts for PF Monte Carlo noise) and
   `test_phase6_followup_e2e_pf.jl` (full SMC² with the PF as the inner
   likelihood, AdvancedHMC.jl + ForwardDiff gradient end-to-end on AR(1)).

2. **Full GPU bootstrap PF end-to-end — done.** `EstimationModel` gained
   two optional fields: `propagate_batch_fn` and `obs_log_weight_batch_fn`.
   When provided, `bootstrap_log_likelihood` calls them on the full
   `(K, n_states)` particle matrix instead of the per-particle loop —
   so the model side runs natively on `CuArray`. The PF bookkeeping
   (cumsum, Liu-West shrinkage, clip, Δlog-w) was also vectorised so
   the same source compiles for both `Array` (CPU) and `CuArray` (GPU)
   buffers. `clip_to_bounds!` has both a scalar-loop method (CPU-fast,
   ForwardDiff-friendly) and a broadcast method (GPU-portable); the
   dispatcher picks the right one. The systematic-resample binary-search
   step transfers a `(K,)` cumsum vector to CPU, computes indices, and
   leaves the heavy gather on the device — a textbook charter §14
   hybrid pattern (scalar bookkeeping crosses PCIe, particle states
   stay GPU-resident).

   Verified by `test_phase6_followup_gpu_pf.jl`: same AR(1) likelihood
   evaluated three ways — CPU per-particle, CPU batched, GPU batched —
   all three agree within PF Monte Carlo σ.

Phase 6 follow-up #2.5 — fused single-kernel GPU PF — landed:

   The framework's `bootstrap_log_likelihood` (CuArray-backed) is
   correct end-to-end but **launch-bound**: each PF step decomposes
   into ~10 small CUDA kernel launches × T steps ≈ thousands of
   launches per call. A demo at K = 1M fp32 reports `nvidia-smi`
   sm % near 0 — the GPU sits idle waiting for the next launch.

   The fix is a single `KernelAbstractions.@kernel` that fuses the
   per-particle PF trajectory into ONE launch (one thread per
   particle, the full T-step loop inside the kernel body). Demo:
   [`version_1_Julia/tools/bench_gpu_bistable_fused.jl`](../../version_1_Julia/tools/bench_gpu_bistable_fused.jl)
   — at K = 1M fp32 / T = 432 / RTX 5090:

   - 2.45e10 particle-steps/sec (sustained); **11,529× faster** than
     CPU at K = 5k Float64.
   - `nvidia-smi dmon` reports sm % = 100 for 10 consecutive 1-s
     samples during a 10 s sustained run.
   - 18 ms cold-JIT → 2.1 ms warm sustained per call.
   - 3.25 GiB VRAM for noise grids (K × (T+1) Float32 × 2).

   Caveat: the fused kernel runs bootstrap PF **without per-step
   resampling** (the trade-off that makes the trajectory
   embarrassingly parallel). Acceptable for T ≤ ~500 + K ≥ 10⁵; for
   longer horizons use the framework's multi-launch path.

   **What's still genuinely out of scope** in this port:
   - **Enzyme reverse-mode AD** as the production backend instead of
     ForwardDiff. Charter §13 names Enzyme as the preferred AD; the
     hooks are in `HMC.jl` (`build_target(...; ad_backend=:Enzyme)`),
     but the Enzyme path needs a per-model audit because Enzyme's
     mutation analysis is stricter than ForwardDiff's.
   - **Full FSA-v5 model** wired against this framework end-to-end.
     The framework is model-agnostic (charter §17), so this lives in
     `models/fsa_high_res/` and is its own port effort.
   - **General fused-kernel framework support**: the bistable demo
     above hand-rolls one kernel per model. A model-agnostic
     `EstimationModel.fused_step_kernel!` field that compiles into
     the framework's generic GPU PF driver is a follow-up.

## How this composes with the Python `smc2fc/`

The Julia package sits **alongside** the Python `smc2fc/` (charter §17:
"the Python `smc2fc/` package continues to exist during and after the
port"). The intended joint pipeline is:

```
LaTeX equations  →  LEAN4 reference (Part I: model side)
                     └── differential test ───┐
                                                ▼
Python smc2fc (live, GPU fast path)  ←─→  Julia SMC2FC (typed reference + GPU fast path)
                                                ▲
                                                └── this package
```

A regression in either Python or Julia is caught by the differential
test against the LEAN4 binary. The two languages keep each other honest;
neither is the single source of truth.
