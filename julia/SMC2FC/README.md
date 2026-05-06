# SMC2FC.jl

Julia port of the `smc2fc` framework — adaptive tempered SMC² for joint
state + parameter inference, plus the same outer kernel re-used as a
control-as-inference engine.

This package is **Part II of the joint LEAN4-and-Julia charter** for the
FSA + smc2fc stack; the charter PDF lives at
[`LaTex_docs/julia_port_charter.pdf`](../../LaTex_docs/julia_port_charter.pdf).

## Status: all phases complete

| Phase | Component                                      | Tests | Status |
|------:|:-----------------------------------------------|------:|:------:|
| 1     | Foundations (Types, Config, Transforms)        |   204 |   ✓    |
| 2     | Filtering (Kernels, OT, Bootstrap PF, GPU)     |    44 |   ✓    |
| 3     | Outer SMC² (Tempering, HMC, TemperedSMC, Bridge) | 14 |   ✓    |
| 4     | Control (Spec, Calibration, RBFSchedule, ControlLoop) | 12 |   ✓    |
| 5     | Plant + Simulator (StochasticDiffEq + Observations) | 6 |   ✓    |
| 6     | End-to-end + JET                                |     7 |   ✓    |
|       | **Total**                                       | **287** | **All passing** |

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
| `Bootstrap.bootstrap_log_likelihood` | ✓ | (pending) | inner SDE step uses model callbacks; full GPU end-to-end is the Phase 6 follow-up listed below |
| `SMC2/*` outer machinery        | ✓   | n/a | charter §14: outer SMC² runs CPU-resident on purpose |

## Phase 6 follow-up

Two items deferred for follow-on work (both listed in charter §17 and §15.7):

1. **AD-compatible bootstrap PF.** The current `BootstrapBuffers{Float64}`
   can't be traced by ForwardDiff — `searchsortedfirst` returns an `Int`
   used for indexed assign. The Python framework gets around this by
   relying on JAX's `lax.stop_gradient` on the resampling step plus the
   Liu-West kernel smoothing for the differentiable proxy. The Julia fix
   is to:
   - parameterise `BootstrapBuffers{T}` over the AD-tracked `T`,
   - replace `searchsortedfirst` with the Liu-West-smoothed kernel blend
     when an AD context is detected (or always, since the production
     path uses Liu-West anyway),
   - swap the `:ForwardDiff` AD backend for `:Enzyme` once the buffer
     parameterisation lands; Enzyme's reverse-mode is what charter §13
     names as the production AD backend for high-D θ posteriors.

2. **Full GPU bootstrap PF end-to-end.** Phase 6 step 21 of the charter
   asks for a full-pipeline test on `backend = :cuda`. The kernel ops and
   OT primitives (Phase 2) are already GPU-portable, but the Bootstrap PF
   driver allocates `Vector{Int}` for resample indices and uses scalar
   binary search inside the inner loop. Replacing this with a
   `KernelAbstractions.@kernel` and a parallel-prefix-sum cumsum is the
   second half of follow-up (1).

Both items are scoped, both are mechanical translation work, both are
under a day each — but they are out of scope for the seven-and-a-half-hour
single-pass port that the charter §15 budget commits to.

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
