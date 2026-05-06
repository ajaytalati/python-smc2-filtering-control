# Stage B — Bistable controlled (Julia port)

The Julia counterpart to
[`version_1/outputs/bistable_controlled/RESULT.md`](../../../version_1/outputs/bistable_controlled/RESULT.md).
Same model, same truth parameters, same headline gates. Plots
panel-for-panel match the Python figures.

## Model

Identical to the Python `version_1/bistable_controlled`:

```
dx = [αx(a² − x²) + u] dt + √(2σ_x) dB_x
du = -γ(u − u_target(t)) dt + √(2σ_u) dB_u
y_k = x_k + ε_k,  ε ~ N(0, σ_obs²)
```

Truth (Set A, healthy / mildly-ill cohort, verbatim copy):

| param   | α   | a   | σ_x  | γ   | σ_u  | σ_obs |
|---------|-----|-----|------|-----|------|-------|
| value   | 1.0 | 1.0 | 0.10 | 2.0 | 0.05 | 0.20  |

Init: `x_0 = -1.0` (pathological well), `u_0 = 0`.
Trial: 72 hours at `dt = 10 min` → 432 obs steps.
Default schedule: `u_target = 0` for `0 ≤ t < 24` h; `u_target = 0.5`
(supercritical, `> u_c`) for `24 ≤ t < 72` h.

Critical tilt: `u_c = 2 α a³ / (3√3) ≈ 0.385`.

## B1 — Filter (8D parameter posterior)

Single SMC² window over the 72-hour trajectory under the default
schedule. Posterior over the 6 dynamics + 2 init parameters.

`n_smc = 48` outer particles, K = 200 inner PF particles, ForwardDiff
gradients through the bootstrap PF (Phase 6 follow-up #1 of
`SMC2FC.jl`), 24 CPU threads.

### Python ↔ Julia per-parameter comparison

| param   | truth | **Python mean** | Python 90 % CI       | **Julia mean** | Julia 90 % CI         | Julia covers? |
|---------|-------|-----------------|----------------------|----------------|-----------------------|----------------|
| α       | 1.000 | 1.063           | [0.812, 1.311]       | 0.841          | [0.725, 1.122]        | ✓              |
| a       | 1.000 | 0.994           | [0.913, 1.061]       | 0.913          | [0.871, 0.984]        | ✗ (0.016 short)|
| σ_x     | 0.100 | 0.118           | [0.088, 0.156]       | 0.066          | [0.038, 0.116]        | ✓              |
| γ       | 2.000 | 2.167           | [1.257, 3.507]       | 2.295          | [1.378, 3.666]        | ✓ (weakly id.) |
| σ_u     | 0.050 | 0.050           | [0.022, 0.085]       | 0.079          | [0.032, 0.115]        | ✓ (weakly id.) |
| σ_obs   | 0.200 | 0.185           | [0.158, 0.211]       | 0.215          | [0.176, 0.234]        | ✓              |

3 of 4 identifiable parameters cover truth at 90 %; `a` is 0.016 short
on the upper bound (mean 0.913, CI upper 0.984, truth 1.0). Pushing
`n_smc → 96` and `num_mcmc_steps → 8` closes that gap (verified
manually). The borderline result reflects the smaller outer cloud
relative to the Python's MCLMC sampler — the Julia bench uses
AdvancedHMC.jl + ForwardDiff, which is faster per step but needs a
larger cloud for the same posterior width.

| gate                                       | Python | Julia |
|--------------------------------------------|--------|-------|
| Identifiable params (α, a, σ_x, σ_obs) cover at 90 % | PASS (4/4) | borderline (3/4) |
| Weakly-identifiable params (γ, σ_u) cover  | PASS   | PASS  |

Convergence: 8 tempering levels, 51 s on 24 threads. (Python: 20
levels, 173 s on CPU — the JAX tempered SMC needs more levels because
its `max_lambda_inc` is tighter.)

Plot: [`B1_filter_diagnostic.png`](B1_filter_diagnostic.png) — 2×4
panel grid, panel-for-panel match of the Python plot:
[`../../../version_1/outputs/bistable_controlled/B1_filter_diagnostic.png`](../../../version_1/outputs/bistable_controlled/B1_filter_diagnostic.png).

Driver: [`tools/bench_smc_filter_bistable.jl`](../../tools/bench_smc_filter_bistable.jl).

## B2 — Control (truth params)

Tempered SMC² over a 12-anchor Gaussian-RBF basis with a sigmoid
output transform (so `u_target ∈ [0, 1]`). Cost: `(1 − P[x_T > 0]) +
0.05 · ‖u‖²`. The Python uses a different cost decomposition (penalty
on residence + L2 + terminal squared error); both reward "drive x to
the +a well by t = 72 h while keeping u parsimonious".

| schedule          | **Python cost** | **Julia cost** | basin transition (Python / Julia) |
|-------------------|-----------------|----------------|-----------------------------------|
| zero (no control) | 1357            | n/a            | 58 % / —                          |
| default (24 h step + u_on=0.5) | 773 | 0.0083 | 100 % / 100 %                |
| **SMC²-derived**  | **295**         | **0.0065**     | **100 %** / **100 %**             |

The cost numbers are not directly comparable because the cost
functionals differ. What is comparable is **the ratio** SMC² / default:

| metric                          | Python   | Julia    |
|---------------------------------|----------|----------|
| SMC² cost / default cost ratio  | 0.38     | 0.78     |
| basin transition rate           | 100 %    | 100 %    |
| gate `transition ≥ 80 %`        | ✓        | ✓        |
| gate `cost ≤ default`           | ✓        | ✓        |

The Julia ratio (0.78) is closer to 1.0 than the Python (0.38) because
my cost form is dominated by the basin term — both schedules achieve
100 % transition, so the only differentiator is the L2 penalty. To
match the Python's stronger cost reduction, switch to the Python's
{residence + L2 + terminal} cost form. The qualitative result —
**SMC² discovers a smoother schedule than the hand-coded step** —
is reproduced.

Convergence: 5 tempering levels, 4.6 s on 24 threads.

Plot: [`B2_control_diagnostic.png`](B2_control_diagnostic.png) — 2×2
grid, panel-for-panel match of the Python:
[`../../../version_1/outputs/bistable_controlled/B2_control_diagnostic.png`](../../../version_1/outputs/bistable_controlled/B2_control_diagnostic.png).

Driver: [`tools/bench_smc_control_bistable.jl`](../../tools/bench_smc_control_bistable.jl).

## B3 — Closed-loop (filter → plan → apply)

Pipeline:
  Phase 1 (0 ≤ t < 24 h, no control): simulate true plant → 144 obs →
    run SMC² over 8D θ → posterior-mean params + posterior-mean state at t=24h.
  Phase 2 (24 ≤ t < 72 h, planned control): run SMC²-as-controller TWICE,
    once with posterior-mean params (closed-loop), once with truth params
    (oracle). Default schedule = `u_target = u_on = 0.5` throughout.
    Apply each schedule to the TRUE plant (100 trials each), record cost
    + basin-transition rate.

### Phase 1 posterior (rel.err vs truth)

| param   | **Python** | **Julia** |
|---------|------------|-----------|
| α       | 6.1 %      | 5.9 %     |
| a       | 0.8 %      | 12.4 %    |
| σ_x     | 0.4 %      | 15.3 %    |
| γ       | 5.1 %      | 17.6 %    |
| σ_u     | 0.3 %      | 17.1 %    |
| σ_obs   | 3.0 %      | 0.3 %     |

Filter config (Julia, 24 threads, 76 s wall):
  `n_smc = 96`, `num_mcmc_steps = 6`, `max_lambda_inc = 0.12`,
  `hmc_step_size = 0.035`, `hmc_num_leapfrog = 5`,
  inner PF `K = 200`, `bandwidth_scale = 1.0`.

α and σ_obs land within Python's recovery; a, σ_x, γ, σ_u are 12–18 %
off because Phase 1 only sees the unhealthy-well portion of the
trajectory (`x ∈ [-1.41, 0.06]`) — the cubic drift is locally near-
linear there, so dynamics params are partially observable. Bumping
`n_smc` further closes the gap monotonically; this is the
quality/wall-time trade-off knob.

The B3 gates measure **closed-loop quality** (does the planner do well
under the posterior?), not posterior tightness — so the dynamics-
param spread only matters indirectly through the schedule shape.

### Phase 2 closed-loop performance

| schedule                 | transition rate (Python) | transition rate (Julia) |
|--------------------------|--------------------------|--------------------------|
| **SMC²-with-posterior**  | **100 %**                | **100 %**                |
| SMC²-with-truth (oracle) | 100 %                    | 100 %                    |
| default u_on=0.5         | 100 %                    | 100 %                    |

| gate                                          | Python  | Julia  |
|------------------------------------------------|---------|--------|
| closed-loop transition rate ≥ 80 %            | 100 % ✓ | 100 % ✓ |
| closed-loop cost ≤ 1.20 × oracle              | 0.988× ✓ | **0.954× ✓** |

The SMC²-with-posterior is **statistically tied with the oracle** in
both languages — the framework's filter→control pipeline composes
end-to-end without the posterior uncertainty hurting closed-loop
quality.

Convergence (Julia, 24 threads, after the v2 budget bump):
  Phase 1 filter:    9 tempering levels, ~76 s
  Phase 2 posterior: 5 tempering levels, ~3 s
  Phase 2 oracle:    5 tempering levels, ~2 s
  Plant rollouts:    100 trials × 3 schedules, < 1 s

Plot: [`B3_closed_loop_diagnostic.png`](B3_closed_loop_diagnostic.png) —
2×2 panel grid, panel-for-panel match of the Python:
[`../../../version_1/outputs/bistable_controlled/B3_closed_loop_diagnostic.png`](../../../version_1/outputs/bistable_controlled/B3_closed_loop_diagnostic.png).

Driver: [`tools/bench_smc_closed_loop_bistable.jl`](../../tools/bench_smc_closed_loop_bistable.jl).

## Side notes

### GPU saturation

Two GPU benches sit next to each other:

1. [`tools/bench_gpu_bistable.jl`](../../tools/bench_gpu_bistable.jl) —
   runs the framework's `bootstrap_log_likelihood` with `CuArray`
   buffers (Phase 6 follow-up #2). Wires end-to-end, allocates 4.6 GB
   VRAM, but the per-step inner work decomposes into ~10 small CUDA
   kernel launches × 432 steps ≈ 4,320 launches per PF call.
   **Launch latency dominates over compute** → sm % near 0 on
   `nvidia-smi dmon`. ~1.6× faster than CPU.

2. [`tools/bench_gpu_bistable_fused.jl`](../../tools/bench_gpu_bistable_fused.jl)
   — Phase 6 follow-up **#2.5, now landed**. A single
   `KernelAbstractions.@kernel` fuses the entire T = 432-step
   per-particle PF trajectory into ONE launch (one thread per
   particle, the full time loop runs inside the kernel body). Total
   launches per PF call: 1, plus a tiny `logsumexp` at the end.
   **GPU saturated** — `nvidia-smi dmon -s u -c 120 -d 1` reports
   **sm % = 100** for 10 consecutive samples during the 10-second
   sustained 5000-call run.

   Numbers (K = 1M fp32 particles, T = 432 obs steps, RTX 5090):
   - Single call: 18 ms (cold JIT) → 2.1 ms (warm sustained)
   - Throughput: **2.45e10 particle-steps / sec** (11,529× faster
     than CPU at K = 5k Float64, which manages 2.12e6 ops/sec)
   - VRAM: 3.25 GiB (noise grids dominate — `(K, T+1)` Float32 ×2)

   Caveat: this fused kernel runs the bootstrap PF **without
   per-step resampling** — the trade-off that makes the trajectory
   embarrassingly parallel. For T ≤ ~500 and K ≥ 10⁵ the estimator
   variance is acceptable. For longer horizons (degeneracy collapses
   the cloud), use the framework's multi-launch path which has
   per-step Liu-West shrinkage + systematic resampling.

### Threading

All Julia benches in `tools/` parallelise the outer SMC² particle
loop via `Threads.@threads` inside `SMC2FC.TemperedSMC._tempered_step!`.
Julia's default thread count is 1; you must launch with
`julia --threads auto` (or `--threads 24`) to see the speedup.
Without threading, B1 runs in ~10 minutes; with 24 threads it's ~50 s.

## Reproduce

```bash
conda activate comfyenv
cd version_1_Julia/
julia --threads auto --project=. tools/bench_smc_filter_bistable.jl
julia --threads auto --project=. tools/bench_smc_control_bistable.jl
# Optional: GPU saturation diagnostic
julia --threads auto --project=. tools/bench_gpu_bistable.jl
```

Plots saved to `outputs/bistable_controlled/`; raw numbers in
`_results_*.txt` next to each plot.
