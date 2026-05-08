# FINDINGS — v1.5 closed-loop three-way comparison (T=7d, seed=42)

> Pre-flight run for the planned T=28d headline experiment. All
> artefacts in this directory are reproducible from `tools/launchers/
> run_v15_T28d_compare.sh 7 42` (`version_1_5_Python_JAX/`).

## Setup

| | Julia stack | Python+JAX stack |
|---|---|---|
| Bench | `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` | `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py` |
| Filter outer SMC² | N=32, K_pf=200 | N=32, K_pf=200 |
| Filter HMC | num_mcmc=3, ε=0.05, L=4 | num_mcmc=3, ε=0.05, L=4 |
| Controller outer SMC² | N=256, n_inner=64 | N=256, n_inner=64 (matched) |
| Controller HMC | num_mcmc=8, ε=0.2, L=16 | num_mcmc=8, ε=0.2, L=16 |
| T_days | 7 | 7 |
| step | 60 min (BINS_PER_DAY=24) | 60 min (BINS_PER_DAY=24) |
| seed | 42 | 42 |
| Hardware | RTX 5090, 575 W TDP, 32 GB | same |

Both stacks now use the JAX-native compile-once SMC kernel path
respectively (Python: `make_gk_dpf_v3_lite_log_density_compileonce` +
`run_smc_window_native` + `run_tempered_smc_loop_native`; Julia:
`run_outer_smc` + `controller_plan` over `FSAv1ControlGPUTarget`).
Configurations are matched 1:1 after Phase-A CLI parity work.

## Three-way diagnostic table

```
                                    Baseline (Φ=1.0)    Python+JAX MPC         Julia MPC
  ────────────────────────────────────────────────────────────────────────────
  mean A (last 7 days)                        0.0935            0.0803            0.0815
    improvement vs baseline                        —            -14.1%            -12.8%
  F-violation rate (frac)                     0.0000            0.0000            0.0000
  posterior MSE to truth (final)                   —            0.5502            0.0868
  ────────────────────────────────────────────────────────────────────────────
  ──────────────────── macro (during the closed-loop run) ────────────────────
  total wall time (min)              ~0 (plant-only)               4.4              37.2
  mean GPU utilisation (%)                         —              23.2              24.8
  peak GPU memory (GB)                             —              7.05             12.64
  mean GPU power (W)                               —                85               100
  ────────────────────────────────────────────────────────────────────────────
  ─────────────── micro (profile_gpu, isolated kernel timing) ────────────────
  filter kernel time / call (ms)                   —               4.7              81.0
    effective TFLOPS                               —              0.01              0.00
    util vs 5090 peak (%)                          —               0.0               0.0
  controller cost / call (ms)                      —              23.4               0.5
    effective TFLOPS                               —              0.05              2.61
    util vs 5090 peak (%)                          —               0.0               2.5
```

Plot: [`comparison.png`](comparison.png) (6-panel; A trajectory, applied
Φ, two posterior medians, GPU SM utilisation, per-stride wall time).

## Verdict

**Headline:** the simple H1 vs H2 framing from the plan does not
match the observed pattern. Three findings emerge instead:

### Finding 1 — H1 is NOT supported by filter behaviour

Julia's posterior MSE on the 10 estimated parameters is **0.087 vs
Python's 0.550** (6× better). The Julia filter is genuinely doing a
better job inferring the truth than the Python filter at this
horizon and seed. Whatever's wrong, it isn't a filter-orchestration
bug: the posterior over (params, state) Julia hands to the controller
is closer to truth than Python's.

### Finding 2 — H2 has a twist: same GPU util, very different per-call cost

Both stacks show **~25% mean GPU utilisation** for the full
closed-loop run (Python 23.2%, Julia 24.8%). Neither is saturating.

But Python is 8.5× faster wall-time (4.4 min vs 37.2 min). The
microbench shows where the cost lives:

- **Filter kernel time / call: Python 4.7 ms vs Julia 81.0 ms** — Julia
  is **17× slower per call** on the filter PF kernel.
- Controller cost / call: Python 23.4 ms vs Julia 0.5 ms — Python is
  47× slower on the controller. (Consistent with Julia's
  KernelAbstractions controller running fp32 vs JAX's fp64 — fp32 is
  ~64× faster on the 5090.)

The bench is filter-dominated (filter runs every stride, controller
only every K=2 strides), so Julia's slow filter dominates wall time.

The 17× per-call gap is **CPU-side overhead, not GPU compute**. The
GPU utilisation pattern in `comparison.png` panel (2,0) makes this
obvious: Python shows tall narrow bursts to ~80-100% with idle gaps
(typical compile-once-then-fire pattern); Julia shows a continuous
~25-30% level (typical "spending most of the time in CPU launches /
host syncs / memory marshalling"). Same average util, completely
different shape.

### Finding 3 — Both controllers fail the baseline gate (NEW: H3)

**Neither MPC beats the constant Φ=1.0 baseline.** Both lose ~13% on
mean A. The plan's decision-rule for "neither MPC beats baseline" was:
"unexpected; either both have the same orchestration bug or the model
itself is mis-tuned for this scenario."

The Julia per-replan log shows the controller picks Φ̄≈0.28
(rest-heavy) at strides 4, 6, 8, 10 and only ramps to Φ̄≈1.66 at the
final replan (stride 12). The cost function (`-∫A + λ_F · F-barrier`,
no Φ² penalty) plus the v1 Stuart-Landau dynamics admit a "rest cures
all" optimum that the controller is finding. This is a **cost-function
calibration issue** — call it **H3** — not orchestration or GPU.

The original "rest cures all" pathology of v1 was supposed to be
closed by v2's coupling B to Φ explicitly. v1.5 keeps v1's drift but
adds the F_max penalty. In a 7-day horizon the controller may
genuinely prefer to rest first, then sprint at the end — which is what
both stacks are doing. Whether T=28d gives the controller enough
horizon to find a sustainable Banister periodisation is the open
question the headline run will answer.

## Recommendations

In priority order:

1. **Investigate Julia's per-call filter overhead (H2).** The 17×
   gap vs JAX is the easiest performance win. Worth profiling
   specifically what the segmented PF kernel is doing on each call —
   suspect either (a) Liu-West / OT resample sync that round-trips to
   CPU per particle, or (b) θ-marshalling overhead on each
   `gpu_log_density` invocation. `nsys profile` would show this
   directly.

2. **Re-examine the controller cost function (H3).** The current
   cost lets the controller pick Φ̄≈0.28 in early replans. Either:
   (a) the planning horizon at each replan is too short to see the
   "build up B then sprint" reward, or (b) the cost needs an explicit
   "minimum mean-A over the planning window" term, or (c) the
   `lam_barrier=1.0` on F is too weak to prevent the rest pathology.
   Confirm with a T=28d run before changing anything.

3. **The 25% GPU util ceiling is a ceiling for both stacks.**
   Neither bench is saturating the 5090 at the matched config.
   Bumping particle counts (per CLAUDE.md, v2 saturates at N=256/K=400)
   would help both — but won't change the qualitative result if H3
   (controller cost) is the binding constraint.

## What's NOT relevant

- The "MANDATORY native path" warning in CLAUDE.md is correctly
  followed by the Python bench: it uses
  `make_gk_dpf_v3_lite_log_density_compileonce`,
  `run_smc_window_native`, `run_smc_window_bridge_native`, and
  `run_tempered_smc_loop_native` already. The 25% Python GPU util is
  not from the BlackJAX-recompile footgun; it's from the bench being
  small enough that XLA-compile + per-stride overhead dominates the
  GPU work.
- Lean4 is out of scope for closed-loop comparison (no SMC², no GPU);
  Lean4's role as the model-math oracle is already discharged via the
  triangulated 121-case 1e-6 diff test.

## Followup

A T=28d run with the same seed + matched config is in progress to
confirm H3 holds at longer horizon. Updates to this file when that run
completes.
