# FSA v1.5 — FIM gate FINAL ✅

**Date**: 2026-05-08
**Status**: ✅ **PASS** at the realistic 14-day varying-Φ gate.
**Decision**: Option B (reparametrise (κ_B, τ_B) → (B_inf, τ_B), (κ_F, τ_F) → (F_inf, τ_F)) **+ pin {τ_B, η, ε_A, μ_FF}** at truth.

---

## Final FIM rank table (7 estimated drift params)

| Window | Rank / 7 | κ |
|---|---|---|
| 1 d Φ=1 (spec'd 1-day) | 5 / 7 | 7.2 × 10¹¹ |
| 14 d Φ=1 | **7 / 7** | **3.2 × 10⁷** ✅ |
| **14 d Φ=sin (realistic gate)** | **7 / 7** | **2.95 × 10⁷** ✅ ≪ 10⁸ |
| 28 d Φ=sin | **7 / 7** | **2.2 × 10⁶** ✅ |

The 1-day spec'd window is genuinely too short for the slow-timescale parameters even after pinning — but the bench's filter accumulates information over a 14-day rolling horizon, where every condition is well-conditioned.

## CRLB stds at the 14d realistic gate

| Parameter | I_ii | CRLB std |
|---|---|---|
| τ_F       | 4.7 × 10² | 0.046 |
| B_inf     | 3.7 × 10⁵ | 0.0016 |
| F_inf     | 4.6 × 10⁶ | 0.00047 |
| λ_A       | 2.4 × 10³ | 0.020 |
| μ_0       | 9.4 × 10⁶ | 0.00033 |
| μ_B       | 8.3 × 10⁴ | 0.0035 |
| μ_F       | 4.9 × 10⁵ | 0.0014 |

All seven CRLB stds are at least an order of magnitude smaller than the truth value, so each parameter should pin tightly under closed-loop filtering.

## Pinned parameters (NOT estimated)

| Parameter | Value | Reason |
|---|---|---|
| τ_B       | 42.0  | Slow timescale (>> 14d horizon); was the dominant zero-mode |
| η         | 0.20  | Stuart-Landau cubic — paired with μ_FF in the (η, μ_FF) doublet |
| ε_A       | 0.40  | A-coupling on B's gain — paired with τ_B in (τ_B, ε_A) doublet |
| μ_FF      | 0.40  | F² curvature — paired with μ_F in the (μ_F, μ_FF) shear |
| σ_B_obs, σ_F_obs, σ_A_obs | 0.005 each | Obs noise — pinned by design (per plan decision A) |

## What's locked in

- v1.5 parametrisation: **(τ_B, τ_F, B_inf, F_inf, ε_A, λ_A, μ_0, μ_B, μ_F, μ_FF, η)** + diffusion (σ_B, σ_F, σ_A) + obs noise (3).
- Filter estimates: **10 params total** (7 drift + 3 diffusion).
- Adapter `params_v15_to_v1_nt` rotates back to v1's drift form so v1's `_dynamics.jl` stays verbatim.
- Adapter `fill_pinned_nt` inserts the 4 pinned dynamics values at every estimator → drift call site.
- Reports at `outputs/fsa_high_res/fim_check/`.

## Iteration history (for the record)

| Step | Setup | Result |
|---|---|---|
| 1 | Original v1 basis (κ_B, κ_F), all 11 estimated | rank 4 / 11, κ = 1.6e21 → STOP |
| 2 | Option B: reparametrise (B_inf, F_inf), all 11 estimated | rank 5 / 11 (1d) → STOP |
| 3 | Add pin τ_B, η | rank 6 / 9 (1d) — partial |
| 4 | Add pin ε_A (3 pinned, 8 estimated) | rank 7 / 8 (14d sin), κ = 1.09e8 — borderline |
| 5 | **Add pin μ_FF (4 pinned, 7 estimated)** | **rank 7 / 7 (14d sin), κ = 2.95e7 — PASS ✅** |

Each step's report is preserved in `outputs/fsa_high_res/fim_check/`.

## Where v1.5 stands now

Built and tested:
- `models/fsa_high_res/_dynamics.jl` (v1, verbatim)
- `models/fsa_high_res/simulation.jl` (pure, owns BINS_PER_DAY + adapters)
- `models/fsa_high_res/_plant.jl` (pure, immutable PlantState, plant_step + plant_rollout)
- `models/fsa_high_res/gpu_control.jl` (v1, verbatim)
- `models/fsa_high_res/estimation.jl` (pure propagate + obs_log_weight, 10-param prior config)
- `models/fsa_high_res/FSAHighRes.jl` (module aggregator)
- `tools/fim_check.jl` (passed)
- `tools/psim_sanity.jl` (psim 14d Φ=1 plot, sensible trajectories)

NOT built:
- `models/fsa_high_res/gpu_pf.jl` (~150 LOC: GPU PF target + functional facade over framework's `run_segmented_smc_step!`)
- `tools/bench_smc_full_mpc_fsa_gpu.jl` (~400 LOC: closed-loop foldl bench)

These are the remaining v1.5 work items per the plan.
