# FSA v1.5 — FIM gate report (after option B reparametrisation)

**Date**: 2026-05-08
**Status**: ⚠ **STILL HALTED** — option B improves the situation but does not reach the gate threshold (rank=11, κ≤10⁸).
**Action**: Surface to user; await guidance on whether to (a) accept rank-deficient FIM and proceed, (b) combine B with another option, or (c) try a different reparametrisation.

---

## What changed

Per user instruction: applied option B from the prior FIM gate report — replaced `(κ_B, τ_B)` with `(B_inf = κ_B · τ_B, τ_B)` and similarly `(κ_F, τ_F)` with `(F_inf, τ_F)`. v1's `_dynamics.jl` is still verbatim; the basis rotation is implemented as an adapter `params_v15_to_v1_nt` in `Simulation`, called at every site that touches `drift()` (plant, propagate, FIM check). v1's `gpu_control.jl` is also unchanged — when the bench wires it, it calls the adapter to convert posterior-mean params into v1 form before constructing the controller target.

Files modified:

- `models/fsa_high_res/simulation.jl` — `DEFAULT_PARAMS` now contains `:B_inf, :F_inf` instead of `:kappa_B, :kappa_F`. New `params_v15_to_v1_nt` adapter.
- `models/fsa_high_res/_plant.jl` — `plant_step` calls the adapter.
- `models/fsa_high_res/estimation.jl` — `PARAM_NAMES` and `PARAM_PRIOR_CONFIG` updated to v1.5 basis.
- `tools/fim_check.jl` — `DRIFT_PARAM_NAMES` and `rollout_det` updated.

---

## Headline comparison

### 1-day Φ=1 window (the spec'd gate)

| Basis | Rank / 11 | κ | λ_max | λ_min |
|---|---|---|---|---|
| v1 (κ_B, κ_F)        | 4 / 11 | 1.6 × 10²¹ | 3.62 × 10⁵ | -2.74 × 10⁻¹² |
| v1.5 (B_inf, F_inf) ← new | **5 / 11** | 1.07 × 10¹⁸ | 6.22 × 10³ | 5.80 × 10⁻¹⁵ |

Marginal improvement at 1 day. The dominant eigenvalues SHRANK in the new basis — that's expected, because the κ_B-dominant mode in the old basis had info ≈ τ_B² × info-on-B_inf; reparametrising redistributes that factor, so the largest eigenvalue is smaller but the smallest (in well-identified directions) is similar. Total Frobenius norm of FIM is approximately preserved (it's a basis change after all).

### Window sweep, v1.5 basis

| n_days | Φ | Rank / 11 | κ |
|---|---|---|---|
| 1 | const | 5 | 1.07 × 10¹⁸ |
| 1 | sin Φ ∈ [0.2, 1.5] | 6 | 2.59 × 10¹³ |
| 7 | const | 6 | 1.55 × 10¹³ |
| 7 | sin | 6 | 1.56 × 10¹⁰ |
| 14 | const | **8** | 4.90 × 10¹⁰ |
| 14 | sin (~bench info horizon) | **8** | **2.18 × 10⁹** |
| 28 | const | 9 | 5.15 × 10⁹ |
| 28 | sin | **9** | **1.05 × 10⁹** |

Best result tested: 28 days varying-Φ, **rank 9 / 11**, κ ≈ 10⁹. Still 2 eigenmodes near zero. The bench's actual 14-day horizon gets to **rank 8 / 11**, κ ≈ 2 × 10⁹ — both fail the original gate.

### Improvement vs option A (no reparametrisation)

| Window | v1 basis rank | v1.5 basis rank | Δ |
|---|---|---|---|
| 1 d const | 4 | 5 | +1 |
| 14 d const | 6 | 8 | +2 |
| 28 d const | 7 | 9 | +2 |
| 28 d sin | 8 | 9 | +1 |

Modest but consistent improvement. Two more directions become identifiable at the cost of the same orthogonal rotation.

---

## The persistent weak modes (14 d sin-Φ, v1.5 basis)

Bottom 3 eigenvalues + dominant loadings:

**Mode 1 (weakest)** — λ ≈ 4.8 × 10⁻³
- ε_A loading: −0.74
- τ_B loading: +0.67
- (others small)
- **Interpretation**: trade-off between the autonomic-coupling factor and the B-relaxation timescale. They both modulate B's dynamics in similar ways at typical operating points.

**Mode 2** — λ ≈ 7.6 × 10⁻³
- η loading: −0.72
- μ_FF loading: +0.62
- μ_F loading: −0.29
- **Interpretation**: trade-off in the Stuart-Landau bifurcation: the cubic damping η versus the F² curvature in µ. Both shape A's amplitude–saturation relationship.

**Mode 3** — λ ≈ 8.8 × 10⁻²
- τ_B loading: +0.74
- ε_A loading: +0.67
- (orthogonal direction in the (τ_B, ε_A) plane)
- **Interpretation**: same 2-param subspace as mode 1, the orthogonal direction within it. The pair (τ_B, ε_A) is anisotropically informed — strong along one direction, weak along the other.

So the genuine residual identifiability problems live in **TWO 2-parameter subspaces**:

1. **(τ_B, ε_A)** — chronic-B timescale × autonomic coupling on B's gain
2. **(η, μ_FF)** — Stuart-Landau cubic × F² bifurcation curvature

Both are physically interpretable: the data is informative about the *combined* effect of each pair on B and A dynamics, but cannot separately tease them apart at TRUTH_PARAMS over a 14-day window.

---

## Decisions for the user

The plan's gate failed again. Per the same plan I should not push past it autonomously. The options are:

### B + accept (override the gate)

Proceed to `gpu_pf.jl` + closed-loop bench at rank 8/11, κ ≈ 2 × 10⁹.

- The closed-loop controller depends most directly on the well-identified parameters (B_inf, F_inf, μ_0, μ_F) for its cost surface — those have CRLB stds 0.013–0.060.
- The filter posterior will be wide on (τ_B, ε_A) and (η, μ_FF), but those directions are mostly orthogonal to the controller's optimum.
- Empirical question whether closed-loop accuracy degrades.

### B + pin τ_B and η (option A on top)

Pin the two leading parameters in the worst eigenmodes at truth. The matrix shrinks 11→9; preliminary check suggests this would give full rank 9.

- Cleanest gate-passing solution.
- Loses ability to estimate τ_B and η from data — but they're already not really estimable at v1.5's bench horizon.

### B + C (add G1 centring on µ)

Replace `µ = µ_0 + µ_B·B − µ_F·F − µ_FF·F²` with the centred form `µ = µ_0' + µ_B·B − a·(F − F_typ)² − b·(F − F_typ)` (around F_typ = 0.20). v2 went to G1 specifically for this reason. Addresses the (η, μ_FF) collapse but introduces v2's drift form (and brings back the §2.7 fp32 cancellation question, which we wanted to avoid).

- Best chance of full rank 11.
- Erodes the "v1.5 is structurally simpler than v2" promise.

### Different reparametrisation entirely

Some other choice not in the original menu — e.g. drop ε_A entirely AND drop η entirely (option D from the prior report applied to both weak doublets). Reduces to 9 params with full rank likely.

---

## My recommendation (junior-engineer hedge, again)

Pure option B alone doesn't pass the gate. The cheapest path to a clean PASS is **B + pin τ_B and η** (rank 9 / 9 likely, no further reformulation needed). For the v1.5 stated purpose ("clean diagnostic harness, NOT production model"), pinning two slow / unobservable parameters at truth is consistent with the spirit of the original plan.

If you want to keep all 11 params estimable, **B alone with rank 8 override** is the next-cheapest path — but the bench's ChEES-HMC kernel may have mixing issues on the 3 weak directions.

I am not picking. Awaiting your call.

---

## Files

- This report: `outputs/fsa_high_res/fim_check/FIM_GATE_REPORT_optB.md`
- Prior report: `outputs/fsa_high_res/fim_check/FIM_GATE_REPORT.md`
- Updated tool: `tools/fim_check.jl`
- Adapter: `models/fsa_high_res/simulation.jl :: params_v15_to_v1_nt`
