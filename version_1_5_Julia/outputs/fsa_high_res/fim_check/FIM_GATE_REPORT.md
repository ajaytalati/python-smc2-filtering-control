# FSA v1.5 — FIM gate report (HALTED)

**Date**: 2026-05-08
**Status**: 🛑 **STOP** — identifiability issue flagged at TRUTH_PARAMS.
**Action**: Do not proceed to closed-loop bench. Surface findings to user; await re-parametrisation guidance per the v1.5 plan.

---

## Headline

The deterministic-trajectory FIM at TRUTH_PARAMS is **rank-deficient** for the v1.5 parametrisation under every window I tested:

| Window | Φ schedule | Rank / 11 | κ (cond. number) |
|---|---|---|---|
| 1 day, 24 bins | constant Φ=1 | **4 / 11** | 1.6 × 10²¹ |
| 3 days, 72 bins | constant Φ=1 | 5 / 11 | 8.0 × 10¹⁷ |
| 7 days, 168 bins | constant Φ=1 | 5 / 11 | 1.3 × 10¹⁵ |
| 14 days, 336 bins | constant Φ=1 | 6 / 11 | 3.6 × 10¹² |
| 28 days, 672 bins | constant Φ=1 | 7 / 11 | 1.7 × 10¹¹ |
| 14 days | sin Φ ∈ [0.2, 1.5] | 6 / 11 | 1.4 × 10¹¹ |
| 28 days | sin Φ ∈ [0.2, 1.5] | **8 / 11** | 3.0 × 10¹⁰ |

Even at the most informative setting tested (28 days, varying Φ), three of the eleven drift parameters remain effectively un-identifiable. The plan's GATE was rank = 11 and κ ≤ 10⁸; both are missed.

---

## Spec'd 1-day FIM (the actual gate result)

Configuration:
- Step: 60 min ⇒ BINS_PER_DAY = 24
- Window: 1 day = 24 bins, Φ_t = 1 constant
- Init state: (B = 0.05, F = 0.30, A = 0.10)
- Obs noise (pinned): σ_B_obs = σ_F_obs = σ_A_obs = 0.005

**Eigenvalue spectrum** (descending):

| # | λ | comment |
|---|---|---|
| 1 | 3.62 × 10⁵ | dominant — informed by κ_B |
| 2 | (~ 1 × 10⁵) | informed by κ_F |
| 3 | 3.57 × 10³ | informed by μ_0 |
| 4 | 1.44 × 10⁻² | borderline — first weak mode |
| 5 | 8.55 × 10⁻⁴ | weak |
| 6 | 3.82 × 10⁻⁶ | very weak |
| 7 | 4.30 × 10⁻⁷ | very weak |
| 8 | 8.61 × 10⁻⁹ | numerically zero |
| 9 | 6.95 × 10⁻¹¹ | numerically zero |
| 10 | −1.87 × 10⁻¹³ | numerically zero |
| 11 | −2.74 × 10⁻¹² | numerically zero |

Numerical rank (tol = λ_max × 10⁻⁸ = 3.62 × 10⁻³) = **4 / 11**.

**Per-parameter information (FIM diagonal) and CRLB std at 1-day window:**

| Parameter | I_ii | CRLB std | identified? |
|---|---|---|---|
| κ_B       | 3.62e+05 | 0.0017 | ✅ very strong |
| κ_F       | 3.04e+05 | 0.0018 | ✅ very strong |
| μ_0       | 3.25e+03 | 0.018  | ✅ strong |
| μ_F       | 2.81e+02 | 0.060  | ✅ strong |
| μ_FF      | 2.43e+01 | 0.20   | ⚠ moderate |
| τ_F       | 1.32e+01 | 0.27   | ⚠ moderate |
| μ_B       | 9.51e+00 | 0.32   | ⚠ moderate |
| λ_A       | 5.25e+00 | 0.44   | ⚠ moderate |
| ε_A       | 4.71e-01 | 1.46   | ❌ weak |
| η         | 3.11e-01 | 1.79   | ❌ weak |
| τ_B       | 3.14e-04 | 56.4   | ❌ very weak |

Note: the diagonal info is misleading on its own (it ignores correlations). The eigendecomposition above is what determines actual identifiability, and only 4 directions are well-resolved.

---

## Persistent degeneracies (28-day varying-Φ window)

Even the most-informative window I tested still has three near-zero eigenmodes. The dominant loading on each:

**Mode 1** — λ ≈ 0.18, weakest by 100×:
- **τ_B** (loading +0.98) — the chronic-B time constant is 42 days; even a 28-day window sees only ~50 % of an exponential relaxation. Cannot be pinned.

**Mode 2** — λ ≈ 0.59:
- **μ_FF** (loading −0.90), with μ_F (+0.38) and η (+0.19) admixture. Stuart-Landau curvature on F is poorly resolved because F barely leaves [0.2, 0.4] under the FSA dynamics at TRUTH_PARAMS — its quadratic-in-F coupling cannot be teased apart from the linear-in-F coupling.

**Mode 3** — λ ≈ 22:
- **ε_A** (loading −0.98). The B-coupling factor `1 + ε_A · A` produces only ~4 % modulation of κ_B's contribution at A ≈ 0.10. Tiny effect → weak signal.

**Physical interpretation:** these aren't pathological — they reflect real properties of the experimental setup:

- **τ_B is too slow** for any realistic bench horizon (42 d).
- **μ_FF needs F-variation** that doesn't happen at typical operating points.
- **ε_A needs A-variation** that's also limited under standard training.

---

## What this means for v1.5

Per the plan's gate logic and the user's instruction:

> "if this flags identifiability issues STOP and report to me me and we will reparametrize"

I have **HALTED**. The closed-loop bench (`gpu_pf.jl`, `bench_smc_full_mpc_fsa_gpu.jl`) is **not** built. The v1.5 model files that ARE built so far:

```
version_1_5_Julia/
├── Project.toml                                # done (no SMC2FC yet)
├── models/fsa_high_res/
│   ├── _dynamics.jl                            # done — copy from v1
│   ├── simulation.jl                           # done — pure
│   ├── _plant.jl                               # done — pure functional
│   ├── gpu_control.jl                          # done — copy from v1
│   ├── estimation.jl                           # done — pure
│   └── FSAHighRes.jl                           # done — aggregator
└── tools/
    └── fim_check.jl                            # done — produced this report
```

The plant + obs + estimation (filter primitives) are all written and ready — the FIM check exercises them end-to-end and they work. The missing pieces are `gpu_pf.jl` (GPU PF target) and the closed-loop bench. I've stopped before those.

---

## Re-parametrisation options for the user to choose between

Each option is independent; the user can pick one or combine.

### A) Pin the three weakest params at truth (drop them from the filter)

**Pinned params:** τ_B, μ_FF, ε_A → fixed at truth values.
**Filter estimates:** the remaining 8 + 3 σ_obs (already pinned) = 8 dynamics params.

**Pros:** smallest change; no reformulation. `PARAM_PRIOR_CONFIG` shrinks from 14 to 11 entries (8 drift + 3 diffusion). The filter posterior will be tight on the 8 it estimates; the 3 pinned ones never move.

**Cons:** if τ_B / μ_FF / ε_A vary in real subjects, we'd be ignoring real heterogeneity. For a synthetic-only diagnostic harness this is fine.

This is the most pragmatic choice and matches what v2 effectively does for σ_obs.

### B) Reparametrise (κ_B, τ_B) → effective gain at steady state

`κ_B · τ_B` is the steady-state value of B at Φ = 1 (assuming A doesn't move much). Define a new parameter `B∞ = κ_B · τ_B` and let the filter estimate `(B∞, τ_B)` instead of `(κ_B, τ_B)`. The filter has lots of info on B∞ (it's the long-run mean of B), and τ_B remains weakly identified but with a more interpretable companion parameter.

Same trick for (κ_F, τ_F) → `F∞ = κ_F · τ_F`.

**Pros:** keeps all 11 params in the filter; just rotates the parameter basis.
**Cons:** requires rewriting v1's `_dynamics.jl` (which we wanted to keep verbatim). Breaks the "v1.5 == v1 dynamics" promise.

### C) Reparametrise (μ_F, μ_FF) into Horner / centred form

The Stuart-Landau µ is `μ_0 + μ_B·B − μ_F·F − μ_FF·F²`. Rewrite as `µ_0' + µ_B·B − a·(F − F_typ)² − b·(F − F_typ)` (around F_typ = 0.20, the typical operating point). Then `a = μ_FF` and `b = μ_F − 2·μ_FF·F_typ`. This is exactly v2's G1 reparametrisation idea — and the writeup §2.7 says v2 went to G1 specifically for filter identifiability.

**Pros:** addresses the (μ_FF, μ_F) degeneracy directly. Makes the Stuart-Landau coefficients orthogonal at typical F values.
**Cons:** brings v1.5 closer to v2's parametrisation, defeating part of the v1.5 / v2 contrast.

### D) Drop ε_A entirely

The ε_A · A term modulates B's response by ~4 % at A=0.10. Setting ε_A = 0 (no autonomic modulation of B's gain) might be acceptable for the synthetic diagnostic — it's a small-effect parameter that's nuisance-rather-than-signal in v1.5.

**Pros:** simplest. One fewer parameter, no reformulation.
**Cons:** removes one of the FSA-specific autonomic-coupling effects.

---

## My recommendation (junior-engineer hedge)

Option A (pin τ_B, μ_FF, ε_A at truth) is the smallest-change path that keeps v1.5 a "minimum-viable closed-loop bridge" instead of a "mini reparametrisation project". It does cost ε_A, which is a real FSA effect — but for v1.5's stated purpose ("clean diagnostic harness for the closed-loop pipeline machinery, NOT a production model"), pinning seems acceptable.

If the user wants to keep ε_A as estimable, Option C (G1-style centring around F_typ on (μ_F, μ_FF)) plus pinning τ_B alone could get to rank 9 / 11 — and combining that with Option D's drop of ε_A gets to rank 9 / 10 at typical operating points.

I am **not** going to pick on my own. Surfacing to you per the plan.

---

## Files

- This report: `outputs/fsa_high_res/fim_check/FIM_GATE_REPORT.md`
- The FIM tool: `tools/fim_check.jl`
- The plant / sim / est / dynamics + gpu_control modules are all in place under `models/fsa_high_res/`
- v1.5's closed-loop bench is **not** written — awaiting your re-parametrisation choice.
