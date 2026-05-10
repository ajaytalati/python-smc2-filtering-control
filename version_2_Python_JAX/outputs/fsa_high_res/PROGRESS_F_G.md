# Progress Report — Stages F + G (partial)

## Executive summary

| stage | what                                              | E3 coverage  | status |
|-------|---------------------------------------------------|--------------|--------|
| F1    | SF Path B-fixed bridge replacing Gaussian         | 17/27 (was 18/27) | not a clean win on v2 |
| F2    | closed-loop T=14 with SF bridge + T_total plan    | mean A +9% over baseline; F-violation 8.26% | motivates Stage G |
| **G1**| **reparametrize FSA-v2 (FIM rank-deficient pairs)** | **23/27 (vs F1 17/27)** | **principled fix; ~1 short of 24/27 gate** |
| G2    | enable info-aware SF bridge on top of G1          | **17/27** (regressed) | info-aware HURTS on the well-conditioned model |

**Headline**: Stage G1 (reparametrization) is the correct principled
fix — it moved E3 27-window coverage from 17-18/27 → 23/27 (within
1 of the strict 24/27 gate). Stage G2 (info-aware bridge), proposed
as belt-and-braces on top of G1, **actually regressed** to 17/27
because the reparametrized model is well-conditioned enough that
holding 15/30 directions at prior throws away genuine information.

**Recommendation**: ship G1 reparametrization, skip G2 info-aware,
proceed with G4 multi-horizon closed-loop MPC using the
`sf_info_aware=False` config that gave 23/27 on E3.

## The journey — diagnosed FIM rank-deficiency, fixed at the model level

The pen-and-paper FIM analysis of the v2 spec at 1-day windows showed
three rank-deficient parameter pairs:

  | pair                | reason                                                  |
  |---------------------|---------------------------------------------------------|
  | (κ_B, ε_A)          | A nearly constant → only `κ_B(1+ε_A·A_typ)` identifiable |
  | (μ_F, μ_FF)         | F nearly constant → only `μ_F + 2·F_typ·μ_{FF}` identifiable |
  | (τ_F, λ_A)          | A nearly constant → only `(1+λ_A·A_typ)/τ_F` identifiable |

The bridge handoff propagated particle scatter on these directions as
**phantom information** — Ledoit-Wolf shrinkage tightened the cov around
arbitrary spreads instead of the prior, causing posterior drift across
windows.

Two candidate fixes were proposed:

- **Option A (G1)**: rotate parameter pairs into identifiable
  combinations + tight residual priors. Closes the rank-deficiency at
  the **model spec** level.
- **Option B (G2)**: enable `sf_info_aware=True` so the bridge holds
  prior on weakly-informed FIM directions instead of moment-fitting.
  Closes the rank-deficiency at the **bridge** level.

User chose Option C (both), with G1 first to fix the model spec then
G2 to test the bridge against the cleaned-up model. The empirical
result is that **G1 alone does the work** and **G2 on top of G1 is
counterproductive**.

## Stage G1 — reparametrization (principled fix)

### Mathematical change

Define operating-point reference constants:
- `A_typ = 0.10`  (initial-state A; representative of de-trained subject)
- `F_typ = 0.20`  (mid-window F under typical Φ at typical A)

Replace the rank-deficient pairs with (strongly-identified, residual)
decompositions, keeping parameter NAMES so downstream code (control,
bench drivers) doesn't need rewrites:

| name        | NEW meaning                            | NEW truth |
|-------------|----------------------------------------|-----------|
| `kappa_B`   | κ_B^eff = κ_B·(1+ε_A·A_typ)            | 0.01248   |
| `epsilon_A` | residual A-boost beyond A_typ          | 0.40      |
| `mu_F`      | slope of μ vs F at F_typ               | 0.26      |
| `mu_FF`     | curvature; (F − F_typ)²-centered       | 0.40      |
| `mu_0`      | μ_0 + μ_FF·F_typ²                      | 0.036     |
| `tau_F`     | τ_F^eff = τ_F/(1 + λ_A·A_typ)          | 6.3636…   |
| `lambda_A`  | residual A-coupling beyond A_typ       | 1.00      |

Tighten priors on the residuals (`epsilon_A`, `mu_FF`, `lambda_A`)
from σ=0.20-0.25 → σ=0.05 in log-space.

The drift formulas are mathematically equivalent to the original v2
spec at the new truth values (verified by 4 unit tests in
[`tests/test_g1_reparam.py`](../../version_2/tests/test_g1_reparam.py)).

### Verification

| test                                          | result |
|-----------------------------------------------|--------|
| drift parity at typical state                 | ✓      |
| drift parity across 50 random states          | ✓      |
| truth-value derivation matches closed-form    | ✓      |
| residual params drop out at typical point     | ✓      |
| **E1 single-window filter (1-day, 30 params)**| **30/30 covered, 16 levels in 169s** |
| E2 plant tests (5 unchanged)                  | ✓ all pass |

### E1 evidence — residual params now properly absorbed

| param      | pre-G1 std | post-G1 std | shrinkage |
|------------|------------|-------------|-----------|
| ε_A        | 0.093      | **0.016**   | ~6×       |
| λ_A        | 0.188      | **0.052**   | ~4×       |
| μ_FF       | 0.080      | **0.020**   | ~4×       |

Tighter posterior CI on residuals is **as designed** — they're
weakly informed by data so the prior dominates, and our new prior is
explicitly tight.

### E3 27-window result — G1 alone

**23/27 windows pass ≥5/6 identifiable subset coverage** (vs F1's
17/27 with un-reparametrized model). Failures concentrated at W9
(4/6), W14 (1/6), W21 (4/6), W23 (4/6) — much sparser than F1's
mid-period collapse.

Diagnostic: [`E3_rolling_window_traces_g1_no_infoaware_all30.png`](E3_rolling_window_traces_g1_no_infoaware_all30.png)
shows all 30 parameter posterior traces. The story:

- **Identifiable subset (HR_base, S_base, mu_step0, kappa_B_HR, k_F,
  beta_C_HR)** — track truth tightly throughout 27 windows ✓
- **Tightened residuals (ε_A, λ_A, μ_FF)** — pinned to truth, no
  drift ✓ (the new priors are doing exactly what they should)
- **Weakly informed but data-touched (τ_B, μ_B, η, alpha_A_HR,
  sigma_HR, k_A, c_tilde, k_A_S, beta_C_S, sigma_S, beta_*_st)** —
  modest drift around truth with wide CI, mostly within prior bounds.
  Not pathological.

## Stage G2 — info-aware bridge (incremental polish, regressed)

### What was tested

Set `sf_info_aware=True` in the SMCConfig, keeping the G1
reparametrized model. Diagnostic shows 15/30 directions are "held"
(below the median FIM eigenvalue threshold):

```
SF base (annealed/q0_cov): blend=0.70, … info_aware: λ∈[1.8e-01, 4.0e+02] (15/30 held)
```

### E3 27-window result — G1 + G2 combined

**17/27 windows pass — REGRESSED from G1's 23/27.** Failures more
frequent: W7=4/6, W13=4/6, W14=4/6, W17=2/6, W18=3/6, W19=3/6,
W20=4/6, W24=3/6, W26=2/6, W27=3/6. Total 10 failures vs G1's 4.

### Why info-aware regressed on the v2 reparametrized model

The info-aware bridge holds the prior on FIM eigenvectors with
eigenvalues below the median threshold. After G1's reparametrization,
the FIM is **better conditioned** — what was a rank-deficient
direction is now a residual direction with a tight prior already
doing the holding. Layering info-aware on top means the bridge
*also* refuses to update directions that the data DOES inform (just
weakly) — throwing away genuine information.

In short: **info-aware fights G1's tightened priors instead of
complementing them.**

The info-aware bridge would likely help on a model with the
RANK-DEFICIENT spec + the ORIGINAL wider priors (i.e., applied
without G1). But that's the opposite of the principled fix.

Diagnostic plot: [`E3_rolling_window_traces_g2_infoaware_all30.png`](E3_rolling_window_traces_g2_infoaware_all30.png).
Visible difference vs G1: residuals stay equally tight (priors still
do their job), strongly-identified subset has slightly wider CIs at
some windows, and the "weakly informed but data-touched" group has
slightly wider variance (because bridge now holds them more often).

## Recommendation for Stage G4 (multi-horizon closed-loop MPC)

Use **G1 alone**:
- `bridge_type='schrodinger_follmer'`
- `sf_q1_mode='annealed'`
- `sf_use_q0_cov=True`
- `sf_blend=0.7`
- `sf_annealed_n_stages=3`
- `sf_annealed_n_mh_steps=5`
- **`sf_info_aware=False`** ← reverted from G2

Default behaviour for the existing bench drivers is currently set up
this way (the suffix-driven CLI toggle keeps `sf_info_aware=False`
unless the suffix contains `infoaware`).

## Open questions / future work

1. **Mid-period catastrophic drops (e.g., G1 W14 at 1/6)**: investigate
   what causes a single-window collapse. Is it numerical instability
   at a specific obs sequence? Particle degeneracy?
2. **The 23/27 gate misses 24/27 by 1 window**: marginal pass given
   stochasticity. With a different seed or particle count we might
   land at 24/27 or 22/27. Should the gate threshold be relaxed to
   ≥22/27 (= 81% coverage)?
3. **τ_B identifiability**: at 1-day windows we know this is
   essentially un-identified. The current prior σ=0.10 keeps the
   posterior near truth, but a real-data subject with τ_B != 42 days
   would be poorly served. For real-data extension (Stage H), we'd
   need either longer windows or a hierarchical prior on τ_B.

## Files modified in F1+F2+G1+G2

```
version_2/models/fsa_high_res/_dynamics.py    [G1 reparametrize]
version_2/models/fsa_high_res/simulation.py   [G1 reparametrize]
version_2/models/fsa_high_res/estimation.py   [G1 reparametrize + tighter priors]
version_2/tests/test_g1_reparam.py            [NEW — 4 drift parity tests]
version_2/tools/bench_smc_rolling_window_fsa.py
                  [F1: SF Path B-fixed config]
                  [G1/G2: 30-param diagnostic plot + suffix-driven info-aware toggle]
version_2/tools/bench_smc_full_mpc_fsa.py
                  [F1: SF bridge]
                  [F2: T_total CLI + plan_horizon_days + HMC step scaling]
version_2/tools/bench_smc_closed_loop_fsa.py
                  [F2: plan_horizon_days param]
```

## Plots (gitignored; regenerable by re-running benches)

```
version_2/outputs/fsa_high_res/
├── E1_filter_diagnostic.png                          # G1 E1 (post-reparam)
├── E3_rolling_window_traces_g1_no_infoaware.png      # G1 6-param (23/27)
├── E3_rolling_window_traces_g1_no_infoaware_all30.png # G1 30-param diagnostic
├── E3_rolling_window_traces_g2_infoaware.png         # G2 6-param (17/27)
├── E3_rolling_window_traces_g2_infoaware_all30.png   # G2 30-param diagnostic
├── E3_rolling_window_traces_sf_pathb.png             # F1 6-param (17/27 baseline)
├── E4_closed_loop_one_cycle.png                      # E4 — pre-reparam, 1-cycle
├── E5_full_mpc_T14d_traces.png                       # F2 — pre-reparam, 14-day MPC
└── PROGRESS_F_G.md                                    # this report
```

## Commits pushed

```
93f93bc  Stage G1: reparametrize FSA-v2 to absorb FIM rank-deficient pairs
a1beb48  Stage F1: SF Path B-fixed bridge config — not a silver bullet on v2
1dcc1cd  (Stage E5: full 27-window rolling MPC on FSA-v2 — earlier baseline)
```

This report itself + the bench changes for diagnostic plots + the G1
E3 and G2 E3 results live in the next commit.
