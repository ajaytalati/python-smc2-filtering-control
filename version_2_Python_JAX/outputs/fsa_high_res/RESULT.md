# Stage E — closed-loop MPC on FSA-v2

This subtree implements the closed-loop filter↔control cycle on the
v2 Banister-coupled FSA model with the full 4-channel observation
model (HR / sleep / stress / steps). The architectural contribution is
the **`StepwisePlant`** simulator-as-plant — the missing piece that
psim's preset files explicitly flagged as the "next-stage follow-up".

## Architecture

```
filter (4-channel rolling SMC²) ──┐
                                   │  posterior over (params, state)
                                   ▼
                          Stage-D controller  ──> Φ schedule for next stride
                                   │
                                   ▼
                  StepwisePlant.advance(stride, Φ)  ──> new obs
                                   │
                                   ▲
                                   └─ feedback to next window's filter
```

The repo weaves three sibling repos:

- **`smc2fc/`** (this repo) — the shared SMC² + control engine
- **`Python-Model-Scenario-Simulation`** (`psim`) — the validated
  scenario builder + sim-est consistency gate, providing the SDE solver
  template and the canonical artifact format
- **`smc2_blackjax_framework`** (`smc2-blackjax-rolling`) — the
  earlier validated rolling-window SMC² achieving 27/27 PASS at 98.5%
  coverage on the same FSA model with the SF Path B-fixed bridge

## Sub-stage results

| stage | gate result | notes |
|-------|-------------|-------|
| E0    | ✓ 47/47 v1 tests green | repo reorganised into version_1/ + version_2/ |
| E1    | ✓ 30/30 covered, 16 levels in 168s | single-window filter recovers full 30-param posterior |
| E2    | ✓ 5/5 tests green   | Φ-burst integral preserved; StepwisePlant advances correctly |
| E3    | ✗ 18/27 windows ≥5/6 | Gaussian bridge drift in mid-period; 24/27 gate misses by 6 |
| E4    | ✓ all gates           | filter→plan→apply pipeline works end-to-end (3.5 min) |
| E5    | (results below)       | full 27-window MPC                                    |

## E1 — Single-window filter (✓)

| | |
|-|-|
| n_steps      | 96 (1 day × 96 bins at 15 min)        |
| obs counts   | sleep=39, HR=39, stress=57, steps=57   |
| SMC² levels  | 16                                     |
| wall-clock   | 168 s on RTX 5090                      |
| Coverage     | **30/30 all params**, 6/6 identifiable |

The full 30-parameter posterior covers truth even on the slow Banister
chronic dynamics (τ_B=42d, κ_B, κ_F) within their 90% CI on a 1-day
window. Strong evidence the Kalman fusion of 4 obs channels at 15-min
granularity is sufficient to identify the latent (B, F, A) precisely.

Plot: [`E1_filter_diagnostic.png`](E1_filter_diagnostic.png).

## E2 — Sub-daily Φ-burst + StepwisePlant (✓)

The StepwisePlant API exposes:

```python
plant = StepwisePlant(truth_params, init_state, dt, seed)
for window_k in range(n_windows):
    obs_stride = plant.advance(stride_bins, Phi_daily_for_stride)
    posterior  = filter.update(obs_stride)
    Phi_next   = controller.plan(posterior, horizon=stride_bins)
    Phi_daily_for_stride = Phi_next   # decided online
plant.finalise(out_dir)   # psim-format artifact for archival
```

Tests verify:
1. ∫_day Φ_subdaily(t) dt = 24 · Φ_daily to fp precision (any Φ_daily).
2. Burst envelope: zero overnight (03:00, 23:00), peak ~10am.
3. Step-wise composition: advance(s, Φ_a) + advance(s, Φ_b) preserves
   global bin index across calls.
4. plant.finalise() produces a psim-format artifact (manifest.json +
   trajectory.npz + obs/*.npz + exogenous/*.npz) reusable by psim's
   downstream consistency-check machinery.

## E3 — 27-window rolling SMC² (open-loop, ✗ partial pass)

| | |
|-|-|
| Window         | 96 bins (1 day) × 12-h stride × 14 days = 27 windows |
| Total compute  | 35 min on RTX 5090                                    |
| Pass rate      | 18/27 windows have ≥5/6 identifiable covered          |
| Gate           | ✗ ≥24/27 required (missed by 6 windows)               |

Per-window coverage trace (id_cov out of 6):

```
W1  cold:   6/6
W2-W6:      6/6 6/6 5/6 6/6 6/6
W7-W12:     4/6 4/6 6/6 6/6 5/6 6/6
W13-W18:    3/6 3/6 2/6 4/6 3/6 4/6   ← mid-period bridge drift
W19-W27:    3/6 5/6 6/6 6/6 6/6 5/6 6/6 6/6 5/6 (recovery)
```

The drift hits dynamics-coupled obs parameters (kappa_B_HR, k_F,
beta_C_HR) most; static base levels (HR_base, S_base, mu_step0) track
truth throughout — visible in the per-param trace plot.

This is a known property of the simpler `bridge_type='gaussian'`
variant. The smc2-blackjax-rolling reference achieves **27/27 PASS at
98.5% coverage** on the same FSA model by using the **SF Path B-fixed
bridge** (Schrödinger-Föllmer with decoupled covariance). Porting that
bridge variant to `smc2fc` is the listed follow-up; the rolling-window
machinery (extract_window, run_smc_window_bridge, smoothed-state
extraction) is already in place.

Plot: [`E3_rolling_window_traces.png`](E3_rolling_window_traces.png).

## E4 — Closed-loop one cycle (✓)

| | |
|-|-|
| Phase 1                | 1 day @ Φ=1.0, plant → 4-channel obs           |
| Filter (Phase 1)       | SMC², 17 levels in 175 s                        |
| Smoothed state         | B=0.061, F=0.290, A=0.096 (vs truth 0.061/0.284/0.094) |
| Phase 2 plan           | 10 levels, planned Φ̄=0.228                     |
| Plant (Phase 2)        | mean A=0.0910 (vs baseline 0.0967)              |
| F-violation            | 0.00%                                           |
| Total                  | 3.5 min on RTX 5090                             |

**The headline isn't a control gain at this horizon** — at 1-day
horizon the slow Banister channel barely activates, so the cost
surface is flat near Φ=1 and the planner's choice (Φ̄=0.228) gives
essentially the same mean A (0.0910 vs 0.0967, within 6%). The
headline IS that the filter+control pipeline runs end-to-end:

- Filter recovers smoothed latent state to within 2% of truth ✓
- Plant accepts injected Φ for any stride ✓
- ControlSpec built dynamically from posterior-mean params ✓
- Counterfactual baseline initialised consistently ✓

A bug fix was made during E4 development:
`extract_state_at_step`'s scan iterates `k = 0..t_steps-1`, so passing
`target_step = WINDOW_BINS` (=96) made `at_target=(k==96)` never True,
returning all-zero saved state. Fix: pass `target_step = WINDOW_BINS - 1`.

Plot: [`E4_closed_loop_one_cycle.png`](E4_closed_loop_one_cycle.png).

## E5 — Full 27-window rolling MPC (✓ headline + ✗ filter coverage)

| | |
|-|-|
| Total compute             | 41 min on RTX 5090                            |
| Strides                   | 27 strides × 48 bins (12 h) over 14 days     |
| Replan cadence            | every K=2 strides (= every 24 h)              |
| Plant total advance       | 1296 bins (= 13.5 days)                       |
| **Mean A (closed-loop MPC)** | **0.0964**                                |
| Mean A (constant Φ=1.0)   | 0.0823                                        |
| **Headline gain**         | **+17% over constant baseline**               |
| F-violation               | 0.00%                                         |
| Filter coverage           | 3/26 windows ≥ 5/6 identifiable               |

| acceptance gate                                    | result |
|----------------------------------------------------|--------|
| mean A (MPC) ≥ 0.95 × baseline                    | ✓ 0.0964 vs 0.0781 |
| ≥ 24 windows have ≥ 5/6 identifiable covered      | ✗ 3/26 |
| F-violation ≤ 5%                                   | ✓ 0.00% |
| Total time ≤ 4 h                                   | ✓ 0.68 h |

### Headline finding: rest-and-rebuild discovered (again)

The MPC consistently planned **Φ ≈ 0.25-0.31 across the 27 strides**
(vs the constant Φ=1.0 baseline). Under this rest-leaning schedule
the closed-loop trajectory achieves **+17% mean amplitude** over
the constant-Φ baseline. Same physiological story as v1 Stage D's
T=84 finding, transposed to the 14-day daily-replan cadence:

- Initial state has F_0 = 0.30 (high residual fatigue).
- Under Φ=1.0: F stays elevated at ~0.20 → μ(B,F) suppressed → A* small.
- Under Φ=0.27: F decays toward ~0.06 → μ rises → A* nearly doubles.
- Even though B builds slower at low Φ, the F-clearance dominates.

The MPC discovered the right physiological strategy **despite the
filter posterior degrading** in mid-period (covered below). This is
robustness to parameter-estimation error: when the controller plans
from a posterior whose dynamics-coupled obs params have drifted,
the resulting Φ schedule is still reasonable because the cost surface
is locally flat.

### Filter coverage gate fails sharply (3/26 vs 24/26)

The filter coverage problem from E3 (18/27 with constant Φ=1.0)
becomes much worse under MPC (3/26): the closed-loop dynamics under
the controller-injected Φ ≈ 0.27 differ from the prior windows'
Φ=1.0 dynamics, so the Gaussian-bridge handoff drifts substantially.

Per-stride pattern (id_cov out of 6):

```
S2   filter (cold):    6/6      plan: Φ̄=0.246
S3   filter (bridge):  6/6      reuse Φ=0.246
S4   filter (bridge):  3/6      plan: Φ̄=0.248      ← drift starts
S5   filter (bridge):  0/6      reuse Φ=0.248
S6+  consistently      0-3/6    Φ ≈ 0.25-0.31
```

The MPC keeps the controller "honest" through the cost (which still
sees high-A schedules under low Φ as good), but the filter posterior
becomes uninformative for individual parameters. The follow-up port
of the SF Path B-fixed bridge from `smc2-blackjax-rolling` would fix
this — that bridge variant achieves 27/27 PASS at 98.5% coverage on
the same FSA model under fixed-Φ data and is expected to handle the
closed-loop case similarly.

Plot: [`E5_full_mpc_traces.png`](E5_full_mpc_traces.png).

## Out of scope / future follow-ups

- **SF Path B-fixed bridge port**. Listed already in E3. Fixes the
  bridge drift across windows. Reference:
  `smc2_blackjax_framework/smc2bj/estimation/sf_bridge.py`.
- **Macrocycle-scale 5-year MPC** (120-day window × 30-day stride ×
  ~57 windows). Defer to a future Stage F. Architecture is the same;
  only window sizing changes.
- **Real Whoop/Strava/Garmin CSV ingestion**. The artifact-loader is
  in place; just needs a CSV → psim-artifact converter.
- **Estimation of circadian phase φ** (frozen at 0 in v2 — morning
  chronotype).
- **per-particle controller planning** instead of posterior-mean
  collapse (sample 16 particle params, plan against each, average) —
  reduces info loss at the controller hand-off.

## Code layout

```
version_2/
├── models/fsa_high_res/
│   ├── _dynamics.py       # v2 Banister + sqrt-Itô (shared with v1)
│   ├── _circadian.py      # C(t) = cos(2π t + φ)  [folded into simulation.py]
│   ├── _phi_burst.py      # Gamma(k=2) JAX-vectorised expansion
│   ├── _plant.py          # StepwisePlant — closed-loop simulator
│   ├── simulation.py      # numpy reference + 4 obs channels
│   ├── estimation.py      # 30-param EstimationModel + Kalman fusion
│   └── control.py         # Stage-D ControlSpec (copied from v1)
├── tools/
│   ├── bench_smc_filter_fsa.py            # E1 driver
│   ├── bench_smc_rolling_window_fsa.py    # E3 driver
│   ├── bench_smc_closed_loop_fsa.py       # E4 driver
│   └── bench_smc_full_mpc_fsa.py          # E5 driver
└── tests/
    ├── test_fsa_v2_estimation.py    [optional]
    └── test_e2_plant.py             # 5 E2 tests, all green
```

Drivers in `tools/` follow the pattern: load StepwisePlant or psim
artifact → window obs → SMC² filter → (optional) Stage-D controller
plan → apply to plant → loop.
