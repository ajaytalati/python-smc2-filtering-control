# python-smc2-filtering-control

**SMC² for Bayesian filtering and stochastic optimal control**, demonstrated
end-to-end on three test models — two with closed-form ground truth, one
the user's actual research target (a 3-state physiological SDE).

The repo exercises **two pillars** of the same outer tempered-SMC engine:

1. **Filtering side** — Bayesian inference of SDE parameters and latent
   states from time-series observations. Rolling-window SMC² with a
   Schrödinger-Föllmer bridge (information-aware, FIM-keyed
   per-eigenvector blend).
2. **Control side** — stochastic optimal control via the
   control-as-inference duality (Toussaint 2009; Levine 2018; Kappen
   2005). The same outer kernel is reused with the marginal log-
   likelihood replaced by `-β · J(u)` where `J` is a cost functional
   and `u` is the control schedule.

## Three test models, all gates passing

| Stage | What | Headline gate | Result |
|-------|------|---------------|--------|
| **A1** scalar OU LQG filter | PF log p(y\|truth) vs analytical Kalman LL | bias < 5 nats | **−0.18 nats** ✓ |
| **A2** scalar OU LQG control (open-loop) | SMC² / open-loop u=0 cost ratio | ∈ [0.95, 1.10] | **1.036** ✓ |
| **A3** scalar OU LQG control (state feedback) | SMC² / MC LQG cost ratio | ∈ [0.95, 1.10] | **0.995** ✓ |
|        | K RMS error vs Riccati gains | < 25% | **20%** ✓ |
| **B1** bistable filter | 90% CI covers truth (identifiable params) | 4/4 | **8/8** ✓ |
| **B2** bistable control | basin transition rate, cost vs default | ≥ 80%, ≤ default | **100%, 38%** ✓ |
| **B3** bistable closed-loop | transition rate, cost vs oracle | ≥ 80%, ≤ 1.20× | **100%, 0.99×** ✓ |
| **D**  FSA control v2 (Banister) — T=42d  | mean ∫A/T vs sedentary, vs constant Φ=1; F-viol | ≥ 1.40×, match within 3%, ≤ 5% | **1.48×, 1.005×, 0.05%** ✓ |
|        T=56d (1.33 τ_B)                  | same (gain over const baseline grows)  | — | SMC² **+20%** over const, +127% over sedentary ✓ |
|        T=84d (2.00 τ_B)                  | same (front-loaded periodisation)       | — | SMC² **+28%** over const, +208% over sedentary ✓ |
|        T=28d (0.67 τ_B)                  | sub-canonical: too short for any schedule | — | gates fail by design — horizon < τ_B ✗ |

See [`outputs/scalar_ou_lqg/RESULT.md`](outputs/scalar_ou_lqg/RESULT.md),
[`outputs/bistable_controlled/RESULT.md`](outputs/bistable_controlled/RESULT.md),
and [`outputs/fsa_high_res/RESULT.md`](outputs/fsa_high_res/RESULT.md)
for full details.

## Headline plots

### A3 — state-feedback recovers LQR/LQG

![A3 state-feedback](outputs/scalar_ou_lqg/A3_state_feedback_diagnostic.png)

Tempered SMC over the gain vector `K` discovers the analytical Riccati
solution. SMC²-mean-gain cost (5.61) matches the MC LQG cost (5.63)
within 0.5%; both are 40% below open-loop (9.29). Gain shape tracks
the Riccati decay through the horizon.

### B2 — SMC² discovers front-loaded intervention

![B2 control](outputs/bistable_controlled/B2_control_diagnostic.png)

SMC²-as-controller (top-left) finds that `u_target ≈ 1.5` at t=0
decaying through the horizon is far cheaper than the hand-coded
default's "wait 24h then u_on=0.5" step. Same 100% basin-transition
success rate, **62% lower cost**. The SMC² didn't know about u_c or
"early intervention is better" — just minimised expected cost.

### B3 — Closed-loop pipeline (filter then plan)

![B3 closed-loop](outputs/bistable_controlled/B3_closed_loop_diagnostic.png)

Phase 1 (0-24h, no control): observe x in the pathological well.
Filter to get parameter posterior. Use posterior mean to plan Phase 2
schedule. The closed-loop trajectory under SMC²-with-posterior (blue)
is statistically indistinguishable from the truth-params oracle
(purple dashed).

### D — FSA Banister control: SMC² discovers front-loaded periodisation

![D FSA control](outputs/fsa_high_res/D_v2_T84_diagnostic.png)

3-state physiological SDE — **Banister-coupled v2** (fitness B
accrues from training Φ; strain F also driven by Φ; Stuart-Landau
amplitude A modulated by μ(B, F)). Square-root Itô diffusion
(Jacobi for B, CIR for F and A) so each state stays in its
physiological domain without clipping. Single control input Φ(t).
Cost rewards time-averaged amplitude `∫A(t)dt` with a soft
F-overshoot barrier (no Φ²-effort penalty — the F-barrier already
penalises overtraining endogenously).

Run at the long horizon T = 84 d (= 2 · τ_B), SMC² discovers a
clean **front-loaded periodisation pattern**: Φ ≈ 2.5 at t = 0
ramping down through ≈ 2.0 in the build phase to ≈ 1.5 in the
maintain phase. F brushes the F_max = 0.40 ceiling around day 30
and decays as autonomic feedback (`λ_A·A`) accelerates F-clearance.
A grows from 0.10 to its plateau at ≈ 1.0 by day 60.

Headline: **mean ∫A/T = 0.645, +28% over the best constant
baseline** (Φ=1.0 → 0.503), **+208% over sedentary** (Φ=0 → 0.209),
F-violation 1.28% (well under 5% gate). The shape is exactly what
canonical Banister periodisation prescribes — **discovered by SMC²
without being told**.

This is v2; v1 was rejected during user review for admitting a
degenerate "rest cures all" optimum. See
[`outputs/fsa_high_res/RESULT.md`](outputs/fsa_high_res/RESULT.md)
for the full discussion.

## Layout

```
smc2fc/                # framework package
  estimation_model.py    # base dataclass for filtering models
  _likelihood_constants.py
  core/
    tempered_smc.py      # outer SMC² engine + bridge dispatch
    sf_bridge.py         # Schrödinger-Föllmer bridge (info-aware variant)
    config.py            # SMCConfig
    mass_matrix.py       # diagonal mass-matrix re-estimation
    sampling.py
  filtering/
    gk_dpf_v3_lite.py    # locally-optimal Pitt-Shephard inner-PF
    _gk_kernel.py        # ESS helper
    resample.py          # OT-rescue Sinkhorn resampler
    transport_kernel.py
    sinkhorn.py
    project.py
  transforms/
    unconstrained.py     # constrained ↔ unconstrained transforms
  simulator/
    sde_model.py         # SDEModel container
    sde_observations.py
    sde_solver_diffrax.py
models/
  scalar_ou_lqg/
    simulation.py        drift / diffusion / Gaussian obs
    _dynamics.py         pure-JAX drift, IMEX, obs log-prob
    estimation.py        SMC² EstimationModel
    bench_kalman.py      analytical scalar Kalman + smoother + MLE
    bench_lqr.py         analytical LQR (Riccati) + LQG MC + open-loop
  bistable_controlled/
    simulation.py        carried from public-dev (270 LoC)
    estimation.py        carried (365 LoC; locally-optimal PF)
    control.py           ControlSpec + RBF schedule + cost
    sim_plots.py
  fsa_high_res/
    _dynamics.py         pure-JAX 3-state Banister drift + sqrt-Itô EM (~140 LoC)
    control.py           ControlSpec + 8-RBF Φ schedule + ∫A cost (~370 LoC)
smc2fc/control/        # control engine (extracted Stage C, ~520 LoC)
  control_spec.py        ControlSpec dataclass
  config.py              SMCControlConfig
  tempered_smc_loop.py   run_tempered_smc_loop
  calibration.py         calibrate_beta_max + build_crn_noise_grids
  rbf_schedules.py       Gaussian-RBF schedule basis
  diagnostics.py         plot helpers + evaluate_gates
tools/
  bench_smc_control_ou.py                A2 driver
  bench_smc_control_ou_state_feedback.py A3 driver
  bench_smc_filter_bistable.py           B1 driver
  bench_smc_control_bistable.py          B2 driver
  bench_smc_closed_loop_bistable.py      B3 driver
  bench_smc_control_fsa.py               D  driver
tests/
  test_sf_bridge.py                       25 carried tests
  test_kalman_lqr_baseline.py             5 analytical bench tests
  test_scalar_ou_filter_matches_kalman.py 2 PF-vs-Kalman tests
outputs/
  scalar_ou_lqg/
    RESULT.md
    A2_control_diagnostic.png
    A3_state_feedback_diagnostic.png
  bistable_controlled/
    RESULT.md
    B1_filter_diagnostic.png
    B2_control_diagnostic.png
    B3_closed_loop_diagnostic.png
  fsa_high_res/
    RESULT.md
    D_v2_T84_diagnostic.png    # T=84d (long-horizon, +28% over baseline)
    D_v2_T42_diagnostic.png    # T=42d (canonical τ_B; flat optimum, SMC matches baseline)
```

## Setup

```bash
git clone <this repo>
cd python-smc2-filtering-control
pip install -e ".[test]"

# unit tests (fast, ~1 minute)
pytest tests/ -v       # 47 tests, all green

# headline benchmarks (each ~1-3 min on CPU; D needs a GPU, ~15 min on RTX 5090)
python tools/bench_smc_control_ou.py                  # A2
python tools/bench_smc_control_ou_state_feedback.py   # A3
python tools/bench_smc_filter_bistable.py             # B1 (~3 min)
python tools/bench_smc_control_bistable.py            # B2 (~10 min)
python tools/bench_smc_closed_loop_bistable.py        # B3 (~2 min)
python tools/bench_smc_control_fsa.py 84               # D  (~55 min on GPU; pass T=42 for ~30 min run)
```

## What's carried over (and what's fresh)

The principled framework primitives are carried from prior experimental
repos with `smc2bj → smc2fc` import rewrites:

- **Outer tempered-SMC engine** (`smc2fc/core/tempered_smc.py`) — the
  heart of both filter and control sides.
- **Schrödinger-Föllmer bridge** (`smc2fc/core/sf_bridge.py`) including
  the recently-developed FIM-keyed information-aware variant. 25/25
  tests carry over green.
- **Locally-optimal Pitt-Shephard inner PF** (`smc2fc/filtering/gk_dpf_v3_lite.py`)
  with OT rescue + Liu-West correction.
- **Bistable controlled model** (`models/bistable_controlled/`) — 270
  LoC simulation + 365 LoC estimation, refactored only against the
  new package paths.

Everything new lives in `models/scalar_ou_lqg/` (Stage A model + Kalman/LQR
benches) and `tools/` (the per-stage benchmark scripts).

## References

- Toussaint, M. (2009). "Robot trajectory optimization using approximate
  inference." ICML.
- Levine, S. (2018). "Reinforcement learning and control as
  probabilistic inference." arXiv:1805.00909.
- Kappen, H. J. (2005). "Path integrals and symmetry breaking for
  optimal control theory." J. Stat. Mech.
- Andrieu, C., Doucet, A., Holenstein, R. (2010). "Particle Markov
  chain Monte Carlo methods." JRSS-B.

## License

MIT.
