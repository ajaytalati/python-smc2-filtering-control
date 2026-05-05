# python-smc2-filtering-control

**SMC² for Bayesian filtering and stochastic optimal control** —
demonstrated end-to-end on three test models, then carried through to
a closed-loop MPC pipeline on a 3-state physiological SDE (the user's
actual research target).

The repo is structured in **two top-level versioned subtrees** that
share a common framework (`smc2fc/`):

```
python-smc2-filtering-control/
├── smc2fc/                  # SHARED framework (filter + control + simulator engines)
├── version_1/               # Stages A-D — filter side validated; control side
│   │                        #   shipped on FSA-v2 (Banister) for canonical horizons
│   │                        #   T = 42, 56, 84 d. All gates passing.
│   └── README.md            # ← per-stage RESULT.md + headline plots
├── version_2/               # Stage E — closed-loop MPC: filter + 27-window
│   │                        #   rolling SMC² + Stage-D controller + step-wise
│   │                        #   simulator-as-plant. Two models:
│   │                        #     models/fsa_high_res/  — FSA-v2 (shipped, +17%)
│   │                        #     models/swat/          — 4-state SWAT (in development)
│   └── README.md
└── README.md                (you are here)
```

Project conventions, debugging methodology, and per-model gotchas live in
[`CLAUDE.md`](CLAUDE.md) — read that first if you're new to the repo.

## What's in v1 (`version_1/`)

Both pillars demonstrated bit-by-bit on simple test models with
closed-form ground truth, then carried to FSA-v2 for fully-observed
control:

| Stage | Model            | Pillar  | Headline                                                  |
|-------|------------------|---------|-----------------------------------------------------------|
| A1    | scalar OU LQG    | filter  | PF log-likelihood matches analytical Kalman to −0.18 nats |
| A2    | scalar OU LQG    | control | SMC² / open-loop ratio = 1.036 ∈ [0.95, 1.10]             |
| A3    | scalar OU LQG    | control | SMC² / MC LQG ratio = 0.995; K RMS err = 20% < 25%        |
| B1    | bistable         | filter  | 90% CI covers truth on 8/8 estimable params               |
| B2    | bistable         | control | basin transition 100%, cost 38% of default                |
| B3    | bistable         | both    | filter→plan closed-loop: cost 0.99× oracle                |
| D     | FSA-v2 (Banister)| control | T=84 d: mean ∫A/T = 1.28× constant baseline, F-viol 1.28% |

47 tests, 0 regressions. Headline plot per stage in
[`version_1/outputs/`](version_1/outputs/). Full narrative in
[`version_1/README.md`](version_1/README.md).

To re-run any v1 benchmark:
```bash
cd version_1
PYTHONPATH=.:.. pytest tests/                                  # 47/47 green
PYTHONPATH=.:.. python tools/bench_smc_control_fsa.py 84       # Stage D, ~55 min on RTX 5090
```

## What's in v2 (`version_2/`)

**Stage E**: the second pillar in earnest — closed-loop MPC on FSA-v2
with the full 4-channel observation model (HR, sleep, stress, steps)
and rolling-window SMC². The architectural contribution is the
**step-wise simulator-as-plant** that lets a controller decide the
next stride's training stimulus Φ from the filter's posterior at the
current time, rather than pre-committing to a full-horizon schedule.

```
filter (4-channel rolling SMC²) ─┐
                                  │  posterior over (params, state)
                                  ▼
                          Stage-D controller (Φ schedule for next stride)
                                  │
                                  ▼
                  StepwisePlant.advance(stride, Φ)  → new obs
                                  │
                                  ▲
                                  └─ feedback to next window's filter
```

| Sub-stage | Goal                                                  | Status |
|-----------|-------------------------------------------------------|--------|
| E1        | v2 model + 4-channel filter on 1-day window           | ✓ 30/30 covered, 16 levels in 168 s |
| E2        | Sub-daily Φ-burst + StepwisePlant simulator-as-plant  | ✓ 5/5 tests, integral preserved to fp |
| E3        | 27-window rolling SMC² (open-loop, Φ=1)               | ✗ 18/27 ≥5/6 (Gaussian-bridge drift) |
| E4        | Closed-loop MPC, single cycle                         | ✓ pipeline + smoothed state recovers truth |
| E5        | Full 27-window rolling MPC                            | ✓ +17% mean A over const Φ=1.0 baseline |

The Stage E4-E5 headline: **even with imperfect filter coverage, the
closed-loop MPC discovers a physiologically correct rest-leaning
schedule** (mean Φ ≈ 0.27 vs baseline Φ=1.0) that achieves **+17%
mean amplitude over the constant baseline** under starting state with
high residual fatigue. The controller is robust to filter parameter
drift because the cost surface near optimum is locally flat.

Window structure matches the validated reference
([smc2-blackjax-rolling](https://github.com/ajaytalati/smc2-blackjax-rolling)
fsa_high_res C0: 98.5% mean coverage, 27-of-27 PASS): **1-day windows
× 12-hour stride × 14-day total = 27 windows**.

Full narrative in [`version_2/README.md`](version_2/README.md).

A second v2 model — **SWAT** (4-state Sleep-Wake-Adenosine-Testosterone
SDE with three exogenous control variates V_h vitality, V_n chronic load,
V_c phase shift) — is being ported into the same closed-loop framework
under [`version_2/models/swat/`](version_2/models/swat/). Active
development happens on `feat/import-swat-from-dev-repo`; periodic checkpoints
land on master. Reuses the same outer SMC² engine, plant adapter, and
controller as FSA-v2.

## Adjacent repos

Stage E weaves three sibling repos:

- **[Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)**
  — canonical home for SDE models in the 3-file convention (`simulation.py`
  + `estimation.py` + `sim_plots.py`).
- **[Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation)**
  (`psim`) — the mandatory sim-est consistency validation gate
  between (1) and the SMC² estimation work. Synthesises forward-SDE
  scenarios + emits canonical artifacts (manifest.json + npz/).
- **[smc2-blackjax-rolling](https://github.com/ajaytalati/smc2-blackjax-rolling)**
  — earlier validated rolling-window SMC² implementation, the source of
  the SF Path B-fixed bridge variant + 27-window driver template.

## Setup

```bash
git clone https://github.com/ajaytalati/python-smc2-filtering-control
cd python-smc2-filtering-control
pip install -e ".[test]"

# v1 unit tests (~1 min)
cd version_1 && PYTHONPATH=.:.. pytest tests/ -v       # 47 tests, all green

# v1 headline benchmarks
cd version_1 && PYTHONPATH=.:.. python tools/bench_smc_control_fsa.py 84
```

## References

- Toussaint, M. (2009). "Robot trajectory optimization using approximate
  inference." ICML.
- Levine, S. (2018). "Reinforcement learning and control as
  probabilistic inference." arXiv:1805.00909.
- Kappen, H. J. (2005). "Path integrals and symmetry breaking for
  optimal control theory." J. Stat. Mech.
- Andrieu, C., Doucet, A., Holenstein, R. (2010). "Particle Markov
  chain Monte Carlo methods." JRSS-B.
- Banister, E.W. (1991). "Modeling elite athletic performance." In
  *Physiological testing of elite athletes*.

## License

MIT.
