# version_2 — Stage E: closed-loop MPC on FSA-v2

This subtree contains the closed-loop MPC pipeline that connects the
filter side and control side of the framework, applied to the FSA-v2
(Banister-coupled) physiological SDE with the full 4-channel
observation model (HR, sleep, stress, steps).

## Architecture

```
filter (4-channel rolling SMC²)
        │   posterior over (params, latent state)
        ▼
Stage-D controller — plans Φ schedule for next stride
        │
        ▼
StepwisePlant.advance(stride_bins, Φ)  →  new obs
        │
        ▲
        └─────  feedback to next window's filter
```

**Window structure**: 1-day windows (96 bins at 15-min) × 12-hour
stride (48 bins) × 14 days = **27 windows** total. Matches the
validated [smc2-blackjax-rolling](https://github.com/ajaytalati/smc2-blackjax-rolling)
fsa_high_res C0 reference (98.5% mean coverage, 27-of-27 PASS).

**Replan cadence**: every K=2 windows (= every 24 hours, at the wake
boundary), the controller plans a fresh 1-day Φ; otherwise the
previously-planned Φ is reapplied.

## Sub-stages

| Stage | Goal                                                          |
|-------|---------------------------------------------------------------|
| E1    | v2 model in 3-file convention + psim consistency + filter     |
| E2    | Sub-daily Φ-burst + StepwisePlant simulator-as-plant          |
| E3    | 27-window rolling-window SMC² (open-loop, frozen Φ)            |
| E4    | Closed-loop MPC, single cycle (filter → plan → apply)         |
| E5    | Full 27-window rolling MPC                                     |

## Layout

```
version_2/
├── models/fsa_high_res/    # v2 dynamics + circadian + Φ-burst + plant + estimation + control
├── tools/                  # bench scripts: filter / rolling / closed-loop / full MPC
├── tests/                  # E2 step-wise composition tests + sim-est consistency mirror
└── outputs/fsa_high_res/   # per-stage diagnostic plots + RESULT.md
```

## Adjacent repos

Stage E weaves three sibling repos:

- **`Python-Model-Development-Simulation/version_2/models/fsa_high_res/`**
  — canonical home for the v2 model in 3-file convention.
- **`Python-Model-Scenario-Simulation` (psim)** —
  sim-est consistency validation gate before any SMC² estimation runs.
- **`smc2-blackjax-rolling`** — earlier validated rolling-window SMC²;
  source of the SF Path B-fixed bridge variant + 27-window driver
  template.

See top-level [README.md](../README.md) for cross-repo wiring.

## Running

```bash
cd version_2

# E1 single-window filter
PYTHONPATH=.:.. python tools/bench_smc_filter_fsa.py

# E3 27-window rolling SMC² (open-loop)
PYTHONPATH=.:.. python tools/bench_smc_rolling_window_fsa.py

# E4 closed-loop, single cycle
PYTHONPATH=.:.. python tools/bench_smc_closed_loop_fsa.py

# E5 full 27-window MPC
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa.py
```

Once Stage E ships, headline plots + RESULT.md will be in
[`outputs/fsa_high_res/`](outputs/fsa_high_res/).
