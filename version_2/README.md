# version_2 — closed-loop SMC²-MPC

Closed-loop MPC pipeline that connects the filter side and control side of the framework. Two models live here:

- **`fsa_high_res`** — 3-state Banister-coupled physiological SDE (B, F, A) with a 4-channel obs model (HR, sleep, stress, steps). Single control input Φ. Shipped through 27-window rolling MPC.
- **`swat`** — 4-state Sleep-Wake-Adenosine-Testosterone SDE (W, Z, a, T) with the same 4-channel obs model. Three exogenous control variates (V_h vitality, V_n chronic load, V_c phase shift). Currently being imported on branch `feat/import-swat-from-dev-repo` — Phase 3.6 formulation matches the upstream `Python-Model-Development-Simulation/SWAT_dev` repo byte-equivalent.

The framework engines (`smc2fc/core`, `smc2fc/filtering`, `smc2fc/control`) are model-agnostic; both models drop in via the 3-file convention.

## Architecture

```
filter (4-channel rolling SMC²)
        │   posterior over (params, latent state)
        ▼
controller (tempered SMC over RBF schedule)
        │   plans control schedule for next stride
        ▼
StepwisePlant.advance(stride_bins, controls)  →  new obs
        │
        ▲
        └─────  feedback to next window's filter
```

## Layout

```
version_2/
├── models/
│   ├── fsa_high_res/                 # FSA-v2 model (3-file convention + plant + control)
│   │   ├── _dynamics.py              # pure-JAX drift/diffusion + IMEX
│   │   ├── _phi_burst.py             # daily Φ → sub-daily morning-loaded burst
│   │   ├── _plant.py                 # StepwisePlant: simulator-as-plant
│   │   ├── simulation.py             # forward sim + 4 obs samplers + DEFAULT_PARAMS
│   │   ├── estimation.py             # SMC² EstimationModel
│   │   └── control.py                # ControlSpec + 8-RBF Φ basis + ∫A cost
│   └── swat/                         # SWAT 4-state model
│       ├── _dynamics.py              # 4-state Stuart-Landau drift + Jacobi/state-INDEP diffusion
│       ├── _v_schedule.py            # V_h/V_n/V_c daily → bin expansion
│       ├── _plant.py                 # StepwisePlant for 3-D control input
│       ├── simulation.py             # forward sim + 4 obs samplers + scenario presets
│       ├── estimation.py             # SMC² EstimationModel (G1-reparam: E_crit, mu_E)
│       ├── control.py                # ControlSpec for (V_h, V_n, V_c) + ∫T cost + λ_E·∫E
│       └── sim_plots.py              # diagnostic panels (latents/obs/entrainment)
├── tools/                            # bench drivers + launchers + plot tools
│   ├── bench_smc_filter_fsa.py       # single-window filter (Stage E1)
│   ├── bench_smc_rolling_window_fsa.py   # 27-window open-loop rolling SMC²
│   ├── bench_smc_closed_loop_fsa.py  # filter→plan→apply, single cycle (E4)
│   ├── bench_smc_full_mpc_fsa.py     # full 27-window MPC (E5) — FSA-v2
│   ├── bench_smc_full_mpc_swat.py    # full closed-loop MPC — SWAT
│   ├── bench_lqg_baseline_fsa.py     # LQG baseline for FSA-v2
│   ├── compare_g4_lqg.py             # SMC² vs LQG comparison
│   ├── plot_param_traces.py          # post-hoc parameter trace plots
│   ├── regenerate_swat_dev_panels.py # post-hoc dev-repo-style panels for SWAT
│   ├── load_g4_run.py                # load saved run artifacts
│   ├── check_t28_gate.py             # standalone T=28d gate check
│   ├── test_inference_single_stride.py
│   └── launchers/                    # tmux-friendly shell launchers
│       ├── run_horizon.sh            # FSA T=14/28/42/56/84 sweep
│       ├── run_swat_horizon.sh       # SWAT horizon sweep
│       ├── run_swat_overnight_chain.sh  # canonical SWAT chain (auto-activates comfyenv)
│       ├── run_h1h_sweep.sh
│       └── run_t42_only.sh
├── tests/
│   ├── test_e2_plant.py              # Φ-burst integral + StepwisePlant composition
│   ├── test_g1_reparam.py            # G1 drift-parity (FSA reparametrization)
│   ├── test_h1h_grid.py              # FSA_STEP_MINUTES grid coarsening sanity
│   ├── test_jax_native_smc.py        # JAX-native tempered SMC vs BlackJAX equivalence
│   └── test_lqg.py                   # LQG/Riccati smoke tests
└── outputs/
    ├── fsa_high_res/                 # FSA results: RESULT.md, GPU_TUNING_RTX5090.md, g4_runs/
    └── swat/                         # SWAT results: CHANGELOG.md, experiments/runNN_<tag>/
```

## Running

Activate `comfyenv` first — it has all required packages (JAX/CUDA, BlackJAX, diffrax) installed:

```bash
conda activate comfyenv
cd version_2

# Tests
PYTHONPATH=.:.. pytest tests/ -v

# FSA-v2 benches
PYTHONPATH=.:.. python tools/bench_smc_filter_fsa.py
PYTHONPATH=.:.. python tools/bench_smc_rolling_window_fsa.py
PYTHONPATH=.:.. python tools/bench_smc_closed_loop_fsa.py
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa.py [T_days]

# SWAT benches
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_swat.py [T_days] \
    --step-minutes 15 \
    --scenario {pathological,set_A}

# Long unattended runs go through tmux + the launchers
tmux new -s swat -d "$PWD/tools/launchers/run_swat_overnight_chain.sh"
```

## FSA-v2 — window structure

1-day windows × 12-hour stride × 14 days = **27 windows**. Replan every K=2 windows (= once per day at the wake boundary).

## SWAT — window structure

3-hour stride × 1-day filter window × T_days. Replan every **6 hours wall-clock** (= every K=2 strides at STRIDE_HOURS=3), independent of `--step-minutes`. Default `--step-minutes 15`; sub-hour resolution is required because SWAT's fast-subsystem (W, Z, a) timescale is ~30–60 min and the sleep/wake transitions identify many obs-side parameters (κ, λ, α_HR, c̃, W_thresh, …). FSA-v2's h=1h does **not** generalize.

Scenario presets (`--scenario`):

- `pathological` (default): cold-start `T_0=0`, `V_h=0`, `V_n=4`, `V_c=12h`. Controller must drive recovery across the Stuart-Landau bifurcation.
- `set_A`: healthy baseline `V_h=1`, `V_n=0.2`, `V_c=0`. Sanity check.

`SWAT_LAMBDA_E` env var overrides the `λ_E·∫E_dyn` shaping weight in the cost (default 1.0). The cost is `-∫T dt + λ_E·∫E_dyn` — `λ_E=0` reverts to pure ∫T.

Currently pinned (FIM rank deficiency): `tau_T`, `lambda_amp_Z`. A principled reparametrization (absorb `tau_T` into rates; single `lambda_amp`) is queued as future work — see `project_upgrade_plans/`.

## Per-run results

- **FSA-v2**: see [`outputs/fsa_high_res/RESULT.md`](outputs/fsa_high_res/RESULT.md), [`outputs/fsa_high_res/GPU_TUNING_RTX5090.md`](outputs/fsa_high_res/GPU_TUNING_RTX5090.md), and `outputs/fsa_high_res/g4_runs/T<NN>d_replanK2_*/`.
- **SWAT**: per-run summaries in [`outputs/swat/CHANGELOG.md`](outputs/swat/CHANGELOG.md); per-run artifacts under `outputs/swat/experiments/runNN_<tag>/` (parameter traces, latent panels, observation panels, entrainment plot, manifest.json + data.npz).
