# python-smc2-filtering-control

**SMC² for Bayesian filtering and stochastic optimal control**, demonstrated
end-to-end on simple test models with closed-form ground truth.

The repo exercises **two pillars** of the same outer tempered-SMC engine:

1. **Filtering side** — Bayesian inference of SDE parameters and latent
   states from time-series observations. Rolling-window SMC² with a
   Schrödinger-Föllmer bridge (information-aware, FIM-keyed
   per-eigenvector blend).
2. **Control side** — stochastic optimal control via the
   control-as-inference duality. The same outer kernel is reused with
   the marginal log-likelihood replaced by `-β · J(u)` where `J` is a
   cost functional and `u` is the control schedule.

## Two test models

| Model | Stage | Why |
|---|---|---|
| **scalar OU LQG** (`models/scalar_ou_lqg/`) | A | Smallest principled filter+control problem with closed-form Kalman, LQR, and joint LQG via the separation principle. Tightest analytical gates. |
| **bistable controlled** (`models/bistable_controlled/`) | B | 2-state cubic-drift + OU-control SDE with a saddle-node bifurcation. Pedagogical "push the subject out of the pathological basin" demo. |

## Layout

```
smc2fc/                # framework package
  estimation_model.py    # base dataclass for filtering models
  _likelihood_constants.py
  core/
    tempered_smc.py      # outer SMC² engine + bridge dispatch
    sf_bridge.py         # Schrödinger-Föllmer bridge (info-aware variant)
    config.py            # SMCConfig
    mass_matrix.py
    sampling.py
  filtering/
    gk_dpf_v3_lite.py    # locally-optimal Pitt-Shephard inner-PF
    _gk_kernel.py
    resample.py          # OT-rescue Sinkhorn resampler
    transport_kernel.py
    sinkhorn.py
    project.py
    kalman.py            # (Stage A) analytical scalar Kalman filter
    ekf.py               # (Stage B) extended KF baseline
  control/               # (extracted after Stage A2 + Stage B2)
  transforms/
    unconstrained.py
  simulator/
    sde_model.py
    sde_observations.py
    sde_solver_diffrax.py
models/
  scalar_ou_lqg/         # (Stage A)
  bistable_controlled/   # (Stage B — carried from Python-Model-Development-Simulation)
drivers/
  scalar_ou_lqg/         # (Stage A)
  bistable_controlled/   # (Stage B)
tests/
outputs/
tools/
```

## Setup

```bash
git clone <this repo>
cd python-smc2-filtering-control
pip install -e ".[test]"
pytest tests/test_sf_bridge.py -v       # 25 carried tests should pass
```

## Status

This is a fresh repo, bootstrapped from prior experimental work. Stage 0
(scaffold, carry over the principled filtering primitives + SF bridge)
is complete. Stages A and B are in flight.
