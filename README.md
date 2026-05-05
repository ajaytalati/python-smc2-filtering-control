# python-smc2-filtering-control

**SMC² for Bayesian filtering and stochastic optimal control** on
physiological SDE models. The same outer tempered-SMC engine drives
both pillars:

- **Filter side** — Bayesian inference of SDE parameters and latent
  states from noisy time-series observations. Rolling-window SMC²
  with a Schrödinger–Föllmer bridge between windows.
- **Control side** — stochastic optimal control via the
  control-as-inference duality (Toussaint 2009 / Levine 2018 /
  Kappen 2005). The same kernel is reused with the marginal
  log-likelihood replaced by `−β · J(u)`.

Closed-loop MPC composes the two: filter → posterior → controller
plans schedule → step-wise simulator-as-plant → next window's filter.

## Layout

```
python-smc2-filtering-control/
├── smc2fc/         framework (tempered-SMC engine, SF-bridge, inner PF, control loop, simulator)
├── version_2/      closed-loop MPC pipeline; two physiological SDE models
│   └── models/
│       ├── fsa_high_res/   Banister-coupled 3-state model (HR / sleep / stress / steps obs)
│       └── swat/           Sleep-Wake-Adenosine-Testosterone 4-state model
├── claude_plans/   project plans + debugging methodology (kept in sync with `~/.claude/plans/`)
└── CLAUDE.md       project conventions, debugging methodology, per-model gotchas
```

**Read [`CLAUDE.md`](CLAUDE.md) first** if you're new to the repo —
it covers the senior-files principle, controller-only debugging,
the plan-archive policy, the JAX/conda env, and per-model gotchas.

## Setup

```bash
git clone https://github.com/ajaytalati/python-smc2-filtering-control
cd python-smc2-filtering-control
conda activate comfyenv          # JAX/CUDA, BlackJAX, diffrax already installed
pip install -e ".[test]"
```

Drivers and tests run from inside `version_2/` with `PYTHONPATH=.:..`
on the path so both the version dir (for sibling `models.X` imports)
and the repo root (for `smc2fc...` imports) resolve:

```bash
cd version_2
PYTHONPATH=.:.. pytest tests/ -v
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa.py 14    # FSA-v2 closed-loop, 14-day horizon
PYTHONPATH=.:.. python tools/bench_smc_full_mpc_swat.py 14 --scenario pathological
```

## Adjacent repos

Per-model dev sandboxes — models are developed and validated there
(identifiability, stiffness, plant-vs-estimator reconciliation,
likelihood, controller checks) before being ported into
`version_2/models/<name>/` for closed-loop runs.

- **[FSA_model_dev](https://github.com/ajaytalati/FSA_model_dev)**
  — dev sandbox for the FSA-v2 (Banister-coupled) model.
- **[SWAT_model_dev](https://github.com/ajaytalati/SWAT_model_dev)**
  — dev sandbox for the SWAT (Sleep-Wake-Adenosine-Testosterone)
  model.

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
