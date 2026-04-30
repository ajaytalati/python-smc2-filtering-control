# SWAT cross-repo consistency check

Reproducible verification that the two presentations of the corrected
SWAT model — the **7-state form** in
[Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation)
and the **4-state form** in
[Python-Model-Validation](https://github.com/ajaytalati/Python-Model-Validation)
— produce **identical testosterone trajectories** under matching
control values. Result: sub-picosecond agreement across a 14-day
window for all three canonical scenarios.

This artefact resolved an audit ambiguity during the SWAT port plan:
Phase 0a (sync Repo C against Repo A) and Phase 0b (sync psim against
Repo A) were **dropped** as a result of the empirical evidence below.
The two repos are already mathematically consistent.

## Numerical result

| Scenario | T(14 days) — 7-state | T(14 days) — 4-state | max\|ΔT\| over 14 d |
|:---|---:|---:|---:|
| A — healthy basin | 0.826506 | 0.826506 | 1.08 × 10⁻¹² |
| B — amplitude collapse | 0.014368 | 0.014368 | 8.76 × 10⁻¹³ |
| D — phase shift | 0.013506 | 0.013506 | 2.05 × 10⁻¹⁵ |

## How to reproduce

### Requirements

- Python 3.10+
- `jax`, `jaxlib`, `diffrax`, `numpy`, `matplotlib`
- Local clones of both upstream repos:
  - `https://github.com/ajaytalati/Python-Model-Development-Simulation`
  - `https://github.com/ajaytalati/Python-Model-Validation`

### Configuration

Set these two paths near the top of `swat_consistency_check.py`:

```python
REPO_A = "/path/to/Python-Model-Development-Simulation/version_1"
REPO_C = "/path/to/Python-Model-Validation/src"
```

### Run

```bash
JAX_PLATFORMS=cpu python swat_consistency_check.py
```

Prints the per-scenario T(14 days) values and max pointwise residuals,
and writes three PNGs into `/tmp/swat_consistency_plots/`:

- `set_A.png` — healthy basin
- `set_B.png` — amplitude collapse
- `set_D.png` — phase shift

Each plot has two panels:
- **Top**: T(t) over 14 days. Solid blue = 7-state form. Dashed
  orange = 4-state form. The two lines overlap.
- **Bottom**: pointwise residual |T_A − T_C| on a log scale, showing
  the difference stays at floating-point precision throughout.

## Why this exists

After the V_h-anabolic structural fix landed in Repo A (PR #11), an
audit raised the question: are the two downstream vendorings
(validation copy in Repo C, optimal-control copy in OT-Control)
actually consistent with the master copy, or did they drift during
re-vendoring? The function signatures look different — 7-state vs
4-state — and that visual difference can be mistaken for a
model-level discrepancy.

This script answers the question empirically: feed both forms the
same V_h, V_n, V_c values, integrate both with `diffrax.Tsit5`,
compare. The trajectories overlap to floating-point precision, so
the apparent signature difference is presentation only.

## Implication for the SMC²-MPC port

The two forms are deliberately split for different downstream
consumers:

| form | shape | callers |
|---|---|---|
| 7-state estimation form | `drift(y[7], t, params_array, PI)` — V_h, V_n in state vector as per-subject constants | SMC² posterior estimation (Repo A's `_dynamics.py`, `simulation.py`, `estimation.py`) |
| 4-state control form | `swat_drift(t, x[4], u[3], params_dict)` — V_h, V_n, V_c as exogenous controls | OT-Control engine + Repo C validation tests + the planned SMC²-MPC controller |

The SMC²-MPC port should import:
- 7-state form from Repo A's `models.swat._dynamics` for the
  filter side (the SMC² propagate / log-density evaluator),
- 4-state form from Repo C's `model_validation.models.swat.vendored_dynamics`
  for the controller side (the MPC cost / horizon roll-out).

Both must match Repo A's `swat-validated-<date>-<sha>` tag (see
Repo C's `snapshots/manifest.json`).

## Artefacts in this folder

- `swat_consistency_check.py` — the verification script.
- `README.md` — this file.

(The PNG outputs are written to `/tmp/swat_consistency_plots/` at
runtime, not committed — re-run the script to regenerate.)
