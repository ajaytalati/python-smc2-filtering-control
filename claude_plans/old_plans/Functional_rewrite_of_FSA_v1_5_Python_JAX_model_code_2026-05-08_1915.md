# Functional rewrite of FSA v1.5 Python+JAX model code

> Archived from plan mode: 2026-05-08 19:15.

## Context

We're rewriting the FSA high-res model code from
`version_1_5_Python_JAX/models/fsa_high_res/` into a sibling directory
`version_1_5_Python_JAX_functional/models/fsa_high_res/` so that it is
**purely functional**:

- No `for` / `while` loops — use `jax.lax.scan` and `jax.vmap`.
- Total functions — same inputs always give the same outputs.
- No mutation after construction — replace `np.empty` + index assignment
  with `scan` carries; drop `object.__setattr__` hacks.
- **Immutable, typed records** (`typing.NamedTuple`) instead of plain
  Python dicts for params / init-state / obs — gives explicit field
  types that map cleanly to LEAN4 records.

This is a **stepping-stone for a LEAN4 port** of the same model. The
LEAN4 specification will be derived from this functional Python by hand.

The existing code is already mostly functional: `_dynamics.py` is fully
pure, `simulation.py` has no loops, the inner JIT'd cost rollout in
`control.py` already uses `jax.lax.scan`. The work is concentrated in
**5 imperative spots** plus a **dict → NamedTuple migration** across
the public API of the 6 model files.

## Decisions (confirmed with user)

1. **Target path:** `version_1_5_Python_JAX_functional/models/fsa_high_res/`
   — repo convention (plural `models/`, `fsa_high_res/`).
2. **Drop `_traj_sample_fn`** — the `object.__setattr__` hack on the
   frozen `ControlSpec` is dead in v1.5 Python (no consumer).
   `build_control_spec` returns the `ControlSpec` only.
3. **Migrate `params` to NamedTuple** now (and likewise `init_state`,
   `obs`, `pinned_params`). LEAN4-friendly. Boundary conversion to
   `dict` only where the `smc2fc` framework requires it.
4. **Keep** `os.environ.setdefault('JAX_ENABLE_X64', 'True')` at module
   import in `control.py` — driver-resilient.

## Phase 1 audit summary

| File | Imperative content | Action |
|---|---|---|
| `__init__.py` | None — docstring only. | Copy verbatim. |
| `_dynamics.py` | None — already uses `jax.lax.scan` for sub-stepping. | Switch `params: dict` → `params: ParamsV1`. Else copy. |
| `simulation.py` | Module-load env-var parse for `BINS_PER_DAY`. | Keep env-var parse (project convention). Migrate `DEFAULT_PARAMS`, `INIT_STATE`, `PINNED_PARAMS` to NamedTuples; rewrite `params_v15_to_v1`, `fill_pinned`, `sample_obs_bfa` against typed records. |
| `_plant.py` | **`plant_rollout` has a `for k in range(n)` loop** (lines 154-167) writing into `np.empty` buffers. | Replace with `jax.lax.scan`. Switch dict params → NamedTuple. Already returns immutable `PlantState` (NamedTuple). |
| `estimation.py` | **`obs_log_weight` `for m in range(M)` loop** (155-162); **`propagate` `for m in range(M)` loop** (188-197). | Replace both with `jax.vmap`. Switch dict params → NamedTuple. Build dict at framework boundary (`EstimationModel.frozen_params`). |
| `control.py` | (a) `object.__setattr__(spec, '_traj_sample_fn', ...)` on frozen dataclass (line 212). | (a) Drop the setattr; return `ControlSpec` only. (b) Switch dict params → NamedTuple. (c) Keep the `os.environ.setdefault` line. |

## New typed records (added in `simulation.py`)

```python
from typing import NamedTuple
import jax.numpy as jnp

class ParamsV15(NamedTuple):
    """14 dynamics params in the v1.5 basis (B_inf, F_inf) + 3 obs sigmas."""
    tau_B:       float
    tau_F:       float
    B_inf:       float
    F_inf:       float
    epsilon_A:   float
    lambda_A:    float
    mu_0:        float
    mu_B:        float
    mu_F:        float
    mu_FF:       float
    eta:         float
    sigma_B:     float
    sigma_F:     float
    sigma_A:     float
    sigma_B_obs: float
    sigma_F_obs: float
    sigma_A_obs: float

class ParamsV1(NamedTuple):
    """14 dynamics params in the v1 basis (kappa_B, kappa_F) + 3 obs sigmas.
    What `drift_jax` and `diffusion_state_dep` consume."""
    tau_B:       float
    tau_F:       float
    kappa_B:     float
    kappa_F:     float
    epsilon_A:   float
    lambda_A:    float
    mu_0:        float
    mu_B:        float
    mu_F:        float
    mu_FF:       float
    eta:         float
    sigma_B:     float
    sigma_F:     float
    sigma_A:     float
    sigma_B_obs: float
    sigma_F_obs: float
    sigma_A_obs: float

class PinnedParams(NamedTuple):
    tau_B:     float
    eta:       float
    epsilon_A: float
    mu_FF:     float

class InitState(NamedTuple):
    B: float
    F: float
    A: float

class Obs(NamedTuple):
    obs_B: float
    obs_F: float
    obs_A: float
```

`typing.NamedTuple` is a registered JAX pytree, so these flow naturally
through `jit` / `scan` / `vmap`. Field access is `params.kappa_B` (not
`params['kappa_B']`). Floats stay floats; arrays stay arrays.

The constants change shape:
```python
DEFAULT_PARAMS = ParamsV15(tau_B=42.0, tau_F=7.0, B_inf=0.504, ...)
INIT_STATE     = InitState(B=0.05, F=0.30, A=0.10)
PINNED_PARAMS  = PinnedParams(tau_B=42.0, eta=0.20, epsilon_A=0.40, mu_FF=0.40)
```

## Refactor plan, file-by-file

### `__init__.py`
Copy verbatim.

### `_dynamics.py`
- `TRUTH_PARAMS` → `TRUTH_PARAMS_V1: ParamsV1` (NamedTuple instead of dict).
- `drift_jax(y, params: ParamsV1, Phi_t)` — change `params['kappa_B']` →
  `params.kappa_B` and similar (10 sites).
- `diffusion_state_dep(y, params: ParamsV1)` — same field-access change.
- `em_step_substepped` — already pure, just propagates the typed `params`.

### `simulation.py`
- Add the 5 NamedTuple definitions above.
- Build `DEFAULT_PARAMS: ParamsV15`, `INIT_STATE: InitState`,
  `PINNED_PARAMS: PinnedParams` from `_dynamics.TRUTH_PARAMS_V1` at module
  load. Pure expressions — no loops.
- `params_v15_to_v1(p: ParamsV15) -> ParamsV1` — replace `isinstance(p, dict)`
  branch with a single typed signature; the Julia `@match` site collapses
  to one explicit field-by-field constructor.
- `fill_pinned(estimated, pinned=None)` —
  signature `(estimated: <10-field record>, pinned: PinnedParams = PINNED_PARAMS) -> ParamsV15`.
  We can either use a separate `EstimatedDynParams` NamedTuple for the
  10-field input or keep it as a dict at this single boundary
  (the filter feeds floats by name from a JAX array).
  Default proposal: small `EstimatedDynParams` NamedTuple for symmetry.
- `sample_obs_bfa(state, params: ParamsV15, key) -> Obs` — return typed
  record, not dict. Pure function, already loop-free.
- Keep the `os.environ.get('FSA_STEP_MINUTES', '60')` parse + `BINS_PER_DAY`
  + `DT_BIN_DAYS` module-level constants. The parse itself is pure
  (env → int); only the `ValueError` raise is a side effect, which is
  the project's intentional fail-loud behaviour.

### `_plant.py` — replace the for-loop in `plant_rollout`

`PlantState` already a NamedTuple. `_reflect_unit`, `init_plant_state`,
`plant_step` already pure.

```python
def plant_rollout(s0: PlantState, Phi_subdaily, params: ParamsV15,
                  dt: float, key0) -> RolloutResult:
    n = int(len(Phi_subdaily))
    Phi_arr = jnp.asarray(Phi_subdaily, dtype=jnp.float64)

    def step(carry, k):
        state, key = carry
        key, sde_key, obs_key = jax.random.split(key, 3)
        sde_noise = jax.random.normal(sde_key, (3,), dtype=jnp.float64)
        next_state, obs = plant_step(
            state, Phi_arr[k], params, dt,
            sde_noise=sde_noise, obs_noise_key=obs_key,
        )
        emitted = jnp.array([
            next_state.bfa[0], next_state.bfa[1], next_state.bfa[2],
            obs.obs_B, obs.obs_F, obs.obs_A,
        ])
        return (next_state, key), emitted

    (final_state, _), stacked = jax.lax.scan(step, (s0, key0), jnp.arange(n))

    return RolloutResult(
        final_state=final_state,
        trajectory=stacked[:, 0:3],
        obs_B=stacked[:, 3],
        obs_F=stacked[:, 4],
        obs_A=stacked[:, 5],
        Phi=Phi_arr,
    )
```

`RolloutResult` is a NamedTuple replacing the previous return dict.

`plant_step` keeps the existing two-path obs sampling
(`obs_noise` array OR `obs_noise_key` PRNG) but returns an `Obs`
NamedTuple instead of a dict.

### `estimation.py` — replace two for-loops with `jax.vmap`

(1) `obs_log_weight(particles, obs: Obs, params: ParamsV15)`:

```python
def obs_log_weight(particles, obs: Obs, params: ParamsV15):
    parts = jnp.asarray(particles, dtype=jnp.float64)
    return jax.vmap(
        lambda b, f, a: obs_log_weight_one(b, f, a, obs, params)
    )(parts[:, 0], parts[:, 1], parts[:, 2])
```

(2) `propagate(particles, Phi_t, params: ParamsV15, dt, key)`:

```python
def propagate(particles, Phi_t, params: ParamsV15, dt, key):
    M = particles.shape[0]
    params_v1 = params_v15_to_v1(params)
    sqrt_dt = math.sqrt(dt)
    keys = jax.random.split(key, M)

    def one(y, k):
        d = drift_jax(y, params_v1, Phi_t)
        sigma = diffusion_state_dep(y, params_v1)
        noise = jax.random.normal(k, (3,), dtype=jnp.float64)
        y_pred = y + d * dt + sigma * sqrt_dt * noise
        B = jnp.where(y_pred[0] < 0.0, -y_pred[0],
                      jnp.where(y_pred[0] > 1.0, 2.0 - y_pred[0], y_pred[0]))
        F = jnp.abs(y_pred[1])
        A = jnp.abs(y_pred[2])
        return jnp.array([B, F, A])

    return jax.vmap(one)(jnp.asarray(particles, dtype=jnp.float64), keys)
```

The framework callbacks (`_propagate_fn_framework`,
`_diffusion_fn_framework`, `_obs_log_weight_fn_framework`,
`align_obs_fn`, `_shard_init_fn`, `_make_init_state_fn`) already pure;
keep their JAX-array signatures but build `ParamsV15` from the
constrained filter vector inside (replace `_params_v15_from_filter_vector`'s
dict-builder with a `ParamsV15(...)` constructor).

`HIGH_RES_FSA_V15_ESTIMATION = EstimationModel(...)` — this is the
boundary with `smc2fc`. The `frozen_params` field of `EstimationModel`
is typed as `Dict` upstream, so we **convert at the boundary**:

```python
HIGH_RES_FSA_V15_ESTIMATION = EstimationModel(
    ...,
    frozen_params={
        **PINNED_PARAMS._asdict(),
        'sigma_B_obs': SIGMA_B_OBS_FROZEN,
        'sigma_F_obs': SIGMA_F_OBS_FROZEN,
        'sigma_A_obs': SIGMA_A_OBS_FROZEN,
    },
    ...,
)
```

`NamedTuple._asdict()` is built-in. No mutation. `smc2fc` is untouched.

### `control.py` — drop the setattr-hack, switch dict→NamedTuple

- Keep line 23 `os.environ.setdefault('JAX_ENABLE_X64', 'True')` (per
  user decision).
- `build_control_spec(params_v15: ParamsV15 = DEFAULT_PARAMS, init_state: InitState = INIT_STATE, ...)`
  — typed signatures.
- `_make_em_step_fn(params_v15: ParamsV15, ...)` — call
  `params_v15_to_v1(params_v15)` then convert to per-field JAX scalars
  for the closure (same as today, just field-access not dict-access).
- `truth_params={k: float(v) for k, v in p_v15.items()}` becomes
  `truth_params=p_v15._asdict()` — `ControlSpec.truth_params` is typed
  `Dict[str, float]` upstream and stays a dict at the boundary.
- **Remove line 212** `object.__setattr__(spec, '_traj_sample_fn', traj_sample_fn)`.
  Drop `traj_sample_fn` from the function entirely (kept as a closure
  that's never returned). `build_control_spec` returns just the
  `ControlSpec`.
- The inner closures (`schedule_from_theta`, `cost_fn`, `em_step`)
  already use `jax.lax.scan` and are pure — no change.

## Critical files (paths)

Source (read-only reference):
- `version_1_5_Python_JAX/models/fsa_high_res/__init__.py`
- `version_1_5_Python_JAX/models/fsa_high_res/_dynamics.py`
- `version_1_5_Python_JAX/models/fsa_high_res/simulation.py`
- `version_1_5_Python_JAX/models/fsa_high_res/_plant.py`
- `version_1_5_Python_JAX/models/fsa_high_res/estimation.py`
- `version_1_5_Python_JAX/models/fsa_high_res/control.py`

Target (new):
- `version_1_5_Python_JAX_functional/models/__init__.py` (empty)
- `version_1_5_Python_JAX_functional/models/fsa_high_res/__init__.py`
- `version_1_5_Python_JAX_functional/models/fsa_high_res/_dynamics.py`
- `version_1_5_Python_JAX_functional/models/fsa_high_res/simulation.py`
- `version_1_5_Python_JAX_functional/models/fsa_high_res/_plant.py`
- `version_1_5_Python_JAX_functional/models/fsa_high_res/estimation.py`
- `version_1_5_Python_JAX_functional/models/fsa_high_res/control.py`

Existing utilities reused (no duplication):
- `smc2fc.estimation_model.EstimationModel` (frozen dataclass)
- `smc2fc.control.ControlSpec` (frozen dataclass)
- `smc2fc.control.RBFSchedule`
- `smc2fc.control.calibration.build_crn_noise_grids`
- `smc2fc._likelihood_constants.HALF_LOG_2PI`

## Verification plan

Activate env and run from inside the new dir per `CLAUDE.md`:

```bash
conda activate comfyenv
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX_functional
PYTHONPATH=.:.. python -c "from models.fsa_high_res import simulation, _dynamics, _plant, estimation, control; print('imports OK')"
```

1. **Import smoke** — every module imports without error.
2. **`drift_jax` / `diffusion_state_dep` numerical equivalence** —
   small synthetic test: same `(y, params, Phi_t)` between old (dict)
   and new (NamedTuple) implementations agree to ≤1e-15 (only field
   access changed).
3. **`plant_step` numerical equivalence** — same `(state, Phi_t,
   params, dt, sde_noise, obs_noise)` → identical `(next_state, obs)`
   bit-for-bit between old and new.
4. **`plant_rollout` equivalence** — same `key0` and `Phi_subdaily` →
   trajectory and obs arrays agree to ≤1e-12 between old (for-loop) and
   new (`jax.lax.scan`). Per-step RNG split sequence is byte-identical.
5. **`obs_log_weight` and `propagate` equivalence** — same inputs →
   same outputs between old (for-loop, numpy) and new (`jax.vmap`,
   jnp). Tolerance ≤1e-12.
6. **End-to-end smoke** — minimal closed-loop run of
   `bench_smc_full_mpc_fsa_v15.py` adapted to import from the new
   module path, T_total=2 days, confirm completion + posterior
   summary numbers match the old at ≤1e-10.

If any test fails, the migration is treated as broken — no merging
until equivalence holds.
