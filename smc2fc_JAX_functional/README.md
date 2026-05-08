# smc2fc_JAX_functional

Purely-functional rewrite of the `smc2fc` framework, byte-identical
in numerical output to the imperative source.

## Layout

```
smc2fc_JAX_functional/
    smc2fc/               # the framework (drop-in replacement)
        core/             # outer SMC² engine
        filtering/        # inner GK-DPF v3-lite
        control/          # controller side (incl. LQG)
        simulator/        # generic SDE solver + obs channels
        transforms/       # unconstrained ↔ constrained
        estimation_model.py
        _likelihood_constants.py
    tools/
        bench_hmc_vs_nuts.py   # moved out of smc2fc/core/ (not used)
    README.md (this file)
```

## Use

```bash
JAX_ENABLE_X64=True \
PYTHONPATH=smc2fc_JAX_functional:.. \
python tools/bench_smc_full_mpc_fsa_v15.py …
```

`PYTHONPATH=smc2fc_JAX_functional:…` puts the new tree first; `import
smc2fc` then resolves to this rewrite.

## What changed

The framework was already ~85% functionally pure — only 35
`for`/`while` statements across 5,600 lines, public dataclasses
already `@dataclass(frozen=True)`. The rewrite tightened the
remaining 15%:

| Change | Where |
|---|---|
| 10 bound-clipping `for + .at[i].set()` loops → `jnp.clip(arr, lo, hi)` | `filtering/gk_dpf_v3_lite.py` |
| K-means seeding/iteration imperative loops → recursive helpers | `core/tempered_smc.py` |
| MoG-component imperative fit loop → comprehension + `np.stack` | `core/tempered_smc.py` |
| `diag_buf` drain `for` loops → `tuple(map(...))` | `core/tempered_smc.py` |
| Per-dim init-particle for-loop → vectorised `jnp.where`-style | `core/sampling.py` |
| Unconstrained-transform-array build loop → comprehensions | `transforms/unconstrained.py` |
| Topological-sort `for + .remove()` → recursive `_topo_sorted_channels` | `simulator/sde_observations.py` |
| Numpy post-compute cleanup loops → vectorised `np.clip` | `simulator/sde_solver_diffrax.py` |
| Inside-JIT `for idx in det_idxs` → `functools.reduce` | `simulator/sde_solver_diffrax.py` |
| LQG `for + traj.append` → `functools.reduce` | `control/lqg/controller.py` |
| 7 matplotlib `for` loops → `tuple(map(...))` | `control/diagnostics.py` |
| `acceptance_gates` `for + out[name] = ...` → `functools.reduce` | `control/diagnostics.py` |
| 3 non-frozen `@dataclass` → frozen | `core/config.py` |

## What did NOT change

* The 3 adaptive-tempering `while` loops in `core/tempered_smc.py`
  and `control/tempered_smc_loop.py`. These are documented exceptions:
  they read a device→host scalar each iteration to decide
  termination, print live diagnostics, and call into JIT'd kernels.
  Migrating to `jax.lax.while_loop` would forbid the live prints.
* All `jax.lax.scan` / `jax.lax.fori_loop` / `jax.vmap` patterns
  already in the source.
* The numerical recipes (k-means, Ledoit-Wolf shrinkage, Sinkhorn,
  Bures-Wasserstein interpolation, Liu-West correction). Bit-for-bit
  preservation was the success criterion.

## Verification

The v1.5 closed-loop SMC²-MPC bench (4 strides, T=2 days, seed=42)
was used as the integration gate. Every `trajectory.npz` key (13
total — MPC trajectory, baseline, accumulated obs, posterior cloud,
window mask, etc.) compared bit-identical to the artefact baseline
at `version_1_5_Python_JAX_functional/outputs/full_T2/`.

Per-phase bit-identical checks were green throughout.

## Functional purity invariants

* **Zero `for`/`while` STATEMENTS in framework body** apart from the
  3 documented adaptive-tempering `while` loops.
* **Zero non-frozen dataclasses.**
* **Zero `object.__setattr__` hacks.**
* **Zero module-level state mutation** (env vars, prints, file I/O at
  import time) — except the `os.environ.setdefault(...)` /
  `jax.config.update(...)` pair at the top of
  `simulator/sde_solver_diffrax.py`, which is defensive driver-side
  setup matching the original module's behaviour.
