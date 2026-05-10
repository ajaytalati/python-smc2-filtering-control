# Functional rewrite of the smc2fc framework

> Archived from plan mode: 2026-05-08 20:16.

## Context

`smc2fc/` is the engine of the project — the outer SMC² kernel that powers
both filter-side estimation and control-as-inference. ~5,600 lines across
30 Python files in 6 subpackages. It is consumed by every `models/<m>/`
directory and every `tools/bench_*` driver.

The recent v1.5 model rewrite (`version_1_5_Python_JAX_functional/`) showed
that a purely-functional rewrite plus Google-style docstrings is
straightforward when the source is already JAX-first. The user has now
asked for the same treatment of `smc2fc` itself, in a sandbox at
`smc2fc_Jax_functional/`. The end goal is a LEAN4 spec port of the whole
stack; this rewrite is the stepping-stone.

**Audit headline:** the framework is **~85% already functionally pure**.
Only **35 `for`/`while` statements** in the entire codebase, and most are
benign (build-then-return helpers, on-host adaptive control flow, post-
compute numpy cleanup, or matplotlib plotting). Mutation sites: 8 total,
all in localised buffer-build patterns. Public dataclasses are already
`@dataclass(frozen=True)`. Three non-frozen dataclasses in
`core/config.py` are pure config objects, never mutated.

The work is therefore not a rewrite-from-scratch but a focused tightening
plus exhaustive equivalence testing.

## Audit summary (concrete inventory)

Catalogued by an Explore agent. The full inventory is reproduced as a
table below; one row per imperative spot.

| File | Line | Construct | Category | Action |
|---|---|---|---|---|
| `core/config.py` | 11 | `@dataclass class SMCConfig` (non-frozen) | freeze | freeze |
| `core/config.py` | 77 | `@dataclass class RollingConfig` (non-frozen) | freeze | freeze |
| `core/config.py` | 88 | `@dataclass class MissingDataConfig` (non-frozen) | freeze | freeze |
| `core/tempered_smc.py` | 45 | k-means seeding `for k in range(1, K)` + `centres[k]=` | (b) build-helper | replace with comprehension + `np.stack` |
| `core/tempered_smc.py` | 52 | k-means iteration `for _ in range(n_iter)` | (b) | wrap in `functools.reduce` over rebinds |
| `core/tempered_smc.py` | 89 | MoG-component fit `for k in range(K)` + 3 array writes | (b) | comprehensions then `np.stack` |
| `core/tempered_smc.py` | 170 | adaptive tempering `while float(state.tempering_param) < 1.0:` | (d) on-host control | keep as-is — inherently sequential |
| `core/tempered_smc.py` | 204 | `diag_buf.append({...})` | (d) deferred print buffer | replace with tuple-accumulation |
| `core/tempered_smc.py` | 216 | `for i, d in enumerate(diag_buf):` (drain after) | (d) | comprehension-driven print |
| `core/tempered_smc.py` | 421 | second adaptive `while` (bridge variant) | (d) | keep as-is |
| `core/tempered_smc.py` | 456 | `diag_buf.append({...})` (bridge variant) | (d) | tuple-accumulation |
| `core/tempered_smc.py` | 471 | second drain loop | (d) | comprehension |
| `core/sampling.py` | 23 | `for i in range(n_dim): arr.at[:,i].set(...)` | (a) JAX-incompatible-ish | vectorise via `jax.vmap` |
| `filtering/gk_dpf_v3_lite.py` | 126-128, 205-207, 220-222, 248, 299, 463, 568, 588, 628, 684 | bound-clipping `for i, (lo, hi) in enumerate(bounds)` + `.at[:, i].set(jnp.clip(...))` (10 sites) | (b) JAX-functional already | vectorise with `jnp.clip(particles, lo_arr, hi_arr)` |
| `transforms/unconstrained.py` | 49 | `for i, (name, …) in enumerate(all_config.items())` builds metadata arrays | (b) | comprehension + `np.array` |
| `simulator/sde_solver_diffrax.py` | 60, 66, 128-129, 154-157, 186-189, 191-192 | numpy post-compute cleanup loops over `model.deterministic_indices` and `model.bounds` | (d) | replace with comprehension/np.where |
| `simulator/sde_observations.py` | 44 | `for _ in range(max_iterations)` topological-sort while-list-mutation | (d) | functional sort: `_topo_sorted(channels) -> tuple` |
| `simulator/sde_observations.py` | 48-58 | `for ch in remaining[:]` + `remaining.remove(ch)` | (d) | same |
| `control/tempered_smc_loop.py` | 266 | adaptive tempering `while …` | (d) | keep as-is |
| `control/tempered_smc_loop.py` | 292 | `n_temp += 1` counter | (d) | keep — local |
| `control/diagnostics.py` | 44, 91, 95, 97, 142, 148, 185 | matplotlib loops | (d) | replace with `tuple(map(…))` |
| `control/lqg/controller.py` | 138 | `for _ in range(n_steps): traj.append(x.copy())` | (b) | replace with `jax.lax.scan` or `jnp.stack(comprehension)` |

(a) JAX-incompatible loop that needs `scan`/`vmap` — **1 spot**
(b) Build-then-return helper — **~13 spots** (mostly already functional in semantics; vectorising buys clarity not purity)
(c) Genuine list/dict mutation needing `tuple(map(…))` — **0 spots**
(d) On-host control flow / matplotlib / topological sort — **~21 spots** (declarative refactor; `while` adaptive-tempering loops kept as-is — see Decisions §3)

**Key observations:**

1. **No `object.__setattr__` hacks anywhere.** (The v1.5 model dir was the only offender.)
2. **No module-level mutation.** `JAX_ENABLE_X64` and similar env vars are set in driver scripts, not in the framework.
3. The two `while` loops in `tempered_smc.py` and `tempered_smc_loop.py` are **intentionally on-host**: they adaptively decide tempering λ per step from device→host scalars. Replacing with `jax.lax.while_loop` would require the body to be jittable, which would also require dropping the live diagnostic prints. **Keeping them as-is is the right call** unless the user explicitly wants them inside a jit boundary.

## Decisions (defaults; please confirm)

1. **Sandbox layout**: `smc2fc_Jax_functional/smc2fc/...` — i.e. a top-
   level dir `smc2fc_Jax_functional/` containing a `smc2fc/` subpackage
   that mirrors the source 1-for-1. Drivers consume the new code by
   prepending `smc2fc_Jax_functional/` to `PYTHONPATH`. No git-repo
   creation; this is a directory in the existing repo, matching the
   convention of `version_1_5_Python_JAX_functional/`.

2. **Frozen-dataclass vs NamedTuple policy**: Keep the existing frozen
   dataclasses (`EstimationModel`, `ControlSpec`, `RBFSchedule`,
   `SMCControlConfig`) as-is. They are already immutable and well-suited
   to default arguments / methods / properties. Convert the 3 non-frozen
   `core/config.py` dataclasses to `frozen=True` (no field-shape change).
   Introduce new `typing.NamedTuple` records only where new types are
   needed for accumulator-style tuple records (e.g. `DiagnosticRecord`).

3. **Adaptive `while` loops** (2 sites — `core/tempered_smc.py:170,421`,
   `control/tempered_smc_loop.py:266`): keep as Python on-host loops.
   Their bodies print live diagnostics and call into JIT'd kernels; the
   loop *itself* is sequential by definition (each λ-step depends on the
   ESS of the previous). This is acceptable in functional Python — the
   overall function has pure inputs/outputs (the loop just sequences
   state through pure kernel calls). Document this explicitly in
   Google-style docstrings.

4. **Strictness of "no `for`/`while`"**: comprehensions (list, dict,
   generator, set) ARE allowed — they are declarative expressions, not
   imperative loops. `for`/`while` STATEMENTS at function-body level
   are the targets to eliminate. Two exceptions: the adaptive-tempering
   `while` loops in §3, documented as on-host control flow.

5. **Docstring style**: Google-style throughout (Args, Returns, Raises,
   Note, Example sections). Module-level docstrings explain the
   functional contract of each module. Matches the v1.5 bench rewrite.

6. **Test strategy**: per-module numerical equivalence vs the existing
   imperative source (≤1e-12 max abs diff under same seed/inputs), plus
   integration tests at phase boundaries (the v1.5 functional bench at
   `--smoke` and at T=2 days with bit-identical artefact comparison).
   No mocks; real JAX, real GPU when available.

7. **Migration scope**: rewrite every file in `smc2fc/`. Files that are
   already pure get a Google-docstring + minor cleanup pass (no semantic
   change). Files with imperative content get the substantive refactor.

## Phased execution

Bottom-up: leaves first, framework boundary last. Each phase ends with a
green test gate before the next phase begins. If a phase fails its gate,
roll back and diagnose before continuing.

### Phase 0 — Setup (≈30 min)

1. Create `smc2fc_Jax_functional/smc2fc/` with the same subpackage tree.
2. Verbatim-copy every file from `smc2fc/` to the new tree. (No edits.)
3. Confirm `PYTHONPATH=smc2fc_Jax_functional:.. python -c "import smc2fc"`
   resolves to the new tree.
4. Run the v1.5 functional `diff_vs_imperative.py` against the new
   tree (should pass byte-for-byte, since the new tree is a verbatim
   copy).

**Gate:** `import smc2fc` succeeds; v1.5 diff test passes.

### Phase 1 — Leaves (≈2 h)

Files with no `smc2fc`-internal dependencies:

| File | Lines | Action |
|---|---|---|
| `_likelihood_constants.py` | 20 | Google docstring; no logic change. |
| `__init__.py` | 17 | Google docstring; verify all exports. |
| `transforms/unconstrained.py` | 129 | Refactor `build_transform_arrays` loop (line 49) → comprehensions; Google docstrings. |
| `simulator/sde_observations.py` | 64 | Refactor topological sort (lines 44-58) → functional `_topo_sorted` returning a tuple; Google docstrings. |
| `simulator/sde_solver_diffrax.py` | 194 | Refactor 5 numpy cleanup loops → comprehensions / `np.where`; Google docstrings. |
| `simulator/sde_model.py` | 149 | Already pure; Google docstring pass. |
| `simulator/__init__.py` | 0 | (empty) |

**Tests:**
- `test_unconstrained.py` — for each prior config (lognormal, normal, mixed), verify `unconstrained_to_constrained(theta_unc)` matches the old impl to ≤1e-12.
- `test_topo_sort.py` — for a synthetic channel-DAG, verify the new `_topo_sorted` returns the same order as the old loop.
- `test_sde_solver.py` — for a 3-state SDE, verify the new solver outputs match the old to ≤1e-12.

**Gate:** all Phase 1 tests green.

### Phase 2 — Filtering (≈3 h)

| File | Lines | Action |
|---|---|---|
| `filtering/sinkhorn.py` | 86 | Already pure (uses `jax.lax.fori_loop`); docstring pass. |
| `filtering/_gk_kernel.py` | 364 | Already pure; docstring pass. |
| `filtering/transport_kernel.py` | 75 | Already pure; docstring pass. |
| `filtering/project.py` | 46 | Already pure; docstring pass. |
| `filtering/resample.py` | 102 | Already pure; docstring pass. |
| `filtering/gk_dpf_v3_lite.py` | 709 | Vectorise 10 bound-clipping loops → `jnp.clip(particles, lo_arr, hi_arr)`; Google docstrings on every `def`. **Largest file in framework.** |
| `filtering/__init__.py` | 0 | (empty) |

**Tests:**
- `test_gk_dpf_log_density.py` — call `make_gk_dpf_v3_lite_log_density_compileonce(...)` on the v1.5 estimation model with a synthetic `grid_obs`, verify the log-density at a known particle cloud matches the old impl to ≤1e-12.
- `test_gk_dpf_extract.py` — verify `extract_state_at_step(...)` matches.
- `test_resample.py` — verify OT resample output identical.

**Gate:** all Phase 2 tests green; then run the v1.5 functional bench
in `--smoke` mode (single warmup stride, no filter fire) and confirm it
completes — exercises imports + first-call paths.

### Phase 3 — Core engine (≈4 h)

| File | Lines | Action |
|---|---|---|
| `core/config.py` | 109 | Freeze 3 dataclasses (`SMCConfig`, `RollingConfig`, `MissingDataConfig`); Google docstrings. |
| `core/sampling.py` | 31 | Refactor line-23 `for i in range(n_dim)` → `jax.vmap`; Google docstrings. |
| `core/mass_matrix.py` | 20 | Already pure; docstring pass. |
| `core/sf_bridge.py` | 589 | Already pure (no loops); Google docstrings on every `def`. **Second-largest file in core.** |
| `core/tempered_smc.py` | 483 | Replace k-means + MoG numpy buffer-build loops → comprehensions; replace `diag_buf.append` → tuple-accumulation. Keep the 2 adaptive `while` loops, document as on-host. Google docstrings throughout. |
| `core/jax_native_smc.py` | 424 | Already pure (`lax.scan`-based); Google docstrings. |
| `core/jax_native_smc_nuts.py` | 426 | Already pure; Google docstrings. |
| `core/bench_hmc_vs_nuts.py` | 81 | Tool, not framework code — leave as-is (verbatim copy with Google docstring header) or move to `tools/`. Open question. |
| `core/__init__.py` | 0 | (empty) |

**Tests:**
- `test_smc_config.py` — instantiating frozen `SMCConfig`, verify field
  immutability via `pytest.raises(FrozenInstanceError)` on assignment.
- `test_sf_bridge.py` — for a 5D-Gaussian bridge problem, verify
  `sf_bridge` outputs (KL, sample mean/cov) match old to ≤1e-10.
- `test_tempered_smc_kmeans.py` — synthetic bimodal posterior, verify
  k-means clustering matches the old impl in label assignment + centre
  positions.
- `test_tempered_smc_mog_fit.py` — verify MoG bridge (`mus`, `L_chols`,
  `log_weights`) matches old.
- `test_run_smc_window.py` — full filter-window run with the v1.5 model:
  verify (a) final particle cloud, (b) tempering levels `n_temp`, (c)
  log-marginal-likelihood (if returned) all match old to ≤1e-10.
- `test_jax_native_smc.py` — same for the compile-once native path.

**Gate:** all Phase 3 tests green; then run the v1.5 functional bench
at T=2 days, n_smc=16, k_pf=64. Compare ALL `trajectory.npz` keys to
the baseline already on disk at
`version_1_5_Python_JAX_functional/outputs/full_T2/`. Bit-identical
required (we already verified the imperative-vs-rewrite bench runs as
identical). **This is the make-or-break gate for the whole framework.**

### Phase 4 — Control (≈3 h)

| File | Lines | Action |
|---|---|---|
| `control/control_spec.py` | 93 | Already frozen-dataclass; Google docstrings. |
| `control/config.py` | 40 | Already frozen-dataclass; Google docstrings. |
| `control/rbf_schedules.py` | 74 | Already frozen-dataclass; Google docstrings. |
| `control/calibration.py` | 87 | Already pure; Google docstrings. |
| `control/diagnostics.py` | 196 | Replace 7 matplotlib loops → `tuple(map(…))`; Google docstrings. (matplotlib is intrinsically stateful but loop *statements* removed.) |
| `control/tempered_smc_loop.py` | 339 | Same `diag_buf.append` → tuple pattern; keep adaptive `while`; Google docstrings. |
| `control/lqg/__init__.py` | 58 | Already pure; Google docstrings. |
| `control/lqg/linearize.py` | 62 | Already pure; Google docstrings. |
| `control/lqg/riccati.py` | 115 | Already pure; Google docstrings. |
| `control/lqg/controller.py` | 208 | Refactor `nominal_trajectory` `for + traj.append` (line 138) → `jax.lax.scan` or `jnp.stack(comprehension)`; Google docstrings. |
| `control/__init__.py` | 47 | Google docstrings; verify exports. |

**Tests:**
- `test_rbf_schedule.py` — design matrix + theta-decode round-trip vs old.
- `test_calibration.py` — `build_crn_noise_grids` + `calibrate_beta_max`
  vs old, ≤1e-12.
- `test_tempered_smc_loop.py` — controller loop on the v1.5 control spec,
  verify final `mean_theta` and `n_temp_levels` match old to ≤1e-10.
- `test_lqg_controller.py` — for a 3D LQG problem, verify riccati gain +
  nominal trajectory match old.

**Gate:** all Phase 4 tests green; v1.5 bench T=2 days passes
bit-identical artefact check (re-run the Phase 3 gate with the now-
fully-converted framework).

### Phase 5 — Top-level (≈1 h)

| File | Lines | Action |
|---|---|---|
| `estimation_model.py` | 182 | Already pure (frozen dataclass + properties); Google docstrings. |
| `__init__.py` | 17 | Re-export audit; Google docstring. |

**Tests:**
- `test_estimation_model.py` — instantiation immutability check; property
  values vs hand-computed values for the v1.5 spec.

**Gate:** all tests green; v1.5 bench T=2 days passes; the v1.5
`diff_vs_imperative.py` (model-side test) passes against the new
framework.

### Phase 6 — Integration test (≈1 h, mostly waiting)

Run the **full** v1.5 closed-loop bench at production scale (T=2 days,
n_smc=64, k_pf=128, ctrl_n_smc=64) with seed=42. Compare every
`trajectory.npz` key to the baseline at
`version_1_5_Python_JAX_functional/outputs/full_T2/`. **Bit-identical**
is the success criterion. (We already established that the imperative
and pure-Python rewrites of the bench produce bit-identical output, so
any divergence here is on the framework side.)

If divergence: bisect by phase — re-test Phase 3 alone, Phase 4 alone,
etc. Each phase has its own bit-identical gate, so the regression must
have slipped through one of those gates.

**Gate:** bench runs end-to-end, every `.npz` key bit-identical.

### Phase 7 — Documentation polish (≈1 h)

- Pass through every file checking that:
  - Module docstring explains the functional contract.
  - Every `def` has Google-style `Args`/`Returns`/`Raises`.
  - Type hints on every function signature.
  - No remaining `for`/`while` STATEMENTS (only comprehensions, plus the
    documented adaptive-tempering `while` loops with explanatory
    comments).
- Cross-link related modules in docstrings (e.g. `core/tempered_smc.py`
  references `core/sf_bridge.py` for the bridge-proposal call).
- Verify no `TODO`/`XXX`/`FIXME` markers introduced in the new tree.

**Gate:** `grep -rn '^[[:space:]]*for [a-zA-Z_]' smc2fc_Jax_functional/`
returns only the documented adaptive-tempering loops.

## Test infrastructure

A new `smc2fc_Jax_functional/tests/` directory holds:

```
tests/
    __init__.py              (empty)
    conftest.py              (pytest fixtures: rng, v1.5 EstimationModel)
    helpers.py               (load_old_smc2fc, load_new_smc2fc — sys.path tricks)
    test_unconstrained.py
    test_topo_sort.py
    test_sde_solver.py
    test_gk_dpf_log_density.py
    test_gk_dpf_extract.py
    test_resample.py
    test_smc_config.py
    test_sf_bridge.py
    test_tempered_smc_kmeans.py
    test_tempered_smc_mog_fit.py
    test_run_smc_window.py
    test_jax_native_smc.py
    test_rbf_schedule.py
    test_calibration.py
    test_tempered_smc_loop.py
    test_lqg_controller.py
    test_estimation_model.py
    test_v15_bench_artefacts.py    (the integration gate — runs bench, compares .npz)
```

Run order: `JAX_ENABLE_X64=True PYTHONPATH=.:.. pytest tests/ -v`.

The `helpers.py` module uses the `_load_pkg` trick from
`version_1_5_Python_JAX_functional/diff_vs_imperative.py` to import both
`smc2fc` (old) and `smc2fc` (new) under different synthetic top-level
package names within the same Python process.

## Critical files (paths)

Source (read-only reference):
- `smc2fc/...` (the existing imperative tree)

Target (new):
- `smc2fc_Jax_functional/smc2fc/...` (mirrored tree)
- `smc2fc_Jax_functional/tests/...`

Existing utilities reused:
- `version_1_5_Python_JAX_functional/diff_vs_imperative.py` → template
  for the `_load_pkg` trick.
- `version_1_5_Python_JAX_functional/tools/bench_smc_full_mpc_fsa_v15.py`
  → integration-gate driver.
- `version_1_5_Python_JAX_functional/outputs/full_T2/trajectory.npz` →
  bit-identical artefact baseline.

## Risk register

1. **Silent semantic drift**: vectorising a `for + .at[i].set()` loop is
   bit-equivalent in pure float64, but rounding differences may surface
   at fp32 (the project's hot-loop convention). Mitigation: every
   equivalence test runs at fp64; if a fp32 path is touched, an
   additional fp32 test pins the result.

2. **`sys.path` collisions during testing**: importing both `smc2fc`
   (old) and `smc2fc_Jax_functional/smc2fc` (new) in the same process
   is fragile. Mitigation: use the proven `_load_pkg(label, root_dir)`
   pattern from the v1.5 diff test; isolate each import behind a
   synthetic top-level name; clear `sys.modules` between loads.

3. **Adaptive-tempering reproducibility**: the on-host `while` loops
   read `float(state.tempering_param)` to break — this device→host
   transfer is non-deterministic in async-dispatch JAX if not awaited.
   Mitigation: existing code already calls `float(...)` which forces
   sync; preserve this behaviour.

4. **Performance regression**: vectorised `jnp.clip(particles, lo, hi)`
   may compile to different XLA than the loop-with-`.at[]` form. Per
   CLAUDE.md ("hot inner loops in fp32") this is the GPU-saturation
   path. Mitigation: at the integration gate, also record wall-clock
   per stride; flag any stride that's >10% slower than the baseline.

5. **Time cost**: total ≈14 h of focused work. The largest individual
   files are `gk_dpf_v3_lite.py` (709 lines, ~3 h) and
   `core/tempered_smc.py` (483 lines, ~2 h). Phases are ordered so a
   pause after Phase 3 leaves the framework in a half-converted but
   fully-tested state.

## Open questions for the user

Before kicking off Phase 0, I would like to confirm:

1. **Sandbox layout** — is `smc2fc_Jax_functional/smc2fc/` (subpackage
   nested one level under the sandbox dir) the right shape, or do you
   want `smc2fc_Jax_functional/` to BE the package root (importable as
   `import smc2fc_Jax_functional`)? The former preserves `import smc2fc`
   in driver code; the latter is more explicit about which framework is
   loaded.

2. **`bench_hmc_vs_nuts.py`** — this file lives under `smc2fc/core/` but
   looks like a benchmarking *tool* not framework code. Move it to
   `tools/` in the sandbox, or leave at `core/`?

3. **NUTS path** — `core/jax_native_smc_nuts.py` exists alongside the
   HMC variant. Both are already pure. Are both consumed in production,
   or is NUTS deprecated? (Affects whether to test it against an
   imperative baseline that may already be unused.)

4. **Performance budget** — strict bit-identical (≤1e-12) is the
   default. Any module where you'd accept a slightly higher tolerance
   (e.g. ≤1e-8) in exchange for a cleaner refactor?

5. **Adaptive-tempering `while` loops** (3 sites) — keep as on-host
   Python with documented justification, or you want them inside
   `jax.lax.while_loop` (loses live diagnostic prints, gains pure-JAX
   compilation)?

6. **Time horizon** — the 14-hour estimate is for one focused pass. Do
   you want me to do this all in one session, or break at phase
   boundaries for review?
