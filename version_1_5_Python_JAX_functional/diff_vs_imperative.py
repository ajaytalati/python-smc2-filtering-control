"""Numerical equivalence test: functional rewrite vs imperative source.

Runs side-by-side on the same RNG keys + inputs and asserts that the
new (purely-functional, NamedTuple, jax.lax.scan / jax.vmap)
implementation matches the old (dict-based, Python-for-loop) source
to within tight tolerances.

Run:
    cd version_1_5_Python_JAX_functional
    JAX_ENABLE_X64=True PYTHONPATH=.:..:../version_1_5_Python_JAX \
        python diff_vs_imperative.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import sys
import numpy as np
import jax
import jax.numpy as jnp


def _add_path(p):
    if p not in sys.path:
        sys.path.insert(0, p)


# We need both module trees on sys.path. The functional rewrite lives at
# `<repo>/version_1_5_Python_JAX_functional/models/fsa_high_res/` and
# the imperative source at `<repo>/version_1_5_Python_JAX/models/fsa_high_res/`.
# Both are accessed via `from models.fsa_high_res import …` — we'd
# normally rename one but loading both via importlib bypasses that.

import importlib.util


def _load_pkg(label, root_dir):
    """Load `models.fsa_high_res` from a given root_dir, register
    under a synthetic top-level package name `label` so we can import
    both side-by-side.
    """
    import types
    pkg = types.ModuleType(label)
    pkg.__path__ = [root_dir]
    sys.modules[label] = pkg

    models_pkg_init = os.path.join(root_dir, 'models', '__init__.py')
    fsa_pkg_init    = os.path.join(root_dir, 'models', 'fsa_high_res', '__init__.py')
    if not os.path.exists(models_pkg_init):
        # Imperative source has no __init__.py at models/, but the
        # functional one does — for the import to work we need an
        # empty package marker.
        with open(models_pkg_init, 'a'):
            pass

    # Bind 'models' under our label so 'from models.fsa_high_res import X'
    # inside the loaded modules resolves to this tree.
    models_mod = types.ModuleType('models')
    models_mod.__path__ = [os.path.join(root_dir, 'models')]
    sys.modules['models'] = models_mod

    fsa_pkg = types.ModuleType('models.fsa_high_res')
    fsa_pkg.__path__ = [os.path.join(root_dir, 'models', 'fsa_high_res')]
    sys.modules['models.fsa_high_res'] = fsa_pkg

    submodules = ['_dynamics', 'simulation', '_plant', 'estimation']
    out = {}
    for sm in submodules:
        spec = importlib.util.spec_from_file_location(
            f'models.fsa_high_res.{sm}',
            os.path.join(root_dir, 'models', 'fsa_high_res', sm + '.py'),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        out[sm] = mod
    return out


REPO = '/home/ajay/Repos/python-smc2-filtering-control'
NEW_ROOT = os.path.join(REPO, 'version_1_5_Python_JAX_functional')
OLD_ROOT = os.path.join(REPO, 'version_1_5_Python_JAX')


def main():
    print("=" * 72)
    print("Loading NEW (functional) implementation")
    print("=" * 72)
    new = _load_pkg('new_pkg', NEW_ROOT)
    new_dyn = new['_dynamics']
    new_sim = new['simulation']
    new_plant = new['_plant']
    new_est = new['estimation']

    print("=" * 72)
    print("Loading OLD (imperative) implementation")
    print("=" * 72)
    # Reset top-level 'models' caches so the second load picks up the
    # imperative tree.
    for k in list(sys.modules.keys()):
        if k.startswith('models'):
            del sys.modules[k]
    old = _load_pkg('old_pkg', OLD_ROOT)
    old_dyn = old['_dynamics']
    old_sim = old['simulation']
    old_plant = old['_plant']
    old_est = old['estimation']

    print()
    print("=" * 72)
    print("TEST 1 — drift_jax / diffusion_state_dep equivalence at TRUTH")
    print("=" * 72)

    y_test = jnp.array([0.05, 0.30, 0.10], dtype=jnp.float64)
    Phi_t = 1.0

    # OLD: dict-based params
    old_params_v1 = old_dyn.TRUTH_PARAMS
    # NEW: NamedTuple params
    new_params_v1 = new_dyn.TRUTH_PARAMS_V1

    d_old = old_dyn.drift_jax(y_test, old_params_v1, Phi_t)
    d_new = new_dyn.drift_jax(y_test, new_params_v1, Phi_t)
    diff = float(jnp.max(jnp.abs(d_old - d_new)))
    print(f"  drift_jax max abs diff: {diff:.2e}")
    assert diff < 1e-15, f"drift_jax mismatch: {diff}"

    s_old = old_dyn.diffusion_state_dep(y_test, old_params_v1)
    s_new = new_dyn.diffusion_state_dep(y_test, new_params_v1)
    diff = float(jnp.max(jnp.abs(s_old - s_new)))
    print(f"  diffusion_state_dep max abs diff: {diff:.2e}")
    assert diff < 1e-15, f"diffusion_state_dep mismatch: {diff}"

    print()
    print("=" * 72)
    print("TEST 2 — plant_step equivalence (one EM step under truth)")
    print("=" * 72)

    key0 = jax.random.PRNGKey(7)
    sde_key, obs_key = jax.random.split(key0)
    sde_noise = jax.random.normal(sde_key, (3,), dtype=jnp.float64)
    obs_noise = jax.random.normal(obs_key, (3,), dtype=jnp.float64)
    dt = 1.0 / 24.0

    # OLD: dict params, dict obs return
    old_state0 = old_plant.init_plant_state()
    old_next, old_obs = old_plant.plant_step(
        old_state0, Phi_t, old_sim.DEFAULT_PARAMS, dt,
        sde_noise=sde_noise, obs_noise=obs_noise,
    )

    # NEW: NamedTuple params, NamedTuple obs return
    new_state0 = new_plant.init_plant_state()
    new_next, new_obs = new_plant.plant_step(
        new_state0, Phi_t, new_sim.DEFAULT_PARAMS, dt,
        sde_noise=sde_noise, obs_noise=obs_noise,
    )

    bfa_diff = float(jnp.max(jnp.abs(old_next.bfa - new_next.bfa)))
    print(f"  next_state.bfa max abs diff: {bfa_diff:.2e}")
    assert bfa_diff < 1e-15, f"plant_step bfa mismatch: {bfa_diff}"

    obs_diff = max(
        abs(float(old_obs['obs_B']) - float(new_obs.obs_B)),
        abs(float(old_obs['obs_F']) - float(new_obs.obs_F)),
        abs(float(old_obs['obs_A']) - float(new_obs.obs_A)),
    )
    print(f"  obs (B, F, A) max abs diff: {obs_diff:.2e}")
    assert obs_diff < 1e-15, f"plant_step obs mismatch: {obs_diff}"

    print()
    print("=" * 72)
    print("TEST 3 — plant_rollout equivalence (n=24 bins, one day)")
    print("=" * 72)

    n = 24
    Phi_subdaily = np.full(n, 1.0)
    key_roll = jax.random.PRNGKey(11)

    old_res = old_plant.plant_rollout(
        old_plant.init_plant_state(),
        Phi_subdaily, old_sim.DEFAULT_PARAMS, dt, key_roll,
    )
    new_res = new_plant.plant_rollout(
        new_plant.init_plant_state(),
        Phi_subdaily, new_sim.DEFAULT_PARAMS, dt, key_roll,
    )

    traj_diff = float(np.max(np.abs(np.asarray(old_res['trajectory']) -
                                     np.asarray(new_res.trajectory))))
    print(f"  trajectory max abs diff: {traj_diff:.2e}")
    assert traj_diff < 1e-12, f"trajectory mismatch: {traj_diff}"

    obs_b_diff = float(np.max(np.abs(np.asarray(old_res['obs_B']) -
                                      np.asarray(new_res.obs_B))))
    obs_f_diff = float(np.max(np.abs(np.asarray(old_res['obs_F']) -
                                      np.asarray(new_res.obs_F))))
    obs_a_diff = float(np.max(np.abs(np.asarray(old_res['obs_A']) -
                                      np.asarray(new_res.obs_A))))
    print(f"  obs_B max abs diff: {obs_b_diff:.2e}")
    print(f"  obs_F max abs diff: {obs_f_diff:.2e}")
    print(f"  obs_A max abs diff: {obs_a_diff:.2e}")
    assert max(obs_b_diff, obs_f_diff, obs_a_diff) < 1e-12, "rollout obs mismatch"

    print()
    print("=" * 72)
    print("TEST 4 — obs_log_weight equivalence (M=64 particles)")
    print("=" * 72)

    M = 64
    rng = np.random.default_rng(2026)
    particles = rng.uniform(size=(M, 3)) * np.array([1.0, 0.5, 0.5])
    obs_dict = {'obs_B': 0.07, 'obs_F': 0.31, 'obs_A': 0.11}

    old_lw = old_est.obs_log_weight(
        particles, obs_dict, dict(old_sim.DEFAULT_PARAMS),
    )
    # NEW: Obs NamedTuple, ParamsV15 NamedTuple
    new_obs_t = new_sim.Obs(**obs_dict)
    new_lw = new_est.obs_log_weight(
        particles, new_obs_t, new_sim.DEFAULT_PARAMS,
    )
    diff = float(np.max(np.abs(np.asarray(old_lw) - np.asarray(new_lw))))
    print(f"  obs_log_weight max abs diff: {diff:.2e}")
    assert diff < 1e-12, f"obs_log_weight mismatch: {diff}"

    print()
    print("=" * 72)
    print("TEST 5 — propagate equivalence (M=64 particles, one EM step)")
    print("=" * 72)

    key_prop = jax.random.PRNGKey(31)
    parts0 = rng.uniform(size=(M, 3)) * np.array([1.0, 0.5, 0.5])
    parts0 = parts0.astype(np.float64)
    Phi_t = 1.0
    dt = 1.0 / 24.0

    old_p = old_est.propagate(
        parts0, Phi_t, dict(old_sim.DEFAULT_PARAMS), dt, key_prop,
    )
    new_p = new_est.propagate(
        parts0, Phi_t, new_sim.DEFAULT_PARAMS, dt, key_prop,
    )
    diff = float(np.max(np.abs(np.asarray(old_p) - np.asarray(new_p))))
    print(f"  propagate max abs diff: {diff:.2e}")
    assert diff < 1e-12, f"propagate mismatch: {diff}"

    print()
    print("=" * 72)
    print("ALL TESTS PASSED ✓")
    print("=" * 72)


if __name__ == '__main__':
    main()
