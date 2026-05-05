"""Bench-level smoke tests for the Stage 1 / 2 / 3 FSA-v5 bench drivers.

These do NOT execute the full benches (each ~10-60 min); they only
exercise the import surface, CLI-flag parsing, and the spec-construction
helpers so typos / refactor regressions are caught fast (< 30 s).

Run from the repo root:
    cd version_3 && PYTHONPATH=.:.. pytest tests/test_fsa_v5_bench_smoke.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import pytest

# Force CPU + X64 BEFORE importing JAX or the bench modules.
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

# Repo root + version_3 dir on sys.path so `from version_3.models.fsa_v5...`
# inside the bench drivers resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_V3_DIR = _REPO_ROOT / "version_3"
for p in (_REPO_ROOT, _V3_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_bench(filename: str):
    """Load a bench module from `tools/<filename>` without executing main()."""
    full = _V3_DIR / "tools" / filename
    spec = importlib.util.spec_from_file_location(full.stem, full)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Stage 1 ──

def test_stage1_filter_only_imports_clean():
    """Stage 1 driver imports without error and exposes expected symbols."""
    mod = _load_bench("bench_smc_filter_only_fsa_v5.py")
    for sym in ('main', '_simulate_synthetic_full', 'extract_window',
                '_allocate_run_dir', 'TRAINED_ATHLETE_STATE',
                'DEFAULT_PHI_B', 'DEFAULT_PHI_S',
                'WINDOW_BINS', 'STRIDE_BINS', 'N_DAYS_TOTAL'):
        assert hasattr(mod, sym), f"Stage 1 driver missing {sym}"


def test_stage1_run_dir_allocator():
    """`_allocate_run_dir` creates next runNN_<tag>/ folder."""
    import tempfile
    mod = _load_bench("bench_smc_filter_only_fsa_v5.py")
    with tempfile.TemporaryDirectory() as tmp:
        out, n = mod._allocate_run_dir(Path(tmp), "smoke_test")
        assert out.exists() and out.is_dir()
        assert out.name.startswith("run")
        assert "smoke_test" in out.name
        assert n == 1
        # Second call should bump to 02
        out2, n2 = mod._allocate_run_dir(Path(tmp), "smoke_test")
        assert n2 == 2 and out2 != out


# ── Stage 2 ──

def test_stage2_controller_only_imports_clean():
    """Stage 2 driver imports + advertises soft/hard/gradient_ot variants."""
    mod = _load_bench("bench_controller_only_fsa_v5.py")
    for sym in ('main', '_build_spec_for_cost_variant',
                '_build_cost_chance_constrained',
                'SCENARIOS', 'TRAINED_ATHLETE_STATE',
                'DEFAULT_BETA', 'DEFAULT_ALPHA', 'DEFAULT_A_TARGET'):
        assert hasattr(mod, sym), f"Stage 2 driver missing {sym}"
    assert set(mod.SCENARIOS) == {'healthy', 'sedentary', 'overtrained'}


@pytest.mark.parametrize("cost_kind", ['soft', 'hard'])
def test_stage2_spec_builds_for_each_cost(cost_kind):
    """Both soft and hard control specs build + the cost_fn returns a finite scalar."""
    mod = _load_bench("bench_controller_only_fsa_v5.py")
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5

    spec, overrides = mod._build_spec_for_cost_variant(
        cost=cost_kind, n_steps=14 * 96, n_anchors=8,
        init_state=mod.TRAINED_ATHLETE_STATE.copy(),
        truth_params=dict(DEFAULT_PARAMS_V5),
        dt=mod.DT, alpha=0.05, A_target=2.0, beta=50.0,
        lam_phi=0.1, lam_chance=100.0,
    )
    assert spec.theta_dim == 16   # 2 * n_anchors
    assert spec.n_steps == 14 * 96

    # Cost at theta=0 should be finite. With trained-athlete init +
    # constant Phi=(0.30, 0.30) we expect near-zero violations and
    # mean_A_integral ~ 11 over 14 days, so cost ~ -11.
    val = float(spec.cost_fn(jnp.zeros(spec.theta_dim)))
    assert np.isfinite(val), f"cost_fn(theta=0) returned non-finite: {val}"
    assert val < 0.0, (
        f"cost at trained-athlete + healthy Phi should be negative "
        f"(controller wants to maximise A_integral); got {val}")
    # Hard variant must skip HMC inside the tempered-SMC loop.
    if cost_kind == 'hard':
        assert overrides == {'num_mcmc_steps': 0}, (
            f"hard variant must override num_mcmc_steps=0 to avoid running "
            f"HMC over a non-differentiable indicator; got {overrides}")
    else:
        assert overrides == {}


def test_stage2_schedule_decoder_shape():
    """schedule_from_theta produces (n_steps, 2) bimodal output."""
    mod = _load_bench("bench_controller_only_fsa_v5.py")
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5

    n_steps = 14 * 96
    spec, _ = mod._build_spec_for_cost_variant(
        cost='soft', n_steps=n_steps, n_anchors=8,
        init_state=mod.TRAINED_ATHLETE_STATE.copy(),
        truth_params=dict(DEFAULT_PARAMS_V5),
        dt=mod.DT, alpha=0.05, A_target=2.0, beta=50.0,
        lam_phi=0.1, lam_chance=100.0,
    )
    sched = spec.schedule_from_theta(jnp.zeros(spec.theta_dim))
    assert sched.shape == (n_steps, 2), f"unexpected schedule shape {sched.shape}"
    assert float(sched.min()) >= 0.0
    assert float(sched.max()) <= 3.0   # Phi_max bound


# ── Stage 3 ──

def test_stage3_full_mpc_imports_clean():
    """Stage 3 driver imports + has the posterior-aware cost wrapper."""
    mod = _load_bench("bench_smc_full_mpc_fsa_v5.py")
    for sym in ('main', '_build_spec_for_cost_variant_posterior',
                '_build_cost_chance_constrained_posterior',
                '_posterior_to_theta_stacked',
                'extract_window', 'SCENARIOS', 'TRAINED_ATHLETE_STATE'):
        assert hasattr(mod, sym), f"Stage 3 driver missing {sym}"


def test_stage3_posterior_to_theta_stacked():
    """`_posterior_to_theta_stacked` produces a dict-of-arrays keyed by param names."""
    mod = _load_bench("bench_smc_full_mpc_fsa_v5.py")
    from version_3.models.fsa_v5.estimation import HIGH_RES_FSA_V5_ESTIMATION

    em = HIGH_RES_FSA_V5_ESTIMATION
    names = list(em.all_names)
    n_smc, n_params = 7, len(names)
    fake_post = np.random.randn(n_smc, n_params).astype(np.float64)
    stacked = mod._posterior_to_theta_stacked(fake_post, names)
    assert set(stacked) == set(names)
    for k, v in stacked.items():
        assert v.shape == (n_smc,), f"{k} has wrong shape {v.shape}"


@pytest.mark.parametrize("cost_kind", ['soft', 'hard'])
def test_stage3_spec_builds_with_posterior_cloud(cost_kind):
    """Stage 3 spec builds from a posterior cloud + cost_fn evaluates."""
    mod = _load_bench("bench_smc_full_mpc_fsa_v5.py")
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5
    from version_3.models.fsa_v5.estimation import HIGH_RES_FSA_V5_ESTIMATION

    em = HIGH_RES_FSA_V5_ESTIMATION
    names = list(em.all_names)
    truth = dict(DEFAULT_PARAMS_V5)
    n_smc = 5
    posterior = np.array([[truth[n] for n in names]] * n_smc, dtype=np.float64)
    theta_stacked = mod._posterior_to_theta_stacked(posterior, names)
    weights = jnp.full((n_smc,), 1.0 / n_smc, dtype=jnp.float64)

    spec, overrides = mod._build_spec_for_cost_variant_posterior(
        cost=cost_kind, n_steps=14 * 96, n_anchors=8,
        init_state=mod.TRAINED_ATHLETE_STATE.copy(),
        theta_stacked=theta_stacked, weights=weights,
        dt=mod.DT, alpha=0.05, A_target=2.0, beta=50.0,
        lam_phi=0.1, lam_chance=100.0,
    )
    assert spec.theta_dim == 16
    val = float(spec.cost_fn(jnp.zeros(spec.theta_dim)))
    assert np.isfinite(val) and val < 0.0
    if cost_kind == 'hard':
        assert overrides == {'num_mcmc_steps': 0}


# ── Cross-bench sanity: soft and hard agree on a healthy-island theta ──

def test_soft_and_hard_agree_at_healthy_island():
    """Trained-athlete + Phi=(0.30, 0.30) is well inside the healthy island,
    so the trajectory never violates the basin separatrix. The hard
    indicator is therefore zero everywhere and the soft sigmoid (at
    typical beta) is also ~0. Both variants should return ~the same
    cost in this corner case (effort + -A_integral terms dominate)."""
    mod_s = _load_bench("bench_controller_only_fsa_v5.py")
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5

    spec_soft, _ = mod_s._build_spec_for_cost_variant(
        cost='soft', n_steps=14 * 96, n_anchors=8,
        init_state=mod_s.TRAINED_ATHLETE_STATE.copy(),
        truth_params=dict(DEFAULT_PARAMS_V5),
        dt=mod_s.DT, alpha=0.05, A_target=2.0, beta=50.0,
        lam_phi=0.1, lam_chance=100.0,
    )
    spec_hard, _ = mod_s._build_spec_for_cost_variant(
        cost='hard', n_steps=14 * 96, n_anchors=8,
        init_state=mod_s.TRAINED_ATHLETE_STATE.copy(),
        truth_params=dict(DEFAULT_PARAMS_V5),
        dt=mod_s.DT, alpha=0.05, A_target=2.0, beta=50.0,
        lam_phi=0.1, lam_chance=100.0,
    )
    theta_zero = jnp.zeros(16)
    val_soft = float(spec_soft.cost_fn(theta_zero))
    val_hard = float(spec_hard.cost_fn(theta_zero))
    assert abs(val_soft - val_hard) < 1.0, (
        f"soft and hard variants disagree by > 1.0 on healthy-island "
        f"corner: soft={val_soft}, hard={val_hard}")


if __name__ == '__main__':
    # Allow `python tests/test_fsa_v5_bench_smoke.py` for a quick local run
    test_stage1_filter_only_imports_clean()
    print("PASS: Stage 1 imports")
    test_stage1_run_dir_allocator()
    print("PASS: Stage 1 run-dir allocator")
    test_stage2_controller_only_imports_clean()
    print("PASS: Stage 2 imports")
    for kind in ('soft', 'hard'):
        test_stage2_spec_builds_for_each_cost(kind)
    print("PASS: Stage 2 spec builds (soft/hard)")
    test_stage2_schedule_decoder_shape()
    print("PASS: Stage 2 schedule decoder shape")
    test_stage3_full_mpc_imports_clean()
    print("PASS: Stage 3 imports")
    test_stage3_posterior_to_theta_stacked()
    print("PASS: Stage 3 posterior stacker")
    for kind in ('soft', 'hard'):
        test_stage3_spec_builds_with_posterior_cloud(kind)
    print("PASS: Stage 3 spec builds with posterior")
    test_soft_and_hard_agree_at_healthy_island()
    print("PASS: soft/hard agree at healthy island")
    print("All bench smoke tests passed.")
