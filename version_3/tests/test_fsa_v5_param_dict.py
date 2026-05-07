"""Regression tests for the FSA-v5 parameter dictionaries.

History: until 2026-05-06 the ``DEFAULT_PARAMS`` dict in
``simulation.py`` had two ``'sigma_S'`` keys (state-noise 0.008 and
stress-channel obs noise 4.0). Python silently kept the second, which
meant any code path that read ``params['sigma_S']`` for state-noise
purposes received the obs value. The plant and the SMC² filter
worked around this with hard-coded constants. This was renamed:

  * ``sigma_S``     — Jacobi diffusion scale on Strength state (0.008)
  * ``sigma_S_obs`` — Gaussian noise on the stress channel obs (4.0)

These tests lock in the post-rename invariant so any future edit that
re-introduces the collision (or routes the wrong sigma into either
path) fails loudly.
"""

import math
import pytest


def test_default_params_sigma_S_state_value():
    """``sigma_S`` is unambiguously the state-noise scale post-rename."""
    from version_3.models.fsa_v5 import DEFAULT_PARAMS_V5
    assert DEFAULT_PARAMS_V5['sigma_S'] == 0.008, (
        f"sigma_S must be the state-noise scale (0.008). "
        f"Found {DEFAULT_PARAMS_V5['sigma_S']}. If you've moved the obs "
        f"noise back into this key, the Strength state-noise is now wrong."
    )


def test_default_params_sigma_S_obs_value():
    """The stress-channel obs noise lives under its own key."""
    from version_3.models.fsa_v5 import DEFAULT_PARAMS_V5
    assert DEFAULT_PARAMS_V5['sigma_S_obs'] == 4.0, (
        f"sigma_S_obs must be the stress-channel obs noise (4.0). "
        f"Found {DEFAULT_PARAMS_V5.get('sigma_S_obs')!r}."
    )


def test_default_params_no_duplicate_keys():
    """No dict-literal duplicates anywhere in DEFAULT_PARAMS."""
    import ast
    src_path = 'version_3/models/fsa_v5/simulation.py'
    with open(src_path) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if not (isinstance(tgt, ast.Name) and tgt.id == 'DEFAULT_PARAMS'):
                continue
            d = node.value
            if not isinstance(d, ast.Dict):
                continue
            keys = [k.value for k in d.keys if isinstance(k, ast.Constant)]
            from collections import Counter
            dups = {k: v for k, v in Counter(keys).items() if v > 1}
            assert not dups, (
                f"DEFAULT_PARAMS dict literal has duplicate keys: {dups}. "
                f"This is exactly the bug that motivated the sigma_S rename."
            )
            return
    pytest.fail("Could not locate DEFAULT_PARAMS assignment in simulation.py")


def test_estimation_state_sigmas_are_correct_state_values():
    """Estimation pipeline state-noise constants stay at the canonical values."""
    from version_3.models.fsa_v5.estimation import (
        SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
        SIGMA_A_FROZEN, SIGMA_K_FROZEN,
    )
    assert SIGMA_B_FROZEN == 0.010
    assert SIGMA_S_FROZEN == 0.008
    assert SIGMA_F_FROZEN == 0.012
    assert SIGMA_A_FROZEN == 0.020
    assert SIGMA_K_FROZEN == 0.005


def test_estimation_prior_uses_sigma_S_obs_key():
    """Estimation prior config registers ``sigma_S_obs`` (not ``sigma_S``)
    as the estimated stress-channel obs noise, centred at log(4.0)."""
    from version_3.models.fsa_v5.estimation import PARAM_PRIOR_CONFIG
    assert 'sigma_S_obs' in PARAM_PRIOR_CONFIG, (
        "PARAM_PRIOR_CONFIG must use sigma_S_obs (not sigma_S) for the "
        "stress-channel obs noise. If you've reverted to sigma_S, the "
        "name collision is back."
    )
    assert 'sigma_S' not in PARAM_PRIOR_CONFIG, (
        "sigma_S must NOT appear in PARAM_PRIOR_CONFIG — diffusion sigmas "
        "are FROZEN at canonical values, not estimated. The estimated "
        "obs noise lives under sigma_S_obs."
    )
    family, args = PARAM_PRIOR_CONFIG['sigma_S_obs']
    assert family == 'lognormal'
    log_mean, _ = args
    assert math.isclose(log_mean, math.log(4.0), rel_tol=1e-4)


def test_plant_reads_state_sigmas_from_truth_params():
    """The plant's diffusion vector must come from ``self.truth_params``,
    not hard-coded literals (the historical workaround). Source check."""
    src = open('version_3/models/fsa_v5/_plant.py').read()
    # Either of these spellings is acceptable:
    needed = ["self.truth_params['sigma_B']",
              "self.truth_params[\"sigma_B\"]"]
    assert any(s in src for s in needed), (
        "Plant must read state-noise sigmas from self.truth_params now "
        "that the sigma_S collision has been resolved. The hard-coded "
        "vector workaround should have been removed."
    )


def test_truth_params_v5_has_clean_sigma_S():
    """``TRUTH_PARAMS_V5`` (in _dynamics.py) has only the state-noise key
    — no obs-channel sigma_S appears there since obs params live in
    DEFAULT_PARAMS, not TRUTH_PARAMS."""
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5
    assert TRUTH_PARAMS_V5['sigma_S'] == 0.008
    assert 'sigma_S_obs' not in TRUTH_PARAMS_V5, (
        "TRUTH_PARAMS_V5 should not carry obs-channel keys; obs noise "
        "(sigma_S_obs) belongs in DEFAULT_PARAMS only."
    )
