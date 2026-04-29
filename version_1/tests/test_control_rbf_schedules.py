"""Tests for smc2fc.control.rbf_schedules."""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax.numpy as jnp
import numpy as np
import pytest

from smc2fc.control.rbf_schedules import RBFSchedule


def test_design_matrix_shape():
    rbf = RBFSchedule(n_steps=100, dt=0.1, n_anchors=8)
    Phi = rbf.design_matrix()
    assert Phi.shape == (100, 8)


def test_design_matrix_values_at_anchor_centres_are_near_one():
    """Each column's max should be near 1 (centred Gaussian RBF; the
    grid discretisation can put the actual max at most one step away
    from the centre, so we allow ~5% slack)."""
    rbf = RBFSchedule(n_steps=50, dt=0.2, n_anchors=5)
    Phi = np.asarray(rbf.design_matrix())
    col_max = Phi.max(axis=0)
    np.testing.assert_allclose(col_max, np.ones(5), atol=0.05)


@pytest.mark.parametrize('output,sign_check', [
    ('softplus', lambda y: (y >= 0).all()),
    ('sigmoid',  lambda y: ((y >= 0) & (y <= 1)).all()),
    ('identity', lambda y: True),
])
def test_output_transforms(output, sign_check):
    rbf = RBFSchedule(n_steps=20, dt=0.5, n_anchors=4, output=output)
    theta = jnp.array([1.0, -2.0, 0.5, -1.5])
    schedule = np.asarray(rbf.from_theta(theta))
    assert schedule.shape == (20,)
    assert sign_check(schedule), f"output={output!r} produced bad signs"


def test_invalid_output_raises():
    rbf = RBFSchedule(n_steps=10, dt=0.1, n_anchors=3, output='nonsense')
    with pytest.raises(ValueError, match='unknown output transform'):
        rbf.from_theta(jnp.array([1.0, 0.0, -1.0]))
