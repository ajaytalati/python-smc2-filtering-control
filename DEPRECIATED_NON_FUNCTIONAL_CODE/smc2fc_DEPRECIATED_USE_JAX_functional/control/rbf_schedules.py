"""Gaussian-RBF schedule basis for SMC² control.

Used by B2 and B3 (bistable) and Stage D (FSA) to parameterise a
schedule u_target(t) as a small number of RBF anchors over the
horizon, evaluated at every grid step. Output transforms: identity,
softplus (≥ 0), or sigmoid (∈ [0, 1]).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RBFSchedule:
    """Gaussian-RBF schedule basis with optional output transform.

    Attributes:
        n_steps: number of grid steps.
        dt: step size (in the model's natural units).
        n_anchors: number of RBF anchors evenly spaced over [0, T_total].
        width_factor: RBF width as a multiple of the anchor spacing
            (1.0 → adjacent anchors overlap at 60% intensity; 0.5 →
            sharper anchors; 2.0 → smoother).
        output: 'identity' | 'softplus' | 'sigmoid'.

    Methods:
        design_matrix() → Phi shape (n_steps, n_anchors)
        from_theta(theta) → schedule shape (n_steps,)
    """

    n_steps: int
    dt: float
    n_anchors: int
    width_factor: float = 1.0
    output: str = 'identity'

    def design_matrix(self) -> jnp.ndarray:
        """Build the (n_steps, n_anchors) Gaussian RBF design matrix."""
        T_total = self.n_steps * self.dt
        centres = jnp.linspace(0.0, T_total, self.n_anchors)
        width = (T_total / max(self.n_anchors, 1)) * self.width_factor
        t_grid = jnp.arange(self.n_steps) * self.dt
        return jnp.exp(
            -0.5 * ((t_grid[:, None] - centres[None, :]) / width) ** 2
        )

    def from_theta(self, theta: jnp.ndarray, Phi: jnp.ndarray | None = None
                     ) -> jnp.ndarray:
        """Build a schedule grid from RBF coefficients θ.

        Args:
            theta: shape (n_anchors,)
            Phi:   optional pre-computed design matrix. If None,
                   built on the fly (slow inside JIT — pass it in).

        Returns:
            schedule of shape (n_steps,) with the configured output
            transform applied.
        """
        if Phi is None:
            Phi = self.design_matrix()
        raw = jnp.einsum('a,ta->t', theta, Phi)
        if self.output == 'identity':
            return raw
        elif self.output == 'softplus':
            return jax.nn.softplus(raw)
        elif self.output == 'sigmoid':
            return jax.nn.sigmoid(raw)
        else:
            raise ValueError(f"unknown output transform: {self.output!r}")
