"""Shared math constants for likelihood evaluation.

Date:    16 April 2026
Version: 1.0

Adding ``HALF_LOG_2PI`` to every Gaussian log-density term is a
no-op for posterior shape (constant offset) but matters for any
absolute-LL comparison: AIC/BIC, model selection, cross-method
consistency checks, marginal-likelihood validation against
analytical baselines.

The pre-v6.4 codebase silently dropped this constant in five
places (HR, stress, OU obs ×2, LogNormal step).  v6.4 standardises:
EVERY Gaussian (or LogNormal) likelihood includes ``- HALF_LOG_2PI``.
"""

import math

HALF_LOG_2PI: float = 0.5 * math.log(2.0 * math.pi)
"""Half-log-2π ≈ 0.9189385332046727."""
