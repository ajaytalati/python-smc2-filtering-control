"""smc2fc — SMC² for filtering and control (purely-functional rewrite).

A small framework demonstrating two pillars on simple test models:

  * Bayesian filtering via tempered SMC² with a Schrödinger-Föllmer
    bridge across rolling windows.
  * Stochastic optimal control via the same outer kernel, exploiting
    the control-as-inference duality (Toussaint 2009; Levine 2018;
    Kappen 2005).

Test models:

  * ``models.scalar_ou_lqg`` (Stage A): scalar linear-Gaussian SDE +
    quadratic cost; closed-form Kalman + LQR + LQG.
  * ``models.bistable_controlled`` (Stage B): 2-state cubic-drift +
    OU-control SDE with a saddle-node bifurcation.
  * ``models.fsa_high_res`` (v1.5): 3-state Banister-coupled human-
    fitness model with 3 direct-Gaussian obs channels.

Functional contract:

  * No ``for``/``while`` STATEMENTS in the framework body — all
    iteration goes through ``jax.lax.scan``, ``jax.vmap``,
    ``functools.reduce``, or list/dict comprehensions. Two intentional
    on-host adaptive-tempering ``while`` loops in
    ``core/tempered_smc.py`` and ``control/tempered_smc_loop.py`` are
    documented in their respective module docstrings.
  * No mutation after construction — every public dataclass is
    ``@dataclass(frozen=True)`` or a ``typing.NamedTuple``.
  * No ``object.__setattr__`` hacks anywhere.
  * No module-level state mutation (env vars, prints, file I/O at
    import time).

Attributes:
    __version__: Semantic version of the framework.
"""

__version__ = "0.1.0"
