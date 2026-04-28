"""smc2fc — SMC² for filtering and control.

A small framework demonstrating two pillars on simple test models:
  - Bayesian filtering via tempered SMC² with a Schrödinger-Föllmer
    bridge across rolling windows.
  - Stochastic optimal control via the same outer kernel, exploiting
    the control-as-inference duality (Toussaint 2009; Levine 2018;
    Kappen 2005).

Test models:
  - ``models.scalar_ou_lqg`` (Stage A): scalar linear-Gaussian SDE +
    quadratic cost; closed-form Kalman + LQR + LQG.
  - ``models.bistable_controlled`` (Stage B): 2-state cubic-drift +
    OU-control SDE with a saddle-node bifurcation.
"""

__version__ = "0.1.0"
