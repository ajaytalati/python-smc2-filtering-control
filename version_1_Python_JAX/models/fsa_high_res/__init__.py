"""High-res FSA model — Stage D fully-observed control.

3-state physiological SDE:
  B  fitness     (Jacobi, [0, 1] bounded)
  F  strain      (CIR, ≥ 0)
  A  amplitude   (Stuart-Landau, ≥ 0)

Carried over from the public-dev model in
``Python-Model-Development-Simulation/version_1/models/fsa_high_res/``,
keeping only the dynamics for the fully-observed control side. The
4 mixed observation channels (HR, sleep, stress, steps) and the
29-parameter estimation prior are dropped — Stage D treats the
state as directly observable.
"""
