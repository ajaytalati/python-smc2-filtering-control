"""FSA v1.5 Python+JAX — Banister-coupled, 3-channel direct-Gaussian obs.

Cross-validation port of `version_1_5_Julia/models/fsa_high_res/`. The
Julia is the production stack; this Python+JAX is a third independent
implementation, differentially tested against the Julia at 1e-6.

The v1.5 model uses v1's drift formulas (NOT v2's reparametrized G1
form) — so `_dynamics.py` is essentially a copy of v1's `_dynamics.py`,
not v2's.
"""
