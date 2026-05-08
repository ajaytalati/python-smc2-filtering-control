"""Outer SMC² engine.

Modules:

    * :mod:`smc2fc.core.config` — frozen tuning records.
    * :mod:`smc2fc.core.sampling` — prior-draw helper.
    * :mod:`smc2fc.core.mass_matrix` — diagonal mass-matrix estimator
      for the per-tempering-level HMC kernel.
    * :mod:`smc2fc.core.sf_bridge` — Schrödinger-Föllmer base-measure
      fitter for warm-start bridge SMC.
    * :mod:`smc2fc.core.tempered_smc` — adaptive tempered SMC² (cold
      start + bridge variants), BlackJAX-backed kernel.
    * :mod:`smc2fc.core.jax_native_smc` — compile-once JAX-native HMC
      kernel that bypasses BlackJAX's per-stride JIT recompile.
    * :mod:`smc2fc.core.jax_native_smc_nuts` — NUTS variant of the
      compile-once kernel.
"""
