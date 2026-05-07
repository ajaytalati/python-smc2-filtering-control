
Here is what the senior developer has code for you,

## What got built

**New code in [`models/fsa_high_res/control_v5.py`](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/models/fsa_high_res/control_v5.py):**

- `_jax_mu_bar(A, Phi_B, Phi_S, params)` — pure-JAX mirror of `_mu_bar`, JIT-friendly.
- `_jax_find_A_sep(...)` — separatrix root-finder via 64-point grid scan + 40-step JAX bisection. Pure JAX, JIT/vmap/grad-friendly. Drop-in replacement for `scipy.optimize.brentq`.
- `_stack_particle_dicts` + `_ensure_v5_keys` — input coercion helpers.
- `_compute_cost_internals` + `_aggregate` — shared body, fully JIT-able.
- `_cost_hard_jit` + `_cost_soft_jit` — JIT-decorated inner cores.
- **`evaluate_chance_constrained_cost_hard(...)`** — for pure-SMC² importance weighting. Indicator-based.
- **`evaluate_chance_constrained_cost_soft(...)`** — for HMC. Sigmoid surrogate with `beta` annealing parameter.
- **`evaluate_chance_constrained_cost`** — back-compat alias for `_hard`.
- The legacy NumPy/SciPy implementation is renamed `_evaluate_chance_constrained_cost_legacy` and kept as a debug reference.

**Tests in [`tests/test_fsa_v5_smoke.py`](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/tests/test_fsa_v5_smoke.py):** added 4 new tests, total 8 passing in 8.5s:

- `test_v5_cost_hard_jits` — hard variant runs.
- `test_v5_cost_soft_jits` — soft variant runs.
- `test_v5_cost_soft_grad_finite` — `jax.grad` returns finite, non-trivial gradients.
- `test_v5_cost_soft_to_hard_limit` — at β=10⁴ the soft variant matches hard within 10⁻³.

**Docs:** [`LaTex_docs/FSA_version_5_technical_guide.tex`](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/LaTex_docs/FSA_version_5_technical_guide.tex) §5 rewritten to document the two variants + back-compat alias + decision tree for the smc2fc port. PDF rebuilt cleanly at 22 pages.

**Boundary:** [`models/fsa_high_res/__init__.py`](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/models/fsa_high_res/__init__.py) exposes both new variants alongside the existing alias.

## Performance verification

100-particle, 84-day, 96-bin/day cloud (the LaTeX guide's benchmark spec):

|Variant|First call (incl. JIT)|Cached call|Target|
|---|---|---|---|
|Hard|0.86s|**0.229s**|< 2s ✓|
|Soft|0.67s|**0.157s**|(similar) ✓|

8× under budget on the cached call. Comfortably fast enough to sit inside an HMC inner kernel that calls the cost thousands of times.

## What to tell the smc2fc agent

> The chance-constrained cost in `control_v5.py` is now JIT-friendly. You have two interchangeable variants: `evaluate_chance_constrained_cost_hard` for pure-SMC² importance weighting (no gradients needed), and `evaluate_chance_constrained_cost_soft` for HMC architectures (sigmoid surrogate with `beta` annealing). Both are `@jax.jit`-compiled and `vmap`-ed over particles. The existing `evaluate_chance_constrained_cost` symbol is preserved as an alias for the hard variant. Use `build_control_spec_v5` (gradient-OT) only as a back-compat fallback in tests that compare against v3 behaviour. Don't ship gradient-OT as the production cost — that gives up the v5 novelty.

---

My current preference is to prefer Option C as it is more mathematically principled.

The real test though is empirical - and that is why you must run both Options B and C to test what the behaviors are - does the controller perform noticeably better (i.e. accurately) for either option 