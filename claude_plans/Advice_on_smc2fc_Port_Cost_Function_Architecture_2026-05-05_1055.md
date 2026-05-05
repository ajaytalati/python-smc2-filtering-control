# Advice on the smc2fc Port: Cost-Function Architecture Choice

> Archived from plan mode: 2026-05-05 10:55.

## Context

The `smc2fc` coding agent is making a design call about which v5 cost
function to use inside its tempered-SMC controller. Two candidates exist
in the FSA-v5 codebase that we just shipped:

* `build_control_spec_v5` in
  [`models/fsa_high_res/control.py`](models/fsa_high_res/control.py#L286)
  — pure JAX, fast, JIT-friendly, gradient-friendly. **But** it is the
  v3-era smooth cost: $-\int(A+B+S)dt + \lambda_\Phi \int\|\Phi\|^2 dt +
  \lambda_{\rm barrier}\int \max(F-F_{\max}, 0)^2 dt$. This is precisely
  the formulation §5 / §9.6 of `FSA_version_5_technical_guide.tex` calls
  *the wrong cost for v5*.
* `evaluate_chance_constrained_cost` in
  [`models/fsa_high_res/control_v5.py`](models/fsa_high_res/control_v5.py#L230)
  — the v5 main novelty per the LaTeX guide. Computes $\Pr[A_t < A_{\rm
  sep}(\Phi_t)]$ on a particle cloud, with an analytical separatrix
  $A_{\rm sep}$. **But** the implementation has three JIT-blockers:
  * `scipy.optimize.brentq` ([line 166](models/fsa_high_res/control_v5.py#L166)) for separatrix root-finding,
  * a Python `for k in range(n_steps)` loop over bins ([line 330](models/fsa_high_res/control_v5.py#L330)),
  * a Python `for i in range(n_particles)` loop ([line 349](models/fsa_high_res/control_v5.py#L349)).

The smc2fc agent's diagnosis is technically correct: my reference cost
isn't JIT-able as-is. The agent's *conclusion* (ship gradient-OT and
move on) is what Ajay flagged as "lazy" and a possible threat to the v5
mathematical behaviour. This plan is the response.

## Honest assessment of the stakes

The two costs are not equivalent. Picking gradient-OT means the
controller is solving:

> "maximise time-averaged $A+B+S$ with a soft barrier on $F$"

which lacks the structural feature that defines v5: the closed-island
basin geometry and the bistability annulus. The smooth cost cannot
represent the chance constraint $\Pr[A_t < A_{\rm sep}(\Phi_t)] \le
\alpha$ because indicators of state events are non-differentiable. Two
specific consequences:

1. **Bistable regime.** Inside the bistable annulus, the optimal
   schedule under the chance-constrained cost backs off when state
   uncertainty pushes $A$ near $A_{\rm sep}$. The gradient-OT cost has
   no such mechanism — it sees $A$ as a smooth thing to maximise and
   doesn't know about the basin boundary.
2. **N=1 calibration with rare detraining episodes.** If the cost
   doesn't penalise basin-escape probability, the controller can
   schedule into the bistable annulus and "get lucky" most of the time,
   exposing the subject to occasional catastrophic collapse without it
   appearing as a cost.

So the agent's gradient-OT path *will* produce a working controller,
and it will look right on the v0.1 happy path. It will not exhibit the
v5 main novelty. That is the gap this plan addresses.

## Three architecture options the smc2fc agent could land on

**Option A — Gradient-OT only (the agent's proposed call).** Ship v0.1
with the smooth cost. Document the limitation. Plan a future upgrade.
Lowest cost; no v5 novelty.

**Option B — Smooth-relaxed chance constraint.** Replace the indicator
$\mathbb{1}[A_t < A_{\rm sep}(\Phi_t)]$ with a sigmoid surrogate
$\sigma(\beta(A_{\rm sep} - A_t))$. As $\beta \to \infty$ this
recovers the indicator; for finite $\beta$ it's smooth and HMC-friendly.
The cost becomes a Lagrangian:
\[
J = \lambda_\Phi \int\|\Phi\|^2 dt
   + \lambda_{\rm chance} \int \sigma\big(\beta\,(A_{\rm sep}(\Phi_t) - A_t)\big)\, dt
\]
Pure JAX, JIT-able, gradient-defined. *Captures the v5 novelty in the
limit but lets you anneal the temperature $\beta$ during inference.*

**Option C — Pure SMC² importance weighting.** Drop HMC inside the
controller. Use the SMC tempering schedule itself as the rejection
mechanism: each parameter particle is re-weighted by its empirical
violation rate. No gradients needed; indicator-based weighting is fine
under a sampling-based outer loop. Most structurally correct;
*requires the smc2fc controller architecture to support indicator-based
particle weighting.*

## Recommendation

**Do the upstream rewrite of `control_v5.py` to be JIT-friendly. This
work belongs in `FSA_model_dev`, not in `smc2fc`.**

Specific changes (~150–250 lines in `control_v5.py`):

1. **JAX-bisection separatrix root-finder.**
   Replace `scipy.optimize.brentq` with a `jax.lax.while_loop`
   bisection. Bracket: $[A_{\rm low}, A_{\rm high}] = [10^{-4}, 2.0]$
   (physiological upper bound on $A$). If no sign change in the
   bracket, return $\pm\infty$ as currently defined. Pure JAX,
   jittable, vmappable, gradient-friendly.

2. **`vmap` over particles.**
   Replace the `for i in range(theta_arr.shape[0])` with `jax.vmap`
   over the leading particle dimension. Requires per-particle param
   dicts to be a structured array (use `jax.tree.map` to stack and
   unstack).

3. **`vmap` over bins for $A_{\rm sep}$.**
   Replace `for k in range(n_steps)` with `jax.vmap(find_A_sep_v5,
   in_axes=(0, 0, None))(Phi_B_arr, Phi_S_arr, params)` once
   `find_A_sep_v5` is JAX-bisection.

4. **Two cost variants exposed.**
   * `evaluate_chance_constrained_cost_hard(...)` — uses indicator
     `(A_traj < A_sep).astype(float)`. Returns the same metrics as
     today. Suitable for **Option C**: importance weighting in pure
     SMC²-only architecture.
   * `evaluate_chance_constrained_cost_soft(...)` — uses sigmoid
     surrogate `sigmoid(beta * (A_sep - A_traj))` with a `beta`
     parameter. Suitable for **Option B**: HMC-friendly with
     temperature annealing.

5. **Tests.**
   Extend `tests/test_fsa_v5_smoke.py` with:
   * Hard variant matches the current implementation byte-for-byte
     (regression).
   * Soft variant approaches hard variant as $\beta \to \infty$.
   * Both variants are jittable (call inside `jax.jit` succeeds).
   * Hard variant runs in $O(\text{seconds})$ for 100 particles × 84
     days × 96 bins/day on CPU.

6. **Docstring updates** explaining the trade-off (B vs C) and which
   smc2fc architecture each variant is for.

After this, the smc2fc agent has a real choice: **B** (HMC + soft
chance-constraint with $\beta$ annealing) or **C** (pure SMC² with hard
weighting). Both deliver the v5 novelty. Gradient-OT becomes the
fallback for back-compat tests, not the default.

## Critical files

* [`models/fsa_high_res/control_v5.py`](models/fsa_high_res/control_v5.py)
  — the file to rewrite. Lines 70–170 (the
  `find_A_sep_v5` function) and 230–360 (the `evaluate_chance_constrained_cost`
  function) are the JIT-blockers.
* [`tests/test_fsa_v5_smoke.py`](tests/test_fsa_v5_smoke.py) — extend
  with the three new tests.
* [`LaTex_docs/FSA_version_5_technical_guide.tex`](LaTex_docs/FSA_version_5_technical_guide.tex)
  — §5.3 "Reference implementation" needs an updated subsection
  describing the hard/soft variants and the architecture B/C choice.

## Verification

1. `.fsa_venv/bin/python -m pytest tests/test_fsa_v5_smoke.py -v` —
   all existing v5 smoke tests pass + new JIT/gradient/regression
   tests pass.
2. `.fsa_venv/bin/python -c "from models.fsa_high_res.control_v5
   import evaluate_chance_constrained_cost_hard,
   evaluate_chance_constrained_cost_soft; ..."` — both variants
   import and run.
3. Profile: hard-variant cost evaluation on a 100-particle,
   84-day, 96-bin/day cloud should take <2 seconds wall-clock on CPU.
4. Gradient check: `jax.grad` of soft-variant cost wrt a synthetic
   parameter dict returns finite, non-zero gradients.
5. Regression: hard-variant numerical output on the existing 10-particle,
   14-day smoke test matches the byte-equivalent of the current
   implementation (i.e., the JAX rewrite preserves behaviour).

## Decision (user-confirmed): rewrite BOTH variants

**Hard-indicator variant** for pure-SMC² importance weighting *and*
**soft-sigmoid variant** for HMC with temperature annealing — both
JIT-friendly. The smc2fc agent picks one at integration time based on
their controller's outer-loop architecture; gradient-OT becomes the
back-compat fallback only.

Estimated work: ~200 lines + tests, plus a short LaTeX section update.

## Concrete spec for the rewrite

### A. New helper: `_jax_bisection`
A pure-JAX scalar bisection root-finder using `jax.lax.while_loop`.
Used by both variants for the per-bin separatrix $A_{\rm sep}$.

```python
def _jax_bisection(f, a, b, max_iter=50, tol=1e-6):
    """Bisection on a scalar JAX function f. Caller responsible for
    bracketing — if sign(f(a)) == sign(f(b)) on entry, returns nan.
    """
```
~25 lines.

### B. New helper: `_jax_find_A_sep`
Drop-in JAX replacement for the existing numpy `find_A_sep_v5`. Uses a
fixed bracket $[10^{-4}, 2.0]$ (physiological upper bound on $A$).
Returns a scalar — the separatrix root, or `+inf` if no root in
bracket (collapsed regime), or `-inf` if no positive root because
$\bar\mu(0) > 0$ (mono-stable healthy regime, consistent with current
NumPy version).

The current `find_A_sep_v5` will remain in the file as a debug /
visualisation utility; the new `_jax_find_A_sep` is what the cost
functions call.

### C. Cost variant 1: `evaluate_chance_constrained_cost_hard`
Replace the existing function. Returns the same dict, but:
- Replace the per-bin Python loop with `jax.vmap(_jax_find_A_sep)` over
  the $\Phi$ schedule.
- Replace the per-particle Python loop with `jax.vmap` over a
  particle-batched parameter pytree.
- Replace `scipy.optimize.brentq` with `_jax_bisection`.
- Indicator: hard `(A_traj < A_sep_per_bin).astype(float)`.
- Decorate with `@jax.jit`.

### D. Cost variant 2: `evaluate_chance_constrained_cost_soft`
Same body as variant 1 but with the indicator replaced by:
```python
indicator = jax.nn.sigmoid(beta * (A_sep_per_bin - A_traj) / scale)
```
Extra args: `beta` (default 50.0) and `scale` (default 0.1, units of
$A$). Returns the same dict.

### E. Back-compat shim
Keep the existing function name `evaluate_chance_constrained_cost` as
an alias for `evaluate_chance_constrained_cost_hard`. This way the
smoke test and `__init__.py` re-export don't change.

### F. Tests in `tests/test_fsa_v5_smoke.py`
Add four tests:
1. `test_v5_cost_hard_jits` — `jax.jit(evaluate_chance_constrained_cost_hard)`
   compiles and runs on a 5-particle, 1-day cloud.
2. `test_v5_cost_soft_jits` — same for the soft variant.
3. `test_v5_cost_soft_grad_finite` — `jax.grad` of soft variant wrt
   a parameter dict returns finite, non-trivial gradients.
4. `test_v5_cost_soft_to_hard_limit` — soft variant with $\beta=10000$
   reproduces hard-variant violation rate within 1e-3.

### G. LaTeX update
A short subsection in `FSA_version_5_technical_guide.tex` §5
("Reference implementation") explaining the hard/soft variants, the
beta-annealing strategy for HMC, and the architecture-B-vs-C choice.
~30 lines of prose.

## Critical files

* [`models/fsa_high_res/control_v5.py`](models/fsa_high_res/control_v5.py)
  — primary edit target.
* [`tests/test_fsa_v5_smoke.py`](tests/test_fsa_v5_smoke.py) — add 4
  tests.
* [`LaTex_docs/FSA_version_5_technical_guide.tex`](LaTex_docs/FSA_version_5_technical_guide.tex)
  — short §5 update.
* `models/fsa_high_res/__init__.py` — keep existing
  `evaluate_chance_constrained_cost` symbol; optionally also export
  the `_hard` / `_soft` variants.

## Verification

1. `.fsa_venv/bin/python -m pytest tests/test_fsa_v5_smoke.py -v` —
   all 8 tests pass (4 existing + 4 new).
2. `.fsa_venv/bin/python -c "import jax; from models.fsa_high_res
   import evaluate_chance_constrained_cost; jit = jax.jit(...);
   jit(...)"` — confirms back-compat alias is JIT-able.
3. Wall-clock profile: hard variant on 100-particle, 84-day,
   96-bin/day cloud should complete in <2s on CPU after first JIT
   compilation.
4. Gradient check: $\partial J_{\rm soft} / \partial \mu_K$ at the
   v5 truth, finite and of order 1e-4 to 1e-2 (sanity).
5. LaTeX `latexmk -pdf FSA_version_5_technical_guide.tex` compiles
   cleanly with the §5 update.

## Final advice for the user to send to the smc2fc agent

> "Don't ship gradient-OT as the production cost — that gives up the
> v5 novelty (the chance-constrained formulation is the whole point
> of v5 control). Wait for the upstream rewrite of `control_v5.py`,
> which will land in the `FSA_model_dev` repo shortly. You'll then
> have two interchangeable v5 cost functions:
>
>   * `evaluate_chance_constrained_cost_hard` — for pure-SMC²
>     architectures where you weight particles by their empirical
>     violation rate at each tempering step. Indicator-based, no
>     gradients required.
>   * `evaluate_chance_constrained_cost_soft` — for HMC architectures
>     where you need smooth gradients. Sigmoid surrogate with a
>     `beta` knob you can anneal during inference.
>
> Both are JIT-able and vmap-friendly. The existing
> `evaluate_chance_constrained_cost` symbol is preserved as an alias
> for the hard variant so existing imports keep working.
>
> Use `build_control_spec_v5` (gradient-OT) only as a back-compat
> fallback in tests that need to compare against v3 behaviour."
