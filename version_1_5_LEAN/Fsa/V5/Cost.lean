import Fsa.V5.Types

/-!
# FSA-v5 cost-function primitives — chance-constraint separator

Single source of truth for the bifurcation parameter $\bar\mu(A; \Phi)$
and the bistable separatrix $A_{\rm sep}(\Phi)$ used by the v5
chance-constrained cost. Transcribed line-by-line from
`models/fsa_high_res/control_v5.py:429-521` (functions `_jax_mu_bar`
and `_jax_find_A_sep`) and the LaTeX equations at
`LaTex_docs/sections/14_v5_sedentary_collapse.tex` (`eq:v5-mubar`,
§10.4 closed-island topology).

Per the LEAN4-first charter (`LaTex_docs/lean4_first_charter.pdf` §5):
this file is both the formal specification AND the executable
reference. The Python is differentially tested against the LEAN4
binary on every PR.

## Structural prevention of Bug 2 (particle-0 separator template)

The historical Python `_compute_cost_internals` collapsed the SMC²
particle ensemble to particle-0 before computing `A_sep`, then
broadcast that single template's separator across all particles'
trajectories — mathematically wrong because the separator depends on
each particle's own bifurcation parameters.

Here, `aSepGrid` takes the FULL particle list and returns one
separator per particle per bin. The output type is
`Array (Array Float)` of shape `(n_particles, n_steps)`, NOT
`Array Float` of shape `(n_steps,)`. The buggy collapse cannot be
written: the function signature forbids it.
-/

namespace Fsa.V5

/-! ## Slow-manifold effective Stuart-Landau coefficient

`muBar A Φ params` mirrors `_jax_mu_bar` at `control_v5.py:429-462`.
This is $\bar\mu(A; \Phi)$ for the v5 closed-island bifurcation: the
slow-manifold $B^*(A), S^*(A), F^*(A)$ are substituted, and the
Hill-deconditioning subtractions are applied.
-/

/-- $\bar\mu(A; \Phi)$ — the FSA-v5 effective Stuart-Landau coefficient
on the slow manifold. Maps to `_jax_mu_bar` at
`control_v5.py:429-462`. -/
def muBar (A : Float) (phi : BimodalPhi) (p : Params) : Float :=
  let phi_B := phi.Phi_B
  let phi_S := phi.Phi_S
  let a_B := (1.0 + p.epsilon_AB * A) / (1.0 + p.epsilon_AB * A_TYP)
  let a_S := (1.0 + p.epsilon_AS * A) / (1.0 + p.epsilon_AS * A_TYP)
  let a_F := (1.0 + p.lambda_A * A)  / (1.0 + p.lambda_A * A_TYP)
  let B   := p.tau_B * p.kappa_B * a_B * phi_B
  let S   := p.tau_S * p.kappa_S * a_S * phi_S
  let KFB := p.KFB_0 + p.tau_K * p.mu_K * phi_B
  let KFS := p.KFS_0 + p.tau_K * p.mu_K * phi_S
  let F   := p.tau_F * (KFB * phi_B + KFS * phi_S) / a_F
  let F_dev := F - F_TYP
  let n   := p.n_dec
  let Bn  := Float.pow (max B 0.0) n
  let Sn  := Float.pow (max S 0.0) n
  let Bdn := Float.pow p.B_dec n
  let Sdn := Float.pow p.S_dec n
  let dec_B := p.mu_dec_B * Bdn / (Bn + Bdn)
  let dec_S := p.mu_dec_S * Sdn / (Sn + Sdn)
  p.mu_0
    + p.mu_B * B + p.mu_S * S
    - p.mu_F * F - p.mu_FF * F_dev * F_dev
    - dec_B - dec_S

/-! ## Separatrix root-finder

`findASep` mirrors `_jax_find_A_sep` at `control_v5.py:465-521`.
Returns:
  * `-Float.inf` — mono-stable healthy regime ($A=0$ unstable; no separatrix)
  * `+Float.inf` — mono-stable collapsed regime
  * finite `Float` — bistable regime; the smaller positive root of
    $g(A) = \bar\mu(A) - \eta A^2$.
-/

/-- Number of grid points the separatrix scan uses. Mirrors the
    Python default `n_grid=64`. Hard-coded so the algorithm is
    pure-Float (no Nat-parameterised arrays in the hot path). -/
private def N_SEP_GRID : Nat := 64
/-- Bisection iteration count. Mirrors the Python default
    `n_bisect=40`. -/
private def N_SEP_BISECT : Nat := 40
/-- Lower bound of the search bracket. Mirrors `A_min=1e-4`. -/
private def A_MIN : Float := 1e-4
/-- Upper bound of the search bracket. Mirrors `A_max=2.0`. -/
private def A_MAX : Float := 2.0

/-- Linear-spaced grid over `[A_MIN, A_MAX]` with `N_SEP_GRID` points.
    Mirrors `jnp.linspace(A_min, A_max, n_grid)`. -/
private def aGrid : Array Float :=
  let n := N_SEP_GRID
  let step := if n ≤ 1 then 0.0 else (A_MAX - A_MIN) / (n - 1).toFloat
  Array.range n |>.map (fun i => A_MIN + i.toFloat * step)

/-- Scan the grid for the FIRST sign change `g(a_i) < 0 ∧ g(a_{i+1}) > 0`
    and return its index. Returns `0` if no sign change exists.
    Mirrors `jnp.argmax(sign_chg.astype(jnp.int32))`. -/
private def firstSignChangeIdx (gVals : Array Float) : Nat := Id.run do
  let n := gVals.size
  if n < 2 then return 0
  for i in [0:n-1] do
    let gi  := gVals[i]!
    let gi1 := gVals[i+1]!
    if gi < 0.0 && gi1 > 0.0 then
      return i
  return 0

/-- Bisection: given `g_at : Float → Float` and bracket `[lo, hi]`,
    iterate `N_SEP_BISECT` times. Mirrors the `jax.lax.scan` body at
    `control_v5.py:507-516`. If `g(mid) < 0`, root is in `[mid, hi]`;
    else in `[lo, mid]`. -/
private def bisect (g_at : Float → Float) (lo hi : Float) : Float := Id.run do
  let mut a := lo
  let mut b := hi
  for _ in [0:N_SEP_BISECT] do
    let mid   := 0.5 * (a + b)
    let g_mid := g_at mid
    if g_mid < 0.0 then
      a := mid
    else
      b := mid
  return 0.5 * (a + b)

/-- Find the bistable separatrix `A_sep(Φ)` under one particle's
    parameters. Three-way return matches the Python's mathematical
    contract: `-inf` (healthy), `+inf` (collapsed), finite (bistable).

    Maps to `_jax_find_A_sep` at `control_v5.py:465-521`. -/
def findASep (phi : BimodalPhi) (p : Params) : Float :=
  let g_at (A : Float) : Float := muBar A phi p - p.eta * A * A
  let gVals := aGrid.map g_at
  -- mono-stable healthy: g(A_min) > 0 ⇒ A=0 unstable ⇒ no separator
  let isHealthy := (gVals[0]!) > 0.0
  -- find first sign change from negative to positive
  let n := gVals.size
  let hasSep : Bool := Id.run do
    if n < 2 then return false
    for i in [0:n-1] do
      if (gVals[i]!) < 0.0 && (gVals[i+1]!) > 0.0 then
        return true
    return false
  let idx := firstSignChangeIdx gVals
  let a0  := aGrid[idx]!
  let b0  := aGrid[idx + 1]!
  let root := bisect g_at a0 b0
  if isHealthy then -1.0 / 0.0           -- -inf
  else if hasSep then root
  else 1.0 / 0.0                         -- +inf

/-! ## Soft chance-constraint surrogate

`softChancePenalty` mirrors the differentiable surrogate used in the
GPU cost kernel. It returns σ(β·(thr−val)/scale) · (thr−val)² —
a smooth gate (≈1 below `thr`, ≈0 above) multiplied by the quadratic
violation depth, so the integrand grows the further `val` drops below
`thr` rather than saturating at a bounded value.
-/

/-- Soft chance-constraint surrogate: σ(β·(thr−val)/scale) · (thr−val)².
    Smooth everywhere (HMC-friendly); ≈ 0 for val above thr;
    grows quadratically for val below thr.
    Mirrors `gpu_control_v5.jl` inner term. -/
def softChancePenalty (val thr beta scale : Float) : Float :=
  let d := thr - val
  sigmoid (beta * d / scale) * d * d

/-! ## Per-particle, per-bin separator grid (kills Bug 2)

The output shape is `(n_particles, n_steps)`. The historical Python
function `_compute_cost_internals` collapsed this to `(n_steps,)` by
using particle-0's params as a "template" for all particles — a
mathematically incorrect simplification, since the separator depends
on each particle's bifurcation parameters.

The signature below makes the buggy collapse impossible to write:
the result is one `Float` per (particle, bin), keyed independently. -/

/-- Compute the separator at every (particle, bin) pair.

Returns shape `(n_particles, n_steps)`.

The Python `_jax_find_A_sep` was vmapped over the schedule using
particle-0 as a template; here we vmap over BOTH the particle axis
AND the schedule axis. Bug 2 was the missing outer vmap. -/
def aSepGrid (particles : Array Params) (schedule : Array BimodalPhi)
    : Array (Array Float) :=
  particles.map (fun p => schedule.map (fun phi => findASep phi p))

end Fsa.V5
