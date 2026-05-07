"""Reference implementation of the FSA-v5 chance-constrained control cost.

This module is the v5 main novelty in code form. The mathematical content is
laid out in ``LaTex_docs/sections/13_cost_function_v4.tex`` §9.6 — equations
(eq:chance-constraint) and (eq:v4-chance-formulation), reproduced here for
quick reference:

  Constraint   :  Pr[ A_t < A_sep(Phi_t) ]  <=  alpha     for all t in [0, T]
  Optimisation :  min_theta  lambda_Phi * integral( ||Phi||^2 dt )
                  subject to:
                      E[ integral( A dt ) ]  >=  A_target
                      Pr[ A_t < A_sep(Phi_t) ]  <=  alpha

The v5 closed-island basin geometry of LaTeX §10 makes this the
structurally correct cost: keep the autonomic state above the bistable
separatrix A_sep(Phi) at all times, with a chance budget alpha on
crossings, while expending minimal training effort.

Why this lives here (not in smc2fc)
-----------------------------------
The actual receding-horizon SMC$^2$ controller lives in the smc2fc
repository — it owns the outer parameter SMC loop, the chance-constraint
particle rejection, the controller-state, etc. This file provides a
**reference cost evaluation** that takes a parameter-particle cloud,
forward-simulates each particle under a candidate Phi schedule, and
returns the per-particle violation rates and aggregate metrics. The
smc2fc controller is responsible for closing the loop: re-weighting,
rejecting, or scoring schedules using these returned quantities.

Single source of truth
----------------------
- Drift              : ``_dynamics.drift_jax``  (canonical v5 drift)
- Diffusion          : ``_dynamics.diffusion_state_dep``
- Bifurcation parameter mu_bar : derived in-line below; matches
  ``tools/stability_basins_v4.py:mu_bar`` and §10.2 (eq:v5-mubar)
- Separatrix root-finding : Brent on ``g(A) = mu_bar(A;Phi) - eta*A^2``;
  semantics match ``tools/stability_basins_v4.py:find_A_separatrix``

LaTeX cross-references
----------------------
- §9.6 equations 23, 24 (chance-constrained formulation)
- §10 (v5 closed-island, source of A_sep)
- §11.6 recommendation (4): chance constraint on basin membership

Usage example
-------------
::

    import jax.numpy as jnp
    from models.fsa_high_res._dynamics import TRUTH_PARAMS_V5
    from models.fsa_high_res.control_v5 import evaluate_chance_constrained_cost

    # 10 particles, all at the v5 truth (a smoke-test cloud)
    n_particles = 10
    theta = jnp.tile(_truth_to_vec(TRUTH_PARAMS_V5), (n_particles, 1))
    weights = jnp.ones(n_particles) / n_particles

    # 28-day Phi schedule (96 bins/day = 2688 bins)
    Phi = jnp.full((28 * 96, 2), 0.30)

    out = evaluate_chance_constrained_cost(
        theta, weights, Phi, dt=1.0/96, alpha=0.05, A_target=2.0,
        truth_params_template=TRUTH_PARAMS_V5,
    )
    print(out['weighted_violation_rate'], out['mean_effort'])
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
import jax
import jax.numpy as jnp

from version_3.models.fsa_v5._dynamics import (
    A_TYP, F_TYP, drift_jax, diffusion_state_dep, TRUTH_PARAMS_V5,
)


# ===========================================================================
# Closed-form quantities used to compute A_sep(Phi)
# ===========================================================================
# These are the slow-manifold equilibrium expressions derived in §7.1
# (cascade reduction). They also appear in tools/stability_basins_v4.py;
# this module re-derives them locally to avoid a hard dependency on the
# tools/ folder (which is not part of the smc2fc import boundary).

def _K_star(Phi_B, Phi_S, params):
    """Slow-manifold K equilibrium (closed-form linear)."""
    KFB = params['KFB_0'] + params['tau_K'] * params['mu_K'] * Phi_B
    KFS = params['KFS_0'] + params['tau_K'] * params['mu_K'] * Phi_S
    return KFB, KFS


def _BS_star(A, Phi_B, Phi_S, params):
    """Slow-manifold (B, S) equilibrium given A."""
    a_B = (1.0 + params['epsilon_AB'] * A) / (1.0 + params['epsilon_AB'] * A_TYP)
    a_S = (1.0 + params['epsilon_AS'] * A) / (1.0 + params['epsilon_AS'] * A_TYP)
    B = params['tau_B'] * params['kappa_B'] * a_B * Phi_B
    S = params['tau_S'] * params['kappa_S'] * a_S * Phi_S
    return B, S


def _F_star(A, Phi_B, Phi_S, params):
    """Slow-manifold F equilibrium given (A, K^*)."""
    KFB, KFS = _K_star(Phi_B, Phi_S, params)
    a_F = (1.0 + params['lambda_A'] * A) / (1.0 + params['lambda_A'] * A_TYP)
    return params['tau_F'] * (KFB * Phi_B + KFS * Phi_S) / a_F


def _mu_bar(A, Phi_B, Phi_S, params):
    """Effective Stuart-Landau coefficient on the slow manifold (v5).

    This is the v5 version of the bifurcation parameter — it includes
    the Hill-deconditioning subtractions. See §10.2, equation
    (eq:v5-mubar). When ``params['mu_dec_B'] = params['mu_dec_S'] = 0``
    this reduces to the v4 form.
    """
    B, S = _BS_star(A, Phi_B, Phi_S, params)
    F = _F_star(A, Phi_B, Phi_S, params)
    F_dev = F - F_TYP
    n     = params.get('n_dec', 4.0)
    B_dec = params.get('B_dec', 0.0)
    S_dec = params.get('S_dec', 0.0)
    mu_dec_B = params.get('mu_dec_B', 0.0)
    mu_dec_S = params.get('mu_dec_S', 0.0)
    Bn  = max(B, 0.0) ** n
    Sn  = max(S, 0.0) ** n
    Bdn = B_dec ** n
    Sdn = S_dec ** n
    dec_B = mu_dec_B * Bdn / (Bn + Bdn) if mu_dec_B > 0 else 0.0
    dec_S = mu_dec_S * Sdn / (Sn + Sdn) if mu_dec_S > 0 else 0.0
    return (params['mu_0']
            + params['mu_B'] * B + params['mu_S'] * S
            - params['mu_F'] * F - params['mu_FF'] * F_dev * F_dev
            - dec_B - dec_S)


def find_A_sep_v5(Phi_B, Phi_S, params, A_max=2.5, n_grid=4000):
    """Unstable separatrix A_sep(Phi) — the basin boundary in the bistable
    regime.

    Definition: the smaller positive root of
        g(A; Phi) := mu_bar(A; Phi) - eta * A^2 = 0.
    Returns ``+inf`` outside the bistable annulus (where the regime is
    either mono-stable healthy — there A=0 is unstable, no separatrix —
    or mono-stable collapsed — no positive root exists at all).
    Convention: ``+inf`` so that "A_t < A_sep" is False everywhere
    (constraint trivially satisfied) in the mono-stable healthy regime,
    and True everywhere (constraint violated) in the mono-stable
    collapsed regime.

    Maps to §7.5 / §10 separatrix definition.
    """
    eta = params['eta']

    def g(a):
        return float(_mu_bar(a, Phi_B, Phi_S, params) - eta * a * a)

    A_grid = np.linspace(1e-7, A_max, n_grid)
    g_vals = np.array([g(a) for a in A_grid])
    sign_flips = np.where(np.diff(np.sign(g_vals)) != 0)[0]
    roots = []
    for idx in sign_flips:
        try:
            roots.append(brentq(g, A_grid[idx], A_grid[idx + 1]))
        except ValueError:
            continue
    roots.sort()
    if len(roots) >= 2:
        return roots[0]   # bistable: lower root = unstable separatrix
    if len(roots) == 1:
        return -float('inf')   # mono-stable healthy: A=0 unstable, no sep
    # No positive roots → mono-stable collapsed: separatrix is "everywhere"
    return float('inf')


# ===========================================================================
# Forward simulator wrapper used by the cost function
# ===========================================================================

def _make_forward_rollout_fn(dt):
    """Build a JIT-compiled per-particle forward-simulator (no noise).

    For the deterministic cost evaluation we run drift-only RK4-style
    Euler steps. Stochastic per-particle CRN rollouts can be substituted
    by replacing this helper without touching the rest of the file.
    """
    def rk4_step(y, params, Phi_t):
        k1 = drift_jax(y, params, Phi_t)
        k2 = drift_jax(y + 0.5 * dt * k1, params, Phi_t)
        k3 = drift_jax(y + 0.5 * dt * k2, params, Phi_t)
        k4 = drift_jax(y + dt * k3, params, Phi_t)
        y_new = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # Domain clipping (matches _dynamics.drift_jax convention)
        y_new = y_new.at[0].set(jnp.clip(y_new[0], 0.0, 1.0))
        y_new = y_new.at[1].set(jnp.clip(y_new[1], 0.0, 1.0))
        y_new = y_new.at[2].set(jnp.maximum(y_new[2], 0.0))
        y_new = y_new.at[3].set(jnp.maximum(y_new[3], 0.0))
        y_new = y_new.at[4].set(jnp.maximum(y_new[4], 0.0))
        y_new = y_new.at[5].set(jnp.maximum(y_new[5], 0.0))
        return y_new

    @jax.jit
    def rollout(y0, params, Phi_schedule):
        """Forward-roll one particle's trajectory.

        Args:
            y0: shape (6,) initial state.
            params: dict of v5 parameters.
            Phi_schedule: shape (n_steps, 2) per-bin Phi.

        Returns:
            traj: shape (n_steps, 6).
        """
        def step(y, k):
            y_next = rk4_step(y, params, Phi_schedule[k])
            return y_next, y_next
        n_steps = Phi_schedule.shape[0]
        _, traj = jax.lax.scan(step, y0, jnp.arange(n_steps))
        return traj

    return rollout


# ===========================================================================
# THE CHANCE-CONSTRAINT COST EVALUATION (LaTeX §9.6 eq 23/24)
# ===========================================================================

def _evaluate_chance_constrained_cost_legacy(
    theta_particles,        # (n_particles, theta_dim)  OR list of param dicts
    weights,                # (n_particles,) — must sum to 1
    Phi_schedule,           # (n_steps, 2)
    *,
    dt: float = 1.0 / 96,   # bin width (days); default = 15 min (FSA_STEP_MINUTES=15)
    alpha: float = 0.05,    # chance-constraint budget on basin escapes
    A_target: float = 2.0,  # minimum E[∫A dt] target
    truth_params_template: dict | None = None,
    initial_state=None,     # (6,) — defaults to a moderate-trained athlete
) -> dict:
    """Legacy NumPy/SciPy reference evaluator for the FSA-v5 chance-constrained cost.

    KEPT AS A DEBUG / READABILITY REFERENCE. Not JIT-able (uses
    ``scipy.optimize.brentq`` and Python loops over bins and particles).
    Production callers should use ``evaluate_chance_constrained_cost_hard``
    or ``evaluate_chance_constrained_cost_soft`` defined below.

    Takes a particle cloud over the model parameters, forward-simulates
    each particle's deterministic trajectory under the candidate Phi
    schedule, and returns aggregate constraint-violation metrics
    suitable for the SMC$^2$ outer loop to either reject candidates or
    score them.

    Args:
        theta_particles: either a (n_particles, theta_dim) ndarray of
            estimated parameter values OR a list of param-dict objects
            (one per particle). The list-of-dicts form is what
            ``smc2fc`` typically maintains; the ndarray form is for
            quick smoke testing where ``theta_dim`` matches a fixed
            parameter ordering.
        weights: shape (n_particles,) particle weights, summing to 1.
        Phi_schedule: shape (n_steps, 2) per-bin (Phi_B, Phi_S).
        dt: bin width in days (default 1/96 = 15 min, matches production).
        alpha: chance-constraint budget — fraction of bins for which
            ``A_t < A_sep(Phi_t)`` is allowed.
        A_target: minimum required time-averaged autonomic exposure
            (the second constraint of equation 24).
        truth_params_template: a base param dict — used to fill in any
            keys not in the per-particle theta vector (e.g. observation
            coefs, frozen v5 deconditioning). Default: ``TRUTH_PARAMS_V5``.
        initial_state: shape (6,) start of each rollout. Default chosen
            to be a moderate-trained athlete inside the v5 island.

    Returns:
        dict with keys
            ``mean_effort``               — average over particles of
                ∫ ||Phi||^2 dt (deterministic in Phi alone).
            ``mean_A_integral``           — weighted-mean ∫ A_t dt across particles.
            ``violation_rate_per_particle`` — (n_particles,) array; the
                fraction of bins where ``A_t < A_sep(Phi_t)`` per particle.
            ``weighted_violation_rate``   — single scalar = weighted mean
                of the per-particle rates.
            ``satisfies_chance_constraint`` — bool: True iff weighted
                rate <= alpha.
            ``satisfies_target``           — bool: True iff
                ``mean_A_integral >= A_target``.
            ``A_sep_per_bin``             — (n_steps,) analytical
                separatrix at each bin, evaluated under
                ``truth_params_template`` (constant across particles for
                speed; chance constraint is then about state, not
                parameter, uncertainty).

    Notes:
        - The per-particle violation rate uses a SHARED
          ``A_sep(Phi_t)`` curve evaluated at ``truth_params_template``,
          not per particle. This is the engineering simplification that
          §11.6 recommends: state-uncertainty dominates the chance
          constraint; parameter uncertainty is handled separately by
          the SMC$^2$ outer loop. For full-fidelity evaluation, the
          smc2fc controller can wrap this function in a per-particle
          loop with per-particle ``A_sep``.
        - This is a deterministic cost (no per-particle SDE noise).
          Each particle differs only in its parameter values, not in
          its noise realisation. Replacing ``_make_forward_rollout_fn``
          with a stochastic variant (CRN noise + per-particle RNG key)
          is a one-function change.
        - Effort cost is deterministic in Phi alone, so identical
          across particles — we report it as a scalar.

    Maps to LaTeX §9.6 equations 23, 24.
    """
    if truth_params_template is None:
        truth_params_template = TRUTH_PARAMS_V5

    # Initial state — moderate-trained athlete, well inside the v5 island.
    if initial_state is None:
        initial_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07])
    else:
        initial_state = jnp.asarray(initial_state)

    Phi_schedule = jnp.asarray(Phi_schedule, dtype=jnp.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if abs(weights.sum() - 1.0) > 1e-6:
        weights = weights / weights.sum()
    n_steps = int(Phi_schedule.shape[0])

    # ── Effort cost (deterministic in Phi) ───────────────────────────
    # ∫ ||Phi||^2 dt. Same for every particle.
    effort = float(jnp.sum(Phi_schedule ** 2) * dt)

    # ── Per-bin separatrix A_sep(Phi_t) at the template params ──────
    # Computed in numpy once (not per particle, see Notes).
    A_sep_per_bin = np.empty(n_steps, dtype=np.float64)
    Phi_np = np.asarray(Phi_schedule)
    for k in range(n_steps):
        A_sep_per_bin[k] = find_A_sep_v5(
            float(Phi_np[k, 0]), float(Phi_np[k, 1]),
            truth_params_template,
        )
    A_sep_jax = jnp.asarray(A_sep_per_bin)

    # ── Forward-simulate each particle ───────────────────────────────
    rollout_fn = _make_forward_rollout_fn(dt)

    # Coerce theta_particles into a list of parameter dicts. Smoke-test
    # callers may pass an ndarray of shape (n_particles, n_keys); we
    # build per-particle dicts by overlaying onto the template. For
    # smc2fc-style callers passing a list of dicts, we use them directly.
    if isinstance(theta_particles, list):
        per_particle_dicts = theta_particles
    else:
        per_particle_dicts = []
        theta_arr = np.asarray(theta_particles)
        for i in range(theta_arr.shape[0]):
            d = dict(truth_params_template)
            # Best-effort assignment: pair the row entries with the
            # template's keys in insertion order. For real smc2fc
            # callers, pass list-of-dicts to be explicit.
            for j, k in enumerate(truth_params_template.keys()):
                if j < theta_arr.shape[1]:
                    d[k] = float(theta_arr[i, j])
            per_particle_dicts.append(d)

    n_particles = len(per_particle_dicts)
    A_integral_per_particle = np.zeros(n_particles)
    violation_rate_per_particle = np.zeros(n_particles)

    for i, p in enumerate(per_particle_dicts):
        # Build a JAX-compatible params dict with all keys drift_jax expects.
        p_jax = {k: jnp.asarray(float(v)) for k, v in p.items()}
        # Ensure v5 Hill keys are present (fill from template if missing)
        for k in ('B_dec', 'S_dec', 'mu_dec_B', 'mu_dec_S', 'n_dec'):
            if k not in p_jax:
                p_jax[k] = jnp.asarray(float(truth_params_template[k]))
        traj = rollout_fn(initial_state, p_jax, Phi_schedule)   # (n_steps, 6)
        A_traj = np.asarray(traj[:, 3])
        A_integral_per_particle[i] = float(np.sum(A_traj) * dt)
        # Violation rate: fraction of bins where A_t < A_sep(Phi_t).
        # Note: A_sep can be ±inf in mono-stable regimes — comparison
        # handles those naturally (A < +inf is always True ⇒ all-violate
        # in collapsed regime; A < -inf is always False ⇒ no violations
        # in healthy regime).
        violations = (A_traj < A_sep_per_bin).astype(np.float64)
        violation_rate_per_particle[i] = float(np.mean(violations))

    weighted_violation_rate = float(np.sum(weights * violation_rate_per_particle))
    mean_A_integral         = float(np.sum(weights * A_integral_per_particle))

    return {
        'mean_effort':                  effort,
        'mean_A_integral':              mean_A_integral,
        'violation_rate_per_particle':  violation_rate_per_particle,
        'weighted_violation_rate':      weighted_violation_rate,
        'satisfies_chance_constraint':  bool(weighted_violation_rate <= alpha),
        'satisfies_target':             bool(mean_A_integral >= A_target),
        'A_sep_per_bin':                A_sep_per_bin,
    }


# =============================================================================
# JIT-FRIENDLY IMPLEMENTATION  (smc2fc-port-ready)
# =============================================================================
# The legacy implementation above uses ``scipy.optimize.brentq`` plus Python
# loops; that's fine for a 10-particle smoke test but is too slow to sit
# inside a tempered-SMC outer loop or an HMC chain that calls the cost a
# few thousand times per inference run.
#
# Two production-ready variants are defined below:
#
#   * ``evaluate_chance_constrained_cost_hard``  — uses the indicator
#     ``(A_traj < A_sep).astype(float)``. Suitable for pure SMC$^2$
#     architectures where the controller weights $\theta$-particles by
#     their empirical violation rate at each tempering step.
#     Indicator-based; not differentiable wrt $\theta$.
#
#   * ``evaluate_chance_constrained_cost_soft`` — uses the sigmoid
#     surrogate ``sigmoid(beta * (A_sep - A_traj) / scale)``. Suitable
#     for HMC architectures that need smooth gradients. Anneal $\beta$
#     toward $\infty$ during inference to recover the hard limit.
#
# Both delegate to a shared inner core ``_compute_cost_internals`` that
# is fully JIT-able and vmap-able. The user-facing alias
# ``evaluate_chance_constrained_cost`` points to the hard variant, so
# existing imports continue to work.

import functools


def _jax_mu_bar(A, Phi_B, Phi_S, params):
    """JIT-friendly counterpart to the legacy ``_mu_bar``.

    Computes the slow-manifold effective Stuart-Landau coefficient
    $\\bar\\mu(A; \\Phi)$ in pure JAX. ``params`` is a dict whose values
    are JAX scalars (or 0-d arrays). All v5 keys must be present
    (``B_dec, S_dec, mu_dec_B, mu_dec_S, n_dec``).

    Mirrors LaTeX §10.2, equation (eq:v5-mubar). Drops the legacy's
    Python ``if mu_dec_*>0`` guard since the v5 truth parameter set
    always has positive thresholds (``B_dec, S_dec > 0``), so
    ``Bn + Bdn > 0`` always.
    """
    a_B = (1.0 + params['epsilon_AB'] * A) / (1.0 + params['epsilon_AB'] * A_TYP)
    a_S = (1.0 + params['epsilon_AS'] * A) / (1.0 + params['epsilon_AS'] * A_TYP)
    a_F = (1.0 + params['lambda_A']  * A) / (1.0 + params['lambda_A']  * A_TYP)
    B   = params['tau_B'] * params['kappa_B'] * a_B * Phi_B
    S   = params['tau_S'] * params['kappa_S'] * a_S * Phi_S
    KFB = params['KFB_0'] + params['tau_K'] * params['mu_K'] * Phi_B
    KFS = params['KFS_0'] + params['tau_K'] * params['mu_K'] * Phi_S
    F   = params['tau_F'] * (KFB * Phi_B + KFS * Phi_S) / a_F

    F_dev = F - F_TYP
    n     = params['n_dec']
    Bn    = jnp.power(jnp.maximum(B, 0.0), n)
    Sn    = jnp.power(jnp.maximum(S, 0.0), n)
    Bdn   = jnp.power(params['B_dec'], n)
    Sdn   = jnp.power(params['S_dec'], n)
    dec_B = params['mu_dec_B'] * Bdn / (Bn + Bdn)
    dec_S = params['mu_dec_S'] * Sdn / (Sn + Sdn)
    return (params['mu_0']
            + params['mu_B'] * B + params['mu_S'] * S
            - params['mu_F'] * F - params['mu_FF'] * F_dev * F_dev
            - dec_B - dec_S)


def _jax_find_A_sep(Phi_B, Phi_S, params,
                     A_min: float = 1e-4,
                     A_max: float = 2.0,
                     n_grid: int = 64,
                     n_bisect: int = 40):
    """JIT-friendly separatrix root-finder.

    Mathematical contract identical to the legacy ``find_A_sep_v5``:
      * ``-inf``           — mono-stable healthy regime ($A=0$ unstable),
                             so the constraint $A_t < A_{\\rm sep}$ is
                             trivially satisfied for any $A_t > 0$.
      * finite scalar in $(A_{\\rm min}, A_{\\rm max})$
                           — bistable regime; the smaller positive root
                             of $g(A) = \\bar\\mu(A) - \\eta A^2$.
      * ``+inf``           — mono-stable collapsed regime; the constraint
                             $A_t < A_{\\rm sep}$ is violated for any
                             finite $A_t$.

    Implementation: scan a 64-point grid for a sign change from negative
    to positive, then bisect within that bracket for ``n_bisect``
    iterations. Pure JAX, JIT-able, vmap-able, gradient-friendly.
    """
    eta = params['eta']

    def g_at(A):
        return _jax_mu_bar(A, Phi_B, Phi_S, params) - eta * A * A

    # Sample $g$ on a uniform grid in $[A_{\rm min}, A_{\rm max}]$.
    A_grid = jnp.linspace(A_min, A_max, n_grid)
    g_vals = jax.vmap(g_at)(A_grid)

    # Mono-stable healthy: $g(A_{\rm min}) > 0$ ⇒ $A=0$ is unstable, no separatrix.
    is_healthy = g_vals[0] > 0.0

    # Find first sign change from negative to positive (the smaller positive root).
    sign_chg = (g_vals[:-1] < 0.0) & (g_vals[1:] > 0.0)
    has_sep = jnp.any(sign_chg)
    idx     = jnp.argmax(sign_chg.astype(jnp.int32))   # 0 if no change
    a0      = A_grid[idx]
    b0      = A_grid[idx + 1]

    # Bisect within $[a_0, b_0]$.
    def bisect_body(state, _):
        lo, hi = state
        mid    = 0.5 * (lo + hi)
        g_mid  = g_at(mid)
        # If g(mid) < 0, root is in [mid, hi]; else in [lo, mid].
        new_lo = jnp.where(g_mid < 0.0, mid, lo)
        new_hi = jnp.where(g_mid < 0.0, hi, mid)
        return (new_lo, new_hi), None

    (lo, hi), _ = jax.lax.scan(bisect_body, (a0, b0), None, length=n_bisect)
    root = 0.5 * (lo + hi)

    # Three-way return matches legacy semantics.
    return jnp.where(is_healthy, -jnp.inf,
                      jnp.where(has_sep, root, jnp.inf))


def _stack_particle_dicts(particles_list):
    """Convert ``[ {param:scalar, ...}, ... ]`` → ``{param: (n_particles,), ...}``.

    Validates that every particle has the same keys. JAX arrays in the
    output, dtype float64.
    """
    if not particles_list:
        raise ValueError("particles_list is empty")
    keys = list(particles_list[0].keys())
    for i, p in enumerate(particles_list):
        if set(p.keys()) != set(keys):
            raise ValueError(f"particle {i} has different keys")
    return {k: jnp.array([float(p[k]) for p in particles_list], dtype=jnp.float64)
            for k in keys}


def _ensure_v5_keys(theta_stacked, truth_template):
    """Fill any missing v5 Hill / cascade keys from a template.

    Estimated parameter vectors from ``smc2fc`` typically only carry the
    37 estimated keys; the 8 frozen dynamics keys (KFB_0, KFS_0, tau_K,
    n_dec, B_dec, S_dec, mu_dec_B, mu_dec_S) plus the 6 frozen diffusion
    keys aren't in the particle. This helper fills them in from the
    truth template, broadcasting to the leading particle dimension.
    """
    n_particles = next(iter(theta_stacked.values())).shape[0]
    needed = TRUTH_PARAMS_V5.keys()
    out = dict(theta_stacked)
    for k in needed:
        if k not in out:
            out[k] = jnp.full((n_particles,), float(truth_template[k]),
                               dtype=jnp.float64)
    return out


def _compute_cost_internals(theta_stacked, weights, Phi_schedule,
                             initial_state, dt):
    """Shared body for hard + soft variants — fully JIT-friendly.

    Returns a tuple
        ``(effort, A_traj_per_particle, A_sep_per_particle_per_bin)``
    where:
      * ``effort`` is a scalar (Φ-only, deterministic).
      * ``A_traj_per_particle`` is shape (n_particles, n_steps) — the
        autonomic-amplitude trajectory of each particle under the
        deterministic (drift-only) v5 dynamics.
      * ``A_sep_per_particle_per_bin`` is shape (n_particles, n_steps)
        — the analytical separatrix evaluated at each bin's $\\Phi$
        using **each particle's own parameters**. The separatrix
        depends on the bifurcation parameters mu_0, mu_B, mu_S, mu_F,
        mu_FF, eta, B_dec, S_dec, mu_dec_B, mu_dec_S, n_dec — all of
        which differ across the SMC² particle population, so collapsing
        to a single template (the historical particle-0 simplification,
        Bug 2 in the 2026-05-06 audit) was incorrect for multi-particle
        runs.
    """
    # 1. Per-particle, per-bin separatrix.
    #    Inner vmap: over the Phi schedule for one particle's params.
    #    Outer vmap: over the SMC² particle axis.
    rollout_fn = _make_forward_rollout_fn(dt)

    def per_particle_internals(params_single):
        # A_sep at every bin under THIS particle's bifurcation params.
        A_sep_seq = jax.vmap(
            lambda Phi_t: _jax_find_A_sep(Phi_t[0], Phi_t[1], params_single)
        )(Phi_schedule)
        # Drift-only rollout under THIS particle's params.
        traj = rollout_fn(initial_state, params_single, Phi_schedule)
        A_seq = traj[:, 3]
        return A_seq, A_sep_seq

    A_traj_per_particle, A_sep_pp = jax.vmap(per_particle_internals)(theta_stacked)

    # 2. Effort cost: deterministic in Φ, identical across particles.
    effort = jnp.sum(Phi_schedule ** 2) * dt

    return effort, A_traj_per_particle, A_sep_pp


def _aggregate(A_traj_pp, A_sep_pp, weights, dt, indicator,
                alpha, A_target, effort):
    """Common aggregation given a per-particle-per-bin indicator array.

    ``indicator`` shape: (n_particles, n_steps), values in [0, 1].
    For the hard variant: 0/1 indicator. For the soft variant: sigmoid.
    ``A_sep_pp`` shape: (n_particles, n_steps) — per-particle separatrix.
    """
    # Per-particle violation rate
    vrpp = jnp.mean(indicator, axis=1)
    weighted_vr = jnp.sum(weights * vrpp)

    # Per-particle ∫A dt
    A_int_pp = jnp.sum(A_traj_pp, axis=1) * dt
    weighted_A_int = jnp.sum(weights * A_int_pp)

    return {
        'mean_effort':                       effort,
        'mean_A_integral':                   weighted_A_int,
        'violation_rate_per_particle':       vrpp,
        'weighted_violation_rate':           weighted_vr,
        'satisfies_chance_constraint':       weighted_vr <= alpha,
        'satisfies_target':                  weighted_A_int >= A_target,
        # Backwards-compat key kept; shape changed from (n_steps,) to
        # (n_particles, n_steps) post Bug-2 fix. Callers that only
        # consume scalar gates aren't affected; callers that plot need
        # to take a particle-axis aggregate (e.g. weighted mean).
        'A_sep_per_bin':                     A_sep_pp,
    }


# ── JIT'd inner cores (one per indicator variant) ─────────────────────────

@functools.partial(jax.jit, static_argnames=('dt', 'alpha', 'A_target'))
def _cost_hard_jit(theta_stacked, weights, Phi_schedule, initial_state,
                    dt, alpha, A_target):
    effort, A_traj_pp, A_sep_pp = _compute_cost_internals(
        theta_stacked, weights, Phi_schedule, initial_state, dt)
    # No broadcast — A_sep_pp already has shape (n_particles, n_steps).
    indicator = (A_traj_pp < A_sep_pp).astype(jnp.float64)
    return _aggregate(A_traj_pp, A_sep_pp, weights, dt, indicator,
                       alpha, A_target, effort)


@functools.partial(jax.jit, static_argnames=('dt', 'alpha', 'A_target',
                                              'beta', 'scale'))
def _cost_soft_jit(theta_stacked, weights, Phi_schedule, initial_state,
                    dt, alpha, A_target, beta, scale):
    effort, A_traj_pp, A_sep_pp = _compute_cost_internals(
        theta_stacked, weights, Phi_schedule, initial_state, dt)
    # Sigmoid surrogate: large beta → hard indicator. ``A_sep_pp`` and
    # ``A_traj_pp`` both have shape (n_particles, n_steps) so the
    # subtraction is element-wise — no broadcast.
    # NOTE: ``A_sep_pp`` may be ±inf in mono-stable regimes — sigmoid
    # handles this gracefully (sigmoid(+inf) = 1, sigmoid(-inf) = 0).
    indicator = jax.nn.sigmoid(beta * (A_sep_pp - A_traj_pp) / scale)
    return _aggregate(A_traj_pp, A_sep_pp, weights, dt, indicator,
                       alpha, A_target, effort)


# ── User-facing entry points (do input coercion outside JIT) ──────────────

def evaluate_chance_constrained_cost_hard(
    theta_particles,
    weights,
    Phi_schedule,
    *,
    dt: float = 1.0 / 96,
    alpha: float = 0.05,
    A_target: float = 2.0,
    truth_params_template: dict | None = None,
    initial_state=None,
) -> dict:
    """JIT-friendly chance-constrained cost (hard indicator).

    Architecturally suitable for **pure-SMC$^2$** controllers where each
    parameter particle is re-weighted by its empirical violation rate
    at each tempering step. The hard indicator
    ``(A_traj < A_sep_per_bin)`` is non-differentiable wrt $\\theta$, so
    this variant is NOT suitable for HMC inner kernels — use
    ``evaluate_chance_constrained_cost_soft`` for that.

    Args:
        theta_particles: either a list of param dicts (legacy form) or
            a dict-of-arrays where each leaf has shape (n_particles,).
            Missing v5 frozen keys are filled from
            ``truth_params_template``.
        weights: shape (n_particles,) particle weights, will be
            normalised to sum to 1.
        Phi_schedule: shape (n_steps, 2) per-bin (Phi_B, Phi_S).
        dt: bin width in days (default 1/96 = 15 min).
        alpha: chance-constraint budget — fraction of bins for which
            $A_t < A_{\\rm sep}(\\Phi_t)$ is allowed.
        A_target: minimum required time-averaged autonomic exposure.
        truth_params_template: dict used to fill missing keys. Default
            ``TRUTH_PARAMS_V5``.
        initial_state: shape (6,) start of each rollout. Default chosen
            to be a moderate-trained athlete inside the v5 island.

    Returns:
        Same dict shape as the legacy ``evaluate_chance_constrained_cost``,
        with all numeric values now JAX arrays (caller may cast to
        Python floats if needed).
    """
    if truth_params_template is None:
        truth_params_template = TRUTH_PARAMS_V5
    if initial_state is None:
        initial_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                                    dtype=jnp.float64)
    else:
        initial_state = jnp.asarray(initial_state, dtype=jnp.float64)

    # Coerce theta_particles to a stacked dict of JAX arrays
    if isinstance(theta_particles, list):
        theta_stacked = _stack_particle_dicts(theta_particles)
    elif isinstance(theta_particles, dict):
        theta_stacked = {k: jnp.asarray(v, dtype=jnp.float64)
                          for k, v in theta_particles.items()}
    else:
        raise ValueError(
            "theta_particles must be a list of dicts or a dict of arrays; "
            f"got {type(theta_particles)}.")

    theta_stacked = _ensure_v5_keys(theta_stacked, truth_params_template)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    Phi_schedule = jnp.asarray(Phi_schedule, dtype=jnp.float64)

    return _cost_hard_jit(theta_stacked, weights, Phi_schedule,
                           initial_state, float(dt), float(alpha), float(A_target))


def evaluate_chance_constrained_cost_soft(
    theta_particles,
    weights,
    Phi_schedule,
    *,
    dt: float = 1.0 / 96,
    alpha: float = 0.05,
    A_target: float = 2.0,
    beta: float = 50.0,         # surrogate temperature (large = hard limit)
    scale: float = 0.1,         # units of A — typical separation distance
    truth_params_template: dict | None = None,
    initial_state=None,
) -> dict:
    """JIT-friendly chance-constrained cost (sigmoid surrogate).

    Architecturally suitable for **HMC**-based controllers (or any
    gradient-based optimiser). The indicator is replaced by:

        sigmoid(beta * (A_sep - A_traj) / scale)

    which is differentiable everywhere. As ``beta → ∞`` this recovers
    the hard indicator. Recommended use: anneal ``beta`` upward over
    the inference run (small β → smooth & easy gradients; large β →
    structurally correct).

    All other arguments and return shape are identical to
    ``evaluate_chance_constrained_cost_hard``.
    """
    if truth_params_template is None:
        truth_params_template = TRUTH_PARAMS_V5
    if initial_state is None:
        initial_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07],
                                    dtype=jnp.float64)
    else:
        initial_state = jnp.asarray(initial_state, dtype=jnp.float64)

    if isinstance(theta_particles, list):
        theta_stacked = _stack_particle_dicts(theta_particles)
    elif isinstance(theta_particles, dict):
        theta_stacked = {k: jnp.asarray(v, dtype=jnp.float64)
                          for k, v in theta_particles.items()}
    else:
        raise ValueError(
            "theta_particles must be a list of dicts or a dict of arrays; "
            f"got {type(theta_particles)}.")

    theta_stacked = _ensure_v5_keys(theta_stacked, truth_params_template)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    weights = weights / jnp.sum(weights)
    Phi_schedule = jnp.asarray(Phi_schedule, dtype=jnp.float64)

    return _cost_soft_jit(theta_stacked, weights, Phi_schedule,
                           initial_state, float(dt), float(alpha),
                           float(A_target), float(beta), float(scale))


# ── Back-compat alias ─────────────────────────────────────────────────────
# The legacy NumPy implementation lives at
# ``_evaluate_chance_constrained_cost_legacy`` (kept as a debug
# reference). The public name now points to the fast hard variant.
evaluate_chance_constrained_cost = evaluate_chance_constrained_cost_hard


# ===========================================================================
# Smoke test
# ===========================================================================

def _smoke_test():
    """Compare legacy / hard / soft variants on a 10-particle cloud."""
    print("=== control_v5 smoke test ===")
    n_particles = 10
    particles = [dict(TRUTH_PARAMS_V5) for _ in range(n_particles)]
    weights = np.ones(n_particles) / n_particles

    n_days = 14
    n_steps = n_days * 96
    Phi = np.tile([0.30, 0.30], (n_steps, 1))

    print("\n--- Legacy NumPy reference ---")
    out_legacy = _evaluate_chance_constrained_cost_legacy(
        particles, weights, Phi,
        dt=1.0/96, alpha=0.05, A_target=2.0,
        truth_params_template=TRUTH_PARAMS_V5,
    )
    print(f"  effort={out_legacy['mean_effort']:.4f}  "
          f"A_int={out_legacy['mean_A_integral']:.4f}  "
          f"violations={out_legacy['weighted_violation_rate']:.4f}")

    print("\n--- JIT-friendly hard variant ---")
    out_hard = evaluate_chance_constrained_cost_hard(
        particles, weights, Phi,
        dt=1.0/96, alpha=0.05, A_target=2.0)
    print(f"  effort={float(out_hard['mean_effort']):.4f}  "
          f"A_int={float(out_hard['mean_A_integral']):.4f}  "
          f"violations={float(out_hard['weighted_violation_rate']):.4f}")

    print("\n--- JIT-friendly soft variant (β=50) ---")
    out_soft = evaluate_chance_constrained_cost_soft(
        particles, weights, Phi,
        dt=1.0/96, alpha=0.05, A_target=2.0, beta=50.0)
    print(f"  effort={float(out_soft['mean_effort']):.4f}  "
          f"A_int={float(out_soft['mean_A_integral']):.4f}  "
          f"violations={float(out_soft['weighted_violation_rate']):.4f}")

    return out_legacy, out_hard, out_soft


if __name__ == '__main__':
    _smoke_test()
