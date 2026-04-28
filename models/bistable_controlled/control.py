"""Bistable_controlled control task spec.

Builds and exports BISTABLE_CONTROL_SPEC — the model-specific
ControlSpec consumed by smc2fc.control.run_tempered_smc_loop.

Task: drive the bistable health variable x from the pathological
well (x = -1) to the healthy well (x = +1) by scheduling the OU
control target u_target(t).

Schedule parameterisation: 6 Gaussian RBF anchors over the 72-hour
horizon, output transform softplus (≥ 0).

Cost functional:
    J(theta) = E_traj[ ∫(x - x_target)^2 dt + λ · ∫u_target^2 dt
                       + barrier(x < -0.5) · dt + 5·(x_T - x_target)^2 ]
"""

from __future__ import annotations

import math
import os

os.environ.setdefault('JAX_ENABLE_X64', 'True')

import jax
import jax.numpy as jnp
import numpy as np

from smc2fc.control import ControlSpec, RBFSchedule
from smc2fc.control.calibration import build_crn_noise_grids


# ── Truth parameters (Set A from public-dev) ──────────────────────────

TRUTH_PARAMS = dict(
    alpha=1.0, a=1.0, sigma_x=0.10,
    gamma=2.0, sigma_u=0.05, sigma_obs=0.20,
)

INIT_STATE = dict(x=-1.0, u=0.0)

EXOGENOUS = dict(
    T_total=72.0,                # hours
    T_intervention=24.0,
    u_on=0.5,                    # hand-coded default reference
    dt_hours=10.0 / 60.0,        # 10 min outer step
)

U_CRIT = (2.0 * TRUTH_PARAMS['alpha'] * TRUTH_PARAMS['a'] ** 3
          / (3.0 * math.sqrt(3.0)))


# ── Cost-functional builder (closure over CRN noise grids + RBF basis) ──

def _build_cost_and_traj_fns(
    *,
    n_inner: int,
    n_steps: int,
    dt: float,
    rbf: RBFSchedule,
    Phi: jnp.ndarray,
    x_target: float = 1.0,
    lam: float = 0.5,
    x_barrier: float = -0.5,
    seed: int = 42,
):
    """Build (cost_fn, traj_sample_fn, schedule_from_theta) closures.

    cost_fn(theta) → scalar mean cost over CRN noise grid.
    traj_sample_fn(theta, key) → single (x, u) trajectory.
    schedule_from_theta(theta) → u_target(t) grid for plotting.
    """
    alpha = TRUTH_PARAMS['alpha']
    a = TRUTH_PARAMS['a']
    gamma = TRUTH_PARAMS['gamma']
    sigma_x = jnp.sqrt(2 * TRUTH_PARAMS['sigma_x'])
    sigma_u = jnp.sqrt(2 * TRUTH_PARAMS['sigma_u'])
    sqrt_dt = jnp.sqrt(dt)

    grids = build_crn_noise_grids(
        n_inner=n_inner, n_steps=n_steps, n_channels=2, seed=seed,
    )
    fixed_wx = grids['wiener'][:, :, 0]
    fixed_wu = grids['wiener'][:, :, 1]
    fixed_x0 = jnp.full((n_inner,), INIT_STATE['x'], dtype=jnp.float64)
    fixed_u0 = jnp.full((n_inner,), INIT_STATE['u'], dtype=jnp.float64)

    @jax.jit
    def schedule_from_theta(theta: jnp.ndarray) -> jnp.ndarray:
        return rbf.from_theta(theta, Phi=Phi)

    @jax.jit
    def cost_fn(theta: jnp.ndarray) -> jnp.ndarray:
        u_target = schedule_from_theta(theta)

        def trial(x0, u0, wx_seq, wu_seq):
            def step(carry, k):
                x, u, c = carry
                u_tgt = u_target[k]
                dx = (alpha * x * (a ** 2 - x ** 2) + u) * dt
                du = -gamma * (u - u_tgt) * dt
                x_next = x + dx + sigma_x * sqrt_dt * wx_seq[k]
                u_next = u + du + sigma_u * sqrt_dt * wu_seq[k]
                stage = ((x - x_target) ** 2
                         + lam * u_tgt ** 2
                         + 50.0 * jax.nn.softplus(-(x - x_barrier) - 1.0)) * dt
                return (x_next, u_next, c + stage), None

            (x_T, _, total), _ = jax.lax.scan(
                step, (x0, u0, jnp.float64(0.0)), jnp.arange(n_steps),
            )
            return total + 5.0 * (x_T - x_target) ** 2

        return jnp.mean(jax.vmap(trial)(fixed_x0, fixed_u0,
                                           fixed_wx, fixed_wu))

    @jax.jit
    def traj_sample_fn(theta: jnp.ndarray, key) -> tuple[jnp.ndarray, jnp.ndarray]:
        u_target = schedule_from_theta(theta)
        wx_seq = jax.random.normal(key, (n_steps,))
        key2, _ = jax.random.split(key)
        wu_seq = jax.random.normal(key2, (n_steps,))

        def step(carry, args):
            x, u = carry
            k, wx, wu = args
            u_tgt = u_target[k]
            dx = (alpha * x * (a ** 2 - x ** 2) + u) * dt
            du = -gamma * (u - u_tgt) * dt
            x_next = x + dx + sigma_x * sqrt_dt * wx
            u_next = u + du + sigma_u * sqrt_dt * wu
            return (x_next, u_next), (x_next, u_next)

        (_, _), traj = jax.lax.scan(
            step,
            (jnp.float64(INIT_STATE['x']), jnp.float64(INIT_STATE['u'])),
            (jnp.arange(n_steps), wx_seq, wu_seq),
        )
        return traj

    return cost_fn, traj_sample_fn, schedule_from_theta


# ── Acceptance gates ──────────────────────────────────────────────────

def _make_acceptance_gates(*, n_steps: int, dt: float,
                            schedule_from_theta_for_default,
                            traj_sample_fn,
                            cost_fn_eval,
                            n_trial_traj: int = 100,
                            seed: int = 23):
    """Build the gate callables that operate on a result dict.

    The gates evaluate:
      1. cost_smc <= cost_default  (SMC² better than the hand-coded default)
      2. basin transition rate >= 80% under the SMC²-derived schedule
    """
    rng = np.random.default_rng(seed)

    # Default schedule: u_target = 0 for first 24h, u_on=0.5 thereafter
    t_grid = np.arange(n_steps) * dt
    default_schedule = np.where(
        t_grid < EXOGENOUS['T_intervention'],
        0.0, EXOGENOUS['u_on']
    )

    def _basin_transition_rate(schedule_arr):
        sx = math.sqrt(2 * TRUTH_PARAMS['sigma_x'])
        su = math.sqrt(2 * TRUTH_PARAMS['sigma_u'])
        sqrt_dt = math.sqrt(dt)
        successes = 0
        for trial in range(n_trial_traj):
            wx = rng.standard_normal(n_steps)
            wu = rng.standard_normal(n_steps)
            x = INIT_STATE['x']
            u = INIT_STATE['u']
            success = False
            for k in range(n_steps):
                t = k * dt
                u_tgt = float(schedule_arr[k])
                dx = (TRUTH_PARAMS['alpha'] * x
                       * (TRUTH_PARAMS['a'] ** 2 - x ** 2) + u)
                du = -TRUTH_PARAMS['gamma'] * (u - u_tgt)
                x = x + dt * dx + sx * sqrt_dt * wx[k]
                u = u + dt * du + su * sqrt_dt * wu[k]
                if t >= 60.0 and x > 0.5:
                    success = True
                    break
            successes += int(success)
        return successes / n_trial_traj

    default_cost = float(cost_fn_eval(jnp.asarray(default_schedule)))

    def gate_cost_lower_than_default(result):
        smc_cost = float(cost_fn_eval(jnp.asarray(result['mean_schedule'])))
        passed = smc_cost <= default_cost
        return passed, smc_cost, (
            f"SMC² mean cost = {smc_cost:.2f}  vs default = {default_cost:.2f}  "
            f"({'passes' if passed else 'fails'} ≤ default)"
        )

    def gate_basin_transition_rate(result):
        rate = _basin_transition_rate(result['mean_schedule'])
        passed = rate >= 0.80
        return passed, rate, (
            f"basin transition rate = {rate:.0%} "
            f"({'passes' if passed else 'fails'} ≥ 80%)"
        )

    return {
        'cost_lower_than_default':  gate_cost_lower_than_default,
        'basin_transition_>=_80%':  gate_basin_transition_rate,
    }, default_schedule, default_cost


# ── Plot helper (model-specific 2x2 layout) ───────────────────────────

def diagnostic_plot(result, out_path: str,
                      default_schedule: np.ndarray,
                      traj_sample_fn,
                      n_traj: int = 5,
                      seed: int = 7):
    """4-panel B2 diagnostic plot."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    spec = result['spec']
    n_steps = spec.n_steps
    dt = spec.dt
    t_grid = np.arange(n_steps) * dt
    smc_schedule = np.asarray(result['mean_schedule'])

    # Top-left: SMC²-derived schedule
    axes[0, 0].plot(t_grid, smc_schedule, '-', color='steelblue', lw=2,
                       label='SMC² posterior-mean u_target(t)')
    axes[0, 0].axhline(U_CRIT, color='red', linestyle=':', alpha=0.7,
                          label=f'u_c = {U_CRIT:.3f}')
    axes[0, 0].axhline(EXOGENOUS['u_on'], color='gray', linestyle=':',
                          alpha=0.7, label=f'default u_on = {EXOGENOUS["u_on"]}')
    axes[0, 0].set_xlabel('time (h)')
    axes[0, 0].set_ylabel('u_target')
    axes[0, 0].set_title('SMC²-derived control schedule')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: default schedule
    axes[0, 1].plot(t_grid, default_schedule, '-', color='darkred', lw=2,
                       label='default u_target(t)')
    axes[0, 1].axhline(U_CRIT, color='red', linestyle=':', alpha=0.7,
                          label=f'u_c = {U_CRIT:.3f}')
    axes[0, 1].axvline(EXOGENOUS['T_intervention'], color='gray',
                          linestyle='--', alpha=0.5,
                          label=f'T_intervention = {EXOGENOUS["T_intervention"]} h')
    axes[0, 1].set_xlabel('time (h)')
    axes[0, 1].set_ylabel('u_target')
    axes[0, 1].set_title('Default (hand-coded) schedule')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom row: sample x trajectories under each schedule
    rng_key = jax.random.PRNGKey(seed)
    for traj_i in range(n_traj):
        rng_key, sub = jax.random.split(rng_key)
        traj_smc = traj_sample_fn(jnp.asarray(result['mean_theta']), sub)
        x_smc = np.asarray(traj_smc[0])
        axes[1, 0].plot(t_grid, x_smc, alpha=0.6, lw=1)

    for ax, title in zip(axes[1, :], ['x(t) under SMC² schedule',
                                          'x(t) under default schedule']):
        ax.axhline(-1.0, color='red', linestyle=':', alpha=0.5,
                     label='x=-1 (sick)')
        ax.axhline(+1.0, color='green', linestyle=':', alpha=0.5,
                     label='x=+1 (well)')
        ax.set_xlabel('time (h)')
        ax.set_ylabel('x (health)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Bistable controlled — '
                  f'{result["n_temp_levels"]} tempering levels in '
                  f'{result["elapsed_s"]:.0f} s on '
                  f'{"GPU" if jax.devices()[0].platform == "gpu" else "CPU"}',
                  fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


# ── Build the spec ────────────────────────────────────────────────────

def build_control_spec(
    *,
    n_anchors: int = 6,
    n_inner: int = 32,
    seed: int = 42,
) -> ControlSpec:
    """Construct BISTABLE_CONTROL_SPEC with cost_fn, gates, etc."""
    dt = EXOGENOUS['dt_hours']
    n_steps = int(round(EXOGENOUS['T_total'] / dt))

    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors,
                        output='softplus')
    Phi = rbf.design_matrix()

    cost_fn, traj_sample_fn, schedule_from_theta = _build_cost_and_traj_fns(
        n_inner=n_inner, n_steps=n_steps, dt=dt, rbf=rbf, Phi=Phi,
        seed=seed,
    )

    # Build cost_fn_eval at higher MC count for gate evaluation (stable cost
    # estimates) — same machinery, fresh seed.
    cost_fn_eval, _, _ = _build_cost_and_traj_fns(
        n_inner=500, n_steps=n_steps, dt=dt, rbf=rbf, Phi=Phi, seed=99,
    )
    # cost_fn_eval as built takes theta (RBF coeffs) — but for gate evaluation
    # we want to evaluate cost on a SCHEDULE directly. Provide a wrapper.
    # The schedule -> raw cost evaluator needs the inner forward sim.
    # Build it explicitly:
    alpha = TRUTH_PARAMS['alpha']
    a = TRUTH_PARAMS['a']
    gamma = TRUTH_PARAMS['gamma']
    sigma_x = jnp.sqrt(2 * TRUTH_PARAMS['sigma_x'])
    sigma_u = jnp.sqrt(2 * TRUTH_PARAMS['sigma_u'])
    sqrt_dt = jnp.sqrt(dt)
    grids = build_crn_noise_grids(
        n_inner=500, n_steps=n_steps, n_channels=2, seed=99,
    )
    fixed_wx = grids['wiener'][:, :, 0]
    fixed_wu = grids['wiener'][:, :, 1]
    fixed_x0 = jnp.full((500,), INIT_STATE['x'], dtype=jnp.float64)
    fixed_u0 = jnp.full((500,), INIT_STATE['u'], dtype=jnp.float64)

    @jax.jit
    def cost_eval_on_schedule(u_target):
        x_target = 1.0
        lam = 0.5
        x_barrier = -0.5

        def trial(x0, u0, wx_seq, wu_seq):
            def step(carry, k):
                x, u, c = carry
                u_tgt = u_target[k]
                dx = (alpha * x * (a ** 2 - x ** 2) + u) * dt
                du = -gamma * (u - u_tgt) * dt
                x_next = x + dx + sigma_x * sqrt_dt * wx_seq[k]
                u_next = u + du + sigma_u * sqrt_dt * wu_seq[k]
                stage = ((x - x_target) ** 2
                         + lam * u_tgt ** 2
                         + 50.0 * jax.nn.softplus(-(x - x_barrier) - 1.0)) * dt
                return (x_next, u_next, c + stage), None

            (x_T, _, total), _ = jax.lax.scan(
                step, (x0, u0, jnp.float64(0.0)), jnp.arange(n_steps),
            )
            return total + 5.0 * (x_T - x_target) ** 2

        return jnp.mean(jax.vmap(trial)(fixed_x0, fixed_u0,
                                           fixed_wx, fixed_wu))

    gates, default_schedule, default_cost = _make_acceptance_gates(
        n_steps=n_steps, dt=dt,
        schedule_from_theta_for_default=schedule_from_theta,
        traj_sample_fn=traj_sample_fn,
        cost_fn_eval=cost_eval_on_schedule,
    )

    spec = ControlSpec(
        name='bistable_controlled',
        version='1.0',
        dt=dt,
        n_steps=n_steps,
        n_substeps=1,
        initial_state=jnp.array([INIT_STATE['x'], INIT_STATE['u']]),
        truth_params=dict(TRUTH_PARAMS),
        theta_dim=n_anchors,
        sigma_prior=1.5,
        cost_fn=cost_fn,
        schedule_from_theta=schedule_from_theta,
        acceptance_gates=gates,
    )
    # Stash extras on the spec for the per-model bench script to use.
    object.__setattr__(spec, '_traj_sample_fn',     traj_sample_fn)
    object.__setattr__(spec, '_default_schedule',   default_schedule)
    object.__setattr__(spec, '_default_cost',       default_cost)
    object.__setattr__(spec, '_cost_eval_on_schedule', cost_eval_on_schedule)
    object.__setattr__(spec, '_diagnostic_plot',    diagnostic_plot)
    return spec


BISTABLE_CONTROL_SPEC = build_control_spec()
