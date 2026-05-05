"""Stage 2 (FSA-v5): controller-only bench. Filter NOT run.

Goal: isolate controller behaviour from filter quality. Plant runs under
the controller's chosen schedule; truth params are KNOWN; current state
is the actual plant.state (no posterior inference). Bugs in the
controller side -- RBF schedule, cost composition, integrator, tempering
schedule -- surface here without paying the filter's per-stride cost.

Mirrors the methodology in `claude_plans/controller_only_test_methodology.md`,
adapted for FSA-v5's bimodal control + chance-constrained cost.

Cost variants (set via `--cost`):

  * `--cost soft`  -- Variant B per Ajay's two-cost test plan. Uses
    `evaluate_chance_constrained_cost_soft` (sigmoid surrogate with
    `--beta` knob). Suitable for HMC inner kernels because the
    surrogate is differentiable everywhere.
  * `--cost hard`  -- Variant C. Uses
    `evaluate_chance_constrained_cost_hard` (true indicator). Run with
    `num_mcmc_steps=0` so the tempered-SMC outer loop relies on
    importance weighting alone (no HMC inside, since the indicator is
    non-differentiable).
  * `--cost gradient_ot`  -- back-compat fallback. Uses
    `build_control_spec_v5` directly. NOT shipped as production; only
    for sanity-checking against v3-era controllers.

Acceptance gates:

  1. Applied schedule stays inside [0, Phi_max] (no controller flailing).
  2. Mean A over the run >= A_target (default 2.0, LaTeX §8 Test 5).
  3. Weighted violation rate <= alpha (default 0.05) per the post-hoc
     legacy chance-constraint evaluator (re-evaluated on the actual
     plant trajectory, not the controller's internal estimate).
  4. Replans differ meaningfully across windows (controller adapts).

Run:
    cd version_3 && PYTHONPATH=.:.. python tools/bench_controller_only_fsa_v5.py \
        --cost soft --scenario healthy --T-days 14 --replan-K 2
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

# JAX persistent compilation cache (matches v2 driver pattern)
import pathlib as _pathlib
_CACHE_DIR = _pathlib.Path.home() / ".jax_compilation_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(_CACHE_DIR))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '1')

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# ── Window structure (same as Stage 1; LaTeX §6 confirms 15-min grid)
DT              = 1.0 / 96.0    # 15 min in days
BINS_PER_DAY    = 96
STRIDE_BINS     = 48            # 12 hours

# Defaults
DEFAULT_T_DAYS    = 14
DEFAULT_REPLAN_K  = 2           # replan every 2 strides = 24h
DEFAULT_BETA      = 50.0        # soft variant temperature
DEFAULT_ALPHA     = 0.05
DEFAULT_A_TARGET  = 2.0
DEFAULT_LAM_PHI   = 0.1
DEFAULT_LAM_CHANCE = 100.0

# Trained-athlete state (LaTeX §8 Test 1 setup, reused by Tests 2/3)
TRAINED_ATHLETE_STATE = np.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07])

SCENARIOS = {
    'healthy':     {  # LaTeX §8 Test 2 -- inside the healthy island
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (0.30, 0.30),
        'description':      'Trained athlete + Phi=(0.30, 0.30) baseline',
    },
    'sedentary':   {  # LaTeX §8 Test 1 -- detraining, Phi=(0,0) collapses A
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (0.0, 0.0),
        'description':      'Trained athlete + Phi=(0.0, 0.0) baseline (decond)',
    },
    'overtrained': {  # LaTeX §8 Test 3 -- Phi=(1.0, 1.0) drives F runaway
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (1.0, 1.0),
        'description':      'Trained athlete + Phi=(1.0, 1.0) baseline (overtrn)',
    },
}


# ── CLI parsing ──────────────────────────────────────────────────────

def _pop_named_arg(name: str, default, cast=str):
    """Pop `--name <value>` from sys.argv. Returns cast(value) or default."""
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 >= len(sys.argv):
            raise SystemExit(f"{name} requires a value")
        val = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return cast(val)
    return default


def _next_run_number(experiments_dir: Path) -> int:
    if not experiments_dir.exists():
        return 1
    nums = []
    for p in experiments_dir.iterdir():
        if p.is_dir() and p.name.startswith('run'):
            stem = p.name[3:].split('_', 1)[0]
            try:
                nums.append(int(stem))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def _allocate_run_dir(repo_root: Path, run_tag: str) -> tuple[Path, int]:
    exp_dir = repo_root / "outputs" / "fsa_v5" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    n = _next_run_number(exp_dir)
    out_dir = exp_dir / f"run{n:02d}_{run_tag}"
    out_dir.mkdir(exist_ok=True)
    return out_dir, n


# ── Cost-fn factories ────────────────────────────────────────────────
# Each returns a (cost_fn, schedule_from_theta, theta_dim, ctrl_cfg_overrides)
# tuple for the requested cost variant.

def _build_cost_chance_constrained(
    *,
    cost_kind: str,            # 'soft' or 'hard'
    n_steps: int,
    n_anchors: int,
    init_state: np.ndarray,
    truth_params: dict,
    dt: float,
    alpha: float,
    A_target: float,
    beta: float,
    lam_phi: float,
    lam_chance: float,
    n_truth_particles: int = 1,
):
    """Wrap evaluate_chance_constrained_cost_{soft,hard} as a smc2fc cost_fn.

    Lagrangian aggregation per LaTeX §5.5:

        J = lam_phi * mean_effort - mean_A_integral
            + lam_chance * max(0, weighted_violation_rate - alpha)**2

    For Stage 2 (controller-only): theta_particles = N copies of the
    truth-params dict (n_truth_particles, default 1). The
    chance-constrained cost averages over these; with truth-only it
    collapses to a single deterministic forward rollout.
    """
    from version_3.models.fsa_v5.control_v5 import (
        _cost_soft_jit, _cost_hard_jit, _stack_particle_dicts,
        _ensure_v5_keys,
    )
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5
    from smc2fc.control import RBFSchedule

    # RBF schedule (output_dim=2 for bimodal Phi)
    Phi_default, Phi_max = 0.30, 3.0
    rbf = RBFSchedule(n_steps=n_steps, dt=dt, n_anchors=n_anchors, output='identity')
    Phi_design = rbf.design_matrix()
    p_ratio = Phi_default / Phi_max
    c_Phi = float(math.log(p_ratio / (1.0 - p_ratio)))

    @jax.jit
    def schedule_from_theta(theta):
        theta_B = theta[:n_anchors]
        theta_S = theta[n_anchors:]
        raw_B = c_Phi + jnp.einsum('a,ta->t', theta_B, Phi_design)
        raw_S = c_Phi + jnp.einsum('a,ta->t', theta_S, Phi_design)
        out_B = Phi_max * jax.nn.sigmoid(raw_B)
        out_S = Phi_max * jax.nn.sigmoid(raw_S)
        return jnp.stack([out_B, out_S], axis=1)

    # Pre-stack the truth-only particle cloud once. Outside cost_fn so JIT
    # treats it as a closed-over constant.
    truth_list = [dict(truth_params) for _ in range(n_truth_particles)]
    theta_stacked = _stack_particle_dicts(truth_list)
    theta_stacked = _ensure_v5_keys(theta_stacked, TRUTH_PARAMS_V5)
    weights = jnp.full((n_truth_particles,), 1.0 / n_truth_particles,
                       dtype=jnp.float64)
    init_state_jax = jnp.asarray(init_state, dtype=jnp.float64)

    if cost_kind == 'soft':
        scale = 0.1   # units of A; typical separatrix distance

        @jax.jit
        def cost_fn(theta_ctrl):
            Phi_schedule = schedule_from_theta(theta_ctrl)
            out = _cost_soft_jit(theta_stacked, weights, Phi_schedule,
                                  init_state_jax, dt, alpha, A_target,
                                  beta, scale)
            violation_excess = jnp.maximum(
                0.0, out['weighted_violation_rate'] - alpha)
            return (lam_phi * out['mean_effort']
                    - out['mean_A_integral']
                    + lam_chance * violation_excess ** 2)

    elif cost_kind == 'hard':
        @jax.jit
        def cost_fn(theta_ctrl):
            Phi_schedule = schedule_from_theta(theta_ctrl)
            out = _cost_hard_jit(theta_stacked, weights, Phi_schedule,
                                  init_state_jax, dt, alpha, A_target)
            violation_excess = jnp.maximum(
                0.0, out['weighted_violation_rate'] - alpha)
            return (lam_phi * out['mean_effort']
                    - out['mean_A_integral']
                    + lam_chance * violation_excess ** 2)
    else:
        raise ValueError(f"cost_kind must be 'soft' or 'hard', got {cost_kind!r}")

    theta_dim = 2 * n_anchors
    return cost_fn, schedule_from_theta, theta_dim


def _build_spec_for_cost_variant(
    *, cost: str, n_steps: int, n_anchors: int,
    init_state: np.ndarray, truth_params: dict,
    dt: float, alpha: float, A_target: float, beta: float,
    lam_phi: float, lam_chance: float,
):
    """Return (ControlSpec, ctrl_cfg_overrides) tuple for the requested cost."""
    from smc2fc.control import ControlSpec

    if cost in ('soft', 'hard'):
        cost_fn, schedule_from_theta, theta_dim = _build_cost_chance_constrained(
            cost_kind=cost,
            n_steps=n_steps, n_anchors=n_anchors,
            init_state=init_state, truth_params=truth_params,
            dt=dt, alpha=alpha, A_target=A_target, beta=beta,
            lam_phi=lam_phi, lam_chance=lam_chance,
        )
        spec = ControlSpec(
            name=f'fsa_v5_stage2_{cost}',
            version='5.0',
            dt=dt, n_steps=n_steps, n_substeps=1,
            initial_state=jnp.asarray(init_state, dtype=jnp.float64),
            truth_params=dict(truth_params),
            theta_dim=theta_dim, sigma_prior=1.5, prior_mean=0.0,
            cost_fn=cost_fn, schedule_from_theta=schedule_from_theta,
            acceptance_gates={},
        )
        # Variant C ('hard'): skip HMC inside leapfrog (indicator has zero
        # gradients). Tempering alone drives the outer SMC.
        if cost == 'hard':
            ctrl_cfg_overrides = {'num_mcmc_steps': 0}
        else:
            ctrl_cfg_overrides = {}
        return spec, ctrl_cfg_overrides

    elif cost == 'gradient_ot':
        # Back-compat fallback: use the v5-flavoured gradient-OT spec
        # the senior provided. NOTE: this spec ignores `init_state`
        # arg and `truth_params` arg -- it bakes in TRUTH_PARAMS_V5 +
        # a fixed INIT_STATE per the model file. Only suitable as a
        # back-compat sanity check, not Stage 2 production.
        from version_3.models.fsa_v5.control import build_control_spec_v5
        T_total_days = n_steps * dt
        spec = build_control_spec_v5(
            T_total_days=T_total_days, dt_days=dt, n_anchors=n_anchors,
        )
        return spec, {}

    else:
        raise ValueError(f"--cost must be one of soft/hard/gradient_ot, got {cost!r}")


# ── Main loop ────────────────────────────────────────────────────────

def main():
    cost           = _pop_named_arg('--cost', 'soft', str)
    scenario_key   = _pop_named_arg('--scenario', 'healthy', str)
    T_total_days   = _pop_named_arg('--T-days', DEFAULT_T_DAYS, int)
    replan_K       = _pop_named_arg('--replan-K', DEFAULT_REPLAN_K, int)
    beta           = _pop_named_arg('--beta', DEFAULT_BETA, float)
    alpha          = _pop_named_arg('--alpha', DEFAULT_ALPHA, float)
    A_target       = _pop_named_arg('--A-target', DEFAULT_A_TARGET, float)
    lam_phi        = _pop_named_arg('--lam-phi', DEFAULT_LAM_PHI, float)
    lam_chance     = _pop_named_arg('--lam-chance', DEFAULT_LAM_CHANCE, float)
    n_anchors      = _pop_named_arg('--n-anchors', 8, int)
    # v2-production controller particle counts: n_smc=1024, n_inner=128.
    # Smaller numbers (e.g. v2 dev-config 256/32) under-saturate the
    # RTX 5090. Per CLAUDE.md "GPU saturated post-driver update": these
    # are the values that hit 97% util / 80% VRAM.
    n_smc          = _pop_named_arg('--n-smc', 1024, int)
    n_inner        = _pop_named_arg('--n-inner', 128, int)
    auto_tag = (f"stage2_controller_{cost}_{scenario_key}_"
                 f"T{T_total_days}d_K{replan_K}")
    run_tag        = _pop_named_arg('--run-tag', auto_tag, str)

    if scenario_key not in SCENARIOS:
        raise SystemExit(f"--scenario must be one of {list(SCENARIOS)}; got {scenario_key!r}")
    if cost not in ('soft', 'hard', 'gradient_ot'):
        raise SystemExit(f"--cost must be one of soft/hard/gradient_ot; got {cost!r}")

    scenario = SCENARIOS[scenario_key]
    init_state = scenario['init_state'].astype(np.float64).copy()
    baseline_phi = scenario['baseline_phi']

    repo_root = Path(__file__).resolve().parent.parent
    out_dir, run_num = _allocate_run_dir(repo_root, run_tag)

    print("=" * 76)
    print(f"  Stage 2 (FSA-v5) -- controller-only, cost={cost}, scenario={scenario_key}")
    print(f"  {scenario['description']}")
    print(f"  run dir:  {out_dir.relative_to(repo_root.parent)}")
    print("=" * 76)

    # Imports deferred until after env vars set
    from version_3.models.fsa_v5._plant import StepwisePlant
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5
    from version_3.models.fsa_v5.control_v5 import (
        evaluate_chance_constrained_cost_hard,
        evaluate_chance_constrained_cost_soft,
    )
    from smc2fc.control import SMCControlConfig
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native

    truth = dict(DEFAULT_PARAMS_V5)
    plant = StepwisePlant(
        truth_params=dict(truth),
        state=init_state.copy(),
        seed_offset=42,
    )

    n_strides = (T_total_days * BINS_PER_DAY) // STRIDE_BINS
    print(f"  device:     {jax.devices()[0].platform.upper()}")
    print(f"  T_total:    {T_total_days} days  ({n_strides} strides of {STRIDE_BINS} bins)")
    print(f"  replan K:   every {replan_K} strides ({replan_K * STRIDE_BINS / BINS_PER_DAY * 24:.0f} h)")
    print(f"  cost:       {cost}" + (f"  (beta={beta})" if cost == 'soft' else ''))
    print(f"  n_anchors:  {n_anchors}  (theta_dim = {2*n_anchors})")
    print(f"  n_smc:      {n_smc}, n_inner: {n_inner}")
    print(f"  alpha:      {alpha}, A_target: {A_target}")
    print()

    # Plan horizon = full remaining T (myopic-on-full-horizon, like FSA-v2 E5)
    plan_n_steps = T_total_days * BINS_PER_DAY

    # Build initial control spec at the trained-athlete state
    spec, cfg_overrides = _build_spec_for_cost_variant(
        cost=cost, n_steps=plan_n_steps, n_anchors=n_anchors,
        init_state=plant.state.copy(), truth_params=truth,
        dt=DT, alpha=alpha, A_target=A_target, beta=beta,
        lam_phi=lam_phi, lam_chance=lam_chance,
    )

    # v2-production controller config (mirrors
    # version_2/tools/bench_smc_full_mpc_fsa.py:184-191):
    base_cfg = dict(
        n_smc=n_smc, n_inner=n_inner, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=10, hmc_step_size=0.2, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0, max_temp_steps=30,
        n_calibration_samples=256,
        log_every_n_steps=5,
    )
    base_cfg.update(cfg_overrides)
    ctrl_cfg = SMCControlConfig(**base_cfg)

    print(f"  num_mcmc_steps: {ctrl_cfg.num_mcmc_steps}"
          + (" (HMC skipped for hard variant)" if cost == 'hard' else ''))
    print()

    # ── Closed-loop controller-only run ──
    applied_phi_per_stride = np.zeros((n_strides, 2))   # diagnostic
    plant_state_per_stride = np.zeros((n_strides + 1, 6))
    plant_state_per_stride[0] = plant.state.copy()
    replan_records = []

    # Initial schedule: baseline_phi for the first replan period
    current_schedule = np.tile(baseline_phi, (replan_K, 1))   # (replan_K, 2)

    total_t0 = time.time()
    for k in range(n_strides):
        # On replan boundary, plan fresh schedule from current state
        if k % replan_K == 0:
            print(f"  Stride {k+1:>2}/{n_strides} REPLAN  state={plant.state.round(3).tolist()}",
                  end='', flush=True)
            t_replan = time.time()
            spec, _ = _build_spec_for_cost_variant(
                cost=cost, n_steps=plan_n_steps, n_anchors=n_anchors,
                init_state=plant.state.copy(), truth_params=truth,
                dt=DT, alpha=alpha, A_target=A_target, beta=beta,
                lam_phi=lam_phi, lam_chance=lam_chance,
            )
            result = run_tempered_smc_loop_native(
                spec=spec, cfg=ctrl_cfg, seed=42 + k * 1000,
                print_progress=False,
            )
            elapsed_replan = time.time() - t_replan
            mean_schedule = np.asarray(result['mean_schedule'])  # (plan_n_steps, 2)
            # Decode: take the schedule for the next replan_K strides
            current_schedule_bins = mean_schedule[:replan_K * STRIDE_BINS]
            # Average across daily blocks to get daily Phi (n_days, 2)
            n_replan_days = max(1, int(round(replan_K * STRIDE_BINS / BINS_PER_DAY)))
            current_schedule = (current_schedule_bins
                                .reshape(n_replan_days, BINS_PER_DAY, 2)
                                .mean(axis=1))   # (n_replan_days, 2)
            # Ensure at least replan_K days available even when replan_K maps to <1d
            if current_schedule.shape[0] < max(1, replan_K // 2):
                pad_days = max(1, replan_K // 2) - current_schedule.shape[0] + 1
                current_schedule = np.concatenate([
                    current_schedule, np.tile(current_schedule[-1:], (pad_days, 1))
                ], axis=0)
            print(f"  cost={float(result['particle_costs'].mean()):+.3f}, "
                  f"n_temp={result['n_temp_levels']}, "
                  f"plan={current_schedule.round(2).tolist()}, "
                  f"{elapsed_replan:.1f}s")

            replan_records.append({
                'stride':            k,
                'state_at_replan':   plant.state.copy(),
                'mean_schedule':     mean_schedule,
                'mean_theta':        np.asarray(result['mean_theta']),
                'particle_costs':    np.asarray(result['particle_costs']),
                'n_temp_levels':     int(result['n_temp_levels']),
                'elapsed_s':         elapsed_replan,
                'beta_max':          float(result['beta_max']),
            })
            cur_replan_idx_in_block = 0
        else:
            cur_replan_idx_in_block += 1

        # Pull this stride's daily Phi from the replanned schedule.
        # Each daily slot covers BINS_PER_DAY bins; one stride covers
        # STRIDE_BINS = BINS_PER_DAY/2 bins, so we pass 1 day's worth
        # of schedule (StepwisePlant.advance handles partial slices).
        n_days_for_stride = max(1, STRIDE_BINS // BINS_PER_DAY + 1)
        daily_idx = min(cur_replan_idx_in_block, current_schedule.shape[0] - 1)
        phi_this_stride = current_schedule[daily_idx:daily_idx + 1]   # (1, 2)

        plant.advance(STRIDE_BINS, phi_this_stride)
        plant_state_per_stride[k + 1] = plant.state.copy()
        applied_phi_per_stride[k] = phi_this_stride[0]

    total_elapsed = time.time() - total_t0
    print()
    print(f"  Total: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s) for "
          f"{n_strides} strides + {len(replan_records)} replans")

    # ── Post-hoc trajectory analysis ──
    full_traj = np.concatenate(plant.history['trajectory'], axis=0)
    A_traj = full_traj[:, 3]
    A_integral_observed = float(np.sum(A_traj) * DT)
    mean_A = float(A_traj.mean())

    # Re-evaluate the chance-constraint on the actual plant trajectory
    # (different from the controller's internal estimate, which uses the
    # cost evaluator's deterministic SDE rollout from each replan's
    # init state).
    full_phi = np.concatenate(plant.history['Phi_value'], axis=0)
    full_phi = np.asarray(full_phi).reshape(-1, 2)
    n_steps_full = min(full_phi.shape[0], full_traj.shape[0])
    if cost == 'gradient_ot':
        # Use the post-hoc legacy hard cost as a witness
        from version_3.models.fsa_v5.control_v5 import evaluate_chance_constrained_cost_hard as posthoc_eval
    else:
        from version_3.models.fsa_v5.control_v5 import evaluate_chance_constrained_cost_hard as posthoc_eval
    posthoc = posthoc_eval(
        [dict(truth)], np.array([1.0]),
        full_phi[:n_steps_full],
        dt=DT, alpha=alpha, A_target=A_target,
        truth_params_template=truth,
        initial_state=init_state,
    )
    weighted_violation_rate = float(posthoc['weighted_violation_rate'])
    posthoc_mean_A_integral = float(posthoc['mean_A_integral'])

    # ── Acceptance gates ──
    Phi_max_bound = 3.0
    gate_bounds = bool(np.all((applied_phi_per_stride >= 0.0)
                              & (applied_phi_per_stride <= Phi_max_bound)))
    gate_A_target = bool(A_integral_observed >= A_target)
    gate_violation = bool(weighted_violation_rate <= alpha)
    if len(replan_records) >= 2:
        last_two = np.array([replan_records[-1]['mean_theta'],
                             replan_records[-2]['mean_theta']])
        gate_adapt = bool(np.linalg.norm(last_two[0] - last_two[1]) > 1e-3)
    else:
        gate_adapt = True   # only 1 replan, can't tell
    all_pass = gate_bounds and gate_A_target and gate_violation and gate_adapt

    print()
    print(f"  Acceptance gates:")
    print(f"    {'PASS' if gate_bounds else 'FAIL'}  schedule in [0, {Phi_max_bound}]: "
          f"max applied = {float(applied_phi_per_stride.max()):.2f}")
    print(f"    {'PASS' if gate_A_target else 'FAIL'}  integral A dt >= {A_target}: "
          f"observed = {A_integral_observed:.2f}")
    print(f"    {'PASS' if gate_violation else 'FAIL'}  weighted violation <= {alpha}: "
          f"observed = {weighted_violation_rate:.4f}")
    print(f"    {'PASS' if gate_adapt else 'FAIL'}  controller adapts across replans")
    print(f"  {'ALL GATES PASS' if all_pass else 'ONE OR MORE GATES FAIL'}")

    # ── Save artifacts ──
    np.savez(
        out_dir / "trajectory.npz",
        trajectory=full_traj,
        applied_phi_per_stride=applied_phi_per_stride,
        plant_state_per_stride=plant_state_per_stride,
        full_phi=full_phi,
    )
    np.savez(
        out_dir / "replan_records.npz",
        stride=np.array([r['stride'] for r in replan_records]),
        mean_thetas=np.stack([r['mean_theta'] for r in replan_records]),
        mean_schedules=np.stack([r['mean_schedule'] for r in replan_records]),
        n_temp_levels=np.array([r['n_temp_levels'] for r in replan_records]),
        elapsed_s=np.array([r['elapsed_s'] for r in replan_records]),
    )

    manifest = {
        "schema_version": 1,
        "stage": 2,
        "bench": "bench_controller_only_fsa_v5",
        "run_tag": run_tag,
        "run_number": run_num,
        "fsa_model_dev_pin": "7075436628fa8c202cf62241666fe90230c46ac1",
        "cost_variant": cost,
        "scenario": {
            "name":             scenario_key,
            "description":      scenario['description'],
            "init_state":       init_state.tolist(),
            "baseline_phi":     list(baseline_phi),
        },
        "T_total_days":     T_total_days,
        "step_minutes":     15,
        "BINS_PER_DAY":     BINS_PER_DAY,
        "STRIDE_BINS":      STRIDE_BINS,
        "DT":               DT,
        "n_strides":        n_strides,
        "replan_K":         replan_K,
        "n_replans":        len(replan_records),
        "cost_kwargs": {
            "alpha":         alpha,
            "A_target":      A_target,
            "beta":          beta if cost == 'soft' else None,
            "lam_phi":       lam_phi,
            "lam_chance":    lam_chance,
        },
        "ctrl_cfg": {
            "n_smc":              ctrl_cfg.n_smc,
            "n_inner":             ctrl_cfg.n_inner,
            "sigma_prior":         ctrl_cfg.sigma_prior,
            "target_ess_frac":     ctrl_cfg.target_ess_frac,
            "max_lambda_inc":      ctrl_cfg.max_lambda_inc,
            "num_mcmc_steps":      ctrl_cfg.num_mcmc_steps,
            "hmc_step_size":       ctrl_cfg.hmc_step_size,
            "hmc_num_leapfrog":    ctrl_cfg.hmc_num_leapfrog,
            "beta_max_target_nats": ctrl_cfg.beta_max_target_nats,
            "max_temp_steps":      ctrl_cfg.max_temp_steps,
        },
        "summary": {
            "total_compute_s":           total_elapsed,
            "total_compute_min":         total_elapsed / 60.0,
            "device":                    jax.devices()[0].platform.upper(),
            "mean_A_traj":               mean_A,
            "A_integral_observed":       A_integral_observed,
            "posthoc_mean_A_integral":   posthoc_mean_A_integral,
            "weighted_violation_rate":   weighted_violation_rate,
            "applied_phi_max":           float(applied_phi_per_stride.max()),
            "applied_phi_min":           float(applied_phi_per_stride.min()),
            "final_state":               plant.state.tolist(),
            "gates": {
                "schedule_in_bounds":     gate_bounds,
                "A_integral_geq_target":  gate_A_target,
                "violation_leq_alpha":    gate_violation,
                "controller_adapts":      gate_adapt,
                "all_pass":               all_pass,
            },
        },
    }
    with open(out_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Diagnostic plots ──
    state_names = ['B', 'S', 'F', 'A', 'K_FB', 'K_FS']
    t_days = np.arange(full_traj.shape[0]) * DT

    # Plot 1: latent trajectory
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, name in enumerate(state_names):
        ax = axes[i]
        ax.plot(t_days, full_traj[:, i], color='steelblue', lw=1.0)
        ax.set_xlabel('days')
        ax.set_ylabel(name)
        ax.set_title(f'{name}')
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f'Stage 2 (FSA-v5) -- {scenario_key} scenario, cost={cost}: '
        f'mean A={mean_A:.3f}, integral A dt={A_integral_observed:.2f}, '
        f'violation={weighted_violation_rate:.3f}',
        fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "latent_trajectory.png", dpi=120)
    plt.close()

    # Plot 2: applied schedule per stride (bimodal)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    stride_days = np.arange(n_strides) * STRIDE_BINS / BINS_PER_DAY
    axes[0].plot(stride_days, applied_phi_per_stride[:, 0], 'o-',
                  color='steelblue', label='Phi_B (aerobic)')
    axes[0].axhline(baseline_phi[0], color='gray', linestyle=':',
                     label=f'baseline {baseline_phi[0]:.2f}')
    axes[0].set_ylabel('Phi_B')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(stride_days, applied_phi_per_stride[:, 1], 'o-',
                  color='darkorange', label='Phi_S (strength)')
    axes[1].axhline(baseline_phi[1], color='gray', linestyle=':',
                     label=f'baseline {baseline_phi[1]:.2f}')
    axes[1].set_xlabel('days')
    axes[1].set_ylabel('Phi_S')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.suptitle(
        f'Stage 2 (FSA-v5) -- applied bimodal Phi per stride '
        f'({cost} / {scenario_key})',
        fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "applied_schedule.png", dpi=120)
    plt.close()

    print()
    print(f"  Artifacts written:")
    print(f"    {out_dir}/manifest.json")
    print(f"    {out_dir}/trajectory.npz")
    print(f"    {out_dir}/replan_records.npz")
    print(f"    {out_dir}/latent_trajectory.png")
    print(f"    {out_dir}/applied_schedule.png")
    print("=" * 76)


if __name__ == '__main__':
    main()
