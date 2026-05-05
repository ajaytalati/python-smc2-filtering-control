"""Stage 3 (FSA-v5): full closed-loop SMC2-MPC bench. Filter + controller.

End-to-end pipeline verification. Plant runs under the controller's
chosen schedule; the filter's posterior over (params, state) is fed into
the controller every replan window; the controller plans, the plant
advances, the filter ingests fresh obs.

Combines the structures of:
  * `bench_smc_filter_only_fsa_v5.py` -- the rolling-window filter
  * `bench_controller_only_fsa_v5.py` -- the cost wrapper + controller
  * `version_2/tools/bench_smc_full_mpc_fsa.py` -- the FSA-v2 closed-loop
    template (driver structure, posterior-mean state extraction)

Cost variants behind `--cost {soft,hard,gradient_ot}`:
  * `--cost soft`  -- Variant B per Ajay's two-cost test plan. HMC inside.
  * `--cost hard`  -- Variant C. Pure-SMC2 importance weighting (no HMC).
  * `--cost gradient_ot`  -- back-compat fallback.

Acceptance gates (all four required for Stage 3 PASS):

  1. Mean integral A dt over the run >= A_target (default 2.0).
  2. Weighted violation rate <= alpha (default 0.05) on the post-hoc
     legacy chance-constraint evaluation.
  3. Filter id_cov: posterior 90% CI covers truth on >=80% of estimable
     params for >=80% of windows.
  4. Total compute <= 4 hours on RTX 5090.

Run:
    cd version_3 && PYTHONPATH=.:.. python tools/bench_smc_full_mpc_fsa_v5.py \
        --cost soft --T-days 14 --replan-K 2
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')
# Disable JAX/XLA pre-allocating ~75% of GPU memory at first JIT (Ajay's
# request: nvtop should show actual per-PID demand, not the preallocated
# block). Mirrors the v2 launchers in version_2/tools/launchers/.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

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


# ── Time grid (LaTeX §6 -- 15-min bins are mandatory) ────────────────
DT              = 1.0 / 96.0
BINS_PER_DAY    = 96
WINDOW_BINS     = 96     # 1 day
STRIDE_BINS     = 48     # 12 hours

# Defaults
DEFAULT_T_DAYS    = 14
DEFAULT_REPLAN_K  = 2          # replan every 2 strides = daily
DEFAULT_BETA      = 50.0
DEFAULT_ALPHA     = 0.05
DEFAULT_A_TARGET  = 2.0
DEFAULT_LAM_PHI   = 0.1
DEFAULT_LAM_CHANCE = 100.0

TRAINED_ATHLETE_STATE = np.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07])

SCENARIOS = {
    'healthy':     {
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (0.30, 0.30),
        'description':      'Trained athlete + Phi=(0.30, 0.30) baseline',
    },
    'sedentary':   {
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (0.0, 0.0),
        'description':      'Trained athlete + Phi=(0.0, 0.0) baseline (decond)',
    },
    'overtrained': {
        'init_state':       TRAINED_ATHLETE_STATE,
        'baseline_phi':     (1.0, 1.0),
        'description':      'Trained athlete + Phi=(1.0, 1.0) baseline (overtrn)',
    },
}


def _pop_named_arg(name: str, default, cast=str):
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


def extract_window(obs_data, start: int, end: int):
    """Slice obs_data to bins [start, end), re-indexing t_idx to [0, end-start)."""
    window = {}
    for ch_name, ch_data in obs_data.items():
        t_idx = np.asarray(ch_data['t_idx'])
        mask = (t_idx >= start) & (t_idx < end)
        new_ch = {'t_idx': (t_idx[mask] - start).astype(np.int32)}
        n_t = len(t_idx)
        for key in ch_data:
            if key == 't_idx':
                continue
            val = ch_data[key]
            arr = np.asarray(val)
            if arr.ndim >= 1 and len(arr) == n_t:
                new_ch[key] = arr[mask]
            else:
                new_ch[key] = val
        window[ch_name] = new_ch
    return window


# ── Cost-fn factory: posterior particles version ─────────────────────
# Differs from Stage 2 by accepting a posterior particle cloud (theta_stacked)
# rather than a truth-only single-particle. The cloud comes from the
# filter posterior; the cost evaluator averages over it to capture
# parameter uncertainty.

def _build_cost_chance_constrained_posterior(
    *,
    cost_kind: str,
    n_steps: int,
    n_anchors: int,
    init_state: np.ndarray,
    theta_stacked: dict,        # dict of (n_particles,)-shaped arrays
    weights: jnp.ndarray,       # (n_particles,)
    dt: float, alpha: float, A_target: float,
    beta: float, lam_phi: float, lam_chance: float,
    bin_stride: int = 4,        # only used by 'soft_fast'
):
    from version_3.models.fsa_v5.control_v5 import (
        _cost_soft_jit, _cost_hard_jit, _ensure_v5_keys,
    )
    from version_3.models.fsa_v5.control_v5_fast import _cost_soft_fast_jit
    from version_3.models.fsa_v5._dynamics import TRUTH_PARAMS_V5
    from smc2fc.control import RBFSchedule

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

    theta_stacked = _ensure_v5_keys(theta_stacked, TRUTH_PARAMS_V5)
    init_state_jax = jnp.asarray(init_state, dtype=jnp.float64)

    if cost_kind == 'soft':
        scale = 0.1

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

    elif cost_kind == 'soft_fast':
        # Variant: fp32 + relaxed bisection + sub-sampled bins. Cast
        # the posterior cloud + weights + init_state to fp32 once outside
        # the JIT body.
        scale = 0.1
        theta_stacked_f32 = {k: v.astype(jnp.float32)
                             for k, v in theta_stacked.items()}
        weights_f32 = weights.astype(jnp.float32)
        init_state_f32 = init_state_jax.astype(jnp.float32)

        @jax.jit
        def cost_fn(theta_ctrl):
            Phi_schedule = schedule_from_theta(theta_ctrl).astype(jnp.float32)
            out = _cost_soft_fast_jit(theta_stacked_f32, weights_f32,
                                       Phi_schedule, init_state_f32,
                                       dt, alpha, A_target,
                                       beta, scale, bin_stride)
            violation_excess = jnp.maximum(
                0.0, out['weighted_violation_rate'] - alpha)
            return jnp.float64(lam_phi * out['mean_effort']
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
        raise ValueError(
            f"cost_kind must be 'soft' / 'soft_fast' / 'hard', got {cost_kind!r}")

    theta_dim = 2 * n_anchors
    return cost_fn, schedule_from_theta, theta_dim


def _build_spec_for_cost_variant_posterior(
    *, cost: str, n_steps: int, n_anchors: int,
    init_state: np.ndarray, theta_stacked: dict, weights: jnp.ndarray,
    dt: float, alpha: float, A_target: float, beta: float,
    lam_phi: float, lam_chance: float,
    bin_stride: int = 4,
):
    from smc2fc.control import ControlSpec

    if cost in ('soft', 'hard', 'soft_fast'):
        cost_fn, schedule_from_theta, theta_dim = (
            _build_cost_chance_constrained_posterior(
                cost_kind=cost,
                n_steps=n_steps, n_anchors=n_anchors,
                init_state=init_state, theta_stacked=theta_stacked,
                weights=weights, dt=dt, alpha=alpha, A_target=A_target,
                beta=beta, lam_phi=lam_phi, lam_chance=lam_chance,
                bin_stride=bin_stride,
            )
        )
        spec = ControlSpec(
            name=f'fsa_v5_stage3_{cost}', version='5.0',
            dt=dt, n_steps=n_steps, n_substeps=1,
            initial_state=jnp.asarray(init_state, dtype=jnp.float64),
            truth_params={},   # not used by cost_fn (uses posterior cloud)
            theta_dim=theta_dim, sigma_prior=1.5, prior_mean=0.0,
            cost_fn=cost_fn, schedule_from_theta=schedule_from_theta,
            acceptance_gates={},
        )
        if cost == 'hard':
            ctrl_cfg_overrides = {'num_mcmc_steps': 0}
        elif cost == 'soft_fast':
            # Reverted: see version_3/tools/bench_controller_only_fsa_v5.py
            # for the rationale. Same HMC config as `soft`; only the
            # cost-fn-side optimisations remain in `soft_fast`.
            ctrl_cfg_overrides = {}
        else:
            ctrl_cfg_overrides = {}
        return spec, ctrl_cfg_overrides

    elif cost == 'gradient_ot':
        from version_3.models.fsa_v5.control import build_control_spec_v5
        T_total_days = n_steps * dt
        spec = build_control_spec_v5(
            T_total_days=T_total_days, dt_days=dt, n_anchors=n_anchors,
        )
        return spec, {}

    else:
        raise ValueError(f"--cost must be one of soft/hard/gradient_ot, got {cost!r}")


def _posterior_to_theta_stacked(particles_constrained: np.ndarray,
                                  param_names: list[str]) -> dict:
    """Convert (n_smc, n_params) constrained posterior array into a
    dict-of-arrays for the chance-constrained cost evaluator."""
    return {name: jnp.asarray(particles_constrained[:, i], dtype=jnp.float64)
            for i, name in enumerate(param_names)}


# ── Main ─────────────────────────────────────────────────────────────

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
    # 5090-saturation point per CLAUDE.md (N=256/K=400 saturates; bigger
    # is just more wall-clock). Override via --n-smc / --n-pf / --n-inner
    # if you want the slower production posterior (1024/800/128).
    n_smc          = _pop_named_arg('--n-smc', 256, int)    # filter outer + controller
    n_pf           = _pop_named_arg('--n-pf', 400, int)     # filter inner
    n_inner        = _pop_named_arg('--n-inner', 64, int)   # controller cost-MC
    bin_stride     = _pop_named_arg('--bin-stride', 4, int) # only used by --cost soft_fast
    auto_tag = (f"stage3_full_mpc_{cost}_{scenario_key}_"
                 f"T{T_total_days}d_K{replan_K}")
    run_tag        = _pop_named_arg('--run-tag', auto_tag, str)

    if scenario_key not in SCENARIOS:
        raise SystemExit(f"--scenario must be one of {list(SCENARIOS)}")
    if cost not in ('soft', 'soft_fast', 'hard', 'gradient_ot'):
        raise SystemExit(f"--cost must be one of soft/soft_fast/hard/gradient_ot")

    scenario = SCENARIOS[scenario_key]
    init_state = scenario['init_state'].astype(np.float64).copy()
    baseline_phi = scenario['baseline_phi']

    repo_root = Path(__file__).resolve().parent.parent
    out_dir, run_num = _allocate_run_dir(repo_root, run_tag)

    print("=" * 76)
    print(f"  Stage 3 (FSA-v5) -- full closed-loop MPC, cost={cost}, "
          f"scenario={scenario_key}")
    print(f"  {scenario['description']}")
    print(f"  run dir:  {out_dir.relative_to(repo_root.parent)}")
    print("=" * 76)

    from version_3.models.fsa_v5._plant import StepwisePlant
    from version_3.models.fsa_v5.simulation import DEFAULT_PARAMS_V5
    from version_3.models.fsa_v5.estimation import (
        HIGH_RES_FSA_V5_ESTIMATION,
    )
    from version_3.models.fsa_v5.control_v5 import (
        evaluate_chance_constrained_cost_hard as posthoc_eval,
    )
    from smc2fc.control import SMCControlConfig
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native
    from smc2fc.core.config import SMCConfig
    # v2-production native path: compile-once factory + Partial-wrapped
    # log_density per stride. Avoids per-window JIT recompile.
    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density_compileonce,
    )

    truth = dict(DEFAULT_PARAMS_V5)
    em = HIGH_RES_FSA_V5_ESTIMATION
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}
    all_param_names = list(em.all_names)

    plant = StepwisePlant(
        truth_params=dict(truth),
        state=init_state.copy(),
        seed_offset=42,
    )

    n_strides = (T_total_days * BINS_PER_DAY) // STRIDE_BINS
    plan_n_steps = T_total_days * BINS_PER_DAY

    print(f"  device:     {jax.devices()[0].platform.upper()}")
    print(f"  T_total:    {T_total_days} days  ({n_strides} strides)")
    print(f"  WINDOW:     {WINDOW_BINS} bins / STRIDE: {STRIDE_BINS} bins")
    print(f"  replan K:   every {replan_K} strides")
    print(f"  cost:       {cost}" + (f"  beta={beta}" if cost == 'soft' else ''))
    print(f"  filter:     n_pf={n_pf}, n_smc(filter)=128, "
          f"controller n_smc={n_smc}, n_inner={n_inner}")
    print()

    # v2-production filter config (mirrors bench_smc_full_mpc_fsa.py:163-176).
    # Critical: n_smc_particles=1024 outer + n_pf_particles=800 inner is the
    # GPU-saturating point; smaller numbers leave the 5090 idle.
    smc_filter_cfg = SMCConfig(
        n_smc_particles=n_smc, n_pf_particles=n_pf,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed', sf_use_q0_cov=True, sf_blend=0.7,
        sf_annealed_n_stages=3, sf_annealed_n_mh_steps=5,
        sf_info_aware=False,
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.15,
    )

    # v2-production controller config (mirrors bench_smc_full_mpc_fsa.py:184-191).
    # `num_mcmc_steps=0` override is applied per-replan when --cost hard
    # (indicator gradient is zero -> HMC inside leapfrog is wasted).
    base_ctrl_cfg = dict(
        n_smc=n_smc, n_inner=n_inner, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        num_mcmc_steps=10, hmc_step_size=0.2, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0, max_temp_steps=30,
        n_calibration_samples=256,
        log_every_n_steps=5,
    )

    # ── State/obs accumulators across the run ──
    prev_particles = None
    fixed_init_state = jnp.asarray(init_state)
    accumulated_obs = {ch: {'t_idx': [], **{}} for ch in
                        ('obs_HR', 'obs_sleep', 'obs_stress',
                         'obs_steps', 'obs_volumeload', 'Phi', 'C')}
    window_records = []
    replan_records = []
    applied_phi_per_stride = np.zeros((n_strides, 2))
    plant_state_per_stride = np.zeros((n_strides + 1, 6))
    plant_state_per_stride[0] = plant.state.copy()

    # ── Build the compile-once log-density factory ONCE before the loop ──
    # v2 production pattern (bench_smc_full_mpc_fsa.py:250-260).
    log_density_factory = make_gk_dpf_v3_lite_log_density_compileonce(
        model=em, n_particles=smc_filter_cfg.n_pf_particles,
        bandwidth_scale=smc_filter_cfg.bandwidth_scale,
        ot_ess_frac=smc_filter_cfg.ot_ess_frac,
        ot_temperature=smc_filter_cfg.ot_temperature,
        ot_max_weight=smc_filter_cfg.ot_max_weight,
        ot_rank=smc_filter_cfg.ot_rank, ot_n_iter=smc_filter_cfg.ot_n_iter,
        ot_epsilon=smc_filter_cfg.ot_epsilon,
        dt=DT, t_steps=WINDOW_BINS,
    )
    T_arr = log_density_factory._transforms

    # Initial schedule = baseline. The first replan (after window 1
    # is full) updates this.
    current_schedule_per_day = np.tile(baseline_phi, (T_total_days, 1)).astype(np.float64)

    total_t0 = time.time()
    for s in range(n_strides):
        # 1. Plant advance using the planned schedule (pull this stride's day's slice)
        n_days_into_run = s * STRIDE_BINS // BINS_PER_DAY
        # provide a generous slice (>= 1 day) for the plant.advance API
        days_slice = max(1, STRIDE_BINS // BINS_PER_DAY + 1)
        phi_slice = current_schedule_per_day[
            n_days_into_run:n_days_into_run + days_slice + 1]
        if phi_slice.shape[0] == 0:
            phi_slice = np.tile(current_schedule_per_day[-1:], (days_slice + 1, 1))
        out_stride = plant.advance(STRIDE_BINS, phi_slice)
        plant_state_per_stride[s + 1] = plant.state.copy()
        # average daily Phi we actually applied for diagnostics
        applied_phi_per_stride[s] = phi_slice[0]

        # 2. Accumulate obs into a single global record
        for ch in ('obs_HR', 'obs_sleep', 'obs_stress',
                   'obs_steps', 'obs_volumeload', 'Phi', 'C'):
            ch_data = out_stride[ch]
            for key in ch_data:
                accumulated_obs[ch].setdefault(key, []).append(
                    np.asarray(ch_data[key]))

        # 3. Build full obs dict so far (concat lists)
        obs_data_full = {}
        for ch_name in accumulated_obs:
            ch_dict = {}
            for key, parts in accumulated_obs[ch_name].items():
                if not parts:
                    continue
                if key == 't_idx':
                    ch_dict['t_idx'] = np.concatenate(parts).astype(np.int32)
                else:
                    arr = np.concatenate([np.atleast_1d(p) for p in parts])
                    ch_dict[key] = arr
            obs_data_full[ch_name] = ch_dict

        end_bin = (s + 1) * STRIDE_BINS
        start_bin = max(0, end_bin - WINDOW_BINS)

        if end_bin - start_bin < WINDOW_BINS:
            print(f"  Stride {s+1:>2}/{n_strides}  warm-up "
                  f"(window not full yet)")
            continue

        # 4. Filter window
        print(f"  Stride {s+1:>2}/{n_strides}  bins {start_bin:>4}-{end_bin:>4}",
              end='', flush=True)
        window_obs = extract_window(obs_data_full, start_bin, end_bin)
        grid_obs = em.align_obs_fn(window_obs, WINDOW_BINS, DT)

        # v2-production native path: bind dynamic data via Partial; the
        # underlying jitted scan caches across strides.
        key0_stride = jax.random.PRNGKey(42 + s * 1000)
        w_start_arr = jnp.asarray(start_bin, dtype=jnp.int32)
        ld = jax.tree_util.Partial(
            log_density_factory,
            grid_obs=grid_obs,
            fixed_init_state=fixed_init_state,
            w_start=w_start_arr,
            key0=key0_stride,
        )
        if prev_particles is None:
            init_tag = 'cold'
            particles_unc, elapsed_f, n_temp_f = run_smc_window_native(
                ld, em, T_arr, cfg=smc_filter_cfg,
                initial_particles=None, seed=42 + s * 1000,
            )
        else:
            init_tag = 'bridge'
            particles_unc, elapsed_f, n_temp_f = run_smc_window_bridge_native(
                new_ld=ld, prev_particles=prev_particles,
                model=em, T_arr=T_arr, cfg=smc_filter_cfg,
                seed=42 + s * 1000,
            )
        particles_unc = np.asarray(particles_unc)
        particles_constrained = np.array([
            np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
            for p in particles_unc
        ])
        n_id_covered = sum(
            1 for name in all_param_names
            if np.quantile(particles_constrained[:, name_to_idx[name]], 0.05) <= truth[name]
                <= np.quantile(particles_constrained[:, name_to_idx[name]], 0.95)
        )
        print(f" ({init_tag}) {n_temp_f}lvl/{elapsed_f:.0f}s, "
              f"id={n_id_covered}/{len(all_param_names)}", end='', flush=True)
        prev_particles = particles_unc

        # vmap-batched smoothed state extract via the compile-once factory.
        # Same Partial wraps the dynamic data as the bridge log_density
        # so the cache hits.
        n_extract = min(10, particles_unc.shape[0])
        us_extract = jnp.asarray(particles_unc[:n_extract])
        target_step_arr = jnp.asarray(STRIDE_BINS, dtype=jnp.int32)
        extract_partial = jax.tree_util.Partial(
            log_density_factory.extract_state_at_step,
            grid_obs=grid_obs,
            fixed_init_state=fixed_init_state,
            w_start=w_start_arr,
            key0=key0_stride,
            target_step=target_step_arr,
        )
        states = jax.vmap(extract_partial)(us_extract)
        smoothed_state = np.asarray(jnp.mean(states, axis=0))
        fixed_init_state = jnp.asarray(smoothed_state)

        window_records.append({
            'stride':            s,
            'start_bin':         start_bin, 'end_bin': end_bin,
            'n_temp':            int(n_temp_f),
            'elapsed_s':         float(elapsed_f),
            'init_tag':          init_tag,
            'particles_constrained': particles_constrained,
            'id_covered':        n_id_covered,
        })

        # 5. Replan if scheduled
        if (s + 1) % replan_K == 0:
            theta_stacked = _posterior_to_theta_stacked(
                particles_constrained, all_param_names)
            weights = jnp.full((particles_constrained.shape[0],),
                                1.0 / particles_constrained.shape[0],
                                dtype=jnp.float64)
            spec, cfg_overrides = _build_spec_for_cost_variant_posterior(
                cost=cost, n_steps=plan_n_steps, n_anchors=n_anchors,
                init_state=smoothed_state, theta_stacked=theta_stacked,
                weights=weights,
                dt=DT, alpha=alpha, A_target=A_target, beta=beta,
                lam_phi=lam_phi, lam_chance=lam_chance,
                bin_stride=bin_stride,
            )
            ctrl_cfg_kwargs = dict(base_ctrl_cfg)
            ctrl_cfg_kwargs.update(cfg_overrides)
            ctrl_cfg = SMCControlConfig(**ctrl_cfg_kwargs)

            t_replan = time.time()
            res_ctrl = run_tempered_smc_loop_native(
                spec=spec, cfg=ctrl_cfg, seed=42 + s,
                print_progress=False,
            )
            elapsed_replan = time.time() - t_replan
            mean_schedule = np.asarray(res_ctrl['mean_schedule'])  # (plan_n_steps, 2)
            n_days_in_plan = T_total_days
            sched_per_day = (mean_schedule[:n_days_in_plan * BINS_PER_DAY]
                             .reshape(n_days_in_plan, BINS_PER_DAY, 2)
                             .mean(axis=1))
            current_schedule_per_day = sched_per_day.astype(np.float64)
            phi_summary = (f"day0=({sched_per_day[0,0]:.2f},{sched_per_day[0,1]:.2f}), "
                           f"day_mid=({sched_per_day[n_days_in_plan//2,0]:.2f},"
                           f"{sched_per_day[n_days_in_plan//2,1]:.2f})")
            print(f"  plan: {res_ctrl['n_temp_levels']}lvl, "
                  f"cost={float(res_ctrl['particle_costs'].mean()):+.2f}, "
                  f"{phi_summary}  ({elapsed_replan:.0f}s)", flush=True)
            replan_records.append({
                'stride':            int(s),
                'plan_per_day':      sched_per_day.copy(),
                'mean_theta':        np.asarray(res_ctrl['mean_theta']),
                'particle_costs':    np.asarray(res_ctrl['particle_costs']),
                'n_temp_levels':     int(res_ctrl['n_temp_levels']),
                'elapsed_s':         elapsed_replan,
                'beta_max':          float(res_ctrl['beta_max']),
            })
        else:
            print()

    total_elapsed = time.time() - total_t0
    print()
    print(f"  Total: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s) "
          f"for {n_strides} strides + {len(replan_records)} replans")

    # ── Post-hoc trajectory analysis ──
    full_traj = np.concatenate(plant.history['trajectory'], axis=0)
    A_traj = full_traj[:, 3]
    A_integral_observed = float(np.sum(A_traj) * DT)
    mean_A = float(A_traj.mean())

    full_phi = np.asarray(np.concatenate(plant.history['Phi_value'], axis=0))
    full_phi = full_phi.reshape(-1, 2)
    n_steps_full = min(full_phi.shape[0], full_traj.shape[0])
    posthoc = posthoc_eval(
        [dict(truth)], np.array([1.0]),
        full_phi[:n_steps_full],
        dt=DT, alpha=alpha, A_target=A_target,
        truth_params_template=truth,
        initial_state=init_state,
    )
    weighted_violation_rate = float(posthoc['weighted_violation_rate'])
    posthoc_mean_A_integral = float(posthoc['mean_A_integral'])

    # ── Acceptance gates (4 standard gates per Stage 3 plan) ──
    n_id_covered_per_window = [r['id_covered'] for r in window_records]
    n_id_total = len(all_param_names)
    cov_thresh = int(np.ceil(0.80 * n_id_total))
    n_pass_id = sum(1 for c in n_id_covered_per_window if c >= cov_thresh)
    pass_thresh_windows = int(np.ceil(0.80 * len(window_records)))

    gate1_A = bool(A_integral_observed >= A_target)
    gate2_violation = bool(weighted_violation_rate <= alpha)
    gate3_id = bool(n_pass_id >= pass_thresh_windows)
    gate4_compute = bool(total_elapsed <= 4 * 3600)
    all_pass = gate1_A and gate2_violation and gate3_id and gate4_compute

    print()
    print(f"  Acceptance gates (Stage 3, all 4 required):")
    print(f"    {'PASS' if gate1_A else 'FAIL'}  integral A dt >= {A_target}: "
          f"observed = {A_integral_observed:.2f}")
    print(f"    {'PASS' if gate2_violation else 'FAIL'}  weighted violation <= {alpha}: "
          f"observed = {weighted_violation_rate:.4f}")
    print(f"    {'PASS' if gate3_id else 'FAIL'}  filter id_cov >= {cov_thresh}/{n_id_total} "
          f"for >= {pass_thresh_windows}/{len(window_records)} windows: "
          f"{n_pass_id}/{len(window_records)}")
    print(f"    {'PASS' if gate4_compute else 'FAIL'}  total compute <= 4h: "
          f"observed = {total_elapsed/3600:.2f}h")
    print(f"  {'ALL GATES PASS' if all_pass else 'ONE OR MORE GATES FAIL'}")

    # ── Save artifacts ──
    np.savez(
        out_dir / "trajectory.npz",
        trajectory=full_traj,
        applied_phi_per_stride=applied_phi_per_stride,
        plant_state_per_stride=plant_state_per_stride,
        full_phi=full_phi,
    )

    posterior_array = np.stack([
        r['particles_constrained'] for r in window_records
    ])
    np.savez(
        out_dir / "posterior.npz",
        posterior=posterior_array,
        param_names=np.array(all_param_names, dtype=object),
        truth_vec=np.array([truth[n] for n in all_param_names]),
        elapsed_s=np.array([r['elapsed_s'] for r in window_records]),
        id_covered=np.array([r['id_covered'] for r in window_records]),
        n_temp=np.array([r['n_temp'] for r in window_records]),
    )
    np.savez(
        out_dir / "replan_records.npz",
        stride=np.array([r['stride'] for r in replan_records]),
        mean_thetas=np.stack([r['mean_theta'] for r in replan_records]),
        plans_per_day=np.stack([r['plan_per_day'] for r in replan_records]),
        n_temp_levels=np.array([r['n_temp_levels'] for r in replan_records]),
        elapsed_s=np.array([r['elapsed_s'] for r in replan_records]),
    )

    manifest = {
        "schema_version": 1,
        "stage": 3,
        "bench": "bench_smc_full_mpc_fsa_v5",
        "run_tag": run_tag,
        "run_number": run_num,
        "fsa_model_dev_pin": "7075436628fa8c202cf62241666fe90230c46ac1",
        "cost_variant": cost,
        "scenario": {
            "name": scenario_key,
            "description": scenario['description'],
            "init_state": init_state.tolist(),
            "baseline_phi": list(baseline_phi),
        },
        "T_total_days": T_total_days,
        "step_minutes": 15,
        "BINS_PER_DAY": BINS_PER_DAY,
        "WINDOW_BINS": WINDOW_BINS,
        "STRIDE_BINS": STRIDE_BINS,
        "DT": DT,
        "n_strides": n_strides,
        "n_filter_windows": len(window_records),
        "n_replans": len(replan_records),
        "replan_K": replan_K,
        "param_names": all_param_names,
        "truth_params": {k: float(v) for k, v in truth.items()},
        "cost_kwargs": {
            "alpha": alpha, "A_target": A_target,
            "beta": beta if cost in ('soft', 'soft_fast') else None,
            "bin_stride": bin_stride if cost == 'soft_fast' else None,
            "lam_phi": lam_phi, "lam_chance": lam_chance,
        },
        "smc_filter_cfg": {
            "n_smc_particles": smc_filter_cfg.n_smc_particles,
            "n_pf_particles": smc_filter_cfg.n_pf_particles,
            "target_ess_frac": smc_filter_cfg.target_ess_frac,
            "max_lambda_inc": smc_filter_cfg.max_lambda_inc,
            "bridge_type": smc_filter_cfg.bridge_type,
            "sf_q1_mode": smc_filter_cfg.sf_q1_mode,
            "sf_use_q0_cov": smc_filter_cfg.sf_use_q0_cov,
            "sf_blend": smc_filter_cfg.sf_blend,
            "num_mcmc_steps": smc_filter_cfg.num_mcmc_steps,
            "hmc_step_size": smc_filter_cfg.hmc_step_size,
            "hmc_num_leapfrog": smc_filter_cfg.hmc_num_leapfrog,
        },
        "ctrl_cfg_base": base_ctrl_cfg,
        "summary": {
            "total_compute_s": total_elapsed,
            "total_compute_h": total_elapsed / 3600.0,
            "device": jax.devices()[0].platform.upper(),
            "mean_A_traj": mean_A,
            "A_integral_observed": A_integral_observed,
            "posthoc_mean_A_integral": posthoc_mean_A_integral,
            "weighted_violation_rate": weighted_violation_rate,
            "n_id_covered_per_window": n_id_covered_per_window,
            "n_pass_id_geq_threshold": n_pass_id,
            "id_cov_threshold": cov_thresh,
            "pass_threshold_windows": pass_thresh_windows,
            "applied_phi_max": float(applied_phi_per_stride.max()),
            "applied_phi_min": float(applied_phi_per_stride.min()),
            "final_state": plant.state.tolist(),
            "gates": {
                "A_integral_geq_target": gate1_A,
                "violation_leq_alpha": gate2_violation,
                "id_cov_geq_thresh_for_window_majority": gate3_id,
                "compute_leq_4h": gate4_compute,
                "all_pass": all_pass,
            },
        },
    }
    with open(out_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── Plots ──
    state_names = ['B', 'S', 'F', 'A', 'K_FB', 'K_FS']
    t_days = np.arange(full_traj.shape[0]) * DT

    # Latent + applied
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, name in enumerate(state_names):
        ax = axes[i]
        ax.plot(t_days, full_traj[:, i], color='steelblue', lw=1.0)
        ax.set_xlabel('days'); ax.set_ylabel(name)
        ax.set_title(f'{name}'); ax.grid(True, alpha=0.3)
    axes[6].plot(np.arange(n_strides) * STRIDE_BINS / BINS_PER_DAY,
                  applied_phi_per_stride[:, 0], 'o-', color='steelblue',
                  label='Phi_B')
    axes[6].plot(np.arange(n_strides) * STRIDE_BINS / BINS_PER_DAY,
                  applied_phi_per_stride[:, 1], 'o-', color='darkorange',
                  label='Phi_S')
    axes[6].set_xlabel('days'); axes[6].set_ylabel('Phi')
    axes[6].set_title('Applied Phi'); axes[6].legend(fontsize=9); axes[6].grid(True, alpha=0.3)
    axes[7].set_visible(False)
    plt.suptitle(f'Stage 3 (FSA-v5) -- {scenario_key} / {cost}: '
                  f'mean A={mean_A:.3f}, '
                  f'integral A dt={A_integral_observed:.2f}, '
                  f'violation={weighted_violation_rate:.3f}',
                  fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "Stage3_full_mpc_summary.png", dpi=120)
    plt.close()

    # Param trace plot (all 37)
    n_params_all = len(all_param_names)
    n_cols = 6
    n_rows = (n_params_all + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3 * n_rows))
    axes = axes.flatten()
    n_w = len(window_records)
    x = np.arange(n_w)
    for i, name in enumerate(all_param_names):
        ax = axes[i]
        means = [r['particles_constrained'][:, i].mean() for r in window_records]
        q05s = [np.quantile(r['particles_constrained'][:, i], 0.05)
                for r in window_records]
        q95s = [np.quantile(r['particles_constrained'][:, i], 0.95)
                for r in window_records]
        per_w_cov = [
            (np.quantile(r['particles_constrained'][:, i], 0.05) <= truth[name]
             <= np.quantile(r['particles_constrained'][:, i], 0.95))
            for r in window_records
        ]
        cov_frac = float(np.mean(per_w_cov))
        color = 'steelblue' if cov_frac >= 0.5 else 'darkorange'
        ax.plot(x, means, 'o-', color=color, markersize=2, lw=0.8)
        ax.fill_between(x, q05s, q95s, alpha=0.25, color=color)
        ax.axhline(truth[name], color='red', linestyle='--', lw=0.8)
        ax.set_title(f'{name}  cov={cov_frac:.0%}', fontsize=9)
        ax.tick_params(labelsize=7); ax.grid(True, alpha=0.3)
    for i in range(n_params_all, len(axes)):
        axes[i].set_visible(False)
    plt.suptitle(f'Stage 3 (FSA-v5) -- closed-loop posterior trace '
                  f'({cost} / {scenario_key}): '
                  f'{n_pass_id}/{n_w} windows >={cov_thresh}/{n_id_total} covered',
                  fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_dir / "Stage3_param_traces.png", dpi=120)
    plt.close()

    # Basin overlay -- daily-mean applied (Phi_B, Phi_S) path on the v5
    # closed-island regime classification. Same diagnostic as Stage 2.
    n_days_run = max(1, n_strides * STRIDE_BINS // BINS_PER_DAY)
    n_strides_per_day = max(1, BINS_PER_DAY // STRIDE_BINS)
    truncated = applied_phi_per_stride[:n_days_run * n_strides_per_day]
    daily_phi = truncated.reshape(n_days_run, n_strides_per_day, 2).mean(axis=1)
    try:
        from version_3.tools.plot_basin_overlay import plot_basin_overlay
        plot_basin_overlay(
            daily_phi, out_dir / "basin_overlay.png",
            title=f"{run_tag}\ncost={cost}, scenario={scenario_key}",
            baseline_phi=tuple(baseline_phi),
        )
        basin_status = f"    {out_dir}/basin_overlay.png"
    except Exception as e:
        basin_status = f"    basin_overlay.png FAILED: {e}"

    print()
    print(f"  Artifacts written:")
    print(f"    {out_dir}/manifest.json")
    print(f"    {out_dir}/trajectory.npz, posterior.npz, replan_records.npz")
    print(f"    {out_dir}/Stage3_full_mpc_summary.png")
    print(f"    {out_dir}/Stage3_param_traces.png")
    print(basin_status)
    print("=" * 76)


if __name__ == '__main__':
    main()
