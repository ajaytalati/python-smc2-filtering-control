"""FSA v1.5 closed-loop SMC²-MPC bench (Python+JAX).

Mirrors `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` and
v2's `bench_smc_full_mpc_fsa.py` but slimmed to v1.5's surface:
3-state (B, F, A), single Φ control, 3 direct-Gaussian obs channels.

Pipeline per stride:
  1. plant_rollout(state, Φ_subdaily, params, dt, key) → trajectory + obs
  2. accumulate obs, build window grid_obs at the right cadence
  3. (every K strides) outer SMC² filter window → posterior particles
  4. (every K strides) tempered SMC² controller → new daily Φ plan
  5. apply new plan for the next K strides

Defaults are intentionally small so the bench runs as a smoke test in
under a minute. The realistic 14-day run (with N_smc=256, K_pf=400)
is a follow-up — the structure here is the integration check.

CLI:
    python tools/bench_smc_full_mpc_fsa_v15.py [--T-days 14] [--smoke]
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description='FSA v1.5 closed-loop SMC²-MPC')
    ap.add_argument('--T-days', type=int, default=14,
                    help='total horizon in days')
    ap.add_argument('--replan-K', type=int, default=2,
                    help='replan every K strides')
    ap.add_argument('--n-smc', type=int, default=32,
                    help='filter outer SMC² particles')
    ap.add_argument('--k-pf', type=int, default=200,
                    help='filter inner PF particles per chain')
    ap.add_argument('--ctrl-n-smc', type=int, default=128,
                    help='controller outer SMC² particles')
    ap.add_argument('--ctrl-n-inner', type=int, default=32,
                    help='controller inner CRN-MC trials')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--smoke', action='store_true',
                    help='single stride only — sanity check that imports + '
                         'first-call paths work')
    ap.add_argument('--out-dir', type=str, default='',
                    help='output dir; empty → auto under outputs/fsa_v15/')
    return ap.parse_args()


def main():
    args = parse_args()

    # Imports here so that env-vars (JAX_ENABLE_X64, XLA_…) are set first.
    from models.fsa_high_res._dynamics import drift_jax, diffusion_state_dep
    from models.fsa_high_res.simulation import (
        DEFAULT_PARAMS, INIT_STATE, BINS_PER_DAY, DT_BIN_DAYS,
    )
    from models.fsa_high_res._plant import init_plant_state, plant_rollout
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V15_ESTIMATION, COLD_START_INIT,
    )
    from models.fsa_high_res.control import build_control_spec

    # Framework entry points (same as v2).
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.jax_native_smc import (
        run_smc_window_native, run_smc_window_bridge_native,
    )
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density_compileonce,
    )
    from smc2fc.control import SMCControlConfig
    from smc2fc.control.tempered_smc_loop import run_tempered_smc_loop_native
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained

    # ── Window / stride geometry ───────────────────────────────────────
    STRIDE_BINS = BINS_PER_DAY // 2          # 12-h stride at default 1-h grid
    WINDOW_BINS = BINS_PER_DAY               # 1-day filter window
    n_strides = (args.T_days * BINS_PER_DAY) // STRIDE_BINS

    if args.smoke:
        n_strides = 1                          # one stride only — smoke test

    print('=' * 76)
    print(f"  FSA v1.5 closed-loop SMC²-MPC  (T = {args.T_days} d, "
          f"step = {60 // (BINS_PER_DAY // 24)} min, "
          f"BINS_PER_DAY = {BINS_PER_DAY})")
    print(f"  device = {jax.devices()[0].platform.upper()}")
    print(f"  strides = {n_strides}, replan every K = {args.replan_K}")
    if args.smoke:
        print("  *** SMOKE MODE: 1 stride only ***")
    print('=' * 76)

    em = HIGH_RES_FSA_V15_ESTIMATION

    # ── Initial conditions ────────────────────────────────────────────
    plant_state = init_plant_state()
    daily_phi_plan = np.full(args.T_days, 1.0, dtype=np.float64)   # Φ ≡ 1.0 baseline

    # Filter config — small defaults so smoke run is fast
    smc_cfg = SMCConfig(
        n_smc_particles=args.n_smc, n_pf_particles=args.k_pf,
        target_ess_frac=0.5, max_lambda_inc=0.20,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed',
        sf_use_q0_cov=True, sf_blend=0.7,
        sf_annealed_n_stages=3, sf_annealed_n_mh_steps=5,
        sf_info_aware=False,
        num_mcmc_steps=3, hmc_step_size=0.05, hmc_num_leapfrog=4,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.20,
    )
    # Controller config
    ctrl_cfg = SMCControlConfig(
        n_smc=args.ctrl_n_smc, n_inner=args.ctrl_n_inner, sigma_prior=1.5,
        target_ess_frac=0.5, max_lambda_inc=0.20,
        num_mcmc_steps=8, hmc_step_size=0.2, hmc_num_leapfrog=16,
        beta_max_target_nats=8.0, max_temp_steps=25,
    )

    # Compile-once log-density factory for the filter
    log_density_factory = make_gk_dpf_v3_lite_log_density_compileonce(
        model=em, n_particles=smc_cfg.n_pf_particles,
        bandwidth_scale=smc_cfg.bandwidth_scale,
        ot_ess_frac=smc_cfg.ot_ess_frac,
        ot_temperature=smc_cfg.ot_temperature,
        ot_max_weight=smc_cfg.ot_max_weight,
        ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
        ot_epsilon=smc_cfg.ot_epsilon,
        dt=DT_BIN_DAYS, t_steps=WINDOW_BINS,
    )
    T_arr = log_density_factory._transforms

    # ── Per-stride accumulators ───────────────────────────────────────
    accumulated_obs = {'obs_B': [], 'obs_F': [], 'obs_A': [], 'Phi': []}
    full_traj = []
    daily_phi_per_stride = []
    replan_history = []

    prev_particles = None
    fixed_init_state = COLD_START_INIT
    last_replan_stride = 0
    key = jax.random.PRNGKey(args.seed)

    total_t0 = time.time()

    for s in range(n_strides):
        t0 = time.time()

        # day_in_plan within the current planning horizon
        day_in_plan = (s - last_replan_stride) * STRIDE_BINS // BINS_PER_DAY
        Phi_today = float(daily_phi_plan[min(day_in_plan,
                                                args.T_days - 1)])
        # Per-bin Φ for this stride (constant within the stride; v1.5
        # has no Φ-burst so no sub-daily expansion is needed).
        Phi_subdaily = np.full(STRIDE_BINS, Phi_today, dtype=np.float64)

        # Plant step
        key, plant_key = jax.random.split(key)
        rollout = plant_rollout(plant_state, Phi_subdaily,
                                 DEFAULT_PARAMS, DT_BIN_DAYS, plant_key)
        plant_state = rollout['final_state']
        full_traj.append(rollout['trajectory'])
        daily_phi_per_stride.append(Phi_today)
        for ch in ('obs_B', 'obs_F', 'obs_A'):
            accumulated_obs[ch].extend(rollout[ch].tolist())
        accumulated_obs['Phi'].extend(Phi_subdaily.tolist())

        # ── Filter step (every stride, on the latest WINDOW_BINS bins) ──
        n_obs = len(accumulated_obs['obs_B'])
        if n_obs >= WINDOW_BINS:
            window_obs = {k: np.asarray(v[-WINDOW_BINS:], dtype=np.float64)
                          for k, v in accumulated_obs.items()}
            grid_obs = em.align_obs_fn(window_obs, WINDOW_BINS, DT_BIN_DAYS)

            key, smc_key = jax.random.split(key)
            ld = jax.tree_util.Partial(
                log_density_factory,
                grid_obs=grid_obs,
                fixed_init_state=fixed_init_state,
                w_start=jnp.asarray(0, dtype=jnp.int32),
                key0=smc_key,
            )

            if prev_particles is None:
                particles, elapsed_f, n_temp = run_smc_window_native(
                    ld, em, T_arr, cfg=smc_cfg,
                    initial_particles=None, seed=args.seed + s * 1000,
                )
            else:
                particles, elapsed_f, n_temp = run_smc_window_bridge_native(
                    new_ld=ld, prev_particles=prev_particles,
                    model=em, T_arr=T_arr, cfg=smc_cfg,
                    seed=args.seed + s * 1000,
                )
            prev_particles = particles
            print(f"  stride {s+1}/{n_strides}: filter {n_temp} levels, "
                  f"{elapsed_f:.1f}s")
        else:
            print(f"  stride {s+1}/{n_strides}: warmup (< window)")

        # ── Replan every K strides (skipped in smoke mode) ──────────────
        if (not args.smoke
            and s > 0
            and s % args.replan_K == 0
            and prev_particles is not None):

            # Posterior-mean params (constrained space)
            samp = np.array([
                np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
                for p in np.asarray(prev_particles)
            ])
            posterior_mean = {n: float(samp[:, i].mean())
                               for i, n in enumerate(em.all_names)}
            # Build v1.5 dict (the 4 pinned values are added by control.py
            # using DEFAULT_PARAMS as fallback).
            from models.fsa_high_res.simulation import (
                fill_pinned, params_v15_to_v1,
            )
            params_v15 = fill_pinned(posterior_mean)
            params_v15.update({k: DEFAULT_PARAMS[k] for k in
                                ('sigma_B_obs', 'sigma_F_obs', 'sigma_A_obs')})

            # Build a fresh ControlSpec around the posterior-mean params,
            # taking the current plant state as the new initial condition.
            t_remaining_days = (n_strides - s) * STRIDE_BINS / BINS_PER_DAY
            init_state_dict = dict(B=float(plant_state.bfa[0]),
                                    F=float(plant_state.bfa[1]),
                                    A=float(plant_state.bfa[2]))
            ctrl_spec = build_control_spec(
                T_total=t_remaining_days, dt_days=DT_BIN_DAYS,
                params_v15=params_v15, init_state=init_state_dict,
                n_inner=args.ctrl_n_inner,
            )

            key, ctrl_key = jax.random.split(key)
            res = run_tempered_smc_loop_native(
                ctrl_spec, ctrl_cfg, ctrl_key,
            )
            # `res` carries `mean_theta` → schedule → daily mean Φ
            mean_theta = jnp.asarray(res['mean_theta'])
            Phi_plan = ctrl_spec.schedule_from_theta(mean_theta)
            # Decode to per-day mean Φ
            Phi_plan_arr = np.asarray(Phi_plan, dtype=np.float64)
            n_plan_bins = Phi_plan_arr.shape[0]
            n_plan_days = n_plan_bins // BINS_PER_DAY
            new_daily = Phi_plan_arr[:n_plan_days * BINS_PER_DAY].reshape(
                n_plan_days, BINS_PER_DAY).mean(axis=1)
            # Splice into daily_phi_plan starting from the next day
            day_now = (s * STRIDE_BINS) // BINS_PER_DAY
            for di in range(min(n_plan_days, args.T_days - day_now)):
                daily_phi_plan[day_now + di] = new_daily[di]
            last_replan_stride = s
            replan_history.append({
                'stride': s,
                'plan_per_day': new_daily.tolist(),
                'n_temp': int(res.get('n_temp', 0)),
            })
            print(f"    replan: new mean Φ over next {n_plan_days} d = "
                  f"{new_daily.mean():.3f}")

        elapsed = time.time() - t0
        print(f"  stride done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    print()
    print(f"  total: {total_elapsed/60:.1f} min "
          f"({total_elapsed:.0f}s) for {n_strides} strides")

    # ── Save artefacts ────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (Path(__file__).resolve().parent.parent / 'outputs'
                   / 'fsa_v15' / f'run_T{args.T_days}'
                   f'{"_smoke" if args.smoke else ""}_seed{args.seed}')
    out_dir.mkdir(parents=True, exist_ok=True)

    full_traj_arr = (np.concatenate(full_traj, axis=0)
                       if full_traj else np.zeros((0, 3)))
    np.savez(out_dir / 'trajectory.npz',
              trajectory=full_traj_arr,
              applied_phi_per_stride=np.array(daily_phi_per_stride),
              accumulated_obs_B=np.array(accumulated_obs['obs_B']),
              accumulated_obs_F=np.array(accumulated_obs['obs_F']),
              accumulated_obs_A=np.array(accumulated_obs['obs_A']),
              accumulated_Phi=np.array(accumulated_obs['Phi']))
    manifest = dict(
        bench='bench_smc_full_mpc_fsa_v15',
        T_days=args.T_days,
        n_strides=n_strides,
        replan_K=args.replan_K,
        n_smc=args.n_smc, k_pf=args.k_pf,
        ctrl_n_smc=args.ctrl_n_smc, ctrl_n_inner=args.ctrl_n_inner,
        smoke=args.smoke,
        seed=args.seed,
        device=jax.devices()[0].platform,
        total_elapsed_s=total_elapsed,
        replan_history=replan_history,
    )
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f"  artefacts written to {out_dir}/")
    print('=' * 76)


if __name__ == '__main__':
    main()
