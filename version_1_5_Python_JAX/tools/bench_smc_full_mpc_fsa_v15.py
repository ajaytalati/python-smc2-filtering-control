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
import csv
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
    # Controller knobs — defaults match the v1.5 Julia bench
    # (`bench_smc_full_mpc_fsa_gpu.jl`) so a no-flag invocation produces
    # the same configuration on both stacks.
    ap.add_argument('--ctrl-n-smc', type=int, default=256,
                    help='controller outer SMC² particles')
    ap.add_argument('--ctrl-n-inner', type=int, default=64,
                    help='controller inner CRN-MC trials per cost evaluation')
    ap.add_argument('--ctrl-num-mcmc', type=int, default=8,
                    help='controller HMC moves per tempering level')
    ap.add_argument('--ctrl-hmc-step', type=float, default=0.2,
                    help='controller HMC leapfrog step size')
    ap.add_argument('--ctrl-hmc-leap', type=int, default=16,
                    help='controller HMC leapfrog trajectory length')
    ap.add_argument('--ctrl-target-nats', type=float, default=8.0,
                    help='controller β_max auto-calibration target (nats)')
    ap.add_argument('--ctrl-max-levels', type=int, default=25,
                    help='controller hard cap on tempering levels per replan')
    ap.add_argument('--ctrl-max-lambda-inc', type=float, default=0.20,
                    help='controller max λ increment per tempering bisection')
    ap.add_argument('--ctrl-sigma-prior', type=float, default=1.5,
                    help='controller prior std on θ_ctrl RBF coefficients')
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
    # Controller config — every knob exposed at CLI so config can match
    # `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` 1:1.
    ctrl_cfg = SMCControlConfig(
        n_smc=args.ctrl_n_smc,
        n_inner=args.ctrl_n_inner,
        sigma_prior=args.ctrl_sigma_prior,
        target_ess_frac=0.5,
        max_lambda_inc=args.ctrl_max_lambda_inc,
        num_mcmc_steps=args.ctrl_num_mcmc,
        hmc_step_size=args.ctrl_hmc_step,
        hmc_num_leapfrog=args.ctrl_hmc_leap,
        beta_max_target_nats=args.ctrl_target_nats,
        max_temp_steps=args.ctrl_max_levels,
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

    # Per-stride telemetry — written to per_stride.csv at end of bench so
    # the comparison script (Phase D) can plot per-stride wall time and
    # tempering levels alongside the Julia equivalent.
    per_stride_log = []   # list of dicts; columns flushed to CSV at end

    # Per-stride posterior particle clouds (constrained space) for the
    # param-traces plot. `posterior_mask[s]=True` iff the filter
    # actually ran on stride s (i.e. we had ≥ WINDOW_BINS of obs).
    n_params = em.n_params
    posterior_particles = np.zeros(
        (n_strides, args.n_smc, n_params), dtype=np.float64,
    )
    posterior_mask = np.zeros(n_strides, dtype=bool)

    prev_particles = None
    fixed_init_state = COLD_START_INIT
    last_replan_stride = 0
    key = jax.random.PRNGKey(args.seed)

    total_t0 = time.time()

    for s in range(n_strides):
        t0 = time.time()
        n_temp_filter_s = 0   # 0 if filter didn't run this stride (warmup)
        n_temp_ctrl_s   = 0   # 0 if no replan happened this stride

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
            # Snapshot constrained posterior into per-stride store for
            # the param-traces plot at end of bench.
            samp_constrained = np.array([
                np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
                for p in np.asarray(particles)
            ])
            posterior_particles[s, :samp_constrained.shape[0], :] = samp_constrained
            posterior_mask[s] = True
            n_temp_filter_s = int(n_temp)
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
            n_temp_ctrl_s = int(res.get('n_temp', 0))
            replan_history.append({
                'stride': s,
                'plan_per_day': new_daily.tolist(),
                'n_temp': n_temp_ctrl_s,
            })
            print(f"    replan: new mean Φ over next {n_plan_days} d = "
                  f"{new_daily.mean():.3f}")

        elapsed = time.time() - t0
        # Per-stride telemetry record (rolling A-mean uses the
        # accumulated trajectory so far, including this stride's bins).
        if full_traj:
            full_so_far = np.concatenate(full_traj, axis=0)
            A_mean_so_far = float(np.mean(full_so_far[:, 2]))
            B_end = float(full_so_far[-1, 0])
            F_end = float(full_so_far[-1, 1])
            A_end = float(full_so_far[-1, 2])
        else:
            A_mean_so_far = B_end = F_end = A_end = float('nan')
        per_stride_log.append(dict(
            stride=s,
            t_wall_s=elapsed,
            n_temp_filter=n_temp_filter_s,
            n_temp_ctrl=n_temp_ctrl_s,
            daily_phi=Phi_today,
            A_mean_so_far=A_mean_so_far,
            B_end=B_end, F_end=F_end, A_end=A_end,
        ))
        print(f"  stride done in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    print()
    print(f"  total: {total_elapsed/60:.1f} min "
          f"({total_elapsed:.0f}s) for {n_strides} strides")

    full_traj_arr = (np.concatenate(full_traj, axis=0)
                       if full_traj else np.zeros((0, 3)))

    # ── Baseline reference: rerun the plant under constant Φ = 1.0
    #    for the same number of bins, so the plot can compare MPC vs
    #    canonical Banister baseline (matches Julia bench schema).
    print("  building baseline (constant Φ = 1.0) reference …")
    n_mpc_bins = full_traj_arr.shape[0]
    if n_mpc_bins > 0:
        baseline_state = init_plant_state()
        Phi_base = np.full(n_mpc_bins, 1.0, dtype=np.float64)
        key, base_key = jax.random.split(key)
        base_rollout = plant_rollout(baseline_state, Phi_base,
                                       DEFAULT_PARAMS, DT_BIN_DAYS, base_key)
        traj_baseline = base_rollout['trajectory']
    else:
        traj_baseline = np.zeros_like(full_traj_arr)

    # ── Save artefacts ────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (Path(__file__).resolve().parent.parent / 'outputs'
                   / 'fsa_v15' / f'run_T{args.T_days}'
                   f'{"_smoke" if args.smoke else ""}_seed{args.seed}')
    out_dir.mkdir(parents=True, exist_ok=True)

    param_names = list(em.all_names)
    # Truth values for each estimated param: read from v1.5 DEFAULT_PARAMS
    # (the basis matches `params_v15` so the names line up directly).
    truth_params = {n: float(DEFAULT_PARAMS[n]) for n in param_names
                     if n in DEFAULT_PARAMS}

    np.savez(out_dir / 'trajectory.npz',
              trajectory_mpc=full_traj_arr,
              trajectory_baseline=traj_baseline,
              applied_phi_per_stride=np.array(daily_phi_per_stride),
              accumulated_obs_B=np.array(accumulated_obs['obs_B']),
              accumulated_obs_F=np.array(accumulated_obs['obs_F']),
              accumulated_obs_A=np.array(accumulated_obs['obs_A']),
              accumulated_Phi=np.array(accumulated_obs['Phi']),
              posterior_particles=posterior_particles,
              posterior_window_mask=posterior_mask,
              param_names=np.array(param_names, dtype=object),
              BINS_PER_DAY=BINS_PER_DAY,
              STRIDE_BINS=STRIDE_BINS,
              dt_days=DT_BIN_DAYS)
    manifest = dict(
        bench='bench_smc_full_mpc_fsa_v15',
        T_days=args.T_days,
        n_strides=n_strides,
        replan_K=args.replan_K,
        n_smc=args.n_smc, k_pf=args.k_pf,
        ctrl_n_smc=args.ctrl_n_smc,
        ctrl_n_inner=args.ctrl_n_inner,
        ctrl_num_mcmc=args.ctrl_num_mcmc,
        ctrl_hmc_step=args.ctrl_hmc_step,
        ctrl_hmc_leap=args.ctrl_hmc_leap,
        ctrl_target_nats=args.ctrl_target_nats,
        ctrl_max_levels=args.ctrl_max_levels,
        ctrl_max_lambda_inc=args.ctrl_max_lambda_inc,
        ctrl_sigma_prior=args.ctrl_sigma_prior,
        smoke=args.smoke,
        seed=args.seed,
        device=jax.devices()[0].platform,
        total_elapsed_s=total_elapsed,
        replan_history=replan_history,
        param_names=param_names,
        truth_params=truth_params,
        BINS_PER_DAY=int(BINS_PER_DAY),
        STRIDE_BINS=int(STRIDE_BINS),
        WINDOW_BINS=int(WINDOW_BINS),
        step_minutes=60 // (BINS_PER_DAY // 24),
    )
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))

    # ── Per-stride telemetry CSV (Phase B; consumed by the comparison
    #    script in Phase D). One row per stride.
    if per_stride_log:
        csv_cols = ['stride', 't_wall_s', 'n_temp_filter', 'n_temp_ctrl',
                    'daily_phi', 'A_mean_so_far', 'B_end', 'F_end', 'A_end']
        with open(out_dir / 'per_stride.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=csv_cols)
            w.writeheader()
            for row in per_stride_log:
                w.writerow(row)

    # ── 4-panel state-trajectory plot (mirrors version_2_Julia/tools/
    #    plot_state_traces.jl exactly: same panels, colours, labels) ──
    if n_mpc_bins > 0:
        plot_path = out_dir / f'v15_T{args.T_days}d_traces.png'
        _plot_state_traces(
            traj_mpc=full_traj_arr,
            traj_baseline=traj_baseline,
            daily_phi_per_stride=np.array(daily_phi_per_stride),
            BINS_PER_DAY=BINS_PER_DAY,
            STRIDE_BINS=STRIDE_BINS,
            dt_days=DT_BIN_DAYS,
            F_max=0.40,
            T_days=args.T_days,
            out_path=plot_path,
        )
        print(f"  wrote {plot_path}")

    # ── Posterior parameter-traces plot (mirrors v2's
    #    plot_param_traces.py: per-param 5/95 quantile band + median +
    #    truth horizontal line, indexed by end-of-window day). ──
    if posterior_mask.any():
        param_path = out_dir / f'v15_T{args.T_days}d_param_traces.png'
        _plot_param_traces(
            posterior_particles=posterior_particles,
            posterior_mask=posterior_mask,
            param_names=param_names,
            truth=truth_params,
            n_strides=n_strides,
            stride_bins=STRIDE_BINS,
            window_bins=WINDOW_BINS,
            bins_per_day=BINS_PER_DAY,
            T_days=args.T_days,
            step_minutes=60 // (BINS_PER_DAY // 24),
            mean_A_mpc=float(np.mean(full_traj_arr[:, 2])),
            mean_A_base=float(np.mean(traj_baseline[:, 2])),
            out_path=param_path,
        )
        print(f"  wrote {param_path}")

    print(f"  artefacts written to {out_dir}/")
    print('=' * 76)


def _plot_state_traces(*, traj_mpc, traj_baseline, daily_phi_per_stride,
                         BINS_PER_DAY, STRIDE_BINS, dt_days, F_max,
                         T_days, out_path):
    """4-panel state-trajectory plot. Direct port of
    `version_2_Julia/tools/plot_state_traces.jl:plot_state_traces` —
    same layout (top: B + F, bottom: A + applied Φ), same colours
    (blue MPC-B, dark-red MPC-F, green MPC-A, orange Φ, grey baseline),
    same labels. Matplotlib backend `Agg` so no X display needed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_bins = traj_mpc.shape[0]
    t_days = np.arange(n_bins) * dt_days
    n_strides = len(daily_phi_per_stride)
    stride_start_days = np.arange(n_strides) * STRIDE_BINS * dt_days

    mean_A_mpc  = float(np.mean(traj_mpc[:, 2]))
    mean_A_base = float(np.mean(traj_baseline[:, 2]))

    blue, darkred, green, orange = '#1f77b4', '#8b0000', '#2ca02c', '#ff7f0e'
    grey, redline = '#7f7f7f', '#d62728'

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), dpi=120)
    fig.suptitle(
        f"FSA v1.5 closed-loop MPC, T={t_days[-1]:.1f}d. "
        f"mean A {mean_A_mpc:.3f} vs baseline {mean_A_base:.3f}",
        fontsize=10,
    )

    # Top-left: B
    ax = axes[0, 0]
    ax.plot(t_days, traj_baseline[:, 0], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='B (baseline)')
    ax.plot(t_days, traj_mpc[:, 0], color=blue, lw=1.6, label='B (MPC)')
    ax.set_title('B trajectory', fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('B', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Top-right: F + F_max
    ax = axes[0, 1]
    ax.plot(t_days, traj_baseline[:, 1], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='F (baseline)')
    ax.plot(t_days, traj_mpc[:, 1], color=darkred, lw=1.6, label='F (MPC)')
    ax.axhline(F_max, color=redline, ls='--', lw=1.0, label='F_max')
    ax.set_title('F trajectory', fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('F', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-left: A
    ax = axes[1, 0]
    ax.plot(t_days, traj_baseline[:, 2], color=grey, ls='--', alpha=0.7,
            lw=1.0, label='A (baseline)')
    ax.plot(t_days, traj_mpc[:, 2], color=green, lw=1.6, label='A (MPC)')
    ax.set_title(
        f'A trajectory  (mean MPC: {mean_A_mpc:.3f}, '
        f'baseline: {mean_A_base:.3f})', fontsize=10,
    )
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('A', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-right: applied daily Φ per stride
    ax = axes[1, 1]
    ax.plot(stride_start_days, daily_phi_per_stride, color=orange, lw=2.0,
            marker='o', markersize=3, label='applied daily Φ')
    ax.axhline(1.0, color=grey, ls='--', lw=1.0, label='baseline Φ=1.0')
    ax.set_title(f'MPC-applied Φ schedule across {n_strides} strides',
                 fontsize=11)
    ax.set_xlabel('time (days)', fontsize=8)
    ax.set_ylabel('daily Φ', fontsize=8)
    ax.set_ylim(0.0, max(1.5, float(np.max(daily_phi_per_stride)) * 1.1))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)


def _plot_param_traces(*, posterior_particles, posterior_mask, param_names,
                         truth, n_strides, stride_bins, window_bins,
                         bins_per_day, T_days, step_minutes,
                         mean_A_mpc, mean_A_base, out_path):
    """Posterior parameter-traces plot. Direct port of
    `version_2_Python_JAX/tools/plot_param_traces.py:main` — one panel
    per parameter showing the 5-95% quantile band + median over rolling
    windows, with the truth value as a horizontal red dashed line.

    `posterior_particles` is (n_strides, n_smc, n_params); `posterior_mask`
    is (n_strides,) of bool indicating which strides have valid filter
    output (warmup strides — before the first full WINDOW — are False).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    end_t_days = (np.arange(n_strides) * stride_bins + window_bins) / bins_per_day
    end_t_days = end_t_days[posterior_mask]

    valid = posterior_particles[posterior_mask]   # (n_valid, n_smc, n_params)
    q05 = np.quantile(valid, 0.05, axis=1)
    q50 = np.quantile(valid, 0.50, axis=1)
    q95 = np.quantile(valid, 0.95, axis=1)

    n_params = len(param_names)
    n_cols = 5
    n_rows = (n_params + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.0, n_rows * 2.2),
                              sharex=True, dpi=120)
    axes = axes.flatten() if n_rows > 1 else axes

    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.fill_between(end_t_days, q05[:, i], q95[:, i],
                        color='C0', alpha=0.3, label='5-95%')
        ax.plot(end_t_days, q50[:, i], color='C0', lw=1.4, label='median')
        if name in truth:
            ax.axhline(truth[name], color='red', lw=1.0, ls='--',
                       label='truth')
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(n_params, len(axes)):
        axes[j].axis('off')

    for ax in axes[-n_cols:]:
        ax.set_xlabel('end of window (days)', fontsize=8)

    fig.suptitle(
        f"FSA-v1.5 posterior parameter traces — T={T_days}d, "
        f"h={step_minutes}min, n_strides={n_strides}, "
        f"mean A {mean_A_mpc:.3f} vs baseline {mean_A_base:.3f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
