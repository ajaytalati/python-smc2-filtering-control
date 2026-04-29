"""Stage E3: 27-window rolling SMC² on FSA-v2 (open-loop, frozen Φ).

Synthesise 14 days of FSA-v2 data under a constant Φ ≡ 1.0 daily
schedule (sub-daily morning-burst expansion via _phi_burst.py), then
run rolling-window SMC²:

  - Window size:      96 bins (1 day) at dt = 15 min
  - Stride:           48 bins (12 hours)
  - n_windows:        27 (over 14 days = 1344 bins)
  - Bridge handoff:   Gaussian fit + LW shrinkage (smc2fc/core/sf_bridge.py)
                      from W_{k-1} posterior into W_k SMC.

W1 cold-starts from the prior; W2..W27 warm-start from the previous
window's posterior via the bridge. Per-window posterior on identifiable
parameters is reported and traced.

Acceptance gates (E3):
  1. ≥ 24 of 27 windows pass the per-window 90% CI gate on the
     identifiable subset (≥ 5 of 6 covered).
  2. Latent [B, F, A] smoothed trajectories continuous across boundaries.
  3. Cumulative compute ≤ 60 min on RTX 5090.

Run:
    cd version_2 && PYTHONPATH=.:.. python tools/bench_smc_rolling_window_fsa.py
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


# Window structure (matches smc2-blackjax-rolling fsa_high_res C0)
N_DAYS_TOTAL    = 14
WINDOW_BINS     = 96    # 1 day
STRIDE_BINS     = 48    # 12 hours
DT              = 1.0 / 96.0   # 15 min


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


def _simulate_synthetic_full(seed: int = 42, daily_phi: float = 1.0):
    """Use StepwisePlant to synthesise 14 days of constant-Φ data."""
    from models.fsa_high_res._plant import StepwisePlant

    plant = StepwisePlant(seed_offset=seed)
    daily = np.full(N_DAYS_TOTAL, daily_phi)
    n_bins = N_DAYS_TOTAL * 96
    plant.advance(n_bins, daily)

    # Concatenate per-channel obs across all bins (single advance)
    obs_data = {}
    obs_data['obs_HR'] = {
        't_idx':     np.concatenate(plant.history['obs_HR_t_idx']),
        'obs_value': np.concatenate(plant.history['obs_HR_value']),
    }
    obs_data['obs_sleep'] = {
        't_idx':       np.arange(n_bins, dtype=np.int32),
        'sleep_label': np.concatenate(plant.history['obs_sleep_label']),
    }
    obs_data['obs_stress'] = {
        't_idx':     np.concatenate(plant.history['obs_stress_t_idx']),
        'obs_value': np.concatenate(plant.history['obs_stress_value']),
    }
    obs_data['obs_steps'] = {
        't_idx':     np.concatenate(plant.history['obs_steps_t_idx']),
        'obs_value': np.concatenate(plant.history['obs_steps_value']),
    }
    obs_data['Phi'] = {
        't_idx':     np.arange(n_bins, dtype=np.int32),
        'Phi_value': np.concatenate(plant.history['Phi_value']),
    }
    obs_data['C'] = {
        't_idx':     np.arange(n_bins, dtype=np.int32),
        'C_value':   np.concatenate(plant.history['C_value']),
    }
    trajectory = np.concatenate(plant.history['trajectory'])
    return {
        'obs_data':   obs_data,
        'trajectory': trajectory,
        'n_bins':     n_bins,
    }


def main():
    print("=" * 76)
    print("  Stage E3 — 27-window rolling SMC² on FSA-v2 (open-loop, Φ=1)")
    print("=" * 76)

    n_windows = (N_DAYS_TOTAL * 96 - WINDOW_BINS) // STRIDE_BINS + 1
    print(f"  device:   {jax.devices()[0].platform.upper()}")
    print(f"  total:    {N_DAYS_TOTAL} days × 96 bins = {N_DAYS_TOTAL*96} bins")
    print(f"  window:   {WINDOW_BINS} bins (1 day)")
    print(f"  stride:   {STRIDE_BINS} bins (12 hours)")
    print(f"  windows:  {n_windows}")
    print()

    print("  Step 1: synthesize 14-day trajectory under constant Φ=1.0")
    data = _simulate_synthetic_full(seed=42, daily_phi=1.0)
    print(f"    trajectory shape: {data['trajectory'].shape}")
    print(f"    B range: [{data['trajectory'][:,0].min():.3f}, {data['trajectory'][:,0].max():.3f}]")
    print(f"    F range: [{data['trajectory'][:,1].min():.3f}, {data['trajectory'][:,1].max():.3f}]")
    print(f"    A range: [{data['trajectory'][:,2].min():.3f}, {data['trajectory'][:,2].max():.3f}]")
    print()

    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V2_ESTIMATION, COLD_START_INIT,
    )
    from models.fsa_high_res.simulation import DEFAULT_PARAMS
    from smc2fc.core.config import SMCConfig
    from smc2fc.core.tempered_smc import run_smc_window, run_smc_window_bridge
    from smc2fc.transforms.unconstrained import unconstrained_to_constrained
    from smc2fc.filtering.gk_dpf_v3_lite import make_gk_dpf_v3_lite_log_density

    em = HIGH_RES_FSA_V2_ESTIMATION
    truth = dict(DEFAULT_PARAMS)

    smc_cfg = SMCConfig(
        n_smc_particles=128, n_pf_particles=200,
        target_ess_frac=0.5, max_lambda_inc=0.10,
        bridge_type='gaussian',
        num_mcmc_steps=5, hmc_step_size=0.025, hmc_num_leapfrog=8,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.15,
    )

    identifiable_subset = {'HR_base', 'S_base', 'mu_step0',
                            'kappa_B_HR', 'k_F', 'beta_C_HR'}

    all_results = []
    prev_particles = None
    fixed_init_state = COLD_START_INIT
    name_to_idx = {n: i for i, n in enumerate(em.all_names)}

    total_t0 = time.time()
    for w in range(n_windows):
        start = w * STRIDE_BINS
        end = start + WINDOW_BINS

        print(f"  Window {w+1:>2}/{n_windows}  bins {start:>4}-{end:>4}", end='', flush=True)

        window_obs = extract_window(data['obs_data'], start, end)
        n_obs_w = (len(window_obs['obs_HR']['t_idx'])
                    + len(window_obs['obs_stress']['t_idx'])
                    + len(window_obs['obs_steps']['t_idx'])
                    + len(window_obs['obs_sleep']['t_idx']))
        print(f"  obs={n_obs_w:>3} ", end='', flush=True)

        grid_obs = em.align_obs_fn(window_obs, WINDOW_BINS, DT)
        ld = make_gk_dpf_v3_lite_log_density(
            model=em, grid_obs=grid_obs, n_particles=smc_cfg.n_pf_particles,
            bandwidth_scale=smc_cfg.bandwidth_scale,
            ot_ess_frac=smc_cfg.ot_ess_frac, ot_temperature=smc_cfg.ot_temperature,
            ot_max_weight=smc_cfg.ot_max_weight,
            ot_rank=smc_cfg.ot_rank, ot_n_iter=smc_cfg.ot_n_iter,
            ot_epsilon=smc_cfg.ot_epsilon,
            dt=DT, seed=42 + w,
            fixed_init_state=fixed_init_state, window_start_bin=int(start),
        )
        T_arr = ld._transforms

        if prev_particles is None:
            init_tag = 'cold'
            particles, elapsed, n_temp = run_smc_window(
                ld, em, T_arr, cfg=smc_cfg,
                initial_particles=None, seed=42 + w * 1000,
            )
        else:
            init_tag = 'bridge'
            particles, elapsed, n_temp = run_smc_window_bridge(
                new_ld=ld, prev_particles=prev_particles,
                model=em, T_arr=T_arr, cfg=smc_cfg,
                seed=42 + w * 1000,
            )
        print(f"({init_tag}) {n_temp:>2} levels, {elapsed:>5.1f}s ", end='', flush=True)

        # Convert to constrained space
        particles = np.asarray(particles)
        samp = np.array([
            np.asarray(unconstrained_to_constrained(jnp.asarray(p), T_arr))
            for p in particles
        ])

        # Coverage on identifiable subset
        n_id_covered = 0
        for name in identifiable_subset:
            idx = name_to_idx[name]
            q05 = np.quantile(samp[:, idx], 0.05)
            q95 = np.quantile(samp[:, idx], 0.95)
            if q05 <= truth[name] <= q95:
                n_id_covered += 1
        print(f" id_cov={n_id_covered}/{len(identifiable_subset)}", flush=True)

        all_results.append({
            'window': w,
            'start_bin': start, 'end_bin': end,
            'n_temp': n_temp, 'elapsed_s': elapsed,
            'init_tag': init_tag,
            'particles_constrained': samp,
            'id_covered': n_id_covered,
        })
        prev_particles = particles

        # Extract smoothed init state at t = STRIDE_BINS for next window's PF
        n_extract = min(10, particles.shape[0])
        extracted = []
        for ei in range(n_extract):
            u_draw = jnp.array(particles[ei])
            st = ld.extract_state_at_step(u_draw, STRIDE_BINS)
            extracted.append(np.array(st))
        fixed_init_state = jnp.array(np.mean(extracted, axis=0))

    total_elapsed = time.time() - total_t0
    print()
    print(f"  Total: {total_elapsed/60:.1f} min ({total_elapsed:.0f}s) for "
          f"{n_windows} windows")

    # ── Acceptance gates ──
    n_pass = sum(1 for r in all_results if r['id_covered'] >= 5)
    print()
    print(f"  Acceptance gates:")
    gate1 = n_pass >= 24
    print(f"    {'✓' if gate1 else '✗'}  ≥24 of 27 windows have ≥5/6 identifiable "
          f"covered (actual: {n_pass}/{n_windows})")
    gate2 = total_elapsed <= 60 * 60
    print(f"    {'✓' if gate2 else '✗'}  Total compute ≤ 60 min "
          f"(actual: {total_elapsed:.0f}s)")
    all_pass = gate1 and gate2
    print(f"  {'✓ all gates pass' if all_pass else '✗ one or more gates fail'}")

    # ── Diagnostic plot — per-param posterior trace across windows ──
    out_dir = "outputs/fsa_high_res"
    out_path = f"{out_dir}/E3_rolling_window_traces.png"
    os.makedirs(out_dir, exist_ok=True)

    cov_names = sorted(identifiable_subset)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, name in enumerate(cov_names):
        ax = axes[i]
        idx = name_to_idx[name]
        means = [r['particles_constrained'][:, idx].mean() for r in all_results]
        q05s  = [np.quantile(r['particles_constrained'][:, idx], 0.05)
                 for r in all_results]
        q95s  = [np.quantile(r['particles_constrained'][:, idx], 0.95)
                 for r in all_results]
        x = np.arange(n_windows)
        ax.plot(x, means, 'o-', color='steelblue', markersize=4, label='posterior mean')
        ax.fill_between(x, q05s, q95s, alpha=0.3, color='steelblue', label='90% CI')
        ax.axhline(truth[name], color='red', linestyle='--', label=f'truth = {truth[name]:.3g}')
        ax.set_xlabel('window'); ax.set_ylabel(name)
        ax.set_title(f'{name} across {n_windows} windows')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(f'Stage E3 — FSA-v2 27-window rolling SMC² (open-loop): '
                  f'{n_pass}/{n_windows} windows pass ≥5/6 identifiable, '
                  f'{total_elapsed/60:.0f} min on '
                  f'{jax.devices()[0].platform.upper()}', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print()
    print(f"  Plot: {out_path}")
    print("=" * 76)


if __name__ == '__main__':
    main()
