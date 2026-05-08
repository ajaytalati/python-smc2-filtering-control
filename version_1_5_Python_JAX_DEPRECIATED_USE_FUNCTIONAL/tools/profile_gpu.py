"""GPU profiling for the FSA v1.5 Python+JAX bench. Direct semantic port
of `version_2_Julia/tools/profile_gpu.jl` — same surface, same metrics,
same RTX-5090 reference (104.8 TFLOPS fp32 peak).

Two profile modes:
  --filter     : profile the GK-DPF v3-lite log-density factory call
                 (the per-particle log-likelihood the filter HMC drives).
  --controller : profile the controller cost_fn (the RBF cost rollout
                 over the planning horizon).
  --both       : default — runs both.

Reports for each kernel:
  - kernel wall time (median over warm calls)
  - threads in flight (batch size × inner-loop length)
  - approximate FLOPs / call (hand-counted dominant fp ops)
  - effective TFLOPS (flops / time)
  - % of RTX-5090 peak (104.8 fp32 TFLOPS)
  - GPU memory used / available (via nvidia-smi)
  - host syncs per call (count of `jax.block_until_ready` invocations,
    JAX's analogue to `CUDA.synchronize`).

Usage:
  PYTHONPATH=.:.. JAX_ENABLE_X64=True \\
    python tools/profile_gpu.py [--filter|--controller|--both] [--verbose] [--trace]

Optional `--trace` wraps the timed loop in `jax.profiler.trace(...)` so
the run can be opened in TensorBoard's Profile tab for kernel-level
breakdown. Off by default — adds ~30 s and ~100 MB to the run.
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse
import shutil
import statistics
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp


# ── RTX 5090 spec ──────────────────────────────────────────────────────
FP32_PEAK_TFLOPS = 104.8       # boost-clock peak per NVIDIA spec sheet


def _bytes_to_gb(b: float) -> float:
    return b / 1e9


def _gpu_mem_state() -> tuple:
    """Query (used_GB, total_GB, util_pct_str) via nvidia-smi.
    Returns (None, None, '?') if nvidia-smi isn't available."""
    if shutil.which('nvidia-smi') is None:
        return None, None, '?'
    try:
        out = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        line = out.stdout.strip().split('\n')[0]
        used_mib, total_mib, util = [s.strip() for s in line.split(',')]
        used_gb  = float(used_mib)  / 1024.0
        total_gb = float(total_mib) / 1024.0
        return used_gb, total_gb, util
    except Exception:
        return None, None, '?'


def _print_gpu_state(label: str) -> None:
    used, total, util = _gpu_mem_state()
    if used is None:
        print(f"  [{label}] GPU mem: <nvidia-smi unavailable>; util: {util}%")
    else:
        pct = 100.0 * used / total if total else 0.0
        print(f"  [{label}] GPU mem: {used:.2f} / {total:.2f} GB used "
              f"({pct:.1f}%);  util: {util}%")


# ── Filter PF profile (mirrors profile_filter in the Julia file) ───────

def profile_filter(*, n_smc: int = 32, k_per_chain: int = 200,
                    verbose: bool = False, n_repeat: int = 5,
                    trace: bool = False) -> float:
    print("=" * 72)
    print("PROFILE: filter PF kernel (GK-DPF v3-lite log-density, vmap'd)")
    print("=" * 72)
    _print_gpu_state("at start")

    # Imports here so XLA env vars (above) are honoured.
    from models.fsa_high_res.simulation import (
        BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE,
    )
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_V15_ESTIMATION, COLD_START_INIT,
    )
    from smc2fc.core.config import SMCConfig
    from smc2fc.filtering.gk_dpf_v3_lite import (
        make_gk_dpf_v3_lite_log_density_compileonce,
    )
    from smc2fc.transforms.unconstrained import constrained_to_unconstrained

    em = HIGH_RES_FSA_V15_ESTIMATION
    WINDOW_BINS = BINS_PER_DAY        # 1-day window at default 60-min grid

    # SMC config — only the fields the factory reads.
    smc_cfg = SMCConfig(
        n_smc_particles=n_smc, n_pf_particles=k_per_chain,
        target_ess_frac=0.5, max_lambda_inc=0.20,
        bridge_type='schrodinger_follmer',
        sf_q1_mode='annealed',
        sf_use_q0_cov=True, sf_blend=0.7,
        sf_annealed_n_stages=3, sf_annealed_n_mh_steps=5,
        sf_info_aware=False,
        num_mcmc_steps=3, hmc_step_size=0.05, hmc_num_leapfrog=4,
        num_mcmc_steps_bridge=3, max_lambda_inc_bridge=0.20,
    )

    print(f"Config: n_smc={n_smc}, K_per_chain={k_per_chain}, "
          f"T_steps={WINDOW_BINS}")

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
    _print_gpu_state("after factory build")

    # Synthetic 1-day obs window — direct Gaussian on each latent at
    # truth + small noise. Matches v1.5's obs schema (3 channels).
    rng = np.random.default_rng(0)
    obs_data = {
        'obs_B': INIT_STATE['B'] + 0.005 * rng.standard_normal(WINDOW_BINS),
        'obs_F': INIT_STATE['F'] + 0.005 * rng.standard_normal(WINDOW_BINS),
        'obs_A': INIT_STATE['A'] + 0.005 * rng.standard_normal(WINDOW_BINS),
        'Phi':   np.ones(WINDOW_BINS, dtype=np.float64),
    }
    grid_obs = em.align_obs_fn(obs_data, WINDOW_BINS, DT_BIN_DAYS)

    # Build a batch of θ at the prior mean in unconstrained space.
    truth_constrained = jnp.array(
        [DEFAULT_PARAMS[n] for n in em.all_names], dtype=jnp.float64,
    )
    theta_unc_one = jnp.asarray(
        constrained_to_unconstrained(truth_constrained, T_arr),
        dtype=jnp.float64,
    )
    theta_batch = jnp.tile(theta_unc_one[None, :], (n_smc, 1))

    # Bind dynamic data into the factory and vmap over θ rows. JIT the
    # vmapped callable so the first call compiles, subsequent calls hit
    # the cache.
    key0 = jax.random.PRNGKey(42)
    ld = jax.tree_util.Partial(
        log_density_factory,
        grid_obs=grid_obs,
        fixed_init_state=COLD_START_INIT,
        w_start=jnp.asarray(0, dtype=jnp.int32),
        key0=key0,
    )
    ld_batched = jax.jit(jax.vmap(ld))

    # Warm up — forces XLA compilation; subsequent calls hit the cache.
    print("warming up (XLA compile)...")
    out = ld_batched(theta_batch)
    jax.block_until_ready(out)
    _print_gpu_state("after warmup")

    # Time many calls.
    print(f"timing {n_repeat} calls...")
    times = []
    if trace:
        jax.profiler.start_trace('./jax_trace_filter')
    for _ in range(n_repeat):
        t0 = time.time()
        out = ld_batched(theta_batch)
        jax.block_until_ready(out)
        times.append(time.time() - t0)
    if trace:
        jax.profiler.stop_trace()

    t_med = statistics.median(times)
    n_threads = n_smc * k_per_chain
    # Hand-counted dominant ops per (chain × particle) per bin.
    # Mirrors Julia's count: drift ~30 ops × 4 substeps + 3-channel
    # diagonal-Gaussian log-pdf ~30 ops + reflection ~10 ops.
    flops_per_thread = WINDOW_BINS * (30 * 4 + 30 + 10)
    flops = n_threads * flops_per_thread
    tflops_eff = flops / t_med / 1e12

    print()
    print("Results:")
    print(f"  median time per call : {t_med:.4f} s")
    print(f"  threads in flight    : {n_threads} ({n_smc} chains × "
          f"{k_per_chain} state particles)")
    print(f"  approx FLOPs / call  : {flops:.2e}")
    print(f"  effective TFLOPS     : {tflops_eff:.2f} / "
          f"{FP32_PEAK_TFLOPS:.1f} peak  =  "
          f"{100 * tflops_eff / FP32_PEAK_TFLOPS:.1f}% util")
    _print_gpu_state("after timing")
    print(f"  host syncs per call  : 1 (block_until_ready)")
    if trace:
        print("  trace dumped to ./jax_trace_filter/")
    if verbose:
        print(f"  raw times (s)        : {[round(t, 4) for t in times]}")
    print()
    return t_med


# ── Controller cost profile (mirrors profile_controller) ───────────────

def profile_controller(*, n_smc: int = 256, n_inner: int = 64,
                        n_anchors: int = 8, T_days: float = 14.0,
                        verbose: bool = False, n_repeat: int = 3,
                        trace: bool = False) -> float:
    print("=" * 72)
    print("PROFILE: controller cost kernel (build_control_spec.cost_fn, vmap'd)")
    print("=" * 72)
    _print_gpu_state("at start")

    from models.fsa_high_res.simulation import DT_BIN_DAYS, BINS_PER_DAY
    from models.fsa_high_res.control import build_control_spec

    n_steps = int(round(T_days * BINS_PER_DAY))
    print(f"Config: n_smc={n_smc}, n_inner={n_inner}, n_anchors={n_anchors}, "
          f"n_steps={n_steps} (T={T_days}d × {BINS_PER_DAY} bins/day)")

    spec = build_control_spec(
        T_total=T_days, dt_days=DT_BIN_DAYS, n_anchors=n_anchors,
        n_inner=n_inner,
    )
    _print_gpu_state("after spec build")

    # Batch of θ at the prior mean (zero in unconstrained space).
    rng = np.random.default_rng(0)
    theta_batch = jnp.asarray(
        rng.standard_normal((n_smc, n_anchors)), dtype=jnp.float64,
    )

    cost_batched = jax.jit(jax.vmap(spec.cost_fn))

    print("warming up (XLA compile)...")
    out = cost_batched(theta_batch)
    jax.block_until_ready(out)
    _print_gpu_state("after warmup")

    print(f"timing {n_repeat} calls...")
    times = []
    if trace:
        jax.profiler.start_trace('./jax_trace_controller')
    for _ in range(n_repeat):
        t0 = time.time()
        out = cost_batched(theta_batch)
        jax.block_until_ready(out)
        times.append(time.time() - t0)
    if trace:
        jax.profiler.stop_trace()

    t_med = statistics.median(times)
    n_threads = n_smc * n_inner
    # Hand-counted dominant ops per (chain × inner trial) per bin.
    # Same shape as Julia's profile_controller: RBF decode (~10 ops ×
    # n_anchors) + drift (~30 ops × n_substeps=4) + diffusion (~12) +
    # cost accumulator (~6).
    flops_per_thread = n_steps * (10 * n_anchors + 30 * 4 + 12 + 6)
    flops = n_threads * flops_per_thread
    tflops_eff = flops / t_med / 1e12

    print()
    print("Results:")
    print(f"  median time per call : {t_med:.4f} s")
    print(f"  threads in flight    : {n_threads} ({n_smc} chains × "
          f"{n_inner} CRN trials)")
    print(f"  approx FLOPs / call  : {flops:.2e}")
    print(f"  effective TFLOPS     : {tflops_eff:.2f} / "
          f"{FP32_PEAK_TFLOPS:.1f} peak  =  "
          f"{100 * tflops_eff / FP32_PEAK_TFLOPS:.1f}% util")
    _print_gpu_state("after timing")
    print(f"  host syncs per call  : 1 (block_until_ready)")
    if trace:
        print("  trace dumped to ./jax_trace_controller/")
    if verbose:
        print(f"  raw times (s)        : {[round(t, 4) for t in times]}")
    print()
    return t_med


# ── Main ──────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description='FSA v1.5 Python+JAX GPU profiler')
    ap.add_argument('--filter', action='store_true',
                    help='profile the filter PF kernel only')
    ap.add_argument('--controller', action='store_true',
                    help='profile the controller cost kernel only')
    ap.add_argument('--both', action='store_true',
                    help='(default) profile both')
    ap.add_argument('--verbose', action='store_true',
                    help='print raw per-call times')
    ap.add_argument('--trace', action='store_true',
                    help='dump jax.profiler trace to ./jax_trace_*/')
    ap.add_argument('--n-smc', type=int, default=32,
                    help='filter outer SMC² particles (matches bench default)')
    ap.add_argument('--k-pf', type=int, default=200,
                    help='filter inner PF particles per chain')
    ap.add_argument('--ctrl-n-smc', type=int, default=256,
                    help='controller outer SMC² particles')
    ap.add_argument('--ctrl-n-inner', type=int, default=64,
                    help='controller inner CRN-MC trials')
    ap.add_argument('--ctrl-T-days', type=float, default=14.0,
                    help='controller planning horizon (days)')
    return ap.parse_args()


def main():
    args = parse_args()
    do_filter = args.filter or args.both or (not args.controller)
    do_ctrl   = args.controller or args.both or (not args.filter)
    if not args.filter and not args.controller and not args.both:
        # No mode flag → default both
        do_filter = do_ctrl = True

    devices = jax.devices()
    print(f"\nJAX device: {devices[0].platform.upper()} "
          f"({devices[0].device_kind if hasattr(devices[0], 'device_kind') else '?'})")
    used, total, _ = _gpu_mem_state()
    if total is not None:
        print(f"GPU memory: {total:.2f} GB total, {total - used:.2f} GB free")
    print()

    if do_filter:
        profile_filter(n_smc=args.n_smc, k_per_chain=args.k_pf,
                        verbose=args.verbose, trace=args.trace)
    if do_ctrl:
        profile_controller(n_smc=args.ctrl_n_smc, n_inner=args.ctrl_n_inner,
                            T_days=args.ctrl_T_days,
                            verbose=args.verbose, trace=args.trace)

    print("Notes:")
    print("  - 'effective TFLOPS' is approximate (counts only the dominant fp")
    print("    ops in the inner loop). Useful for relative comparison across")
    print("    stacks at matched config; not a substitute for jax.profiler.trace.")
    print("  - 'util %' compares to RTX 5090's 104.8 TFLOPS fp32 peak.")
    print("    Saturating to >50%% requires:")
    print("      • enough threads in flight (>= ~1M for 5090's 21504 cores)")
    print("      • compute-bound (not memory-bound) inner loop")
    print("      • no host syncs blocking the stream")
    print("  - JAX `block_until_ready` is the analogue of CUDA's synchronize().")


if __name__ == '__main__':
    main()
