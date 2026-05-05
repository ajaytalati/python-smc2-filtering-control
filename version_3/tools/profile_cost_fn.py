"""Principled JAX profiler for the v5 chance-constrained cost.

Goal: empirically identify what's leaving the RTX 5090 ~80% idle on
the soft_fast controller path, before guessing optimisations.

How it works:

  1. Build a cost_fn for the requested variant (soft / soft_fast).
  2. Warm it up (single call, block_until_ready, drop the JIT-compile time).
  3. Run N timed forward calls + N timed `jax.grad` calls (the leapfrog
     uses both). `block_until_ready` after each so wall-clock includes
     GPU compute, not just host-side dispatch.
  4. Inside a `jax.profiler.start_trace(...)` context capture a TensorBoard
     trace for visual inspection.
  5. Print: warm wall-clock (mean, p95, min, max) for forward + grad,
     plus a kernel-count summary scraped from the trace if available.

Usage:

    cd version_3
    PYTHONPATH=.:.. python tools/profile_cost_fn.py \
        --cost soft_fast --n-smc 128 --n-truth-particles 1 \
        --n-iters 20 --bin-stride 4

  --cost              soft | soft_fast (default soft_fast)
  --n-smc             outer SMC particle count for the cost vmap
                      (default 128 for soft_fast, 256 for soft)
  --n-truth-particles inner per-cost-call particle count (default 1)
  --n-iters           number of timed forward+grad calls (default 20)
  --bin-stride        soft_fast separatrix sub-sample stride (default 4)
  --trace-dir         where to dump the .trace files
                      (default outputs/fsa_v5/profiles/<timestamp>/)
  --no-trace          skip the start_trace pass (just measure wall-clock)

Output goes both to stdout AND to a JSON file under the trace dir, so
we can diff configs.

Use TensorBoard to inspect:
    tensorboard --logdir version_3/outputs/fsa_v5/profiles/
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',
                       os.path.expanduser('~/.jax_compilation_cache'))

import argparse
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp


def _build_cost_fn(*, cost_kind: str, n_steps: int, n_anchors: int,
                     n_truth_particles: int, dt: float, alpha: float,
                     A_target: float, beta: float,
                     lam_phi: float, lam_chance: float,
                     bin_stride: int):
    """Mirror version_3/tools/bench_controller_only_fsa_v5.py:
    _build_cost_chance_constrained but stripped of the Stage-2-loop
    wrapping. Returns a single cost_fn(theta_ctrl) -> scalar."""
    from version_3.models.fsa_v5.control_v5 import (
        _cost_soft_jit, _stack_particle_dicts, _ensure_v5_keys,
        TRUTH_PARAMS_V5,
    )
    from version_3.models.fsa_v5.control_v5_fast import _cost_soft_fast_jit
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

    truth_list = [dict(TRUTH_PARAMS_V5) for _ in range(n_truth_particles)]
    theta_stacked = _stack_particle_dicts(truth_list)
    theta_stacked = _ensure_v5_keys(theta_stacked, TRUTH_PARAMS_V5)
    weights = jnp.full((n_truth_particles,), 1.0 / n_truth_particles,
                        dtype=jnp.float64)
    init_state = jnp.array([0.50, 0.45, 0.20, 0.45, 0.06, 0.07])

    if cost_kind == 'soft':
        scale = 0.1
        @jax.jit
        def cost_fn(theta_ctrl):
            Phi_schedule = schedule_from_theta(theta_ctrl)
            out = _cost_soft_jit(theta_stacked, weights, Phi_schedule,
                                  init_state, dt, alpha, A_target,
                                  beta, scale)
            violation_excess = jnp.maximum(0.0,
                                            out['weighted_violation_rate'] - alpha)
            return (lam_phi * out['mean_effort']
                    - out['mean_A_integral']
                    + lam_chance * violation_excess ** 2)

    elif cost_kind == 'soft_fast':
        scale = 0.1
        theta_stacked_f32 = {k: v.astype(jnp.float32)
                             for k, v in theta_stacked.items()}
        weights_f32 = weights.astype(jnp.float32)
        init_state_f32 = init_state.astype(jnp.float32)
        @jax.jit
        def cost_fn(theta_ctrl):
            Phi_schedule = schedule_from_theta(theta_ctrl).astype(jnp.float32)
            out = _cost_soft_fast_jit(theta_stacked_f32, weights_f32,
                                       Phi_schedule, init_state_f32,
                                       dt, alpha, A_target,
                                       beta, scale, bin_stride)
            violation_excess = jnp.maximum(0.0,
                                            out['weighted_violation_rate'] - alpha)
            return jnp.float64(lam_phi * out['mean_effort']
                                - out['mean_A_integral']
                                + lam_chance * violation_excess ** 2)
    else:
        raise ValueError(f"--cost must be soft / soft_fast; got {cost_kind!r}")

    theta_dim = 2 * n_anchors
    return cost_fn, theta_dim


def _bench_calls(cost_fn, theta, n_iters: int, label: str):
    """Time `cost_fn(theta)` n_iters times; force GPU sync each call."""
    # Warm-up + JIT compile
    out = cost_fn(theta)
    out.block_until_ready()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        out = cost_fn(theta)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return _summarise(times, label)


def _bench_grad(cost_fn, theta, n_iters: int, label: str):
    """Time `jax.grad(cost_fn)(theta)` n_iters times."""
    grad_fn = jax.jit(jax.grad(cost_fn))
    g = grad_fn(theta)
    g.block_until_ready()

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        g = grad_fn(theta)
        g.block_until_ready()
        times.append(time.perf_counter() - t0)
    return _summarise(times, label)


def _summarise(times: list, label: str) -> dict:
    arr = np.array(times) * 1000.0   # ms
    summary = {
        'label':     label,
        'n_calls':   len(arr),
        'mean_ms':   float(arr.mean()),
        'median_ms': float(np.median(arr)),
        'min_ms':    float(arr.min()),
        'max_ms':    float(arr.max()),
        'p95_ms':    float(np.percentile(arr, 95)),
        'std_ms':    float(arr.std()),
    }
    print(f"  {label:20s}  mean={summary['mean_ms']:7.1f}  "
          f"median={summary['median_ms']:7.1f}  "
          f"min={summary['min_ms']:7.1f}  "
          f"max={summary['max_ms']:7.1f}  "
          f"p95={summary['p95_ms']:7.1f}  ms (n={summary['n_calls']})")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cost', default='soft_fast', choices=['soft', 'soft_fast'])
    ap.add_argument('--n-smc', type=int, default=None,
                     help='outer SMC particle count for the vmap. Default: '
                          '128 for soft_fast, 256 for soft. The cost_fn is '
                          'vmapped over this many particles per leapfrog step.')
    ap.add_argument('--n-truth-particles', type=int, default=1)
    ap.add_argument('--n-iters', type=int, default=20)
    ap.add_argument('--n-anchors', type=int, default=8)
    ap.add_argument('--T-days', type=int, default=14)
    ap.add_argument('--bin-stride', type=int, default=4)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--A-target', type=float, default=2.0)
    ap.add_argument('--beta', type=float, default=50.0)
    ap.add_argument('--lam-phi', type=float, default=0.1)
    ap.add_argument('--lam-chance', type=float, default=100.0)
    ap.add_argument('--trace-dir', default=None,
                     help='Directory for jax.profiler.start_trace output. '
                          'Default: outputs/fsa_v5/profiles/<timestamp>_<cfg>/')
    ap.add_argument('--no-trace', action='store_true',
                     help='Skip the start_trace pass; only measure wall-clock.')
    args = ap.parse_args()

    if args.n_smc is None:
        args.n_smc = 128 if args.cost == 'soft_fast' else 256

    print("=" * 76)
    print(f"  v5 cost-fn profiler -- cost={args.cost}")
    print(f"  n_smc(vmap_outer)={args.n_smc}, n_truth_particles={args.n_truth_particles}")
    print(f"  T={args.T_days}d, n_anchors={args.n_anchors}, bin_stride={args.bin_stride}")
    print(f"  device:  {jax.devices()[0]}")
    print("=" * 76)

    n_steps = args.T_days * 96
    cost_fn, theta_dim = _build_cost_fn(
        cost_kind=args.cost,
        n_steps=n_steps, n_anchors=args.n_anchors,
        n_truth_particles=args.n_truth_particles,
        dt=1.0/96, alpha=args.alpha, A_target=args.A_target,
        beta=args.beta, lam_phi=args.lam_phi, lam_chance=args.lam_chance,
        bin_stride=args.bin_stride,
    )

    # Single-particle theta for the bare cost_fn call
    theta = jnp.zeros((theta_dim,))

    # Vmapped cost (mimics the SMC kernel's per-leapfrog vmap of cost_fn
    # over n_smc particles): batched theta shape (n_smc, theta_dim).
    @jax.jit
    def vmapped_cost(thetas_batched):
        return jax.vmap(cost_fn)(thetas_batched).sum()

    @jax.jit
    def vmapped_grad(thetas_batched):
        return jax.grad(vmapped_cost)(thetas_batched)

    rng = jax.random.PRNGKey(0)
    thetas_batched = jax.random.normal(rng, (args.n_smc, theta_dim))

    print()
    print("Warm-up + timed calls (forward + jax.grad), n_iters=", args.n_iters)
    print()
    print("  Per-particle (bare) cost_fn(theta) -- represents the cost of a single")
    print("  particle's leapfrog half-step:")
    s_fwd      = _bench_calls(cost_fn, theta, args.n_iters,
                                'cost_fn forward')
    s_grad     = _bench_grad(cost_fn,  theta, args.n_iters,
                                'jax.grad fwd+bwd')
    print()
    print(f"  Vmapped cost (n_smc={args.n_smc}) -- represents one leapfrog half-step")
    print("  for the entire outer-loop SMC particle cloud at once:")
    s_v_fwd  = _bench_calls(vmapped_cost, thetas_batched, args.n_iters,
                                'vmapped fwd')
    s_v_grad = _bench_calls(vmapped_grad, thetas_batched, args.n_iters,
                                'vmapped grad')

    # Per-leapfrog math: the SMC tempered controller does
    # `num_mcmc_steps × hmc_num_leapfrog` cost+grad evaluations per
    # tempering level. For soft_fast: 5 × 8 = 40. There are typically
    # ~10-15 tempering levels per replan. So ~400-600 vmapped-grad
    # calls per replan.
    leap_per_level = 5 * 8   # soft_fast trim
    levels_per_replan = 12   # rough average from current run
    grad_calls_per_replan = leap_per_level * levels_per_replan
    proj_replan_s = grad_calls_per_replan * s_v_grad['mean_ms'] / 1000.0
    print()
    print(f"  Projection: {grad_calls_per_replan} vmapped-grad calls per replan "
          f"× {s_v_grad['mean_ms']:.1f} ms/call ≈ {proj_replan_s:.1f}s/replan")

    # Save summary JSON
    if args.trace_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_tag = f"{args.cost}_K{args.n_smc}_T{args.T_days}d_truth{args.n_truth_particles}"
        repo_root = Path(__file__).resolve().parent.parent
        trace_dir = repo_root / "outputs" / "fsa_v5" / "profiles" / f"{ts}_{cfg_tag}"
    else:
        trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'cost':                args.cost,
        'n_smc':               args.n_smc,
        'n_truth_particles':   args.n_truth_particles,
        'T_days':              args.T_days,
        'n_anchors':           args.n_anchors,
        'bin_stride':          args.bin_stride,
        'theta_dim':           theta_dim,
        'device':              str(jax.devices()[0]),
        'cost_fn_forward':     s_fwd,
        'cost_fn_grad':        s_grad,
        'vmapped_forward':     s_v_fwd,
        'vmapped_grad':        s_v_grad,
        'projected_replan_s':  proj_replan_s,
    }
    with open(trace_dir / "wallclock_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Wall-clock summary -> {trace_dir}/wallclock_summary.json")

    # Optional jax.profiler trace pass
    if not args.no_trace:
        print()
        print(f"  jax.profiler.start_trace -> {trace_dir}/")
        jax.profiler.start_trace(str(trace_dir))
        for _ in range(args.n_iters):
            with jax.profiler.TraceAnnotation('vmapped_grad'):
                g = vmapped_grad(thetas_batched)
                g.block_until_ready()
        jax.profiler.stop_trace()
        print(f"  Trace files dumped. Inspect with:")
        print(f"      tensorboard --logdir {trace_dir.parent}")

    print()
    print("=" * 76)


if __name__ == '__main__':
    main()
