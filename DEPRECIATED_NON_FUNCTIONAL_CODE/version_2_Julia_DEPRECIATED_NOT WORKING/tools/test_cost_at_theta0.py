#!/usr/bin/env python
"""Bit-for-bit cost comparison Python vs Julia — Python side.

Standing-hypothesis test from julia_fsa_writeup.pdf §2.7 item 1:
fix θ (start with θ = 0), use a saved CRN noise grid, print the
per-trial A_acc / B_acc / barrier_acc that the Python cost_fn
computes. The Julia twin script then reads the same noise and prints
the Julia kernel's values; the comparison localises any A-cost
divergence.

Run from version_2/:
    cd version_2 && PYTHONPATH=.:.. python ../version_2_Julia/tools/test_cost_at_theta0.py

Outputs:
    /tmp/crn_noise_seed42.npy     — (N_INNER, N_STEPS, 3) fp64 Wiener increments
    /tmp/python_cost_at_theta0.npz — per-trial A/B/F/Phi/barrier accumulators
"""

from __future__ import annotations

import os
os.environ.setdefault('JAX_ENABLE_X64', 'True')

import numpy as np
import jax
import jax.numpy as jnp

from models.fsa_high_res._dynamics import (
    TRUTH_PARAMS, drift_jax, diffusion_state_dep,
)
from models.fsa_high_res.control import _make_schedule, INIT_STATE


# ── Config — must match the Julia twin ──────────────────────────────────
OUT_DIR      = "/tmp"
N_INNER      = 32
T_DAYS       = 14
BINS_PER_DAY = 96
N_STEPS      = T_DAYS * BINS_PER_DAY            # 1344
DT           = 1.0 / BINS_PER_DAY               # 15-min outer step
N_SUBSTEPS   = 4
N_ANCHORS    = 8
F_MAX        = 0.40
PHI_MAX      = 3.0
PHI_DEFAULT  = 1.0
SEED         = 42


# ── Step 1: generate fp64 noise and save ────────────────────────────────
rng    = np.random.default_rng(SEED)
wiener = rng.standard_normal((N_INNER, N_STEPS, 3))   # fp64
noise_path = f"{OUT_DIR}/crn_noise_seed{SEED}.npy"
np.save(noise_path, wiener)
print(f"Saved noise → {noise_path}  shape={wiener.shape}  dtype={wiener.dtype}")


# ── Step 2: build em_step + per-trial diagnostic cost in fp64 ───────────
rbf, schedule_from_theta = _make_schedule(
    n_steps=N_STEPS, dt=DT, n_anchors=N_ANCHORS,
    Phi_default=PHI_DEFAULT, Phi_max=PHI_MAX,
)

p_jax  = {k: jnp.asarray(float(v)) for k, v in TRUTH_PARAMS.items()}
sub_dt = DT / float(N_SUBSTEPS)
sqrt_dt = jnp.sqrt(DT)


@jax.jit
def em_step(y, Phi_t, noise_3d):
    def sub_body(y_inner, _):
        return y_inner + sub_dt * drift_jax(y_inner, p_jax, Phi_t), None
    y_det, _ = jax.lax.scan(sub_body, y, jnp.arange(N_SUBSTEPS))
    sigma_y  = diffusion_state_dep(y_det, p_jax)
    y_pred   = y_det + sigma_y * sqrt_dt * noise_3d
    B_pred, F_pred, A_pred = y_pred[0], y_pred[1], y_pred[2]
    B_next = jnp.where(B_pred < 0.0, -B_pred,
                        jnp.where(B_pred > 1.0, 2.0 - B_pred, B_pred))
    F_next = jnp.abs(F_pred)
    A_next = jnp.abs(A_pred)
    return jnp.array([B_next, F_next, A_next])


init_arr = jnp.array([INIT_STATE['B'], INIT_STATE['F'], INIT_STATE['A']])


@jax.jit
def per_trial_components(theta, w_seq):
    """Return (A_acc, B_acc, F_acc, Phi_acc, barrier_acc) for one MC trial.

    Same accumulator semantics as the production cost_fn in
    version_2/models/fsa_high_res/control.py (PRE-step state, not POST-step).
    """
    Phi_arr = schedule_from_theta(theta)

    def step(carry, k):
        y, A_acc, B_acc, F_acc, Phi_acc, barrier_acc = carry
        Phi_t  = Phi_arr[k]
        y_next = em_step(y, Phi_t, w_seq[k])
        # PRE-step accumulation (matches Python's production cost_fn).
        B_acc       = B_acc + y[0] * DT
        F_acc       = F_acc + y[1] * DT
        A_acc       = A_acc + y[2] * DT
        Phi_acc     = Phi_acc + Phi_t * Phi_t * DT
        barrier_acc = barrier_acc + jnp.maximum(y[1] - F_MAX, 0.0) ** 2 * DT
        return (y_next, A_acc, B_acc, F_acc, Phi_acc, barrier_acc), None

    init_carry = (init_arr,
                   jnp.float64(0.0), jnp.float64(0.0),
                   jnp.float64(0.0), jnp.float64(0.0), jnp.float64(0.0))
    (_, A_acc, B_acc, F_acc, Phi_acc, barrier_acc), _ = jax.lax.scan(
        step, init_carry, jnp.arange(N_STEPS),
    )
    return A_acc, B_acc, F_acc, Phi_acc, barrier_acc


per_trial_batched = jax.vmap(per_trial_components, in_axes=(None, 0))


# ── Step 3: evaluate at two θ values ────────────────────────────────────
# θ=0   → Φ ≡ Phi_default = 1.0 (canonical Banister baseline)
# θ_RO  → recovery → overload (anti-symmetric, writeup §2.4 bimodal-test
#         spec: low early, high late). At Phi_max=3, c_Phi=logit(1/3)≈
#         -0.693, so per-anchor sigmoid raw values:
#           θ=-1.5 → raw≈-2.19 → Φ ≈ 0.30
#           θ= 0.0 → raw≈-0.69 → Φ ≈ 1.00
#           θ=+1.5 → raw≈+0.81 → Φ ≈ 2.07
#         Pattern: rest then overload halves of the horizon.
theta0  = jnp.zeros(N_ANCHORS, dtype=jnp.float64)
theta_RO = jnp.array([-1.5, -1.5, -1.5, 0.0, 0.0, +1.5, +1.5, +1.5],
                      dtype=jnp.float64)

fixed_w = jnp.asarray(wiener, dtype=jnp.float64)

results = {}
for label, theta in [("theta0", theta0), ("theta_RO", theta_RO)]:
    A_accs, B_accs, F_accs, Phi_accs, barrier_accs = per_trial_batched(theta, fixed_w)
    Phi_arr = np.asarray(schedule_from_theta(theta))
    results[label] = dict(
        A_acc       = np.asarray(A_accs),
        B_acc       = np.asarray(B_accs),
        F_acc       = np.asarray(F_accs),
        Phi_acc     = np.asarray(Phi_accs),
        barrier_acc = np.asarray(barrier_accs),
        Phi_arr     = Phi_arr,
        theta       = np.asarray(theta),
    )

    print()
    print("=" * 72)
    print(f"Python cost at {label}  (fp64, {N_INNER} CRN trials, T={T_DAYS}d, h=15min)")
    print("=" * 72)
    print(f"  θ           = {np.asarray(theta)}")
    print(f"  Φ(t) summary: min={Phi_arr.min():.3f}  mean={Phi_arr.mean():.3f}  "
          f"max={Phi_arr.max():.3f}")
    Phi_anchors = Phi_arr[::N_STEPS // (N_ANCHORS - 1)][:N_ANCHORS]
    print(f"  Φ at ~anchor times: {Phi_anchors}")
    print()
    print(f"  per-trial A_acc:        mean = {results[label]['A_acc'].mean():.6f}   "
          f"std = {results[label]['A_acc'].std():.6f}")
    print(f"  per-trial F_acc:        mean = {results[label]['F_acc'].mean():.6f}   "
          f"std = {results[label]['F_acc'].std():.6f}")
    print(f"  per-trial barrier_acc:  mean = {results[label]['barrier_acc'].mean():.6f}   "
          f"std = {results[label]['barrier_acc'].std():.6f}")
    print()
    print(f"  cost (-A + barrier):    mean = "
          f"{(-results[label]['A_acc'] + results[label]['barrier_acc']).mean():.6f}")
    print(f"  Mean ∫A/T:              "
          f"{results[label]['A_acc'].mean() / (N_STEPS * DT):.4f}")

# ── Step 4: save per-trial accumulators for the Julia twin ──────────────
out_path = f"{OUT_DIR}/python_cost_at_theta0.npz"
np.savez(out_path,
         # θ=0 results (legacy keys preserved for the existing Julia driver)
         A_acc       = results["theta0"]["A_acc"],
         B_acc       = results["theta0"]["B_acc"],
         F_acc       = results["theta0"]["F_acc"],
         Phi_acc     = results["theta0"]["Phi_acc"],
         barrier_acc = results["theta0"]["barrier_acc"],
         # θ_RO (recovery→overload) results
         A_acc_RO       = results["theta_RO"]["A_acc"],
         B_acc_RO       = results["theta_RO"]["B_acc"],
         F_acc_RO       = results["theta_RO"]["F_acc"],
         Phi_acc_RO     = results["theta_RO"]["Phi_acc"],
         barrier_acc_RO = results["theta_RO"]["barrier_acc"],
         theta_RO       = results["theta_RO"]["theta"],
         Phi_arr_RO     = results["theta_RO"]["Phi_arr"],
)
print(f"\nSaved per-trial results → {out_path}")
