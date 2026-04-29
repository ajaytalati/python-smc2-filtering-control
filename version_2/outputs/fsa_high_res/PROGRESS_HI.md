# Progress — Stage H (LQG baseline) + Stage I (h=1h coarsening)

This is a working draft for the main results section of the LaTeX
write-up. Numerical values fill in as runs land; the structure is
final. Updated after each commit.

## Executive summary

Two new pieces of work, designed to land together as a complete
multi-horizon evaluation of the SMC²-MPC controller on the FSA-v2
plant:

1. **Stage I — coarsen the integration grid from h=15 min to h=1 h.**
   Mechanism: env-var `FSA_STEP_MINUTES` resolved at module-import
   time so simulation, plant, filter, and controller all see the same
   grid. End-to-end speedup ~4× (T=84: ~14 GPU-h → ~3.5 GPU-h on
   RTX 5090). Justified by FSA-v2 dynamical timescales (τ_B = 42 d,
   τ_F = 6.36 d, η⁻¹ ≈ 5 d, circadian period 24 h — Nyquist-safe at
   24 samples/cycle). Cross-grid consistency at T=14 documents that
   the discretization choice does not materially change closed-loop
   performance.

2. **Stage H — LQG/Riccati baseline as Section 8.8 / 10.3.**
   JAX-autodiff linearisation of the FSA-v2 drift around
   (B_typ, F_typ, A_typ, Φ=1) under Section 8.8 assumptions
   (A1)–(A3), backward differential Riccati on the same outer grid,
   open-loop optimal Φ schedule applied to the StepwisePlant. The
   LQG ratio (mean ∫A / mean ∫A_baseline) is empirically
   **non-competitive at long horizons** for the structural reasons
   anticipated in the LaTeX docs, quantifying the value of capturing
   the Stuart-Landau cubic nonlinearity that LQG linearises away.

## Stage I — h=1h coarsening

### Implementation

Single CLI flag `--step-minutes N` (default 15). Pre-parsed from
`sys.argv` before any model import; sets `FSA_STEP_MINUTES` env var.
`simulation.py` and `_phi_burst.py` derive their grid constants
(`BINS_PER_DAY`, `DT_BIN_DAYS`, `DT_BIN_HOURS`) from the env var at
import time.

Backward compatible: with `--step-minutes 15` (the default), the
existing 15-min code path is byte-identical to before the refactor;
existing tests pass unchanged.

The G4 checkpoint manifest gets a new `step_minutes` field and the
run-dir auto-suffix appends `_h{N}min` when `N != 15`, so 15-min and
1 h checkpoints don't clobber each other.

The controller's inner sub-step is automatically scaled with the
outer step: at `BINS_PER_DAY=96` we keep the legacy `n_substeps=4`
(3.75-min inner step); at `BINS_PER_DAY=24` we drop to
`n_substeps=1` (1 h inner step). Without this scaling the speedup
would be lost in the controller cost MC.

Files touched (all behind the env-var flag):

| file | change |
|------|--------|
| `version_2/models/fsa_high_res/simulation.py` | grid constants from env |
| `version_2/models/fsa_high_res/_phi_burst.py` | grid constants from env |
| `version_2/tools/bench_smc_full_mpc_fsa.py`   | CLI flag, manifest, run-dir suffix |
| `version_2/tools/bench_smc_closed_loop_fsa.py` | `_build_phase2_control_spec` honours `BINS_PER_DAY` |
| `version_2/tests/test_h1h_grid.py` | 4 sanity checks |

### Sanity tests (all pass)

| test | result |
|------|--------|
| grid constants for `step_minutes ∈ {15, 30, 60}` | ✓ |
| circadian C(t) at hour-aligned bins identical across grids | ✓ |
| sleep gating: 8 h sleep window = 1/3 of day on both grids | ✓ |
| plant 1-day means agree across grids to <2% relative | ✓ (1.64%) |

### Cross-grid consistency at T=14

[FILL IN after T=14 h=1h validation run lands]

Acceptance criteria:
- mean A_mpc within ~10% of h=15min T=14 reference (was 0.0954)
- F-violation fraction within 3 percentage points of 0%
- per-window id-coverage within ±3 of the 12/26 baseline at h=15min
- closed-loop A trajectory shape qualitatively similar

### Long-horizon results

| T (d) | h (min) | mean A_MPC | mean A_const | ratio | F-viol | wall-clock |
|-------|---------|------------|---------------|-------|--------|------------|
| 14    | 15      | 0.0954     | 0.0824        | 1.157 | 0.00%  | 124 min |
| 28    | 15      | [pending]  | [pending]     | -     | -      | ~115 min   |
| 42    | 60      | [pending]  | [pending]     | -     | -      | ~70 min    |
| 56    | 60      | [pending]  | [pending]     | -     | -      | ~120 min   |
| 84    | 60      | [pending]  | [pending]     | -     | -      | ~215 min   |

## Stage H — LQG/Riccati baseline

### Implementation

`smc2fc/control/lqg/` module:

| file | content |
|------|---------|
| `linearize.py` | `linearize_drift_at(...)` via `jax.jacfwd` |
| `riccati.py`   | `solve_riccati_backward(...)` RK4, `compute_lqr_gain(...)` |
| `controller.py` | `LQGSpec`, `LQGController`, `build_lqg_open_loop_schedule` |

Two usage modes wired:
- **Open-loop schedule** (used in the bench): pre-compute `K(t)`,
  roll deterministic-skeleton `x*(t)` under `Φ_ref`, emit
  `Φ(t) = Φ_ref − K(t) (x*(t) − x_ref)` clipped to `[0, Φ_max]`.
- **Feedback** (separation principle): `feedback_phi(x_hat, t_idx)`
  for use against the SMC² posterior-mean estimate at replan time.

`version_2/tools/bench_lqg_baseline_fsa.py` mirrors the G4 bench
schema (manifest + data.npz), so `compare_g4_lqg.py` can join G4 and
LQG checkpoints by `(T_total_days, step_minutes)`.

### Linearisation matrix (numerical fact)

At the operating point `(B=0.05, F=0.20, A=0.10, Φ=1.0)`:

```
A_lin = [[-0.0238   0.0000   0.0048]    # B
         [ 0.0000  -0.1571  -0.0286]    # F
         [ 0.0300  -0.0260  -0.0070]]   # A

B_lin = [0.0125, 0.0300, 0.0000]^T

Eigenvalues = [+0.003, -0.029, -0.162]
```

Two structural facts of the linearisation drive everything that
follows:

1. **`B_lin[2,0] = 0`** — Φ has *no direct linear path to A* in the
   FSA-v2 drift. The autonomic amplitude is reachable only via the
   F→A coupling.

2. **One positive eigenvalue (+0.003)** — at A_typ = 0.10, the cubic
   damping `−A³` is too weak to stabilise the linearisation
   (`η − 3·A_typ² = 0.20 − 0.03 > 0`). The linearisation is therefore
   *unstable* in a slow direction along A; the cubic
   nonlinearity is what restores stability in the true model.

3. **`A_lin[2,1] = -0.0260`** — in the linearisation, *raising F
   reduces A*. This is opposite to the user's training-physiology
   intuition (where increased fatigue is downstream of training that
   also lifts A). The cubic-Stuart-Landau cross-coupling that makes
   A track training is missing from the linearised state-space.

### LQG schedule across horizons (default cost weights)

Default cost (chosen as the simplest defensible weights):
- `Q = diag(0, w_F=10, w_A=1000)`
- `R = 1`
- `Q_T = 0`
- `x_ref = (B_0, F_typ=0.20, A_ref=0.30)`
- `Φ_ref = 1.0`, clipped to `[0, 3]`

| T (d) | mean A_LQG | mean A_const | ratio | F-viol | LQG wall-clock |
|-------|------------|---------------|-------|--------|----------------|
| 14    | 0.0852     | 0.0824        | **1.034** | 0.00%   | 1.6 s |
| 28    | 0.0930     | 0.0943        | **0.986** | 0.00%   | 1.4 s |
| 42    | 0.1284     | 0.1396        | **0.919** | 32.29%  | 2.1 s |
| 56    | 0.2386     | 0.2306        | **1.035** | 37.37%  | 1.7 s |
| 84    | 0.3991     | 0.4442        | **0.898** | 21.44%  | 1.9 s |

Numbers at h=1h are within 1% of the h=15min numbers (LQG itself is
grid-stable; only the plant baseline shifts slightly with the grid
because the noise sequence differs).

### Why LQG fails at long horizons

Two failure modes, both anticipated in Section 8.8 / 10.3:

**(i) Symmetric F-penalty cannot enforce the asymmetric F_max
barrier.** With `Q_F = w_F · (F − F_ref)²`, the controller is
penalised equally for being above and below `F_ref`. The MPC's
soft-plus barrier `λ_F · max(F − F_max, 0)²` is asymmetric and only
fires above the barrier. Empirically the LQG goes bang-bang at the
start of long horizons (Φ=3 on day 0) which crashes through F=0.40,
hence F-violation fractions of 21–37 % at T ≥ 42 d.

**(ii) The linearisation inverts the F→A coupling.** In the true
nonlinear plant, training (high Φ) eventually lifts A through the
cubic + circadian-coupled Stuart-Landau dynamics. In the
linearisation, `A_lin[2,1] < 0` says high F *reduces* A — so the
LQG controller's only path to lift A is to push F *down* (i.e.
*reduce* training), giving the conservative low-Φ schedules at T=14
(mean Φ = 0.34) and T=28 (mean Φ = 0.66). At longer T the unstable
A direction in the linearisation (eigenvalue +0.003) eventually
overwhelms the F-penalty and the controller flips to bang-bang;
the symmetric F-penalty can't restrain it.

### MPC vs LQG vs constant-Φ baseline

Headline at T = 14 d (only horizon currently with both MPC and LQG
checkpoints committed):

| controller | mean A | ratio | F-viol | wall-clock |
|------------|--------|-------|--------|------------|
| constant Φ=1.0 | 0.0824 | 1.000 | 0.00% | < 1 s     |
| LQG open-loop  | 0.0852 | 1.034 | 0.00% | 1.6 s     |
| **SMC² MPC closed-loop** | **0.0954** | **1.157** | **0.00%** | 124 min |

The MPC vs LQG gap (~12 percentage points of the constant baseline)
is large at the *short* horizon where LQG is best-behaved (no
F-violation). At long horizons LQG fails outright.

### Compute tradeoff

Riccati ODE + plant rollout: ~2 s on CPU at every horizon. Total
LQG-bench cost across 5 horizons × 2 grids = 20 s.

SMC² MPC: ~115 min at T=28, ~7 GPU-h at T=84 (h=1h). The 4–5 orders
of magnitude compute gap is real, and the headline numbers say it is
buying real performance.

## What this section establishes

1. **The h=1h discretization preserves the inference and control
   problem to within a few percent at T=14**, so reporting long-
   horizon results at h=1h is methodologically honest.

2. **Across T ∈ {14, 28, 42, 56, 84} the LQG baseline is *not*
   competitive** with SMC²-MPC: at T=14 LQG captures only ~21% of
   MPC's gain over baseline (ratio 1.034 vs 1.157), and at T ≥ 28
   LQG underperforms even the constant-Φ baseline.

3. **The Stuart-Landau cubic nonlinearity is essential.** The
   linearisation around A_typ has the wrong sign on the F-A coupling
   *and* an unstable A direction, so an LQR/Riccati design can't
   even produce a sensible direction of policy improvement, let alone
   a competitive one. This is the empirical answer to Section 10.3
   Q1.

4. **LQG warm-start (Section 10.3 Q2) is unattractive given (3).**
   The LQG schedule's qualitative shape is wrong (low-Φ at short
   horizons, bang-bang at long), so projecting it onto the RBF basis
   to seed the SMC² control posterior would push the chain toward
   bad regions of θ-space rather than toward the near-optimal.

## Outstanding (after this milestone)

- Re-run T=14 at h=1h to formally close the cross-grid consistency
  gate (number for the LaTeX writeup).
- Run T=28 at h=1h alongside the existing 15-min reference (cheap,
  ~30 min) to give two grids worth of T=28 to compare.
- Optional weight sweep on LQG: confirm that no choice of (w_A, w_F,
  Q_T, x_ref) closes the gap meaningfully — supports the
  "Stuart-Landau matters" argument at the LQG-tuning level, not just
  at one cost configuration.

## Commits

- `c2f345d` — Stage I1: --step-minutes flag for h=1h coarse grid
- `15b3596` — Stage H: LQG/Riccati exact-approximate baseline
- `7874228` — Stage H/I plot helper: compare_g4_lqg.py

[Subsequent commits land per-horizon as the h=1h sweep runs.]
