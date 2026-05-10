# Controller-exploration test at T = 14 d — findings

**Run date:** 2026-05-09 17:11 → 18:05 (54 min wall).
**Plan:** [`claude_plans/Controller_exploration_test_T14_stack_all_aids_2026-05-09_1659.md`](../../../claude_plans/Controller_exploration_test_T14_stack_all_aids_2026-05-09_1659.md)
**Sweep root:** this directory.

## TL;DR — the compute-imbalance hypothesis is NOT supported at this scale

Stacking all four controller exploration aids (`--ctrl-num-mcmc 24 --ctrl-sigma-prior 2.5 --ctrl-max-lambda-inc 0.05 --ctrl-target-ess-frac 0.3 --ctrl-n-inner 128`) produced **dramatic algorithmic changes** on the controller side but **left the closed-loop trajectory essentially unchanged**:

| | unified-HMC T=14 | this run (ctrl-explore stack) | change |
|---|---:|---:|---|
| `mean_A_mpc` | 0.0762 | **0.0766** | +0.5 % |
| `final_A_mpc` | 0.0837 | **0.0842** | +0.6 % |
| `mean_A_baseline` (constant Φ = 1) | 0.1138 | 0.1138 | — |
| MPC vs baseline | −33 % | **−33 %** | identical gap |
| `mean_phi_per_stride` | 0.614 | **0.516** | went *lower* |
| `final_phi_per_stride` | 0.638 | 1.303 | went higher at end |
| `mean_phi_at_stride21` | 1.128 | 0.701 | substantially *lower* |
| total wall (s) | 520 | **3263** | 6.3× |

**The ~24× extra controller compute bought ~0 % improvement in closed-loop A.** Source: [`horizons_results.csv`](horizons_results.csv).

## What the controller-side algorithm DID change (it works — just doesn't help)

From [`docs/controller_hmc_summary.csv`](docs/controller_hmc_summary.csv):

| | unified-HMC T=14 | this run | change |
|---|---:|---:|---|
| total tempering levels (across 13 replans) | 69 | **260** | 3.8× |
| mean accept (%) | 96.87 | 98.38 | +1.5 pp |
| **mean ChEES L** | **16.0** | **28.4** | **picker now selects L > 16** |
| mean ESJD/(ε·L) | 14.12 | 24.85 | 1.76× |
| mean β_max | 226 | 169 | — (auto-calibrated; not directly comparable) |
| mean wall per level (s) | 1.62 | 11.03 | 6.8× |

**The picker is no longer locked at L = 16** — mean L = 28.4, with 20 tempering levels per replan instead of 5. ESJD per unit Hamiltonian time has nearly doubled. Algorithmically, every signal that the exploration aids should improve the controller's mixing did improve. **The controller is mixing better; it's just mixing better around the same posterior, which produces the same Φ schedule.**

## Filter-side sanity (unchanged, as expected)

[`docs/filter_hmc_accept_summary.csv`](docs/filter_hmc_accept_summary.csv): mean accept 73.1 %, mean ChEES L 55.9, 130 levels — within noise of the unified-HMC T=14 baseline (71.4 % / 58.2 / 130). The exploration aids touch only controller-side flags; this confirms no accidental filter-side change.

## Wall split (measured from per_stride.csv)

| | strides | mean wall (s) | total (s) | share |
|---|---:|---:|---:|---:|
| filter-only | 14 | 14.13 | 198 | **6 %** |
| replan (filter + controller) | 13 | 235.50 | — | — |
| controller contribution per replan | 13 | 221.37 | **2878** | **88 %** |
| total | 27 | — | 3263 | 100 % |

The compute imbalance is *now* the opposite of the unified-HMC default (88 / 6 % for controller / filter, vs 16 / 84 % at unified-HMC T=14). **We have decisively given the controller more compute — and it didn't change the closed-loop result.**

## What this rules out and what it leaves open

**Rules out:** "the controller fails at T=14 because compute is too low relative to the filter." Throwing 24× more controller compute at it produced no closed-loop improvement. The compute-imbalance reading was wrong as a *causal* explanation, even though the imbalance is real.

**Still open** (not measured here):

1. **Filter parameter bias affects controller cost surface.** The earlier posterior-trace inspection showed the filter has a consistent ~10 % bias on `tau_F`, `B_inf`, `F_inf` across all sweeps. The controller uses the filter's posterior-mean parameters for its forward-sim cost evaluation. If those are biased, the controller is optimising the *wrong* cost — and no amount of MCMC mixing on a wrong cost recovers the true-cost optimum. **This is the next thing to test.**

2. **CRN noise structure.** The controller's cost is `mean(over n_inner=128 CRN noise trials)` where the noise grid is fixed per cost call (deterministic given seed). Bumping `n_inner` 64 → 128 cut MC variance by √2 but didn't change the *bias* induced by a particular fixed noise realisation. A different fix would be to randomise the CRN seed per replan rather than per controller call.

3. **The cost J really has its minimum at low Φ at T=14d.** μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F² with truth params gives a small positive drift on A only when B is large and F is small. At T=14 d B is still small (start 0.05, grows slowly with τ_B = 42 d), so high-Φ training mainly grows F → drives μ negative → A *shrinks*. So the cost-minimising schedule is genuinely low-Φ at T=14d; the constant-Φ=1 baseline beats MPC by *coincidence* because Φ=1 happens to keep F bounded at ~0.18 (well below F_max = 0.40) while letting B drift up enough to get a small positive μ. **The MPC may be finding the cost-minimising schedule correctly, but the cost J doesn't align with "maximise mean A" — that's a cost-design issue, not a controller failure.** This is the most plausible explanation for what we measured.

## Verdict

**At T = 14 d the controller's closed-loop A-trajectory is not compute-bound.** Stacking exploration aids did improve the MCMC algorithmically (ChEES L ↑, ESJD ↑, more levels) without improving the closed-loop A. Three plausible-but-untested explanations remain (filter parameter bias, CRN noise structure, or cost-functional mis-alignment with mean-A); each is a separate experiment. The compute-imbalance hypothesis is closed.

If the goal is "more mean A at T = 14 d", the next moves are:
1. **Fix the filter parameter bias** (the τ_F / B_∞ / F_∞ bias seen across all sweeps) and re-run with the existing controller defaults.
2. **Reformulate the cost** if "mean A" is what we actually want — e.g. a positive quadratic on A or a penalty on negative-μ regions.

Throwing more compute at the controller at T = 14 d under the current cost+filter is empirically not going to help.

## Plots on disk

- [`v15_T14d_param_traces.png`](T14d_seed42/v15_T14d_param_traces.png), [`v15_T14d_traces.png`](T14d_seed42/v15_T14d_traces.png) — per-stride parameter and state traces.
- [`docs/controller_hmc_diagnostics_T14d.png`](docs/controller_hmc_diagnostics_T14d.png) — six-panel ChEES / accept / ESJD / Δlog-density / per-replan summary plot.
- [`cloud_std_per_stride_all_horizons.png`](cloud_std_per_stride_all_horizons.png) — cloud std vs day (single-horizon overlay; included for parity with prior sweeps).

## Source edits made for this run

Two new CLI flags exposed in [`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`](../../../version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl):
- `--ctrl-target-ess-frac` (default 0.5; was hardcoded inside `ctrl_cfg`).
- `--ctrl-max-lambda-inc` (default 0.20; was hardcoded inside `ctrl_cfg`).

Both default values reproduce pre-flag behaviour exactly. Manifest's `smc_cfg` block now records `ctrl_target_ess_frac` and `ctrl_max_lambda_inc`. No framework edits.

Sibling launcher: [`run_julia_horizon_sweep_ctrl_explore.sh`](../../../version_1_5_Julia/tools/launchers/run_julia_horizon_sweep_ctrl_explore.sh) (T_LIST = 14 by default; the four exploration flags inlined).

Also today, separately from this study: filter-side flag aliases `--filter-<name>` were added for every previously-unprefixed filter flag (back-compat preserved). See the bench source around line 219.
