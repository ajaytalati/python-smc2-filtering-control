# FSA-v1 Julia open-loop GPU controller — port from v2 GPU kernel

> Archived from plan mode: 2026-05-07 18:21.


## Context

The v2 closed-loop SMC²-MPC bench takes ~22 min wall-clock on Julia and is dominated by the filter (~125s × 27 strides). The standing controller-bug investigation (`docs/julia_fsa_writeup.pdf` §2.7) needs a much faster turn-around than that. Open-loop control on the **v1 FSA model** (the predecessor to v2 — same 3-state [B, F, A] Banister-coupled SDE, but simpler drift formulas without the G1 reparametrisation around (A_TYP, F_TYP)) is the right isolation: no filter, single SMC² call at the full horizon, ~minutes per run instead of ~half-hour.

The v1 Python bench (`version_1/tools/bench_smc_control_fsa.py`) already runs this exact open-loop experiment at four horizons (T=28, 42, 56, 84 d) and produced the four `D_v2_T{N}_diagnostic.png` plots in `version_1/outputs/fsa_high_res/`. The v1 Julia model directory has CPU-only dynamics + control plumbing but **no GPU kernel and no bench driver**. The job is to add both, run them, and reproduce the four diagnostic plots from the Julia side.

"EXACTLY the same plots" cannot be pixel-identical because Julia and Python use different RNGs; the practical target is **same 6-panel layout, same panel content, same diagnostic numbers within Monte-Carlo noise** (mean ∫A/T per stride, F-violation fraction, gate pass/fail outcomes).

## What needs to be true at the end

1. `version_1_Julia/models/fsa_high_res/gpu_control.jl` exists and exports `FSAv1ControlGPUTarget` + `make_log_density_fn`, mirroring v2's pattern but with v1 drift formulas.
2. `version_1_Julia/tools/bench_smc_control_fsa_gpu.jl` runs the open-loop SMC² controller at a CLI-supplied T_total_days, dumps a 6-panel diagnostic PNG matching the Python filename pattern.
3. Running the bench at T = 28, 42, 56, 84 produces four PNGs under `version_1_Julia/outputs/fsa_high_res/` whose qualitative numbers (mean ∫A/T, gate pass/fail) line up with `version_1/outputs/fsa_high_res/RESULT.md`.

## Critical drift differences (v1 vs v2)

The v2 GPU kernel formulas must NOT be copied verbatim. v1 specifically:

| Term | v1 (target) | v2 (do not use) |
|---|---|---|
| B-coupling | `aB(A) = 1 + ε_A·A` | `(1 + ε_A·A) / (1 + ε_A·A_TYP)` |
| F-coupling | `aF(A) = 1 + λ_A·A` | `(1 + λ_A·A) / (1 + λ_A·A_TYP)` |
| Stuart-Landau μ | `μ_0 + μ_B·B − μ_F·F − μ_FF·F²` | `μ_0 + μ_B·B − μ_F·F − μ_FF·(F−F_TYP)²` |
| TRUTH μ_0 | 0.02 | 0.036 (= 0.02 + μ_FF·F_TYP²) |
| TRUTH μ_F | 0.10 | 0.26 (= μ_F + 2·μ_FF·F_TYP) |
| TRUTH κ_B | 0.012 | 0.01248 (= κ_B·(1+ε_A·A_TYP)) |

Authoritative v1 source for these formulas: `version_1_Julia/models/fsa_high_res/_dynamics.jl:58-67` and `version_1/models/fsa_high_res/_dynamics.py:70-97`.

There is NO `_phi_burst.jl` and NO `_plant.jl` in v1 — the controller integrates the SDE directly inside the cost kernel, just like v2's `gpu_control.jl` does. No closed-loop machinery needed.

## Files to create

### 1. `version_1_Julia/models/fsa_high_res/gpu_control.jl` (~280 lines)

Direct port of `version_2_Julia/models/fsa_high_res/gpu_control.jl` with these targeted changes inside the `@kernel` body:

- Replace v2's `a_factor_B = (1f0 + p_epsilon_A * A) / (1f0 + p_epsilon_A * A_typ32)` with `a_factor_B = 1f0 + p_epsilon_A * A`.
- Replace v2's `a_factor_F = (1f0 + p_lambda_A * A) / (1f0 + p_lambda_A * A_typ32)` with `a_factor_F = 1f0 + p_lambda_A * A`.
- Replace v2's `F_dev = F - F_typ32; mu_bif = p_mu_0 + p_mu_B*B - p_mu_F*F - p_mu_FF*F_dev*F_dev` with `mu_bif = p_mu_0 + p_mu_B*B - p_mu_F*F - p_mu_FF*F*F`.
- Drop the `A_TYP, F_TYP` imports — v1 has no operating-point constants in the drift.
- Cost is unchanged: `J = -∫A·dt + λ_F·∫max(F-F_max,0)²·dt`, defaults λ_F=1, F_max=0.40, Phi_max=3.0, Phi_default=1.0.
- Module name: `module GPUControl` (same as v2 — kept module-local to `FSAHighRes`).
- Struct rename: `FSAv1ControlGPUTarget` (so it cannot be accidentally confused with v2's `FSAControlGPUTarget`).
- RBF design matrix construction: identical to v2's lines 184-189 (raw Gaussian, NOT row-normalised), σ = T_total/n_anchors.
- CRN noise, `gpu_cost_log_density_batched`, `make_log_density_fn`: ported verbatim.
- Pure-Julia CPU mirror function (`cpu_fp64_per_trial_cost`) included for the same fp32-cancellation cross-check we ran in the v2 test driver.

### 2. `version_1_Julia/tools/bench_smc_control_fsa_gpu.jl` (~250 lines)

Open-loop bench, mirrors `version_1/tools/bench_smc_control_fsa.py`. Pattern from `version_2_Julia/tools/test_max_B_only.jl` (already known to drive `run_tempered_smc_gpu` correctly).

Steps:
1. Parse CLI: `T_total_days` (default 42), seed (default 0).
2. Build an `FSAv1ControlGPUTarget` at T_total_days, h=15min, n_substeps=4, n_anchors=8, n_inner=32, sigma_prior=1.5.
3. Call `SMC2FC.run_tempered_smc_gpu(log_density_fn, M_max, n_smc=256, d=8, prior_mean=0, sigma_prior=1.5, …)` with the horizon-adaptive HMC settings the Python bench uses:
   - T < 50 d: `hmc_step_size=0.30`, leapfrog=16, num_mcmc=15
   - 50 ≤ T < 70 d: `hmc_step_size=0.12`, same leapfrog/num_mcmc
   - T ≥ 70 d: `hmc_step_size=0.05`, same leapfrog/num_mcmc
   - `target_ess_frac=0.5`, `max_lambda_inc=0.10`, `target_nats=8.0`, `max_temp_levels=100`.
4. From the posterior cloud, extract posterior-mean θ, decode it through the RBF basis to a per-bin Φ schedule.
5. For diagnostic plotting, run the cost kernel one more time at posterior-mean θ with a separate RNG to draw 5 sample (B,F,A) trajectories. Compute mean ∫A/T, F-violation fraction, gate pass/fail per the Python bench's gate definitions.
6. Build the 6-panel diagnostic with Plots.jl (already in `version_1_Julia/Project.toml`). Layout matches `version_1/tools/bench_smc_control_fsa.py:91-202`:
   1. Top-left: Φ(t) — SMC² posterior mean vs constant Φ=1 baseline vs sedentary Φ=0.
   2. Top-right: per-particle cost histogram from the final tempering level (returned by `run_tempered_smc_gpu` as the posterior cloud's per-particle log-densities).
   3. Middle-left: 5 B(t) sample paths + mean.
   4. Middle-right: 5 F(t) sample paths + mean, with F_max=0.40 horizontal line.
   5. Bottom-left: 5 A(t) sample paths + mean, with baseline and sedentary mean references.
   6. Bottom-right: bar chart — sedentary, baseline, SMC² mean ∫A/T.
7. Save to `version_1_Julia/outputs/fsa_high_res/D_v2_T{int(T_total_days)}_diagnostic.png` (same filename convention as Python).
8. Print a summary block to stdout: gate pass/fail, mean ∫A/T values, n_temp_levels, β_max, wall time. Optionally append a row to `version_1_Julia/outputs/fsa_high_res/RESULT.md`.

### 3. `version_1_Julia/tools/launchers/run_horizon_sweep.sh` (~10 lines)

Tiny shell driver that loops T ∈ {28, 42, 56, 84} and calls the bench. Logs each run to `version_1_Julia/outputs/fsa_high_res/run_T{N}.log`.

## Files to modify

### `version_1_Julia/models/fsa_high_res/FSAHighRes.jl`

Add `include("gpu_control.jl")` after the existing `include("control.jl")`, then `using .GPUControl` and re-export `FSAv1ControlGPUTarget`, `gpu_cost_log_density_batched`, `make_log_density_fn`. Existing CPU `control.jl` stays untouched.

## Existing utilities to reuse (do not re-implement)

- `SMC2FC.run_tempered_smc_gpu` from `julia/SMC2FC/src/Control/GPUControlSMC.jl` — the parallel-chains tempered SMC² + ChEES-HMC engine. Same call signature as `test_max_B_only.jl:92-100`.
- **`build_rbf`** from `version_2_Julia/models/fsa_high_res/cpu_control.jl:32-47` — already a clean Gaussian-RBF Matrix builder, raw (NOT row-normalised), σ = T_total/n_anchors. Copy this function into v1's new `gpu_control.jl` (or call it from v1's existing `control.jl` if that file already exposes the same helper) instead of inlining the RBF construction. Same matrix is then `Float32.(...)` cast and uploaded to the GPU. Removes a 6-line near-duplicate that already differs subtly between v2's GPU and CPU files.
- `Plots.jl` — already in `version_1_Julia/Project.toml`. Use it for all 6 panels.
- `NPZ.jl` — already a dep; use for any noise-grid IO needed for cross-checks.

## Plan archive (per repo CLAUDE.md)

Copy this plan file out of `~/.claude/plans/` into `/home/ajay/Repos/python-smc2-filtering-control/claude_plans/` as the very first implementation step, with a date+time-stamped filename derived from this plan's title (NOT the auto-generated codename). The archived copy gets a `> Archived from plan mode: <YYYY-MM-DD HH:MM>.` line prepended right under the title. Subsequent meaningful updates push to the archive too with `> Updated: …` audit lines.

## Verification

End-to-end:
1. From `version_1_Julia/`: `julia --project=. tools/bench_smc_control_fsa_gpu.jl 42`. Should complete in under a few minutes on the RTX 5090, save `outputs/fsa_high_res/D_v2_T42_diagnostic.png`, and print the four gate outcomes.
2. Open the produced PNG side-by-side with `version_1/outputs/fsa_high_res/D_v2_T42_diagnostic.png`. Visual check: same 6-panel layout, Φ schedule near constant ≈ 1.0 (per Python's RESULT.md, T=42 SMC² matches baseline within 0.5%), F well under F_max, all 4 gates passing.
3. Run the full sweep via `tools/launchers/run_horizon_sweep.sh`. Compare each PNG to its Python sibling and cross-check the headline numbers in the printed gate summary against `version_1/outputs/fsa_high_res/RESULT.md`:
   - T=28: 2 of 4 gates fail (insufficient horizon) — same outcome on Julia.
   - T=42: all gates pass, SMC² ≈ constant baseline.
   - T=56: all gates pass, ~+20% over baseline.
   - T=84: all gates pass, ~+28% over baseline, +208% over sedentary.
4. Cross-check the cost kernel itself with the same fp32-vs-fp64 mirror test we already ran for v2 (`version_2_Julia/tools/test_cost_at_theta0.jl` template) — at θ=0 the GPU-fp32 result should be within 1e-5 of the CPU-fp64 mirror. Adapt that test driver for v1 by pointing it at the new `FSAv1ControlGPUTarget`. This is a quick sanity check before running the full bench.
