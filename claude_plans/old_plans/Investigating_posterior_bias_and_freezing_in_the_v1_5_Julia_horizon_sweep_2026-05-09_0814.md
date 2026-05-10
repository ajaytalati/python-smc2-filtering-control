# Investigating posterior bias + freezing in the v1.5 Julia horizon sweep

> Archived from plan mode: 2026-05-09 08:17.
> Updated: 2026-05-09 08:21 — Tier 0 executed; H1 confirmed (cloud collapses without Liu–West, recovers with `--liu-west-a 0.97`); secondary finding HMC accept rate = 0% across all baseline runs. Findings report at `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/TIER_0_FINDINGS.md`.

## Context

The 6 h GPU horizon sweep (T = 14, 28, 42, 56, 70, 84 d, seed 42) at
`compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/`
shows a striking pattern in the per-stride parameter trace plots
(`v15_T<N>d_param_traces.png`):

- **All 10 estimated parameters' posterior medians and credible bands
  collapse to a flat line around stride 13–25** (i.e. day 6–12 in real
  time), regardless of horizon length.
- The **frozen value is biased** away from truth, with the bias
  direction varying per parameter (e.g. T = 84 d: `tau_F` ≈ 8 vs truth
  7, `mu_B` ≈ 0.40 vs truth 0.30, `lambda_A` ≈ 0.6 vs truth 1.0).
- Per-stride filter tempering count is **exactly 5** for every stride
  and every horizon (per `manifest.json::smc_cfg` and
  `per_stride.csv::n_temp_filter`); the cap of 30 is never approached.
- The **bias persists and does not shrink** as the horizon (and
  therefore the number of filter strides) grows — adding more
  observations does not move the posterior off its frozen value.

The bias-direction varying per parameter (rather than all in one
direction) is consistent with **wherever the parameter cloud happened
to be when it collapsed**, not a systematic identifiability problem.

## Top three hypotheses (in priority order)

### H1 (PRIMARY, the user's hypothesis): Liu–West shrinkage was off

**The default in the bench driver was `--liu-west-a 0.0`, which
disables the only between-resample θ-cloud rejuvenation step inside
the outer SMC². Without it, the cloud collapses after a few rounds of
systematic resampling and never recovers.**

Code evidence
([bench_smc_full_mpc_fsa_gpu.jl:56,278-290,506](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl)):

```julia
# Line 56 (defaults):
"liu-west-a"      => 0.0,                            # DEFAULT = OFF
...
# Lines 278-290 (inside run_outer_smc, between systematic resample and HMC):
U_resampled = U[indices, :]                          # identical-θ duplicates
if cfg.liu_west_a > 0.0 && cfg.liu_west_a < 1.0
    a = cfg.liu_west_a
    θ_mean = mean(U_resampled; dims = 1)
    θ_std  = std(U_resampled;  dims = 1)
    jitter = sqrt(1 - a^2) .* θ_std .* randn(rng, size(U_resampled))
    U_resampled = a .* U_resampled .+ (1 - a) .* θ_mean .+ jitter
end
# WITH a = 0.0 the if-branch is skipped — U_resampled is unchanged.
```

The HMC moves that follow (`num_mcmc = 3` × `n_temp = 5` = 15 leapfrog
trajectories per stride) use **finite-difference gradients**
(`h_fd = 1e-3`, `parallel_hmc_one_move!` →
[gpu_pf.jl:635-658](version_1_5_Julia/models/fsa_high_res/gpu_pf.jl#L635))
on a stochastic inner-PF log-likelihood. With FD gradients on a noisy
target, 15 HMC moves per stride is not enough to re-inject diversity
into a cloud that systematic resampling just collapsed.

**Why this explains horizon-invariant bias.** Once the cloud collapses
(typically by stride 13–25 ≈ day 6–12), every subsequent stride is
just resampling-from-a-point-cloud. Adding more strides cannot recover
diversity, so the posterior median stays where it froze.

### H2 (SECONDARY, related to H1): No between-window cloud rejuvenation either

**Even with Liu–West on inside a window, the between-window pipeline
is identity-copy: window k's posterior becomes window k+1's starting
cloud at λ=0, with no jitter / shrinkage / kernel-density step in
between.** Liu–West may be necessary but not sufficient.

Code evidence
([bench_smc_full_mpc_fsa_gpu.jl:632-640](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L632)):
the previous posterior `acc.filter_post` is passed in as `U_init`
(read by `run_outer_smc` at lines 224-237 — `U = copy(U_init); λ = 0.0`)
and the tempering loop simply re-tempers the new window's likelihood
on top of it. There is no between-window rejuvenation. This is the
classic Chopin SMC² rolling-window design — it relies entirely on
within-window rejuvenation (Liu–West / OT) to maintain diversity. With
both off (`liu_west_a=0`, `ot_max_weight=0`), nothing keeps the cloud
alive between windows.

If H1 is the cause and Liu–West fixes it, H2 is moot. If turning
Liu–West on doesn't fully fix the bias, H2 says we also need to
think about between-window rejuvenation (e.g. a shrink-then-jitter at
window boundaries).

### H3 (TERTIARY, contextual): The filter only sees 24 obs per stride

**The filter window is fixed at `WINDOW_BINS = 24` (1 day) regardless
of `T_total_days`.** More horizon ≠ more data per filter call — only
more strides.

Code evidence
([bench_smc_full_mpc_fsa_gpu.jl:446-447,634](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L446)):

```julia
bins_per_day = (60 * 24) ÷ args["step-minutes"]
window_bins  = bins_per_day                           # = 24, always
...
t0       = hist_end - window_bins
grid_obs = window_grid_obs(new_obs, new_traj, t0, window_bins)
```

This is the Chopin / Storvik rolling-window convention, not a bug.
But it means the per-stride likelihood is a function of 1 day of
data, so per-stride information content is small — and the filter is
heavily reliant on rejuvenation (H1, H2) to use it. **H3 is not a
fix-target on its own; it is context that amplifies H1/H2.**

## Critical files

- [version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl)
  — bench driver, contains the `run_outer_smc` function and the
  Liu–West shrinkage code path
- [version_1_5_Julia/models/fsa_high_res/gpu_pf.jl](version_1_5_Julia/models/fsa_high_res/gpu_pf.jl)
  — `parallel_hmc_one_move!` (FD-gradient HMC) and the GPU inner PF
- [version_1_5_Julia/models/fsa_high_res/estimation.jl](version_1_5_Julia/models/fsa_high_res/estimation.jl)
  — `PARAM_PRIOR_CONFIG` (LogNormal priors centred at log(truth), σ=0.30)
- [compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/T<N>d_seed42/data.jld2](compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/)
  — saved per stride: `posterior_particles[stride, m, p]` (full 512×10
  cloud per fired-filter stride, per
  [bench_smc_full_mpc_fsa_gpu.jl:735-742](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L735))

## Test plan — gated tiers (user-approved Tier 0 first, then report)

### Tier 0 — combined diagnostic + single short GPU run (~10 min total)

Two parts, run in order; report findings to user before anything else.

**Part A — cloud-collapse diagnostic on existing data (CPU, minutes).**
The existing `data.jld2` files already store the **full 512×10
parameter cloud per stride**, so the cloud-collapse question (the
core claim of H1) can be answered from data already on disk.

- Load `T28d_seed42/data.jld2` → `posterior_particles` (shape
  `(n_strides, 512, 10)`); the array is `exp(U)` per
  [bench_smc_full_mpc_fsa_gpu.jl:740](version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L740),
  so take `log` first to get unconstrained-space std
- Per stride, compute per-parameter cloud std
- Plot `cloud_std(stride)` as 10 panels, one per parameter; repeat
  for all 6 horizons and overlay
- Save the small Julia script as
  `compare_v15_julia_vs_python/src/diag_cloud_collapse.jl` and the
  PNG into the existing horizon-sweep folder

**Part B — single T = 14 d bench with Liu–West on, fast controller config
(~2 min GPU).**

Per the technical guide §"When to deviate from these defaults", the
~4× speed-up recipe is to **leave the filter at saturated config
(`--N-smc 512 --K-per-chain 1000`) but drop the controller side**
(`--ctrl-n-smc` 2048 → 1024, `--ctrl-num-mcmc` 16 → 8). Since the
bias under investigation is on the **filter** side (parameter
posterior collapse / cloud std), the weaker controller schedule
shouldn't mask the Liu–West signal.

```bash
cd version_1_5_Julia
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days 14 --seed 42 \
    --liu-west-a 0.97 \
    --N-smc 512 --K-per-chain 1000 \
    --ctrl-n-smc 1024 --ctrl-num-mcmc 8 \
    --output-dir <SWEEP_ROOT>/T14d_seed42_lw097_fastctrl
```

- Wall: ~2 min on RTX 5090 (T = 14 d baseline was ~7 min at the
  saturated controller; ~4× drop with the fast controller per the
  tech guide). Liu–West itself adds one `randn` per tempering level,
  negligible cost.
- Caveat: the comparison is *not* perfectly apples-to-apples vs the
  existing `T14d_seed42/` baseline (which used the saturated
  controller). What it **is** apples-to-apples for is the question
  of interest: does the parameter cloud collapse with Liu–West off
  vs stay alive with it on? Cloud dynamics depend on the filter side,
  not the controller schedule.
- Use the existing canonical plotters
  `compare_v15_julia_vs_python/src/plot_state_traces.jl` and the
  parameter-traces plotter
- Also run `diag_cloud_collapse.jl` on this new `data.jld2` so we
  can compare cloud-std curves baseline vs Liu–West directly

**Report content (back to user, before any further GPU spend).**

A short markdown/PDF writeup with:

- The cloud-std-vs-stride curves from Part A (does the original run
  show the predicted collapse?)
- The cloud-std-vs-stride curves from Part B (does Liu–West keep
  the cloud alive?)
- Side-by-side parameter-trace plots: existing `T14d_seed42` vs new
  `T14d_seed42_lw097`
- Headline verdict: H1 confirmed / H1 not sufficient / H1 wrong
  — and a one-paragraph recommendation on whether to commit further
  GPU time

### Tier 1 (parked) — full 6-horizon sweep with `--liu-west-a 0.97`, fast controller

Only on user approval after the Tier 0 report. Same launcher
`version_1_5_Julia/tools/launchers/run_julia_horizon_sweep.sh`,
patched to add the **same flag set as Tier 0 Part B**:

```
--liu-west-a 0.97 \
--N-smc 512 --K-per-chain 1000 \
--ctrl-n-smc 1024 --ctrl-num-mcmc 8
```

Total wall ≈ 6 h ÷ 4 ≈ **1.5 h** (vs the original 6 h). Per the
technical guide this trades a "weaker schedule" — i.e. the controller
finds a less aggressive ramp-up — for ~4× wall-time. That trade is
acceptable here because the question this study answers is "does
Liu–West fix the *filter* posterior bias", not "what is the strongest
controller schedule". If H1 + the fast controller still produces
unbiased filter posteriors, we can do a separate full-saturated-controller
re-run later for the final scientific writeup.

## Verification

End-to-end success criterion for the rerun:

- Per-parameter cloud std plotted vs stride does **not** collapse to
  zero — stays of order 0.05–0.30 in log-space (i.e. comparable to
  the prior σ).
- Posterior medians for all 10 parameters lie within ±2 prior-σ of
  truth at the end of every horizon.
- Bias **shrinks** monotonically with horizon (as
  T = 14 → 84 d), as a properly-functioning rolling-window SMC²
  should.

## User decisions (2026-05-09)

1. **TIER 0 ONLY is approved.** User said: "I approve ONLY TIER 0
   do both parts then report back". Both Part A (diagnostic on
   existing `data.jld2`) and Part B (single T = 14 d GPU bench at
   `--liu-west-a 0.97` + fast controller config) are in scope. Then
   stop and report findings. **TIER 1 (full 6-horizon sweep) is
   NOT approved and must NOT be started without a separate
   approval.**
2. **Use the technical guide's "fast" controller config for Part B.**
   Per
   [compare_v15_julia_vs_python/docs/technical_guide_to_current_best_Julia_SMC2FC_config.tex:152-158](compare_v15_julia_vs_python/docs/technical_guide_to_current_best_Julia_SMC2FC_config.tex#L152)
   §"When to deviate from these defaults": keep `--N-smc 512
   --K-per-chain 1000` (filter at saturated config) but drop
   `--ctrl-n-smc` 2048 → 1024 and `--ctrl-num-mcmc` 16 → 8. ~4×
   wall-time reduction at the cost of a weaker controller schedule.
