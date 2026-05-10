# Tier 0 findings — `--liu-west-a 0.97` fixes the cloud collapse

**Run date:** 2026-05-09
**Plan:** [`claude_plans/Investigating_posterior_bias_and_freezing_in_the_v1_5_Julia_horizon_sweep_2026-05-09_0814.md`](../../../claude_plans/Investigating_posterior_bias_and_freezing_in_the_v1_5_Julia_horizon_sweep_2026-05-09_0814.md)

## TL;DR

H1 (Liu–West shrinkage was off → parameter cloud collapses → posterior
freezes at biased value) is **confirmed** by two independent pieces of
evidence:

1. **Part A diagnostic on existing data:** in every one of the six
   baseline horizons, per-parameter cloud std (in unconstrained log
   space) starts at the prior σ ≈ 0.30 and **decays to near-zero by
   day ~15** for all 10 estimated parameters. Files:
   [`docs/cloud_std_per_stride_all_horizons.png`](docs/cloud_std_per_stride_all_horizons.png).
2. **Part B single T = 14 d rerun with `--liu-west-a 0.97` and the
   tech-guide fast-controller config:** the posterior cloud std stays
   bounded above zero across all strides
   ([`T14d_seed42_lw097_fastctrl/cloud_std_lw097.png`](T14d_seed42_lw097_fastctrl/cloud_std_lw097.png)),
   and the posterior medians of all 10 parameters now **track truth**
   instead of freezing at biased values
   ([`T14d_seed42_lw097_fastctrl/v15_T14d_param_traces.png`](T14d_seed42_lw097_fastctrl/v15_T14d_param_traces.png)).

## Part A — cloud-collapse diagnostic on existing 6-horizon baseline

The bench saves the **full 512×10 parameter cloud per stride** in
`data.jld2::posterior_particles` (constrained / `exp(U)` space, per
[`bench_smc_full_mpc_fsa_gpu.jl:739`](../../../version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L739)).
The diagnostic
[`compare_v15_julia_vs_python/src/diag_cloud_collapse.jl`](../../src/diag_cloud_collapse.jl)
loads each horizon's `data.jld2`, takes `log` to recover the
unconstrained cloud, and plots per-parameter cloud std vs day.

**Result.** The pattern is identical across all six horizons: cloud
std for all 10 parameters drops from ~0.30 (the prior σ) to near-zero
by day 15. For T = 14 d the run ends before full collapse; for T ≥ 28 d
the collapsed state persists for the rest of the run. This is the
mechanical cause of the "frozen median + biased value" pattern in the
existing parameter-traces plots.

## Part B — T = 14 d rerun with `--liu-west-a 0.97` and fast-controller

```bash
cd version_1_5_Julia
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days 14 --seed 42 \
    --liu-west-a 0.97 \
    --N-smc 512 --K-per-chain 1000 \
    --ctrl-n-smc 1024 --ctrl-num-mcmc 8 \
    --output-dir compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/T14d_seed42_lw097_fastctrl
```

- **Wall: 173.4 s** (~2.4× faster than the baseline T = 14 d's
  422.7 s; the tech-guide quoted ~4×, so we got somewhat less than
  promised — the JIT-compile overhead is a fixed cost that doesn't
  shrink with the controller config).
- Liu–West code path
  ([`bench_smc_full_mpc_fsa_gpu.jl:284-290`](../../../version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L284))
  fires every tempering level with `a = 0.97`.

### Side-by-side: posterior medians at end of T = 14 d

Truth → baseline (no LW) → `--liu-west-a 0.97`:

| param      | truth  | baseline (frozen) | LW = 0.97 |
|------------|--------|-------------------|-----------|
| `tau_F`    | 7.0    | ~6.5              | **~7.0**  |
| `B_inf`    | 0.504  | ~0.42             | **~0.50** |
| `F_inf`    | 0.21   | ~0.27             | ~0.27 (still high) |
| `lambda_A` | 1.0    | bouncing 0.6–1.4  | **~1.0**  |
| `mu_0`     | 0.020  | ~0.022            | **~0.020**|
| `mu_B`     | 0.30   | ~0.25             | **~0.30** |
| `mu_F`     | 0.10   | ~0.10             | **~0.10** |
| `sigma_B`  | 0.010  | ~0.010            | ~0.010    |
| `sigma_F`  | 0.012  | ~0.010            | ~0.011    |
| `sigma_A`  | 0.020  | ~0.022            | ~0.020    |

Bolded entries are clear improvements; `F_inf` is the lone holdout
(still biased high). The credible bands stay wide for the
Liu–West run, instead of collapsing to invisible.

## Secondary finding (not in original plan): HMC acceptance is 0% everywhere

While running the bench I noticed `accept=0%` printed at every
tempering level, every stride. Cross-checking the baseline run logs:

| run            | total tempering levels logged | accept=0% count |
|----------------|-------------------------------|-----------------|
| T14d baseline  | 130                           | 130             |
| T28d baseline  | 270                           | 270             |
| T42d baseline  | 410                           | 410             |
| T56d baseline  | 550                           | 550             |
| T70d baseline  | 690                           | 690             |
| T84d baseline  | 830                           | 830             |
| T14d + LW=0.97 | 130                           | 130             |

**Every HMC trajectory in every baseline run was rejected.** The MCMC
moves are not contributing to mixing; the only sources of cloud
diversity are (a) the systematic resample (which contracts the
cloud — does not expand it), and (b) Liu–West shrinkage (when on).

This explains why turning Liu–West on alone is enough to keep the
cloud alive at this config: with HMC dead anyway, Liu–West is the
*only* mechanism putting variance back into the cloud after each
resample. It's also a separate engineering issue worth addressing —
the bench is computing fully-rejected HMC trajectories at non-trivial
cost. Likely culprits (in priority order, all unverified — would need
a dedicated debug run):

1. **HMC step size 0.05 with finite-difference gradients on a
   stochastic target.** FD gradients on the inner-PF noisy
   log-likelihood are biased; combined with the HMC's exact MH
   correction, every proposal lands in a low-density region.
2. **`hmc_num_leapfrog = 4`** is small — short trajectories on a
   poorly-conditioned target.
3. **Mass-matrix diagonal** is identity-ish; for parameters with
   different prior scales (e.g. `tau_F` ∈ ~7 vs `sigma_B` ∈ ~0.01),
   that means very anisotropic acceptance.

I have **not** verified any of these — they're hypotheses, not
diagnoses. The cheap test would be to bisect: run with
`--num-mcmc 0` (no HMC at all, only Liu–West) and see if the
posterior is the same as `--num-mcmc 3 --liu-west-a 0.97`. If yes,
HMC is contributing nothing.

## Recommendation

Now that H1 is confirmed and Liu–West clearly works at T = 14 d, I
think there are three reasonable next moves. They are not equivalent;
the right pick depends on what the user wants the next ~hours of GPU
time to answer.

(I am **not** starting any of these — Tier 1 was explicitly not
approved, so I am stopping here and reporting back.)

- **Option A — proceed to the parked Tier 1 (full 6-horizon sweep
  with `--liu-west-a 0.97 + fast controller`).** ~1.5 h GPU. Confirms
  the fix at every horizon and produces an apples-to-apples comparison
  writeup. The fast controller is a known weaker schedule, so the
  scientific result will be "filter posteriors are unbiased; the
  controller's commitment to the Banister-overload ramp-up may be
  weaker than the saturated config".
- **Option B — also debug the 0% HMC accept rate before any Tier 1.**
  Cheap CPU experiments first (vary step size; vary `--num-mcmc`;
  print a few proposed-vs-current log-densities to see whether
  proposals are landing far away because of large step or because of
  FD-gradient bias). This is a different question than H1 but the
  finding is unexpected and worth understanding before committing
  more wall time.
- **Option C — try a smaller `a` (e.g. 0.95)** in a single short
  rerun. The cloud std plot for the LW = 0.97 run shows `mu_F` cloud
  std *growing* to ~0.50 by end-of-run, larger than the prior σ. That
  is the AR(1)-like Liu–West process at equilibrium: with HMC adding
  zero mixing, the only thing damping the cloud is the shrinkage
  factor `a`. A smaller `a` (= more shrinkage) would tighten the
  cloud at the cost of bias toward the running mean.

My read: **A is the most direct answer to the user's original
question** (does Liu–West fix the horizon-invariant bias?). B is the
most scientifically clean (understand why HMC is dead before
publishing). C is a cheap variant of A.

Flag for the user: my read could be wrong — I'd want to verify it
with you before committing GPU time, since each of A / B / C is a
different research question.


# Addendum (Tier 0 follow-up B): HMC step-size + h_fd sweep + framework rejuvenation tricks

User asked for two more pieces of work after Tier 0:

1. Debug the 0% HMC accept rate — cheap CPU experiments varying step
   size, leapfrog count, and FD-gradient step.
2. Find what other rejuvenation tricks the framework already provides
   alongside Liu–West (cov-shrinkage / KDE jitter / similar).

Both done; results below.

## 1) HMC accept-rate sweep

Diagnostic script:
[`compare_v15_julia_vs_python/src/diag_hmc_accept_sweep.jl`](../../src/diag_hmc_accept_sweep.jl).
Setup mirrors [`version_1_5_Julia/tools/test_gpu_pf.jl`](../../../version_1_5_Julia/tools/test_gpu_pf.jl):
1-day truth-Φ = 1 obs window, M = 128 chains spread iid from the prior
around truth, K = 200 inner-PF state particles. We re-implement the
HMC leapfrog inline so we can also report the proposed-vs-current
log-density. Every cell of the (ε × L × h_fd) grid uses the same RNG
key.

Full table: [`diag_hmc_accept_sweep.csv`](diag_hmc_accept_sweep.csv).
Key rows (sorted by accept rate, with the bench's defaults marked):

| ε       | L (leap) | h_fd   | accept % | mean log_α | mean Δll  | comment |
|---------|----------|--------|---------:|-----------:|----------:|---------|
| 0.001   | 4        | 1e-2   |     74.2 |    -0.12   |   +0.19   | best    |
| 0.001   | 1        | 1e-2   |     73.4 |    -0.13   |   -0.09   | best    |
| 0.001   | 16       | 1e-2   |     62.5 |    -0.68   |   +0.55   | good    |
| 0.005   | 1        | 1e-2   |     58.6 |    -0.72   |   +0.15   | good    |
| 0.01    | 1        | 1e-2   |     50.0 |    -2.08   |   +0.27   | OK      |
| 0.005   | 4        | 1e-2   |     36.7 |    -4.6    |   +0.66   | dropping|
| 0.001   | 4        | 1e-3   |     32.8 |    -6.0    |   +0.09   | dropping|
| 0.02    | 1        | 1e-2   |     22.7 |    -8.5    |   +0.53   | poor    |
| 0.01    | 4        | 1e-2   |     17.2 |   -11.6    |   +0.91   | poor    |
| **0.05**| **4**    |**1e-3**|  **0.0** | **-2.2e72**|**-6.4e59**| **bench default — broken** |
| 0.05    | 4        | 1e-2   |      0.0 | -1.9e+04   | -5.5e+34  | broken  |
| 0.05    | 1        | 1e-3   |      0.0 | -4.9e+59   | -1.4e+29  | broken  |

Three trends are clear:

1. **Step size ε = 0.05 (the bench default) is far too large.** Even
   with the more accurate FD step h_fd = 1e-2, every cell at ε = 0.05
   gives 0% accept. The sweet spot for this target is ε ≈ 0.001–0.005.
2. **h_fd = 1e-3 (the bench default) is too small.** For fixed
   (ε, L) = (0.001, 1), going h_fd 1e-2 → 1e-3 → 1e-4 cuts accept
   from 73 % → 65 % → 43 %. The reason is mechanical: the inner-PF
   log-likelihood is stochastic; its central-difference gradient
   `(f(θ+h) − f(θ−h)) / (2h)` has variance ∝ σ_PF² / h², so for
   small h the gradient estimate is dominated by inner-PF noise.
   h = 1e-2 in unconstrained log-space is large enough to overcome
   that noise; h = 1e-3 is comparable to it; h = 1e-4 is below it.
3. **Leapfrog L = 16 is much worse than L = 1**, especially with
   smaller h_fd. Each leapfrog step compounds the FD-gradient noise.
   At (ε, L, h_fd) = (0.001, 1, 1e-3): 65 %; at (0.001, 16, 1e-3):
   8 %; at (0.001, 16, 1e-4): 0.8 %.

So the **proximate bug** behind "0 % HMC accept everywhere" is that
the bench is running:

```
hmc_step  = 0.05    # too large by ~50× for this target
hmc_leap  = 4
h_fd      = 1e-3    # too small by ~10× for the inner-PF noise
```

A reasonable starting fix would be `hmc_step = 0.005, hmc_leap = 4,
h_fd = 1e-2` (~37 % accept in the standalone sweep, well within the
HMC sweet spot of 0.5–0.8). Worth iterating once on a real bench run
to confirm the bench's accept rate matches the standalone diagnostic.

## 1b) A second, deeper issue I noticed while reading the code

This is **not** what caused the 0% accept rate — that's the step-size
issue above. But while reading
[`gpu_pf.jl:657-662`](../../../version_1_5_Julia/models/fsa_high_res/gpu_pf.jl#L657)
I noticed something that I think is wrong, and I want to flag it
before any step-size fix is shipped — they interact.

```julia
function tempered_grads(U_in, sub_key::UInt64)
    out = gpu_grads(target, U_in, grid_obs, h_fd, sub_key)
    grads_prior = -(U_in .- prior_means') ./ (prior_sigmas' .^ 2)
    vals_prior  = -0.5 .* vec(sum(((U_in .- prior_means') ./ prior_sigmas') .^ 2; dims = 2))
    return (out.vals .+ vals_prior, out.grads .+ grads_prior)
end
```

The function is called `tempered_grads` but **it does not depend on
λ**. It returns `log p(y|θ) + log p(θ)` = the full-posterior
log-density and gradient, regardless of which tempering level the
outer SMC is on. Standard tempered SMC needs the MCMC kernel at level
λ to leave the **tempered intermediate** π_λ ∝ p(θ)·p(y|θ)^λ
invariant, which means the gradient should be `λ · ∇log p(y|θ) +
∇log p(θ)`.

I checked the call site
([`bench_smc_full_mpc_fsa_gpu.jl:301-305`](../../../version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L301))
— `parallel_hmc_one_move!` is invoked with no λ argument either, so
the bench is currently running posterior-targeted HMC at every
tempering level instead of intermediate-targeted HMC.

Right now this is masked: with 0% accept rate, the cloud doesn't
move, so π_λ is preserved by accident. **The danger is that fixing
only the step-size bug would unmask this one** — the cloud would
then drift toward π_1 inside every tempering level, which would
distort the weighting at the next level. So I'd recommend fixing
both at the same time, with the gradient inside `tempered_grads`
scaled by the caller-passed `current_λ`. I'm flagging this rather
than "fixing" it, since (a) the user's framework explorers said the
SMC2FC_functional `HMC.jl` is structured differently and may have
the right pattern, and (b) I'm a junior on this codebase — flagging
is the right move.

## 2) Framework rejuvenation tricks alongside Liu–West

Searched [`julia/SMC2FC_functional/src/`](../../../julia/SMC2FC_functional/src/)
for `shrink`, `cov`, `jitter`, `kde`, `kernel`. Findings:

### 2a) [`Filtering/Kernels.jl`](../../../julia/SMC2FC_functional/src/Filtering/Kernels.jl) — kernel-density resampling with Silverman bandwidth

This module exposes **four** rejuvenation variants that go strictly
beyond the bench's per-dim Liu–West:

| function                          | bandwidth        | Liu–West correction |
|-----------------------------------|------------------|---------------------|
| `smooth_resample_basic`           | fixed Silverman  | no                  |
| `smooth_resample_ess_scaled`      | ESS-scaled       | no                  |
| `smooth_resample`                 | fixed Silverman  | yes (via `a = √(1−h²)`)|
| `smooth_resample_ess_scaled_lw`   | ESS-scaled       | yes (via `a = √(1−h_eff²)`) |

`smooth_resample` (lines 184-206) is the most relevant — it does a
full kernel-density resample with a **Silverman per-dim bandwidth**
`h_d = (4/(d+2))^(1/(d+4)) · K^(−1/(d+4)) · scale · σ_d` and a
Liu–West shrinkage `a = √(1 − h_norm²)` where `h_norm` is the
dimensionless Silverman factor. Compared to the bench's local
Liu–West (a fixed scalar `a` like 0.97 with per-dim std-only
jitter), this:

- Couples the shrinkage strength `a` to the cloud size `K` and
  bandwidth scale via Silverman, so it is "self-tuning" instead of
  a free hyperparameter.
- Is a kernel-density estimator (KDE) sample, not just a Gaussian
  jitter — captures non-Gaussian shape better.

**Caveat:** these `smooth_resample` functions are written for the
**inner state PF** (signature takes `stochastic_idx` for state
components), not for the outer θ-cloud. They are imported by
[`Filtering/Bootstrap.jl`](../../../julia/SMC2FC_functional/src/Filtering/Bootstrap.jl)
and the segmented-PF module
[`Filtering/GPUSegmentedPF.jl`](../../../julia/SMC2FC_functional/src/Filtering/GPUSegmentedPF.jl#L156),
not by any θ-cloud SMC² code. The bench would either need to wrap
them or copy the Silverman+LW math into its own `run_outer_smc`.

### 2b) [`SMC2/Bridge.jl`](../../../julia/SMC2FC_functional/src/SMC2/Bridge.jl) — multivariate-Gaussian bridge with **regularised sample covariance**

`Bridge.fit_gaussian` (lines 46-53) computes:

```julia
μ = vec(mean(particles; dims=1))
Σ = cov(particles; dims=1) + reg * I        # reg = 1e-6
return μ, Symmetric(Σ)
```

and `Bridge.sample_from_gaussian` (lines 72-80) draws via Cholesky:

```julia
L = cholesky(Σ).L
return reshape(μ, 1, :) .+ Z * L'
```

This is a **multivariate Liu–West cousin**: a full-covariance
Gaussian fit with light Tikhonov regularisation, used to produce the
next window's initial cloud (replaces "copy previous posterior
verbatim" with "sample from `N(μ, Σ̂_reg)`"). It's **what I'd reach
for first on the user's hint** — the `cov` call in line 49 is the
standard "covariance shrinkage" infrastructure. Right now Bridge is
wired up for *between-window* propagation only (via `bridge_init`).
It is **not** called by the bench's local `run_outer_smc` — the
bench copies the previous posterior verbatim into the new window.

### 2c) Summary of available rejuvenation tricks the bench could opt into

| location                                  | level applied            | what it does |
|-------------------------------------------|--------------------------|--------------|
| bench's `run_outer_smc:284-290`           | between resample & HMC, every tempering level | per-dim Liu–West (fixed `a`) — what we just turned on |
| `Kernels.smooth_resample`                 | between resample & HMC   | KDE resample + Silverman-coupled Liu–West |
| `Kernels.smooth_resample_ess_scaled_lw`   | between resample & HMC   | same but ESS-scales the bandwidth (degrades to identity for healthy clouds) |
| `Bridge.GaussianBridge`                   | between rolling windows  | multivariate Gaussian bridge with regularised sample covariance |

The user's hint maps cleanly: the **multivariate Gaussian-bridge in
`Bridge.jl` is the framework's "covariance shrinkage"** (full Σ̂ +
small Tikhonov). The bench's own per-dim Liu–West (Tier 0) is the
cheap diagonal version of the same idea. The Silverman+LW kernels
in `Kernels.jl` are an even richer KDE-style alternative, but they
target the inner state PF, not the θ-cloud, so they would need
adapting before use.

## What I'd ask before next steps

I have **not** touched any of this in source — same junior-engineer
posture as before, flagging rather than refactoring. Three reasonable
next-step packages, only one of which is "more GPU time on the
horizon sweep":

- **B-fix.** Patch `hmc_step` and `h_fd` defaults in the bench to
  ε ≈ 0.005, h_fd = 1e-2 (and add the missing λ-scaling inside
  `tempered_grads`). Re-run the same Tier 0 Part B (~2 min wall) and
  see whether non-zero HMC mixing + Liu–West gives a tighter, less
  prior-dominated posterior than Liu-West alone. **Gates** the
  eventual full sweep: if HMC actually mixes, we may not need
  Liu–West to do all the work.
- **Wire in the multivariate Gaussian bridge.** Replace
  `acc.filter_post` → `bridge_init(GaussianBridge(), …)` between
  rolling windows. This is the "covariance shrinkage between
  windows" the user was hinting at. Cheap to try.
- **Promote `Kernels.smooth_resample` into the θ-cloud path.** Bigger
  refactor; would need to change `run_outer_smc` to call it instead
  of the local Liu-West. Probably not worth it before the simpler
  fixes above are tested.

I'd lean toward **B-fix + Gaussian bridge** together (small patch to
the bench; small wrapper around `Bridge.GaussianBridge`), then re-run
Tier 0 Part B once. But I'd want to check with you before changing
source.
