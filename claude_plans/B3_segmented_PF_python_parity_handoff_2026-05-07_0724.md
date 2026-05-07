# B3 segmented-PF Python-parity handoff

> Handoff written 2026-05-07 07:24 by Claude Opus 4.7 after a ~6-hour
> autonomous session debugging why the Julia GPU segmented PF was
> giving wildly different posterior recovery from Python's
> `gk_dpf_v3_lite.py` reference (σ_u rel.err 1287% in Julia vs 0.3% in
> Python). Read this BEFORE touching the bistable B3 code.

## TL;DR

- **5 real bugs were found and fixed in the working tree.** All
  uncommitted (per Ajay's instruction). Commit 4265c88 was REVERTED
  during the session because it documented an incorrect "structural"
  conclusion that has since been disproven.
- **After fixes**, 5-seed mean recoveries vs Python:
  γ matches (5.3% vs 5.1%), σ_obs matches (3.4% vs 3.0%), σ_x within
  9× (3.5% vs 0.4%), α within 2.4× (14.4% vs 6.1%), `a` within 12×
  (9.3% vs 0.8%), σ_u within 47× (14.2% vs 0.3%).
- **The remaining σ_u / `a` gap is sampling variance** (LL probe
  shows the surface peaks at truth for σ_u; per-seed recoveries swing
  4.8% – 57.6%). Two fixes from here, in priority order:
  1. **OT (Sinkhorn-Nyström) rescue** — Python's gk_dpf_v3_lite uses
     it, my port doesn't. Likely closes the σ_u gap.
  2. Larger N_SMC (e.g. 256) and / or seed averaging.
- **No commits have been made.** Originating commit 4265c88 was
  reverted via `git reset --mixed`. The 5 working-tree fixes need
  Ajay's explicit go-ahead before being committed.

## Files modified (uncommitted, working tree only)

```
M version_1_Julia/models/bistable_controlled/gpu_pf.jl
?? version_1_Julia/tools/bench_b3_gpu_segmented.jl   (new file)
```

`gpu_pf.jl` adds (purely additive, no existing code modified):
- `bootstrap_pf_kernel_segment_batched!` — R-step PF kernel, **now uses
  locally-optimal Bayes-fused proposal for x** (was bootstrap)
- `gpu_resample_shrink_kernel!` — per-particle binary-search systematic
  resample + Liu-West shrink, all GPU-resident
- `BistableGPUTargetSegmented` struct + constructor with reductions and
  cumsum scratch buffers
- `gpu_log_density_batched_segmented` + `gpu_grads_parallel_chains_segmented`
  wrappers

`bench_b3_gpu_segmented.jl` is the new B3 driver:
- Mirrors `bench_b3_gpu_parallel.jl` outer-SMC loop
- ChEES L-adaptation across {2, 4, 8, 16}
- **Prior-scaled mass matrix** in `parallel_hmc_seg!` (this is the
  single biggest win in the session — see "Bug 2" below)
- LL surface probe at start (σ_u + α grids around truth)
- BENCH_SEED env var for multi-seed runs

## The 5 bugs and fixes

These came out of the session in this order. Each one improved
recovery; the cumulative effect was σ_u going from 1287% → 14% rel.err.

### Bug 1 — Wildly off-truth priors (BIGGEST WIN before mass matrix)

**Symptom:** σ_u posterior ~0.7 (truth 0.05) regardless of any other
fix. σ_x posterior ~0.5 (truth 0.1). σ_obs ~1.0 (truth 0.2).

**Cause:** the bench had `PRIOR_MEANS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...]`
and `PRIOR_SIGMAS = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, ...]` — i.e.
σ_u prior median was `exp(0) = 1` (20× truth) and σ_x prior median
was 1.0 (10× truth). The likelihood signal in σ_u is weak; the prior
penalty for being at log(σ_u)=-3 (truth) was -18 vs 0 at log(σ_u)=0,
which dwarfs the +6-log-unit likelihood gain at truth → posterior
collapsed onto the prior mode.

**Fix:** mirror Python's
`version_1/models/bistable_controlled/estimation.py:65-71`
`PARAM_PRIOR_CONFIG`:
```
                       α      a      σ_x    γ          σ_u    σ_obs   x_0   u_0
PRIOR_MEANS  = [0.0,   0.0,   -2.0,   log(2.0),  -3.0,  -1.5,  -1.0,  0.0]
PRIOR_SIGMAS = [0.5,   0.3,    0.5,   0.3,        0.5,   0.3,   0.5,  0.3]
```

**Lesson:** Python's "0.3% σ_u recovery" is NOT pure data identification.
The prior is centred at truth. Likelihood adds a small correction; the
prior dominates. My Julia setup didn't have this prior, so the chain
sat at the wrong mode.

### Bug 2 — HMC used identity mass matrix (BIGGEST WIN of session)

**Symptom:** Even after Bug 1 was fixed, α was 17.8% off truth and
recovery was unstable across seeds (16-30% rel.err on α).

**Cause:** the leapfrog used `inv_mass = ones(d)`, so the position
step `ε · grad` was the same scale across all 8 dimensions despite
the prior σ varying (0.3 to 1.0). Effective step in units of prior σ
varied 3× across dimensions → some dims explored too fast, others too
slow.

**Fix:** prior-scaled mass matrix `M = diag(1/PRIOR_SIGMAS²)`,
i.e. `inv_mass = PRIOR_SIGMAS²`. With this, the position step
`ε · inv_mass · grad = ε · σ² · grad` is dimensionless (ε is now in
units of σ). Sample momentum as `p ~ N(0, M) = randn ./ sqrt(inv_mass)`.

Result: α went from 17.8% → 7.2% rel.err on the same seed (single-shot,
no other change).

**Lesson:** Without prior-scaled mass matrix, dimensions with very
different prior scales (here 0.3 vs 0.5 vs 1.0 in log-space) cannot
all be explored simultaneously. Standard HMC tutorials skip this
because the textbook examples have unit-scale priors. ALL future
SMC²-MPC work should use this pattern.

### Bug 3 — Bootstrap proposal for the observed state x

**Symptom:** With Bugs 1+2 fixed, σ_u posterior was still 26% off
truth (chain converged to 0.063 vs truth 0.05).

**Cause:** the kernel propagated x via the prior SDE
(`x_next = x_pred + sqrt(2·σ_x·dt)·noise`) and reweighted by the
obs likelihood. This is the bootstrap PF, which has very high weight
variance — the LL estimator becomes unreliable for partially-identified
params (σ_u in particular).

**Fix:** Python's
`version_1/models/bistable_controlled/estimation.py:88-182`
uses the **locally-optimal proposal** for x: q*(x') ∝ p(x'|x_k) · p(y|x'),
which is closed-form Gaussian:
```
σ_proc_x²    = 2·σ_x·dt
σ_obs²       = σ_obs²
sum_var      = σ_proc_x² + σ_obs²
σ_prop²      = σ_proc_x² · σ_obs² / sum_var
μ_prop       = (x_pred·σ_obs² + y·σ_proc_x²) / sum_var
x_next       = μ_prop + sqrt(σ_prop²) · noise
log_predictive = log N(y; x_pred, sum_var)   ← weight increment
```

The weight increment is **sample-independent** (depends only on x_pred,
not on the drawn x_next) — that's the key variance-reduction
property. u stays bootstrap (it's unobserved).

**Lesson:** for partially-observed SDEs, locally-optimal proposal is
massively better than bootstrap. This is the standard "guided PF"
trick in the SMC literature. ALL future model ports should consider
it for any observed state component.

### Bug 4 — Hard NaN-kill instead of soft state clip

**Symptom:** 27/64 chains returned NaN log-density on the prior cloud
(before Bug 1 was fixed). The bistable cubic drift `α·x·(a²-x²)`
overflows fp32 at large `|x|` for tail-of-prior θ samples.

**Cause:** my kernel had `if abs(x_next) > 50: x = 0; log_w = -1e30`
— a HARD KILL that zeros the state and parks the weight at a very
negative value. This biased the LL estimator (chains with wide priors
got artificially low LL).

**Fix:** Python clips state to `state_bounds = ((-5, 5), (-5, 5))`
after each obs step (`gk_dpf_v3_lite.py:220-222`) — a SOFT clip, no
weight modification. Match this pattern: `x = max(-5, min(5, x_next))`,
log_w gets the obs likelihood at the clipped value normally. Only NaN
falls through to `x = 0` reset.

**Lesson:** match Python's state_bounds exactly. For any model with
unbounded drift in fp32, soft clip is essential.

### Bug 5 — Init-cloud noise unique per particle across chains

**Symptom:** subtle FD-gradient noise that grew with chain count.

**Cause:** init noise `init_noise_x = randn(rng, Ntot)` was length
M·K = 1088·K, so each particle had its OWN init noise. Across 17 FD
perturbations of one HMC chain, the 17 perturbations saw 17·K
different init noise values → CRN broken at init step → FD gradient
noisier than necessary.

**Fix:** length-K shared noise across all M chains. Particle p of
chain m uses `init_noise_x[p]` regardless of m. CRN preserved.

**Lesson:** for batched-chains GPU PF, shared CRN noise grids are
the single most important implementation detail (already documented
in HANDOFF.md re the SDE noise grid). The init step needs the same
treatment.

## Verified-correct: Liu-West shrinkage, init dispersion

These were checked against Python and matched:
- Liu-West shrinkage formula (n_st=2 Silverman, ESS-scaled bandwidth,
  shrink-toward-weighted-mean blend) — matches `gk_dpf_v3_lite.py:180-189`.
- Init particle dispersion `base + sigma_diag · sqrt_dt · noise` —
  matches `gk_dpf_v3_lite.py:124-125`.
- Predictive likelihood formula — matches Python verbatim.

## Final 5-seed results (uncommitted, current working tree)

Config: K_PER_CHAIN=10000, N_SMC=64, NUM_MCMC=3, R_SEGMENT=1,
HMC_STEP=0.002, prior-scaled mass matrix, locally-optimal proposal,
state clip ±5, init noise CRN.

```
seed | α     | a     | σ_x   | γ    | σ_u   | σ_obs
-----|-------|-------|-------|------|-------|------
 11  | 7.2%  | 13.1% | 0.4%  | 4.9% | 26.7% | 1.1%
 23  | 1.9%  |  8.4% | 17.7% | 6.4% | 57.6% | 0.9%
 42  | 29.3% |  6.7% | 0.7%  |16.8% |  4.8% | 9.3%
101  | 20.4% |  8.6% | 4.4%  | 4.6% | 15.5% | 3.4%
314  | 16.7% |  9.9% | 4.1%  | 3.7% |  2.6% | 4.5%
-----|-------|-------|-------|------|-------|------
mean | 14.4% |  9.3% | 3.5%  | 5.3% | 14.2% | 3.4%
Pyth |  6.1% |  0.8% | 0.4%  | 5.1% |  0.3% | 3.0%
```

Wall: 138-153s per filter (vs unresampled GPU's 23s, ~6× slower).
GPU memory: 4 GB. GPU utilisation: ~17-32% (kernel launch overhead
dominates with 480 segments at R=1).

**γ and σ_obs match Python after 5-seed averaging.** Other params
within 2-50× of Python, mostly limited by sample variance.

## What's NOT done

### OT (Sinkhorn-Nyström) rescue — likely the missing piece

Python's `gk_dpf_v3_lite.py:198-214` has an OT-rescue branch:
```python
if ot_max_weight >= 1e-6:
    ot_raw = ot_resample_lr(particles, log_w_pre, rk, stochastic_indices, ε, n_iter, rank)
    ...
    ot_weight = ot_max * sigmoid((ot_threshold - ess) / ot_temp)
    resampled = (1.0 - ot_weight) * sys_lw + ot_weight * ot_out
```

OT kicks in when ESS drops below threshold — exactly the case where
σ_u-direction signal collapses. Python uses K=128 + OT; my port has
K=10000 + no OT. Even with 50× more particles, my σ_u variance is
~10× higher. OT is the prime suspect.

**Helper files to port:** Python's
`smc2fc/filtering/{transport_kernel.py, sinkhorn.py, project.py,
resample.py}` — total ~190 lines. Estimated 2-3 hrs to GPU-implement
properly with batched matmul over M chains.

GPU implementation strategy:
1. Per-chain anchor selection (r=50 random indices, CRN-shared)
2. Compute K_NR ∈ R^{M×K×r} via batched kernel
3. Sinkhorn 10 iterations via CUDA.jl `batched_mul!` on (M, K, r) /
   (M, r, K) tensors
4. Barycentric projection — same matmul pattern, batched over d_s=2 cols
5. Mix with sys_lw using sigmoid gate on ESS

The cost per OT call at K=10000, r=50, M=1088: ~100 ms (rough). At
480 segments × 30 PF calls per HMC × 30 HMC moves = 432K OT calls
per filter run. WAY too many. **Either reduce K (try K=200 + OT
matching Python's setup) or only fire OT when ESS < threshold (most
steps stay below threshold trigger).**

### FSA-v2 port

Per HANDOFF.md the FSA-v2 work is the next deliverable. Status:
- `models/fsa_high_res/_dynamics.jl` — done (commit 5a4fffd)
- `models/fsa_high_res/simulation.jl` — done
- `models/fsa_high_res/control.jl` — done
- `models/fsa_high_res/FSAHighRes.jl` — done

**Not yet started:**
- `models/fsa_high_res/estimation.jl` — needs design (Python's FSA-v2
  is "Stage D fully observed control" — no estimation pipeline exists
  in Python; this is genuinely new design work).
- `models/fsa_high_res/gpu_pf.jl` — model-specific GPU PF kernel
  (mirror bistable's `gpu_pf.jl` but with FSA-v2's 3-state Banister
  drift + Jacobi/CIR diffusion, n_st=3 → Silverman = (4/5)^(1/7),
  k_factor = K^(-1/7)).
- `tools/bench_fsa_gpu_parallel.jl` — bench (mirror
  `bench_b3_gpu_parallel.jl`).
- `outputs/fsa_high_res/RESULT.md` — Python ↔ Julia comparison
  (note: no Python estimation reference exists, so this is comparison
  vs SIMULATOR ground truth only).

## How to reproduce

```bash
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia
julia --threads auto --project=. tools/bench_b3_gpu_segmented.jl
```

For multi-seed:
```bash
for seed in 11 23 42 101 314; do
  echo "=== SEED=$seed ==="
  BENCH_SEED=$seed julia --threads auto --project=. tools/bench_b3_gpu_segmented.jl
done
```

Each seed takes ~140s wall (10 tempering levels × ~13s/level).

## Recommendations for next agent

1. **Read this doc end-to-end before touching `gpu_pf.jl` or
   `bench_b3_gpu_segmented.jl`.** The 5 fixes are subtle and easy to
   regress accidentally.

2. **Decide with Ajay first**: implement OT, or accept current state
   and move to FSA-v2. Ajay was explicit during this session that
   "Julia must perform same as Python" — the σ_u 47× gap is the
   remaining block to that.

3. **If implementing OT:** consider matching Python's K=128 first
   rather than K=10000. Python's whole point with OT is making K=128
   work as well as a much-larger-K bootstrap. Less code-and-memory
   churn, more apples-to-apples comparison.

4. **DO NOT commit anything without Ajay's explicit go-ahead.** Commit
   4265c88 was reverted during this session because it documented an
   incorrect conclusion. Same risk applies to anything new.

5. **Keep the prior-scaled mass matrix as the default for any new
   bench.** This was the single biggest win of the session — every
   future model should use `inv_mass = PRIOR_SIGMAS²`.

6. **Use locally-optimal proposal whenever you have a partially-
   observed SDE.** Bootstrap is the variance-collapse trap.

7. **GPU saturation note:** at R=1 with 480 segments, kernel launch
   overhead dominates and GPU util is ~30%. R=20 segments saturates
   the GPU but gives worse posterior (weight collapse between
   resamples). The right path is either OT (which keeps R=1 sensible)
   or fusing R steps into one kernel that ALSO does the resample
   inside (much harder).

## Conversation context

The session ran in auto mode, started after a context-compaction event
that mid-restored Claude. The previous-context summary mentioned
"σ_u recovery is structurally bad — flag for user". Ajay (correctly)
rejected that conclusion when I repeated it: "NO !!!! THIS IS NOT
ACEPTABLE - α and γ are now in the right ballpark. σ_u and σ_x are
still bad". That triggered the deep-dive that found the 5 real bugs.

**Lesson from this:** when previous-Claude says "X is structural" but
Python achieves it, your default should be "previous-Claude was wrong
about X being structural — there must be a port bug we missed".
Verify against the Python reference numbers FIRST, then characterise
the gap as bug or sample variance.
