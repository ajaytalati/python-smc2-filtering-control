# FINDINGS — v1.5 Python Kalman-fusion fix (T=7d, seed=42)

> Re-run after fixing the v1.5 Python+JAX bench's two structural bugs:
> filter degeneracy (now Kalman-fused proposal, mirrors v2) and the
> closed-loop init_state cheat. Same seed, same matched config, same
> Julia run reused as the cross-stack reference.

## What changed since the previous T=7d run

| Issue | Before | After |
|---|---|---|
| Replan splice / horizon (A1+A2) | sawtooth Φ; shrinking horizon | monotone Φ; full T_days every replan |
| **Filter proposal** | bootstrap (`pred_lw=0`) — degenerate with GK-DPF | **Kalman fusion** with sequential-scalar update + Cholesky sample, mirrors v2's `propagate_fn` |
| Controller `init_state` | `plant_state.bfa` (cheating; reads true state) | `smoothed_state` extracted via `log_density_factory.extract_state_at_step` (filter posterior mean) |

## Diagnostic table

```
                                  Baseline (Φ=1.0)   Python+JAX MPC   Julia MPC
mean A (last 7 days)              0.0935             0.0808           0.0815
  vs baseline                          —             -13.6%           -12.8%
posterior MSE to truth                 —             0.0188 ★          0.0868
total wall time (min)                  —             5.3              37.2
mean GPU utilisation (%)               —             40.0             24.8
```

★ = best of all three.

## Verdict

**Filter side: FIXED.** Python's posterior MSE dropped from 1.41 → 0.0188
— a **74× improvement** — and is now **5× better than Julia's 0.087** at
matched config (N=32 outer, K=200 inner). The param-traces plot shows
every one of the 10 estimated params hugging truth with tight bands
(was: tau_F stuck at 3 vs truth 7; sigmas drifting upward unbounded).

The fix was Kalman fusion in `_propagate_fn_framework` (mirrors v2's
working pattern with the same smc2fc framework). The smc2fc GK-DPF
v3-lite filter was designed for guided proposals; using it with
bootstrap (the original v1.5 design) caused the framework's
weight-smoothing kernel + OT regularisation to push particles toward
biased fixed points. Smoking-gun evidence: at higher particle counts
(N=1024/K=800), the bootstrap variant got *worse* (MSE 1.41 → 31.16),
which is the classic "structural mis-spec" signature.

**Controller side: H3 remains.** Mean A still loses to baseline by
−13.6% (Python) / −12.8% (Julia). Both stacks pick rest-heavy
schedules (Python: Φ=1 day 1 → ~0.15 days 2-6 → 0.4 day 7; Julia
similar shape). The cost `J = -∫A + λ_F · F-barrier` (no Φ² penalty)
at T=7d horizon admits the "rest cures all" optimum endogenously
because suppressing Φ keeps F low and avoids any F-barrier cost while
gaining only a small −∫A penalty. The user's reference v2 results
ramp Φ UP at T=14d/T=28d — possibly because the longer horizon makes
the "build B then sprint" payoff dominant, or because v2's G1
reparametrisation around (A_TYP, F_TYP) shifts the bifurcation
balance.

## Recommended next steps for H3

1. **Test at T=14d or T=28d** to see if the longer horizon flips the
   controller out of the rest optimum (matches v2's pattern).
2. **Re-introduce `λ_phi · ∫Φ²` penalty term** — currently 0 in both
   v1.5 and v2 by default, but v2 might tune it differently in
   practice.
3. **Add a "minimum mean-A over planning window" floor** to the
   cost — soft constraint that the controller can't sit at Φ=0 for
   more than e.g. 1-2 days.
4. **Investigate if v1.5's drift basis (kappa_B/F directly, no
   reparametrisation around A_TYP) makes the rest optimum more
   attractive than v2's**.

These are all separate fixes from the filter fix — the filter is now
correct.

## Plot

[`comparison.png`](comparison.png) — 6-panel: A trajectory, applied
Φ, posterior medians (tau_F, B_inf), GPU SM utilisation, per-stride
wall time. Both Python and Julia overlaid; baseline grey dashed.

[`python/v15_T7d_traces.png`](python/v15_T7d_traces.png) and
[`python/v15_T7d_param_traces.png`](python/v15_T7d_param_traces.png)
— Python state + param panels at the new fix.

## Files changed in this fix

- `version_1_5_Python_JAX/models/fsa_high_res/estimation.py` —
  `_propagate_fn_framework` rewritten with sequential-scalar Kalman
  fusion across the 3 direct-Gaussian channels (H = I, R =
  diag(σ_*_obs²)); samples from fused posterior via Cholesky;
  returns `pred_lw = log_pred_total - obs_ll(y_new)` to cancel the
  framework's `obs_log_weight_fn` re-add at line 156 of
  `gk_dpf_v3_lite.py`. Mirrors v2 `propagate_fn` line 217-257.
- `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py` —
  after each filter window, extract `fixed_init_state` via
  `log_density_factory.extract_state_at_step` (mirrors v2 bench
  line 366-367); use this as the controller's `init_state_dict`
  instead of `plant_state.bfa`.
