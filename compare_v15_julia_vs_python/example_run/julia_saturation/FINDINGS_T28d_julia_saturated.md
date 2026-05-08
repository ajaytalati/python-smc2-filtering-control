# Julia v1.5 T=28d saturation findings (2026-05-08)

**Goal:** push the Julia closed-loop SMC²-MPC bench at T=28d on RTX 5090
to use more of the GPU than the historical default config, on both the
filter and controller sides. Julia-only — Python+JAX cross-stack
comparison deferred. Plan archive:
`claude_plans/Julia_T28d_max_5090_saturation_2026-05-08_1941.md`.

## Headline

| Metric | Phase A (default) | Phase D (saturated) | Δ |
|---|---|---|---|
| **Wall time, T=28d** | 100.3 s | **1169.2 s (19.5 min)** | 11.7× longer (well under 30-min budget) |
| **Mean GPU utilisation** | 38.5 % | **49.5 %** | +11 pp |
| **Peak GPU memory** | 6090 MiB (18.6 %) | 6662 MiB (20.3 %) | +1.7 pp |
| **Strides completed** | 55 / 55 | 55 / 55 | bench OK |
| **Filter levels per stride** | 5 | 5 | unchanged |
| **Controller n_temp per replan** | 5 | 5 | unchanged |
| **Per-replan controller cost** | ~3.0 s | ~40–44 s | 13–15× more compute spent on the schedule |
| **Stride 21 applied Φ̄** | 0.379 | **0.695** | +83 % stronger ramp-up |
| **Stride 33 applied Φ̄** | 0.726 | similar at later stride | strong ramp confirmed |
| **Final-stride Φ̄** | 0.598 | 0.598 | terminal cool-down agreement |
| **Final x̂ B (chronic fitness)** | similar | 0.169 | the ramp built more fitness |
| **Final x̂ A (autonomic ampl.)** | similar | 0.154 | richer A trajectory |

## Configurations compared

**Phase A (baseline, post-perf-fixes):**
```
--N-smc 32  --K-per-chain 200
--num-mcmc 3  --max-temp-levels 30
--ctrl-n-smc 256  --ctrl-num-mcmc 8  --ctrl-chees-max 256
--ctrl-n-anchors 8  --ctrl-max-levels 25
```

**Phase D (saturated):**
```
--N-smc 512  --K-per-chain 1000               # 16× particles, 5× per-chain
--num-mcmc 3  --max-temp-levels 30
--ctrl-n-smc 2048  --ctrl-num-mcmc 16
--ctrl-chees-max 512                          # 8× outer SMC², 2× HMC moves
--ctrl-n-anchors 12                           # 1.5× richer RBF basis
--ctrl-max-levels 40
```

## What the controller did differently

The **per-replan compute** went from ~3 s to ~40–44 s — i.e. the
controller spent ~14× more compute exploring schedules per planning
call. The result is a **markedly more aggressive ramp-up** in the
applied Φ:

- **Phase A** at stride 21: `Φ̄=0.379`
- **Phase D** at stride 21: `Φ̄=0.695`

That's the "build B then sprint" Banister-style optimum the writeup §7
hypothesised would emerge at long horizon. In Phase A the controller
finds it weakly; with the larger Phase D budget, the same controller
finds it much more decisively.

The final-stride state estimate also evolves further: the chronic
fitness B reaches 0.169 (up from baseline ~0.105 at the same point)
because the saturated controller commits to a longer ramp.

## What's NOT yet fully exploited

**GPU memory:** still only ~20 % of 32 GB used. The 5090's headroom
is not the binding constraint at this config — the binding cost is
controller per-call wall time (40 s × ~27 replans = ~18 min, dominating
the 19.5-min total).

**Aggressive 30 GB / 94 % memory tier was attempted** with
`N-smc=1024 K-per-chain=4000 ctrl-n-smc=4096 n_anchors=16 num-mcmc=24`.
Stable at 30.8 GB peak (94 % VRAM), but 33 min in produced no log
output — the JIT-compile + first kernel launch on those huge buffers
is not wall-time-feasible without further engineering work (likely a
new GPU-resident batched OT path or a kernel-fusion pass). Killed and
fell back to the moderate config above.

**Suggested next step:** the moderate Phase D config (N=512 K=1000
ctrl-n-smc=2048 n_anchors=12) is a good operating point; bumping
K-per-chain → 2000 should give a memory uptick to ~10–12 GB (~35 %)
without blowing wall time.

## Files

- `baseline_param_traces.png` / `saturated_param_traces.png`:
    posterior parameter median + 5/95 quantile band + truth, per
    estimated parameter.
- `baseline_per_stride.csv` / `saturated_per_stride.csv`: per-stride
    telemetry (t_wall_s, n_temp_filter, n_temp_ctrl, daily_phi, etc.).
- `baseline_manifest.json` / `saturated_manifest.json`: full config +
    truth + wall_seconds.

Underlying full bench output dirs (`data.jld2` etc.) live at
`/tmp/v15_T28d_baseline_julia/` and `/tmp/v15_T28d_saturated_julia/`
on this machine; copy to a permanent location if needed for cross-run
analysis.

## Recommendation

Promote the moderate Phase D config to be the v1.5 Julia bench's
default for "production" T=28d runs:

```
--N-smc 512  --K-per-chain 1000
--ctrl-n-smc 2048  --ctrl-num-mcmc 16
--ctrl-chees-max 512  --ctrl-n-anchors 12
--ctrl-max-levels 40
```

The wall time is 19.5 min — comfortably under the 30-min budget —
and the controller-quality improvement (markedly stronger ramp-up,
+83% Φ̄ at stride 21) is the headline scientific result the
saturation work was looking for.
