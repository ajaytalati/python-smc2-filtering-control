# 6-horizon sweep with filter HMC OFF — measurements

**Run date:** 2026-05-09 09:34 → 11:30. Sweep root: this directory.

## 1. What was run

```
julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
    --T-days <T> --seed 42 \
    --num-mcmc 0 --hmc-step-size 0 --hmc-leapfrog 0 \
    --liu-west-a 0.97 --smooth-resample-bw 1.0 --gaussian-bridge true \
    --N-smc 512 --K-per-chain 1000 \
    --ctrl-n-smc 1024 --ctrl-num-mcmc 8 \
    --collect-ctrl-diagnostics true
```

T ∈ {14, 28, 42, 56, 70, 84} d, seed 42. Filter HMC disabled
(`num_mcmc = 0`). Controller HMC unchanged (ChEES, fast config).

## 2. Measured wall + closed-loop metrics

Source: [`horizons_results.csv`](horizons_results.csv) (one row per
horizon, written by `compare_v15_julia_vs_python/src/extract_horizon_results.py`).

| T_days | wall (s) | mean A MPC | final A MPC | mean A baseline | final A baseline | F violations | peak VRAM (MiB) | mean GPU util % |
|-------:|---------:|-----------:|------------:|----------------:|-----------------:|-------------:|----------------:|----------------:|
| 14     | 133.2    | 0.0766     | 0.0847      | 0.1138          | 0.1819           | 0            | 5399            | 29.6            |
| 28     | 342.0    | 0.0947     | 0.1607      | 0.1808          | 0.3417           | 0            | 5390            | 47.5            |
| 42     | 715.5    | 0.1673     | 0.5205      | 0.2848          | 0.6462           | 0            | 6091            | 58.3            |
| 56     | 1234.6   | 0.3319     | 1.0662      | 0.4010          | 0.8244           | 0            | 5273            | 65.2            |
| 70     | 1790.8   | 0.4914     | 1.1972      | 0.4921          | 0.8937           | 0            | 5272            | 71.0            |
| 84     | 2456.6   | 0.6115     | 1.2376      | 0.5648          | 0.9071           | 0            | 5776            | 74.5            |

## 3. Wall comparison vs original 6-horizon baseline

Baseline source: `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/horizons_results.csv`.

| T_days | this run wall (s) | baseline wall (s) | ratio |
|-------:|------------------:|------------------:|------:|
| 14     | 133.2             | 422.7             | 3.17× |
| 28     | 342.0             | 1259.3            | 3.68× |
| 42     | 715.5             | 2286.6            | 3.20× |
| 56     | 1234.6            | 3979.4            | 3.22× |
| 70     | 1790.8            | 5586.4            | 3.12× |
| 84     | 2456.6            | 7236.4            | 2.95× |

## 4. Controller-HMC measurements (new instrumentation)

Source: [`docs/controller_hmc_summary.csv`](docs/controller_hmc_summary.csv),
aggregated by `compare_v15_julia_vs_python/src/diag_controller_hmc.jl`
from each horizon's `controller_diagnostics.csv`.

| T_days | n_replans | total levels | mean accept % | median accept % | mean ChEES L | mean β_max | mean ESJD/εL | mean ΔlogD | mean wall/level (s) |
|-------:|----------:|-------------:|--------------:|----------------:|-------------:|-----------:|-------------:|-----------:|--------------------:|
| 14     | 13        | 69           | 97.93         | 97.91           | 16.0         | 158.20     | 14.08        | 0.012      | 1.69                |
| 28     | 27        | 136          | 97.43         | 97.74           | 16.0         | 19.96      | 14.03        | 0.079      | 2.37                |
| 42     | 41        | 230          | 96.17         | 96.48           | 16.0         | 7.16       | 13.41        | 0.344      | 3.00                |
| 56     | 55        | 328          | 94.97         | 95.56           | 16.0         | 3.15       | 13.30        | 0.781      | 3.67                |
| 70     | 69        | 412          | 93.49         | 94.78           | 16.0         | 1.82       | 13.20        | 1.194      | 4.27                |
| 84     | 83        | 492          | 92.68         | 94.35           | 16.0         | 1.27       | 13.15        | 1.609      | 4.92                |

**Filter-HMC reference (separate, vacuous in this run):** with
`--num-mcmc 0` the filter HMC loop iterates 0 times per tempering
level; the `accept=0%` lines printed in the bench log are arithmetic
artefacts (`0 / max(1, 0*N_smc) = 0`), not measurements of HMC
quality. The original baseline run logs 0 % accept across 2880
levels and Tier 0 (LW=0.97) logs 0 % across 130 levels — those WERE
measurements of attempted HMC moves.

## 5. Plots on disk (for visual inspection — no claims made about them in this report)

- Per-horizon parameter traces: `T<N>d_seed42/v15_T<N>d_param_traces.png` (×6).
- Per-horizon state traces: `T<N>d_seed42/v15_T<N>d_traces.png` (×6).
- Cloud std (log-space) per stride, all 6 horizons overlaid: [`docs/cloud_std_per_stride_all_horizons.png`](docs/cloud_std_per_stride_all_horizons.png).
- Per-horizon controller-HMC diagnostic 6-panel plots: `docs/controller_hmc_diagnostics_T<N>d.png` (×6).

## 6. Source edits applied (so this run is reproducible)

- [`julia/SMC2FC_functional/src/Control/GPUControlSMC.jl`](../../../julia/SMC2FC_functional/src/Control/GPUControlSMC.jl):
  added `collect_diagnostics::Bool=false` keyword on
  `run_tempered_smc_gpu` (4-tuple return when true); added
  `return_scores::Bool=false` keyword on `chees_pick_L_generic`.
  Existing 3- / 2-tuple return paths unchanged.
- [`version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`](../../../version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl):
  added `--collect-ctrl-diagnostics` flag (default `true`); plumbed
  through `controller_plan` and `stride_step`; writes
  `controller_diagnostics.csv` per run; manifest now records
  `liu_west_a`, `smooth_resample_bw`, `gaussian_bridge`.
- [`version_1_5_Julia/tools/launchers/run_julia_horizon_sweep_no_filter_hmc.sh`](../../../version_1_5_Julia/tools/launchers/run_julia_horizon_sweep_no_filter_hmc.sh):
  new sibling launcher that drives the 6-horizon sweep with the new
  flag set.
- [`compare_v15_julia_vs_python/src/diag_controller_hmc.jl`](../../src/diag_controller_hmc.jl):
  new aggregator + plotter for `controller_diagnostics.csv`.

## 7. What this report does NOT claim (and would need extra work to claim)

- Per-parameter posterior median comparison vs truth or vs baseline.
  Not measured here. To do this: load `data.jld2::posterior_particles`
  for each horizon of both runs, compute the median of the final-stride
  cloud (taking `log` first, since the saved array is in constrained
  space).
- Cloud-std-vs-time numerical curves. Visible in the PNG; not extracted
  to numbers. To do this: re-emit the per-stride per-parameter std
  values as a CSV from `diag_cloud_collapse.jl`.
- Any causal claim about which knob (`liu_west_a`,
  `smooth_resample_bw`, `gaussian_bridge`) is responsible for which
  effect. Not run as ablations.
- Any verdict on "is filter HMC needed". The wall and closed-loop
  metrics above are consistent with "removing it costs nothing", but
  the posterior-side claim requires the median comparison in the first
  bullet above.
