# Julia v1.5 horizon sweep — 'filter HMC not needed' study — Sat May  9 09:33:59 BST 2026

Sweep root: `/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_no_filter_hmc`

Configuration: filter HMC OFF (--num-mcmc 0); all rejuvenation
tricks ON (Liu–West 0.97, Silverman+KDE 1.0, Gaussian bridge);
fast controller (--ctrl-n-smc 1024 --ctrl-num-mcmc 8);
controller-HMC diagnostics enabled.

## Per-horizon wall + exit code

| T_days | start | wall_s | exit_code | dir |
|---|---|---|---|---|
| 14 | 2026-05-09 09:33:59 | 177 | 0 | T14d_seed42 |
| 28 | 2026-05-09 09:36:56 | 386 | 0 | T28d_seed42 |
| 42 | 2026-05-09 09:43:22 | 760 | 0 | T42d_seed42 |
| 56 | 2026-05-09 09:56:02 | 1278 | 0 | T56d_seed42 |
| 70 | 2026-05-09 10:17:20 | 1836 | 0 | T70d_seed42 |
| 84 | 2026-05-09 10:47:56 | 2501 | 0 | T84d_seed42 |
