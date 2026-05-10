# Julia v1.5 horizon sweep — cost-on-B (controller objective swapped from \int A to \int B) — Sat May  9 14:58:48 BST 2026

Sweep root: `/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_cost_on_B`

Configuration: filter HMC ON via framework's
parallel_hmc_one_move_generic! + chees_pick_L_generic (same code
path as the controller HMC); β=next_λ applied; ε=0.005,
h_fd=0.01, ChEES list=[4,8,16,32,64]. All filter rejuvenation
tricks ON. Fast (reduced) controller config.

## Per-horizon wall + exit code

| T_days | start | wall_s | exit_code | dir |
|---|---|---|---|---|
| 14 | 2026-05-09 14:58:48 | 543 | 0 | T14d_seed42 |
| 28 | 2026-05-09 15:07:51 | 1206 | 0 | T28d_seed42 |
