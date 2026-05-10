# Julia v1.5 horizon sweep — filter HMC unified with framework ChEES — Sat May  9 19:19:00 BST 2026

Sweep root: `/home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_ctrl_explore_rebalanced_T14`

Configuration: filter HMC ON via framework's
parallel_hmc_one_move_generic! + chees_pick_L_generic (same code
path as the controller HMC); β=next_λ applied; ε=0.005,
h_fd=0.01, ChEES list=[4,8,16,32,64]. All filter rejuvenation
tricks ON. Fast (reduced) controller config.

## Per-horizon wall + exit code

| T_days | start | wall_s | exit_code | dir |
|---|---|---|---|---|
