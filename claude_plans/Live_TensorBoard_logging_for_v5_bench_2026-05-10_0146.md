# Live TensorBoard logging for the v5 closed-loop SMC²-MPC bench

> Archived from plan mode: 2026-05-10 01:46.

## Context

Today the only way to inspect latent-variable / parameter trajectories produced by the v5 bench is to wait for the run to finish and open the PNGs the launcher writes (`v5_T<N>d_traces.png`, `v5_T<N>d_param_traces.png`, `v5_T<N>d_obs_channels.png`). For long horizons (T=42d, T=112d) that is hours of waiting before you know whether the controller is doing something sensible or the filter is collapsing.

Goal: at the end of every closed-loop stride, push the same series the end-of-run plotters consume into a TensorBoard event file. Run `tensorboard --logdir <SWEEP_ROOT>` once and watch progress live. End-of-run PNGs keep working unchanged — TensorBoard is *additive*, not a replacement.

## Where the hook fires

There is exactly one place to instrument: the foldl in `run_closed_loop_bench_v5` at `bench_loop.jl:319-323`, driven by `run_one_stride_v5` whose return statement at `bench_loop.jl:301-314` carries every quantity needed in scope:

- `p.final_state` — 6D true plant state at end of stride.
- `new_xhat` — 6D filter point estimate (posterior mean of the latent state).
- `new_filter_post` — `(n_smc, 37)` unconstrained posterior cloud, or `nothing` during warmup.
- `p.Phi_B`, `p.Phi_S` — bimodal stimulus this stride.
- `t_stride_s`, `n_temp_filter`, `n_temp_ctrl` — wall + diagnostics.

The conversion from unconstrained → constrained for the 37 estimated params is `exp.(U)` (LogNormal priors), exactly the same step `bench_postproc.jl:81-84` does at end-of-run.

## Files to modify

1. `version_1_5_Julia/Project.toml` — add `TensorBoardLogger` UUID `899adc3e-224a-11e9-021f-63837185c80f` (already pinned in `Manifest.toml`).
2. `version_1_5_Julia/tools_v5/bench/bench_args.jl` — add `--tensorboard true|false` flag, default `false`.
3. `version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl` — conditionally build `TBLogger`, thread through `ctx`, print `tensorboard --logdir` hint at startup.
4. `version_1_5_Julia/tools_v5/bench/bench_loop.jl` — call `log_stride_to_tb!` at end of `run_one_stride_v5` if `ctx.tb_logger !== nothing`.
5. **New file**: `version_1_5_Julia/models/fsa_v5/bench_tb_logging.jl` — helper holding the logger schema.
6. Three launchers (T7d, T42d, T112d) — header note about the new flag.

## What gets logged per stride

- 12 latent state scalars (`state/{truth,posterior_mean}/{B,S,F,A,K_FB,K_FS}`).
- 2 stride-mean Φ scalars (`phi/mean_B`, `phi/mean_S`).
- 3 diagnostics (`wall/stride_s`, `tempering/n_temp_{filter,ctrl}`).
- 111 param posterior quantile scalars (`params/<name>/{q05,q50,q95}` × 37 in `PARAM_NAMES_V5` order), only when filter active.

≈128 scalars/stride when filter is active, 17 during warmup.

## Defaults locked in

- Q1 cadence: every stride.
- Q2 param breadth: all 37, q05/q50/q95.
- Q3 per-bin obs in live feed: skip; end-of-run PNG covers them.
- Q4 flag parsing: lenient (`lowercase(args["tensorboard"]) in ("true","1","yes")`), mirrors `--open-loop`.
- Q5 auto-launch TB server from launcher: NO, just print the command.

## Verification

1. Smoke run with `--tensorboard true` at small scale; expect `tb_events/` with non-empty `events.out.tfevents.*`.
2. Default (no flag) is unchanged — no `tb_events/` dir, no `TensorBoardLogger` load.
3. Bad value parsing matches `--open-loop` convention.
4. End-of-run PNGs identical to a reference run at the same seed.

## Out of scope

- TensorBoard image summaries.
- Live TB for non-bench scripts.
- Wandb / mlflow.
- Reading `tfevents` back into Julia.
