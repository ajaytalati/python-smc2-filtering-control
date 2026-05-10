# Add `--init-phi-B` / `--init-phi-S` flags to override the closed-loop initial controller stimulus

> Archived from plan mode: 2026-05-10 06:28.

## Context

The v5 closed-loop bench currently hardcodes the controller's initial stimulus to `Φ_B = Φ_S = 1.0`. With the default `--replan-K 2`, that means stride 1 of every closed-loop run is driven at constant moderate Φ until the controller's first replan at stride 2. The user wants to start the controller from any chosen Φ pair (e.g. `0.3, 0.3`) without editing source.

## Where it's set

Two active hardcode sites under `version_1_5_Julia/`:

1. `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl:269` — `(fill(Float32(1.0), T_total_bins), fill(Float32(1.0), T_total_bins))`. **Closed-loop initial plan.** Drives the plant until the first replan.
2. `tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl:246` — `base_phi_seq = [(1.0, 1.0) for _ in 1:T_total_bins]`. **Baseline rollout.** Produces grey reference curves on end-of-run plots and `state/baseline/*` series in TensorBoard.

`gpu_control_v5.jl:337 Phi_default = 1.0` is **NOT** related — it's the sigmoid-decoder bias (controller-internal regularisation), not the initial stimulus.

## Replan cadence

`bench_loop.jl:115`: replan fires when `stride_idx % replan_K == 0`. With default `--replan-K 2`, the initial plan covers stride 1 only. With `--replan-K 4`, strides 1–3.

## Scope (confirmed by user)

**Both sites follow the new flag.** The baseline rollout AND closed-loop initial plan both shift to the user's chosen Φ. Comparing MPC vs baseline becomes "MPC vs constant Φ at the user's chosen level" rather than "MPC vs constant Φ=1".

## Files to modify

1. `version_1_5_Julia/tools_v5/bench/bench_args.jl` — two new Float64 flags, default 1.0.
2. `version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl` — read + validate the two values once, replace both hardcoded sites, update the comment header at line 243.

No new files. No model-side changes. No launcher changes.

## Concrete changes

**`bench/bench_args.jl`** — add to defaults:

```julia
"init-phi-B" => 1.0,
"init-phi-S" => 1.0,
```

(Type-driven Float64 parsing already in place.)

**`bench_smc_full_mpc_fsa_v5_gpu.jl`** — single source of truth before line 243:

```julia
init_phi_B = Float64(args["init-phi-B"])
init_phi_S = Float64(args["init-phi-S"])
phi_max = 3.0
(0.0 <= init_phi_B <= phi_max) || error("--init-phi-B must be in [0, $phi_max], got $init_phi_B")
(0.0 <= init_phi_S <= phi_max) || error("--init-phi-S must be in [0, $phi_max], got $init_phi_S")
@info "initial Φ (baseline + closed-loop init plan): B=$init_phi_B S=$init_phi_S"
```

Replace line 246 baseline rollout: `base_phi_seq = [(init_phi_B, init_phi_S) for _ in 1:T_total_bins]`.

Replace line 269 closed-loop fallback: `(fill(Float32(init_phi_B), T_total_bins), fill(Float32(init_phi_S), T_total_bins))`.

Update the line 243 comment header to no longer claim Φ=1.0.

## Verification

1. Default (no flag) reproduces `Φ̄_B=1.000 Φ̄_S=1.000` at stride 1.
2. `--init-phi-B 0.3 --init-phi-S 0.3` → log shows `initial Φ ...: B=0.3 S=0.3`, stride 1 reports `Φ̄_B=0.300 Φ̄_S=0.300`, `Phi_B_per_bin_mpc[1:stride_bins] == 0.3`, `trajectory_baseline` differs from default-flag run.
3. TensorBoard `phi/mean_B == phi/baseline_B == 0.3` at step 1.
4. `--init-phi-B -0.5` errors cleanly before GPU work.
5. Asymmetric `--init-phi-B 0.5 --init-phi-S 0.1` honoured independently.

## Out of scope

- Open-loop closed-loop initial plan override (controller plans full horizon up front).
- Controller's internal `Phi_default = 1.0` (different concept).
- Time-varying initial plan (ramp Φ over warmup).
- Asymmetric baseline vs initial plan (both move together with this scope).
