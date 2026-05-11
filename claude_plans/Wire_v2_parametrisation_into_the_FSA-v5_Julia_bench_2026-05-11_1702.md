# Wire v2 parametrisation into the FSA-v5 Julia bench (controllability verification)

> Archived from plan mode: 2026-05-11 17:02.

## Context

A LaTeX document at [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.pdf](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.pdf) states **four constructive controllability theorems** for the deterministic FSA-v5 ODE under a re-parametrised "v2" parameter set. The companion script [reference_numerics.jl](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/reference_numerics.jl) is fully self-contained Julia (no project deps) and prints the expected `A(100)` values for each theorem under deterministic RK4 (dt=0.05 d).

**What this plan delivers**: wire the v2 parametrisation into the existing SMC²-MPC bench so the four theorems can be numerically corroborated under the bench's fp32 stochastic-EM plant. Per the PDF §10.1, this is "estimated effort < 100 LOC, pure engineering, does not touch any mathematical content."

**Why now**: the canonical FSA-v5 bench at T=100 d (2026-05-11) failed to escape the sedentary basin even with a sophisticated SMC²-MPC controller. The PDF argues this is a *model* issue — the canonical healthy attractor near Φ ≈ (0.30, 0.30) is unreachable from `SEDENTARY_INIT` in T=100 d. The v2 re-parametrisation shifts the attractor to Φ ≈ (1.06, 0.78) and widens its basin substantially. The four theorems verify this constructively. Bench corroboration closes the loop between the proofs and the production code.

**Scope is narrow**: no Lean4 work; no refactoring; no edits outside the three Julia source files named below plus four new launcher scripts, with one extra small touch to `bench_postproc.jl` (see open question Q4 — I'd like to confirm before touching it).

---

## What the launchers actually test

Per the **revised handover §4.6.1**: the four Lean4 theorems are already proved analytically and verified numerically by `reference_numerics.jl` (5 s on the CPU, deterministic, definitive). The Julia bench launchers are *not* a redundant numerical re-verification of those theorems. The launchers test the **full SMC²-MPC closed-loop stack** (plant + stochastic SDE + observations + particle filter + SMC²-MPC controller) under v2 + the simplified cost from four physiologically distinct starting scenarios.

| Launcher          | Starting scenario                              | Controller test                                                      |
|-------------------|------------------------------------------------|----------------------------------------------------------------------|
| `run_thm_6_1_*`   | sedentary state, baseline Φ in pathological corner (0.1, 0.1) | Does the controller diagnose the corner and steer Φ toward the healthy island? |
| `run_thm_6_2_*`   | sedentary state, **default initial Φ** (=0.3, 0.3) | Does the controller find a healthy-island Φ from a generic sedentary start under the simplified cost? Does Φ̄ track the FIM witness (0.86, 1.24)? |
| `run_thm_6_3_*`   | trained state, baseline Φ in over-training corner (2.5, 2.5) | Does the controller recognise over-training and pull Φ back to the island? |
| `run_thm_6_4_*`   | trained state, baseline Φ at maintenance (1.0, 1.0) | Does the controller hold maintenance Φ ≈ (1, 1) or drift?            |

The initial Φ via `--init-phi-B / --init-phi-S` only sets the state at $t=0$ before the first replan — once the controller fires it is free to choose any Φ. It is NOT a constraint, just the scenario seed.

**Bench mode: ALL FOUR use full closed-loop (`--open-loop false`).** No `--open-loop true` anywhere. Handover §4.6 is now explicit.

**Cost**: simplified `J = -A_acc + lam_island · island_acc` for all four; `lam_island = 1.0` for all four (the v2 island-pull is what gives HMC a sensible escape gradient in 6.1 / a hold gradient in 6.2 / a pull-back gradient in 6.3 / a stability gradient in 6.4). All other cost coefficients zeroed per §4.5.

**Expected wall time:** ~4 × Ultra-Fast tier wall ≈ 4 × 50–60 min on RTX 5090.

(PDF §1.2 working assumption 2 — "full observability, filter removed from consideration" — is a scope statement about what the *mathematical* theorems claim, not a bench-mode directive. I conflated them earlier; handover §4.6.1 is now explicit about the distinction.)

---

## The 8 v2 parameter overrides (PDF §2.3)

All other parameters inherit from canonical `TRUTH_PARAMS_V5` unchanged — including `tau_F = 7/(1 + A_TYP) ≈ 6.36 d`, `mu_dec_B = mu_dec_S = 0.10`, `n_dec = 4`, the `K^0_*`, `mu_K`, `tau_K`, all `sigma_*`, `mu_0`, `mu_B`, `mu_S`, `eta`, `epsilon_A*`, `lambda_A`, `A_TYP`, `F_TYP`. **Reference: PDF page 5, second table ("All other parameters are inherited from canonical TRUTH_PARAMS_V5 unchanged").**

| Param     | Canonical                      | v2          | Note                                                |
|-----------|--------------------------------|-------------|-----------------------------------------------------|
| `tau_B`   | 42 d                           | **21 d**    | halved; B\* invariant under (τ,κ)→(τ/2, 2κ)         |
| `kappa_B` | 0.012·(1 + 0.40·A_TYP) = 0.01248 | **0.02496** | doubled, centred form                              |
| `tau_S`   | 60 d                           | **30 d**    | halved                                              |
| `kappa_S` | 0.008·(1 + 0.20·A_TYP) = 0.00816 | **0.01632** | doubled, centred form                              |
| `B_dec`   | 0.07                           | **0.25**    | raised 3.6×; shifts island toward (1,1)             |
| `S_dec`   | 0.07                           | **0.25**    | raised 3.6×                                         |
| `mu_F`    | 0.10 + 2·F_TYP·0.40 = 0.26     | **0.030**   | reward-side fatigue penalty reduced 88%             |
| `mu_FF`   | 0.40                           | **0.020**   | reduced 95%                                         |

---

## Files I'll edit

1. [version_1_5_Julia/models/fsa_v5/simulation_v5.jl](version_1_5_Julia/models/fsa_v5/simulation_v5.jl) — +2 constants, +2 exports.
2. [version_1_5_Julia/tools_v5/bench/bench_args.jl](version_1_5_Julia/tools_v5/bench/bench_args.jl) — +1 default-dict entry.
3. [version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl) — extend `using ...:` import + add dispatch in `main()` + extend `init-preset` dispatch.
4. (Conditional on Q4) [version_1_5_Julia/tools_v5/bench/bench_postproc.jl](version_1_5_Julia/tools_v5/bench/bench_postproc.jl) — replace the two `FULL_PARAMS_V5` reads with reads from `ctx.full_params` / passed-in arg.

## Files I'll create

5. [version_1_5_Julia/tools_v5/launchers/run_thm_6_1_sedentary_basin_collapse_v2.sh](version_1_5_Julia/tools_v5/launchers/run_thm_6_1_sedentary_basin_collapse_v2.sh)
6. [version_1_5_Julia/tools_v5/launchers/run_thm_6_2_constructive_escape_v2.sh](version_1_5_Julia/tools_v5/launchers/run_thm_6_2_constructive_escape_v2.sh)
7. [version_1_5_Julia/tools_v5/launchers/run_thm_6_3_overtraining_collapse_v2.sh](version_1_5_Julia/tools_v5/launchers/run_thm_6_3_overtraining_collapse_v2.sh)
8. [version_1_5_Julia/tools_v5/launchers/run_thm_6_4_maintenance_v2.sh](version_1_5_Julia/tools_v5/launchers/run_thm_6_4_maintenance_v2.sh)

## Files I'll only read (no edits)

- [version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl](version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl) — confirmed lines 182-212 (island cost block) and 283-289 (cost line) already work correctly under v2 because they consume `p_*` fields unpacked from the controller's params dict, which under `--truth-preset v2` will be the v2 values.

---

## Concrete edits

### Edit 1 — `simulation_v5.jl`

Add after the existing `TRUTH_PARAMS_V5` `let` block (currently ends [simulation_v5.jl:135](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L135)):

```julia
"""
    TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32}

v2 re-parametrisation of TRUTH_PARAMS_V5 for the controllability_v2_proofs
theorems. Eight numerical entries differ from canonical; all other entries
(including tau_F, mu_dec_*, n_dec, K^0_*, mu_K, tau_K, sigma_*, mu_0, mu_B,
mu_S, eta, epsilon_A*, lambda_A) are inherited unchanged. See
controllability_v2_proofs.pdf §2.3.
"""
const TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32} = let
    p = copy(TRUTH_PARAMS_V5)
    p[:tau_B]   = 21.0f0
    p[:kappa_B] = 0.012f0 * (1.0f0 + 0.40f0 * A_TYP) * 2.0f0   # 0.02496f0
    p[:tau_S]   = 30.0f0
    p[:kappa_S] = 0.008f0 * (1.0f0 + 0.20f0 * A_TYP) * 2.0f0   # 0.01632f0
    p[:B_dec]   = 0.25f0
    p[:S_dec]   = 0.25f0
    p[:mu_F]    = 0.030f0
    p[:mu_FF]   = 0.020f0
    p
end
```

Add after the existing `TRAINED_ATHLETE_INIT` definition (currently ends [simulation_v5.jl:223](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L223)):

```julia
"""
    TRAINED_ATHLETE_INIT_V2::NamedTuple

Slow-manifold equilibrium of FSA-v5 under TRUTH_PARAMS_V5_RECOMMENDED_V2 at
the v2 island centre Φ = (1.06, 0.78), upper-stable autonomic root A* = 1.238.
Not on the slow manifold under canonical TRUTH_PARAMS_V5. See PDF §2.4.
"""
const TRAINED_ATHLETE_INIT_V2 = (
    B   = 0.7993f0,
    S   = 0.4688f0,
    F   = 0.7925f0,
    A   = 1.2384f0,
    KFB = 0.1414f0,
    KFS = 0.1322f0,
)
```

Update the module's `export` list (lines [28](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L28) and [30](version_1_5_Julia/models/fsa_v5/simulation_v5.jl#L30)) to include `TRUTH_PARAMS_V5_RECOMMENDED_V2` and `TRAINED_ATHLETE_INIT_V2`.

### Edit 2 — `bench_args.jl`

Add immediately after the `"init-preset"` entry ([bench_args.jl:82](version_1_5_Julia/tools_v5/bench/bench_args.jl#L82)), in the same comment-block style as the surrounding flags:

```julia
# Truth-side parameter preset. Selects the dict fed into the plant rollout,
# the open-loop controller's initial plan, and the closed-loop ctx.full_params.
#   "canonical" — TRUTH_PARAMS_V5 (DEFAULT; canonical island centre ~ (0.30, 0.30)).
#   "v2"        — TRUTH_PARAMS_V5_RECOMMENDED_V2 (v2 island centre ~ (1.06, 0.78),
#                  widened basin; for controllability_v2_proofs.pdf theorems).
# Under "v2", use --init-preset TRAINED_ATHLETE_INIT_V2 (the canonical
# trained init is not on the v2 slow manifold).
"truth-preset"    => "canonical",
```

### Edit 3 — `bench_smc_full_mpc_fsa_v5_gpu.jl`

**(a)** Extend the import at lines [53-56](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L53-L56) to add the two new names:

```julia
using .FSAv5.SimulationV5: BINS_PER_DAY, DT_BIN_DAYS,
                           TRUTH_PARAMS_V5, TRUTH_PARAMS_V5_RECOMMENDED_V2,
                           DEFAULT_OBS_PARAMS_V5,
                           FROZEN_PARAMS_V5, SEDENTARY_INIT,
                           TRAINED_ATHLETE_INIT, TRAINED_ATHLETE_INIT_V2
```

**(b)** Keep the top-level `const FULL_PARAMS_V5 = merge(TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5)` at [line 74](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L74) **unchanged** (it remains the canonical default, kept stable for `bench_postproc.jl`'s external read — see Q4).

**(c)** Inside `main()`, immediately after the `is_open_loop` block ([line 124](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L124)), add:

```julia
# ── Truth-side parameter preset selection ──
truth_preset_str = string(args["truth-preset"])
truth_params_v5_local = if truth_preset_str == "canonical"
    TRUTH_PARAMS_V5
elseif truth_preset_str == "v2"
    TRUTH_PARAMS_V5_RECOMMENDED_V2
else
    error("--truth-preset must be \"canonical\" or \"v2\", got: \"$truth_preset_str\"")
end
full_params_v5 = merge(truth_params_v5_local, DEFAULT_OBS_PARAMS_V5)
@info "truth preset: $truth_preset_str"
```

**(d)** Extend the `init-preset` dispatch ([lines 155-163](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L155-L163)) to a 3-way dispatch and add consistency checks:

```julia
init_preset = string(args["init-preset"])
init_state_fn, init_nt = if init_preset == "TRAINED_ATHLETE_INIT"
    (init_plant_state_trained, TRAINED_ATHLETE_INIT)
elseif init_preset == "TRAINED_ATHLETE_INIT_V2"
    (init_plant_state_trained, TRAINED_ATHLETE_INIT_V2)
elseif init_preset == "SEDENTARY_INIT"
    (init_plant_state_sedentary, SEDENTARY_INIT)
else
    error("--init-preset must be one of " *
          "\"TRAINED_ATHLETE_INIT\", \"TRAINED_ATHLETE_INIT_V2\", " *
          "\"SEDENTARY_INIT\"; got: \"$init_preset\"")
end
if init_preset == "TRAINED_ATHLETE_INIT_V2" && truth_preset_str != "v2"
    error("--init-preset TRAINED_ATHLETE_INIT_V2 requires --truth-preset v2 " *
          "(the v2 trained init is not on the canonical slow manifold).")
end
if init_preset == "TRAINED_ATHLETE_INIT" && truth_preset_str == "v2"
    @warn "Canonical TRAINED_ATHLETE_INIT used with --truth-preset v2 — " *
          "not on v2 slow manifold; consider TRAINED_ATHLETE_INIT_V2."
end
```

**(e)** Replace `FULL_PARAMS_V5` with `full_params_v5` at the three consumer sites inside `main()`:
- [line 280](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L280) — baseline `plant_rollout_v5`.
- [line 292](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L292) — open-loop initial `controller_plan_v5`.
- [line 337](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L337) — `ctx.full_params` (this is what flows into the closed-loop foldl and is read by `bench_loop.jl` / `bench_filter.jl`).

**(f)** Update the comment at [line 286](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L286) from `TRUTH_PARAMS_V5` to `$truth_preset_str truth params`.

### Edit 4 — `bench_postproc.jl` (CONDITIONAL on Q4)

If user picks Q4 option (b), make `bench_postproc.jl` read truth params from the run's `ctx` / `final` (already plumbed via `ctx.full_params` at [line 337](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl#L337)) rather than from the global `FULL_PARAMS_V5` const at [bench_postproc.jl:114](version_1_5_Julia/tools_v5/bench/bench_postproc.jl#L114) and [:225](version_1_5_Julia/tools_v5/bench/bench_postproc.jl#L225). About 5 LOC.

### Edits 5-8 — four launcher scripts

All four are slim wrappers in the style of [run_v5_ultra_fast.sh](version_1_5_Julia/tools_v5/launchers/run_v5_ultra_fast.sh) — same nvidia-smi sampler, same Ultra-Fast tier, with a forced `T=100` and the per-theorem flags.

Common cost-zeroing flags (all four launchers): `--ctrl-lam-phi 0 --ctrl-lam-f 0 --ctrl-lam-chance 0 --ctrl-lam-chance-b 0 --ctrl-lam-chance-s 0 --ctrl-lam-b 0 --ctrl-lam-s 0`.

All four use `--open-loop false` (closed-loop). `init-phi-B`/`init-phi-S` seed the closed-loop initial-plan strides 1..(K-1) before the first replan; the controller then takes over. They also seed the baseline rollout (the constant-Φ reference trajectory). For 6.2 the bench's default `init-phi-B = init-phi-S = 0.3` is left in place (no override) — the controller must DISCOVER the FIM witness, not start near it.

| Launcher                                     | `--init-preset`           | `Φ_B init`     | `Φ_S init`     | `lam_island` |
|----------------------------------------------|---------------------------|----------------|----------------|--------------|
| `run_thm_6_1_sedentary_basin_collapse_v2.sh` | `SEDENTARY_INIT`          | 0.1            | 0.1            | 1.0          |
| `run_thm_6_2_constructive_escape_v2.sh`      | `SEDENTARY_INIT`          | default (0.3)  | default (0.3)  | 1.0          |
| `run_thm_6_3_overtraining_collapse_v2.sh`    | `TRAINED_ATHLETE_INIT_V2` | 2.5            | 2.5            | 1.0          |
| `run_thm_6_4_maintenance_v2.sh`              | `TRAINED_ATHLETE_INIT_V2` | 1.0            | 1.0            | 1.0          |

Per-launcher reads from `data.jld2`:
- **Closed-loop A(100)** — the headline number per §5 criteria.
- **Closed-loop Φ̄_B, Φ̄_S** time-averages — used to evaluate whether the controller steered correctly.
- **Baseline A(100)** (constant-Φ reference at the seeded `(init_phi_B, init_phi_S)`) — diagnostic context: shows what would have happened under the passive scenario without the controller. For 6.1 and 6.3 the baseline should reproduce the pathological collapse near A=0; for 6.4 the baseline should sit near A*≈1.24. The contrast with the closed-loop result is what shows the controller's value-add.

---

## Verification

In order. Stop at first failure and surface it.

1. **REPL parameter check** (no GPU needed):
   ```
   cd version_1_5_Julia && julia --project=. -e \
     'include("models/fsa_v5/FSAv5.jl"); using .FSAv5.SimulationV5; \
      for k in (:tau_B,:kappa_B,:tau_S,:kappa_S,:B_dec,:S_dec,:mu_F,:mu_FF); \
        println(k," = ",TRUTH_PARAMS_V5_RECOMMENDED_V2[k]); end'
   ```
   Expect: `21.0f0 0.02496f0 30.0f0 0.01632f0 0.25f0 0.25f0 0.03f0 0.02f0`. Cross-check the `kappa_*` to 4 d.p. against PDF §2.3 table.

2. **Reference numerics reproduction** (no GPU):
   ```
   cd version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/ && \
     julia reference_numerics.jl
   ```
   Expect (PDF §A): μ(SEDENTARY_INIT)=−0.1405; μ̄(0;(1,1))=+0.1188; Thm 6.1 A(100)=0.0000; Thm 6.2 A(100)=1.1605; Thm 6.3 A(100)=0.0000; Thm 6.4 A(100)=1.2460.

3. **Smoke test each launcher at T=2 d** (override the launcher's `--T-days 100` with a trailing `--T-days 2`). Expect a clean run in ~1-2 min, `data.jld2` written, no crashes.

4. **Production runs at T=100 d**. Run each launcher. Expect ~50-60 min wall on RTX 5090 at the Ultra-Fast tier. Stream `nvidia_smi.csv` in the background per existing convention.

5. **Controller-behaviour report**: `outputs/bench_runs/v2_controller_behaviour_<timestamp>.md` (per revised handover §5):
   - One section per launcher.
   - Each section:
     - Launcher command invoked.
     - **Closed-loop A(100)** and **closed-loop Φ̄_B, Φ̄_S** over [0, 100] d.
     - **Baseline A(100)** (constant-Φ reference at the seeded init Φ) for diagnostic contrast.
     - Run wall time.
     - **Controller verdict** line: ✓ worked / ✗ failed (with hypothesis) / ? unclear (with what would clarify), per the §5 table:

| Launcher | Pass criterion (handover §5) |
|---|---|
| 6.1 | A(100) > 0.5 AND Φ̄ *not* near (0.1, 0.1) — controller actively steered out of corner |
| 6.2 | A(100) > 0.5 AND Φ̄ near the v2 island, e.g. Φ_B ∈ [0.6, 1.3], Φ_S ∈ [0.6, 1.3]; bonus check whether Φ̄ tracks the FIM witness (0.86, 1.24) |
| 6.3 | A(100) > 0.5 AND Φ̄ *not* near (2.5, 2.5) — controller pulled back from over-training |
| 6.4 | A(100) ∈ [0.9, 1.5] AND Φ̄ near (1, 1) — controller maintained |

   - One-paragraph commentary on surprising behaviour (e.g., A overshoot of A\*, Φ oscillation).
   - Absolute path to the run dir for traceability.

   Failures are meaningful negative findings about the controller (cost weighting, FIM-island gradient strength, controller exploration) — to be reported honestly with a guess at the failure mode, then surfaced to the user before any tuning.

---

## Resolved decisions (2026-05-11)

All five open items have been answered by the user. Recorded here for traceability during execution.

- **Q1 — init-preset name.** Use `TRAINED_ATHLETE_INIT_V2` (uppercase `_V2`, matches existing all-caps Julia const style).
- **Q2 — Bench mode and what's being tested.** Resolved by handover revision §4.6 / §4.6.1 (2026-05-11): all four launchers use `--open-loop false`; pass/fail is checked against the **closed-loop** trajectory (controller behaviour); the baseline constant-Φ trajectory is reported as diagnostic context only. The launchers are NOT re-verifying the Lean4 theorems (reference_numerics.jl already does that) — they test whether the SMC²-MPC controller stack works under v2 + simplified cost from each starting scenario. My earlier readings were wrong on this point and have been corrected.
- **Q3 — Report format.** Markdown.
- **Q4 — `bench_postproc.jl` edit.** Yes — add it to the editable list (5 LOC) so the run manifest's `truth_params` reflects `--truth-preset v2` instead of always showing canonical. Thread `full_params` through `ctx` (already plumbed at line 337) and read it inside `bench_postproc.jl` at the two call sites (lines 114, 225) instead of the top-level `FULL_PARAMS_V5` const.
- **Q5 — Filter side rewiring.** No. `FSAv5GPUTarget` consumes only NaN-fallback init values; the filter learns params via SMC² from observations. Handover §4.3 step 6(b) is a misstatement.
