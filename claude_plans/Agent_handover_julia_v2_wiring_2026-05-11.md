# Agent handover — Julia wiring for FSA-v5 v2 controllability verification

Date: 2026-05-11
Author of this handover: Claude (Anthropic) under direction of Ajay Talati
Audience: a fresh agent with no prior conversation context, tasked with the Julia-side wiring of the FSA-v5 v2 re-parametrisation.

---

## 1. Mission, in one paragraph

A LaTeX document, `controllability_v2_proofs.tex`, has been produced that states four constructive controllability theorems for FSA-v5 under a new "v2" parameter set. **Your job** is to wire the v2 parametrisation into the Julia code so the existing SMC²-MPC bench can run under v2 and numerically corroborate the four theorems. You have **no Lean4 responsibilities**. You have **no refactoring or general code-cleanup responsibilities**. Your only allowed code edits are the minimal additions described in §4 below.

---

## 2. Required reading before you touch any code

Read these in order. They are listed by priority; you should not skip any.

1. **The streamlined v2 proofs document** — the canonical reference for the parametrisation, initial conditions, witnesses, and theorems:
   - [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.pdf](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/controllability_v2_proofs.pdf)
   - Read end-to-end. Pay particular attention to §2 (Model recall under `RECOMMENDED_V2`), §6 (the four theorems), §10 (TODO and further work).

2. **The standalone Julia reference numerics** — your ground-truth for what the deterministic ODE produces under v2:
   - [version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/reference_numerics.jl](version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/reference_numerics.jl)
   - Run it once (`julia reference_numerics.jl`); the output is your ground truth for the four cases. Reproduce these numbers before proceeding.

3. **The existing canonical truth-parameter definition** — the file you will be editing:
   - [version_1_5_Julia/models/fsa_v5/simulation_v5.jl](version_1_5_Julia/models/fsa_v5/simulation_v5.jl) lines 80–145 (TRUTH_PARAMS_V4, TRUTH_PARAMS_V5, SEDENTARY_INIT, TRAINED_ATHLETE_INIT). Read this entire file.

4. **The bench CLI arg structure** — where the new flag will be added:
   - [version_1_5_Julia/tools_v5/bench/bench_args.jl](version_1_5_Julia/tools_v5/bench/bench_args.jl). Skim the `_parse_args` function (around lines 14–520) to understand the flag taxonomy and the parser. There are existing flags like `--ctrl-lam-island`, `--ctrl-a-thr`, `--init-preset`, etc., which give you the style template.

5. **The bench driver** — where the new flag is consumed:
   - [version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl). Skim lines 200–260 (the cfg dict construction) and the params-passing into the bench plant + controller.

6. **The GPU controller cost kernel** — for verifying the island cost block (§4.3):
   - [version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl](version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl) lines 182–212 (the island gradient surrogate).

7. **The project CLAUDE.md** at the repo root: read for project-wide conventions. Key items relevant to you: `comfyenv` is the conda env name; `JAX_ENABLE_X64=True` for outer-loop work; bench drivers prepend a specific JAX/CUDA env setup; the `version_1_5_Julia/` is Julia not Python.

---

## 3. Hard constraints

These are non-negotiable. Violations require explicit user approval.

### 3.1 Coding constraints (only apply if you are allowed to edit code)

- **Functional programming only.** No global mutable state, no in-place mutation of shared data structures, no setter-style methods. Use `let`, `const`, immutable structs, and pure functions. Where Julia idiom needs mutation (e.g., GPU kernel buffers), confine it tightly and document why.
- **GPU code is Float32 only.** No Float64 inside GPU kernels. Mass-matrix Cholesky, log-weights, parameter posteriors etc.\ stay fp64 on CPU (per the project's `CLAUDE.md` GPU-dtype convention).
- **Avoid CPU computation if possible.** Compute everything on GPU when feasible. If you find CPU code that could be moved to GPU, flag it but do not move it without explicit user approval — that is scope creep.
- **Documentation is Google Professional Software Engineering standards.** No chatty LLM comments. No "Here's the trick: …", no "Note: this is important because…". Every comment is one of: a Julia docstring on a public function (one-line summary + multi-line behavior description + Args + Returns), an inline `#` comment on a non-obvious algorithmic step, or `#!` / `@assert` for invariants. If a comment doesn't pull its weight, delete it.

### 3.2 Scope constraints

- **You must not edit any code outside of the four files named in §4 below**, unless you find a literal bug that blocks your work. In that case, ask the user before editing. ``Refactoring opportunities'' you notice are out of scope.
- **You have no Lean4 responsibilities.** Do not touch `version_1_5_LEAN/Fsa/V5/*.lean` or `version_1_5_LEAN/Main_v5.lean`. The Lean4 proof mechanisation is a separate stream of work.
- **You must not introduce new dependencies** (Julia packages). The bench runs in a fixed environment; new dependencies break reproducibility.

---

## 4. Concrete deliverables

In order. Do them in this order; do not skip ahead.

### 4.1 Define `TRUTH_PARAMS_V5_RECOMMENDED_V2` in `simulation_v5.jl`

In [version_1_5_Julia/models/fsa_v5/simulation_v5.jl](version_1_5_Julia/models/fsa_v5/simulation_v5.jl), after the existing `TRUTH_PARAMS_V5` definition (around line 135), add a new const:

```julia
"""
    TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32}

v2 re-parametrisation of TRUTH_PARAMS_V5, recommended for controllability
analysis from SEDENTARY_INIT. Differs from canonical TRUTH_PARAMS_V5 in 8
numerical entries; all other entries are inherited unchanged.

The 8 v2 overrides:
  - tau_B  : 42.0 -> 21.0       (halved relaxation timescale)
  - kappa_B: 0.01248 -> 0.02496 (doubled centred gain — preserves B* equilibrium)
  - tau_S  : 60.0 -> 30.0
  - kappa_S: 0.00816 -> 0.01632
  - B_dec  : 0.07 -> 0.25        (raised deconditioning threshold)
  - S_dec  : 0.07 -> 0.25
  - mu_F   : 0.26 -> 0.030       (reduced fatigue-linear coefficient)
  - mu_FF  : 0.40 -> 0.020       (reduced fatigue-quadratic coefficient)

Documented in version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/
controllability_v2_proofs.pdf, Section 2.3.
"""
const TRUTH_PARAMS_V5_RECOMMENDED_V2::Dict{Symbol, Float32} = let
    p = copy(TRUTH_PARAMS_V5)
    p[:tau_B]   = 21.0f0
    p[:kappa_B] = 0.012f0 * (1.0f0 + 0.40f0 * A_TYP) * 2.0f0
    p[:tau_S]   = 30.0f0
    p[:kappa_S] = 0.008f0 * (1.0f0 + 0.20f0 * A_TYP) * 2.0f0
    p[:B_dec]   = 0.25f0
    p[:S_dec]   = 0.25f0
    p[:mu_F]    = 0.030f0
    p[:mu_FF]   = 0.020f0
    p
end
```

Verify after adding: the print-out of this const at REPL time should match the values listed in §2.3 of `controllability_v2_proofs.pdf` to 4 decimal places.

### 4.2 Add `TRAINED_ATHLETE_INIT_V2` in `simulation_v5.jl`

Right after the existing `TRAINED_ATHLETE_INIT` definition (around line 220), add a parallel const **without modifying** the existing one (the canonical is still used by tests that depend on canonical parameters):

```julia
"""
    TRAINED_ATHLETE_INIT_V2::NamedTuple

Slow-manifold equilibrium of the FSA-v5 dynamics under
TRUTH_PARAMS_V5_RECOMMENDED_V2 at the v2 island centre Φ = (1.06, 0.78),
with the upper-stable autonomic root A* = 1.238.

Use this as the initial state for trained-athlete trajectories evaluated
under the v2 parametrisation. Do not use it under canonical TRUTH_PARAMS_V5
— it is not a slow-manifold state there.

Documented in version_1_5_LEAN/LaTex_docs/sedentary_basin_controllability_analysis/
controllability_v2_proofs.pdf, Section 2.4.
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

Update the existing `:init-preset` documentation comment in `bench_args.jl` to mention this new preset name (`"TRAINED_ATHLETE_INIT_V2"`).

### 4.3 Add `--truth-preset v2` CLI flag

In [version_1_5_Julia/tools_v5/bench/bench_args.jl](version_1_5_Julia/tools_v5/bench/bench_args.jl):

1. Add a new default entry to the `defaults` dict (alongside the other I/O / model args, around line 80–100):
```julia
"truth-preset" => "canonical",
```

2. The legal values are `"canonical"` (selecting `TRUTH_PARAMS_V5`) and `"v2"` (selecting `TRUTH_PARAMS_V5_RECOMMENDED_V2`).

3. Document the flag with a comment block in the style of the other flag comments (`# Truth-side parameter preset selector. …`).

In [version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl](version_1_5_Julia/tools_v5/bench_smc_full_mpc_fsa_v5_gpu.jl):

4. After argument parsing, dispatch on `args["truth-preset"]` to select the right truth-params dict (functional style — no in-place mutation; use a `truth_params = args["truth-preset"] == "v2" ? TRUTH_PARAMS_V5_RECOMMENDED_V2 : TRUTH_PARAMS_V5` style assignment). The legal-value check should raise an error with a clear message if an unknown preset is passed.

5. **Also** dispatch on `truth-preset` to decide whether `--init-preset TRAINED_ATHLETE_INIT` resolves to the canonical `TRAINED_ATHLETE_INIT` or the new `TRAINED_ATHLETE_INIT_V2`. Recommended: under `truth-preset v2`, treat `--init-preset TRAINED_ATHLETE_INIT` as `TRAINED_ATHLETE_INIT_V2` automatically (warn at startup if the user explicitly passes `TRAINED_ATHLETE_INIT` under v2). Alternatively, accept `--init-preset TRAINED_ATHLETE_INIT_V2` as a separate value and require the user to be explicit. Pick whichever is cleaner; document your choice.

6. Threads the chosen `truth_params` through to: (a) the plant initialisation, (b) the filter `EstimationModel`, (c) the controller `FSAv5ControlGPUTarget`. There may be 2-3 call sites. Use the existing `params_v5` parameter pathway; do not introduce new globals.

### 4.4 Verify the island cost block in `gpu_control_v5.jl`

In [version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl](version_1_5_Julia/models/fsa_v5/gpu_control_v5.jl) lines 182–212 (the "Healthy-island gradient surrogate" block):

1. **Read the block** end-to-end. Confirm it computes the slow-manifold $\mu(A_t; \Phi_t)$ — i.e., it evaluates the bifurcation parameter with the *current* $A$, not $A = 0$ — and produces an `island_acc` increment of the form `(1/β_island) · softplus(β_island · (-mu_bar_isl)) · dt`.

2. **Cross-check against** the document §2.2 (Eq.~6) and §7.2 (Eq.~21). The kernel block should match these expressions. The sign in particular: `softplus(β · (-μ̄))` ≥ 0 always, large when μ̄ << 0 (collapsed), small when μ̄ > 0 (healthy). The final cost contribution is `+ lam_island · island_acc`, so the cost penalises Φ values that produce collapsed-region μ̄.

3. **Do not modify this block** unless you find a literal bug. The block was reviewed by the user already. Just confirm it's still correct under the v2 parametrisation — and the only way it could be wrong is if the v2 dynamics introduce a corner case (e.g., a clipping interaction) the original code didn't anticipate. If you find anything, ask the user before editing.

### 4.5 Verify all other controller cost terms can be switched off

The cost expression to use for verifying the four theorems numerically is:
```
J = -A_acc + lam_island * island_acc
```
i.e., **only** the negative A-reward and the island penalty. All other terms must be zeroed.

Audit the cost line at the end of `fsa_v5_cost_kernel!` (around line 248–256 in `gpu_control_v5.jl`). The full cost is:
```julia
cost_per_thread[i] = lam_Phi * effort_acc
                   - A_acc - lam_b * B_acc - lam_s * S_acc
                   + lam_F * barrier_acc
                   + lam_chance * chance_acc
                   + lam_chance_B * chance_B_acc
                   + lam_chance_S * chance_S_acc
                   + lam_island * island_acc
```

To use the simplified cost, the launcher must pass:
- `--ctrl-lam-phi 0`   (turn off effort term)
- `--ctrl-lam-b 0`     (default, but pass explicitly)
- `--ctrl-lam-s 0`     (default, but pass explicitly)
- `--ctrl-lam-f 0`     (default, but pass explicitly)
- `--ctrl-lam-chance 0`
- `--ctrl-lam-chance-b 0`
- `--ctrl-lam-chance-s 0`
- `--ctrl-lam-island 1.0` (or some positive value to tune)

Document this in the launcher script(s) you produce in §4.6.

### 4.6 Produce four launcher scripts

Under [version_1_5_Julia/tools_v5/launchers/](version_1_5_Julia/tools_v5/launchers/), produce four launchers. **All four use the FULL CLOSED-LOOP SMC²-MPC stack** (plant + observations + particle filter + controller). The initial Φ via `--init-phi-B / --init-phi-S` sets the *starting baseline scenario* only — once the controller's first replan fires, it is free to choose any Φ that minimises the (simplified) cost. **No `--open-loop` use anywhere.**

- `run_thm_6_1_sedentary_basin_collapse_v2.sh`:  
  `--truth-preset v2 --init-preset SEDENTARY_INIT --init-phi-B 0.1 --init-phi-S 0.1` + simplified cost from §4.5. T=100d.
- `run_thm_6_2_constructive_escape_v2.sh`:  
  `--truth-preset v2 --init-preset SEDENTARY_INIT` (default initial Φ) + simplified cost from §4.5. T=100d.
- `run_thm_6_3_overtraining_collapse_v2.sh`:  
  `--truth-preset v2 --init-preset TRAINED_ATHLETE_INIT_V2 --init-phi-B 2.5 --init-phi-S 2.5` + simplified cost from §4.5. T=100d.
- `run_thm_6_4_maintenance_v2.sh`:  
  `--truth-preset v2 --init-preset TRAINED_ATHLETE_INIT_V2 --init-phi-B 1.0 --init-phi-S 1.0` + simplified cost from §4.5. T=100d.

Style: model the launchers on the existing `version_1_5_Julia/tools_v5/launchers/run_v5_ultra_fast.sh` template (Ultra-Fast preset is fine for fast iteration). Place output in `outputs/bench_runs/$(timestamp)_thm_X_X_v2/` for each.

### 4.6.1 What each launcher actually tests (read carefully)

**The four LEAN4 theorems are proved analytically and verified numerically by `reference_numerics.jl`** — a 5-second deterministic computation on the CPU. The Julia bench launchers are NOT redundant numerical verifications of those theorems; the deterministic reference is already definitive.

**What the bench launchers actually test** is the FULL SMC²-MPC CLOSED-LOOP STACK (plant + stochastic SDE + observations + particle filter + SMC²-MPC controller) operating under the v2 parametrisation + simplified cost, from four physiologically distinct starting scenarios:

| Launcher | Starting scenario | What the bench tests |
|---|---|---|
| `run_thm_6_1_*` | sedentary state with baseline policy already in the pathological corner | Does the controller correctly diagnose the sedentary corner and steer Φ toward the healthy island? Or does it stay near the (0.1, 0.1) baseline and let A collapse? |
| `run_thm_6_2_*` | sedentary state with default initial Φ | Does the controller find a healthy-island Φ from the canonical SEDENTARY starting point under simplified cost? (Most direct controller test.) |
| `run_thm_6_3_*` | trained state with baseline policy in the over-training corner | Does the controller recognise over-training and pull Φ back to the island, or does the system collapse? |
| `run_thm_6_4_*` | trained state with the maintenance policy already correct | Does the controller hold the maintenance Φ ≈ (1, 1), or does it drift? |

The "constant baseline setting" (initial Φ via the CLI flags above) is what sets up the scenario; the controller then runs freely. *The initial Φ is not a constraint on the controller; it is just the state at $t = 0$ before the first replan.*

This means the launcher outcomes may differ from the deterministic theorem references (`reference_numerics.jl`) — and **that is expected and informative**. Examples:

- For 6.1, the deterministic reference says "passive Φ = (0.1, 0.1) gives A(100) = 0". The closed-loop bench may give A(100) > 0 if the controller successfully steers Φ off the (0.1, 0.1) corner. That is *the controller working well*, not a failure of the theorem.
- For 6.3, similarly, the controller may pull Φ back from (2.5, 2.5) before A collapses; A(100) > 0 is the controller doing its job.
- For 6.2 and 6.4, the deterministic reference predicts what a controller WOULD do under the FIM-witness or maintenance policy; the bench tests whether the controller actually finds those policies.

**Interpretation principle.** The launchers test the v2 controller, not the v2 theorems. The theorems are verified by `reference_numerics.jl`; the bench tests how the full stochastic stack behaves under v2.

---

## 5. Verification — what counts as "the controller worked"

Per §4.6.1, the launchers test the SMC²-MPC controller under v2 + the simplified cost from four physiologically distinct starting scenarios. **The theorems themselves are verified separately** by `reference_numerics.jl` (deterministic, definitive); the bench is testing controller behaviour.

The simplified cost `J = -A_acc + lam_island · island_acc` rewards (i) high time-integrated A and (ii) staying near the v2 island where $\bar\mu(0; \Phi) > 0$. Under this cost, *any* well-functioning controller should drive the system to the healthy attractor $A^* \approx 1.24$ regardless of starting scenario. So the pass criterion is essentially the same across all four:

| Launcher | Starting scenario | "Controller worked" criterion at T = 100 d |
|---|---|---|
| 6.1 | sedentary + baseline Φ=(0.1, 0.1) | `A(100) > 0.5` AND Φ-time-average is *not* near (0.1, 0.1) (i.e., controller actively steered out of the corner) |
| 6.2 | sedentary, default initial Φ        | `A(100) > 0.5` AND Φ-time-average is near the island (e.g., $\Phi_B \in [0.6, 1.3], \Phi_S \in [0.6, 1.3]$) |
| 6.3 | trained + baseline Φ=(2.5, 2.5)    | `A(100) > 0.5` AND Φ-time-average is *not* near (2.5, 2.5) (controller pulled back from over-training) |
| 6.4 | trained + baseline Φ=(1.0, 1.0)    | `A(100) ∈ [0.9, 1.5]` AND Φ-time-average near (1, 1) (controller maintained) |

If a launcher *fails* — A(100) collapses below 0.1 in 6.1 or 6.3, or the controller actively drifts away from the island in 6.4 — that is a meaningful negative finding about the controller (under v2 + simplified cost), not about the math. Report failures honestly with a guess at the failure mode (cost mis-weighting? FIM-island gradient too weak? controller exploration insufficient?) and ask the user before tuning.

For 6.2 specifically, also check whether the controller's Φ-time-average sits near the FIM-witness $(0.86, 1.24)$ from §7.3 of `controllability_v2_proofs.pdf`. If it does, the FIM-duality argument is empirically reinforced; if not, the controller has found a different (equally valid) escape strategy — worth flagging but not a failure.

Report results in a single markdown file: `outputs/bench_runs/v2_controller_behaviour_$(timestamp).md` with one section per launcher, each section containing:
- The launcher command invoked.
- A summary of the bench output: A(100), Φ-time-average over $[0, 100]$, run wall time.
- A "controller verdict" line per the table above: ✓ worked, ✗ failed (with hypothesis), or ? unclear (with what would clarify).
- A one-paragraph commentary noting any surprising behaviour (e.g., A overshooting $A^*$, Φ oscillating).

---

## 6. Things to be careful about

Issues I noticed while preparing this handover that you should not stumble on.

### 6.1 Float32 in `simulation_v5.jl`

The existing `TRUTH_PARAMS_V5` uses Float32 literals (`42.0f0`). When you write the v2 dict, every numerical entry must also be Float32 (`21.0f0`, not `21.0`). The `let` block with `copy(TRUTH_PARAMS_V5)` preserves type; explicit overrides like `p[:tau_B] = 21.0f0` must use the `f0` suffix.

### 6.2 Theorem 6.3 witness — Φ = (2.5, 2.5), NOT (2, 2)

The document explicitly uses Φ = (2.5, 2.5) for the over-training collapse, not (2, 2) as in the v1 (canonical) analysis. The reason is documented in Theorem 6.3's Remark in `controllability_v2_proofs.pdf`: under v2's faster (B, S) timescales, the autonomic-protective feedback `a_F(A)` keeps F lower at high A, so Φ = (2, 2) at T = 100 d gives A(100) ≈ 0.49 (not collapsed). Φ = (2.5, 2.5) gives the clean collapse. Use 2.5.

### 6.3 The island cost block depends on the SAME parameter dict

The kernel at lines 182–212 uses `p_mu_0, p_mu_B, p_mu_S, p_mu_F, p_mu_FF, p_mu_dec_B, p_mu_dec_S, p_B_dec, p_S_dec, p_n_dec`. These are all unpacked from the controller's `params` dict at kernel-launch time. Under `--truth-preset v2`, the controller's params dict must be `TRUTH_PARAMS_V5_RECOMMENDED_V2` (not `TRUTH_PARAMS_V5`), or the island cost will use v1-like values and the controller will optimise toward the wrong island. Check this in step §4.3 step 6 — the truth_params must flow through to the controller, not just the plant.

### 6.4 The `n_dec` parameter is NOT changed

v2 keeps `n_dec = 4`. Only `B_dec, S_dec, mu_dec_B, mu_dec_S, mu_F, mu_FF, tau_B, tau_S, kappa_B, kappa_S` change (10 values). Wait — `mu_dec_B, mu_dec_S` actually stay at 0.10 unchanged. So 8 values truly change. See §2.3 of the document for the canonical list.

### 6.5 Existing CLI flag parser is strict

`bench_args.jl`'s parser at line ~508 raises `error("Unknown flag: $a")` for any unrecognised flag. After adding `truth-preset` to the defaults dict, that flag becomes legal. But: if you misspell it anywhere downstream the parser will not catch it — it'll just default. Verify with a smoke run that `--truth-preset v2` actually changes behaviour.

### 6.6 `comfyenv` is the conda env

If you need to run Python tooling (the `generate_figures.py` reference is Python, not Julia), source the `comfyenv` first. Per the project CLAUDE.md.

### 6.7 The reference numerics script will SLIGHTLY mismatch the SMC²-MPC bench

The deterministic reference in `reference_numerics.jl` uses fp64 RK4 with `dt = 0.05`. The bench uses fp32 EM with `dt = 1/96 = 0.0104`. So the bench's A(T) will differ from the reference by a few percent. This is why the pass criterion in §5 is qualitative.

---

## 7. Out of scope

Things you should NOT do, even if they look tempting:

- Touch any Lean4 file. The Lean mechanisation is a separate stream.
- Refactor `gpu_control_v5.jl`. The kernel is performance-critical and was reviewed by the user.
- Add new controller cost terms beyond what's in §4.5.
- Modify `SEDENTARY_INIT` (it stays canonical).
- Change `n_dec`, `epsilon_AB`, `epsilon_AS`, `lambda_A`, or any parameter not listed in §4.1's 8 overrides.
- Touch `obs_v5.jl` (the observation model). v2 changes do not touch observations.
- Make the v2 the new default. The canonical `TRUTH_PARAMS_V5` remains the default; v2 is opt-in via `--truth-preset v2`.
- Add new dependencies (Julia packages) or modify `Project.toml`.

---

## 8. Recommended workflow

1. Read §2's required reading. Run `reference_numerics.jl`, confirm the 4 numbers match the document.
2. §4.1: add `TRUTH_PARAMS_V5_RECOMMENDED_V2` to `simulation_v5.jl`. At REPL, print it and verify the 8 overrides.
3. §4.2: add `TRAINED_ATHLETE_INIT_V2`. At REPL, verify it matches the document's §2.4 values.
4. §4.3: add the `--truth-preset` flag wiring. Test with a 2-day smoke run (`./run_v5_ultra_fast.sh 2 --truth-preset v2`).
5. §4.4: read the island cost block; confirm correctness against §7.2 of the document. Do not modify unless bug.
6. §4.5: audit the cost line; confirm all other cost terms can be zeroed.
7. §4.6: write the four launcher scripts. Smoke-test each at T=2 d (under 2 minutes per).
8. Run each launcher at T=100 d. Collect A(100) and Φ-time-average.
9. Write the verification report (§5 format).
10. Hand back to user with the report + the location of all artefacts.

---

## 9. Open questions to flag back to the user as you encounter them

You may discover unknowns. Flag, do not assume.

- **Stochastic noise calibration**: the v2 dynamics are 2× faster; whether the canonical `sigma_B, sigma_S, sigma_F, sigma_A, sigma_K` are still appropriate for v2 is genuinely open. If your bench runs show wild oscillations on A under v2, it may be a noise-calibration issue. Document and ask.
- **Controller hyperparameter tuning under v2**: the Ultra-Fast preset (`ctrl-n-smc 128`, etc.) was tuned for canonical parameters. If 6.2's verification produces a Φ time-average far from (0.86, 1.24), it may be a controller-tuning issue rather than a model issue. Document and ask before changing controller hyperparams.
- **The `--open-loop` flag**: if it exists in `bench_args.jl`, use it for theorems 6.1, 6.3, 6.4 (no controller needed). If it doesn't exist as a clean toggle, ask how to disable the controller for the passive-trajectory verification.

---

## 10. What success looks like

When you are done:

- `simulation_v5.jl` has two new constants: `TRUTH_PARAMS_V5_RECOMMENDED_V2` and `TRAINED_ATHLETE_INIT_V2`. Approximately +50 lines.
- `bench_args.jl` has one new flag: `truth-preset`. Approximately +20 lines.
- `bench_smc_full_mpc_fsa_v5_gpu.jl` has truth-params dispatch logic. Approximately +30 lines.
- `tools_v5/launchers/` has four new launcher scripts.
- `outputs/bench_runs/v2_theorem_verification_$(timestamp).md` reports the four theorems' empirical outcomes at T = 100 d.

Total expected code addition: $\sim 100$ LOC (matching the document's §10.1 estimate) plus the four launcher scripts.

No code anywhere in the codebase is *modified* in place except the small additions to `simulation_v5.jl`, `bench_args.jl`, `bench_smc_full_mpc_fsa_v5_gpu.jl`. The kernel at `gpu_control_v5.jl` is read-only.

---

## 11. Final user-decision items deferred to you

The user has approved this handover but you should expect to ask one or two clarifying questions during execution. Likely areas:

- The exact name/value of the new `--init-preset` enum (e.g., `TRAINED_ATHLETE_INIT_V2` vs `TRAINED_V2` vs just `V2`). User's preference unclear; ask before committing to a name.
- Whether to run the controller under theorem 6.2 with the FIM-selected witness HARDWIRED into the controller's initial plan, or to let the controller find Φ ≈ (0.86, 1.24) on its own via SMC²-MPC. The latter is the more interesting test (the controller should converge to that Φ if the cost shape is right); the former is the safer test (verifies the dynamics agree with the deterministic reference).
- Whether the verification report should be in markdown (default), JSON, or another format the user can pipe into a CI workflow.

When in doubt, ask. Better to pause for 30 seconds than to commit to an unwanted convention.
