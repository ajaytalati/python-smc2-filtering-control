
I want you to create a new Latex document which contains the following snippets from other documents,

Sections 1 , 2, 3 from

- /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/docs/v15_horizon_sweep_writeup.pdf
- /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/docs/v15_horizon_sweep_writeup.tex

Sections 4 From 

- /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.pdf
- /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/docs/julia_vs_python_v15_writeup.tex

---

Then I want you to write a section (or 2 or 3) which basically explains the technical pipeline from Lean4 defintions to Julia code, and then checking using test_lean_diff_v15

This needs to be documented as its currently very vague and NOT reproducible.

This pipeline from Latex math description / specification to the Julia code here

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res

which used by the SMCFC framework needs to be MUCH tigher and algorithmic /documented and evidence based - YOU should be able to trace the data/information flow.

If you can add a flow diagram to the docs that would be a good idea

Currently there are two VITAL GPU based files which are not verified by LEAN4

- /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_control.jl
- /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/gpu_pf.jl

---

Recall the start of this chat,

test_lean_diff_v15.jl

what does this code do?

Read [test_lean_diff_v15.jl](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/index.html?id=85f612d4-c20f-4ebd-9584-7d333880d10b&parentId=8&origin=0229bde6-12a2-4f16-a1d8-3c0649bd4c1d&swVersion=5&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&session=d28e9413-455e-48b0-9cb9-f268a142ac63#)

This is a Julia ↔ Lean4 differential test for the FSA v1.5 model. It checks that the Julia implementation and the Lean4 implementation produce numerically identical outputs for every public function in the model.

**How it works:**

- Spawns the Lean4 binary (`fsa_v15_cli` from `version_1_5_LEAN/`) as a long-lived subprocess and talks to it line-by-line over stdin/stdout using JSON.
- For each function, Julia generates inputs (random states, random params, pre-drawn noise), sends them to Lean, runs the same call locally, then asserts `|Δ| < 1e-6` for single-step ops or `< 1e-4` for integrated ops.

**Functions covered** ([test_lean_diff_v15.jl:15-18](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/diff_test/test_lean_diff_v15.jl#L15-L18)):

- `reflect_unit` — boundary reflection on [0,1]
- `apply_prior` — log-normal / normal prior transforms
- `params_v15_to_v1` — the v1.5 → v1 NamedTuple conversion (one of three `@match` sites)
- `drift` and `diffusion_state_dep` — SDE coefficients
- `em_step_substepped` — 4-substep EM integrator with one Wiener increment
- `plant_step` — single EM step + observation sampler
- `obs_log_weight` — per-particle log-likelihood

**Why pre-drawn noise:** both sides accept the noise arrays as inputs ([test_lean_diff_v15.jl:201-202](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/diff_test/test_lean_diff_v15.jl#L201-L202)), so RNG differences between Julia and Lean don't muddy the comparison — the test is purely about numerical agreement of the deterministic transforms.

In short: it's a cross-language equivalence harness, used to lock the Lean4 port to bit-for-bit parity with the Julia reference implementation.

where are the julia files it compares with (uses)

From the `include` and `using` statements at the top of the test ([test_lean_diff_v15.jl:26-34](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/diff_test/test_lean_diff_v15.jl#L26-L34)):

The Julia side lives in [version_1_5_Julia/models/fsa_high_res/](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/):

- [_dynamics.jl](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/_dynamics.jl) — provides `Dynamics.drift` and `Dynamics.diffusion_state_dep` (the SDE coefficients tested in the `drift` and `diffusion_state_dep` testsets).
- [simulation.jl](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/simulation.jl) — provides `Simulation.DEFAULT_PARAMS`, `Simulation.params_v15_to_v1_nt`, `Simulation.INIT_STATE` (used to build the v1-style NamedTuple, the v1.5 payload, and the obs-noise payload).
- [_plant.jl](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/_plant.jl) — included so the `plant_step` testset has the same single-EM-step plant logic available; the test re-derives the plant step inline rather than calling a function from this file directly.
- [estimation.jl](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/estimation.jl) — provides `Estimation.obs_log_weight` (the per-particle log-likelihood compared in the `obs_log_weight_one` testset).

Let me verify the directory exists.

Bash List FSA high-res model files

IN

ls /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/

OUT

FSAHighRes.jl
_dynamics.jl
_plant.jl
estimation.jl
gpu_control.jl
gpu_pf.jl
simulation.jl

Confirmed — all four files referenced by the test exist in [version_1_5_Julia/models/fsa_high_res/](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/models/fsa_high_res/).

The Lean4 side it diffs against is the binary at `version_1_5_LEAN/.lake/build/bin/fsa_v15_cli` ([test_lean_diff_v15.jl:39-40](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/diff_test/test_lean_diff_v15.jl#L39-L40)), built from the Lean sources in [version_1_5_LEAN/](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/).

where are the LEAN$ files it uses

Bash List Lean4 project root

IN

ls /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/

OUT

Fsa
Fsa.lean
LaTex_docs
Main.lean
README.md
lake-manifest.json
lakefile.lean
lean-toolchain

Bash List Fsa subdirectory

IN

ls /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/

OUT

V15
V15.lean

Bash List V15 module

IN

ls /home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V15/

OUT

Adapters.lean
Dynamics.lean
Estimation.lean
Plant.lean
Truth.lean
Types.lean

Read [Main.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/index.html?id=85f612d4-c20f-4ebd-9584-7d333880d10b&parentId=8&origin=0229bde6-12a2-4f16-a1d8-3c0649bd4c1d&swVersion=5&extensionId=Anthropic.claude-code&platform=electron&vscode-resource-base-authority=vscode-resource.vscode-cdn.net&parentOrigin=vscode-file%3A%2F%2Fvscode-app&session=d28e9413-455e-48b0-9cb9-f268a142ac63#)

The Lean4 source tree lives at [version_1_5_LEAN/](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/). The binary the test calls (`fsa_v15_cli`) is compiled from these:

**Entry point / CLI bridge**

- [Main.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Main.lean) — reads one JSON object per line on stdin, dispatches by `fn` tag (`drift`, `diffusion`, `emStep`, `paramsV15ToV1`, `reflectUnit`, `plantStep`, `obsLogWeight`, `applyPrior`), writes one JSON object per line on stdout. This is the protocol the Julia test speaks.

**Library root**

- [Fsa.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa.lean) — top-level module; pulls in `Fsa.V15`.
- [Fsa/V15.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15.lean) — V15 module aggregator.

**V15 modules** in [Fsa/V15/](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/) — these contain the actual implementations the diff test exercises:

- [Types.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Types.lean) — `PlantState`, `ParamsV15`, `ParamsV1`, etc.
- [Dynamics.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Dynamics.lean) — `drift`, `diffusion_state_dep`, `em_step_substepped`, `reflect_unit`.
- [Adapters.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Adapters.lean) — `paramsV15ToV1` (the 14-field conversion).
- [Plant.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Plant.lean) — `plantStep`.
- [Estimation.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Estimation.lean) — `obsLogWeight`, `applyPrior`.
- [Truth.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/Fsa/V15/Truth.lean) — truth/default parameter values.

**Build files**

- [lakefile.lean](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/lakefile.lean) — Lake build config that produces `fsa_v15_cli`.
- [lean-toolchain](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_LEAN/lean-toolchain) — pins the Lean4 compiler version.

Build with `cd version_1_5_LEAN && lake build` ([test_lean_diff_v15.jl:8-9](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/diff_test/test_lean_diff_v15.jl#L8-L9)) before running the diff test.



