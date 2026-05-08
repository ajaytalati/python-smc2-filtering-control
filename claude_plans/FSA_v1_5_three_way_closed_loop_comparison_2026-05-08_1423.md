# Plan: FSA v1.5 — three-way closed-loop run testing (Julia vs Python+JAX vs constant baseline, T=28d)

> Archived from plan mode: 2026-05-08 14:23.
>
> Replaces the prior FSA v1.5 LEAN4-port plan (now closed; landed in `version_1_5_LEAN/`).

## Context

The v1.5 model is now triangulated across Lean4, Julia, and Python+JAX at the **per-call math level** — 121/121 diff-test cases green at 1e-6. So drift, diffusion, EM step, plant_step, params_v15_to_v1, reflect_unit, apply_prior, obs_log_weight all agree bit-for-bit.

But the user reports the Julia closed-loop bench appears to *underperform*: filter + controller in isolation look reasonable (open-loop tests passed), but the closed-loop chained behaviour does not. Two competing hypotheses:

- **H1**: an **orchestration bug** in Julia's closed-loop loop (filter → posterior mean → controller → schedule → plant → repeat) — i.e. the Julia code wires something wrong that the model-level diff test cannot detect.
- **H2**: Julia is **GPU-underutilised** — the controller side has the right algorithm but is starved of compute compared to Python+JAX, which (per CLAUDE.md) saturates the RTX-5090 at v2's production settings. Same algorithm, less computational power, worse output.

These are very different fixes (code bug vs config bump) and the right diagnostic is a side-by-side closed-loop run with matched configurations + GPU telemetry.

**This is fundamentally a three-way comparison**:
1. Julia closed-loop MPC
2. Python+JAX closed-loop MPC
3. **Constant Φ=1.0 baseline** (no MPC — open-loop reference)

The load-bearing acceptance test is **"does each closed-loop MPC beat the constant baseline?"** — by design, both should. The baseline is the gate; the cross-stack comparison is the diagnostic that tells us *why* if one stack fails the gate.

About Lean4 in this round: Lean4 has no SMC² engine, no controller, and no GPU support — per the lean4-first charter §7 it's CPU-only and out of scope for closed-loop. Lean4's role as the model-math oracle is already discharged by the diff test. So **Lean4 does not participate in this comparison**; both other stacks have already been verified to call the same math.

## Goal

Produce a single artefact directory `outputs/v15_julia_vs_python_T28d/` containing, for both MPC stacks (Julia and Python+JAX):

1. The bench's standard outputs (state-traces PNG, param-traces PNG, manifest, trajectory data) — already includes the Φ=1.0 baseline trajectory inside.
2. A per-stride telemetry CSV (wall time, n_temp_filter, n_temp_ctrl, applied Φ).
3. A GPU telemetry CSV (utilisation, memory, power) sampled at 1 Hz during the run.
4. A **three-way comparison PNG** overlaying Julia MPC, Python MPC, and the constant baseline on every panel + a markdown findings report that explicitly answers (a) does each MPC beat baseline, and (b) if Julia doesn't, is it H1 or H2.

T = **28 days** (user's choice). Single seed for the first pass. The plant trajectories will *not* be bit-identical between stacks (RNGs differ — JAX vs StableRNG vs hash-based subkeys); the comparison is statistical (mean A, F-violation, posterior coverage, wall time, GPU saturation), not bit-equal.

**The baseline data is free**: both benches *already* compute a constant-Φ=1.0 plant rollout (it's the grey dashed line in their existing 4-panel state-traces PNGs), and its trajectory is saved into `trajectory.npz` (key `trajectory_baseline`) and `data.jld2` (key `trajectory_baseline`). No new code is needed to produce baseline data — only to *surface* it as a third explicit axis in the comparison plots and tables. We can also compare the two stacks' baselines as a cheap sanity check that the plant SDEs agree statistically.

## Why we can't just run them today

Four blockers from the Phase-1 explore:

1. **CLI parity gap (Python)**: Python's `bench_smc_full_mpc_fsa_v15.py` has **hard-coded** controller settings (`num_mcmc=8, hmc_step_size=0.2, hmc_num_leapfrog=16, beta_max_target_nats=8.0, max_temp_steps=25, max_lambda_inc=0.20`) that Julia exposes via CLI. Without parity, "matched config" is impossible.
2. **Default mismatch**: Python defaults `ctrl-n-smc=128 / ctrl-n-inner=32`; Julia defaults `ctrl-n-smc=256 / ctrl-n-inner=64`. Same code path, different work.
3. **No GPU telemetry** in either bench. Diagnosing H2 requires `nvidia-smi` snapshots during the run.
4. **No JAX-side microbenchmark profiler.** Julia already has `version_2_Julia/tools/profile_gpu.jl` which times the heavy filter PF + controller cost kernels in isolation and reports effective TFLOPS vs the 5090's 104.8 fp32 peak, plus host-sync counts and GPU-mem footprint. The Python+JAX side has no equivalent — without it, we can't tell whether a per-call kernel is slow because of the algorithm or because of XLA-compilation / host-sync overhead.

## The plan — seven phases

### Phase A: CLI parity on the Python bench

File: `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py`.

Expose the same controller knobs Julia already exposes:

| New flag | Default (matches Julia) |
|---|---|
| `--ctrl-num-mcmc` | 8 |
| `--ctrl-hmc-step` | 0.2 |
| `--ctrl-hmc-leap` | 16 |
| `--ctrl-target-nats` | 8.0 |
| `--ctrl-max-levels` | 25 |
| `--ctrl-max-lambda-inc` | 0.20 |
| `--ctrl-sigma-prior` | 1.5 |

Wire them into the existing `SMCControlConfig(...)` build at `bench_smc_full_mpc_fsa_v15.py:128-142`. Bump Python's defaults for `--ctrl-n-smc 256 --ctrl-n-inner 64` so a no-flag invocation matches Julia.

Estimated effort: ~10 min mechanical edit.

### Phase A1: JAX microbenchmark profiler (mirror of `profile_gpu.jl`)

New file: `version_1_5_Python_JAX/tools/profile_gpu.py`. Direct semantic port of `version_2_Julia/tools/profile_gpu.jl` — same surface, same metrics, same RTX-5090 reference (104.8 TFLOPS fp32 peak).

Two profile modes (`--filter`, `--controller`, `--both`); both use the v1.5 model files from `models/fsa_high_res/`:

**Filter target** (mirrors `profile_filter`):
  - Build the GK-DPF v3-lite log-density with the same factory the bench uses: `smc2fc.filtering.gk_dpf_v3_lite.make_gk_dpf_v3_lite_log_density_compileonce(model=HIGH_RES_FSA_V15_ESTIMATION, n_particles=K_per_chain, …)`.
  - Generate a synthetic 1-day window of obs; bind it via `jax.tree_util.Partial` (the production pattern from v2's bench).
  - Build a (M_max, n_params) θ matrix at the prior mean.
  - Warm up: one call + `jax.block_until_ready(out)` to force JIT compilation.
  - Time N=5 calls with `block_until_ready` after each — the JAX equivalent of `CUDA.synchronize()`.
  - Report: median time per call, threads-in-flight (M_max × K_per_chain), approximate FLOPs/call (same hand-count as Julia: T_steps × ~120 ops × n_threads), effective TFLOPS, % of 104.8 peak, GPU mem before/after via `nvidia-smi`, host syncs (count of `block_until_ready` calls — JAX makes them explicit, so the count is well-defined).

**Controller target** (mirrors `profile_controller`):
  - Build the same `cost_fn` the bench uses: `models.fsa_high_res.control.build_control_spec(...)` then take `spec.cost_fn`.
  - vmap over the M_max θ rows: `cost_batch = jax.jit(jax.vmap(spec.cost_fn))`.
  - Same warm-up + N=3 timed calls + `block_until_ready` pattern.
  - Same metrics (n_threads = M_max × n_inner, etc.).

**Optional bonus (free since JAX has it native)**: under `--trace`, wrap the timed loop in `jax.profiler.trace("./jax_trace")` so the user can open the dump in TensorBoard's Profile tab if they want kernel-level breakdown later. Off by default — adds ~30 s and a 100 MB trace file.

CLI mirrors Julia 1:1: `python tools/profile_gpu.py [--filter|--controller|--both] [--verbose] [--trace]`.

Estimated effort: ~30 min (mechanical port; the only non-trivial bit is the synthetic obs construction, which the v1.5 bench already does).

### Phase B: Per-stride telemetry CSV in both benches

Both benches log wall time + `n_temp` per stride to stdout but don't persist it. Add a `per_stride.csv` write at end-of-run with columns:

```
stride, t_wall_s, n_temp_filter, n_temp_ctrl, daily_phi, A_mean_so_far, B_end, F_end, A_end
```

In Python: append to a list inside the per-stride loop, dump as CSV via `csv.writer` after the bench completes.

In Julia: extend the existing `acc.per_stride_log` named-tuple to include the new columns and write via `CSV.write` (CSV.jl is already a transitive dep) — or just plain `open(...; write=true)` with manual formatting if CSV.jl isn't available.

Estimated effort: ~10 min per side.

### Phase C: GPU telemetry harness + paired-run launcher

A small bash launcher `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh`. The flow:

1. **Microbenchmark profilers first** (cheap, ~30 s each):
   - `julia --project=. tools/profile_gpu.jl --both > julia_profile.log` (existing file, run from `version_2_Julia/` per its expectations — TBD if a v1.5 copy is needed).
   - `python tools/profile_gpu.py --both > python_profile.log` (Phase A1).
   These give "what does the kernel do *per call* on this GPU, in isolation" — the H2 microbenchmark.

2. **Closed-loop bench, Julia side** (the macro test):
   - Start `nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1` in the background, redirecting to `gpu_telemetry.csv`.
   - Run `julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl --T-days 28 --seed 42 …` in the foreground.
   - Kill the nvidia-smi process; move its CSV into the Julia run's output dir.

3. **Closed-loop bench, Python side** — same pattern.

Both closed-loop runs land in a shared parent dir `outputs/v15_julia_vs_python_T28d/{julia,python}/`. Both microbenchmark logs land at the same level: `outputs/v15_julia_vs_python_T28d/{julia_profile.log, python_profile.log}`.

Estimated effort: ~20 min.

### Phase D: Comparison + diagnostic plot

A new Python script `tools/compare_v15_julia_vs_python.py` (under whichever version dir has both `numpy` and `h5py` — Python+JAX dir is the natural home; JLD2 files are HDF5 so `h5py` reads them directly). Inputs: the parent dir from Phase C. Loads:

- Python's `trajectory.npz` + `manifest.json`
- Julia's `data.jld2` + `manifest.json`
- Both `gpu_telemetry.csv`s
- Both `per_stride.csv`s

Produces a single 6-panel three-way diagnostic PNG `comparison.png`:

| Panel | Content (every line a different colour) |
|---|---|
| (0, 0) | A trajectory overlaid: **Python MPC blue, Julia MPC red, baseline grey solid** |
| (0, 1) | Applied daily Φ overlaid: **Python MPC blue, Julia MPC red, baseline horizontal Φ=1.0 grey** |
| (1, 0) | Posterior median for `B_inf` over windows + truth (both stacks; baseline doesn't have a posterior) |
| (1, 1) | Posterior median for `mu_B` over windows + truth (both stacks) |
| (2, 0) | GPU SM utilisation over wall time (both stacks; baseline is plant-only so trivially low) |
| (2, 1) | Per-stride wall time bar (both stacks) |

Plus a stdout summary table with the **baseline as the gate** and **microbenchmark profiler numbers** alongside the live macro telemetry:

```
                                  Baseline (Φ=1.0)   Python+JAX MPC   Julia MPC
mean A (last 7 days)              0.150              0.234            0.187
  improvement vs baseline         —                  +56%             +25%   ← Julia smaller
F-violation rate (frac)           0.000              0.012            0.034
posterior MSE to truth (final)    —                  0.018            0.041   ← Julia worse
─────────────────── macro (during the closed-loop run) ───────────────────
total wall time (min)             ~0 (plant-only)    38.4             71.2
mean GPU utilisation (%)          —                  87.3             34.5   ← H2 macro evidence
peak GPU memory (GB)              —                  14.2             6.8
─────────────────── micro (from profile_gpu, isolated kernel timing) ────────
filter kernel time / call (ms)    —                  18.4             21.7
  effective TFLOPS                —                  72.3             14.6   ← H2 micro evidence
  utilisation vs 5090 peak (%)    —                  69%              14%
controller cost / call (ms)       —                  142              198
  effective TFLOPS                —                  61.0             21.4
host syncs per filter call        —                  1                1
host syncs per ctrl call          —                  1                1
```

The "improvement vs baseline" row is the load-bearing one. **Both MPCs should be positive and meaningful** (rough threshold: ≥ +20%); if either is negative or near zero, that stack failed the gate.

Estimated effort: ~30 min.

### Phase E: Run + interpret

Launch the launcher. Expected wall time on the RTX-5090 with default-tier settings (N_smc=32, K_pf=200, ctrl-n-smc=256, ctrl-n-inner=64): Python ~30–45 min, Julia ~30–60 min (more if H2 is true). Total wall time including both runs: ~1.5–2 h.

Decision rules from the diagnostic table — **the baseline gate runs first**, then the cross-stack comparison:

- **Both MPCs comfortably beat baseline (≥+20% mean A) AND track each other (≤10% relative gap)** → both work; H1+H2 both rejected; the closed loop is fine and the original concern was likely a single bad seed or visual artefact. Move on.
- **Both MPCs beat baseline by similar margins AND Julia GPU << Python GPU** → both work algorithmically; H1 rejected; H2 separately confirmed as a *performance* concern (Julia is slower/under-utilised but still correct). Optional fix: bump Julia's particle counts.
- **Python beats baseline (≥+20%) but Julia ≈ baseline or worse** → **H1 supported**: Python disproves "the model itself doesn't work", so Julia's underperformance is in the orchestration. Drill into per-stride posterior MSE-to-truth; whichever stride first diverges points at the offending boundary (filter→controller hand-off, controller→plant hand-off, or replan-K cadence).
- **Python beats baseline AND Julia ≈ baseline AND Julia GPU << Python GPU** → both H1 and H2 partially true. Fix H2 first (cheap config bump) and re-run; if Julia then matches Python, H1 was a phantom and only H2 was real.
- **Neither MPC beats baseline** → unexpected; either both have the same orchestration bug or the model itself is mis-tuned for this scenario. The diff test rules out math divergence, so the more likely cause is a shared bench-level setting (e.g. wrong plant init, wrong horizon for the controller's planning window). Pause and re-design.

### Phase F: Findings report

A short markdown `outputs/v15_julia_vs_python_T28d/FINDINGS.md`:

- Configurations used (so the run is reproducible)
- The diagnostic table and the comparison.png
- **Baseline-gate verdict**: did Python beat baseline? did Julia beat baseline? by how much?
- **Cross-stack verdict**: H1 / H2 / both / neither
- If H1: at which stride did the divergence first appear, and which orchestration boundary is implicated
- If H2: which Julia config to bump, and the projected wall-time impact

## What to actually edit / create

- `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py` — Phase A (CLI parity) + Phase B (per-stride CSV)
- `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` — Phase B (per-stride CSV)
- `version_1_5_Python_JAX/tools/profile_gpu.py` — Phase A1 JAX profiler (NEW; mirrors `version_2_Julia/tools/profile_gpu.jl`)
- `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh` — Phase C launcher (NEW; runs profilers then closed-loop benches with `nvidia-smi` telemetry)
- `version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py` — Phase D comparison (NEW; consumes both bench outputs + both profile logs + both `nvidia-smi` CSVs)
- `outputs/v15_julia_vs_python_T28d/FINDINGS.md` — Phase F report (NEW, after run)

**Note**: the existing `version_2_Julia/tools/profile_gpu.jl` is hard-coded for v2's surface (`FSAGPUTargetBatched`, v2's 30-param prior config, `WINDOW_BINS=96`). For v1.5 we either (a) port it to a new `version_1_5_Julia/tools/profile_gpu.jl` with the v1.5 `FSAGPUTarget` (10 params, BINS_PER_DAY=24 default), or (b) reuse it as-is and accept that the *Julia* microbenchmark uses v2's surface. Recommendation: **(a)** — a small port — so both microbenchmarks run on v1.5's actual model and produce comparable numbers.

## Existing utilities to reuse

- `version_2_Julia/tools/profile_gpu.jl` — the **template** for the JAX profiler (Phase A1) and for v1.5's own Julia profiler. Same surface (kernel-time / threads / FLOPs / TFLOPS / utilisation%-vs-104.8-peak / GPU-mem / host-syncs), same RTX-5090 reference number. Direct semantic port.
- `version_2_Julia/tools/compare_to_python.jl:1-90` — pattern for loading both formats, but the v1.5 schema differs (10 LogNormal params vs v2's 30 mixed); the *shape* of the comparison code transfers but the keys + param list don't. **Adapt, don't copy.**
- `version_2_Julia/tools/plot_from_python_data.jl` — same situation; structure of "load Python npz from Julia" useful but not directly reusable.
- The existing `_plot_param_traces` helper (Python, in `bench_smc_full_mpc_fsa_v15.py`) and `_plot_param_traces_v15` helper (Julia) — reusable as one of the comparison panels.
- CLAUDE.md `XLA_PYTHON_CLIENT_PREALLOCATE=false` rule — Python bench already has it (line 26); confirm Julia's CUDA setup doesn't pre-allocate either.
- `jax.profiler.trace(...)` (built-in) — optional `--trace` mode in the JAX profiler; produces a TensorBoard-readable kernel breakdown. Off by default to keep the run fast.

## Verification

End-to-end works when:

1. Both benches accept identical CLI flags for the controller side.
2. `tools/launchers/run_v15_T28d_compare.sh` produces `outputs/v15_julia_vs_python_T28d/{python,julia}/{trajectory.*, manifest.json, gpu_telemetry.csv, per_stride.csv}` with no errors.
3. `compare_v15_julia_vs_python.py outputs/v15_julia_vs_python_T28d/` produces a `comparison.png` and prints the diagnostic table.
4. The `FINDINGS.md` says one of {H1, H2, both, neither} with cited evidence from the table + plot.

## Risks / things to flag now, not after

- **JAX RNG vs Julia StableRNG**: trajectories will not be bit-identical. The comparison is statistical (mean A, posterior shape) not bit-equal. This is correct — the diff test handles bit-equal verification at the model level.
- **Wall-time budget**: ~2 h on the RTX-5090 for one matched-config pass. If H2 is heavily true, Julia may take 2× longer; budget 4 h end-to-end including analysis.
- **`ctrl-n-inner` semantics may differ**: Julia's `ctrl-n-inner=64` and Python's `ctrl-n-inner=32` may not mean the *same* thing internally. Verify the symbol is wired identically before declaring config-matched. If not, the comparison may need `--ctrl-n-inner 32` on both as the safer choice.
- **Plot legibility at T=28d**: 28 strides × 2 stacks on a single line plot may be busy. Consider per-day mean overlays rather than per-bin.

## Effort estimate (AI-paced)

| Phase | Effort |
|---|---|
| A: CLI parity | ~10 min |
| A1: JAX profiler (mirror profile_gpu.jl) | ~30 min |
| B: per-stride CSV (both sides) | ~20 min |
| C: launcher (profilers + closed-loop benches + nvidia-smi) | ~20 min |
| D: comparison script (consumes profile logs + bench outputs + nvidia-smi CSVs) | ~30 min |
| E: actual runs (mostly waiting) | ~2 h wall-clock; ~10 min agent attention |
| F: findings report | ~15 min after results land |
| **Total agent time** | **~2 h coding + ~10 min while runs execute** |
| **Wall-clock end-to-end** | **~3–4 h** |

## What this plan does NOT do

- Fix the bug (if H1 is supported). That's a follow-up plan once we know *where* the bug lives.
- Bump Julia's settings to saturate the 5090 unilaterally. We test the *current* defaults first; bumping is the H2 fix to apply *after* the diagnostic, not before.
- Test multiple seeds. Single seed is enough to disambiguate H1 vs H2 at the qualitative level the user asked for. Multi-seed is a follow-up if results are borderline.
- Involve Lean4. Closed-loop is out of scope for the Lean4 port (no SMC², no GPU). Lean4's role as the model-math oracle is already done via the existing diff test.
