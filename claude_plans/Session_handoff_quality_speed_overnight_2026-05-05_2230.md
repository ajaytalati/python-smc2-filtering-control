# Session handoff — quality+speed overnight (2026-05-05 22:30)

> If you're a fresh agent reading this cold, this doc tells you what's
> running, what's done, what's pending, and what to do next. Read it
> end-to-end before touching anything.

## Goal

Optimise the v5 controller's `soft_fast` variant for both **quality**
(basin-overlay path matches the strict `soft` baseline) and **speed**
(GPU saturation on the RTX 5090, and shorter wall-clock).

## Where we are right now

Run 06 (strict `soft`, healthy, T=14d) finished at 21:52 — 99.8 min,
basin path tightly clustered in the healthy island, all metric gates
PASS except the post-hoc violation rate (which is a known evaluator
quirk). The plot lives at
`outputs/fsa_v5/experiments/run06_stage2_ctrl_soft_healthy_T14d_sat/basin_overlay.png`.

Run 09 (`soft_fast` with the FULL Gemini optimisation stack: fp32 cost
+ relaxed bisection + sub-sampled bins + trimmed HMC) finished in
**16.4 min** — 6× faster — and all headline metrics (mean A, ∫A dt,
violation rate) matched Run 06 to <0.5%. **But the basin-overlay path
is visibly worse**: applied Φ range went from [0.15, 0.49] (Run 06)
to [0.12, 1.07] (Run 09), with two daily-mean replans landing in the
collapsed (Φ_B, Φ_S) regime at (~1.07, ~0.28) and (~0.21, ~0.51).
The plant SDE is robust enough to recover so the integrals come out
the same, but the controller schedule is meaningfully noisier.

**Hypothesis under test:** the HMC config trim (`num_mcmc_steps`
10→5, `hmc_num_leapfrog` 16→8, `n_smc` 256→128) cut the controller's
exploration budget by ~16× and that's where the noisy proposals come
from. The cost-fn optimisations (fp32, relaxed bisection,
sub-sampled bins) are mathematical and should be safe.

**Run 11 in flight** (PID 1782342, started 22:16, ~30–40 min target):
soft_fast cost-fn + FULL HMC config. Tests the hypothesis. If basin
overlay matches Run 06, the cost-fn optimisations are clean and the
HMC trim was the lone culprit.

## What's running

| PID | When started | Run tag | Status |
|---|---|---|---|
| 1782342 | 22:16 | `stage2_soft_fast_healthy_T14_full_hmc` (Run 11) | In flight, soft_fast + full HMC. Diagnostic. |
| (queued) | when 1782342 exits | the overnight launcher | Waits for 1782342 then branches based on Run 11 verdict |

Logs live under `/tmp/stage23_sweep/`:
* Run 11 stdout: `/tmp/stage23_sweep/s2_soft_fast_healthy_full_hmc.log`
* Master sweep log: `/tmp/stage23_sweep/quality_speed_overnight_*.log`
* State file: `/tmp/stage23_sweep/quality_speed_state.json` (machine-readable
  current-run + verdict tracker)

## The autonomous launcher

`version_3/tools/launchers/run_quality_speed_overnight.sh` is the
unattended driver. It:

1. Polls PID 1782342 every 5 min until it exits.
2. Reads `applied_phi_max` from Run 11's manifest.
3. **Verdict branch:**
   * If `applied_phi_max ≤ 0.70` → "Run 11 quality OK, HMC trim was
     the bug". Production config = `soft_fast` cost-fn + full HMC.
     Runs the rest of Stage 2 sweep (sedentary, overtrained) +
     Stage 3 healthy. Each ~30–40 min Stage 2, ~1–2 h Stage 3.
   * If `applied_phi_max > 0.70` → "still bad". Runs Ablation A
     (`--bin-stride 1`, no chance-check sub-sampling) to test
     whether sub-sampling is contributing.
4. Records every step to the state file so the next agent can
   read it cold.

To start the launcher when ready:

```bash
cd /home/ajay/Repos/python-smc2-filtering-control
nohup bash version_3/tools/launchers/run_quality_speed_overnight.sh \
    > /tmp/stage23_sweep/quality_speed_overnight_master.log 2>&1 &
```

(Already started? Check: `ps -ef | grep run_quality_speed_overnight`.)

## Bench-driver edits already applied (commits pending)

* `version_3/tools/bench_controller_only_fsa_v5.py` -- `soft_fast`
  branch in `_build_spec_for_cost_variant` no longer overrides HMC
  config. Now uses `mcmc=10`, `leapfrog=16`, `n_smc=256` (the same
  as `soft`). Only the cost-fn-side optimisations (fp32, relaxed
  bisection, sub-sampled bins) remain in `soft_fast`. Print message
  updated to reflect the actual config.
* `version_3/tools/bench_smc_full_mpc_fsa_v5.py` -- same revert.

NOT yet committed because we want to wait for the Run 11 verdict; if
the revert doesn't fix the regression we may need to revert further.

## When Run 11 finishes — what to do

1. Generate the basin overlay plot:
   ```bash
   cd version_3
   PYTHONPATH=.:.. JAX_PLATFORMS=cpu python tools/plot_basin_overlay.py \
       outputs/fsa_v5/experiments/run11_stage2_soft_fast_healthy_T14_full_hmc/
   ```
2. Compare to Run 06's plot:
   * `applied_phi_max` should be < 0.70 (Run 06 was 0.49, Run 09 was 1.07).
   * Path should hug the healthy island.
3. Write a Run 11 entry in `outputs/fsa_v5/CHANGELOG.md` via:
   ```bash
   PYTHONPATH=.:.. python tools/summarize_run.py \
       outputs/fsa_v5/experiments/run11_*/ >> /tmp/run11_summary.md
   ```
4. **If quality recovered:**
   * Commit the bench-driver reverts (already on disk, just `git add` + commit).
   * Confirm the launcher is running the rest of the sweep (ls
     `/tmp/stage23_sweep/`).
5. **If quality still bad** (max Φ > 0.70):
   * Launcher will auto-run Ablation A (`--bin-stride 1`).
   * After that, ablate bisection (revert `_jax_find_A_sep_fast` to
     use `n_grid=64, n_bisect=40` like the strict `_jax_find_A_sep`).

## Principled profiling -- queued, not started

`version_3/tools/profile_cost_fn.py` is written but NOT yet executed
(GPU is busy with Run 11). When the GPU frees up between runs, the
profiler can:

* Time `cost_fn(theta)` forward + `jax.grad` calls under `jax.profiler.start_trace`.
* Project per-replan wall-clock from per-call wall-clock × leapfrog count.
* Dump TensorBoard-compatible traces to
  `version_3/outputs/fsa_v5/profiles/<timestamp>_<cfg>/`.

Run with:
```bash
cd version_3
PYTHONPATH=.:.. python tools/profile_cost_fn.py \
    --cost soft_fast --n-smc 256 --n-truth-particles 1 --n-iters 20
```

Sweep configurations to compare empirically (each ~1–2 min once
JIT-compiled):
* `soft_fast n_smc=256, n_truth_particles=1` (current production candidate)
* `soft_fast n_smc=256, n_truth_particles=32` (parallelism via redundant truth particles)
* `soft_fast n_smc=512, n_truth_particles=1` (more outer SMC particles)
* `soft     n_smc=256, n_truth_particles=1` (the strict baseline for reference)

The expected story: with truth_particles=1 and a small inner-vmap,
each leapfrog step is bottlenecked by the sequential SDE rollout's
1344 RK4 steps and the GPU is parallelism-starved. Bumping
truth_particles or n_smc gives more parallel work. Whichever is
fastest at acceptable quality is the production default.

## Open question for the user

The user authorised overnight GPU time. They want active monitoring +
intervention. **The launcher above runs unattended, so the GPU is
used overnight regardless.** If the agent (this conversation) hits
its token limit before all results are interpreted, the next agent
should:

1. Read this handoff doc.
2. Read `/tmp/stage23_sweep/quality_speed_state.json` to see what's
   been tried.
3. Read the latest CHANGELOG entries.
4. Read each completed run's `manifest.json` and `basin_overlay.png`.
5. Resume from the next pending step.

There is no way to auto-wake the agent at a fixed time -- the
ScheduleWakeup tool only works in `/loop` mode, which this isn't.
The user must start a new session manually.
