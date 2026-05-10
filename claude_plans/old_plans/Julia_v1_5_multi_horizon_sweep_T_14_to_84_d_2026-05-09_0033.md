# Julia v1.5 multi-horizon sweep (T = 14 → 84 d)

> Archived from plan mode: 2026-05-09 00:33.

## Context

We need a clean sweep of the Julia v1.5 closed-loop SMC²-MPC bench at
six planning horizons — T = 14, 28, 42, 56, 70, 84 days — to study
how the controller's behaviour scales with horizon length on the
Julia side (the side we trust). All other config is the saturated
tier from
`compare_v15_julia_vs_python/docs/technical_guide_to_current_best_Julia_SMC2FC_config.pdf`,
which is now the bench's *default* — so the only flag that has to
change between runs is `--T-days`.

The work is:

1. **Quick T=14 run** (foreground, ~10 min) to confirm the launcher
   script is correct and the two PNGs (state traces + parameter
   traces) save into the expected place.
2. **Overnight chain T=28 → T=84** (sequential, single-GPU, ~3.5 h
   total estimated from the 19.5-min T=28 wall in the tech guide).
3. Per-horizon artefacts land in a single sweep root with a
   well-named subdirectory per horizon.

## Tools to use

The user has pointed to the canonical plotting + profiling tools at
`compare_v15_julia_vs_python/src/`:

* `plot_state_traces.jl` — 4-panel state-trajectory plot (B / F / A
  / applied Φ) consumed via `plot_state_traces(data; out_path=…)`.
* `plot_param_traces.jl` — multi-panel posterior parameter traces
  (5–95 % envelope + median + red-dashed truth line) consumed via
  `plot_param_traces(data; out_path=…, T_total_days=…, step_minutes=…,
  stride_bins=…)`.
* `profile_gpu_julia_v15.jl` — micro-profiler that times the filter
  + controller kernels in isolation, reports kernel wall, threads in
  flight, FLOPs/call, effective TFLOPS vs the RTX 5090 fp32 peak, GPU
  memory, and host-sync count. Modes: `--filter`, `--controller`,
  `--both`.

The bench's own auto-plot fallback (`bench_smc_full_mpc_fsa_gpu.jl`
lines 840-854) tries to load `version_2_Julia/tools/plot_state_traces.jl`
which is in the deprecated tree; it `@warn`s and continues on failure.
The launcher will call the canonical plotters at
`compare_v15_julia_vs_python/src/plot_*.jl` explicitly after each
horizon's bench finishes, so the sweep never depends on the bench's
internal auto-plot path.

The profiler is run ONCE at the start of the sweep — kernel walls
do not depend on T-days (same chains × inner-loop dimensions), only
n_steps and n_replans change, which the macro bench already records.

## What the Julia bench already gives us

Confirmed from `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl`:

* All saturated-tier flags are now defaults (verified at lines 39-155
  of the bench): `N-smc=512`, `K-per-chain=1000`, `num-mcmc=3`,
  `max-temp-levels=30`, `ctrl-n-smc=2048`, `ctrl-num-mcmc=16`,
  `ctrl-chees-max=512`, `ctrl-n-anchors=12`, `ctrl-max-levels=40`,
  `replan-K=2`, `seed=42`, `step-minutes=60`. Default `T-days=14`.
  So the only flag we vary is `--T-days`; we'll set `--seed 42`
  explicitly for reproducibility and `--output-dir` per horizon.

* Per run, the bench writes (lines 731-875):
  * `data.jld2` (state + obs + posterior cloud)
  * `manifest.json` (run config + replan history + total wall)
  * `experiment_run.md`
  * `per_stride.csv` (per-stride telemetry)
  * `v15_T<N>d_traces.png` (4-panel state + applied Φ)
  * `v15_T<N>d_param_traces.png` (10-panel parameter posteriors)

  We additionally tee stdout to a `bench.log` per horizon.

* No existing Julia horizon-sweep launcher
  (`version_1_5_Julia/tools/launchers/` does not exist). The pattern
  to mirror is `version_2_Python_JAX/tools/launchers/run_horizon.sh`
  (a single-T launcher), which we generalise to a multi-T loop.

## Estimated per-horizon walls (from tech guide T=28 ≈ 19.5 min)

Linear scaling in `n_replans ≈ T_days / 2` with the replan-K=2
default — taken as a guess and to be replaced by the measured wall
in the final SWEEP_MANIFEST. Numbers below are the budget I expect:

| T_days | wall (guess) |
|---|---|
| 14 | ~10 min |
| 28 | ~20 min (anchor — measured by Julia tech guide at 19.5 min) |
| 42 | ~30 min |
| 56 | ~40 min |
| 70 | ~50 min |
| 84 | ~60 min |
| **total** | **~3.5 h** |

These are guesses; the SWEEP_MANIFEST.md will record measured walls.

## Folder convention

```
compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/
├── SWEEP_MANIFEST.md          # rendered Markdown summary of horizons_results.csv
├── horizons_results.csv       # one row per horizon: full headline metrics table
├── microbench_profile.txt     # one-shot profile_gpu_julia_v15.jl --both output
├── T14d_seed42/
│   ├── data.jld2
│   ├── manifest.json
│   ├── experiment_run.md
│   ├── per_stride.csv         # per-stride wall, n_temp_filter/ctrl, daily_phi, A/B/F_end
│   ├── nvidia_smi.csv         # 1 Hz GPU util + memory samples for the run
│   ├── results.json           # extracted headline metrics for this horizon
│   ├── v15_T14d_traces.png    # canonical plot_state_traces.jl
│   ├── v15_T14d_param_traces.png  # canonical plot_param_traces.jl
│   └── bench.log              # tee'd stdout
├── T28d_seed42/
│   └── ...
├── T42d_seed42/
├── T56d_seed42/
├── T70d_seed42/
└── T84d_seed42/
```

* Sweep root has a YYYY-MM-DD stamp so a future re-run does not
  collide.
* Per-horizon dir is `T<N>d_seed<S>` — readable by humans, sortable
  by horizon, and includes the seed so re-running at a different
  seed coexists.

## Headline metrics extracted per horizon

After each bench finishes, we run a small Python helper that reads
`data.jld2` (via `h5py` — Julia writes JLD2-on-HDF5), `manifest.json`,
`per_stride.csv`, and `nvidia_smi.csv`, and writes:

* `<horizon>/results.json` — all headline metrics for that horizon.
* one row appended to `<sweep_root>/horizons_results.csv`.

Metrics in the row (one per horizon):

| field | source |
|---|---|
| `T_days` | sweep argument |
| `n_strides` | `manifest.json` |
| `n_replans` | `manifest.json` `replan_history` length |
| `total_wall_s` | `manifest.json` `total_elapsed_s` |
| `mean_stride_wall_s` | `per_stride.csv` mean of `t_wall_s` |
| `mean_filter_n_temp` | `per_stride.csv` mean of `n_temp_filter` (non-zero rows) |
| `mean_ctrl_n_temp` | `per_stride.csv` mean of `n_temp_ctrl` (non-zero rows) |
| `mean_gpu_util_pct` | `nvidia_smi.csv` mean of util column |
| `peak_vram_mib` | `nvidia_smi.csv` max of memory column |
| `pct_time_gpu_above_90` | `nvidia_smi.csv` fraction of samples ≥ 90 % |
| `mean_A_mpc` | `data.jld2` `trajectory_mpc[:, 3]` mean |
| `final_A_mpc` | `data.jld2` `trajectory_mpc[end, 3]` |
| `mean_A_baseline` | `data.jld2` `trajectory_baseline[:, 3]` mean |
| `final_A_baseline` | `data.jld2` `trajectory_baseline[end, 3]` |
| `final_B_mpc` | `data.jld2` `trajectory_mpc[end, 1]` |
| `final_F_mpc` | `data.jld2` `trajectory_mpc[end, 2]` |
| `mean_phi_per_stride` | `data.jld2` `daily_phi_per_stride` mean |
| `final_phi_per_stride` | `data.jld2` `daily_phi_per_stride[end]` |
| `mean_phi_at_stride21` | the Julia tech guide's headline metric — `daily_phi_per_stride[21]` (skipped if `n_strides < 22`) |
| `bench_exit_code` | from the launcher loop |

Helper script: a small Python file at
`compare_v15_julia_vs_python/src/extract_horizon_results.py` (new).
Reads one horizon's directory, writes that horizon's `results.json`,
and appends one row to the sweep-level `horizons_results.csv`. We
add `from h5py import File` for the JLD2 reader (the existing repo
uses `h5py` for Julia ↔ Python crossover per the writeup).

After the sweep finishes, a final pass renders `horizons_results.csv`
into Markdown table form at the bottom of `SWEEP_MANIFEST.md`.

## Plan

### Step 0 — Write the launcher (`version_1_5_Julia/tools/launchers/run_julia_horizon_sweep.sh`)

A single bash script that takes a list of T_days values and runs the
bench sequentially. Mirrors the v2 Python `run_horizon.sh` pattern
but loops over multiple T's.

```bash
#!/usr/bin/env bash
# run_julia_horizon_sweep.sh [T1 T2 …]
# Default: 14 28 42 56 70 84
set -u
T_LIST=("${@:-14 28 42 56 70 84}")

REPO="$HOME/Repos/python-smc2-filtering-control"
SWEEP_DATE=$(date +%Y-%m-%d)
SWEEP_ROOT="$REPO/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_${SWEEP_DATE}"
PLOT_DIR="$REPO/compare_v15_julia_vs_python/src"
mkdir -p "$SWEEP_ROOT"

cd "$REPO/version_1_5_Julia"

# ── Step 0: one-shot microbench profile (only if it doesn't already
# exist for this sweep date — kernel walls don't depend on T-days). ──
PROFILE_OUT="$SWEEP_ROOT/microbench_profile.txt"
if [ ! -f "$PROFILE_OUT" ]; then
    echo "[$(date '+%H:%M:%S')] one-shot microbench profile → $PROFILE_OUT"
    julia --project=. \
        "$PLOT_DIR/profile_gpu_julia_v15.jl" --both 2>&1 | tee "$PROFILE_OUT"
fi

MANIFEST="$SWEEP_ROOT/SWEEP_MANIFEST.md"
{
  echo "# Julia v1.5 horizon sweep — $(date)"
  echo
  echo "| T_days | start | wall_s | exit_code | dir |"
  echo "|---|---|---|---|---|"
} > "$MANIFEST"

for T in ${T_LIST[@]}; do
    DIR="$SWEEP_ROOT/T${T}d_seed42"
    mkdir -p "$DIR"
    LOG="$DIR/bench.log"
    SMI_CSV="$DIR/nvidia_smi.csv"
    START_TS=$(date '+%Y-%m-%d %H:%M:%S')
    T0=$(date +%s)
    echo "[$START_TS] T=${T}d → $DIR"

    # ── Start a 1 Hz nvidia-smi sampler in the background, kill it
    # at end-of-bench. Captures GPU util %, used VRAM (MiB), and
    # power draw per second so we can quote per-horizon mean util,
    # peak VRAM, and time-spent-saturated. ──
    {
        echo "timestamp,gpu_util_percent,memory_used_mib,power_w"
        while true; do
            nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw \
                       --format=csv,noheader,nounits \
                | awk -F', ' '{print $1","$2","$3","$4}'
            sleep 1
        done
    } > "$SMI_CSV" &
    SMI_PID=$!

    # ── Run the bench (writes data.jld2 + manifest.json + per_stride.csv) ──
    julia --project=. tools/bench_smc_full_mpc_fsa_gpu.jl \
        --T-days "$T" --seed 42 --output-dir "$DIR" \
        2>&1 | tee "$LOG"
    rc=${PIPESTATUS[0]}

    # ── Stop the GPU sampler. ──
    kill "$SMI_PID" 2>/dev/null
    wait "$SMI_PID" 2>/dev/null

    # ── Force-call the canonical plotters on data.jld2 (the bench's
    # internal auto-plot points at the deprecated v2 tree and may
    # silently @warn). ──
    if [ "$rc" -eq 0 ] && [ -f "$DIR/data.jld2" ]; then
        STATE_PNG="$DIR/v15_T${T}d_traces.png"
        PARAM_PNG="$DIR/v15_T${T}d_param_traces.png"
        julia --project=. -e "
          include(\"$PLOT_DIR/plot_state_traces.jl\")
          d = load_run_data(\"$DIR/data.jld2\")
          plot_state_traces(d; out_path=\"$STATE_PNG\")
          include(\"$PLOT_DIR/plot_param_traces.jl\")
          plot_param_traces(d; out_path=\"$PARAM_PNG\",
                             T_total_days=$T, step_minutes=60, stride_bins=12)
        " 2>&1 | tee -a "$LOG"

        # ── Extract headline metrics → results.json + horizons_results.csv ──
        python "$PLOT_DIR/extract_horizon_results.py" \
            --horizon-dir "$DIR" --T-days "$T" \
            --sweep-csv "$SWEEP_ROOT/horizons_results.csv" \
            --bench-exit-code "$rc" 2>&1 | tee -a "$LOG"
    fi

    T1=$(date +%s)
    WALL=$((T1 - T0))
    echo "| $T | $START_TS | $WALL | $rc | T${T}d_seed42 |" >> "$MANIFEST"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] T=${T}d done in ${WALL}s exit=$rc"
done

# ── Render horizons_results.csv as a Markdown table at the bottom
# of SWEEP_MANIFEST.md so the sweep root is self-documenting. ──
if [ -f "$SWEEP_ROOT/horizons_results.csv" ]; then
    {
        echo
        echo "## Per-horizon results"
        echo
        python -c "
import csv
with open('$SWEEP_ROOT/horizons_results.csv') as f:
    rows = list(csv.reader(f))
hdr, *body = rows
print('| ' + ' | '.join(hdr) + ' |')
print('|' + '|'.join(['---'] * len(hdr)) + '|')
for r in body:
    print('| ' + ' | '.join(r) + ' |')
"
    } >> "$MANIFEST"
fi
```

Single small file, no dependencies beyond the bench + the canonical
plotters at `compare_v15_julia_vs_python/src/`. Works for any subset
of horizons; default behaviour = full sweep. The microbench profile
is one-shot (skipped on rerun if `microbench_profile.txt` already
exists in the sweep dir).

### Step 1 — Quick verification at T=14 (~10 min, foreground)

```bash
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
bash tools/launchers/run_julia_horizon_sweep.sh 14
```

After it completes, verify by reading the directory listing:

```bash
ls compare_v15_julia_vs_python/example_run/julia_horizon_sweep_<date>/T14d_seed42/
```

Expect 6 files: `data.jld2`, `manifest.json`, `experiment_run.md`,
`per_stride.csv`, `v15_T14d_traces.png`, `v15_T14d_param_traces.png`,
plus `bench.log`.

Read both PNGs back and check:
* state traces show 28 strides (T=14 × 2 strides/day);
* param traces have 10 panels with truth lines.

If anything's off, fix the launcher before launching the long run.

### Step 2 — Overnight chain T=28 → T=84

```bash
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia
nohup bash tools/launchers/run_julia_horizon_sweep.sh 28 42 56 70 84 \
    > /tmp/julia_horizon_sweep_$(date +%Y%m%d_%H%M).log 2>&1 &
```

Detached via `nohup`, sequential (single GPU). Estimated ~3.5 h
total. The script writes per-T `bench.log` files under each
horizon's directory; the master `nohup` log captures the
launcher-level lines (start/done timestamps + per-T exit codes)
plus tee'd Julia stdout.

### Step 3 — Post-sweep summary

Once Step 2 finishes:

* The launcher has already written `SWEEP_MANIFEST.md` with the
  measured wall + exit code per horizon.
* For each horizon, read `manifest.json` to extract:
  total_wall_s, n_strides, n_replans, last-stride applied Φ, mean A
  vs baseline.
* Append a "Per-horizon results" table to `SWEEP_MANIFEST.md` so
  the sweep root is self-documenting.

## Critical files

To create:
- `version_1_5_Julia/tools/launchers/run_julia_horizon_sweep.sh`
  — the sweep launcher with embedded GPU sampling + plot + extract
  steps.
- `compare_v15_julia_vs_python/src/extract_horizon_results.py`
  — small Python helper that reads a horizon's `data.jld2` (via
  `h5py`), `manifest.json`, `per_stride.csv`, and `nvidia_smi.csv`
  and writes both per-horizon `results.json` and a sweep-level
  `horizons_results.csv` (creates with header on first call,
  appends row on subsequent calls).

To create at run time (the bench writes them):
- `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_<date>/SWEEP_MANIFEST.md`
- `compare_v15_julia_vs_python/example_run/julia_horizon_sweep_<date>/T<N>d_seed42/...` × 6

Read-only references:
- `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` — the
  bench (defaults already saturated; we only set `--T-days`,
  `--seed`, `--output-dir`).
- `compare_v15_julia_vs_python/docs/technical_guide_to_current_best_Julia_SMC2FC_config.pdf`
  — config provenance.
- `compare_v15_julia_vs_python/src/plot_state_traces.jl`,
  `plot_param_traces.jl`, `profile_gpu_julia_v15.jl` — canonical
  tools called explicitly by the launcher.
- `version_2_Python_JAX/tools/launchers/run_horizon.sh` — pattern
  to mirror for the bash structure.

## Verification

* **Step 1 gate (T=14 smoke):**
  * `T14d_seed42/` contains `data.jld2`, `manifest.json`,
    `experiment_run.md`, `per_stride.csv`, `nvidia_smi.csv`,
    `results.json`, `bench.log`, `v15_T14d_traces.png`,
    `v15_T14d_param_traces.png`.
  * `SWEEP_MANIFEST.md` has one row showing exit_code = 0.
  * `microbench_profile.txt` exists in the sweep root.
  * `horizons_results.csv` exists with one row of measured headline
    metrics (mean A_MPC, peak VRAM, mean GPU util, etc.).
  * Both PNGs render correctly when opened.
  * If any of the above is missing — fix the launcher / extractor
    script before Step 2.

* **Step 2 gate (overnight chain):** every horizon directory has
  the same 6 artefacts; every row of `SWEEP_MANIFEST.md` shows
  `exit_code = 0`. If any horizon failed, its `bench.log` survives
  for postmortem (the loop continues to the next horizon on
  failure rather than aborting — `set -u` only catches unset vars,
  not non-zero exits).

* **Sanity:** `T<N>d_traces.png` filenames match the horizon
  (e.g. `T84d_seed42/v15_T84d_traces.png`, not `v15_T28d_traces.png`).
  The plotter's output name is built from the `--T-days` value via
  the launcher's `T${T}d` interpolation, so this is automatic, but
  worth eyeballing on the T=14 verification.

## Open questions

None at this point. The sweep set (14, 28, 42, 56, 70, 84) and seed
(42) match the user's note; everything else is the bench's
saturated-tier default per the tech guide.
