# compare_v15_julia_vs_python/

Self-contained snapshot of the Python+JAX vs Julia comparison work for FSA v1.5
closed-loop SMC²-MPC. Created so a fresh agent (or human reader) can pick up
the work without crawling the whole repo.

## Layout

```
compare_v15_julia_vs_python/
├── README.md                               (this file)
├── docs/
│   ├── julia_vs_python_v15_writeup.tex     ← main LaTeX technical report
│   ├── julia_vs_python_v15_writeup.pdf     ← compiled output (build below)
│   └── PLAN_ARCHIVE.md                     ← original plan that drove this work
├── src/                                    ← snapshot of the actual scripts in use
│   ├── compare_v15_julia_vs_python.py      ← three-way diagnostic (Python+JAX, Julia, baseline)
│   ├── profile_gpu_python.py               ← JAX microbench (mirrors Julia's profile_gpu.jl)
│   ├── profile_gpu_julia_v15.jl            ← v1.5 Julia GPU microbench
│   ├── profile_gpu_julia_v2_deprecated.jl  ← original v2 profile (deprecated; for reference)
│   ├── bench_python_v15.py                 ← v1.5 Python+JAX closed-loop bench
│   ├── bench_julia_v15.jl                  ← v1.5 Julia closed-loop bench
│   └── run_v15_T28d_compare.sh             ← paired-run launcher (profilers + benches + nvidia-smi)
└── example_run/                            ← T=7d, seed=42 run with the latest Python KF fix
    ├── FINDINGS_T7d_seed42.md
    ├── summary_T7d_seed42.txt
    ├── comparison_T7d_seed42.png           ← 6-panel three-way diagnostic plot
    ├── python_T7d_traces.png               ← 4-panel state traces (Python MPC vs baseline)
    ├── python_T7d_param_traces.png         ← 10-panel posterior parameter traces (Python)
    ├── julia_T7d_traces.png                ← 4-panel state traces (Julia MPC vs baseline)
    ├── julia_T7d_param_traces.png          ← 10-panel posterior parameter traces (Julia)
    ├── julia_profile.log
    └── python_profile.log
```

## Where the live code actually lives

Files in `src/` are static snapshots — useful for reading and as a stable
reference, but the **live, runnable code** lives in the main repo:

| Snapshot path | Live path |
|---|---|
| `src/bench_python_v15.py` | `version_1_5_Python_JAX/tools/bench_smc_full_mpc_fsa_v15.py` |
| `src/bench_julia_v15.jl` | `version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl` |
| `src/profile_gpu_python.py` | `version_1_5_Python_JAX/tools/profile_gpu.py` |
| `src/profile_gpu_julia_v15.jl` | `version_1_5_Julia/tools/profile_gpu.jl` |
| `src/compare_v15_julia_vs_python.py` | `version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py` |
| `src/run_v15_T28d_compare.sh` | `version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh` |

To run anything new, work from the live paths. `src/` is for archival reading.

## Quickstart for the next agent

```bash
# Activate env
conda activate comfyenv

# Compile the report (read this FIRST)
cd compare_v15_julia_vs_python/docs
pdflatex -interaction=nonstopmode julia_vs_python_v15_writeup.tex

# Run a fresh paired comparison (T=7d, seed=42; ~45 min total)
bash /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX/tools/launchers/run_v15_T28d_compare.sh 7 42

# Then run the three-way diagnostic on the run dir
cd /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX
JAX_ENABLE_X64=True PYTHONPATH=.:.. \
  python tools/compare_v15_julia_vs_python.py \
  /home/ajay/Repos/python-smc2-filtering-control/outputs/v15_julia_vs_python_T7d_seed42
```

## What's been done; what's next

See `docs/julia_vs_python_v15_writeup.tex` (the comprehensive report) and
`example_run/FINDINGS_T7d_seed42.md` (the most recent diagnostic findings).

The headline pointer for "what's next":
- The **Julia code has obvious GPU-utilisation headroom** (~25% mean util at
  current default config). Speed-up + GPU-saturation work is the biggest
  open item.
- A new functional Julia library at `julia/SMC2FC_functional/` is the
  intended drop-in replacement for the current `julia/SMC2FC/`. The proof-
  of-principle bench at
  `julia/SMC2FC_functional/benchmarks/bench_smc_full_mpc_fsa_v15_functional_gpu.jl`
  should become the default Julia entry point going forward.
- The controller (H3) still picks rest-heavy schedules at T=7d on both
  stacks. v2's reference plots show the schedule ramps UP at T=14d/T=28d,
  so the next experiment is a longer-horizon run after the Julia perf
  work lands.
