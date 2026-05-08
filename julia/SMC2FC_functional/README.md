# SMC2FC_functional

A parallel Julia rewrite of [`julia/SMC2FC/`](../SMC2FC/) with a functional API
surface and Google-style docstrings. The two libraries are designed to run
**side-by-side in the same Julia session**, so the new implementation can be
checked against the existing port and against the Python reference in
[`smc2fc/`](../../smc2fc/).

## Goals

1. Functional API: top-level functions take immutable inputs and return
   immutable outputs; pre-allocated buffers live behind an immutable
   `Workspace` container; no `!`-mutating functions are exposed publicly.
2. Idiomatic Julian: multiple dispatch, immutability, dispatch-driven
   specialisation, fewer per-call allocations.
3. Google-style docstrings on every file and every function.

## Plan archive

Full design plan, pros / cons, and phase breakdown:

- [`claude_plans/Surgical_Julian_rewrite_of_Julia_SMC2FC_port_2026-05-08_1406.md`](../../claude_plans/Surgical_Julian_rewrite_of_Julia_SMC2FC_port_2026-05-08_1406.md)

## Three-library comparison

```bash
cd julia/SMC2FC_functional
julia --project=. benchmarks/compare_three_libraries.jl
```

This prints a table comparing posterior means / log-likelihoods between:
- the existing Julia port at `julia/SMC2FC/`,
- this library,
- the Python reference at `smc2fc/`.
