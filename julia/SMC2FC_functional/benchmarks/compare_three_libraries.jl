"""
    compare_three_libraries.jl

Side-by-side comparison harness for:

1. `SMC2FC` — the existing Julia port at `julia/SMC2FC/`.
2. `SMC2FC_functional` — this library.
3. `smc2fc` — the Python reference at `smc2fc/`.

# What this script does

For each canonical tiny example (transforms round-trip, ESS sanity check,
RBF schedule decoding, prior-cost calibration), evaluate the same
quantity through every library and print a compact table of
max-abs-difference and max-rel-difference.

# How to run

```bash
cd julia/SMC2FC_functional
julia --project=. benchmarks/compare_three_libraries.jl
```

Expects both `SMC2FC` and `SMC2FC_functional` `Pkg.dev`'d into the
project. The Python comparison is invoked via a shell-out — see
`run_python_reference()` below — so `python` and `smc2fc` must be on
PATH (e.g. via `conda activate comfyenv`).

# Functional contract

This script only **reads** from the libraries; it does not mutate any
shared state. Each comparison block is a pure function of its inputs.
"""

using Random
using Statistics
using Printf

# ── Load both Julia libraries side-by-side ──────────────────────────────────

# Paths — relative to this file's directory.
const SCRIPT_DIR     = @__DIR__
const REPO_ROOT      = abspath(joinpath(SCRIPT_DIR, "..", "..", ".."))
const SMC2FC_DIR     = joinpath(REPO_ROOT, "julia", "SMC2FC")
const SMC2FC_FN_DIR  = joinpath(REPO_ROOT, "julia", "SMC2FC_functional")
const SMC2FC_PY_PKG  = joinpath(REPO_ROOT, "smc2fc")

println("== Comparing three SMC²-FC libraries ==")
println("  Existing Julia port: $SMC2FC_DIR")
println("  Functional Julia   : $SMC2FC_FN_DIR")
println("  Python reference   : $SMC2FC_PY_PKG")
println()

# Both Julia libraries need to be on the load path.
import Pkg
Pkg.activate(SMC2FC_FN_DIR)
# Add the existing SMC2FC port as a `dev` dep if not already present.
# We're already activated in SMC2FC_functional, so we don't `develop` it.
let proj = Pkg.project().dependencies
    if !haskey(proj, "SMC2FC")
        Pkg.develop(path = SMC2FC_DIR)
    end
end

using SMC2FC
using SMC2FC_functional
const FN = SMC2FC_functional

# Python is invoked via `run(`python -c "..."`)` — see helper below.

# ── Helpers ─────────────────────────────────────────────────────────────────

"""
    diff_summary(name, a, b) -> NamedTuple

Print and return max-abs and max-rel differences between two scalars or
arrays.
"""
function diff_summary(name, a, b)
    aa, bb = collect(a), collect(b)
    abs_diff = maximum(abs.(aa .- bb))
    denom    = maximum(abs.(aa)) + 1e-30
    rel_diff = abs_diff / denom
    @printf("  %-40s  max|Δ| = %10.3e   max|Δ|/|a| = %10.3e\n",
            name, abs_diff, rel_diff)
    return (max_abs = abs_diff, max_rel = rel_diff)
end

"""
    run_python_reference(snippet) -> String

Run a small Python program with `smc2fc` on PATH and return its stdout.
Used to extract scalar reference values from the Python implementation.

# Notes
- Caller is responsible for getting `comfyenv` (the project's conda env)
    activated before running this script.
- Output is parsed by the caller; any parsing strictness is the caller's
    problem.
"""
function run_python_reference(snippet::AbstractString)
    cmd = `python -c $snippet`
    return read(cmd, String)
end

# ── Comparison blocks ───────────────────────────────────────────────────────

"""
    compare_transforms() -> Nothing

Compare lognormal / normal / beta / vonmises log-priors across the two
Julia libraries and (optionally) Python. Both Julia libraries should
agree to floating-point precision because they share formulas.
"""
function compare_transforms()
    println("--- 1. Transforms (per-component log-prior) ---")
    u = 0.3

    cases = [
        ("lognormal(μ=0, σ=1)", (
            SMC2FC.log_prior_unconstrained(SMC2FC.LogNormalPrior(0.0, 1.0), u),
            FN.log_prior_unconstrained(FN.LogNormalPrior(0.0, 1.0), u),
        )),
        ("normal(μ=2, σ=0.5)", (
            SMC2FC.log_prior_unconstrained(SMC2FC.NormalPrior(2.0, 0.5), u),
            FN.log_prior_unconstrained(FN.NormalPrior(2.0, 0.5), u),
        )),
        ("vonmises(μ=0, κ=4)", (
            SMC2FC.log_prior_unconstrained(SMC2FC.VonMisesPrior(0.0, 4.0), u),
            FN.log_prior_unconstrained(FN.VonMisesPrior(0.0, 4.0), u),
        )),
        ("beta(α=2, β=5)", (
            SMC2FC.log_prior_unconstrained(SMC2FC.BetaPrior(2.0, 5.0), u),
            FN.log_prior_unconstrained(FN.BetaPrior(2.0, 5.0), u),
        )),
    ]
    for (name, (a, b)) in cases
        diff_summary(name, [a], [b])
    end
    println()
end

"""
    compare_ess() -> Nothing

Compare `compute_ess` on the same log-weight vector. Both Julia
libraries should agree to floating-point precision.
"""
function compare_ess()
    println("--- 2. compute_ess on a fixed log-weight vector ---")
    rng = MersenneTwister(42)
    log_w = randn(rng, 256)
    ess_orig = SMC2FC.compute_ess(log_w)
    ess_fn   = FN.compute_ess(log_w)
    diff_summary("ESS (orig vs functional)", [ess_orig], [ess_fn])
    @printf("    orig = %.6f, functional = %.6f\n", ess_orig, ess_fn)
    println()
end

"""
    compare_rbf_schedule() -> Nothing

Compare `RBFBasis` design matrix and `schedule_from_theta` decoding
between the two Julia libraries.
"""
function compare_rbf_schedule()
    println("--- 3. RBF schedule decoding ---")
    n_steps, dt, n_anchors = 20, 0.1, 4
    rng = MersenneTwister(0)
    θ   = randn(rng, n_anchors)

    b_orig = SMC2FC.RBFBasis(n_steps, dt, n_anchors)
    b_fn   = FN.RBFBasis(n_steps, dt, n_anchors)

    Φ_orig = SMC2FC.design_matrix(b_orig)
    Φ_fn   = FN.design_matrix(b_fn)
    diff_summary("design_matrix Φ", Φ_orig, Φ_fn)

    s_orig = SMC2FC.schedule_from_theta(b_orig, θ; Φ = Φ_orig)
    s_fn   = FN.schedule_from_theta(b_fn,   θ; Φ = Φ_fn)
    diff_summary("schedule_from_theta", s_orig, s_fn)
    println()
end

"""
    compare_calibrate_beta_max() -> Nothing

Compare `calibrate_beta_max` on a quadratic cost. Both libraries should
match to ~1e-12 since they share the same RNG seed.
"""
function compare_calibrate_beta_max()
    println("--- 4. calibrate_beta_max on a quadratic cost ---")
    cost(θ) = 0.5 * sum(abs2, θ)

    β_orig, μ_orig, σ_orig = SMC2FC.calibrate_beta_max(
        cost; theta_dim = 4, sigma_prior = 1.0,
        n_samples = 256, target_nats = 4.0, seed = 0,
    )
    β_fn, μ_fn, σ_fn = FN.calibrate_beta_max(
        cost; theta_dim = 4, sigma_prior = 1.0,
        n_samples = 256, target_nats = 4.0, seed = 0,
    )
    diff_summary("β_max", [β_orig], [β_fn])
    diff_summary("prior_cost_mean", [μ_orig], [μ_fn])
    diff_summary("prior_cost_std",  [σ_orig], [σ_fn])
    println()
end

# ── Optional Python comparison ──────────────────────────────────────────────

"""
    PYTHON_BIN

Path to the Python interpreter used by the comparison block. The project
uses the `comfyenv` conda env per its CLAUDE.md, so we hard-code the
matching binary. Falls back to `python` on PATH if the conda binary
isn't present.
"""
const PYTHON_BIN = let
    candidate = "/home/ajay/miniconda3/envs/comfyenv/bin/python"
    isfile(candidate) ? candidate : "python"
end

"""
    run_python_reference_bin(snippet) -> String

Like `run_python_reference` but uses `PYTHON_BIN` (the project's conda
env) rather than `python` on PATH.
"""
function run_python_reference_bin(snippet::AbstractString)
    cmd = `$PYTHON_BIN -c $snippet`
    return read(cmd, String)
end

"""
    compare_python_transforms() -> Nothing

Cross-check the vector-level transforms against the Python reference at
`smc2fc.transforms.unconstrained`. Builds a 4-component synthetic
transform-array dict `T` (one component per prior type), evaluates
`log_prior_unconstrained(u, T)` and `constrained_to_unconstrained(θ, T)`
on Python, and compares against the same operations in the two Julia
libraries.

Skipped if `PYTHON_BIN` is not runnable or `smc2fc` is not importable.
"""
function compare_python_transforms()
    println("--- 5. Python `smc2fc.transforms.unconstrained` comparison ---")
    snippet = """
import sys
sys.path.insert(0, '$REPO_ROOT')
try:
    import numpy as np, jax.numpy as jnp
    from smc2fc.transforms.unconstrained import (
        log_prior_unconstrained, constrained_to_unconstrained,
        unconstrained_to_constrained,
    )
    # Synthetic 4-component T: index 0 lognormal(0,1),
    # 1 normal(2,0.5), 2 beta(2,5), 3 vonmises(0,4).
    T = {
        'is_log':   jnp.array([1,0,0,0], dtype=jnp.float32),
        'is_logit': jnp.array([0,0,1,0], dtype=jnp.float32),
        'is_ident': jnp.array([0,1,0,1], dtype=jnp.float32),
        'is_ln':    jnp.array([1,0,0,0], dtype=jnp.float32),
        'is_norm':  jnp.array([0,1,0,0], dtype=jnp.float32),
        'is_vm':    jnp.array([0,0,0,1], dtype=jnp.float32),
        'is_bt':    jnp.array([0,0,1,0], dtype=jnp.float32),
        'ln_mu':    jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        'ln_sigma': jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
        'n_mu':     jnp.array([0.0, 2.0, 0.0, 0.0], dtype=jnp.float32),
        'n_sigma':  jnp.array([1.0, 0.5, 1.0, 1.0], dtype=jnp.float32),
        'vm_mu':    jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
        'vm_kappa': jnp.array([1.0, 1.0, 1.0, 4.0], dtype=jnp.float32),
        'beta_a':   jnp.array([1.0, 1.0, 2.0, 1.0], dtype=jnp.float32),
        'beta_b':   jnp.array([1.0, 1.0, 5.0, 1.0], dtype=jnp.float32),
    }
    theta = jnp.array([1.5, -0.7, 0.3, 0.4], dtype=jnp.float32)
    u     = constrained_to_unconstrained(theta, T)
    theta2= unconstrained_to_constrained(u, T)
    lp    = float(log_prior_unconstrained(u, T))
    # Emit one line per field, space-separated, prefixed with the field
    # name. Easy for Julia to split.
    def line(name, vals):
        return name + ' ' + ' '.join(repr(float(v)) for v in vals)
    print(line('theta',  np.asarray(theta)))
    print(line('u',      np.asarray(u)))
    print(line('theta2', np.asarray(theta2)))
    print('lp ' + repr(lp))
except Exception as e:
    import traceback
    print('PYTHON_ERROR: ' + str(e))
    traceback.print_exc()
"""
    out = try
        run_python_reference_bin(snippet)
    catch err
        @warn "Python comparison skipped (could not run python): $err"
        return
    end

    if occursin("PYTHON_ERROR", out)
        @warn "Python comparison skipped: $out"
        return
    end

    # Parse the per-field lines emitted by Python. Each line is
    # `<name> <v1> <v2> ...`.
    fields = Dict{String,Vector{Float64}}()
    for raw in split(out, '\n')
        line = strip(raw)
        isempty(line) && continue
        toks = split(line)
        length(toks) < 2 && continue
        try
            fields[String(toks[1])] = parse.(Float64, toks[2:end])
        catch
            # ignore non-data lines (e.g. JAX warnings)
        end
    end
    if !haskey(fields, "theta") || !haskey(fields, "lp")
        @warn "Could not parse Python output; raw stdout follows."
        println(out)
        return
    end
    py_theta  = fields["theta"]
    py_u      = fields["u"]
    py_theta2 = fields["theta2"]
    py_lp     = fields["lp"][1]

    # Same priors in Julia (use the existing port and the new functional
    # library — both should agree with Python within fp32 tolerance,
    # because the Python pipeline runs in fp32 by default).
    priors = SMC2FC.PriorType[
        SMC2FC.LogNormalPrior(0.0, 1.0),
        SMC2FC.NormalPrior(2.0, 0.5),
        SMC2FC.BetaPrior(2.0, 5.0),
        SMC2FC.VonMisesPrior(0.0, 4.0),
    ]
    priors_fn = FN.PriorType[
        FN.LogNormalPrior(0.0, 1.0),
        FN.NormalPrior(2.0, 0.5),
        FN.BetaPrior(2.0, 5.0),
        FN.VonMisesPrior(0.0, 4.0),
    ]
    theta_jl = py_theta
    u_orig  = SMC2FC.constrained_to_unconstrained(theta_jl, priors)
    u_fn    = FN.constrained_to_unconstrained(theta_jl, priors_fn)
    theta2_orig = SMC2FC.unconstrained_to_constrained(u_orig, priors)
    theta2_fn   = FN.unconstrained_to_constrained(u_fn, priors_fn)
    lp_orig = SMC2FC.log_prior_unconstrained(u_orig, priors)
    lp_fn   = FN.log_prior_unconstrained(u_fn, priors_fn)

    println("  Python u   = $py_u")
    println("  orig Julia = $u_orig")
    println("  fn Julia   = $u_fn")

    diff_summary("u: Python vs orig",     py_u, u_orig)
    diff_summary("u: Python vs functional", py_u, u_fn)
    diff_summary("u: orig vs functional",   u_orig, u_fn)
    println()

    diff_summary("θ round-trip: Py vs orig", py_theta2, theta2_orig)
    diff_summary("θ round-trip: Py vs fn",   py_theta2, theta2_fn)
    println()

    diff_summary("log-prior: Python vs orig",     [py_lp], [lp_orig])
    diff_summary("log-prior: Python vs functional", [py_lp], [lp_fn])
    diff_summary("log-prior: orig vs functional",   [lp_orig], [lp_fn])
    println()
end

# ── Run all comparisons ─────────────────────────────────────────────────────

function main()
    compare_transforms()
    compare_ess()
    compare_rbf_schedule()
    compare_calibrate_beta_max()
    compare_python_transforms()
    println("== Comparison complete ==")
end

main()
