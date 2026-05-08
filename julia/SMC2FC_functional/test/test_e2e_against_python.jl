"""
    test_e2e_against_python.jl

Skeleton for end-to-end regression tests that compare `SMC2FC_functional`
outputs to a Python reference snapshot.

# How to populate the snapshot

Run the Python reference once to dump posterior particles + log-
likelihoods to a `.npz` file under `test/snapshots/`. The expected
shape of each canonical example is documented below the per-example
testset.

# Status

This file is a skeleton — the snapshots are not committed. To enable a
test, drop the corresponding `.npz` into `test/snapshots/` and remove
the `@test_skip` marker.
"""

using Test
using Random
using SMC2FC_functional

const SNAPSHOT_DIR = joinpath(@__DIR__, "snapshots")

# ── Tiny scalar OU example ─────────────────────────────────────────────────

@testset "End-to-end vs Python — tiny scalar OU (placeholder)" begin
    snap = joinpath(SNAPSHOT_DIR, "tiny_ou.npz")
    if !isfile(snap)
        @test_skip "snapshot $snap missing — run the Python reference and drop the file in"
    else
        # Expected snapshot keys:
        #   posterior_particles : (n_smc, d_theta) Float64
        #   log_marginal       : scalar Float64
        #   posterior_mean     : (d_theta,) Float64
        #
        # When implementing: load via NPZ.jl, run the equivalent
        # `bootstrap_log_likelihood` + `run_smc_window` here, compare
        # posterior means within rtol=1e-3 and log-marginal within 0.1
        # nats.
        @test true
    end
end

# ── Tiny FSA-v2 closed-loop example ────────────────────────────────────────

@testset "End-to-end vs Python — tiny FSA control (placeholder)" begin
    snap = joinpath(SNAPSHOT_DIR, "tiny_fsa_control.npz")
    if !isfile(snap)
        @test_skip "snapshot $snap missing — run the Python reference and drop the file in"
    else
        # Expected snapshot keys:
        #   mean_schedule : (n_steps,) Float64
        #   posterior_particles : (n_smc, theta_dim) Float64
        @test true
    end
end
