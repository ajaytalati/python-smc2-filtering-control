#!/usr/bin/env julia
# Gate 6 runner: activate a temporary project that has both
# `SMC2FC_functional` (via `dev`) and `JET` available, then `include`
# the gate's test file.
#
# This avoids editing the package's main Project.toml.
#
# Run from anywhere:
#     julia run_gate6.jl

using Pkg

const SMC2FC_FN_DIR = abspath(joinpath(@__DIR__, ".."))

# Temp env (deleted automatically when Julia exits).
Pkg.activate(; temp = true)
Pkg.develop(path = SMC2FC_FN_DIR)
Pkg.add("JET")

# Now include the gate test in this environment.
include(joinpath(@__DIR__, "test_gate6_jet.jl"))
