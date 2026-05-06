using SMC2FC
using Test
using Random

# ── Phase 1 — Foundations ────────────────────────────────────────────────────
include("test_phase1_types.jl")
include("test_phase1_config.jl")
include("test_phase1_transforms.jl")

# ── Phase 2 — Filtering ──────────────────────────────────────────────────────
include("test_phase2_kernels.jl")
include("test_phase2_ot.jl")
include("test_phase2_bootstrap.jl")
include("test_phase2_gpu.jl")

# ── Phase 3 — Outer SMC² ─────────────────────────────────────────────────────
include("test_phase3_smc2.jl")

# ── Phase 4 — Control ────────────────────────────────────────────────────────
include("test_phase4_control.jl")

# ── Phase 5 — Plant + Simulator ──────────────────────────────────────────────
include("test_phase5_simulator.jl")

# ── Phase 6 — End-to-end smoke + correctness ─────────────────────────────────
include("test_phase6_e2e.jl")

# Phase 3+ tests are added as later phases land.

# ── JET.jl static analysis (charter §18 audit gate) ──────────────────────────
include("test_jet.jl")

