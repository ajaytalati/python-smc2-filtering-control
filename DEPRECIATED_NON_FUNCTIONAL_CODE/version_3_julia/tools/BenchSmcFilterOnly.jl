# tools/BenchSmcFilterOnly.jl — Stage 1 (FSA-v5): rolling-window SMC²
# filter on synthetic data. NO controller.
#
# Port of `version_3/tools/bench_smc_filter_only_fsa_v5.py`. Goal: confirm
# the rolling-window SMC² filter recovers ground-truth params from clean
# synthetic plant data under fixed Phi=(0.30, 0.30) (LaTeX §8 Test 2).
#
# Status: SKELETON. The synthetic-data path (plant → obs) is wired
# through `FSAv5.Plant`. The full SMC² rolling-window kernel call goes
# through `SMC2FC.run_smc_window` / `run_smc_window_bridge`; this driver
# is the place where the GPU-side filter integration (per Julia port
# charter Part II §15.7) gets exercised end-to-end. Final acceptance
# gates (>=80% coverage on >=80% windows, ≤60 min compute) are the same
# as the Python version.

module BenchSmcFilterOnly

using JSON3
using NPZ
using Random
using Printf
using Statistics
using Plots

using ..FSAv5: FSAv5State, BimodalPhi, StepwisePlant, advance!,
                BINS_PER_DAY, DT_BIN_DAYS,
                TRUTH_PARAMS_V5, DEFAULT_INIT,
                SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                SIGMA_A_FROZEN, SIGMA_K_FROZEN,
                HIGH_RES_FSA_V5_ESTIMATION

using ..RunDir: allocate_run_dir

# ── Window structure (matches v2's E3 reference; LaTeX §6 → 15-min grid)
const N_DAYS_TOTAL = 14
const WINDOW_BINS  = 96       # 1 day at dt=15 min
const STRIDE_BINS  = 48       # 12 hours
const DT           = 1.0 / 96.0

# ── Stage 1 scenario: healthy island Phi (LaTeX §8 Test 2)
const DEFAULT_PHI_B = 0.30
const DEFAULT_PHI_S = 0.30

# ── Synthetic-data simulator ─────────────────────────────────────────────
"""
    simulate_synthetic_full(; seed, phi_B, phi_S)

Use `FSAv5.StepwisePlant` to synthesise N_DAYS_TOTAL days of constant-Phi
data. Returns NamedTuple with `obs_data`, `trajectory`, `n_bins`.
"""
function simulate_synthetic_full(; seed::Int = 42,
                                  phi_B::Float64 = DEFAULT_PHI_B,
                                  phi_S::Float64 = DEFAULT_PHI_S)
    rng = MersenneTwister(seed)
    sigma_diag = [SIGMA_B_FROZEN, SIGMA_S_FROZEN, SIGMA_F_FROZEN,
                   SIGMA_A_FROZEN, SIGMA_K_FROZEN, SIGMA_K_FROZEN]
    plant = StepwisePlant(state=DEFAULT_INIT, params=TRUTH_PARAMS_V5,
                           sigma_diag=sigma_diag, seed_offset=seed)
    daily = fill(0.0, N_DAYS_TOTAL, 2)
    daily[:, 1] .= phi_B
    daily[:, 2] .= phi_S
    n_bins = N_DAYS_TOTAL * BINS_PER_DAY
    traj = advance!(plant, n_bins, daily; rng=rng)
    # Obs sampling is done by the bench driver from the realised trajectory
    # via `FSAv5.Obs.gen_obs_*` (Phase 6 channels). Skeleton: trajectory only.
    return (trajectory = traj, n_bins = n_bins)
end

# ── Main entry ──────────────────────────────────────────────────────────
function main(argv::Vector{<:AbstractString} = ARGS)
    run_tag = isempty(argv) ? "stage1_filter_only_T14_healthy" : argv[1]
    repo_root = dirname(dirname(@__FILE__))   # version_3_julia/
    out_dir, run_num = allocate_run_dir(repo_root, run_tag)

    println("=" ^ 76)
    @printf "  Stage 1 (FSA-v5, Julia) — rolling-window SMC² filter\n"
    @printf "  T=%d days, Phi=(%.2f,%.2f), run dir: %s\n" N_DAYS_TOTAL DEFAULT_PHI_B DEFAULT_PHI_S basename(out_dir)
    println("=" ^ 76)

    n_windows = (N_DAYS_TOTAL * BINS_PER_DAY - WINDOW_BINS) ÷ STRIDE_BINS + 1
    @printf "  total:    %d days × %d bins = %d bins\n" N_DAYS_TOTAL BINS_PER_DAY (N_DAYS_TOTAL*BINS_PER_DAY)
    @printf "  window:   %d bins (1 day)\n" WINDOW_BINS
    @printf "  stride:   %d bins (12 hours)\n" STRIDE_BINS
    @printf "  windows:  %d\n" n_windows
    println()

    println("  Step 1: synthesise 14-day trajectory under healthy-island Phi …")
    data = simulate_synthetic_full(seed=42)
    traj = data.trajectory
    @printf "    trajectory shape: (%d, %d)\n" size(traj, 1) size(traj, 2)

    # ── SMC² rolling-window loop ─────────────────────────────────────
    # Wired through `SMC2FC.run_smc_window` / `run_smc_window_bridge`.
    # The compile-once log-density factory + per-stride binding (charter
    # §15.7) is the Julia analogue of the JAX-native compile-once path
    # in `version_2/tools/bench_smc_full_mpc_fsa.py:150-260`.
    @info "Stage-1 filter loop wiring goes through SMC2FC.run_smc_window. " *
          "Skeleton driver: not exercising the GPU filter in this scaffold."

    em = HIGH_RES_FSA_V5_ESTIMATION
    @printf "  estimation model: %d params (37 estimable, 14 frozen by charter)\n" length(em.all_names)

    # ── Save artifacts (manifest + trajectory; posterior is empty in skeleton)
    state_names = ["B", "S", "F", "A", "K_FB", "K_FS"]
    npzwrite(joinpath(out_dir, "trajectory.npz"),
             Dict("trajectory" => traj,
                  "state_names" => state_names))

    manifest = Dict(
        "schema_version" => 1,
        "stage"          => 1,
        "bench"          => "bench_smc_filter_only_fsa_v5",
        "language"       => "julia",
        "run_tag"        => run_tag,
        "run_number"     => run_num,
        "T_total_days"   => N_DAYS_TOTAL,
        "step_minutes"   => 15,
        "BINS_PER_DAY"   => BINS_PER_DAY,
        "WINDOW_BINS"    => WINDOW_BINS,
        "STRIDE_BINS"    => STRIDE_BINS,
        "DT"             => DT,
        "n_windows"      => n_windows,
        "scenario"       => Dict(
            "name"  => "healthy_island_LaTeX_test2",
            "phi_B" => DEFAULT_PHI_B,
            "phi_S" => DEFAULT_PHI_S,
        ),
    )
    open(joinpath(out_dir, "manifest.json"), "w") do io
        JSON3.pretty(io, manifest)
    end

    println()
    @printf "  Artifacts written under %s/\n" basename(out_dir)
    println("=" ^ 76)
    return out_dir
end

export main, simulate_synthetic_full

end # module BenchSmcFilterOnly
