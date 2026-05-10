#!/usr/bin/env julia
# psim sanity — drive `plant_rollout` for 14 days at Φ=1 from INIT_STATE.
# Plots trajectory + obs samples. Pure functional driver: no mutable state.

ENV["FSA_STEP_MINUTES"] = "60"

using Printf
using Plots
using StaticArrays

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Plant: PlantState, plant_rollout, init_plant_state
using .FSAHighRes.Simulation: BINS_PER_DAY, DT_BIN_DAYS, DEFAULT_PARAMS, INIT_STATE


function main()
    n_days = 14
    n_bins = n_days * BINS_PER_DAY
    Φ_const = fill(Float32(1.0), n_bins)

    s0  = init_plant_state()
    out = plant_rollout(s0, Φ_const, DEFAULT_PARAMS, DT_BIN_DAYS, UInt64(0xC0FFEE))

    t_days = collect(0:n_bins-1) .* DT_BIN_DAYS
    @printf("psim run: %d days at Φ=1 from INIT_STATE\n", n_days)
    @printf("  BINS_PER_DAY = %d   DT_BIN_DAYS = %.6f\n", BINS_PER_DAY, DT_BIN_DAYS)
    @printf("  init  state: B = %.4f, F = %.4f, A = %.4f\n", INIT_STATE.B, INIT_STATE.F, INIT_STATE.A)
    @printf("  final state: B = %.4f, F = %.4f, A = %.4f\n",
            out.final_state.bfa[1], out.final_state.bfa[2], out.final_state.bfa[3])

    plt = plot(layout = (2, 2), size = (1300, 700), dpi = 120,
                fontfamily = "Helvetica", framestyle = :box,
                grid = true, gridalpha = 0.3,
                plot_title = "FSA v1.5 psim — 14d at Φ=1 from INIT_STATE",
                plot_titlefontsize = 11)

    plot!(plt[1], t_days, out.trajectory[:, 1]; lw = 1.6, label = "B (latent)",
          color = RGB(0x1f/255, 0x77/255, 0xb4/255))
    plot!(plt[1], t_days, out.obs_B; lw = 0.8, alpha = 0.5, label = "obs_B",
          color = RGB(0x9d/255, 0xc8/255, 0xe6/255))
    title!(plt[1], "B (fitness)"; titlefontsize = 11)
    xlabel!(plt[1], "time (days)"); ylabel!(plt[1], "B")

    plot!(plt[2], t_days, out.trajectory[:, 2]; lw = 1.6, label = "F (latent)",
          color = RGB(0x8b/255, 0x00/255, 0x00/255))
    plot!(plt[2], t_days, out.obs_F; lw = 0.8, alpha = 0.5, label = "obs_F",
          color = RGB(0xfd/255, 0xae/255, 0x9b/255))
    title!(plt[2], "F (fatigue)"; titlefontsize = 11)
    xlabel!(plt[2], "time (days)"); ylabel!(plt[2], "F")

    plot!(plt[3], t_days, out.trajectory[:, 3]; lw = 1.6, label = "A (latent)",
          color = RGB(0x2c/255, 0xa0/255, 0x2c/255))
    plot!(plt[3], t_days, out.obs_A; lw = 0.8, alpha = 0.5, label = "obs_A",
          color = RGB(0x9c/255, 0xd0/255, 0x90/255))
    title!(plt[3], "A (autonomic amplitude)"; titlefontsize = 11)
    xlabel!(plt[3], "time (days)"); ylabel!(plt[3], "A")

    plot!(plt[4], t_days, out.Phi; lw = 1.6, color = RGB(0xff/255, 0x7f/255, 0x0e/255),
          label = "Φ = 1")
    title!(plt[4], "control input Φ"; titlefontsize = 11)
    xlabel!(plt[4], "time (days)"); ylabel!(plt[4], "Φ")
    ylims!(plt[4], 0, 1.5)

    out_dir = joinpath(REPO_ROOT, "outputs", "fsa_high_res", "psim_sanity")
    mkpath(out_dir)
    out_path = joinpath(out_dir, "psim_14d_phi1.png")
    savefig(plt, out_path)
    @printf("\nwrote %s\n", out_path)
end

main()
