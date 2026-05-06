# Stage A2 — open-loop schedule (raw 20-D pulse)
# Stage A3 — state-feedback gain vector K
#
# Mirror of:
#   version_1/tools/bench_smc_control_ou.py
#   version_1/tools/bench_smc_control_ou_state_feedback.py
#
# Each stage produces ONE PNG at `outputs/scalar_ou_lqg/<stage>_diagnostic.png`,
# panel-for-panel matching the Python plot:
#   A2: 1×2 — open-loop schedule (left) + cost histogram with refs (right)
#   A3: 1×2 — gain vs Riccati with ±1σ band (left) + cost histogram (right)

using Random
using Statistics: mean, std
using Printf
using Plots
using Plots.PlotMeasures: mm
gr()                    # GR backend — fast, matplotlib-shaped output

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "scalar_ou_lqg"))

using SMC2FC
include(joinpath(ROOT, "utils", "diagnostics.jl"))
using .Diagnostics: plot_cost_histogram!, STEELBLUE
using ScalarOULQG: build_open_loop_cost_fn, build_state_feedback_cost_fn,
                    lqr_riccati, lqr_optimal_cost,
                    lqg_optimal_cost_mc, open_loop_zero_control_cost_mc,
                    Control

T_steps = Control.TRUTH.T

# ── Analytical references ──────────────────────────────────────────────────
println("="^70)
println("Stages A2 + A3 — scalar OU LQG control")
println("="^70)

riccati = lqr_riccati(
    a = Control.TRUTH.a, b = Control.TRUTH.b,
    q = Control.TRUTH.q, r = Control.TRUTH.r, s = Control.TRUTH.s,
    sigma_w = Control.TRUTH.sigma_w, dt = Control.TRUTH.dt, T = T_steps,
)
lqr_perfect = lqr_optimal_cost(
    riccati = riccati,
    x0_mean = Control.TRUTH.x0_mean, x0_var = Control.TRUTH.x0_var,
    sigma_w = Control.TRUTH.sigma_w, dt = Control.TRUTH.dt, T = T_steps,
)
lqg_mc = lqg_optimal_cost_mc(
    a = Control.TRUTH.a, b = Control.TRUTH.b,
    q = Control.TRUTH.q, r = Control.TRUTH.r, s = Control.TRUTH.s,
    sigma_w = Control.TRUTH.sigma_w, sigma_v = Control.TRUTH.sigma_v,
    dt = Control.TRUTH.dt, T = T_steps,
    x0_mean = Control.TRUTH.x0_mean, x0_var = Control.TRUTH.x0_var,
    n_trials = 5000, seed = 0,
).mean_cost
ol_mc = open_loop_zero_control_cost_mc(
    a = Control.TRUTH.a, b = Control.TRUTH.b,
    q = Control.TRUTH.q, r = Control.TRUTH.r, s = Control.TRUTH.s,
    sigma_w = Control.TRUTH.sigma_w,
    dt = Control.TRUTH.dt, T = T_steps,
    x0_mean = Control.TRUTH.x0_mean, x0_var = Control.TRUTH.x0_var,
    n_trials = 5000, seed = 0,
).mean_cost

println("LQR perfect: $(round(lqr_perfect, digits=3))   ",
        "MC LQG: $(round(lqg_mc, digits=3))   ",
        "open-loop u=0: $(round(ol_mc, digits=3))")

cfg_outer = SMCConfig(
    n_smc_particles  = 128,
    target_ess_frac  = 0.5,
    num_mcmc_steps   = 5,
    max_lambda_inc   = 0.20,
    hmc_step_size    = 0.05,
    hmc_num_leapfrog = 6,
)

# ── Stage A2 ───────────────────────────────────────────────────────────────
println("\n--- Stage A2: open-loop schedule ---")

cost_open_loop_eval = build_open_loop_cost_fn(n_inner = 2000, seed = 99)
cost_open_loop_smc  = build_open_loop_cost_fn(n_inner = 64,   seed = 42)

spec_A2 = ControlSpec(
    name              = "scalar_ou_open_loop", version = "v1_julia",
    dt                = Control.TRUTH.dt, n_steps = T_steps,
    initial_state     = [Control.TRUTH.x0_mean],
    theta_dim         = T_steps, sigma_prior = 1.5,
    cost_fn           = cost_open_loop_smc,
    schedule_from_theta = identity,
)
result_A2 = run_tempered_smc_loop(spec_A2, cfg_outer, MersenneTwister(2026);
                                   calib_n = 128, target_nats = 6.0)

u_post_A2     = vec(mean(result_A2.particles; dims = 1))
particle_costs_A2 = [cost_open_loop_eval(@view result_A2.particles[i, :])
                      for i in 1:cfg_outer.n_smc_particles]
smc_cost_A2   = cost_open_loop_eval(u_post_A2)
ratio_A2      = smc_cost_A2 / ol_mc
println(@sprintf("SMC² mean cost: %.3f   ratio SMC²/open-loop: %.3f  %s",
                  smc_cost_A2, ratio_A2,
                  0.95 ≤ ratio_A2 ≤ 1.10 ? "PASS" : "FAIL"))

# ── A2 plot — panel-for-panel match ────────────────────────────────────────
default(size = (1300, 400), fontfamily = "Helvetica", legendfontsize = 8,
        titlefontsize = 11, framestyle = :box, grid = true, gridalpha = 0.3,
        left_margin = 10mm, right_margin = 4mm,
        bottom_margin = 8mm, top_margin = 4mm)

t_grid = (0:T_steps-1) .* Control.TRUTH.dt

p_left = plot(t_grid, u_post_A2; lw = 1.5, color = STEELBLUE,
                marker = :circle, ms = 5,
                label = "SMC² mean schedule",
                xlabel = "time (s)", ylabel = "u (control)",
                title = "Open-loop schedule")
hline!(p_left, [0.0]; color = :black, alpha = 0.3, lw = 0.5, label = nothing)

p_right = plot()
plot_cost_histogram!(p_right, particle_costs_A2;
                       references = [
                           "LQR perfect" => lqr_perfect,
                           "MC LQG"      => lqg_mc,
                           "open-loop"   => ol_mc,
                           "SMC² mean"   => smc_cost_A2,
                       ],
                       title = "SMC² cost distribution vs analytical references",
                       xlabel = "cost")

p_A2 = plot(p_left, p_right; layout = (1, 2), size = (1300, 400))
out_A2 = joinpath(ROOT, "outputs", "scalar_ou_lqg", "A2_control_diagnostic.png")
mkpath(dirname(out_A2))
savefig(p_A2, out_A2)
println("Saved: $out_A2")

# ── Stage A3 ───────────────────────────────────────────────────────────────
println("\n--- Stage A3: state-feedback gain ---")

cost_sf_eval = build_state_feedback_cost_fn(n_inner = 2000, seed = 99)
cost_sf_smc  = build_state_feedback_cost_fn(n_inner = 64,   seed = 42)

spec_A3 = ControlSpec(
    name              = "scalar_ou_state_feedback", version = "v1_julia",
    dt                = Control.TRUTH.dt, n_steps = T_steps,
    initial_state     = [Control.TRUTH.x0_mean],
    theta_dim         = T_steps, sigma_prior = 1.5,
    prior_mean        = fill(2.0, T_steps),    # bias toward Riccati range
    cost_fn           = cost_sf_smc,
    schedule_from_theta = identity,
)
result_A3 = run_tempered_smc_loop(spec_A3, cfg_outer, MersenneTwister(2026);
                                   calib_n = 128, target_nats = 6.0)

K_post = vec(mean(result_A3.particles; dims = 1))
K_std  = vec(std(result_A3.particles;  dims = 1))
particle_costs_A3 = [cost_sf_eval(@view result_A3.particles[i, :])
                      for i in 1:cfg_outer.n_smc_particles]
smc_cost_A3       = cost_sf_eval(K_post)
riccati_cost      = cost_sf_eval(riccati.gains)
ratio_A3_lqg = smc_cost_A3 / lqg_mc
K_rms        = sqrt(mean((K_post .- riccati.gains).^2) / mean(riccati.gains.^2))

println(@sprintf("SMC² mean cost: %.3f   ratio SMC²/LQG: %.3f  %s",
                  smc_cost_A3, ratio_A3_lqg,
                  0.95 ≤ ratio_A3_lqg ≤ 1.10 ? "PASS" : "FAIL"))
println(@sprintf("K RMS error vs Riccati: %.3f   %s",
                  K_rms, K_rms < 0.25 ? "PASS" : "FAIL"))

# ── A3 plot — panel-for-panel match ────────────────────────────────────────
DARKRED = RGB(0x8b/255, 0x00/255, 0x00/255)

p_left3 = plot(t_grid, K_post .+ K_std; fillrange = K_post .- K_std,
                fillalpha = 0.3, lw = 0, color = STEELBLUE,
                label = "SMC ±1σ", framestyle = :box,
                xlabel = "time (s)", ylabel = "feedback gain K_k",
                title = "State-feedback gain: SMC² posterior vs Riccati")
plot!(p_left3, t_grid, K_post; lw = 1.5, color = STEELBLUE,
       marker = :circle, ms = 5, label = "SMC² posterior mean K_k")
plot!(p_left3, t_grid, riccati.gains; lw = 1.5, color = DARKRED, ls = :dash,
       marker = :rect, ms = 5, label = "LQR Riccati K_k*")

p_right3 = plot()
plot_cost_histogram!(p_right3, particle_costs_A3;
                       references = [
                           "LQR perfect" => lqr_perfect,
                           "MC LQG"      => lqg_mc,
                           "open-loop"   => ol_mc,
                           "SMC² mean K" => smc_cost_A3,
                           "Riccati K"   => riccati_cost,
                       ],
                       title = "SMC² state-feedback cost vs analytical references",
                       xlabel = "cost")

p_A3 = plot(p_left3, p_right3; layout = (1, 2), size = (1300, 400))
out_A3 = joinpath(ROOT, "outputs", "scalar_ou_lqg", "A3_state_feedback_diagnostic.png")
savefig(p_A3, out_A3)
println("Saved: $out_A3")

println("\n--- summary ---")
println(@sprintf("A2 ratio (SMC²/OL):   %.3f     (Python: 1.036)", ratio_A2))
println(@sprintf("A3 ratio (SMC²/LQG):  %.3f     (Python: 0.995)", ratio_A3_lqg))
println(@sprintf("A3 K RMS error:       %.3f     (Python: 0.20)",  K_rms))

# Stash numbers for RESULT.md generation
open(joinpath(ROOT, "outputs", "scalar_ou_lqg", "_results.txt"), "w") do io
    println(io, "lqr_perfect=", lqr_perfect)
    println(io, "lqg_mc=",      lqg_mc)
    println(io, "ol_mc=",       ol_mc)
    println(io, "smc_cost_A2=", smc_cost_A2)
    println(io, "ratio_A2=",    ratio_A2)
    println(io, "smc_cost_A3=", smc_cost_A3)
    println(io, "ratio_A3=",    ratio_A3_lqg)
    println(io, "K_rms=",       K_rms)
    println(io, "riccati_cost=", riccati_cost)
end
