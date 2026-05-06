# Stages A2 (open-loop) + A3 (state-feedback) — scalar OU LQG control.
#
# Mirror of `version_1/tools/bench_smc_control_ou.py` and
# `bench_smc_control_ou_state_feedback.py`. Two SMC²-as-controller runs:
#   A2: search over θ = u (20-D raw pulse schedule, identity decoder)
#   A3: search over θ = K (20-D state-feedback gain vector)
# Each is compared to the analytical Riccati closed form.
#
# Run from `version_1_Julia/`:
#   julia --project=. tools/bench_smc_control_ou.jl

using Random
using Statistics: mean
using Plots

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "scalar_ou_lqg"))

using SMC2FC
using ScalarOULQG: build_open_loop_cost_fn, build_state_feedback_cost_fn,
                    lqr_riccati, lqr_optimal_cost,
                    lqg_optimal_cost_mc, open_loop_zero_control_cost_mc,
                    Control

T_steps = Control.TRUTH.T

# ── Riccati / LQG / open-loop baselines ────────────────────────────────────
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

println("LQR (perfect-info) cost: ", round(lqr_perfect, digits=4))
println("LQG (MC, partial obs)  : ", round(lqg_mc,    digits=4))
println("Open-loop (u=0) cost   : ", round(ol_mc,     digits=4))

# ── A2: Open-loop schedule SMC² ────────────────────────────────────────────
println("\n--- A2: Open-loop u-schedule (raw 20-D pulse) ---")

cost_open_loop_eval = build_open_loop_cost_fn(n_inner = 2000, seed = 99)
cost_open_loop_smc  = build_open_loop_cost_fn(n_inner = 64,   seed = 42)

spec_open = ControlSpec(
    name              = "scalar_ou_open_loop",
    version           = "v1_julia",
    dt                = Control.TRUTH.dt,
    n_steps           = T_steps,
    initial_state     = [Control.TRUTH.x0_mean],
    theta_dim         = T_steps,
    sigma_prior       = 1.5,
    cost_fn           = cost_open_loop_smc,
    schedule_from_theta = identity,            # raw schedule
)

cfg_outer = SMCConfig(
    n_smc_particles  = 64,
    target_ess_frac  = 0.5,
    num_mcmc_steps   = 5,
    max_lambda_inc   = 0.20,
    hmc_step_size    = 0.05,
    hmc_num_leapfrog = 6,
)

result_A2 = run_tempered_smc_loop(spec_open, cfg_outer, MersenneTwister(2026);
                                    calib_n = 128, target_nats = 6.0)
u_post_A2 = vec(mean(result_A2.particles; dims = 1))
smc_cost_A2 = cost_open_loop_eval(u_post_A2)
ratio_A2 = smc_cost_A2 / ol_mc
println("SMC² mean cost: ", round(smc_cost_A2, digits=4))
println("SMC² / open-loop cost ratio: ", round(ratio_A2, digits=4),
        "  (gate: 0.95 ≤ ratio ≤ 1.10  →  ", 0.95 ≤ ratio_A2 ≤ 1.10 ? "PASS" : "FAIL", ")")

# ── A3: State-feedback gain SMC² ───────────────────────────────────────────
println("\n--- A3: State-feedback gain vector K ---")

cost_sf_eval = build_state_feedback_cost_fn(n_inner = 2000, seed = 99)
cost_sf_smc  = build_state_feedback_cost_fn(n_inner = 64,   seed = 42)

spec_sf = ControlSpec(
    name              = "scalar_ou_state_feedback",
    version           = "v1_julia",
    dt                = Control.TRUTH.dt,
    n_steps           = T_steps,
    initial_state     = [Control.TRUTH.x0_mean],
    theta_dim         = T_steps,
    sigma_prior       = 1.5,
    prior_mean        = fill(2.0, T_steps),    # bias toward Riccati range
    cost_fn           = cost_sf_smc,
    schedule_from_theta = identity,
)

result_A3 = run_tempered_smc_loop(spec_sf, cfg_outer, MersenneTwister(2026);
                                    calib_n = 128, target_nats = 6.0)
K_post = vec(mean(result_A3.particles; dims = 1))
smc_cost_A3 = cost_sf_eval(K_post)
ratio_A3_lqg = smc_cost_A3 / lqg_mc
ratio_A3_ol  = smc_cost_A3 / ol_mc
K_rms = sqrt(mean((K_post .- riccati.gains).^2) / mean(riccati.gains.^2))
println("SMC² mean cost: ", round(smc_cost_A3, digits=4))
println("SMC² / LQG  cost ratio: ", round(ratio_A3_lqg, digits=4),
        "  (gate: 0.95 ≤ ratio ≤ 1.10  →  ", 0.95 ≤ ratio_A3_lqg ≤ 1.10 ? "PASS" : "FAIL", ")")
println("K RMS error vs Riccati: ", round(K_rms, digits=4),
        "  (gate: < 0.25  →  ", K_rms < 0.25 ? "PASS" : "FAIL", ")")

# ── Diagnostic plot ─────────────────────────────────────────────────────────
default(size = (1200, 800), fontfamily = "Helvetica")

# A2 plot
p1 = plot(0:T_steps-1, u_post_A2, lw = 2, color = :steelblue,
          label = "SMC²-mean u (A2)", title  = "A2 — open-loop schedule",
          xlabel = "step k", ylabel = "u_k")
hline!(p1, [0.0], color = :gray, lw = 0.5, label = nothing)

p2 = bar(["LQR perfect", "LQG MC", "Open-loop", "SMC²-A2"],
         [lqr_perfect, lqg_mc, ol_mc, smc_cost_A2],
         title = "A2 — cost comparison",
         legend = false, ylabel = "expected cost",
         color = [:gray, :purple, :black, :steelblue])

# A3 plot
p3 = plot(0:T_steps-1, riccati.gains, lw = 3, color = :red,
          label = "Riccati K_k (analytical)",
          title = "A3 — state-feedback gain",
          xlabel = "step k", ylabel = "K_k")
plot!(p3, 0:T_steps-1, K_post, lw = 2, color = :steelblue,
       label = "SMC²-mean K (A3)", marker = :circle, ms = 4)

p4 = bar(["LQR perfect", "LQG MC", "Open-loop", "SMC²-A3"],
         [lqr_perfect, lqg_mc, ol_mc, smc_cost_A3],
         title = "A3 — cost comparison",
         legend = false, ylabel = "expected cost",
         color = [:gray, :purple, :black, :steelblue])

plot(p1, p2, p3, p4, layout = (2, 2),
      plot_title = "Stage A2 + A3 — scalar OU LQG control (Julia)")

out_path = joinpath(ROOT, "outputs", "scalar_ou_lqg",
                     "A2_A3_control_diagnostic_julia.png")
mkpath(dirname(out_path))
savefig(out_path)
println("\nSaved: $out_path")

println()
println("Summary")
println("  A2 ratio (SMC²/OL): $(round(ratio_A2,    digits=3))")
println("  A3 ratio (SMC²/LQG): $(round(ratio_A3_lqg, digits=3))")
println("  A3 K RMS error    : $(round(K_rms,        digits=3))")
