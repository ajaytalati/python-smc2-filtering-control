# Stage B2 — bistable_controlled SMC²-as-controller.
#
# Mirror of `version_1/tools/bench_smc_control_bistable.py` with the
# `models/bistable_controlled/control.py:diagnostic_plot` 2×2 layout:
#
#   [0,0] SMC²-derived control schedule (with u_c, default u_on hlines)
#   [0,1] Default (hand-coded) schedule (with u_c hline + T_intervention vline)
#   [1,0] x(t) under SMC²: n_traj sample trajectories
#   [1,1] x(t) under default: n_traj sample trajectories
#
# Run from version_1_Julia/:
#   julia --project=. tools/bench_smc_control_bistable.jl

using Random
using Statistics: mean, std
using Printf
using Plots
using Plots.PlotMeasures: mm
gr()

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))

using SMC2FC
include(joinpath(ROOT, "utils", "diagnostics.jl"))
using .Diagnostics: STEELBLUE

using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A,
                            simulate_em, build_basin_cost_fn

# ── Setup ──────────────────────────────────────────────────────────────────
println("="^70)
println("Stage B2 — bistable SMC²-as-controller")
println("="^70)

T_steps = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))

# u_c = 2·α·a³ / (3·√3) — saddle-node bifurcation
const U_CRIT = 2.0 * PARAM_SET_A.alpha * PARAM_SET_A.a^3 / (3.0 * sqrt(3.0))
println(@sprintf("u_c = %.3f, default u_on = %.3f, T_intervention = %.1f h",
                  U_CRIT, EXOGENOUS_A.u_on, EXOGENOUS_A.T_intervention))

# RBF basis with sigmoid output (u_target ∈ [0, 1])
const N_ANCHORS = 12
rbf = RBFBasis(T_steps, EXOGENOUS_A.dt, N_ANCHORS;
               width_factor = 1.0, output = SigmoidOutput())
Φ = design_matrix(rbf)

cost_smc  = build_basin_cost_fn(n_inner = 32,  seed = 42, n_steps = T_steps, lambda_u = 0.05)
cost_eval = build_basin_cost_fn(n_inner = 256, seed = 99, n_steps = T_steps, lambda_u = 0.05)

decode(θ)               = schedule_from_theta(rbf, θ; Φ = Φ)
cost_via_decoder(θ)     = cost_smc(decode(θ))
cost_eval_via_decoder(θ) = cost_eval(decode(θ))

spec = ControlSpec(
    name              = "bistable_b2", version = "v1_julia",
    dt                = EXOGENOUS_A.dt, n_steps = T_steps,
    initial_state     = [INIT_STATE_A.x_0, INIT_STATE_A.u_0],
    theta_dim         = N_ANCHORS, sigma_prior = 1.5,
    cost_fn           = cost_via_decoder,
    schedule_from_theta = decode,
)

cfg_outer = SMCConfig(
    n_smc_particles  = 64,
    target_ess_frac  = 0.5,
    num_mcmc_steps   = 5,
    max_lambda_inc   = 0.20,
    hmc_step_size    = 0.05,
    hmc_num_leapfrog = 6,
)

println("Calibrating β_max + running outer SMC² ...")
result = run_tempered_smc_loop(spec, cfg_outer, MersenneTwister(2026);
                                 calib_n = 128, target_nats = 6.0)
println(@sprintf("SMC² done in %.2f s, %d tempering levels.",
                  result.elapsed, result.n_temp))

θ_post = vec(mean(result.particles; dims = 1))
smc_schedule = decode(θ_post)

# Default schedule: u = 0 for t < T_intervention, u_on after.
t_grid = collect(0:T_steps-1) .* EXOGENOUS_A.dt
default_schedule = [t < EXOGENOUS_A.T_intervention ? 0.0 : EXOGENOUS_A.u_on
                     for t in t_grid]

# ── Sample n_traj closed-loop x trajectories under each schedule ───────────
function sample_trajectory(schedule::AbstractVector; seed::Integer = 0)
    α    = PARAM_SET_A.alpha
    a    = PARAM_SET_A.a
    γ    = PARAM_SET_A.gamma
    sx_sd = sqrt(2.0 * PARAM_SET_A.sigma_x)
    su_sd = sqrt(2.0 * PARAM_SET_A.sigma_u)
    dt   = EXOGENOUS_A.dt
    rng  = MersenneTwister(seed)

    x = zeros(Float64, T_steps); u = zeros(Float64, T_steps)
    x[1] = INIT_STATE_A.x_0; u[1] = INIT_STATE_A.u_0
    for k in 1:T_steps-1
        u_tgt = schedule[k]
        dx = α * x[k] * (a^2 - x[k]^2) + u[k]
        du = -γ * (u[k] - u_tgt)
        x[k+1] = x[k] + dt * dx + sx_sd * sqrt(dt) * randn(rng)
        u[k+1] = u[k] + dt * du + su_sd * sqrt(dt) * randn(rng)
    end
    return x, u
end

const N_TRAJ = 5
traj_smc     = [sample_trajectory(smc_schedule;     seed = s) for s in 1:N_TRAJ]
traj_default = [sample_trajectory(default_schedule; seed = s) for s in 1:N_TRAJ]

# Cost + transition stats for the summary
function basin_transition_rate(schedule; n = 500, seed = 0)
    rng = MersenneTwister(seed)
    α, a, γ = PARAM_SET_A.alpha, PARAM_SET_A.a, PARAM_SET_A.gamma
    sx_sd = sqrt(2.0 * PARAM_SET_A.sigma_x); su_sd = sqrt(2.0 * PARAM_SET_A.sigma_u)
    dt = EXOGENOUS_A.dt
    cnt = 0
    for _ in 1:n
        x = INIT_STATE_A.x_0; u = INIT_STATE_A.u_0
        for k in 1:T_steps
            u_tgt = schedule[k]
            x += dt * (α * x * (a^2 - x^2) + u) + sx_sd * sqrt(dt) * randn(rng)
            u += dt * (-γ * (u - u_tgt))         + su_sd * sqrt(dt) * randn(rng)
        end
        cnt += x > 0 ? 1 : 0
    end
    return cnt / n
end

rate_smc     = basin_transition_rate(smc_schedule;     n = 500, seed = 1)
rate_default = basin_transition_rate(default_schedule; n = 500, seed = 1)
cost_smc_eval = cost_eval(smc_schedule)
cost_default  = cost_eval(default_schedule)
println(@sprintf("\nTransition rate SMC² : %.0f %% (gate ≥ 80 %%)  %s",
                  rate_smc * 100,    rate_smc ≥ 0.80 ? "PASS" : "FAIL"))
println(@sprintf("Transition rate def  : %.0f %%", rate_default * 100))
println(@sprintf("SMC² cost / default  : %.3f  (gate ≤ 1.0)  %s",
                  cost_smc_eval / cost_default,
                  cost_smc_eval ≤ cost_default ? "PASS" : "FAIL"))

# ── B2 plot — 2×2 panel-for-panel match of Python ──────────────────────────
default(size = (1400, 800), fontfamily = "Helvetica",
        legendfontsize = 8, titlefontsize = 11, framestyle = :box,
        grid = true, gridalpha = 0.3,
        left_margin = 8mm, right_margin = 4mm,
        bottom_margin = 6mm, top_margin = 4mm)

DARKRED = RGB(0x8b/255, 0x00/255, 0x00/255)
RED     = RGB(0xd6/255, 0x27/255, 0x28/255)
GREEN   = RGB(0x2c/255, 0xa0/255, 0x2c/255)
GRAY    = RGB(0x7f/255, 0x7f/255, 0x7f/255)

# [0,0] SMC²-derived schedule
p11 = plot(t_grid, smc_schedule; lw = 2, color = STEELBLUE,
            label = "SMC² posterior-mean u_target(t)",
            xlabel = "time (h)", ylabel = "u_target",
            title = "SMC²-derived control schedule")
hline!(p11, [U_CRIT];           color = RED,  ls = :dot, alpha = 0.7,
        label = @sprintf("u_c = %.3f", U_CRIT))
hline!(p11, [EXOGENOUS_A.u_on]; color = GRAY, ls = :dot, alpha = 0.7,
        label = @sprintf("default u_on = %.1f", EXOGENOUS_A.u_on))

# [0,1] Default schedule
p12 = plot(t_grid, default_schedule; lw = 2, color = DARKRED,
            label = "default u_target(t)",
            xlabel = "time (h)", ylabel = "u_target",
            title = "Default (hand-coded) schedule")
hline!(p12, [U_CRIT]; color = RED, ls = :dot, alpha = 0.7,
        label = @sprintf("u_c = %.3f", U_CRIT))
vline!(p12, [EXOGENOUS_A.T_intervention]; color = GRAY, ls = :dash, alpha = 0.5,
        label = @sprintf("T_intervention = %.0f h", EXOGENOUS_A.T_intervention))

# [1,0] x(t) under SMC²: n_traj sample trajectories
p21 = plot(xlabel = "time (h)", ylabel = "x (health)",
            title  = @sprintf("x(t) under SMC² schedule (transition %.0f %%)",
                              rate_smc * 100))
for (i, (x_i, _)) in enumerate(traj_smc)
    plot!(p21, t_grid, x_i; alpha = 0.6, lw = 1, label = nothing)
end
hline!(p21, [-1.0]; color = RED,   ls = :dot, alpha = 0.5, label = "x=-1 (sick)")
hline!(p21, [+1.0]; color = GREEN, ls = :dot, alpha = 0.5, label = "x=+1 (well)")

# [1,1] x(t) under default: n_traj sample trajectories
p22 = plot(xlabel = "time (h)", ylabel = "x (health)",
            title  = @sprintf("x(t) under default schedule (transition %.0f %%)",
                              rate_default * 100))
for (i, (x_i, _)) in enumerate(traj_default)
    plot!(p22, t_grid, x_i; alpha = 0.6, lw = 1, label = nothing)
end
hline!(p22, [-1.0]; color = RED,   ls = :dot, alpha = 0.5, label = "x=-1 (sick)")
hline!(p22, [+1.0]; color = GREEN, ls = :dot, alpha = 0.5, label = "x=+1 (well)")

P = plot(p11, p12, p21, p22; layout = (2, 2), size = (1400, 800),
          plot_title = @sprintf("Bistable controlled — %d tempering levels in %.1f s on CPU",
                                 result.n_temp, result.elapsed),
          plot_titlefontsize = 12)

out_path = joinpath(ROOT, "outputs", "bistable_controlled", "B2_control_diagnostic.png")
mkpath(dirname(out_path))
savefig(P, out_path)
println("\nSaved: $out_path")

# Stash numbers for RESULT.md
open(joinpath(ROOT, "outputs", "bistable_controlled", "_results_B2.txt"), "w") do io
    println(io, "n_temp=", result.n_temp)
    println(io, "elapsed=", result.elapsed)
    println(io, "rate_smc=", rate_smc)
    println(io, "rate_default=", rate_default)
    println(io, "cost_smc=", cost_smc_eval)
    println(io, "cost_default=", cost_default)
    println(io, "u_crit=", U_CRIT)
end
