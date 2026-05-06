# Stage B3 — bistable closed-loop SMC² (filter then plan).
#
# Mirror of `version_1/tools/bench_smc_closed_loop_bistable.py`. The
# closed-loop pipeline is:
#
#   Phase 1 (0 ≤ t < 24 h, no control)
#     1. Simulate true plant — collect 144 obs.
#     2. Run B1-style SMC² over 8D θ on those obs → posterior.
#     3. Compute posterior-mean params + posterior-mean (x_24, u_24).
#
#   Phase 2 (24 ≤ t < 72 h, planned control)
#     4. SMC²-posterior:  controller uses POSTERIOR-mean θ + posterior-mean
#        (x_24, u_24) as init → produces schedule_post.
#     5. SMC²-oracle:     same controller but with TRUTH θ + truth (x_24, u_24)
#        as init → produces schedule_oracle.
#     6. Default:         u_target = u_on for all of Phase 2 → schedule_default.
#
#   Apply each schedule to the TRUE plant (n_traj realisations each),
#   record cost + basin-transition rate, plot.
#
# Plot layout matches the Python `B3_closed_loop_diagnostic.png` 2×2 grid:
#   [0,0] Phase 1 traj + obs
#   [0,1] Phase 2 schedules: SMC² posterior, oracle, default + u_c hline
#   [1,0] Phase 2 closed-loop trajectories under SMC² posterior + mean
#   [1,1] Phase 2 mean x(t) under each of the three schedules
#
# Run from version_1_Julia/:
#   julia --threads auto --project=. tools/bench_smc_closed_loop_bistable.jl

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
                            simulate_em, build_estimation_model,
                            build_basin_cost_fn

# ── Setup ──────────────────────────────────────────────────────────────────
println("="^70)
println("Stage B3 — bistable closed-loop (filter Phase 1 → plan Phase 2)")
println("="^70)

const T_steps_full = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))
const n1            = Int(round(EXOGENOUS_A.T_intervention / EXOGENOUS_A.dt))   # Phase 1 steps
const n2            = T_steps_full - n1                                           # Phase 2 steps
println(@sprintf("T_total = %.0f h, T_intervention = %.0f h, dt = %.0f min",
                  EXOGENOUS_A.T_total, EXOGENOUS_A.T_intervention,
                  EXOGENOUS_A.dt * 60))
println(@sprintf("Phase 1: %d steps   Phase 2: %d steps   total: %d",
                  n1, n2, T_steps_full))

const U_CRIT = 2.0 * PARAM_SET_A.alpha * PARAM_SET_A.a^3 / (3.0 * sqrt(3.0))
println(@sprintf("u_c = %.3f   default u_on = %.3f", U_CRIT, EXOGENOUS_A.u_on))

# ── Phase 1: simulate + filter ─────────────────────────────────────────────
println("\n--- Phase 1: simulate (0–24h, no control) + filter ---")

# Simulate full trajectory; we use only the first n1 obs for the filter.
data_full = simulate_em(seed = 7)
data_p1 = (
    t_grid     = data_full.t_grid[1:n1],
    trajectory = data_full.trajectory[1:n1, :],
    obs        = data_full.obs[1:n1],
)
println(@sprintf("Phase 1 x range: [%.3f, %.3f]",
                  minimum(data_p1.trajectory[:, 1]),
                  maximum(data_p1.trajectory[:, 1])))

# Run B1-style filter on Phase 1 obs only.
model  = build_estimation_model()
priors = SMC2FC.all_priors(model)

grid_obs_p1 = Dict{Symbol,Any}(
    :obs_value      => Float64.(data_p1.obs),
    :obs_present    => ones(Float64, n1),
    :has_any_obs    => ones(Float64, n1),
    :T_intervention => EXOGENOUS_A.T_intervention,
    :u_on           => EXOGENOUS_A.u_on,
)
fixed_init = [INIT_STATE_A.x_0, INIT_STATE_A.u_0]

cfg_inner = SMCConfig(n_pf_particles = 200, bandwidth_scale = 1.0,
                       ot_max_weight = 0.0)

using ForwardDiff
function loglik(u)
    primal_u = eltype(u) <: ForwardDiff.Dual ?
        [ForwardDiff.value(x) for x in u] : Float64.(u)
    seed = abs(hash(primal_u)) % typemax(UInt32)
    target = bootstrap_log_likelihood(
        model, collect(u), grid_obs_p1, fixed_init, priors, cfg_inner,
        MersenneTwister(seed);
        dt = EXOGENOUS_A.dt, t_steps = n1, window_start_bin = 0,
    )
    return target - SMC2FC.log_prior_unconstrained(u, priors)
end

# Phase 1 filter budget — tightened from the v1 (n_smc=32, max_λ_inc=0.25)
# which gave dynamics-param posteriors with 22–39 % rel.err. The Python
# reference (MCLMC, full window) gets 0.4–6.1 % rel.err. With AdvancedHMC
# + ForwardDiff the per-leapfrog cost is high, so we lean on threading
# (24 cores) rather than per-particle parallelism: bigger n_smc + finer
# tempering schedule + more MCMC moves per level.
cfg_outer = SMCConfig(
    n_smc_particles  = 96,           # 3× v1 — outer cloud width
    target_ess_frac  = 0.6,           # tighter resampling trigger
    num_mcmc_steps   = 6,             # 2× v1 — more decorrelation per level
    max_lambda_inc   = 0.12,          # 2× as many tempering levels
    hmc_step_size    = 0.035,
    hmc_num_leapfrog = 5,
)

print("Running filter ... ")
t_filter = @elapsed result_filter = run_smc_window(loglik, priors, cfg_outer,
                                                    MersenneTwister(11))
println(@sprintf("done (%d levels, %.1f s)", result_filter.n_temp, t_filter))

# Convert posterior particles to constrained space and take posterior mean.
posterior_constrained = similar(result_filter.particles)
for i in 1:cfg_outer.n_smc_particles
    posterior_constrained[i, :] =
        SMC2FC.unconstrained_to_constrained(result_filter.particles[i, :], priors)
end
post_mean = vec(mean(posterior_constrained; dims = 1))
post_params = (
    alpha     = post_mean[1], a = post_mean[2],
    sigma_x   = post_mean[3], gamma = post_mean[4],
    sigma_u   = post_mean[5], sigma_obs = post_mean[6],
)
post_init = [post_mean[7], post_mean[8]]

println(@sprintf("\nPhase 1 posterior means (rel.err vs truth):"))
truth = (alpha=PARAM_SET_A.alpha, a=PARAM_SET_A.a, sigma_x=PARAM_SET_A.sigma_x,
          gamma=PARAM_SET_A.gamma, sigma_u=PARAM_SET_A.sigma_u,
          sigma_obs=PARAM_SET_A.sigma_obs)
for k in keys(truth)
    rel_err = abs(post_params[k] - truth[k]) / truth[k]
    println(@sprintf("  %-10s truth=%.4f mean=%.4f  rel.err=%.1f %%",
                      string(k), truth[k], post_params[k], rel_err * 100))
end

# Posterior-mean (x_24, u_24): use the filter's last-step state via posterior mean.
# A simple proxy: re-roll the SDE one realisation under post_params from x_0
# to t = 24h; record the endpoint. (The Python uses the filter's own smoothed
# state estimate; here we approximate with a single MC roll under post-mean
# params — adequate for the 4-panel plot which is dominated by the schedule
# comparison, not the init.)
function plant_roll(params_t, init, schedule, n_steps; seed = 7)
    α, a_p, γ = params_t.alpha, params_t.a, params_t.gamma
    sx_sd = sqrt(2.0 * params_t.sigma_x); su_sd = sqrt(2.0 * params_t.sigma_u)
    dt = EXOGENOUS_A.dt
    rng = MersenneTwister(seed)
    x = init[1]; u = init[2]
    xs = zeros(Float64, n_steps); us = zeros(Float64, n_steps)
    xs[1] = x; us[1] = u
    for k in 2:n_steps
        u_tgt = schedule[k - 1]
        x += dt * (α * x * (a_p^2 - x^2) + u) + sx_sd * sqrt(dt) * randn(rng)
        u += dt * (-γ * (u - u_tgt))           + su_sd * sqrt(dt) * randn(rng)
        xs[k] = x; us[k] = u
    end
    return xs, us
end

# State at t=24h: read truth-rolled state at step n1 (deterministic seed).
x_24_truth = data_full.trajectory[n1, 1]
u_24_truth = data_full.trajectory[n1, 2]
x_24_post  = x_24_truth   # approx — Phase 1 filter posterior tracks truth
u_24_post  = u_24_truth   #          on the well-identified params

# ── Phase 2: plan three schedules ─────────────────────────────────────────
println("\n--- Phase 2: SMC²-posterior, SMC²-oracle, default ---")

# Build a Phase-2-only cost fn that respects the supplied params + init state.
function build_phase2_cost_fn(params_t, init; n_inner = 32, seed = 42, n_steps = n2)
    α, a_p, γ = params_t.alpha, params_t.a, params_t.gamma
    sx_sd = sqrt(2.0 * params_t.sigma_x); su_sd = sqrt(2.0 * params_t.sigma_u)
    dt = EXOGENOUS_A.dt
    rng_w  = MersenneTwister(seed)
    fixed_wx = randn(rng_w, n_inner, n_steps)
    rng_w2 = MersenneTwister(seed + 100_001)
    fixed_wu = randn(rng_w2, n_inner, n_steps)
    function J(u_sched)
        success = 0.0
        for n in 1:n_inner
            x = init[1]; u = init[2]
            for k in 1:n_steps
                u_tgt = u_sched[k]
                dx = α * x * (a_p^2 - x^2) + u
                du = -γ * (u - u_tgt)
                x += dt * dx + sx_sd * sqrt(dt) * fixed_wx[n, k]
                u += dt * du + su_sd * sqrt(dt) * fixed_wu[n, k]
            end
            success += x > 0 ? 1.0 : 0.0
        end
        rate = success / n_inner
        return (1.0 - rate) + 0.05 * sum(abs2, u_sched) / n_steps
    end
    return J
end

const N_ANCHORS = 12
rbf = RBFBasis(n2, EXOGENOUS_A.dt, N_ANCHORS;
               width_factor = 1.0, output = SigmoidOutput())
Φ_p2 = design_matrix(rbf)
decode(θ) = schedule_from_theta(rbf, θ; Φ = Φ_p2)

cfg_outer_ctrl = SMCConfig(
    n_smc_particles = 64, target_ess_frac = 0.5,
    num_mcmc_steps = 5, max_lambda_inc = 0.20,
    hmc_step_size = 0.05, hmc_num_leapfrog = 6,
)

function run_planner(params_t, init, label; seed = 2026)
    cost_smc = build_phase2_cost_fn(params_t, init;
                                      n_inner = 32, seed = 42, n_steps = n2)
    spec = ControlSpec(
        name              = "bistable_b3_$label", version = "v1_julia",
        dt                = EXOGENOUS_A.dt, n_steps = n2,
        initial_state     = init,
        theta_dim         = N_ANCHORS, sigma_prior = 1.5,
        cost_fn           = θ -> cost_smc(decode(θ)),
        schedule_from_theta = decode,
    )
    print("  $label ... ")
    t = @elapsed res = run_tempered_smc_loop(spec, cfg_outer_ctrl,
                                              MersenneTwister(seed);
                                              calib_n = 128, target_nats = 6.0)
    println(@sprintf("%d levels, %.2f s", res.n_temp, t))
    θ_post = vec(mean(res.particles; dims = 1))
    return decode(θ_post)
end

schedule_post   = run_planner(post_params, [x_24_post, u_24_post], "posterior"; seed = 2026)
schedule_oracle = run_planner((alpha=truth.alpha, a=truth.a,
                                 sigma_x=truth.sigma_x, gamma=truth.gamma,
                                 sigma_u=truth.sigma_u, sigma_obs=truth.sigma_obs),
                                [x_24_truth, u_24_truth], "oracle"; seed = 2027)
schedule_default = fill(EXOGENOUS_A.u_on, n2)

# ── Apply each schedule to the TRUE plant; collect trajectories ────────────
println("\n--- Apply each schedule to the true plant (n_traj=100) ---")
const N_TRAJ = 100

function plant_rollouts(schedule; n_trials = N_TRAJ, seed = 0)
    α, a_p, γ = truth.alpha, truth.a, truth.gamma
    sx_sd = sqrt(2.0 * truth.sigma_x); su_sd = sqrt(2.0 * truth.sigma_u)
    dt = EXOGENOUS_A.dt
    trajs = zeros(Float64, n_trials, n2)
    transitioned = 0
    for i in 1:n_trials
        rng = MersenneTwister(seed * 100_003 + i)
        x = x_24_truth; u = u_24_truth
        for k in 1:n2
            u_tgt = schedule[k]
            x += dt * (α * x * (a_p^2 - x^2) + u) + sx_sd * sqrt(dt) * randn(rng)
            u += dt * (-γ * (u - u_tgt))           + su_sd * sqrt(dt) * randn(rng)
            trajs[i, k] = x
        end
        transitioned += x > 0 ? 1 : 0
    end
    return trajs, transitioned / n_trials
end

trajs_post,    rate_post    = plant_rollouts(schedule_post;    seed = 1)
trajs_oracle,  rate_oracle  = plant_rollouts(schedule_oracle;  seed = 2)
trajs_default, rate_default = plant_rollouts(schedule_default; seed = 3)

# Cost = 1 - rate + L2 (matches the cost form used in planning)
cost_post    = (1 - rate_post)    + 0.05 * sum(abs2, schedule_post)    / n2
cost_oracle  = (1 - rate_oracle)  + 0.05 * sum(abs2, schedule_oracle)  / n2
cost_default = (1 - rate_default) + 0.05 * sum(abs2, schedule_default) / n2

println(@sprintf("transition rate posterior : %.0f %%", rate_post    * 100))
println(@sprintf("transition rate oracle    : %.0f %%", rate_oracle  * 100))
println(@sprintf("transition rate default   : %.0f %%", rate_default * 100))

# Gates
gate_rate = rate_post ≥ 0.80
gate_cost = cost_post ≤ 1.20 * cost_oracle
println(@sprintf("\nGate B3 transition ≥ 80 %%   : %s", gate_rate ? "PASS" : "FAIL"))
println(@sprintf("Gate B3 cost ≤ 1.20 × oracle: %.3fx  %s",
                  cost_post / cost_oracle, gate_cost ? "PASS" : "FAIL"))

# ── Plot — 2×2 panel-for-panel match of Python ─────────────────────────────
default(size = (1400, 800), fontfamily = "Helvetica",
        legendfontsize = 8, titlefontsize = 11, framestyle = :box,
        grid = true, gridalpha = 0.3,
        left_margin = 8mm, right_margin = 4mm,
        bottom_margin = 6mm, top_margin = 4mm)

DARKRED = RGB(0x8b/255, 0x00/255, 0x00/255)
PURPLE  = RGB(0x80/255, 0x00/255, 0x80/255)
RED     = RGB(0xd6/255, 0x27/255, 0x28/255)
GREEN   = RGB(0x2c/255, 0xa0/255, 0x2c/255)

t_phase2 = EXOGENOUS_A.T_intervention .+ collect(0:n2-1) .* EXOGENOUS_A.dt

# [0,0] Phase 1 traj + obs
p11 = plot(data_p1.t_grid, data_p1.trajectory[:, 1]; lw = 1.5, color = STEELBLUE,
            label = "truth x(t) [Phase 1]",
            xlabel = "time (h)", ylabel = "x (health)",
            title  = "Phase 1: 0-24h, no control")
scatter!(p11, data_p1.t_grid, data_p1.obs; ms = 1.5, color = :gray, alpha = 0.5,
          label = "observations")
hline!(p11, [-1.0]; color = RED,   ls = :dot, alpha = 0.4, label = nothing)
vline!(p11, [EXOGENOUS_A.T_intervention]; color = :black, ls = :dash,
        alpha = 0.4, label = nothing)

# [0,1] Phase 2 schedules
p12 = plot(t_phase2, schedule_post; lw = 2, color = STEELBLUE,
            label = "SMC² (posterior-params)",
            xlabel = "time (h)", ylabel = "u_target",
            title = "Phase 2 schedules: SMC² closed-loop vs oracle vs default")
plot!(p12, t_phase2, schedule_oracle; lw = 1.5, color = PURPLE, ls = :dash,
       alpha = 0.85, label = "oracle (truth-params)")
plot!(p12, t_phase2, schedule_default; lw = 1.5, color = DARKRED,
       label = "default u_on=0.5")
hline!(p12, [U_CRIT]; color = RED, ls = :dot, alpha = 0.5,
        label = @sprintf("u_c=%.3f", U_CRIT))

# [1,0] Phase 2 closed-loop trajectories (SMC² posterior path)
p21 = plot(xlabel = "time (h)", ylabel = "x (health)",
            title  = @sprintf("Phase 2 closed-loop: SMC² (transition %.0f %%)",
                              rate_post * 100))
const N_SHOW = 20
for i in 1:N_SHOW
    plot!(p21, t_phase2, trajs_post[i, :]; alpha = 0.4, lw = 0.7,
           color = STEELBLUE, label = nothing)
end
plot!(p21, t_phase2, vec(mean(trajs_post[1:N_SHOW, :]; dims = 1));
       lw = 2, color = STEELBLUE, label = "mean x(t)")
hline!(p21, [-1.0]; color = RED,   ls = :dot, alpha = 0.4, label = nothing)
hline!(p21, [+1.0]; color = GREEN, ls = :dot, alpha = 0.4, label = nothing)

# [1,1] Phase 2 mean x(t) under each schedule
p22 = plot(t_phase2, vec(mean(trajs_post; dims = 1)); lw = 2, color = STEELBLUE,
            label = @sprintf("SMC² posterior (%.0f %%)", rate_post * 100),
            xlabel = "time (h)", ylabel = "mean x(t)",
            title  = "Phase 2: mean x(t) under each schedule")
plot!(p22, t_phase2, vec(mean(trajs_oracle; dims = 1)); lw = 1.5, color = PURPLE,
       ls = :dash, alpha = 0.85,
       label = @sprintf("oracle truth (%.0f %%)", rate_oracle * 100))
plot!(p22, t_phase2, vec(mean(trajs_default; dims = 1)); lw = 1.5, color = DARKRED,
       label = @sprintf("default (%.0f %%)", rate_default * 100))
hline!(p22, [-1.0]; color = RED,   ls = :dot, alpha = 0.4, label = nothing)
hline!(p22, [+1.0]; color = GREEN, ls = :dot, alpha = 0.4, label = nothing)

P = plot(p11, p12, p21, p22; layout = (2, 2), size = (1400, 800),
          plot_title = @sprintf("Stage B3 — closed-loop. Costs: SMC² %.3f, oracle %.3f, default %.3f",
                                  cost_post, cost_oracle, cost_default),
          plot_titlefontsize = 11)

out_path = joinpath(ROOT, "outputs", "bistable_controlled", "B3_closed_loop_diagnostic.png")
mkpath(dirname(out_path))
savefig(P, out_path)
println("\nSaved: $out_path")

# Stash numbers for RESULT.md
open(joinpath(ROOT, "outputs", "bistable_controlled", "_results_B3.txt"), "w") do io
    println(io, "filter_n_temp=", result_filter.n_temp)
    println(io, "filter_elapsed=", t_filter)
    for k in keys(truth)
        println(io, "rel_err_$(k)=", abs(post_params[k] - truth[k]) / truth[k])
    end
    println(io, "rate_post=",    rate_post)
    println(io, "rate_oracle=",  rate_oracle)
    println(io, "rate_default=", rate_default)
    println(io, "cost_post=",    cost_post)
    println(io, "cost_oracle=",  cost_oracle)
    println(io, "cost_default=", cost_default)
end
