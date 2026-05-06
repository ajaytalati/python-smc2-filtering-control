# Stage B1 (filter) + B2 (control) — bistable_controlled.
#
# Mirror of the Python `bench_smc_filter_bistable.py` and
# `bench_smc_control_bistable.py`. One bench produces:
#   B1 panel: synthetic trajectory + obs + (optional) PF filter overlay
#   B2 panel: SMC²-as-controller schedule, basin-transition stats
#
# Run from `version_1_Julia/`:
#   julia --project=. tools/bench_bistable.jl

using Random
using Statistics: mean
using Plots

const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "bistable_controlled"))

using SMC2FC
using BistableControlled: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A,
                            simulate_em, build_basin_cost_fn

println("="^70)
println("Stage B1 — bistable filter (truth-params trajectory + obs)")
println("="^70)
println("Truth: α=$(PARAM_SET_A.alpha), a=$(PARAM_SET_A.a), σ_x=$(PARAM_SET_A.sigma_x), γ=$(PARAM_SET_A.gamma), σ_u=$(PARAM_SET_A.sigma_u), σ_obs=$(PARAM_SET_A.sigma_obs)")
println("Schedule: T_intervention=$(EXOGENOUS_A.T_intervention) h, u_on=$(EXOGENOUS_A.u_on) (u_c ≈ 0.385)")
T_steps = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))
println("Grid:  T_total=$(EXOGENOUS_A.T_total) h, dt=$(round(EXOGENOUS_A.dt*60, digits=1)) min, $T_steps steps")

# ── Simulate one trajectory ─────────────────────────────────────────────────
data = simulate_em(seed = 7)
println("\nTrajectory range:")
println("  x: [$(round(minimum(data.trajectory[:,1]), digits=3)), $(round(maximum(data.trajectory[:,1]), digits=3))]")
println("  u: [$(round(minimum(data.trajectory[:,2]), digits=3)), $(round(maximum(data.trajectory[:,2]), digits=3))]")

x_final = data.trajectory[end, 1]
transitioned = x_final > 0
println("Final x = $(round(x_final, digits=3)) → ",
        transitioned ? "TRANSITIONED to healthy well ✓" :
                       "stayed in unhealthy well")

# ── B2: SMC²-as-controller searches over u_target schedule ──────────────────
# Cost = (1 − basin transition rate) + λ‖u‖²
println("\n" * "="^70)
println("Stage B2 — SMC²-as-controller over u_target schedule")
println("="^70)

# Search over a coarse 12-anchor RBF schedule for tractability — same shape
# the Python SMC² controller uses on this model.
const N_ANCHORS = 12
rbf = RBFBasis(T_steps, EXOGENOUS_A.dt, N_ANCHORS;
               width_factor = 1.0, output = SigmoidOutput())   # u_target ∈ [0, 1]
Φ = design_matrix(rbf)

# Cost evaluator: smaller `n_inner` for the SMC²-side search, larger for eval.
cost_smc  = build_basin_cost_fn(n_inner = 32,  seed = 42, n_steps = T_steps, lambda_u = 0.05)
cost_eval = build_basin_cost_fn(n_inner = 256, seed = 99, n_steps = T_steps, lambda_u = 0.05)

# Schedule decoder: RBF coefficients θ → schedule grid (T_steps,)
function decode(θ)
    return schedule_from_theta(rbf, θ; Φ = Φ)
end
function cost_via_decoder(θ)
    return cost_smc(decode(θ))
end
function cost_eval_via_decoder(θ)
    return cost_eval(decode(θ))
end

spec = ControlSpec(
    name              = "bistable_b2",
    version           = "v1_julia",
    dt                = EXOGENOUS_A.dt,
    n_steps           = T_steps,
    initial_state     = [INIT_STATE_A.x_0, INIT_STATE_A.u_0],
    theta_dim         = N_ANCHORS,
    sigma_prior       = 1.5,
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
println("SMC² done in $(round(result.elapsed, digits=2)) s, $(result.n_temp) tempering levels.")

θ_post = vec(mean(result.particles; dims = 1))
schedule = decode(θ_post)
final_cost = cost_eval_via_decoder(θ_post)

# Quick basin-transition stat under this schedule
function basin_transition_rate(u_schedule; n_trials = 500, seed = 0)
    α, a, γ = PARAM_SET_A.alpha, PARAM_SET_A.a, PARAM_SET_A.gamma
    sx_sd = sqrt(2.0 * PARAM_SET_A.sigma_x)
    su_sd = sqrt(2.0 * PARAM_SET_A.sigma_u)
    dt = EXOGENOUS_A.dt
    T_eff = length(u_schedule)
    transitioned = 0
    rng = MersenneTwister(seed)
    for _ in 1:n_trials
        x = INIT_STATE_A.x_0
        u = INIT_STATE_A.u_0
        for k in 1:T_eff
            u_tgt = u_schedule[k]
            dx = α * x * (a^2 - x^2) + u
            du = -γ * (u - u_tgt)
            x += dt * dx + sx_sd * sqrt(dt) * randn(rng)
            u += dt * du + su_sd * sqrt(dt) * randn(rng)
        end
        transitioned += x > 0 ? 1 : 0
    end
    return transitioned / n_trials
end

# Transition rate under the SMC² schedule
sm_rate = basin_transition_rate(schedule; n_trials = 500, seed = 1)
# Transition rate under the Python "default" schedule: u = 0 for t < 24, u = 0.5 after
default_sched = [t < EXOGENOUS_A.T_intervention ? 0.0 : EXOGENOUS_A.u_on
                 for t in (0:T_steps-1) .* EXOGENOUS_A.dt]
def_rate = basin_transition_rate(default_sched; n_trials = 500, seed = 1)
def_cost = cost_eval(default_sched)

println("\nTransition rate (SMC² schedule):    $(round(sm_rate * 100, digits=1)) %")
println("Transition rate (default 24h step): $(round(def_rate * 100, digits=1)) %")
println("Cost (SMC² eval):    $(round(final_cost, digits=4))")
println("Cost (default eval): $(round(def_cost, digits=4))")
gate_smc_transition = sm_rate ≥ 0.80
gate_smc_cost      = final_cost ≤ def_cost
println("Gate B2 (transition ≥ 80 %):     ", gate_smc_transition ? "PASS" : "FAIL")
println("Gate B2 (SMC² cost ≤ default):   ", gate_smc_cost ? "PASS" : "FAIL")

# ── Diagnostic plot ─────────────────────────────────────────────────────────
default(size = (1300, 800), fontfamily = "Helvetica")

# B1: trajectory + obs
p1 = plot(data.t_grid, data.trajectory[:, 1], lw = 2, color = :steelblue,
          label = "x (truth)",
          title = "B1 — bistable filter: simulated trajectory",
          xlabel = "t (h)", ylabel = "x")
plot!(p1, data.t_grid, data.trajectory[:, 2], lw = 1.5, color = :darkorange,
       label = "u")
scatter!(p1, data.t_grid, data.obs, ms = 1.5, alpha = 0.4,
          color = :black, label = "obs")
hline!(p1, [PARAM_SET_A.a, -PARAM_SET_A.a], color = :gray, ls = :dash,
        lw = 0.5, label = nothing)
vline!(p1, [EXOGENOUS_A.T_intervention], color = :red, ls = :dash, lw = 1.2,
        label = "T_intervention")

# u_target (default): step at T=24
p2 = plot(data.t_grid, data.u_target, lw = 2, color = :red, ls = :dash,
          label = "default schedule",
          title = "B2 — SMC² u_target schedule",
          xlabel = "t (h)", ylabel = "u_target")
plot!(p2, data.t_grid, schedule, lw = 2, color = :steelblue,
       label = "SMC²-mean schedule")

# Schedule comparison (a tighter inset showing shape)
p3 = plot(data.t_grid, schedule .- data.u_target, lw = 1.5,
          color = :purple, label = "SMC² − default",
          title = "B2 — schedule shape difference",
          xlabel = "t (h)", ylabel = "Δ u_target")
hline!(p3, [0.0], color = :gray, lw = 0.5, label = nothing)

# Bar of basin rates + costs
p4 = bar(["SMC² rate", "Default rate"], [sm_rate, def_rate],
          ylim = (0, 1), legend = false,
          color = [:steelblue, :red], ylabel = "transition rate",
          title  = "B2 — basin transition + cost")
annotate!(p4, [(1, sm_rate + 0.03,  text(string(round(sm_rate * 100, digits=0), " %"), 9)),
                (2, def_rate + 0.03, text(string(round(def_rate * 100, digits=0), " %"), 9))])

plot(p1, p2, p3, p4, layout = (2, 2),
      plot_title = "Stage B1+B2 — bistable_controlled (Julia)")

out_path = joinpath(ROOT, "outputs", "bistable_controlled",
                     "B1_B2_diagnostic_julia.png")
mkpath(dirname(out_path))
savefig(out_path)
println("\nSaved: $out_path")

println("\nSummary")
println("  B1 trajectory transition (single seed): ", transitioned ? "yes ✓" : "no ✗")
println("  B2 SMC² basin transition rate:           $(round(sm_rate * 100, digits=1)) %")
println("  B2 SMC² cost / default cost:             $(round(final_cost / def_cost, digits=3))")
