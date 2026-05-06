# Stage B1 — bistable filter: SMC² posterior over 8D params on 72h trajectory.
#
# Mirror of `version_1/tools/bench_smc_filter_bistable.py`. Produces
# `outputs/bistable_controlled/B1_filter_diagnostic.png` panel-for-panel
# matching the Python plot:
#
#   2×4 grid:
#     [0,0]  Synthetic trajectory + obs + intervention line
#     [0,1]  Control state (u truth) + target schedule
#     [0,2]  Posterior over alpha
#     [0,3]  Posterior over a
#     [1,0]  Posterior over sigma_x  (identifiable)
#     [1,1]  Posterior over gamma    (weakly id.)
#     [1,2]  Posterior over sigma_u  (weakly id.)
#     [1,3]  Posterior over sigma_obs (identifiable)
#
# Run from version_1_Julia/:
#   julia --project=. tools/bench_smc_filter_bistable.jl

using Random
using Statistics: mean, std, quantile
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
                            Estimation

# ── Truth + simulated data ─────────────────────────────────────────────────
const TRUTH = (
    alpha     = PARAM_SET_A.alpha, a = PARAM_SET_A.a,
    sigma_x   = PARAM_SET_A.sigma_x, gamma = PARAM_SET_A.gamma,
    sigma_u   = PARAM_SET_A.sigma_u, sigma_obs = PARAM_SET_A.sigma_obs,
    x_0       = INIT_STATE_A.x_0, u_0 = INIT_STATE_A.u_0,
)
const PARAM_NAMES = [:alpha, :a, :sigma_x, :gamma, :sigma_u, :sigma_obs]
const IDENTIFIABLE = Set([:alpha, :a, :sigma_x, :sigma_obs])

println("="^70)
println("Stage B1 — bistable filter (SMC² over 8D params)")
println("="^70)

T_steps_full = Int(round(EXOGENOUS_A.T_total / EXOGENOUS_A.dt))
data = simulate_em(seed = 7)

# ── Build EstimationModel + run outer SMC² ─────────────────────────────────
model  = build_estimation_model()
priors = SMC2FC.all_priors(model)

# Use the full 432-obs trajectory at native dt — matches the Python bench.
# Threading (24 cores) keeps the AD budget tractable.
const T_steps = T_steps_full
const dt_eff  = EXOGENOUS_A.dt

grid_obs = Dict{Symbol,Any}(
    :obs_value      => Float64.(data.obs),
    :obs_present    => ones(Float64, T_steps),
    :has_any_obs    => ones(Float64, T_steps),
    :T_intervention => EXOGENOUS_A.T_intervention,
    :u_on           => EXOGENOUS_A.u_on,
)
fixed_init = [INIT_STATE_A.x_0, INIT_STATE_A.u_0]

cfg_inner = SMCConfig(n_pf_particles = 200,
                       bandwidth_scale = 1.0,
                       ot_max_weight = 0.0)

# log-likelihood the outer SMC² consumes — log p(y|θ), no log p(u).
function loglik(u)
    pf_rng = MersenneTwister(abs(hash(ForwardDiffValueOrFloat(u))) % typemax(UInt32))
    target = bootstrap_log_likelihood(
        model, collect(u), grid_obs, fixed_init, priors, cfg_inner, pf_rng;
        dt = dt_eff, t_steps = T_steps, window_start_bin = 0,
    )
    return target - SMC2FC.log_prior_unconstrained(u, priors)
end

# Helper: extract Float64 primal from u (works for plain Float64 + ForwardDiff Dual)
ForwardDiffValueOrFloat(u::AbstractVector{<:Real}) = Float64.(u)
using ForwardDiff
ForwardDiffValueOrFloat(u::AbstractVector{<:ForwardDiff.Dual}) =
    [ForwardDiff.value(x) for x in u]

cfg_outer = SMCConfig(
    n_smc_particles  = 48,
    target_ess_frac  = 0.5,
    num_mcmc_steps   = 4,
    max_lambda_inc   = 0.20,
    hmc_step_size    = 0.04,
    hmc_num_leapfrog = 4,
)

println("Running outer SMC² over 8D θ (n_smc=$(cfg_outer.n_smc_particles), ",
        "K=$(cfg_inner.n_pf_particles), T_obs=$T_steps) ...")
t0 = time()
result = run_smc_window(loglik, priors, cfg_outer, MersenneTwister(2026))
println(@sprintf("Done. %d tempering levels, %.1f s.",
                  result.n_temp, time() - t0))

# Convert posterior particles from unconstrained → constrained per dim
# (lognormal → exp, normal → identity).
n_smc = cfg_outer.n_smc_particles
particles_constrained = similar(result.particles)
for i in 1:n_smc
    θ_c = SMC2FC.unconstrained_to_constrained(result.particles[i, :], priors)
    particles_constrained[i, :] = θ_c
end

# Map name → column index in the 8D θ vector
name_to_idx = Dict(:alpha     => 1, :a => 2, :sigma_x => 3,
                    :gamma    => 4, :sigma_u => 5, :sigma_obs => 6,
                    :x_0      => 7, :u_0 => 8)
means     = vec(mean(particles_constrained; dims=1))
ci_low    = [quantile(@view(particles_constrained[:, i]), 0.05) for i in 1:8]
ci_high   = [quantile(@view(particles_constrained[:, i]), 0.95) for i in 1:8]

# Coverage check on identifiable params
println("\nIdentifiable-parameter 90 % CI coverage:")
let id_covered = true
    for name in [:alpha, :a, :sigma_x, :sigma_obs]
        idx = name_to_idx[name]
        truth_val = TRUTH[name]
        covered = ci_low[idx] ≤ truth_val ≤ ci_high[idx]
        id_covered &= covered
        println(@sprintf("  %-10s truth=%.4f  mean=%.4f  90%%CI=[%.3f, %.3f]  %s",
                          string(name), truth_val, means[idx],
                          ci_low[idx], ci_high[idx], covered ? "✓" : "✗"))
    end
    global all_id_covered = id_covered
    println(id_covered ? "GATE PASS — all identifiable params covered" :
                          "GATE FAIL")
end

# ── Plot — exact panel-for-panel match of the Python figure ────────────────
default(size = (1600, 800), fontfamily = "Helvetica",
        legendfontsize = 7, titlefontsize = 11, framestyle = :box,
        grid = true, gridalpha = 0.3,
        left_margin = 6mm, right_margin = 4mm,
        bottom_margin = 6mm, top_margin = 4mm)

DARKORANGE = RGB(0xff/255, 0x7f/255, 0x0e/255)
RED        = RGB(0xd6/255, 0x27/255, 0x28/255)
GREEN      = RGB(0x2c/255, 0xa0/255, 0x2c/255)

t = data.t_grid
T_i = EXOGENOUS_A.T_intervention

# Top-left: trajectory + obs
p11 = plot(t, data.trajectory[:, 1]; lw = 1.5, color = STEELBLUE, alpha = 0.85,
            label = "truth x(t)",
            xlabel = "time (h)", ylabel = "x (health)",
            title  = "Synthetic trajectory + obs")
scatter!(p11, t, data.obs; ms = 1.5, color = :gray, alpha = 0.5,
          label = "y(t) = x + noise")
hline!(p11, [-1.0]; color = RED,   ls = :dot, alpha = 0.4, label = nothing)
hline!(p11, [+1.0]; color = GREEN, ls = :dot, alpha = 0.4, label = nothing)
vline!(p11, [T_i];  color = :black, ls = :dash, alpha = 0.4, label = nothing)

# Top-second: control state + target
p12 = plot(t, data.trajectory[:, 2]; lw = 1.5, color = DARKORANGE,
            label = "truth u(t)",
            xlabel = "time (h)", ylabel = "u (control)",
            title = "Control state + target schedule")
plot!(p12, t, data.u_target; lw = 1.0, color = RED, ls = :dash,
       label = "u_target(t)")

# Posterior histograms — top row: alpha, a
function posterior_hist(particles, idx, name, truth_val; weak = false)
    p = histogram(@view(particles[:, idx]); bins = 30, color = STEELBLUE,
                   alpha = 0.7, label = "",
                   xlabel = string(name), ylabel = "density",
                   title = "posterior: $name" * (weak ? " (weakly id.)" : " (identifiable)"))
    fmt = abs(truth_val) >= 0.1 ? "%.3f" : "%.4f"
    vline!(p, [truth_val]; color = GREEN, ls = :dash, lw = 2,
            label = @sprintf("truth = %.4f", truth_val))
    vline!(p, [means[idx]]; color = STEELBLUE, ls = :dot, lw = 2,
            label = @sprintf("mean = %.4f", means[idx]))
    return p
end

p13 = posterior_hist(particles_constrained, 1, "alpha", TRUTH.alpha)
p14 = posterior_hist(particles_constrained, 2, "a",     TRUTH.a)

# Bottom row
p21 = posterior_hist(particles_constrained, 3, "sigma_x", TRUTH.sigma_x)
p22 = posterior_hist(particles_constrained, 4, "gamma",   TRUTH.gamma; weak = true)
p23 = posterior_hist(particles_constrained, 5, "sigma_u", TRUTH.sigma_u; weak = true)
p24 = posterior_hist(particles_constrained, 6, "sigma_obs", TRUTH.sigma_obs)

P = plot(p11, p12, p13, p14, p21, p22, p23, p24;
          layout = (2, 4), size = (1600, 800),
          plot_title = "Stage B1 — bistable filter: SMC² posterior vs truth (72h, default supercritical schedule)",
          plot_titlefontsize = 12)

out_path = joinpath(ROOT, "outputs", "bistable_controlled", "B1_filter_diagnostic.png")
mkpath(dirname(out_path))
savefig(P, out_path)
println("\nSaved: $out_path")

# Stash numbers for RESULT.md
open(joinpath(ROOT, "outputs", "bistable_controlled", "_results_B1.txt"), "w") do io
    println(io, "n_temp=", result.n_temp)
    println(io, "elapsed=", time() - t0)
    println(io, "all_id_covered=", all_id_covered)
    for name in PARAM_NAMES
        idx = name_to_idx[name]
        println(io, "$(name)_truth=", TRUTH[name])
        println(io, "$(name)_mean=",  means[idx])
        println(io, "$(name)_ci_low=", ci_low[idx])
        println(io, "$(name)_ci_high=", ci_high[idx])
    end
end
