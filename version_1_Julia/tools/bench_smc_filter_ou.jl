# Stage A1 — scalar OU filter bench.
#
# Mirror of `version_1/tools/bench_smc_filter_*.py`. Generates synthetic
# data with the truth params, runs the Julia bootstrap PF at TRUTH params,
# compares marginal log-likelihood to the closed-form Kalman log-likelihood.
#
# Headline gate (matches Python README): `|PF_LL − Kalman_LL| < 5 nats`.
# The Python result was −0.18 nats; the Julia bootstrap PF should land
# in the same ballpark.
#
# Run with:
#   julia --project=.. tools/bench_smc_filter_ou.jl
# from `version_1_Julia/`.

using Random
using Statistics: mean, std
using Plots

# Locate package roots and load.
const ROOT = dirname(@__DIR__)
push!(LOAD_PATH, joinpath(ROOT, "models", "scalar_ou_lqg"))

using SMC2FC
using ScalarOULQG: PARAM_SET_A, INIT_STATE_A, EXOGENOUS_A,
                    simulate_em, simulate_diffeq,
                    kalman_filter, build_estimation_model

# ── 1. Generate data with truth params ──────────────────────────────────────
println("="^70)
println("Stage A1 — scalar OU filter")
println("Truth: a=$(PARAM_SET_A.a), b=$(PARAM_SET_A.b), σ_w=$(PARAM_SET_A.sigma_w), σ_v=$(PARAM_SET_A.sigma_v)")
println("Grid:  T=$(EXOGENOUS_A.T), dt=$(EXOGENOUS_A.dt)")
println("="^70)

const DATA_SEED = 1     # Julia MT seed 1 → x_0 ≈ -0.14, comparable to
                         # NumPy's `default_rng(0)` draw used in the Python
                         # `test_scalar_ou_filter_matches_kalman.py` test.
                         # The PF initialises tight around 0 (σ_w·√dt ≈ 0.067)
                         # so a small |x_0| keeps the PF cloud near the truth.

data_em     = simulate_em(seed = DATA_SEED)
data_diffeq = simulate_diffeq(seed = DATA_SEED)
println("Trajectory peak (EM):       ", round(maximum(abs.(data_em.trajectory)),     digits=4))
println("Trajectory peak (DiffEq):   ", round(maximum(abs.(data_diffeq.trajectory)), digits=4))

# ── 2. Closed-form Kalman log-likelihood (ground truth) ─────────────────────
kalman_res = kalman_filter(
    y       = data_em.obs,
    u       = data_em.u,
    a       = PARAM_SET_A.a,
    b       = PARAM_SET_A.b,
    sigma_w = PARAM_SET_A.sigma_w,
    sigma_v = PARAM_SET_A.sigma_v,
    dt      = EXOGENOUS_A.dt,
    x0_mean = INIT_STATE_A.x_0,
    x0_var  = EXOGENOUS_A.x0_var,
)
println("\nKalman log-likelihood: ", round(kalman_res.log_likelihood, digits=4), " nats")

# ── 3. Julia bootstrap PF at truth params ───────────────────────────────────
model = build_estimation_model()
priors = SMC2FC.all_priors(model)

# Unconstrained-space coordinates of (a, b, σ_w, σ_v, x_0) at truth.
# Lognormal → log; normal → identity.
u_truth = [
    log(PARAM_SET_A.a),
    log(PARAM_SET_A.b),
    log(PARAM_SET_A.sigma_w),
    log(PARAM_SET_A.sigma_v),
    INIT_STATE_A.x_0,                # x_0 prior is Normal(0, 1) → identity
]

# grid_obs uses Symbol keys (Julia convention)
grid_obs = Dict{Symbol,Any}(
    :obs_value   => data_em.obs,
    :obs_present => ones(Float64, EXOGENOUS_A.T),
    :u_value     => data_em.u,
)
fixed_init = [INIT_STATE_A.x_0]

# Inner PF cfg: K=1500 particles, no Liu-West contamination, no OT rescue
# (we want the cleanest log-lik estimator for a Kalman-comparable baseline).
cfg = SMCConfig(n_pf_particles = 1500, bandwidth_scale = 0.0, ot_max_weight = 0.0)

# Run the PF a few times across seeds to characterise PF MC variance.
const N_REPS = 8
pf_lls = Float64[]
for seed in 1:N_REPS
    pf_target = bootstrap_log_likelihood(
        model, u_truth, grid_obs, fixed_init, priors, cfg,
        MersenneTwister(seed);
        dt = EXOGENOUS_A.dt, t_steps = EXOGENOUS_A.T, window_start_bin = 0,
    )
    # bootstrap_log_likelihood returns log p(y|θ) + log p(u). Subtract the prior.
    lp = SMC2FC.log_prior_unconstrained(u_truth, priors)
    push!(pf_lls, pf_target - lp)
end
pf_mean = mean(pf_lls)
pf_std  = std(pf_lls)
println("PF log-likelihood (mean over $N_REPS seeds): ",
        round(pf_mean, digits=4), " ± ", round(pf_std, digits=4))
println("PF − Kalman bias: ", round(pf_mean - kalman_res.log_likelihood, digits=4), " nats")

bias  = abs(pf_mean - kalman_res.log_likelihood)
gate  = bias < 5.0
println(gate ? "GATE PASS — bias < 5 nats" : "GATE FAIL")

# ── 4. Diagnostic plot ──────────────────────────────────────────────────────
default(size = (1100, 700), fontfamily = "Helvetica")
t = data_em.t_grid

p1 = plot(t, data_em.trajectory[:, 1], lw = 2, label = "x (truth)",
          title = "A1 Stage — scalar OU filter", xlabel = "t", ylabel = "x")
plot!(p1, t, kalman_res.means, lw = 2, color = :red, label = "Kalman mean")
plot!(p1, t, kalman_res.means .+ 1.96 .* sqrt.(kalman_res.covars),
       fillrange = kalman_res.means .- 1.96 .* sqrt.(kalman_res.covars),
       fillalpha = 0.15, color = :red, label = "Kalman 95 % CI")
scatter!(p1, t, data_em.obs, ms = 3, color = :black, label = "obs")

p2 = bar(["Kalman", "PF (mean)"], [kalman_res.log_likelihood, pf_mean],
         title  = "Marginal log-likelihood at truth",
         ylabel = "log p(y | θ_truth)",
         legend = false, color = [:red, :steelblue])
annotate!(p2, [(2, pf_mean + 0.5,
                 text("σ = $(round(pf_std, digits=2))", 9, :left))])

p3 = histogram(pf_lls, bins = 6,
               title  = "PF log-lik across $N_REPS seeds",
               xlabel = "log p (PF)", ylabel = "count",
               legend = false, color = :steelblue)
vline!(p3, [kalman_res.log_likelihood], color = :red, lw = 2, label = "Kalman")

p4 = plot(t, data_em.trajectory[:, 1], lw = 2, label = "EM (hand-rolled)",
          title  = "Simulator: EM vs StochasticDiffEq",
          xlabel = "t", ylabel = "x")
plot!(p4, t, data_diffeq.trajectory[:, 1], lw = 2, ls = :dash,
       color = :green, label = "DiffEq EM solver")

plot(p1, p2, p3, p4, layout = (2, 2))
out_path = joinpath(ROOT, "outputs", "scalar_ou_lqg", "A1_filter_diagnostic_julia.png")
mkpath(dirname(out_path))
savefig(out_path)
println("\nSaved: $out_path")

println("\n", gate ? "✓ A1 GATE PASS" : "✗ A1 GATE FAIL")
