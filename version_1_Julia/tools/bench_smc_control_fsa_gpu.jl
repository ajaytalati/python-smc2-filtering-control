#!/usr/bin/env julia
# Stage D — open-loop SMC²-as-controller on the FSA-v1 model (Julia/GPU).
#
# Direct port of `version_1/tools/bench_smc_control_fsa.py` to the Julia GPU
# pipeline (FSAv1ControlGPUTarget + run_tempered_smc_gpu). Same horizon
# sweep (T = 28, 42, 56, 84 d), same 6-panel diagnostic plot, same output
# filename convention as the Python bench.
#
# Run:
#   cd version_1_Julia
#   julia --project=. tools/bench_smc_control_fsa_gpu.jl [T_total_days] [seed]
#
# Defaults: T_total_days=42, seed=42.
#
# Output:
#   version_1_Julia/outputs/fsa_high_res/D_v2_T<N>_diagnostic.png

using CUDA, Random, Statistics, Printf, LinearAlgebra
using Plots
using SMC2FC: run_tempered_smc_gpu

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.GPUControl: FSAv1ControlGPUTarget, gpu_cost_log_density_batched,
                                make_log_density_fn, build_rbf_design
using .FSAHighRes.Control: build_control
using .FSAHighRes.Dynamics: TRUTH_PARAMS
using .FSAHighRes.Simulation: INIT_STATE, EXOGENOUS, simulate_em


# ── CLI ─────────────────────────────────────────────────────────────────
const T_TOTAL_DAYS = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 42.0
const SEED         = length(ARGS) >= 2 ? parse(Int,     ARGS[2]) : 42

# ── Config — mirrors version_1/tools/bench_smc_control_fsa.py ──────────
const BINS_PER_DAY = 96
const DT_DAYS      = 1.0 / BINS_PER_DAY
const N_SUBSTEPS   = 4
const N_ANCHORS    = 8
const N_INNER      = 32
const F_MAX        = 0.40
const PHI_MAX      = 3.0
const PHI_DEFAULT  = 1.0
const SIGMA_PRIOR  = 1.5
const TARGET_NATS  = 8.0
const TARGET_ESS   = 0.5
const MAX_LAMBDA   = 0.10
const NUM_MCMC     = 15
const HMC_LEAP     = 16
const MAX_LEVELS   = 100
const N_SMC        = 256
const CHEES_LS     = [16, 32, 64, 128, 256]

# Horizon-adaptive HMC step size (matches Python lines 64-69).
const HMC_STEP = T_TOTAL_DAYS >= 70.0 ? 0.05 :
                 T_TOTAL_DAYS >= 50.0 ? 0.12 : 0.30

const N_STEPS = Int(round(T_TOTAL_DAYS / DT_DAYS))
const M_MAX   = N_SMC * (1 + 2 * N_ANCHORS)   # 256 × 17 = 4352


# ── Utilities ───────────────────────────────────────────────────────────
sigmoid_(x) = 1.0 / (1.0 + exp(-x))
logit_(p)   = log(p / (1.0 - p))

"""
Decode posterior-mean θ → per-bin Φ schedule (n_steps,) — Float64.
"""
function decode_schedule(theta::Vector{Float64}, design::Matrix{Float64};
                          Phi_max::Real = PHI_MAX,
                          Phi_default::Real = PHI_DEFAULT)
    c_Phi = logit_(Phi_default / Phi_max)
    raw = c_Phi .+ design * theta
    return Phi_max .* sigmoid_.(raw)
end


"""
    eval_mean_A_under(Phi_arr, n_trials, rng) -> (mean_A, F_violation_frac)

Monte-Carlo evaluate the time-averaged ∫A/T and the F-violation fraction
under a given Φ schedule. Mirrors Python's `_evaluate_schedule_jit` loop in
control.py:`_build_gates`.
"""
function eval_mean_A_under(Phi_arr::AbstractVector{<:Real}, n_trials::Int, rng::AbstractRNG;
                            params = TRUTH_PARAMS)
    n_steps = length(Phi_arr)
    A_means = zeros(Float64, n_trials)
    F_viols = zeros(Float64, n_trials)
    for t in 1:n_trials
        result = simulate_em(; params=params,
                              init_state=INIT_STATE,
                              Phi_schedule=Float64.(Phi_arr),
                              dt=DT_DAYS, n_substeps=N_SUBSTEPS,
                              seed=rand(rng, UInt32))
        A_means[t] = mean(result.A)
        F_viols[t] = sum(result.F .> F_MAX) / n_steps
    end
    return mean(A_means), mean(F_viols)
end


# ── Build the cost target ───────────────────────────────────────────────
println("="^76)
@printf("  Stage D — SMC²-as-controller on FSA-v1  (T = %.0f days)\n", T_TOTAL_DAYS)
println("="^76)
@printf("  device:        %s\n", CUDA.name(CUDA.device()))
@printf("  initial state: B=%.2f, F=%.2f, A=%.2f\n",
        INIT_STATE.B, INIT_STATE.F, INIT_STATE.A)
@printf("  horizon:       %d outer × %d sub-steps = %.1f days\n",
        N_STEPS, N_SUBSTEPS, N_STEPS * DT_DAYS)
@printf("  theta_dim:     %d (Φ-only control, %d RBF anchors)\n", N_ANCHORS, N_ANCHORS)
@printf("  HMC step:      %.3f (T-adaptive)\n", HMC_STEP)

target = FSAv1ControlGPUTarget(
    n_inner   = N_INNER, M_max = M_MAX,
    n_steps   = N_STEPS, n_anchors = N_ANCHORS, n_substeps = N_SUBSTEPS,
    dt        = DT_DAYS,
    F_max     = F_MAX, Phi_max = PHI_MAX, Phi_default = PHI_DEFAULT,
    lam_F     = 1.0,
    sigma_prior = SIGMA_PRIOR,
    params    = TRUTH_PARAMS,
    init_state = [INIT_STATE.B, INIT_STATE.F, INIT_STATE.A],
    noise_seed = SEED,
)

# RBF design matrix — identical to what's inside the GPU kernel; reused for
# decoding posterior θ to per-bin Phi for plotting.
design = build_rbf_design(N_STEPS, DT_DAYS, N_ANCHORS)

# ── Baseline + sedentary references (constant Φ) ────────────────────────
print("  computing baseline (Φ=$PHI_DEFAULT) and sedentary (Φ=0) references… ")
rng_ref  = MersenneTwister(SEED + 1)
Phi_const = fill(PHI_DEFAULT, N_STEPS)
Phi_zero  = fill(0.0, N_STEPS)
baseline_mean_A,  baseline_F_viol = eval_mean_A_under(Phi_const, 50, rng_ref)
sedentary_mean_A, _               = eval_mean_A_under(Phi_zero,  50, rng_ref)
println("done.")
@printf("  baseline mean ∫A/T:  %.3f   (constant Φ=%.1f, F-violation %.2f%%)\n",
        baseline_mean_A, PHI_DEFAULT, 100*baseline_F_viol)
@printf("  sedentary mean ∫A/T: %.3f   (constant Φ=0)\n", sedentary_mean_A)

# ── Run SMC² controller ─────────────────────────────────────────────────
println()
println("  running parallel-chains tempered SMC² + ChEES-HMC…")
log_density_fn = make_log_density_fn(target)
rng_smc = MersenneTwister(SEED)

t0 = time()
posterior, n_temp, β_max = run_tempered_smc_gpu(
    log_density_fn, M_MAX, N_SMC, N_ANCHORS,
    0.0, SIGMA_PRIOR, rng_smc;
    target_nats     = TARGET_NATS,
    target_ess_frac = TARGET_ESS,
    max_lambda_inc  = MAX_LAMBDA,
    max_temp_levels = MAX_LEVELS,
    num_mcmc_steps  = NUM_MCMC,
    hmc_step_size   = HMC_STEP,
    hmc_num_leapfrog = HMC_LEAP,
    chees_L_candidates = CHEES_LS,
    h_fd            = 1e-4,
    calib_n         = 64,
    verbose         = false,
)
elapsed = time() - t0
@printf("  done: %d tempering levels in %.1fs   (β_max=%.2f)\n",
        n_temp, elapsed, β_max)

# Per-particle costs at the final temperature (for the histogram panel).
ll_final = log_density_fn(posterior)        # -cost per particle
particle_costs = -ll_final

# Posterior-mean θ → schedule
theta_post = vec(mean(posterior; dims=1))
Phi_smc = decode_schedule(theta_post, design)

# Sample 5 trajectories under the SMC² posterior-mean schedule (for plots).
rng_traj = MersenneTwister(SEED + 7)
n_traj   = 5
trajs    = Array{Float64}(undef, n_traj, N_STEPS, 3)
for i in 1:n_traj
    res = simulate_em(; params=TRUTH_PARAMS,
                        init_state=INIT_STATE,
                        Phi_schedule=Float64.(Phi_smc),
                        dt=DT_DAYS, n_substeps=N_SUBSTEPS,
                        seed=rand(rng_traj, UInt32))
    trajs[i, :, 1] = res.B
    trajs[i, :, 2] = res.F
    trajs[i, :, 3] = res.A
end
smc_mean_A = mean(trajs[:, :, 3])

# ── Acceptance gates (mirror Python) ────────────────────────────────────
gate1_pass = smc_mean_A >= 0.97 * baseline_mean_A
gate2_pass = smc_mean_A >= 1.40 * sedentary_mean_A
mean_phi   = mean(Phi_smc)
gate3_pass = (mean_phi >= 0.5) && (mean_phi <= 2.5)
# F-violation fraction over the 5 trajectory samples
F_viol_frac = mean([sum(trajs[i, :, 2] .> F_MAX) for i in 1:n_traj]) / N_STEPS
gate4_pass = F_viol_frac <= 0.05

println()
println("  Acceptance gates:")
@printf("    [%s] mean A matches baseline (within 3%%): SMC %.3f vs base*0.97 %.3f\n",
        gate1_pass ? "PASS" : "FAIL", smc_mean_A, 0.97 * baseline_mean_A)
@printf("    [%s] mean A ≥ 1.40 × sedentary:           SMC %.3f vs sed*1.40   %.3f\n",
        gate2_pass ? "PASS" : "FAIL", smc_mean_A, 1.40 * sedentary_mean_A)
@printf("    [%s] mean Φ ∈ [0.5, 2.5]:                 mean Φ = %.3f\n",
        gate3_pass ? "PASS" : "FAIL", mean_phi)
@printf("    [%s] F-violation ≤ 5%%:                   %.2f%%\n",
        gate4_pass ? "PASS" : "FAIL", 100*F_viol_frac)
println()

# ── 6-panel diagnostic plot ─────────────────────────────────────────────
out_dir  = joinpath(REPO_ROOT, "outputs", "fsa_high_res")
mkpath(out_dir)
T_tag    = "T$(Int(round(T_TOTAL_DAYS)))"
out_path = joinpath(out_dir, "D_v2_$(T_tag)_diagnostic.png")

t_grid = (0:N_STEPS-1) .* DT_DAYS

# Panel 1: Φ(t) schedule
p1 = plot(t_grid, Phi_smc, color=:darkorange, lw=2, label="SMC² Φ(t)",
           xlabel="time (days)", ylabel="Φ (training strain rate)",
           title="SMC²-derived Φ(t) schedule",
           ylim=(-0.1, max(PHI_MAX, maximum(Phi_smc)) + 0.1),
           legend=:topright, grid=true)
hline!(p1, [PHI_DEFAULT], color=:gray, linestyle=:dot, alpha=0.7,
       label="baseline Φ = $(PHI_DEFAULT)")
hline!(p1, [0.0], color=:red, linestyle=:dot, alpha=0.4,
       label="sedentary Φ = 0")

# Panel 2: per-particle cost histogram
p2 = histogram(particle_costs, bins=30, color=:steelblue, alpha=0.7,
                label="SMC² per-particle cost",
                xlabel="cost", ylabel="density",
                title="SMC² per-particle cost distribution",
                grid=true, legend=:topright)

# Panel 3: B(t) trajectories
p3 = plot(xlabel="time (days)", ylabel="B (fitness)",
           title="B trajectory (Banister chronic) under SMC² schedule",
           ylim=(-0.05, 1.05), grid=true, legend=:topright)
for i in 1:n_traj
    plot!(p3, t_grid, trajs[i, :, 1], color=:steelblue, lw=0.7, alpha=0.4,
           label=(i == 1 ? "samples" : nothing))
end
plot!(p3, t_grid, vec(mean(trajs[:, :, 1]; dims=1)), color=:steelblue, lw=2,
       label="mean B(t)")

# Panel 4: F(t) trajectories
p4 = plot(xlabel="time (days)", ylabel="F (strain / fatigue)",
           title="F trajectory (Banister acute) under SMC² schedule",
           grid=true, legend=:topright)
for i in 1:n_traj
    plot!(p4, t_grid, trajs[i, :, 2], color=:darkred, lw=0.7, alpha=0.4,
           label=(i == 1 ? "samples" : nothing))
end
plot!(p4, t_grid, vec(mean(trajs[:, :, 2]; dims=1)), color=:darkred, lw=2,
       label="mean F(t)")
hline!(p4, [F_MAX], color=:red, linestyle=:dash, alpha=0.5,
       label="F_max = $(F_MAX)")

# Panel 5: A(t) trajectories
p5 = plot(xlabel="time (days)", ylabel="A (amplitude)",
           title="A trajectory under SMC² schedule (headline)",
           grid=true, legend=:topright)
for i in 1:n_traj
    plot!(p5, t_grid, trajs[i, :, 3], color=:green, lw=0.7, alpha=0.4,
           label=(i == 1 ? "samples" : nothing))
end
plot!(p5, t_grid, vec(mean(trajs[:, :, 3]; dims=1)), color=:green, lw=2,
       label="mean A(t)")
hline!(p5, [baseline_mean_A], color=:gray, linestyle=:dot, alpha=0.7,
       label=@sprintf("baseline mean A = %.3f", baseline_mean_A))
hline!(p5, [sedentary_mean_A], color=:red, linestyle=:dot, alpha=0.5,
       label=@sprintf("sedentary mean A = %.3f", sedentary_mean_A))

# Panel 6: bar chart — sedentary vs baseline vs SMC²
labels_bar = ["sedentary\n(Φ=0)",
              "baseline\n(Φ=$(PHI_DEFAULT))",
              "SMC²\n(time-varying)"]
values_bar = [sedentary_mean_A, baseline_mean_A, smc_mean_A]
p6 = bar(labels_bar, values_bar, color=[:salmon, :gray, :steelblue],
         ylabel="mean ∫A / T  (time-averaged amplitude)",
         title="Mean amplitude comparison",
         legend=false, grid=true)

ttl = @sprintf("Stage D — FSA-v1 (Banister) control:  T = %.0f d, %d tempering levels in %.0fs on GPU",
                T_TOTAL_DAYS, n_temp, elapsed)
fig = plot(p1, p2, p3, p4, p5, p6;
            layout = (3, 2),
            size   = (1400, 1100),
            plot_title = ttl,
            plot_titlefontsize = 11)

savefig(fig, out_path)
@printf("  Plot: %s\n", out_path)
println("="^76)
