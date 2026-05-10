#!/usr/bin/env julia
# §2.6 trivial-cost end-to-end test: J(Φ) = -∫₀ᵀ B(t) dt, λ_F = 0.
#
# Under the FSA SDE, dB/dt = κ_B · a_B(A) · Φ - B/τ_B is monotone in Φ.
# The optimum is to pin Φ = Φ_max for every bin, driving B toward its
# highest equilibrium κ_B · Φ_max · τ_B. If the controller finds this
# pinned-high schedule, §2.6 rules out (i) the FSA SDE integration,
# (ii) the RBF decoder, (iii) the GPU-kernel cost-accumulation pathway,
# and (iv) the framework parallel-chains HMC. Any remaining controller
# bug is then confined to the A-cost surface.
#
# Implementation notes
# --------------------
# - The production cost kernel `fsa_cost_kernel!` in
#   models/fsa_high_res/gpu_control.jl is hard-coded to Eq 37
#   (-∫A dt + λ_F·∫max(F-F_max,0)² dt) and is not allowed to be edited.
#   This driver therefore inlines its OWN copy of the kernel and target
#   with three extra state-integrand weights (w_B, w_F, w_A). At default
#   weights (1, 0, 0) and λ_F = 0 the cost is -∫B dt; at (0, 0, 1) and
#   λ_F = 1 it reproduces Eq 37 byte-for-byte (modulo the harmless extra
#   B/F accumulator adds), so the same driver also serves as a sanity
#   cross-check against the production kernel.
#
# - Planning horizon is FIXED at 14 d per §2.11 of the writeup, INDEPENDENT
#   of bench duration. The original Julia bench had a shrinking-horizon
#   bug that drove the late-bench plan into a recover-only basin; the
#   correct convention is one chronic-B time constant (τ_B = 14 d) of
#   lookahead at every replan. This driver keeps the planning horizon
#   under the `--H-plan-days` flag (default 14) and the plant rollout
#   under `--T-bench-days` (default 14). The two coincide at default
#   so the §2.6 numbers reproduce exactly.
#
# - After SMC² returns the posterior, the driver runs a single open-loop
#   plant rollout (MPC schedule + Φ=1 baseline) and writes a JLD2 in the
#   schema that `tools/plot_state_traces.jl` consumes, then auto-invokes
#   the plotter to produce the 4-panel B/F/A/Φ figure — same auto-plot
#   pattern as bench_smc_full_mpc_fsa_gpu.jl:609-636.

using Dates


# ── CLI parsing — must run BEFORE the model import so FSA_STEP_MINUTES is
#    set in the environment by the time the model module reads it. ─────────

function _parse_args(argv::Vector{String})
    defaults = Dict{String,Any}(
        "T-bench-days"   => 14,        # how long the plant rolls forward
        "H-plan-days"    => 14,        # FIXED planning horizon (§2.11; τ_B)
        "step-minutes"   => 15,
        "n-smc"          => 64,
        "n-inner"        => 16,
        "num-mcmc"       => 3,
        "hmc-leap"       => 8,
        "hmc-step"       => 0.2,
        "max-levels"     => 15,
        "target-nats"    => 8.0,
        "w-B"            => 1.0,
        "w-F"            => 0.0,
        "w-A"            => 0.0,
        "lam-F"          => 0.0,
        "seed"           => 42,
        "output-dir"     => "",
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if startswith(a, "--")
            key = a[3:end]
            if !haskey(defaults, key)
                error("Unknown flag: $a (known: $(sort(collect(keys(defaults)))))")
            end
            i += 1
            i <= length(argv) || error("Missing value for $a")
            v = argv[i]
            if defaults[key] isa Int
                defaults[key] = parse(Int, v)
            elseif defaults[key] isa Float64
                defaults[key] = parse(Float64, v)
            else
                defaults[key] = v
            end
            i += 1
        else
            error("Unrecognized arg: $a")
        end
    end
    return defaults
end

const ARGS_DICT = _parse_args(copy(ARGS))
ENV["FSA_STEP_MINUTES"] = string(ARGS_DICT["step-minutes"])

if ARGS_DICT["T-bench-days"] > ARGS_DICT["H-plan-days"]
    error("--T-bench-days ($(ARGS_DICT["T-bench-days"])) > --H-plan-days " *
          "($(ARGS_DICT["H-plan-days"])). The driver runs a single open-loop " *
          "plan; rolling the plant past the plan end would require closed-loop " *
          "replanning, which is out of scope for this trivial-cost test. Either " *
          "shorten T-bench or extend H-plan.")
end

@info "loading model + framework (FSA_STEP_MINUTES=$(ENV["FSA_STEP_MINUTES"]))..."


# ── Imports ───────────────────────────────────────────────────────────────

using CUDA, KernelAbstractions
using Random, Statistics, Printf, LinearAlgebra
using JLD2

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
include(joinpath(REPO_ROOT, "models", "fsa_high_res", "FSAHighRes.jl"))
using .FSAHighRes
using .FSAHighRes.Dynamics: TRUTH_PARAMS, A_TYP, F_TYP
using .FSAHighRes.PhiBurst: BINS_PER_DAY
using .FSAHighRes.Plant: StepwisePlant, advance_subdaily!
using .FSAHighRes.Simulation: DEFAULT_PARAMS

using SMC2FC: run_tempered_smc_gpu


# ── Local cost kernel — verbatim copy of fsa_cost_kernel! from
#    models/fsa_high_res/gpu_control.jl with three state-integrand weights
#    (w_B, w_F, w_A) added. The accumulators for B and F are added in the
#    same Kahan-compensated style as the existing A_acc / barrier_acc, so
#    fp32 cancellation behaviour is identical to the production kernel. At
#    weights (0, 0, 1) and λ_F = 1 the cost reduces to Eq 37; at (1, 0, 0)
#    and λ_F = 0 it is the trivial -∫B dt monotone test of §2.6. ──────────

@kernel function fsa_cost_kernel_weighted!(
    cost_per_thread,
    theta_per_chain,
    rbf_design,
    init_state,
    p_tau_B, p_tau_F, p_kappa_B, p_kappa_F,
    p_epsilon_A, p_lambda_A,
    p_mu_0, p_mu_B, p_mu_F, p_mu_FF,
    p_eta, p_sigma_B, p_sigma_F, p_sigma_A,
    p_a_typ_inv_B, p_a_typ_inv_F,
    p_mu_const, p_mu_lin_F,
    fixed_w,
    c_Phi, Phi_max, F_max, lam_F,
    w_B, w_F, w_A,
    dt, n_substeps, n_steps, n_anchors, M, n_inner,
)
    i = @index(Global, Linear)
    if i <= M * n_inner
        m_chain = ((i - 1) ÷ n_inner) + 1
        t_trial = ((i - 1) % n_inner) + 1

        sub_dt  = dt / Float32(n_substeps)
        sqrt_dt = sqrt(dt)
        eps_B   = 1f-4

        B, F, A = init_state[1], init_state[2], init_state[3]

        B_acc,   B_comp   = 0f0, 0f0
        F_acc,   F_comp   = 0f0, 0f0
        A_acc,   A_comp   = 0f0, 0f0
        bar_acc, bar_comp = 0f0, 0f0

        @inbounds for k in 1:n_steps
            raw = c_Phi
            for a in 1:n_anchors
                raw = fma(theta_per_chain[m_chain, a], rbf_design[k, a], raw)
            end
            Phi_t = Phi_max / (1f0 + exp(-raw))

            y_B0     = fma(B, dt, -B_comp)
            t_B0     = B_acc + y_B0
            B_comp   = (t_B0 - B_acc) - y_B0
            B_acc    = t_B0

            y_F0     = fma(F, dt, -F_comp)
            t_F0     = F_acc + y_F0
            F_comp   = (t_F0 - F_acc) - y_F0
            F_acc    = t_F0

            y_A      = fma(A, dt, -A_comp)
            t_A      = A_acc + y_A
            A_comp   = (t_A - A_acc) - y_A
            A_acc    = t_A

            bdiff    = max(F - F_max, 0f0)
            bar_val  = bdiff * bdiff * dt
            y_bar    = bar_val - bar_comp
            t_bar    = bar_acc + y_bar
            bar_comp = (t_bar - bar_acc) - y_bar
            bar_acc  = t_bar

            inv_tau_B = 1f0 / p_tau_B
            inv_tau_F = 1f0 / p_tau_F

            for sub in 1:n_substeps
                mu_bif = fma(F, (p_mu_lin_F - p_mu_FF * F),
                             fma(p_mu_B, B, p_mu_const))

                a_factor_B = fma(p_epsilon_A, A, 1f0) * p_a_typ_inv_B
                a_factor_F = fma(p_lambda_A,  A, 1f0) * p_a_typ_inv_F

                drift_B = fma(p_kappa_B * a_factor_B, Phi_t, -B * inv_tau_B)
                drift_F = fma(p_kappa_F, Phi_t, -(a_factor_F * inv_tau_F * F))
                drift_A = fma(mu_bif, A, -(p_eta * A * A * A))

                B = fma(sub_dt, drift_B, B)
                F = fma(sub_dt, drift_F, F)
                A = fma(sub_dt, drift_A, A)
            end

            B_cl        = max(eps_B, min(1f0 - eps_B, B))
            sigma_B_eff = p_sigma_B * sqrt(B_cl * (1f0 - B_cl))
            sigma_F_eff = p_sigma_F * sqrt(max(0f0, F))
            sigma_A_eff = p_sigma_A * sqrt(max(0f0, A))

            B = fma(sigma_B_eff * sqrt_dt, fixed_w[t_trial, k, 1], B)
            F = fma(sigma_F_eff * sqrt_dt, fixed_w[t_trial, k, 2], F)
            A = fma(sigma_A_eff * sqrt_dt, fixed_w[t_trial, k, 3], A)

            B = B < 0f0 ? -B : (B > 1f0 ? 2f0 - B : B)
            F = abs(F)
            A = abs(A)
        end

        # J = -(w_B·∫B dt + w_F·∫F dt + w_A·∫A dt) + λ_F · ∫max(F-F_max,0)² dt
        state_int = fma(w_A, A_acc, fma(w_F, F_acc, w_B * B_acc))
        cost_per_thread[i] = Float64(fma(lam_F, bar_acc, -state_int))
    end
end


# ── Local target struct ──────────────────────────────────────────────────

mutable struct FSAWeightedGPUTarget
    n_inner::Int
    M_max::Int
    n_steps::Int
    n_anchors::Int
    n_substeps::Int
    dt::Float32
    F_max::Float32
    Phi_max::Float32
    Phi_default::Float32
    c_Phi::Float32
    lam_F::Float32
    w_B::Float32; w_F::Float32; w_A::Float32
    sigma_prior::Float64
    p_tau_B::Float32; p_tau_F::Float32
    p_kappa_B::Float32; p_kappa_F::Float32
    p_epsilon_A::Float32; p_lambda_A::Float32
    p_mu_0::Float32; p_mu_B::Float32; p_mu_F::Float32; p_mu_FF::Float32
    p_eta::Float32
    p_sigma_B::Float32; p_sigma_F::Float32; p_sigma_A::Float32
    p_a_typ_inv_B::Float32; p_a_typ_inv_F::Float32
    p_mu_const::Float32;    p_mu_lin_F::Float32
    rbf_design::CuArray{Float32,2}
    init_state::CuArray{Float32,1}
    fixed_w::CuArray{Float32,3}
    theta_per_chain::CuArray{Float32,2}
    cost_per_thread::CuArray{Float64,1}
    kernel::Any
end


function FSAWeightedGPUTarget(; n_inner::Int, M_max::Int,
                                n_steps::Int, n_anchors::Int = 8,
                                n_substeps::Int = 4,
                                dt::Real,
                                F_max::Real = 0.40,
                                Phi_max::Real = 3.0,
                                Phi_default::Real = 1.0,
                                lam_F::Real = 0.0,
                                w_B::Real = 1.0,
                                w_F::Real = 0.0,
                                w_A::Real = 0.0,
                                sigma_prior::Real = 1.5,
                                params,
                                init_state::AbstractVector,
                                noise_seed::Int = 42)
    p_ratio = Phi_default / Phi_max
    c_Phi   = log(p_ratio / (1.0 - p_ratio))

    T_total = n_steps * dt
    t_grid  = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length=n_anchors))
    σ = T_total / n_anchors
    M_design = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for k in 1:n_steps, j in 1:n_anchors
        d = t_grid[k] - anchors[j]
        M_design[k, j] = exp(-0.5 * (d / σ)^2)
    end

    rng = MersenneTwister(noise_seed)
    noise_cpu = randn(rng, Float32, n_inner, n_steps, 3)

    a_typ_inv_B = 1.0 / (1.0 + Float64(params.epsilon_A) * Float64(A_TYP))
    a_typ_inv_F = 1.0 / (1.0 + Float64(params.lambda_A)  * Float64(A_TYP))
    mu_const    = Float64(params.mu_0) - Float64(params.mu_FF) * Float64(F_TYP)^2
    mu_lin_F    = 2.0 * Float64(params.mu_FF) * Float64(F_TYP) - Float64(params.mu_F)

    return FSAWeightedGPUTarget(
        n_inner, M_max, n_steps, n_anchors, n_substeps,
        Float32(dt),
        Float32(F_max), Float32(Phi_max), Float32(Phi_default),
        Float32(c_Phi),
        Float32(lam_F),
        Float32(w_B), Float32(w_F), Float32(w_A),
        Float64(sigma_prior),
        Float32(params.tau_B), Float32(params.tau_F),
        Float32(params.kappa_B), Float32(params.kappa_F),
        Float32(params.epsilon_A), Float32(params.lambda_A),
        Float32(params.mu_0), Float32(params.mu_B), Float32(params.mu_F),
        Float32(params.mu_FF), Float32(params.eta),
        Float32(params.sigma_B), Float32(params.sigma_F), Float32(params.sigma_A),
        Float32(a_typ_inv_B), Float32(a_typ_inv_F),
        Float32(mu_const),    Float32(mu_lin_F),
        CuArray(Float32.(M_design)),
        CuArray(Float32.(init_state)),
        CuArray(noise_cpu),
        CUDA.zeros(Float32, M_max, n_anchors),
        CUDA.zeros(Float64, M_max * n_inner),
        fsa_cost_kernel_weighted!(CUDABackend(), 256),
    )
end


function gpu_cost_log_density_batched_weighted(target::FSAWeightedGPUTarget,
                                                theta_unc::AbstractMatrix{Float64})
    M = size(theta_unc, 1)
    M ≤ target.M_max || throw(ArgumentError("M=$M > M_max=$(target.M_max)"))
    @assert size(theta_unc, 2) == target.n_anchors

    theta_cpu_f32 = Float32.(theta_unc)
    copyto!(view(target.theta_per_chain, 1:M, :), theta_cpu_f32)

    Ntot = M * target.n_inner
    target.kernel(
        view(target.cost_per_thread, 1:Ntot),
        view(target.theta_per_chain, 1:M, :),
        target.rbf_design,
        target.init_state,
        target.p_tau_B, target.p_tau_F,
        target.p_kappa_B, target.p_kappa_F,
        target.p_epsilon_A, target.p_lambda_A,
        target.p_mu_0, target.p_mu_B, target.p_mu_F, target.p_mu_FF,
        target.p_eta,
        target.p_sigma_B, target.p_sigma_F, target.p_sigma_A,
        target.p_a_typ_inv_B, target.p_a_typ_inv_F,
        target.p_mu_const,    target.p_mu_lin_F,
        target.fixed_w,
        target.c_Phi, target.Phi_max,
        target.F_max, target.lam_F,
        target.w_B, target.w_F, target.w_A,
        target.dt, target.n_substeps,
        target.n_steps, target.n_anchors,
        M, target.n_inner;
        ndrange = Ntot,
    )
    KernelAbstractions.synchronize(CUDABackend())

    cost_cpu = Array(view(target.cost_per_thread, 1:Ntot))
    cost_mat = reshape(cost_cpu, target.n_inner, M)
    out = Vector{Float64}(undef, M)
    @inbounds for m in 1:M
        s = 0.0
        for t in 1:target.n_inner
            s += cost_mat[t, m]
        end
        out[m] = -s / target.n_inner
    end
    return out
end


function make_log_density_fn_weighted(target::FSAWeightedGPUTarget)
    return function (U::AbstractMatrix{Float64})
        return gpu_cost_log_density_batched_weighted(target, U)
    end
end


# ── RBF decoder (CPU) — exact mirror of the kernel's RAW Gaussian basis ──

function decode_per_bin_phi(theta::Vector{Float64}, n_steps::Int, n_anchors::Int,
                            dt::Float64, phi_max::Float64, c_Phi::Float64)
    T_total = n_steps * dt
    t_grid = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length = n_anchors))
    σ_rbf = T_total / n_anchors
    out = zeros(Float64, n_steps)
    for k in 1:n_steps
        raw = c_Phi
        for j in 1:n_anchors
            v = exp(-0.5 * ((t_grid[k] - anchors[j]) / σ_rbf)^2)
            raw += theta[j] * v
        end
        out[k] = phi_max / (1.0 + exp(-raw))
    end
    return out
end


# ── Run config ───────────────────────────────────────────────────────────

const D            = 8
const N_SMC        = ARGS_DICT["n-smc"]
const N_INNER      = ARGS_DICT["n-inner"]
const N_ANCHORS    = 8
const M_MAX        = N_SMC * (1 + 2 * D)
const SIGMA_PRIOR  = 1.5

const T_BENCH_DAYS = ARGS_DICT["T-bench-days"]
const H_PLAN_DAYS  = ARGS_DICT["H-plan-days"]
const N_PLAN_STEPS = H_PLAN_DAYS * BINS_PER_DAY     # planning horizon (§2.11)
const STRIDE_BINS  = BINS_PER_DAY
const N_STRIDES    = T_BENCH_DAYS
const DT           = 1.0 / BINS_PER_DAY
const PHI_MAX      = 3.0
const PHI_DEFAULT  = 1.0

const W_B   = Float64(ARGS_DICT["w-B"])
const W_F   = Float64(ARGS_DICT["w-F"])
const W_A   = Float64(ARGS_DICT["w-A"])
const LAM_F = Float64(ARGS_DICT["lam-F"])
const SEED  = ARGS_DICT["seed"]


# ── Build target ─────────────────────────────────────────────────────────

target = FSAWeightedGPUTarget(
    n_inner    = N_INNER,
    M_max      = M_MAX,
    n_steps    = N_PLAN_STEPS,
    n_anchors  = N_ANCHORS,
    n_substeps = 4,
    dt         = DT,
    F_max      = 0.40,
    Phi_max    = PHI_MAX,
    Phi_default = PHI_DEFAULT,
    lam_F      = LAM_F,
    w_B        = W_B,
    w_F        = W_F,
    w_A        = W_A,
    sigma_prior = SIGMA_PRIOR,
    params     = TRUTH_PARAMS,
    init_state = [0.05, 0.30, 0.10],
    noise_seed = SEED,
)

@info "GPU device: $(CUDA.name(CUDA.device()))"
@info @sprintf("Cost: J(Φ) = -(%.2f·∫B + %.2f·∫F + %.2f·∫A) dt + %.2f·∫max(F-F_max,0)² dt",
                W_B, W_F, W_A, LAM_F)
@info @sprintf("Planning horizon (§2.11): H_plan = %d d  (%d bins)", H_PLAN_DAYS, N_PLAN_STEPS)
@info @sprintf("Plant rollout:            T_bench = %d d (%d bins)",
                T_BENCH_DAYS, N_STRIDES * STRIDE_BINS)
@info @sprintf("SMC² config: D=%d N_SMC=%d N_INNER=%d num_mcmc=%d hmc_leap=%d hmc_step=%.2f",
                D, N_SMC, N_INNER, ARGS_DICT["num-mcmc"],
                ARGS_DICT["hmc-leap"], ARGS_DICT["hmc-step"])


# ── Run SMC² ─────────────────────────────────────────────────────────────

t0 = time()
log_density_fn = make_log_density_fn_weighted(target)
rng = MersenneTwister(SEED)
posterior, n_temp, β_max = run_tempered_smc_gpu(
    log_density_fn, M_MAX, N_SMC, D,
    0.0, SIGMA_PRIOR, rng;
    target_nats        = ARGS_DICT["target-nats"],
    target_ess_frac    = 0.5,
    max_lambda_inc     = 0.20,
    max_temp_levels    = ARGS_DICT["max-levels"],
    num_mcmc_steps     = ARGS_DICT["num-mcmc"],
    hmc_step_size      = ARGS_DICT["hmc-step"],
    hmc_num_leapfrog   = ARGS_DICT["hmc-leap"],
    chees_L_candidates = [16, 32, 64, 128, 256],
    h_fd               = 1e-4,
    calib_n            = 64,
    verbose            = true,
)
wall = time() - t0


# ── Decode posterior-mean θ → per-bin Φ schedule ─────────────────────────

theta_post_mean = vec(mean(posterior; dims = 1))
const C_PHI = log((PHI_DEFAULT / PHI_MAX) / (1.0 - PHI_DEFAULT / PHI_MAX))
phi_per_bin = decode_per_bin_phi(theta_post_mean, N_PLAN_STEPS, N_ANCHORS, DT,
                                  PHI_MAX, C_PHI)


# ── Plant rollout: MPC schedule + Φ=1 baseline ───────────────────────────

mpc_plant  = StepwisePlant(truth_params = copy(DEFAULT_PARAMS),
                            init_state   = (B = 0.05, F = 0.30, A = 0.10),
                            dt           = DT,
                            seed_offset  = SEED)
base_plant = StepwisePlant(truth_params = copy(DEFAULT_PARAMS),
                            init_state   = (B = 0.05, F = 0.30, A = 0.10),
                            dt           = DT,
                            seed_offset  = SEED + 1000)

daily_phi_per_stride = Float32[]
for d in 0:N_STRIDES-1
    slice = phi_per_bin[d * STRIDE_BINS + 1 : (d + 1) * STRIDE_BINS]
    advance_subdaily!(mpc_plant,  Float32.(slice))
    advance_subdaily!(base_plant, fill(Float32(1.0), STRIDE_BINS))
    push!(daily_phi_per_stride, Float32(mean(slice)))
end

traj_mpc  = vcat(mpc_plant.history[:trajectory]...)
traj_base = vcat(base_plant.history[:trajectory]...)


# ── Output dir + JLD2 ────────────────────────────────────────────────────

out_dir = ARGS_DICT["output-dir"]
if isempty(out_dir)
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    out_dir = joinpath(REPO_ROOT, "outputs", "fsa_high_res",
                        "test_max_B_only", "run_$timestamp")
end
mkpath(out_dir)

data_path = joinpath(out_dir, "data.jld2")
JLD2.jldopen(data_path, "w") do f
    f["trajectory_mpc"]       = Float32.(traj_mpc)
    f["trajectory_baseline"]  = Float32.(traj_base)
    f["daily_phi_per_stride"] = daily_phi_per_stride
    f["BINS_PER_DAY"]         = BINS_PER_DAY
    f["STRIDE_BINS"]          = STRIDE_BINS
    f["dt_days"]              = DT
    f["F_max"]                = 0.40
    f["theta_post_mean"]      = theta_post_mean
    f["phi_per_bin"]          = Float32.(phi_per_bin)
    f["n_temp"]               = n_temp
    f["beta_max"]             = β_max
    f["wall_seconds"]         = wall
    f["weights"]              = [W_B, W_F, W_A]
    f["lam_F"]                = LAM_F
    f["T_bench_days"]         = T_BENCH_DAYS
    f["H_plan_days"]          = H_PLAN_DAYS
    f["seed"]                 = SEED
end
@info "wrote $data_path"


# ── Auto-invoke plot_state_traces.jl ─────────────────────────────────────
# Same world-age workaround as bench_smc_full_mpc_fsa_gpu.jl:618-621.

try
    let plotters_dir = @__DIR__
        include(joinpath(plotters_dir, "plot_state_traces.jl"))
        data_dict = Base.invokelatest(Main.load_run_data, data_path)
        traces_path = joinpath(out_dir,
            "test_max_B_only_T$(T_BENCH_DAYS)d_traces.png")
        Base.invokelatest(Main.plot_state_traces, data_dict; out_path=traces_path)
        @info "wrote $traces_path"
    end
catch e
    @warn "plot generation failed (data.jld2 still saved)" exception=(e, catch_backtrace())
end


# ── Console summary ──────────────────────────────────────────────────────

println("="^72)
@printf("Result: %d tempering levels, β_max = %.3f, %.1fs wall\n",
        n_temp, β_max, wall)
println("-"^72)
@printf("Cost weights (w_B, w_F, w_A) = (%.2f, %.2f, %.2f)   λ_F = %.2f\n",
        W_B, W_F, W_A, LAM_F)
@printf("H_plan = %d d (FIXED, §2.11)   T_bench = %d d   step = %d min\n",
        H_PLAN_DAYS, T_BENCH_DAYS, ARGS_DICT["step-minutes"])
println()
@printf("Posterior mean θ = %s\n",
        "[" * join([@sprintf("%+.3f", v) for v in theta_post_mean], ", ") * "]")
println()
@printf("Per-day applied Φ̄ (mean over each day's bins):\n")
for d in 0:N_STRIDES-1
    @printf("  day %2d  Φ̄ = %.3f\n", d, daily_phi_per_stride[d + 1])
end
println()
@printf("Φ summary over plan: min = %.3f   mean = %.3f   max = %.3f   (Φ_max = %.2f)\n",
        minimum(phi_per_bin), mean(phi_per_bin), maximum(phi_per_bin), PHI_MAX)
println()
@printf("Mean A: MPC = %.4f   baseline = %.4f\n",
        mean(traj_mpc[:, 3]), mean(traj_base[:, 3]))
@printf("Mean B: MPC = %.4f   baseline = %.4f\n",
        mean(traj_mpc[:, 1]), mean(traj_base[:, 1]))
@printf("Mean F: MPC = %.4f   baseline = %.4f\n",
        mean(traj_mpc[:, 2]), mean(traj_base[:, 2]))
println()
@printf("data.jld2:   %s\n", data_path)
@printf("trace PNG:   %s\n",
        joinpath(out_dir, "test_max_B_only_T$(T_BENCH_DAYS)d_traces.png"))
println("="^72)
