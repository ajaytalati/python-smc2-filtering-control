"""
    BenchController

High-level solver for the FSA-v5 Model Predictive Control (MPC) problem. 
Uses tempered Sequential Monte Carlo (SMC²) to optimize training stimulus plans.

# Mathematical Specification

The controller finds the optimal bimodal stimulus schedule Φ(t) = [Φ_B(t), Φ_S(t)] 
by optimizing the RBF coefficient vector θ ∈ ℝ^(2·n_anchors).

## 1. Stimulus Decoding (RBF Basis)
The stimulus for channel i ∈ {B, S} is decoded via a logistic transformation:
    Φ_i(t) = Φ_max / (1 + exp(-[c_Φ + Σ θ_j · R_j(t)]))
where R_j(t) are Gaussian Radial Basis Functions.

## 2. Objective Function (Multi-Objective Reward)
The controller minimizes the cost J(θ), which maximizes:
    J_reward = - ∫ [A(t) + B(t) + S(t)] dt
subject to:
    J_effort = λ_Φ · ∫ [Φ_B² + Φ_S²] dt
    J_safety = λ_chance · ∫ σ(β · (A_sep - A) / scale) dt

Implementation Note: The solver leverages `Float32` precision and a strictly 
stateless, pure functional architecture to ensure compatibility with GPU-accelerated 
SMC² kernels.
"""
module BenchController

using Statistics
using StableRNGs
using StaticArrays
using CUDA

# Re-import dynamics and kernel definitions
import ..SimulationV5: A_TYP, F_TYP
import ..GPUControlV5: FSAv5ControlGPUTarget, make_log_density_fn_v5
import SMC2FC_functional: run_tempered_smc_gpu

export controller_plan_v5


# =============================================================================
# 1. OPTIMAL CONTROL SOLVER (SMC²)
# =============================================================================

"""
    controller_plan_v5(params_v5, init_state, T_total_bins, n_substeps, dt, 
                       ctrl_cfg, key; collect_diagnostics=false) -> NamedTuple

Executes a tempered SMC² optimization to find the optimal training plan.

# Arguments
- `params_v5::Dict{Symbol, Float32}`: Centered model parameters.
- `init_state::SVector{6, Float32}`: Current 6D latent state.
- `T_total_bins::Int`: Horizon length in discrete bins.
- `n_substeps::Int`: Integration sub-steps for the EM solver.
- `dt::Float32`: Bin width in days.
- `ctrl_cfg::NamedTuple`: Optimizer settings (n_smc, lam_phi, etc.).
- `key::UInt64`: RNG seed for reproducibility.

# Returns
- `Phi_B_plan::Vector{Float32}`: Optimized aerobic stimulus schedule.
- `Phi_S_plan::Vector{Float32}`: Optimized strength stimulus schedule.
- `theta::Vector{Float32}`: Posterior mean RBF coefficients.
- `n_temp_ctrl::Int`: Number of tempering levels used.
- `diagnostics`: HMC diagnostic traces (if requested).
"""
function controller_plan_v5(params_v5::Dict{Symbol, Float32},
                              init_state::SVector{6, Float32},
                              T_total_bins::Int,
                              n_substeps::Int,
                              dt::Float32,
                              ctrl_cfg::NamedTuple,
                              key::UInt64;
                              collect_diagnostics::Bool = false)
    
    # ── Solver Configuration ──
    n_anchors::Int = ctrl_cfg.n_anchors
    theta_dim::Int = 2 * n_anchors  # Bimodal schedule: [B_anchors..., S_anchors...]

    # Initialize GPU target with current posterior parameters
    ctrl_target = FSAv5ControlGPUTarget(
        n_inner    = ctrl_cfg.n_inner,
        M_max      = ctrl_cfg.M_max,
        n_steps    = T_total_bins,
        n_anchors  = n_anchors,
        n_substeps = n_substeps,
        dt         = dt,
        F_max      = 0.40f0,
        Phi_max    = 3.0f0,
        Phi_default = ctrl_cfg.phi_default,
        lam_Phi    = ctrl_cfg.lam_phi,
        lam_F      = ctrl_cfg.lam_f,
        lam_chance = ctrl_cfg.lam_chance,
        A_thr      = 0.05f0,
        beta_chance = 50.0f0,
        scale_chance = 0.10f0,
        sigma_prior = ctrl_cfg.sigma_prior,
        params     = params_v5,
        init_state = Vector{Float32}(init_state),
        noise_seed = Int(key & typemax(Int32)),
    )

    log_density_fn = make_log_density_fn_v5(ctrl_target)
    rng_ctrl = StableRNG(key)

    # ── Tempered SMC² Logic ──
    smc_kwargs = (
        target_nats        = ctrl_cfg.target_nats,
        target_ess_frac    = ctrl_cfg.target_ess_frac,
        max_lambda_inc     = ctrl_cfg.max_lambda_inc,
        max_temp_levels    = ctrl_cfg.max_levels,
        num_mcmc_steps     = ctrl_cfg.num_mcmc,
        hmc_step_size      = ctrl_cfg.hmc_step,
        hmc_num_leapfrog   = ctrl_cfg.hmc_leap,
        chees_L_candidates = ctrl_cfg.chees_L_candidates,
        h_fd               = 1.0f-4,
        calib_n            = 64,
        verbose            = false,
    )

    U_ctrl::Matrix{Float64}, n_temp_ctrl::Int, diagnostics = if collect_diagnostics
        out = run_tempered_smc_gpu(
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, theta_dim,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
            collect_diagnostics = true,
        )
        (out[1], out[2], out[4])
    else
        out = run_tempered_smc_gpu(
            log_density_fn, ctrl_cfg.M_max, ctrl_cfg.n_smc, theta_dim,
            0.0, ctrl_cfg.sigma_prior, rng_ctrl;
            smc_kwargs...,
        )
        (out[1], out[2], NamedTuple[])
    end

    # Extract posterior mean in Float32
    theta::Vector{Float32} = Float32.(vec(mean(U_ctrl; dims = 1)))

    # ── Plan Decoding ──
    plan = _decode_phi_plans(theta, n_anchors, T_total_bins, dt, 
                             Float32(ctrl_target.c_Phi))

    return (
        Phi_B_plan  = plan.Phi_B, 
        Phi_S_plan  = plan.Phi_S,
        n_temp_ctrl = n_temp_ctrl, 
        theta       = theta,
        diagnostics = diagnostics
    )
end


# =============================================================================
# 2. RBF DECODER (Internal)
# =============================================================================

"""
    _decode_phi_plans(theta, n_anchors, n_steps, dt, c_Phi) -> NamedTuple

Pure functional decoder: transforms RBF coefficients into time-resolved 
training intensities. Uses Float32 for bit-equivalence with the GPU kernel.
"""
function _decode_phi_plans(theta::Vector{Float32}, n_anchors::Int, 
                            n_steps::Int, dt::Float32, c_Phi::Float32)
    
    T_total::Float32 = n_steps * dt
    t_grid::Vector{Float32}  = collect(0:(n_steps-1)) .* dt
    anchors::Vector{Float32} = collect(range(0.0f0, T_total; length = n_anchors))
    σ_rbf::Float32   = T_total / n_anchors
    
    phi_B = zeros(Float32, n_steps)
    phi_S = zeros(Float32, n_steps)
    
    for k in 1:n_steps
        raw_B::Float32 = c_Phi
        raw_S::Float32 = c_Phi
        
        for j in 1:n_anchors
            # Gaussian RBF Kernel
            dist::Float32  = t_grid[k] - anchors[j]
            basis::Float32 = exp(-0.5f0 * (dist / σ_rbf)^2)
            
            raw_B += theta[j]             * basis
            raw_S += theta[n_anchors + j] * basis
        end
        
        # Logistic saturation [0, 3.0]
        phi_B[k] = 3.0f0 / (1.0f0 + exp(-raw_B))
        phi_S[k] = 3.0f0 / (1.0f0 + exp(-raw_S))
    end
    
    return (Phi_B = phi_B, Phi_S = phi_S)
end

end # module BenchController
