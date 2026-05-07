# FSA-v2 control task spec — port of `version_2/models/fsa_high_res/control.py`.
#
# Cost functional (mean over MC noise grid, common random numbers):
#
#     J(θ) = E_τ[ -∫ A(t) dt
#                 + λ_Φ · ∫ Φ(t)² dt
#                 + λ_F · ∫ max(F(t) - F_max, 0)² dt ]
#
# Schedule: 8 Gaussian RBF anchors over the horizon, sigmoid output with
# logit-bias offset so θ at the prior mean → Φ ≈ Φ_default. Search dim 8.

module Control

using LinearAlgebra
using Random
using Statistics

using ..PhiBurst: BINS_PER_DAY, DT_BIN_DAYS
using ..Dynamics: drift, diffusion_state_dep, em_step_substepped, A_TYP, F_TYP,
                   TRUTH_PARAMS as DYN_TRUTH_PARAMS
using ..Simulation: INIT_STATE, EXOGENOUS


# ── RBF schedule basis ───────────────────────────────────────────────────

"""
    build_rbf(n_steps::Int, dt::Float64, n_anchors::Int) -> Matrix(n_steps, n_anchors)

Build a Gaussian-RBF design matrix. Anchors equally spaced on [0, T_total],
bandwidth σ = (T_total / n_anchors) for a smooth basis.
"""
function build_rbf(n_steps::Integer, dt::Real, n_anchors::Integer)
    # RAW Gaussian RBF basis (NOT row-normalised) — matches Python's
    # `smc2fc/control/rbf_schedules.py:RBFSchedule.design_matrix`.
    # Row-normalisation here would smear per-anchor θ variation into a
    # weighted average and produce a flat schedule regardless of θ.
    T_total = n_steps * dt
    t_grid  = collect(0:n_steps-1) .* dt
    anchors = collect(range(0.0, T_total; length=n_anchors))
    σ = T_total / n_anchors
    M = Matrix{Float64}(undef, n_steps, n_anchors)
    @inbounds for k in 1:n_steps, j in 1:n_anchors
        d = t_grid[k] - anchors[j]
        M[k, j] = exp(-0.5 * (d / σ)^2)
    end
    return M
end


"""
    schedule_from_theta_fsa(theta, design, c_Phi, Phi_max) -> Vector(n_steps)

Decode RBF coefficients θ ∈ ℝ^n_anchors → Φ(t) ∈ [0, Phi_max].
Φ(t) = Phi_max · sigmoid(c_Phi + Σ_a design[t, a] · θ[a])
"""
@inline function schedule_from_theta_fsa(theta::AbstractVector,
                                          design::AbstractMatrix,
                                          c_Phi::Real, Phi_max::Real)
    raw = c_Phi .+ design * theta
    return Phi_max .* (1.0 ./ (1.0 .+ exp.(-raw)))
end


# ── Cost functional ──────────────────────────────────────────────────────

"""
    ControlBundle

Bundle holding the closures the SMC²-as-controller consumes:
- `decoder(theta)` → Phi(t)
- `cost_fn(theta)` → scalar cost
- `traj_sample_fn(theta, rng)` → (n_steps, 3) trajectory
"""
struct ControlBundle{D,C,T,P}
    decoder::D
    cost_fn::C
    traj_sample_fn::T
    params::P
    n_steps::Int
    n_anchors::Int
    n_inner::Int
    dt::Float64
    T_total_days::Float64
    F_max::Float64
    Phi_max::Float64
end


"""
    build_control(; T_total_days=42.0, dt_days=1/96, n_substeps=4,
                    n_anchors=8, n_inner=32, F_max=0.40, Phi_max=3.0,
                    Phi_default=1.0, lam_phi=0.0, lam_barrier=1.0,
                    seed=42) -> ControlBundle

Construct an FSA-v2 control bundle for the given horizon.
"""
function build_control(; T_total_days::Real = EXOGENOUS.T_total,
                         dt_days::Real = EXOGENOUS.dt_days,
                         n_substeps::Integer = EXOGENOUS.n_substeps,
                         n_anchors::Integer = 8,
                         n_inner::Integer = 32,
                         F_max::Real = EXOGENOUS.F_max,
                         Phi_max::Real = EXOGENOUS.Phi_max,
                         Phi_default::Real = EXOGENOUS.Phi_default,
                         lam_phi::Real = 0.0,
                         lam_barrier::Real = 1.0,
                         seed::Integer = 42,
                         params = DYN_TRUTH_PARAMS,
                         init_state::AbstractVector = [Float64(INIT_STATE.B),
                                                        Float64(INIT_STATE.F),
                                                        Float64(INIT_STATE.A)])
    n_steps = Int(round(T_total_days / dt_days))
    design  = build_rbf(n_steps, dt_days, n_anchors)
    p_ratio = Phi_default / Phi_max
    c_Phi   = log(p_ratio / (1.0 - p_ratio))

    decoder = θ -> schedule_from_theta_fsa(θ, design, c_Phi, Phi_max)

    init_state = Float64.(collect(init_state))

    # CRN noise grid (matches Python build_crn_noise_grids).
    rng_seed = MersenneTwister(seed)
    fixed_w = randn(rng_seed, n_inner, n_steps, 3)

    function cost_fn(theta::AbstractVector)
        Phi_arr = decoder(theta)
        total = 0.0
        @inbounds for trial in 1:n_inner
            y = copy(init_state)
            A_acc = 0.0; Phi_acc = 0.0; barrier_acc = 0.0
            for k in 1:n_steps
                Phi_t = Phi_arr[k]
                noise = fixed_w[trial, k, :]
                y_next = em_step_substepped(y, params, noise, Phi_t, dt_days;
                                              n_substeps=n_substeps)
                A_acc += y[3] * dt_days                           # ∫A dt
                Phi_acc += Phi_t * Phi_t * dt_days                 # ∫Φ² dt
                barrier_acc += max(y[2] - F_max, 0.0)^2 * dt_days  # ∫max(F-F_max,0)² dt
                y = y_next
            end
            total += -A_acc + lam_phi * Phi_acc + lam_barrier * barrier_acc
        end
        return total / n_inner
    end

    function traj_sample_fn(theta::AbstractVector,
                              rng::AbstractRNG = Random.GLOBAL_RNG)
        Phi_arr = decoder(theta)
        traj = Matrix{Float64}(undef, n_steps, 3)
        y = copy(init_state)
        @inbounds for k in 1:n_steps
            noise = randn(rng, 3)
            y = em_step_substepped(y, params, noise, Phi_arr[k], dt_days;
                                    n_substeps=n_substeps)
            traj[k, :] = y
        end
        return traj
    end

    return ControlBundle(decoder, cost_fn, traj_sample_fn, params,
                          n_steps, n_anchors, n_inner, Float64(dt_days),
                          Float64(T_total_days), Float64(F_max),
                          Float64(Phi_max))
end


export build_rbf, schedule_from_theta_fsa
export ControlBundle, build_control

end # module Control
