# FSA-v2 cost functional + RBF schedule decoder — port of
# `version_1/models/fsa_high_res/control.py`.
#
# Cost (mean over MC noise grid, common random numbers):
#
#     J(θ) = E_τ [ −∫A(t)dt + λ_Φ·∫Φ(t)²dt + λ_F·∫max(F(t)−F_max, 0)²dt ]
#
# The schedule is parameterised by 8 Gaussian RBF anchors with a
# sigmoid output transform biased so θ = 0 → Φ ≈ Φ_default.

module Control

using Random: AbstractRNG, MersenneTwister
using Statistics: mean
import ..Dynamics: TRUTH_PARAMS, drift, diffusion_state_dep, em_step_substepped
import ..Simulation: INIT_STATE, EXOGENOUS

export build_control, ControlBundle, schedule_from_theta_fsa

# ── Schedule decoder: θ ∈ ℝ^n_anchors → Φ(t) ───────────────────────────────

"""
    build_rbf(n_steps, dt, n_anchors; width_factor = 1.0) -> Φ_design (n_steps, n_anchors)

Build a Gaussian-RBF design matrix; same shape as Python's `RBFSchedule`.
"""
function build_rbf(n_steps::Integer, dt::Real, n_anchors::Integer;
                    width_factor::Real = 1.0)
    T_total = n_steps * dt
    centres = collect(range(0.0, T_total; length = n_anchors))
    width   = (T_total / max(n_anchors, 1)) * width_factor
    t_grid  = collect(range(0.0; length = n_steps, step = dt))
    Φ = zeros(Float64, n_steps, n_anchors)
    @inbounds for a in 1:n_anchors, t in 1:n_steps
        δ = (t_grid[t] - centres[a]) / width
        Φ[t, a] = exp(-0.5 * δ * δ)
    end
    return Φ
end

@inline _sigmoid(x) = 1.0 / (1.0 + exp(-x))
@inline _logit(p)   = log(p / (1.0 - p))

"""
    schedule_from_theta_fsa(θ::AbstractVector, Φ_design;
                              Phi_max, Phi_default) -> Vector

Decode 8-D RBF coefficients θ into a (n_steps,) Φ schedule.

    Φ(t) = Φ_max · sigmoid( c_Φ + θ' · Φ_design[t, :] )

with `c_Φ = logit(Φ_default / Φ_max)` so θ = 0 → Φ ≡ Φ_default.
"""
function schedule_from_theta_fsa(θ::AbstractVector,
                                   Φ_design::AbstractMatrix;
                                   Phi_max::Real     = EXOGENOUS.Phi_max,
                                   Phi_default::Real = EXOGENOUS.Phi_default)
    c_Φ = _logit(Phi_default / Phi_max)
    raw = Φ_design * θ              # (n_steps,)
    return Phi_max .* _sigmoid.(c_Φ .+ raw)
end


# ── Cost functional with CRN noise grid ─────────────────────────────────────

"""
    ControlBundle holds everything the SMC²-as-controller needs for a
    single horizon: the schedule decoder + the cost callable + the
    trajectory sampler for diagnostics.
"""
struct ControlBundle{Fsched,Fcost,Ftraj}
    n_steps::Int
    dt::Float64
    n_anchors::Int
    Φ_design::Matrix{Float64}
    schedule_from_theta::Fsched
    cost_fn::Fcost
    traj_sample_fn::Ftraj
end

"""
    build_control(; T_total_days, dt_days, n_anchors, n_inner, seed,
                    lam_phi, lam_barrier, F_max, n_substeps,
                    Phi_max, Phi_default, params)

Build the schedule decoder + cost callable + trajectory sampler for
one horizon. Default params, dt, F_max, Phi_*, n_substeps come from
`Simulation.EXOGENOUS` and `Dynamics.TRUTH_PARAMS`.
"""
function build_control(; T_total_days::Real = EXOGENOUS.T_total,
                          dt_days::Real     = EXOGENOUS.dt_days,
                          n_anchors::Int    = 8,
                          n_inner::Int      = 32,
                          seed::Int         = 42,
                          lam_phi::Real     = 0.0,
                          lam_barrier::Real = 1.0,
                          F_max::Real       = EXOGENOUS.F_max,
                          n_substeps::Int   = EXOGENOUS.n_substeps,
                          Phi_max::Real     = EXOGENOUS.Phi_max,
                          Phi_default::Real = EXOGENOUS.Phi_default,
                          params            = TRUTH_PARAMS)
    n_steps  = Int(round(T_total_days / dt_days))
    Φ_design = build_rbf(n_steps, dt_days, n_anchors)

    # Pre-sample CRN Wiener increments (n_inner, n_steps, 3)
    rng_w   = MersenneTwister(seed)
    fixed_w = randn(rng_w, n_inner, n_steps, 3)

    init_arr = [INIT_STATE.B, INIT_STATE.F, INIT_STATE.A]

    function _decode(θ)
        return schedule_from_theta_fsa(θ, Φ_design;
                                          Phi_max = Phi_max,
                                          Phi_default = Phi_default)
    end

    function _cost(θ::AbstractVector)
        Φ_arr = _decode(θ)
        total = 0.0
        for n in 1:n_inner
            y = copy(init_arr)
            A_acc = 0.0; Φ_acc = 0.0; bar_acc = 0.0
            for k in 1:n_steps
                Φ_t = Φ_arr[k]
                noise = view(fixed_w, n, k, :)
                y_next = em_step_substepped(y, params, noise, Φ_t, dt_days;
                                              n_substeps = n_substeps)
                A_acc   += y[3] * dt_days                       # use OLD A (matches Python step accumulator)
                Φ_acc   += Φ_t * Φ_t * dt_days
                bar_acc += max(y[2] - F_max, 0.0)^2 * dt_days
                y = y_next
            end
            total += -A_acc + lam_phi * Φ_acc + lam_barrier * bar_acc
        end
        return total / n_inner
    end

    function _traj_sample(θ::AbstractVector; rng_seed::Int = 0)
        Φ_arr = _decode(θ)
        rng = MersenneTwister(rng_seed)
        traj = zeros(Float64, n_steps, 3)
        y = copy(init_arr)
        for k in 1:n_steps
            noise = randn(rng, 3)
            y = em_step_substepped(y, params, noise, Φ_arr[k], dt_days;
                                     n_substeps = n_substeps)
            traj[k, :] = y
        end
        return traj
    end

    return ControlBundle(
        n_steps, Float64(dt_days), n_anchors, Φ_design,
        _decode, _cost, _traj_sample,
    )
end

end # module Control
