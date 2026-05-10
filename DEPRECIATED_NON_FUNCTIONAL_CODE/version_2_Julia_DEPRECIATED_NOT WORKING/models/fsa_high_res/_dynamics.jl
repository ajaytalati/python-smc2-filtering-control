# FSA-v2 dynamics — port of `version_2/models/fsa_high_res/_dynamics.py`
# (G1-reparametrized).
#
# State [B, F, A]:
#   B  fitness     (Banister chronic, Jacobi diffusion in [0, 1])
#   F  fatigue     (Banister acute,   CIR diffusion in [0, ∞))
#   A  amplitude   (Stuart-Landau,    CIR diffusion in [0, ∞))
#
# Drift (/day) — G1-reparametrized around (A_TYP=0.10, F_TYP=0.20):
#
#   F_dev = F - F_TYP
#   μ(B,F) = μ_0 + μ_B·B - μ_F·F - μ_FF·F_dev²
#
#   a_factor_B = (1 + ε_A·A) / (1 + ε_A·A_TYP)
#   dB/dt = κ_B · a_factor_B · Φ - B / τ_B
#
#   a_factor_F = (1 + λ_A·A) / (1 + λ_A·A_TYP)
#   dF/dt = κ_F·Φ - a_factor_F / τ_F · F
#
#   dA/dt = μ·A - η·A³
#
# At the operating point A = A_TYP, F = F_TYP both a_factor_X reduce to 1
# and μ collapses to μ_0 + μ_B·B - μ_F·F.
#
# Diffusion (state-dependent Itô — unchanged from v2):
#   σ_B · √(B(1-B)) · dW_B   (Jacobi)
#   σ_F · √F        · dW_F   (CIR)
#   σ_A · √A        · dW_A   (CIR)

module Dynamics

# ── Operating-point reference constants (G1) ─────────────────────────────
const A_TYP   = 0.10     # initial-state A; representative of de-trained subject
const F_TYP   = 0.20     # mid-window F under typical Φ at typical A
const PHI_TYP = 1.0      # canonical Banister default (1 unit of TRIMP/day)


# ── Truth parameters (Set A v2, G1-reparametrized) ───────────────────────
# Drift formulas at these new truth values are mathematically equivalent to
# the v2 spec (the reparametrization is a coordinate change).

const TRUTH_PARAMS = (
    # Banister timescales + (effective) gains
    tau_B    = 42.0,
    tau_F    = 7.0 / (1.0 + 1.00 * A_TYP),       # = 6.3636…  τ_F^eff
    kappa_B  = 0.012 * (1.0 + 0.40 * A_TYP),     # = 0.01248  κ_B^eff
    kappa_F  = 0.030,
    epsilon_A = 0.40,                             # residual (tighter prior)
    lambda_A  = 1.00,                             # residual (tighter prior)

    # Stuart-Landau bifurcation parameter (reparametrized around F_TYP)
    mu_0  = 0.02 + 0.40 * (F_TYP ^ 2),           # = 0.036    μ_0^eff
    mu_B  = 0.30,
    mu_F  = 0.10 + 2.0 * F_TYP * 0.40,           # = 0.26     μ_F^eff
    mu_FF = 0.40,                                 # residual curvature
    eta   = 0.20,

    # State-dependent diffusion (frozen)
    sigma_B = 0.010,
    sigma_F = 0.012,
    sigma_A = 0.020,
)


# ── Drift ────────────────────────────────────────────────────────────────

"""
    drift(y::AbstractVector, params, Φ_t::Real) -> Vector

Per-day drift d[B, F, A]/dt — G1-reparametrized. `params` is anything
with `.tau_B`, `.kappa_B`, … fields (NamedTuple, ComponentVector, struct).
"""
@inline function drift(y::AbstractVector, params, Φ_t::Real)
    B, F, A = y[1], y[2], y[3]

    F_dev = F - F_TYP
    μ = params.mu_0 + params.mu_B * B -
        params.mu_F * F - params.mu_FF * F_dev * F_dev

    a_factor_B = (1.0 + params.epsilon_A * A) / (1.0 + params.epsilon_A * A_TYP)
    dB = params.kappa_B * a_factor_B * Φ_t - B / params.tau_B

    a_factor_F = (1.0 + params.lambda_A * A) / (1.0 + params.lambda_A * A_TYP)
    dF = params.kappa_F * Φ_t - a_factor_F / params.tau_F * F

    dA = μ * A - params.eta * A^3
    return [dB, dF, dA]
end


# ── Drift on positionally-indexed parameter vector ───────────────────────
# The framework's `propagate_fn` interface receives params as a flat
# `Vector` indexed by integer position (matching Python's `params[_PI[k]]`).
# Used by estimation.jl. Caller passes the index-of-param dict as `pi`.

@inline function drift_indexed(y::AbstractVector, params::AbstractVector,
                                pi::Dict{Symbol,Int}, Φ_t::Real)
    B, F, A = y[1], y[2], y[3]
    F_dev = F - F_TYP
    μ = params[pi[:mu_0]] + params[pi[:mu_B]] * B -
        params[pi[:mu_F]] * F - params[pi[:mu_FF]] * F_dev * F_dev

    a_factor_B = (1.0 + params[pi[:epsilon_A]] * A) /
                 (1.0 + params[pi[:epsilon_A]] * A_TYP)
    dB = params[pi[:kappa_B]] * a_factor_B * Φ_t - B / params[pi[:tau_B]]

    a_factor_F = (1.0 + params[pi[:lambda_A]] * A) /
                 (1.0 + params[pi[:lambda_A]] * A_TYP)
    dF = params[pi[:kappa_F]] * Φ_t - a_factor_F / params[pi[:tau_F]] * F

    dA = μ * A - params[pi[:eta]] * A^3
    return [dB, dF, dA]
end


# ── Diffusion ────────────────────────────────────────────────────────────

"""
    diffusion_state_dep(y, params) -> Vector

State-dependent diagonal diffusion (unchanged from v2). Each component
vanishes at its domain boundary (B at 0 or 1, F and A at 0).
"""
@inline function diffusion_state_dep(y::AbstractVector, params)
    B, F, A = y[1], y[2], y[3]
    return [
        params.sigma_B * sqrt(max(B * (1.0 - B), 0.0)),
        params.sigma_F * sqrt(max(F, 0.0)),
        params.sigma_A * sqrt(max(A, 0.0)),
    ]
end


# ── Substepped Euler-Maruyama with boundary reflection ───────────────────

"""
    em_step_substepped(y, params, noise, Φ_t, dt; n_substeps=4)

Substepped EM mirroring `_dynamics.py:imex_step_substepped`:
- `n_substeps` deterministic drift sub-steps,
- ONE Wiener increment of variance σ(y_det)²·dt at the outer boundary,
- boundary reflection (B → -B if B<0; B → 2-B if B>1; F, A → |·|).
"""
function em_step_substepped(y::AbstractVector, params,
                            noise::AbstractVector, Φ_t::Real, dt::Real;
                            n_substeps::Integer = 4)
    sub_dt = dt / float(n_substeps)
    y_inner = collect(y)
    @inbounds for _ in 1:n_substeps
        y_inner = y_inner .+ sub_dt .* drift(y_inner, params, Φ_t)
    end

    σ_y    = diffusion_state_dep(y_inner, params)
    y_pred = y_inner .+ σ_y .* sqrt(dt) .* noise

    B_pred, F_pred, A_pred = y_pred[1], y_pred[2], y_pred[3]
    B_next = B_pred < 0.0 ? -B_pred :
             (B_pred > 1.0 ? 2.0 - B_pred : B_pred)
    F_next = abs(F_pred)
    A_next = abs(A_pred)
    return [B_next, F_next, A_next]
end


export A_TYP, F_TYP, PHI_TYP, TRUTH_PARAMS
export drift, drift_indexed, diffusion_state_dep, em_step_substepped

end # module Dynamics
