# FSA-v2 (Banister-coupled) dynamics — direct port of
# `version_1/models/fsa_high_res/_dynamics.py`.
#
# State [B, F, A]:
#   B  fitness     (Banister chronic, Jacobi diffusion in [0, 1])
#   F  fatigue     (Banister acute,   CIR diffusion in [0, ∞))
#   A  amplitude   (Stuart-Landau,    CIR diffusion in [0, ∞))
#
# Drift (/day):
#   μ(B, F) = μ_0 + μ_B·B − μ_F·F − μ_FF·F²
#   dB/dt = κ_B · (1 + ε_A · A) · Φ  −  B / τ_B
#   dF/dt = κ_F · Φ                  −  (1 + λ_A · A) / τ_F · F
#   dA/dt = μ · A − η · A³
#
# Diffusion (state-dependent Itô):
#   σ_B · √(B(1−B)) · dW_B   (Jacobi)
#   σ_F · √F        · dW_F   (CIR)
#   σ_A · √A        · dW_A   (CIR)
#
# Single exogenous control: Φ(t), training-strain rate ≥ 0.

module Dynamics

# ── Truth parameters (Set A v2, verbatim Python copy) ───────────────────────

const TRUTH_PARAMS = (
    # Banister timescales + gains
    tau_B    = 42.0,
    tau_F    =  7.0,
    kappa_B  = 0.012,
    kappa_F  = 0.030,

    # A-coupling
    epsilon_A = 0.40,
    lambda_A  = 1.00,

    # Stuart-Landau
    mu_0  = 0.02,
    mu_B  = 0.30,
    mu_F  = 0.10,
    mu_FF = 0.40,
    eta   = 0.20,

    # State-dependent diffusion
    sigma_B = 0.010,
    sigma_F = 0.012,
    sigma_A = 0.020,
)


# ── Drift ───────────────────────────────────────────────────────────────────

"""
    drift(y::AbstractVector, params, Φ_t::Real) -> Vector

Per-day drift d[B, F, A]/dt at the given state and control input.
"""
@inline function drift(y::AbstractVector, params, Φ_t::Real)
    B, F, A = y[1], y[2], y[3]
    μ  = params.mu_0 + params.mu_B * B -
          params.mu_F * F - params.mu_FF * F * F
    dB = params.kappa_B * (1.0 + params.epsilon_A * A) * Φ_t - B / params.tau_B
    dF = params.kappa_F * Φ_t -
          (1.0 + params.lambda_A * A) / params.tau_F * F
    dA = μ * A - params.eta * A^3
    return [dB, dF, dA]
end


# ── Diffusion ───────────────────────────────────────────────────────────────

"""
    diffusion_state_dep(y, params) -> Vector

State-dependent diagonal diffusion. Each component vanishes at its
domain boundary (B at 0 or 1, F and A at 0) so the SDE keeps each
state in its physiological range without clipping.
"""
@inline function diffusion_state_dep(y::AbstractVector, params)
    B, F, A = y[1], y[2], y[3]
    return [
        params.sigma_B * sqrt(max(B * (1.0 - B), 0.0)),
        params.sigma_F * sqrt(max(F, 0.0)),
        params.sigma_A * sqrt(max(A, 0.0)),
    ]
end


# ── Substepped Euler-Maruyama with boundary reflection ─────────────────────

"""
    em_step_substepped(y, params, noise::AbstractVector, Φ_t, dt;
                        n_substeps = 4)

Substepped EM: `n_substeps` deterministic drift sub-steps followed by
ONE Wiener increment of variance σ(y)²·dt at the outer boundary.
Boundaries are enforced by reflection (`B → -B` if B < 0; `B → 2-B` if
B > 1; `F, A → |·|`).
"""
function em_step_substepped(y::AbstractVector, params,
                              noise::AbstractVector, Φ_t::Real, dt::Real;
                              n_substeps::Integer = 4)
    sub_dt = dt / float(n_substeps)
    y_inner = copy(y)
    for _ in 1:n_substeps
        y_inner = y_inner .+ sub_dt .* drift(y_inner, params, Φ_t)
    end

    σ_y    = diffusion_state_dep(y_inner, params)
    y_pred = y_inner .+ σ_y .* sqrt(dt) .* noise

    # Boundary reflection
    B_pred, F_pred, A_pred = y_pred[1], y_pred[2], y_pred[3]
    B_next = B_pred < 0.0 ? -B_pred :
              (B_pred > 1.0 ? 2.0 - B_pred : B_pred)
    F_next = abs(F_pred)
    A_next = abs(A_pred)
    return [B_next, F_next, A_next]
end

end # module Dynamics
