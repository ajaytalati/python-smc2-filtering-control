"""
    DynamicsV5

Core dynamics engine for the FSA-v5 model, providing deterministic drift, 
state-dependent diffusion, and Euler-Maruyama integration steps.

This module is a machine-precision transcription of the formal Lean4 
specification (`version_1_5_LEAN/Fsa/V5/Drift.lean`).

# State Vector y(t) = [B, S, F, A, KFB, KFS]
- B: Aerobic Fitness (Banister chronic, Jacobi diffusion in [0,1])
- S: Strength Capacity (Banister chronic, Jacobi diffusion in [0,1])
- F: Unified Fatigue (CIR diffusion in [0, ∞))
- A: Autonomic Amplitude (Stuart-Landau, CIR diffusion in [0, ∞))
- KFB: Aerobic Fatigue Gain (Busso variable-dose)
- KFS: Strength Fatigue Gain (Busso variable-dose)

# Drift Equations (/day)
- dB/dt    = κ_B · a_B(A) · Φ_B − B / τ_B
- dS/dt    = κ_S · a_S(A) · Φ_S − S / τ_S
- dF/dt    = K_FB · Φ_B + K_FS · Φ_S − a_F(A)/τ_F · F
- dA/dt    = μ(B,S,F) · A − η · A³
- dK_FB/dt = (K_FB_0 − K_FB) / τ_K + μ_K · Φ_B
- dK_FS/dt = (K_FS_0 − K_FS) / τ_K + μ_K · Φ_S

# Bifurcation Parameter μ(B,S,F)
μ = μ_0 + μ_B B + μ_S S − μ_F F − μ_FF (F − F_TYP)²
    − μ_dec_B · h_n(B; B_dec) − μ_dec_S · h_n(S; S_dec)
where h_n(x; x_dec) = x_dec^n / (x^n + x_dec^n) is the Hill weight.

Implementation Note: All mathematical operations use `Float32` to ensure high-
performance execution on GPU targets.
"""
module DynamicsV5

using StaticArrays

import ..SimulationV5: A_TYP, F_TYP

export drift_v5, diffusion_v5, em_step_v5


# =============================================================================
# 1. DETERMINISTIC DRIFT (dY/dt)
# =============================================================================

"""
    drift_v5(y, params, phi) -> SVector{6, Float32}

Calculates the per-day drift for the 6D latent state.
Implements Parameter Centering around typical operating points (A_TYP, F_TYP) 
to maximize filter identifiability.

Arguments:
- `y`: Current state vector [B, S, F, A, K_FB, K_FS].
- `params`: Parameter set (Dict or NamedTuple).
- `phi`: Stimulus tuple (Phi_B, Phi_S).
"""
@inline function drift_v5(y::AbstractVector{Float32}, params,
                            phi::Tuple{Float32, Float32})
    B, S, F, A, KFB, KFS = y[1], y[2], y[3], y[4], y[5], y[6]
    Phi_B, Phi_S = phi[1], phi[2]

    # ── Bifurcation parameter μ(B, S, F) ──
    # Centering on F_TYP reduces multi-collinearity in the quadratic cliff.
    F_dev = F - F_TYP
    n     = _get(params, :n_dec)
    Bn    = max(B, 0.0f0)^n
    Sn    = max(S, 0.0f0)^n
    Bdn   = _get(params, :B_dec)^n
    Sdn   = _get(params, :S_dec)^n
    dec_B = _get(params, :mu_dec_B) * Bdn / (Bn + Bdn)
    dec_S = _get(params, :mu_dec_S) * Sdn / (Sn + Sdn)

    μ = _get(params, :mu_0) +
         _get(params, :mu_B) * B +
         _get(params, :mu_S) * S -
         _get(params, :mu_F) * F -
         _get(params, :mu_FF) * F_dev * F_dev -
         dec_B - dec_S

    # ── Chronic adaptation dynamics (Banister) ──
    # a_factor modulates gains based on autonomic readiness (A).
    a_factor_B = (1.0f0 + _get(params, :epsilon_AB) * A) /
                  (1.0f0 + _get(params, :epsilon_AB) * A_TYP)
    dB = _get(params, :kappa_B) * a_factor_B * Phi_B -
          B / _get(params, :tau_B)

    a_factor_S = (1.0f0 + _get(params, :epsilon_AS) * A) /
                  (1.0f0 + _get(params, :epsilon_AS) * A_TYP)
    dS = _get(params, :kappa_S) * a_factor_S * Phi_S -
          S / _get(params, :tau_S)

    # ── Fatigue & Sensitivity (Busso) ──
    a_factor_F = (1.0f0 + _get(params, :lambda_A) * A) /
                  (1.0f0 + _get(params, :lambda_A) * A_TYP)
    dF = KFB * Phi_B + KFS * Phi_S -
          a_factor_F / _get(params, :tau_F) * F

    # ── Autonomic state (Stuart-Landau) ──
    dA = μ * A - _get(params, :eta) * A * A * A

    # ── Fatigue Gain relaxation ──
    dKFB = (_get(params, :KFB_0) - KFB) / _get(params, :tau_K) +
            _get(params, :mu_K) * Phi_B
    dKFS = (_get(params, :KFS_0) - KFS) / _get(params, :tau_K) +
            _get(params, :mu_K) * Phi_S

    return SVector{6, Float32}(dB, dS, dF, dA, dKFB, dKFS)
end


# =============================================================================
# 2. STOCHASTIC DIFFUSION
# =============================================================================

"""
    diffusion_v5(y, params) -> SVector{6, Float32}

State-dependent diagonal diffusion magnitudes.
- Jacobi √(x(1-x)) ensures B,S remain in [0, 1].
- CIR √x ensures F,A,K remain non-negative.
"""
@inline function diffusion_v5(y::AbstractVector{Float32}, params)
    B, S, F, A, KFB, KFS = y[1], y[2], y[3], y[4], y[5], y[6]
    return SVector{6, Float32}(
        _get(params, :sigma_B) * sqrt(max(B * (1.0f0 - B), 0.0f0)),
        _get(params, :sigma_S) * sqrt(max(S * (1.0f0 - S), 0.0f0)),
        _get(params, :sigma_F) * sqrt(max(F, 0.0f0)),
        _get(params, :sigma_A) * sqrt(max(A, 0.0f0)),
        _get(params, :sigma_K) * sqrt(max(KFB, 0.0f0)),
        _get(params, :sigma_K) * sqrt(max(KFS, 0.0f0)),
    )
end


# =============================================================================
# 3. NUMERICAL INTEGRATION (Euler-Maruyama)
# =============================================================================

const _EPS_B = 1.0f-4
const _EPS_S = 1.0f-4
const _EPS_A = 1.0f-4

@inline _clamp01(x::Real, ε::Real) = clamp(x, ε, 1.0f0 - ε)

"""
    em_step_v5(y, phi, params, sigma_diag, dt, noise) -> SVector{6, Float32}

Advances the 6D latent state by one bin step using the Euler-Maruyama scheme.
Includes post-step clipping to ensure states remain within their physical domains.
"""
function em_step_v5(y::AbstractVector{Float32},
                     phi::Tuple{Float32, Float32},
                     params,
                     sigma_diag::AbstractVector{Float32},
                     dt::Float32,
                     noise::AbstractVector{Float32})
    d_y      = drift_v5(y, params, phi)
    sqrt_dt  = sqrt(dt)

    # State-dependent diffusion magnitudes.
    B_cl   = _clamp01(y[1], _EPS_B)
    S_cl   = _clamp01(y[2], _EPS_S)
    F_cl   = max(y[3], 0.0f0)
    A_cl   = max(y[4], 0.0f0)
    KFB_cl = max(y[5], 0.0f0)
    KFS_cl = max(y[6], 0.0f0)

    g_B   = sqrt(B_cl * (1.0f0 - B_cl))
    g_S   = sqrt(S_cl * (1.0f0 - S_cl))
    g_F   = sqrt(F_cl)
    g_A   = sqrt(A_cl + _EPS_A)
    g_KFB = sqrt(KFB_cl)
    g_KFS = sqrt(KFS_cl)

    # Update: Y_{n+1} = Y_n + Δt·drift(Y_n) + σ·g(Y_n)·√Δt·ξ
    yB_next   = y[1] + dt * d_y[1] + sigma_diag[1] * g_B   * sqrt_dt * noise[1]
    yS_next   = y[2] + dt * d_y[2] + sigma_diag[2] * g_S   * sqrt_dt * noise[2]
    yF_next   = y[3] + dt * d_y[3] + sigma_diag[3] * g_F   * sqrt_dt * noise[3]
    yA_next   = y[4] + dt * d_y[4] + sigma_diag[4] * g_A   * sqrt_dt * noise[4]
    yKFB_next = y[5] + dt * d_y[5] + sigma_diag[5] * g_KFB * sqrt_dt * noise[5]
    yKFS_next = y[6] + dt * d_y[6] + sigma_diag[6] * g_KFS * sqrt_dt * noise[6]

    # Post-step domain enforcement
    return SVector{6, Float32}(
        _clamp01(yB_next, _EPS_B),
        _clamp01(yS_next, _EPS_S),
        max(yF_next,   0.0f0),
        max(yA_next,   0.0f0),
        max(yKFB_next, 0.0f0),
        max(yKFS_next, 0.0f0),
    )
end


# =============================================================================
# 4. INTERNAL DISPATCH HELPERS
# =============================================================================

@inline _get(p::Dict{Symbol, Float32}, k::Symbol) = p[k]
@inline _get(p::NamedTuple, k::Symbol)           = getproperty(p, k)
@inline _get(p::Dict{Symbol, Float64}, k::Symbol) = Float32(p[k])

end # module DynamicsV5
