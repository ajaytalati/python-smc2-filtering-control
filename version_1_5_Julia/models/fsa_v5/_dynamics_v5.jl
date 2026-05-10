# FSA-v5 deterministic dynamics — pure Julia transcription of
# `version_1_5_LEAN/Fsa/V5/Drift.lean`. The Lean file is the single
# source of truth; this file is differentially tested against the Lean
# binary at machine precision.
#
# State [B, S, F, A, K_FB, K_FS]:
#   B   aerobic fitness   (Banister chronic, Jacobi diffusion in [0,1])
#   S   strength capacity (Banister chronic, Jacobi diffusion in [0,1])
#   F   unified fatigue   (CIR diffusion in [0, ∞))
#   A   autonomic         (Stuart-Landau, CIR diffusion in [0, ∞))
#   K_FB aerobic fatigue gain   (Busso, CIR diffusion in [0, ∞))
#   K_FS strength fatigue gain  (Busso, CIR diffusion in [0, ∞))
#
# Drift (/day) — see tech guide §2.2 (lines 217-231):
#   dB/dt    = κ_B  · a_B(A) · Φ_B  − B / τ_B
#   dS/dt    = κ_S  · a_S(A) · Φ_S  − S / τ_S
#   dF/dt    = K_FB · Φ_B + K_FS · Φ_S − a_F(A)/τ_F · F
#   dA/dt    = μ(B,S,F) · A − η · A³
#   dK_FB/dt = (K_FB^0 − K_FB) / τ_K + μ_K · Φ_B
#   dK_FS/dt = (K_FS^0 − K_FS) / τ_K + μ_K · Φ_S
#
# v5 bifurcation parameter μ(B,S,F) — tech guide eq:mu-v5 (line 242):
#   μ = μ_0 + μ_B B + μ_S S − μ_F F − μ_FF (F − F_TYP)²
#       − μ_dec_B · h_n(B; B_dec) − μ_dec_S · h_n(S; S_dec)
# where h_n(x; x_dec) = x_dec^n / (x^n + x_dec^n) is the Hill weight.
# Setting μ_dec_B = μ_dec_S = 0 recovers FSA-v4 exactly.
#
# Diffusion (state-dependent Itô) — tech guide §2.4:
#   σ_B √(B(1−B)) dW_B    (Jacobi for B, S; vanishes at boundary)
#   σ_S √(S(1−S)) dW_S
#   σ_F √F        dW_F    (CIR for F, A, K_*; vanishes at zero)
#   σ_A √A        dW_A
#   σ_K √K_FB     dW_KFB
#   σ_K √K_FS     dW_KFS

module DynamicsV5

using StaticArrays

import ..SimulationV5: A_TYP, F_TYP

export drift_v5, diffusion_v5, em_step_v5


# ── Drift ──────────────────────────────────────────────────────────────
#
# Transcribed from `Fsa/V5/Drift.lean:35-95`. `params` may be a
# `Dict{Symbol, Float64}` (the convenient form used by tests and the
# bench driver) or any object with the v5 parameter fields as
# properties / keys. The code uses dict-keyword access uniformly to keep
# the call surface simple; for a NamedTuple-keyed call site, use
# `SimulationV5.params_dict_to_nt` first.

"""
    drift_v5(y, params, phi) -> SVector{6, Float64}

Per-day drift d[B, S, F, A, K_FB, K_FS]/dt at the given state and
bimodal control input `phi = (Phi_B, Phi_S)`.

Mirrors `Fsa.V5.drift` (`Fsa/V5/Drift.lean:35-95`).
"""
@inline function drift_v5(y::AbstractVector, params,
                            phi::Tuple{<:Real, <:Real})
    B, S, F, A, KFB, KFS = y[1], y[2], y[3], y[4], y[5], y[6]
    Phi_B, Phi_S = phi[1], phi[2]

    # ── v5 bifurcation parameter μ(B, S, F) ───────────────────────────
    # See tech guide §2.2 eq:mu-v5; Lean Drift.lean:45-67.
    F_dev = F - F_TYP
    n     = _get(params, :n_dec)
    Bn    = max(B, 0.0)^n
    Sn    = max(S, 0.0)^n
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

    # ── Aerobic capacity B (Banister; A-modulated gain) ───────────────
    # Lean Drift.lean:69-72.
    a_factor_B = (1.0 + _get(params, :epsilon_AB) * A) /
                  (1.0 + _get(params, :epsilon_AB) * A_TYP)
    dB = _get(params, :kappa_B) * a_factor_B * Phi_B -
          B / _get(params, :tau_B)

    # ── Strength capacity S ───────────────────────────────────────────
    # Lean Drift.lean:74-77.
    a_factor_S = (1.0 + _get(params, :epsilon_AS) * A) /
                  (1.0 + _get(params, :epsilon_AS) * A_TYP)
    dS = _get(params, :kappa_S) * a_factor_S * Phi_S -
          S / _get(params, :tau_S)

    # ── Unified fatigue F (FSA-v4 dynamic gains) ──────────────────────
    # Lean Drift.lean:79-82.
    a_factor_F = (1.0 + _get(params, :lambda_A) * A) /
                  (1.0 + _get(params, :lambda_A) * A_TYP)
    dF = KFB * Phi_B + KFS * Phi_S -
          a_factor_F / _get(params, :tau_F) * F

    # ── Autonomic amplitude A (Stuart-Landau) ─────────────────────────
    # Lean Drift.lean:84-86.
    dA = μ * A - _get(params, :eta) * A * A * A

    # ── Busso variable-dose K dynamics ────────────────────────────────
    # Lean Drift.lean:88-93. Linear relaxation to baseline + stimulus
    # damage. Slow-manifold equilibrium K* = K_0 + tau_K · mu_K · Phi.
    dKFB = (_get(params, :KFB_0) - KFB) / _get(params, :tau_K) +
            _get(params, :mu_K) * Phi_B
    dKFS = (_get(params, :KFS_0) - KFS) / _get(params, :tau_K) +
            _get(params, :mu_K) * Phi_S

    return SVector{6, Float64}(dB, dS, dF, dA, dKFB, dKFS)
end


# ── Diffusion ──────────────────────────────────────────────────────────

"""
    diffusion_v5(y, params) -> SVector{6, Float64}

State-dependent diagonal diffusion vector. Each component vanishes at
the relevant boundary so the SDE keeps each state in its physiological
range. Mirrors `Fsa.V5.diffusion` (`Fsa/V5/Drift.lean:105-117`).

Note: K_FB and K_FS share the single scale `sigma_K` — empirically
justified per the tech guide because K dynamics are slow.
"""
@inline function diffusion_v5(y::AbstractVector, params)
    B, S, F, A, KFB, KFS = y[1], y[2], y[3], y[4], y[5], y[6]
    return SVector{6, Float64}(
        _get(params, :sigma_B) * sqrt(max(B * (1.0 - B), 0.0)),
        _get(params, :sigma_S) * sqrt(max(S * (1.0 - S), 0.0)),
        _get(params, :sigma_F) * sqrt(max(F, 0.0)),
        _get(params, :sigma_A) * sqrt(max(A, 0.0)),
        _get(params, :sigma_K) * sqrt(max(KFB, 0.0)),
        _get(params, :sigma_K) * sqrt(max(KFS, 0.0)),
    )
end


# ── EM step ────────────────────────────────────────────────────────────
# Mirrors `Fsa.V5.emStep` at `Fsa/V5/Plant.lean:49-80`.
#
# The Lean version uses CLAMP / FLOOR (not reflection) at the
# boundary — diverges from v1.5 which uses reflection on B. Tech guide
# §6.4 lines 794-803 mandates clamp.

const _EPS_B = 1.0e-4
const _EPS_S = 1.0e-4
const _EPS_A = 1.0e-4

@inline _clamp01(x::Real, ε::Real) = clamp(x, ε, 1.0 - ε)

"""
    em_step_v5(y, phi, params, sigma_diag, dt, noise) -> SVector{6, Float64}

One Euler-Maruyama step at a single bin.

Arguments mirror `Fsa.V5.emStep` (`Fsa/V5/Plant.lean:49-80`):
  - `y`           : current 6D state.
  - `phi`         : `(Phi_B, Phi_S)` per-bin stimulus.
  - `params`      : v5 parameter dict (28 fields).
  - `sigma_diag`  : 6-vector of diffusion-scale overrides. Pass
                     `[params[:sigma_B], params[:sigma_S], params[:sigma_F],
                       params[:sigma_A], params[:sigma_K], params[:sigma_K]]`
                     for the production setting.
  - `dt`          : bin width in days (e.g. 1/96).
  - `noise`       : 6-vector of standard-normal samples drawn upstream
                     (pre-drawn for diff-test bit-equivalence).

Returns the next state with B, S clipped to `[ε, 1-ε]` and F, A,
K_FB, K_FS floored at 0 (clamp/floor convention from the tech guide,
NOT reflection).
"""
function em_step_v5(y::AbstractVector,
                     phi::Tuple{<:Real, <:Real},
                     params,
                     sigma_diag::AbstractVector,
                     dt::Real,
                     noise::AbstractVector)
    d_y      = drift_v5(y, params, phi)
    sqrt_dt  = sqrt(dt)

    # State-dependent diffusion magnitudes.
    # Lean Plant.lean:54-66.
    B_cl   = _clamp01(y[1], _EPS_B)
    S_cl   = _clamp01(y[2], _EPS_S)
    F_cl   = max(y[3], 0.0)
    A_cl   = max(y[4], 0.0)
    KFB_cl = max(y[5], 0.0)
    KFS_cl = max(y[6], 0.0)

    g_B   = sqrt(B_cl * (1.0 - B_cl))
    g_S   = sqrt(S_cl * (1.0 - S_cl))
    g_F   = sqrt(F_cl)
    g_A   = sqrt(A_cl + _EPS_A)
    g_KFB = sqrt(KFB_cl)
    g_KFS = sqrt(KFS_cl)

    # y_next = y + dt · drift + sigma_diag · g(y) · √dt · noise
    yB_next   = y[1] + dt * d_y[1] + sigma_diag[1] * g_B   * sqrt_dt * noise[1]
    yS_next   = y[2] + dt * d_y[2] + sigma_diag[2] * g_S   * sqrt_dt * noise[2]
    yF_next   = y[3] + dt * d_y[3] + sigma_diag[3] * g_F   * sqrt_dt * noise[3]
    yA_next   = y[4] + dt * d_y[4] + sigma_diag[4] * g_A   * sqrt_dt * noise[4]
    yKFB_next = y[5] + dt * d_y[5] + sigma_diag[5] * g_KFB * sqrt_dt * noise[5]
    yKFS_next = y[6] + dt * d_y[6] + sigma_diag[6] * g_KFS * sqrt_dt * noise[6]

    # Boundary handling — clamp/floor.
    return SVector{6, Float64}(
        _clamp01(yB_next, _EPS_B),
        _clamp01(yS_next, _EPS_S),
        max(yF_next,   0.0),
        max(yA_next,   0.0),
        max(yKFB_next, 0.0),
        max(yKFS_next, 0.0),
    )
end


# ── Internal helper: dual access for Dict / NamedTuple params ─────────
# v1.5 accomplishes this via `@match`; here we use a plain dispatch on
# the access-method since `getproperty` and `getindex` cover the two
# cases without needing the Match.jl dependency.

@inline _get(p::Dict{Symbol, T}, k::Symbol) where {T}    = p[k]
@inline _get(p::NamedTuple, k::Symbol)                    = getproperty(p, k)

end # module DynamicsV5
