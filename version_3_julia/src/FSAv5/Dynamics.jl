# FSAv5/Dynamics.jl — deterministic drift + state-dependent diffusion.
#
# Maps line-by-line to:
#   • LEAN  spec : `FSA_model_dev/lean/Fsa/V5/Drift.lean` (drift, diffusion)
#   • Python    : `models/fsa_high_res/_dynamics.py:177-258` (drift_jax)
#                 `models/fsa_high_res/_dynamics.py:265-287` (diffusion_state_dep)
#   • LaTeX     : §11.1 (full equations) + §10.2 eq:v5-mubar (Hill deconditioning)
#
# Per the LEAN4-first charter: this file is the Julia executable
# counterpart to the LEAN reference. The differential test asserts
# Lean-binary output ≈ Julia output ≈ Python output to within 1e-6
# absolute. Disagreement would be a Julia bug.

# ── Drift ──────────────────────────────────────────────────────────────────

"""
    dynamics_drift(y::FSAv5State, p::DynParams, phi::BimodalPhi)::FSAv5State

FSA-v5 deterministic drift. Returns ẏ = `[dB, dS, dF, dA, dKFB, dKFS]`.

Mirrors `Fsa.V5.drift` in `Drift.lean` and `drift_jax` at
`_dynamics.py:177-258`. The 6D state derivative is computed from the
v5 bifurcation parameter μ̄(B,S,F) (with Hill deconditioning
subtractions), then plugged into the Stuart-Landau / Banister /
unified-fatigue / Busso-K equations.

Setting `p.mu_dec_B = p.mu_dec_S = 0` recovers FSA-v4 numerics
exactly — the v5 Hill terms vanish.
"""
@inline function dynamics_drift(y::FSAv5State, p::DynParams, phi::BimodalPhi)::FSAv5State
    # Destructure (mirrors Lean `let B := y.B; let S := y.S; ...`).
    (B, S, F, A, KFB, KFS) = y
    (phi_B, phi_S) = phi

    # ── Bifurcation parameter μ̄(B, S, F) — FSA-v5 ──
    # See LaTeX §10.2 eq:v5-mubar; Python `_dynamics.py:213-229`.
    #   μ_v5 = μ_v4(B,S,F)
    #        - μ_dec_B · B_dec^n / (B^n + B_dec^n)
    #        - μ_dec_S · S_dec^n / (S^n + S_dec^n)
    F_dev = F - F_TYP
    n     = p.n_dec
    Bn    = max(B, 0.0) ^ n
    Sn    = max(S, 0.0) ^ n
    Bdn   = p.B_dec ^ n
    Sdn   = p.S_dec ^ n
    dec_B = p.mu_dec_B * Bdn / (Bn + Bdn)
    dec_S = p.mu_dec_S * Sdn / (Sn + Sdn)

    mu = p.mu_0 +
         p.mu_B * B + p.mu_S * S -
         p.mu_F * F - p.mu_FF * F_dev * F_dev -
         dec_B - dec_S

    # ── Aerobic capacity B (Banister; autonomic-modulated gain) ──
    # `_dynamics.py:233-234`.
    a_factor_B = (1.0 + p.epsilon_AB * A) / (1.0 + p.epsilon_AB * A_TYP)
    dB = p.kappa_B * a_factor_B * phi_B - B / p.tau_B

    # ── Strength capacity S ──
    # `_dynamics.py:238-239`.
    a_factor_S = (1.0 + p.epsilon_AS * A) / (1.0 + p.epsilon_AS * A_TYP)
    dS = p.kappa_S * a_factor_S * phi_S - S / p.tau_S

    # ── Unified fatigue F (FSA-v4 dynamic gains) ──
    # `_dynamics.py:244-245`.
    a_factor_F = (1.0 + p.lambda_A * A) / (1.0 + p.lambda_A * A_TYP)
    dF = KFB * phi_B + KFS * phi_S - a_factor_F / p.tau_F * F

    # ── Autonomic amplitude A (Stuart-Landau) ──
    # `_dynamics.py:250`.
    dA = mu * A - p.eta * A * A * A

    # ── Busso variable-dose K dynamics (FSA-v4) ──
    # `_dynamics.py:255-256`.
    dKFB = (p.KFB_0 - KFB) / p.tau_K + p.mu_K * phi_B
    dKFS = (p.KFS_0 - KFS) / p.tau_K + p.mu_K * phi_S

    return FSAv5State(dB, dS, dF, dA, dKFB, dKFS)
end

# ── Diffusion (state-dependent diagonal) ───────────────────────────────────

"""
    dynamics_diffusion(y::FSAv5State, p::DynParams)::FSAv5State

Per-component diffusion magnitudes σ_i(y) for the SDE
`dy = drift dt + σ(y) dW`. Mirrors `Fsa.V5.diffusion` in `Drift.lean`
and `diffusion_state_dep` at `_dynamics.py:265-287`.

  - B, S use Jacobi-style: σ_i √(x_i (1 - x_i))   keeps x_i ∈ [0, 1].
  - F, A, K_* use CIR-style: σ_i √(x_i)            keeps x_i ≥ 0.
  - K_FB and K_FS share `sigma_K` (per the canonical Python).

Returns a `FSAv5State` of magnitudes (not signed; positive root only).
"""
@inline function dynamics_diffusion(y::FSAv5State, p::DynParams)::FSAv5State
    (B, S, F, A, KFB, KFS) = y
    return FSAv5State(
        p.sigma_B * sqrt(max(B * (1.0 - B), 0.0)),
        p.sigma_S * sqrt(max(S * (1.0 - S), 0.0)),
        p.sigma_F * sqrt(max(F, 0.0)),
        p.sigma_A * sqrt(max(A, 0.0)),
        p.sigma_K * sqrt(max(KFB, 0.0)),
        p.sigma_K * sqrt(max(KFS, 0.0)),
    )
end

export dynamics_drift, dynamics_diffusion
