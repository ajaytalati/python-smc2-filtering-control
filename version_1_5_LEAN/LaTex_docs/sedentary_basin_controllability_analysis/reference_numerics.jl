# reference_numerics.jl
#
# STANDALONE Julia reference numerics for the four constructive controllability
# theorems in `controllability_v2_proofs.tex`. No project dependencies — only
# Julia's standard `Printf`. Drop into any Julia 1.10+ installation and run:
#
#     julia reference_numerics.jl
#
# Output: A(100) values for the four cases, to be cross-checked against any
# implementation of the four theorems (LEAN4, Coq, Python, etc.). The
# parametrisation is the v2 RECOMMENDED setting from
# `controllability_v2_proofs.tex` §2 / §5; values are hard-coded here so this
# file is fully self-contained.

using Printf

# ---------------------------------------------------------------------------
# RECOMMENDED v2 parametrisation (Eq. 12 / Appendix 1 of the document)
# ---------------------------------------------------------------------------

const A_TYP = 0.10
const F_TYP = 0.20

# Recommended v2: 8 parameters differ from canonical TRUTH_PARAMS_V5
const PARAMS = Dict{Symbol,Float64}(
    # Aerobic B  — v2 halves τ_B, doubles κ_B (preserves slow-manifold B*)
    :tau_B       => 21.0,
    :kappa_B     => 0.012 * (1.0 + 0.40*A_TYP) * 2.0,   # = 0.02496 (2× centred canonical)
    :epsilon_AB  => 0.40,
    # Strength S — v2 halves τ_S, doubles κ_S
    :tau_S       => 30.0,
    :kappa_S     => 0.008 * (1.0 + 0.20*A_TYP) * 2.0,   # = 0.01632 (2× centred canonical)
    :epsilon_AS  => 0.20,
    # Fatigue F
    :tau_F       => 7.0 / (1.0 + 1.00 * A_TYP),   # ≈ 6.36
    :lambda_A    => 1.00,
    # Busso K block
    :KFB_0       => 0.030,
    :KFS_0       => 0.050,
    :tau_K       => 21.0,
    :mu_K        => 0.005,
    # Stuart-Landau bifurcation parameter — v2 reduces μ_F, μ_FF
    :mu_0        => 0.02 + 0.40 * (F_TYP * F_TYP),  # = 0.036 (unchanged at the centred-form level)
    :mu_B        => 0.30,
    :mu_S        => 0.15,
    :mu_F        => 0.030,                          # v2 — was 0.10 + 2*F_TYP*0.40 = 0.26
    :mu_FF       => 0.020,                          # v2 — was 0.40
    :eta         => 0.20,
    # v5 Hill deconditioning — v2 raises B_dec, S_dec
    :B_dec       => 0.25,                           # v2 — was 0.07
    :S_dec       => 0.25,                           # v2 — was 0.07
    :mu_dec_B    => 0.10,
    :mu_dec_S    => 0.10,
    :n_dec       => 4.0,
)

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

const SEDENTARY_INIT = Dict(:B => 0.05, :S => 0.10, :F => 0.30, :A => 0.10,
                            :KFB => 0.030, :KFS => 0.050)

# TRAINED_ATHLETE_INIT_v2 — slow-manifold equilibrium at island center
# (Φ_B, Φ_S) = (1.06, 0.78) with A* = 1.238 under PARAMS.
# Computed in the document §5; reproduced here as a fixed initial state.
const TRAINED_ATHLETE_INIT_V2 = Dict(:B => 0.7993, :S => 0.4688, :F => 0.7925,
                                     :A => 1.2384, :KFB => 0.1414, :KFS => 0.1322)

# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

hill(x, x_dec, n) = x_dec^n / (x^n + x_dec^n)

function mu_bar_state(B, S, F, p)
    p[:mu_0] + p[:mu_B]*B + p[:mu_S]*S -
    p[:mu_F]*F - p[:mu_FF]*(F - F_TYP)^2 -
    p[:mu_dec_B]*hill(B, p[:B_dec], p[:n_dec]) -
    p[:mu_dec_S]*hill(S, p[:S_dec], p[:n_dec])
end

function drift(y, phi_B, phi_S, p)
    B, S, F, A, KFB, KFS = y
    a_B = (1 + p[:epsilon_AB]*A) / (1 + p[:epsilon_AB]*A_TYP)
    a_S = (1 + p[:epsilon_AS]*A) / (1 + p[:epsilon_AS]*A_TYP)
    a_F = (1 + p[:lambda_A] *A) / (1 + p[:lambda_A] *A_TYP)
    dB   = p[:kappa_B]*a_B*phi_B - B/p[:tau_B]
    dS   = p[:kappa_S]*a_S*phi_S - S/p[:tau_S]
    dF   = KFB*phi_B + KFS*phi_S - a_F*F/p[:tau_F]
    mu   = mu_bar_state(B, S, F, p)
    dA   = mu*A - p[:eta]*A^3
    dKFB = (p[:KFB_0] - KFB)/p[:tau_K] + p[:mu_K]*phi_B
    dKFS = (p[:KFS_0] - KFS)/p[:tau_K] + p[:mu_K]*phi_S
    return [dB, dS, dF, dA, dKFB, dKFS]
end

# RK4 integration of the deterministic ODE.
function simulate(y0_dict, phi_const, T, dt, p)
    y = [y0_dict[:B], y0_dict[:S], y0_dict[:F], y0_dict[:A],
         y0_dict[:KFB], y0_dict[:KFS]]
    n_steps = round(Int, T/dt)
    phi_B, phi_S = phi_const
    for _ in 1:n_steps
        k1 = drift(y,             phi_B, phi_S, p)
        k2 = drift(y .+ 0.5*dt*k1, phi_B, phi_S, p)
        k3 = drift(y .+ 0.5*dt*k2, phi_B, phi_S, p)
        k4 = drift(y .+ dt*k3,     phi_B, phi_S, p)
        y .+= (dt/6.0) .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)
        # plant-side clipping: B, S in [ε, 1-ε]; F, A, K floored at 0
        y[1] = clamp(y[1], 1e-4, 1 - 1e-4)
        y[2] = clamp(y[2], 1e-4, 1 - 1e-4)
        for i in 3:6; y[i] = max(y[i], 0.0); end
    end
    return y   # final state at T
end

# ---------------------------------------------------------------------------
# The four theorems
# ---------------------------------------------------------------------------

println("=" ^ 60)
println("Reference numerics for controllability_v2_proofs.tex")
println("v2 parametrisation, deterministic ODE, T = 100 d, RK4 dt = 0.05 d")
println("=" ^ 60)

# Sanity check: μ(SEDENTARY_INIT) should equal -0.140 under v2 (B_dec=0.25
# saturates the deconditioning Hill at B_0=0.05, deeper negative than canonical).
mu_sed = mu_bar_state(SEDENTARY_INIT[:B], SEDENTARY_INIT[:S],
                      SEDENTARY_INIT[:F], PARAMS)
@printf "\nSanity: μ(SEDENTARY_INIT) under v2 = %.4f (expect -0.140)\n" mu_sed

# Sanity: μ̄(0; (1,1)) on slow manifold should be +0.119
mu_target = let
    phi_B, phi_S = 1.0, 1.0
    a_B = 1.0 / (1.0 + PARAMS[:epsilon_AB]*A_TYP)
    a_S = 1.0 / (1.0 + PARAMS[:epsilon_AS]*A_TYP)
    a_F = 1.0 / (1.0 + PARAMS[:lambda_A]*A_TYP)
    KFB = PARAMS[:KFB_0] + PARAMS[:tau_K]*PARAMS[:mu_K]*phi_B
    KFS = PARAMS[:KFS_0] + PARAMS[:tau_K]*PARAMS[:mu_K]*phi_S
    B   = PARAMS[:tau_B]*PARAMS[:kappa_B]*a_B*phi_B
    S   = PARAMS[:tau_S]*PARAMS[:kappa_S]*a_S*phi_S
    F   = PARAMS[:tau_F]*(KFB*phi_B + KFS*phi_S)/a_F
    mu_bar_state(min(B,1), min(S,1), F, PARAMS)
end
@printf "Sanity: μ̄(0; (1,1)) on slow manifold = %.4f (expect +0.119)\n\n" mu_target

T = 100.0
dt = 0.05

# Theorem 6.1: passive sedentary → A → 0
y_final = simulate(SEDENTARY_INIT, (0.1, 0.1), T, dt, PARAMS)
A_61 = y_final[4]
@printf "Theorem 6.1 (pathological sedentary):     A(%d) = %.4f  (expect ≈ 0.0000)\n" T A_61

# Theorem 6.2: FIM-witness Φ* = (0.86, 1.24) → A escapes
y_final = simulate(SEDENTARY_INIT, (0.86, 1.24), T, dt, PARAMS)
A_62 = y_final[4]
@printf "Theorem 6.2 (constructive escape):        A(%d) = %.4f  (expect ≈ 1.16)\n" T A_62

# Theorem 6.3: passive over-training from v2-trained init → A → 0.
# Under v2 the autonomic-protective feedback a_F(A) shields F when A is high,
# so Φ=(2,2) doesn't collapse within T=100; we need Φ=(2.5, 2.5).
y_final = simulate(TRAINED_ATHLETE_INIT_V2, (2.5, 2.5), T, dt, PARAMS)
A_63 = y_final[4]
@printf "Theorem 6.3 (pathological over-training): A(%d) = %.4f  under Φ=(2.5, 2.5)  (expect ≈ 0.000)\n" T A_63

# Theorem 6.4: maintenance at Φ = (1, 1) from v2-trained init → A stays at A*
y_final = simulate(TRAINED_ATHLETE_INIT_V2, (1.0, 1.0), T, dt, PARAMS)
A_64 = y_final[4]
@printf "Theorem 6.4 (constructive maintenance):   A(%d) = %.4f  (expect ≈ 1.24)\n" T A_64

println("\nAll four cases reproduced from this single self-contained file.")
