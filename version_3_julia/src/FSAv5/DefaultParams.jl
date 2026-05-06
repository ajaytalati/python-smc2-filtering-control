# FSAv5/DefaultParams.jl — canonical truth-parameter sets.
#
# Maps to `Fsa.V5.TRUTH_PARAMS_V5` (in `Types.lean:154-162`) and the
# Python `simulation.DEFAULT_PARAMS_V5` (28 dynamics keys + 22 obs
# keys). Numerical values matched to the LEAN reference verbatim;
# the differential test in `tests/test_lean_diff.jl` will assert
# bit-for-bit equality of these constructors.

# ── FSA-v5 truth dynamics parameters (closed-island basin topology) ────────
# Maps to `_dynamics.py:163-170` and `Types.lean:154-162`. Derived from
# v4-recovering defaults; overrides only the v5 deconditioning entries
# (mu_dec_* = 0.10, B_dec = S_dec = 0.07).
#
# This builder produces an immutable canonical instance. Callers that
# want to scan parameters should `copy(TRUTH_PARAMS_V5)` then mutate.

"""
    TRUTH_PARAMS_V5

Canonical FSA-v5 truth dynamics parameters (`DynParams` ComponentVector,
28 fields). Maps to `_dynamics.py:163-170` and the LEAN4 reference at
`Fsa.V5.Types.TRUTH_PARAMS_V5`.

Setting `mu_dec_B = mu_dec_S = 0` recovers v4 numerics exactly; the
default `0.10` activates the v5 closed-island basin topology described
in LaTeX §10.4.
"""
const TRUTH_PARAMS_V5 = make_dyn_params(
    # ── Aerobic Fitness B (linear in Phi_B, decays with τ_B) ──
    tau_B      = 42.0,
    kappa_B    = 0.012 * (1.0 + 0.40 * A_TYP),  # κ_B^eff in G1 form
    epsilon_AB = 0.40,                           # autonomic boost coefficient
    # ── Strength Adaptation S ──
    tau_S      = 60.0,
    kappa_S    = 0.008 * (1.0 + 0.20 * A_TYP),  # κ_S^eff in G1 form
    epsilon_AS = 0.20,
    # ── Unified Fatigue F (driven by K_FB Phi_B + K_FS Phi_S) ──
    tau_F      = 7.0 / (1.0 + 1.00 * A_TYP),    # τ_F^eff in G1 form
    lambda_A   = 1.00,                           # autonomic-fatigue coupling
    # ── Busso Variable-Dose Sensitivity K_FB, K_FS (FSA-v4) ──
    KFB_0      = 0.030,                          # baseline aerobic fatigue gain
    KFS_0      = 0.050,                          # baseline strength fatigue gain
    tau_K      = 21.0,                           # ~3 weeks recovery timescale
    mu_K       = 0.005,                          # 'damage' rate (Busso 2003)
    # ── Stuart-Landau bifurcation parameter mu(B, S, F) ──
    mu_0       = 0.02 + 0.40 * (F_TYP * F_TYP), # baseline autonomic drive
    mu_B       = 0.30,                           # B → A coupling (positive)
    mu_S       = 0.15,                           # S → A coupling (positive)
    mu_F       = 0.10 + 2.0 * F_TYP * 0.40,     # F → A coupling (negative)
    mu_FF      = 0.40,                           # quadratic F penalty around F_TYP
    eta        = 0.20,                           # cubic damping in A
    # ── State-dependent diffusion scales (frozen in production) ──
    sigma_B    = 0.010,
    sigma_S    = 0.008,    # latent-S Jacobi noise — NOT the stress obs noise
    sigma_F    = 0.012,
    sigma_A    = 0.020,
    sigma_K    = 0.005,    # shared by K_FB and K_FS
    # ── FSA-v5 Hill deconditioning (closed-island calibration §10.4) ──
    B_dec      = 0.07,                           # aerobic-fitness threshold
    S_dec      = 0.07,                           # strength threshold
    mu_dec_B   = 0.10,                           # aerobic decond penalty (v5)
    mu_dec_S   = 0.10,                           # strength decond penalty (v5)
    n_dec      = 4.0,                            # Hill exponent (steepness)
)

# ── Default observation-channel parameters ─────────────────────────────────
# Maps to `simulation.py:392-396` and the LEAN4 `Obs.lean` ObsParams
# section. The stress obs noise is `sigma_S_obs = 4.0` — distinct from
# `TRUTH_PARAMS_V5.sigma_S = 0.008` (the latent state noise).

"""
    DEFAULT_OBS_PARAMS_V5

Canonical FSA-v5 observation-channel parameters (`ObsParams`
ComponentVector, 22 fields). Maps to `simulation.py:392-396` and
`Fsa.V5.ObsParams` defaults.
"""
const DEFAULT_OBS_PARAMS_V5 = make_obs_params(
    # ── HR observation channel (5) — sleep-active ──
    HR_base     = 62.0,
    kappa_B_HR  = 12.0,
    alpha_A_HR  = 3.0,
    beta_C_HR   = -2.5,
    sigma_HR    = 2.0,
    # ── Sleep observation channel (3) — Bernoulli logistic ──
    k_C         = 3.0,
    k_A         = 2.0,
    c_tilde     = 0.5,
    # ── Stress observation channel (5) — wake-only.
    # `sigma_S_obs` is the obs noise — distinct from `DynParams.sigma_S`.
    S_base      = 30.0,
    k_F         = 20.0,
    k_A_S       = 8.0,
    beta_C_S    = -4.0,
    sigma_S_obs = 4.0,
    # ── Steps observation channel (6) — log-Gaussian ──
    mu_step0    = 5.5,
    beta_B_st   = 0.8,
    beta_F_st   = 0.5,
    beta_A_st   = 0.3,
    beta_C_st   = -0.8,
    sigma_st    = 0.5,
    # ── VolumeLoad observation channel (3) ──
    beta_S_VL   = 100.0,
    beta_F_VL   = 20.0,
    sigma_VL    = 10.0,
)

# ── Default initial state (trained athlete) ────────────────────────────────
# Maps to `simulation.py:409` `DEFAULT_INIT` and the v5-bench healthy
# scenario init at `bench_controller_only_fsa_v5.py:88-91`. Used as
# the base-case starting point for plant rollouts and tests.

"""
    DEFAULT_INIT

Canonical FSA-v5 trained-athlete initial state (`FSAv5State`). The
"healthy island" reference point used in the LaTeX §10.4 closed-island
verification (mu_bar(0; (0.30, 0.30)) ≈ +0.011, mono-stable healthy).
"""
const DEFAULT_INIT = FSAv5State(0.50, 0.45, 0.20, 0.45, 0.06, 0.07)

# ── Sedentary-init reference (a deconditioned athlete) ─────────────────────
# Used by Stage 1 filter benches that synthesise from low B, S.

"""
    SEDENTARY_INIT

Sedentary / deconditioned reference initial state. Used by some
diagnostic benches that exercise the Hill-deconditioning regime.
"""
const SEDENTARY_INIT = FSAv5State(0.05, 0.10, 0.30, 0.10, 0.030, 0.050)

export TRUTH_PARAMS_V5, DEFAULT_OBS_PARAMS_V5, DEFAULT_INIT, SEDENTARY_INIT
