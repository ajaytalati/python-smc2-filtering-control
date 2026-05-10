# FSA-v5 cost-function primitives — pure Julia transcription of
# `version_1_5_LEAN/Fsa/V5/Cost.lean`.
#
# Three primitives:
#   - `mu_bar(A, phi, params)` — slow-manifold μ̄(A; Φ); the v5 Stuart-Landau
#       drive evaluated on the equilibrium manifold
#       (B*, S*, F*, K_FB*, K_FS*) parametrised by A.
#   - `find_a_sep(phi, params)` — separatrix root A_sep(Φ) of g(A) = μ̄(A) − η A²,
#       returning ±Inf sentinels for the mono-stable healthy / collapsed
#       regimes and a finite root in the bistable regime.
#   - `a_sep_grid(particles, schedule)` — per-particle, per-bin separator
#       matrix of shape `(n_particles, n_steps)`. The signature deliberately
#       prevents "Bug 2" (the historical particle-0 collapse) — every
#       particle gets its own separator under its own params.
#
# Tech guide §4.1 (lines 428-440) gives the slow-manifold equations;
# §4.2 (lines 450-465) the three regimes. The Lean file is the
# line-by-line reference for the constants (N_SEP_GRID = 64,
# N_SEP_BISECT = 40, A_MIN = 1e-4, A_MAX = 2.0).
#
# These primitives are diff-tested against the Lean reference at 1e-6.
# The HARD/SOFT cost wrappers (`evaluate_chance_constrained_cost_*`)
# from tech guide §5 are Julia-only compositions and are tested via the
# §8 smoke tests, not against Lean.

module CostV5

using StaticArrays

import ..SimulationV5: A_TYP, F_TYP

export mu_bar, find_a_sep, a_sep_grid


# ── Algorithm constants — mirror Fsa/V5/Cost.lean ──────────────────────

const _N_SEP_GRID   = 64       # grid points for sign-change scan
const _N_SEP_BISECT = 40       # bisection iterations once a bracket is found
const _A_MIN        = 1.0e-4   # lower bound of search bracket
const _A_MAX        = 2.0      # upper bound of search bracket


# ── Slow-manifold effective Stuart-Landau coefficient μ̄(A; Φ) ─────────

"""
    mu_bar(A, phi, params) -> Float64

The FSA-v5 effective Stuart-Landau coefficient on the slow manifold
under bimodal stimulus `phi = (Phi_B, Phi_S)` and parameters `params`.

Tech guide §4.1 lines 428-440 / Lean `Fsa.V5.muBar`
(`Fsa/V5/Cost.lean:47-69`).

The slow-manifold equilibria of (B, S, F, K_FB, K_FS) are substituted
in closed form (each parametrised by A); the v5 Hill-deconditioning
subtractions are applied. Returns the resulting μ̄(A; Φ) scalar.
"""
function mu_bar(A::Real,
                  phi::Tuple{<:Real, <:Real},
                  params)
    Phi_B, Phi_S = phi[1], phi[2]
    a_B = (1.0 + _get(params, :epsilon_AB) * A) /
           (1.0 + _get(params, :epsilon_AB) * A_TYP)
    a_S = (1.0 + _get(params, :epsilon_AS) * A) /
           (1.0 + _get(params, :epsilon_AS) * A_TYP)
    a_F = (1.0 + _get(params, :lambda_A) * A) /
           (1.0 + _get(params, :lambda_A) * A_TYP)
    B   = _get(params, :tau_B) * _get(params, :kappa_B) * a_B * Phi_B
    S   = _get(params, :tau_S) * _get(params, :kappa_S) * a_S * Phi_S
    KFB = _get(params, :KFB_0) +
           _get(params, :tau_K) * _get(params, :mu_K) * Phi_B
    KFS = _get(params, :KFS_0) +
           _get(params, :tau_K) * _get(params, :mu_K) * Phi_S
    F   = _get(params, :tau_F) * (KFB * Phi_B + KFS * Phi_S) / a_F
    F_dev = F - F_TYP
    n   = _get(params, :n_dec)
    Bn  = max(B, 0.0)^n
    Sn  = max(S, 0.0)^n
    Bdn = _get(params, :B_dec)^n
    Sdn = _get(params, :S_dec)^n
    dec_B = _get(params, :mu_dec_B) * Bdn / (Bn + Bdn)
    dec_S = _get(params, :mu_dec_S) * Sdn / (Sn + Sdn)
    return _get(params, :mu_0) +
            _get(params, :mu_B) * B +
            _get(params, :mu_S) * S -
            _get(params, :mu_F) * F -
            _get(params, :mu_FF) * F_dev * F_dev -
            dec_B - dec_S
end


# ── Separatrix root-finder ─────────────────────────────────────────────

"""
    find_a_sep(phi, params) -> Float64

Find the bistable separatrix `A_sep(Φ)` under `params`. Three-way
return value matching the tech guide §4.2 regimes:

  - `-Inf` → mono-stable healthy: `g(A_min) > 0`, so `A = 0` is unstable
            and there is no separator.
  - `+Inf` → mono-stable collapsed: `g(A) < 0` everywhere on the search
            interval; `A = 0` is the global attractor.
  - finite → bistable regime: the smaller positive root of `g(A) =
            μ̄(A) − η · A²`.

Algorithm: 64-point grid over `[A_MIN, A_MAX]`, scan for the first
sign change `g(a_i) < 0 ∧ g(a_{i+1}) > 0`, then 40-step bisection on
the bracket. Mirrors `Fsa.V5.findASep` (`Fsa/V5/Cost.lean:134-153`).
"""
function find_a_sep(phi::Tuple{<:Real, <:Real}, params)
    g_at = (A) -> mu_bar(A, phi, params) - _get(params, :eta) * A * A
    a_grid = _a_grid()
    g_vals = g_at.(a_grid)

    # mono-stable healthy: g(A_min) > 0 ⇒ A = 0 is unstable
    if g_vals[1] > 0.0
        return -Inf
    end

    # find first sign change negative → positive
    n = length(g_vals)
    sign_change_idx = 0
    for i in 1:(n - 1)
        if g_vals[i] < 0.0 && g_vals[i + 1] > 0.0
            sign_change_idx = i
            break
        end
    end

    if sign_change_idx == 0
        # no sign change: g stays negative ⇒ mono-stable collapsed
        return Inf
    end

    a0 = a_grid[sign_change_idx]
    b0 = a_grid[sign_change_idx + 1]
    return _bisect(g_at, a0, b0)
end

# Linear-spaced grid over [_A_MIN, _A_MAX] with _N_SEP_GRID points.
# Matches `aGrid` in `Fsa/V5/Cost.lean:95-98`.
function _a_grid()
    n = _N_SEP_GRID
    if n <= 1
        return Float64[_A_MIN]
    end
    step = (_A_MAX - _A_MIN) / (n - 1)
    return [_A_MIN + (i - 1) * step for i in 1:n]
end

# 40-step bisection: at each step, narrow the bracket [a, b] toward
# the root of g. If g(mid) < 0 the root is in [mid, b]; else in [a, mid].
# Mirrors `bisect` in `Fsa/V5/Cost.lean:117-127`.
function _bisect(g_at, lo::Real, hi::Real)
    a = float(lo)
    b = float(hi)
    for _ in 1:_N_SEP_BISECT
        mid = 0.5 * (a + b)
        if g_at(mid) < 0.0
            a = mid
        else
            b = mid
        end
    end
    return 0.5 * (a + b)
end


# ── Per-particle, per-bin separator grid ───────────────────────────────

"""
    a_sep_grid(particles, schedule) -> Matrix{Float64}

For each (particle, schedule-bin) pair, compute `find_a_sep`. Returns
a `(n_particles, n_steps)` matrix.

The signature is deliberately structured so that the historical
particle-0-template bug ("Bug 2" in the v5 implementation pass) is
impossible to express: every particle's separator is independently
computed under its own params. Lean reference: `Fsa.V5.aSepGrid`
(`Fsa/V5/Cost.lean:173-175`).
"""
function a_sep_grid(particles::AbstractVector,
                     schedule::AbstractVector)
    n_p = length(particles)
    n_s = length(schedule)
    out = Matrix{Float64}(undef, n_p, n_s)
    @inbounds for i in 1:n_p
        p_i = particles[i]
        for j in 1:n_s
            phi_j = schedule[j]
            phi_t = (phi_j[1], phi_j[2]) :: Tuple{Float64, Float64}
            out[i, j] = find_a_sep(phi_t, p_i)
        end
    end
    return out
end


# ── Internal: dual access for Dict / NamedTuple params ────────────────
@inline _get(p::Dict{Symbol, T}, k::Symbol) where {T}    = p[k]
@inline _get(p::NamedTuple, k::Symbol)                    = getproperty(p, k)

end # module CostV5
