# FSAv5/Cost.jl — chance-constraint primitives (μ̄, A_sep, per-particle grid).
#
# Maps line-by-line to:
#   • LEAN  spec : `FSA_model_dev/lean/Fsa/V5/Cost.lean`
#   • Python    : `models/fsa_high_res/control_v5.py:429-521`
#                 (`_jax_mu_bar`, `_jax_find_A_sep`, `_compute_cost_internals`)
#   • LaTeX     : §10.2 eq:v5-mubar (Hill bifurcation parameter)
#                 §10.4 closed-island Table 1 (regime classification)
#
# Bug 2 structural prevention (charter Part I §4.2): the Python
# `_compute_cost_internals` collapsed the SMC² particle ensemble to
# particle-0 before computing `A_sep`, then broadcast that single
# template across all particles' trajectories — mathematically wrong
# because the separator depends on each particle's bifurcation
# parameters.
#
# Here, `a_sep_grid` returns one separator per particle per bin. The
# Julia type signature `Matrix{Float64}` of shape `(n_particles,
# n_steps)` makes the buggy collapse to `Vector{Float64}` of shape
# `(n_steps,)` impossible.

using SMC2FC: ChanceConstraintMode, HardIndicator, SoftSurrogate

# ── Slow-manifold effective Stuart-Landau coefficient μ̄(A; Φ) ────────────

"""
    mu_bar(A::Float64, phi::BimodalPhi, p::DynParams)::Float64

The FSA-v5 effective Stuart-Landau coefficient on the slow manifold.
Mirrors `Fsa.V5.muBar` in `Cost.lean` and `_jax_mu_bar` at
`control_v5.py:429-462`. Maps to LaTeX §10.2 eq:v5-mubar.

Substitutes the slow-manifold equilibria B*(A), S*(A), F*(A) into the
v4 form, then applies the v5 Hill-deconditioning subtractions.
"""
@inline function mu_bar(A::Float64, phi::BimodalPhi, p::DynParams)::Float64
    (phi_B, phi_S) = phi
    a_B = (1.0 + p.epsilon_AB * A) / (1.0 + p.epsilon_AB * A_TYP)
    a_S = (1.0 + p.epsilon_AS * A) / (1.0 + p.epsilon_AS * A_TYP)
    a_F = (1.0 + p.lambda_A  * A) / (1.0 + p.lambda_A  * A_TYP)
    B   = p.tau_B * p.kappa_B * a_B * phi_B
    S   = p.tau_S * p.kappa_S * a_S * phi_S
    KFB = p.KFB_0 + p.tau_K * p.mu_K * phi_B
    KFS = p.KFS_0 + p.tau_K * p.mu_K * phi_S
    F   = p.tau_F * (KFB * phi_B + KFS * phi_S) / a_F

    F_dev = F - F_TYP
    n     = p.n_dec
    Bn    = max(B, 0.0) ^ n
    Sn    = max(S, 0.0) ^ n
    Bdn   = p.B_dec ^ n
    Sdn   = p.S_dec ^ n
    dec_B = p.mu_dec_B * Bdn / (Bn + Bdn)
    dec_S = p.mu_dec_S * Sdn / (Sn + Sdn)

    return p.mu_0 +
           p.mu_B * B + p.mu_S * S -
           p.mu_F * F - p.mu_FF * F_dev * F_dev -
           dec_B - dec_S
end

# ── Bistable separatrix root-finder ────────────────────────────────────────
# Mirrors `Fsa.V5.findASep` in `Cost.lean` and `_jax_find_A_sep` at
# `control_v5.py:465-521`. Three-way return matches the Python's
# mathematical contract:
#   - `-Inf` — mono-stable healthy regime (A=0 unstable; no separatrix)
#   - finite scalar in (A_MIN, A_MAX) — bistable separator root
#   - `+Inf` — mono-stable collapsed regime
#
# Algorithm: scan a 64-point grid for first sign change of g(A) =
# μ̄(A) - η·A², then bisect 40 iterations. Pure scalar, allocation-
# free in the hot path (no array allocation).

const N_SEP_GRID   = 64
const N_SEP_BISECT = 40
const A_SEP_MIN    = 1e-4
const A_SEP_MAX    = 2.0

@inline _g_at(A::Float64, phi::BimodalPhi, p::DynParams) =
    mu_bar(A, phi, p) - p.eta * A * A

"""
    find_a_sep(phi::BimodalPhi, p::DynParams)::Float64

Locate the bistable separatrix `A_sep(Φ)` under one particle's
parameters. Returns `-Inf` (mono-stable healthy), `+Inf` (mono-stable
collapsed), or the smaller positive root of `g(A) = μ̄(A;Φ) - η·A²` in
the bistable regime.

Mirrors `Fsa.V5.findASep` in `Cost.lean` and `_jax_find_A_sep` at
`control_v5.py:465-521`.
"""
function find_a_sep(phi::BimodalPhi, p::DynParams)::Float64
    # Scan grid for sign change.
    A_min  = A_SEP_MIN
    A_max  = A_SEP_MAX
    n      = N_SEP_GRID
    step   = (A_max - A_min) / (n - 1)

    # Mono-stable healthy: g(A_min) > 0 ⇒ A=0 is unstable, no separator.
    g_first = _g_at(A_min, phi, p)
    is_healthy = g_first > 0.0

    # Find first sign change from negative to positive — first index `i`
    # with g(A_i) < 0 and g(A_{i+1}) > 0. We compute g lazily, only
    # storing the previous value across iterations so the scan is
    # allocation-free.
    has_sep = false
    a0 = A_min
    b0 = A_min + step
    g_prev = g_first
    A_curr = A_min
    for i in 1:(n - 1)
        A_next = A_min + i * step
        g_next = _g_at(A_next, phi, p)
        if !has_sep && g_prev < 0.0 && g_next > 0.0
            has_sep = true
            a0 = A_curr
            b0 = A_next
        end
        g_prev = g_next
        A_curr = A_next
    end

    # Bisection within [a0, b0].
    a, b = a0, b0
    for _ in 1:N_SEP_BISECT
        mid   = 0.5 * (a + b)
        g_mid = _g_at(mid, phi, p)
        if g_mid < 0.0
            a = mid
        else
            b = mid
        end
    end
    root = 0.5 * (a + b)

    # Three-way return matches the Python and Lean contract.
    if is_healthy
        return -Inf
    elseif has_sep
        return root
    else
        return Inf
    end
end

# ── Per-particle, per-bin separator grid (kills Bug 2) ─────────────────────

"""
    a_sep_grid(particles::AbstractVector{<:DynParams},
                schedule::AbstractVector{BimodalPhi})::Matrix{Float64}

Compute `A_sep(Φ_t; θ_m)` at every (particle m, bin t) pair.

Returns a `Matrix{Float64}` of shape `(n_particles, n_steps)`. The
historical Python `_compute_cost_internals` collapsed this to
`Vector{Float64}` of shape `(n_steps,)` by using particle-0's params
as a "template" — mathematically wrong because the separator depends
on each particle's bifurcation parameters. The Julia type signature
above makes the buggy collapse impossible to write.

Mirrors `Fsa.V5.aSepGrid` in `Cost.lean`.
"""
function a_sep_grid(particles::AbstractVector{<:DynParams},
                     schedule::AbstractVector{BimodalPhi})::Matrix{Float64}
    n_p = length(particles)
    n_t = length(schedule)
    out = Matrix{Float64}(undef, n_p, n_t)
    @inbounds for m in 1:n_p
        p_m = particles[m]
        for t in 1:n_t
            out[m, t] = find_a_sep(schedule[t], p_m)
        end
    end
    return out
end

# ── Chance-constraint indicator dispatch ───────────────────────────────────
# Marker types `HardIndicator` and `SoftSurrogate` are defined in SMC2FC
# (`SMC2FC.ChanceConstraintMode`). Two methods of `chance_indicator`
# replace the Python `if cost_kind == 'soft'` branch — the dispatcher
# picks at compile time. Per charter §15.1.

"""
    chance_indicator(::HardIndicator, A_traj_pp, A_sep_pp; kwargs...)
    chance_indicator(::SoftSurrogate, A_traj_pp, A_sep_pp; beta, scale)

Per-particle-per-bin indicator that A_t < A_sep(Φ_t).
- `HardIndicator`: discrete `(A_traj < A_sep)` — non-differentiable
  in θ; suitable for pure-SMC² importance weighting.
- `SoftSurrogate`: `sigmoid(beta · (A_sep - A_traj) / scale)` —
  C^∞ differentiable; suitable for HMC.

Both methods accept ±Inf in `A_sep_pp` (mono-stable regimes):
  - sigmoid(+inf) = 1 (collapsed → always violates)
  - sigmoid(-inf) = 0 (healthy → never violates)
"""
@inline chance_indicator(::HardIndicator, A_traj_pp::AbstractMatrix,
                          A_sep_pp::AbstractMatrix) =
    Float64.(A_traj_pp .< A_sep_pp)

@inline chance_indicator(::SoftSurrogate, A_traj_pp::AbstractMatrix,
                          A_sep_pp::AbstractMatrix;
                          beta::Float64 = 50.0,
                          scale::Float64 = 0.1) =
    @. 1.0 / (1.0 + exp(-beta * (A_sep_pp - A_traj_pp) / scale))

export mu_bar, find_a_sep, a_sep_grid, chance_indicator
export A_SEP_MIN, A_SEP_MAX, N_SEP_GRID, N_SEP_BISECT
