"""
    Transforms.jl

Constrained ↔ unconstrained bijections via multiple dispatch over a
`PriorType` hierarchy. Mirrors `smc2fc/transforms/unconstrained.py`
(Python reference) and `julia/SMC2FC/src/Transforms.jl` (existing port).

Two API surfaces are provided:

1. **General**: `priors::Vector{<:PriorType}` (heterogeneous element types).
    Each invocation dispatches per-element at runtime. Allocates a fresh
    output vector via `similar`.
2. **Hot-path (fast)**: `priors::Tuple{Vararg{PriorType}}` (heterogeneous
    tuple). Each position has a known concrete type, so dispatch is fully
    resolved at compile time. Uses a `MVector` / `SVector` output for a
    fixed parameter count; suitable for inner-loop calls.

The Python reference has to carry seven indicator arrays (`is_log`,
`is_logit`, …) because JAX cannot dispatch on type. Julia can — each prior
gets its own method.
"""

using LogExpFunctions: log1pexp     # numerically-stable log(1 + exp(u))
using StaticArrays: MVector, SVector

# ── Prior-type hierarchy ─────────────────────────────────────────────────────

"""Abstract base of the prior-type hierarchy. Subtypes carry the parameters
of a univariate prior plus its associated bijection."""
abstract type PriorType end

"""
    LogNormalPrior(μ, σ)

LogNormal prior with location `μ` and scale `σ` (both on log-scale).

# Fields
- `μ::Float64`: mean of `log(θ)`.
- `σ::Float64`: standard deviation of `log(θ)`.
"""
struct LogNormalPrior <: PriorType
    μ::Float64
    σ::Float64
end

"""
    NormalPrior(μ, σ)

Normal prior with mean `μ` and standard deviation `σ`. Identity bijection.
"""
struct NormalPrior <: PriorType
    μ::Float64
    σ::Float64
end

"""
    VonMisesPrior(μ, κ)

Von Mises prior on the circle, with location `μ` and concentration `κ`.
Bijection is identity on the unconstrained tangent space.
"""
struct VonMisesPrior <: PriorType
    μ::Float64
    κ::Float64
end

"""
    BetaPrior(α, β)

Beta prior on `(0, 1)` with shape parameters `α, β`. Bijection is the
logit transform (constrained → log(θ / (1−θ))).
"""
struct BetaPrior <: PriorType
    α::Float64
    β::Float64
end

# ── Component-wise bijections ────────────────────────────────────────────────

"""
    to_unconstrained(prior, θ) -> Real

Map a single constrained scalar `θ` to its unconstrained representation
under `prior`.

# Arguments
- `prior::PriorType`: the prior (multiple-dispatched).
- `θ::Real`: constrained-space scalar.

# Returns
- `Real`: unconstrained-space scalar `u`.
"""
to_unconstrained(::LogNormalPrior, θ::Real) = log(max(θ, 1e-30))
to_unconstrained(::NormalPrior,    θ::Real) = θ
to_unconstrained(::VonMisesPrior,  θ::Real) = θ
function to_unconstrained(::BetaPrior, θ::Real)
    c = clamp(θ, 1e-6, 1.0 - 1e-6)
    return log(c / (1.0 - c))
end

"""
    to_constrained(prior, u) -> Real

Inverse of `to_unconstrained`. Maps a single unconstrained scalar `u` back
into the constrained domain of `prior`.

# Arguments
- `prior::PriorType`: the prior.
- `u::Real`: unconstrained-space scalar.

# Returns
- `Real`: constrained-space scalar `θ`.
"""
to_constrained(::LogNormalPrior, u::Real) = exp(clamp(u, -20.0, 20.0))
to_constrained(::NormalPrior,    u::Real) = u
to_constrained(::VonMisesPrior,  u::Real) = u
to_constrained(::BetaPrior,      u::Real) = 1.0 / (1.0 + exp(-u))

# ── Component-wise log prior in unconstrained space ──────────────────────────

"""
    log_prior_unconstrained(prior, u) -> Real

Log-prior density of a single component evaluated in unconstrained space.
The change-of-variables Jacobian is folded in.

# Arguments
- `prior::PriorType`: the prior.
- `u::Real`: unconstrained-space scalar.

# Returns
- `Real`: log p(θ(u)) + log |dθ/du|.
"""
log_prior_unconstrained(p::LogNormalPrior, u::Real) =
    -0.5 * ((u - p.μ) / p.σ)^2 - log(p.σ)

log_prior_unconstrained(p::NormalPrior, u::Real) =
    -0.5 * ((u - p.μ) / p.σ)^2 - log(p.σ)

log_prior_unconstrained(p::VonMisesPrior, u::Real) =
    p.κ * cos(u - p.μ)

log_prior_unconstrained(p::BetaPrior, u::Real) =
    -p.α * log1pexp(-u) - p.β * log1pexp(u)


# ── Vector-level operations (general, allocating) ────────────────────────────

"""
    constrained_to_unconstrained(θ, priors) -> Vector

Map a constrained parameter vector `θ` to its unconstrained representation
component-by-component.

# Arguments
- `θ::AbstractVector{<:Real}`: constrained vector, length `n`.
- `priors::Vector{<:PriorType}`: prior list, length `n`.

# Returns
- `Vector`: a freshly-allocated unconstrained vector. Element type is
    `promote_type(eltype(θ), Float64)`.

# Throws
- `DimensionMismatch`: if `length(θ) != length(priors)`.

# Notes
- Replaces the Python reference's per-call allocating comprehension with
    a single `similar`-based allocation + an `@inbounds` write loop. Same
    big-O cost, no comprehension noise. For zero-allocation hot-path use,
    pass a `Tuple` of priors (see method below).
"""
function constrained_to_unconstrained(θ::AbstractVector{<:Real},
                                      priors::Vector{<:PriorType})
    length(θ) == length(priors) ||
        throw(DimensionMismatch("θ has $(length(θ)) entries, priors has $(length(priors))"))
    out = similar(θ, promote_type(eltype(θ), Float64))
    @inbounds for i in eachindex(θ)
        out[i] = to_unconstrained(priors[i], θ[i])
    end
    return out
end

"""
    unconstrained_to_constrained(u, priors) -> Vector

Inverse of `constrained_to_unconstrained`.

# Arguments
- `u::AbstractVector{<:Real}`: unconstrained vector, length `n`.
- `priors::Vector{<:PriorType}`: prior list, length `n`.

# Returns
- `Vector`: a freshly-allocated constrained vector.

# Throws
- `DimensionMismatch`: if `length(u) != length(priors)`.
"""
function unconstrained_to_constrained(u::AbstractVector{<:Real},
                                      priors::Vector{<:PriorType})
    length(u) == length(priors) ||
        throw(DimensionMismatch("u has $(length(u)) entries, priors has $(length(priors))"))
    out = similar(u, promote_type(eltype(u), Float64))
    @inbounds for i in eachindex(u)
        out[i] = to_constrained(priors[i], u[i])
    end
    return out
end

"""
    log_prior_unconstrained(u, priors) -> Real

Sum of component log priors in unconstrained space.

# Arguments
- `u::AbstractVector{<:Real}`: unconstrained vector.
- `priors::Vector{<:PriorType}`: prior list, same length.

# Returns
- Scalar log-prior — eltype of `u` if known, else `Float64`.

# Throws
- `DimensionMismatch`: if lengths differ.
"""
function log_prior_unconstrained(u::AbstractVector{<:Real},
                                 priors::Vector{<:PriorType})
    length(u) == length(priors) ||
        throw(DimensionMismatch("u has $(length(u)) entries, priors has $(length(priors))"))
    s = zero(eltype(u))
    @inbounds for i in eachindex(u)
        s += log_prior_unconstrained(priors[i], u[i])
    end
    return s
end


# ── Vector-level operations (hot path, tuple-of-priors, type-stable) ─────────

"""
    constrained_to_unconstrained(θ::AbstractVector, priors::Tuple) -> SVector

Hot-path variant: when `priors` is a `Tuple{Vararg{PriorType}}`, every
position has a concrete static type, so all per-element dispatches are
resolved at compile time and the loop fuses cleanly.

Returns a stack-allocated `SVector` of the same length as the tuple.

# Arguments
- `θ::AbstractVector{<:Real}`: constrained vector, length `N` matching the
    tuple length.
- `priors::Tuple`: heterogeneous tuple of `PriorType` instances.

# Returns
- `SVector{N,Float64}`: stack-allocated unconstrained vector.

# Notes
- Use this from inside the SMC² rejuvenation / inner PF, where the same
    tuple of priors is hit many times per outer iteration. Building the
    tuple once and calling this method is faster than the `Vector{<:PriorType}`
    method by the cost of the per-element dynamic dispatch.
"""
@generated function constrained_to_unconstrained(θ::AbstractVector{<:Real},
                                                 priors::Tuple{Vararg{PriorType,N}}) where {N}
    exprs = [:(to_unconstrained(priors[$i], θ[$i])) for i in 1:N]
    return :(@inbounds SVector{$N,Float64}($(exprs...)))
end

"""
    unconstrained_to_constrained(u::AbstractVector, priors::Tuple) -> SVector

Hot-path inverse of the tuple-of-priors `constrained_to_unconstrained`.
See that function's docstring for behaviour.
"""
@generated function unconstrained_to_constrained(u::AbstractVector{<:Real},
                                                 priors::Tuple{Vararg{PriorType,N}}) where {N}
    exprs = [:(to_constrained(priors[$i], u[$i])) for i in 1:N]
    return :(@inbounds SVector{$N,Float64}($(exprs...)))
end

"""
    log_prior_unconstrained(u::AbstractVector, priors::Tuple) -> Real

Hot-path sum of component log priors when `priors` is a tuple. Each
per-element call is compile-time-resolved.
"""
@generated function log_prior_unconstrained(u::AbstractVector{<:Real},
                                            priors::Tuple{Vararg{PriorType,N}}) where {N}
    body = :(zero(eltype(u)))
    for i in 1:N
        body = :($body + log_prior_unconstrained(priors[$i], u[$i]))
    end
    return :(@inbounds $body)
end


# ── Prior-spec construction helpers ──────────────────────────────────────────

const _PRIOR_CTOR = Dict{Symbol,Function}(
    :lognormal => (a) -> LogNormalPrior(a[1], a[2]),
    :normal    => (a) -> NormalPrior(a[1],    a[2]),
    :vonmises  => (a) -> VonMisesPrior(a[1],  a[2]),
    :beta      => (a) -> BetaPrior(a[1],      a[2]),
)

"""
    build_priors(spec) -> Vector{PriorType}

Build a `Vector{PriorType}` from a list of `(kind, args)` pairs.

# Arguments
- `spec::Vector`: each element is `(kind::Symbol, args::Tuple)`. `kind`
    is one of `:lognormal`, `:normal`, `:vonmises`, `:beta`.

# Returns
- `Vector{PriorType}`: prior instances in the order given.

# Throws
- `ErrorException`: on unknown `kind`.

# Example
```julia
priors = build_priors([
    (:lognormal, (0.0, 1.0)),
    (:normal,    (0.0, 0.5)),
    (:beta,      (2.0, 5.0)),
])
```
"""
function build_priors(spec::Vector)
    out = PriorType[]
    for (kind, args) in spec
        ctor = get(_PRIOR_CTOR, kind, nothing)
        ctor === nothing && error("unknown prior kind: $kind")
        push!(out, ctor(args))
    end
    return out
end


# ── Splitting combined θ into (params, init_states) ──────────────────────────

"""
    split_theta(θ, n_params) -> (params_view, init_view)

Split a combined parameter+init-state vector into its two pieces.

# Arguments
- `θ::AbstractVector`: combined vector of length `n_params + n_init`.
- `n_params::Int`: how many leading components are model parameters.

# Returns
- `Tuple{SubArray,SubArray}`: views (zero-allocation) — first is the
    parameter block, second is the init-state block.

# Notes
- Returned views share storage with `θ`; do not mutate them in caller code
    that expects `θ` to be unchanged. Per the functional API contract,
    callers in this library do not mutate.
"""
function split_theta(θ::AbstractVector, n_params::Int)
    return (@view θ[1:n_params]), (@view θ[n_params+1:end])
end
