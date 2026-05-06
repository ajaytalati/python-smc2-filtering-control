# Transforms.jl — constrained ↔ unconstrained bijections via multiple
# dispatch over a `PriorType` hierarchy.
#
# Replaces `smc2fc/transforms/unconstrained.py`. The Python implementation
# carries seven indicator arrays (`is_log`, `is_logit`, `is_ident`, `is_ln`,
# `is_norm`, `is_vm`, `is_bt`) because JAX cannot dispatch on type. Julia
# can — each prior gets its own method, the dispatcher picks at compile time,
# the indicator arrays disappear.
#
# Charter: LaTex_docs/julia_port_charter.pdf §15.1 (multiple dispatch).

using LogExpFunctions: log1pexp     # numerically-stable log(1 + exp(u))

# ── Prior-type hierarchy ─────────────────────────────────────────────────────

abstract type PriorType end

struct LogNormalPrior <: PriorType
    μ::Float64
    σ::Float64
end

struct NormalPrior <: PriorType
    μ::Float64
    σ::Float64
end

struct VonMisesPrior <: PriorType
    μ::Float64
    κ::Float64
end

struct BetaPrior <: PriorType
    α::Float64
    β::Float64
end

# ── Component-wise bijections ────────────────────────────────────────────────
# Mirrors the per-component formulas in
#   smc2fc/transforms/unconstrained.py:76-96
# exactly, including the clamp ranges (1e-30, 1e-6, ±20).

# constrained → unconstrained
to_unconstrained(::LogNormalPrior, θ::Real) = log(max(θ, 1e-30))
to_unconstrained(::NormalPrior,    θ::Real) = θ
to_unconstrained(::VonMisesPrior,  θ::Real) = θ              # identity in domain
function to_unconstrained(::BetaPrior, θ::Real)
    c = clamp(θ, 1e-6, 1.0 - 1e-6)
    return log(c / (1.0 - c))
end

# unconstrained → constrained
to_constrained(::LogNormalPrior, u::Real) = exp(clamp(u, -20.0, 20.0))
to_constrained(::NormalPrior,    u::Real) = u
to_constrained(::VonMisesPrior,  u::Real) = u                # identity in domain
to_constrained(::BetaPrior,      u::Real) = 1.0 / (1.0 + exp(-u))   # sigmoid

# ── Component-wise log prior in unconstrained space ──────────────────────────
# Mirrors smc2fc/transforms/unconstrained.py:99-116 verbatim. The change-of-
# variables Jacobian is folded in: lognormal already lives in u = log θ space
# so the prior is Gaussian on u; beta uses log σ(u) + log σ(-u) which is the
# correct Jacobian-adjusted Beta density on the logit scale.

log_prior_unconstrained(p::LogNormalPrior, u::Real) =
    -0.5 * ((u - p.μ) / p.σ)^2 - log(p.σ)

log_prior_unconstrained(p::NormalPrior, u::Real) =
    -0.5 * ((u - p.μ) / p.σ)^2 - log(p.σ)

log_prior_unconstrained(p::VonMisesPrior, u::Real) =
    p.κ * cos(u - p.μ)

# log σ(u) = -log(1 + exp(-u)) = -log1pexp(-u). Numerically stable.
log_prior_unconstrained(p::BetaPrior, u::Real) =
    -p.α * log1pexp(-u) - p.β * log1pexp(u)


# ── Vector-level operations over a prior list ────────────────────────────────
# In Python these were flat element-wise multiplies against the indicator
# arrays. In Julia, `priors::Vector{<:PriorType}` is heterogeneous in concrete
# subtype, so we map element-by-element. The dispatcher inlines each method.

"""
    constrained_to_unconstrained(θ::AbstractVector, priors::Vector{<:PriorType})

Map a constrained parameter vector `θ` to its unconstrained representation `u`,
component-by-component, using the prior-type hierarchy for dispatch.
"""
function constrained_to_unconstrained(θ::AbstractVector{<:Real},
                                      priors::Vector{<:PriorType})
    length(θ) == length(priors) ||
        throw(DimensionMismatch("θ has $(length(θ)) entries, priors has $(length(priors))"))
    return [to_unconstrained(priors[i], θ[i]) for i in eachindex(θ)]
end

"""
    unconstrained_to_constrained(u::AbstractVector, priors::Vector{<:PriorType})

Inverse of `constrained_to_unconstrained`.
"""
function unconstrained_to_constrained(u::AbstractVector{<:Real},
                                      priors::Vector{<:PriorType})
    length(u) == length(priors) ||
        throw(DimensionMismatch("u has $(length(u)) entries, priors has $(length(priors))"))
    return [to_constrained(priors[i], u[i]) for i in eachindex(u)]
end

"""
    log_prior_unconstrained(u::AbstractVector, priors::Vector{<:PriorType})

Sum of component log priors in unconstrained space. Returns a scalar.
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


# ── Prior-spec construction helpers ──────────────────────────────────────────
# Build a `Vector{PriorType}` from a `Vector{Tuple{Symbol,Tuple}}` mirroring
# the Python OrderedDict format `(name, (type, args))`.

const _PRIOR_CTOR = Dict{Symbol,Function}(
    :lognormal => (a) -> LogNormalPrior(a[1], a[2]),
    :normal    => (a) -> NormalPrior(a[1],    a[2]),
    :vonmises  => (a) -> VonMisesPrior(a[1],  a[2]),
    :beta      => (a) -> BetaPrior(a[1],      a[2]),
)

"""
    build_priors(spec::Vector{<:Tuple{Symbol,<:Tuple}})

Build a `Vector{PriorType}` from a list of `(prior_kind, args)` pairs.
`prior_kind` is one of `:lognormal`, `:normal`, `:vonmises`, `:beta`.
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
# Mirrors smc2fc/transforms/unconstrained.py:119-129.

"""
    split_theta(θ::AbstractVector, n_params::Int)

Split a combined parameter+init-state vector into its two pieces.
Returns a tuple `(params, init_states)` of views (zero-allocation).
"""
function split_theta(θ::AbstractVector, n_params::Int)
    return (@view θ[1:n_params]), (@view θ[n_params+1:end])
end
