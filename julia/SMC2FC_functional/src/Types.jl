"""
    Types.jl

Parametric types that carry dimensional and architectural distinctions
through the type system.

The Python reference (`smc2fc/`) collapses everything to `jnp.ndarray` and
recovers structure at runtime via Pytree shape introspection. The Julia
version uses parametric `struct`s and marker types so that distinctions
which JAX has to recover at runtime are made compile-time facts here.

Mirrors the structural intent of `julia/SMC2FC/src/Types.jl` (the existing
port) but the API surface is identical — these are immutable types, so the
"functional API" rule has nothing to enforce on this file.
"""

using StaticArrays: SVector
using ComponentArrays: ComponentVector
using StructArrays: StructArray
using CUDA: CuArray

# ── Backends ────────────────────────────────────────────────────────────────

"""
Marker hierarchy for execution backends.

`AbstractBackend` is dispatched on by allocation helpers so the same generic
code can return either a `CPUBackend`-allocated `Array` or a
`CUDABackend`-allocated `CuArray`.
"""
abstract type AbstractBackend end

"""CPU backend marker — allocations return plain `Array`s."""
struct CPUBackend  <: AbstractBackend end

"""CUDA-GPU backend marker — allocations return `CuArray`s via CUDA.jl."""
struct CUDABackend <: AbstractBackend end

# ── State and parameter aliases ─────────────────────────────────────────────

"""
    State{N,T}

Stack-allocated, fixed-size state vector of `N` components of element type `T`.

# Notes
- This is a type alias for `SVector{N,T}` from StaticArrays.jl.
- The inner SDE / particle-filter loop relies on this being stack-allocated;
    using a regular `Vector` here regresses badly versus the JAX reference
    because JAX unrolls small arrays into registers.
"""
const State{N,T} = SVector{N,T}

"""
    DynParams{T,A}

Flat `AbstractVector{T}` with field-name-style indexing (e.g. `p.k_FB`)
provided by `ComponentArrays.ComponentVector`. The component layout is
model-specific and supplied at construction time; AdvancedHMC.jl and
Enzyme.jl both consume this type directly.
"""
const DynParams{T,A<:AbstractVector{T}} = ComponentVector{T,A}

# ── Particle cloud ──────────────────────────────────────────────────────────

"""
    Particle{N,T}

A single particle in the inner filter: an `N`-dimensional state plus a
scalar log-weight, both with element type `T`.

# Fields
- `state::SVector{N,T}`: state vector (stack-allocated).
- `log_weight::T`: log-importance-weight.
"""
struct Particle{N,T}
    state::SVector{N,T}
    log_weight::T
end

"""
    ParticleCloud{N,T}

Cloud of `Particle{N,T}` stored as a `StructArray`. The underlying memory
is contiguous *per field* (Struct-of-Arrays layout), so GPU access is
coalesced and `for p in particles` still reads naturally.
"""
const ParticleCloud{N,T} = StructArray{Particle{N,T}}

# ── Hybrid CPU / GPU type boundaries ────────────────────────────────────────

"""
    GPUFilterState{T}

GPU-resident state of the inner particle filter.

# Fields
- `particles::CuArray{T,2}`: `(n_pf, d_state)` matrix of state particles.
- `weights::CuArray{T,1}`: `(n_pf,)` vector of log-weights.
"""
struct GPUFilterState{T}
    particles::CuArray{T,2}
    weights::CuArray{T,1}
end

"""
    CPUParameterCloud{T}

CPU-resident outer parameter cloud used by SMC² and HMC rejuvenation.

# Fields
- `theta::Matrix{T}`: `(n_smc, d_theta)` parameter samples.
- `log_lik::Vector{T}`: `(n_smc,)` per-particle log-likelihood.
"""
struct CPUParameterCloud{T}
    theta::Matrix{T}
    log_lik::Vector{T}
end

# ── Marker types for multiple dispatch ──────────────────────────────────────

"""Abstract base for warm-start bridge kinds."""
abstract type BridgeKind end

"""Single-Gaussian + Liu–West shrinkage warm-start bridge."""
struct GaussianBridge <: BridgeKind end

"""Bures–Wasserstein-geodesic Schrödinger–Föllmer warm-start bridge."""
struct SchrodingerFollmerBridge <: BridgeKind end

"""Abstract base for chance-constraint surrogate kinds."""
abstract type ChanceConstraintMode end

"""Soft (smooth) chance-constraint surrogate."""
struct SoftSurrogate <: ChanceConstraintMode end

"""Hard (indicator) chance-constraint surrogate."""
struct HardIndicator <: ChanceConstraintMode end

"""Abstract base for SF q1-sampling modes."""
abstract type SFQ1Mode end

"""Single-step importance-sampling q1 estimator."""
struct SFQ1ImportanceSampling <: SFQ1Mode end

"""K-stage tempered-SMC + RW-MH q1 estimator."""
struct SFQ1AnnealedSMC <: SFQ1Mode end
