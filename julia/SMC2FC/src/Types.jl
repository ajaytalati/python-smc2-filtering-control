# Types.jl — parametric types that encode dimensional distinctions
# the Python `jnp.ndarray` erases.
#
# Charter: LaTex_docs/julia_port_charter.pdf §15.2 + §14.4 (architectural type
# boundaries). The marker structs in this file power the multiple-dispatch
# substitutes for Python's flag-driven branching (charter §15.1).

using StaticArrays: SVector
using ComponentArrays: ComponentVector
using StructArrays: StructArray
using CUDA: CuArray

# ── Backends ─────────────────────────────────────────────────────────────────
# `AbstractBackend` powers the GPU/CPU dispatch. Functions that need to allocate
# arrays accept a backend marker and return either `Array` or `CuArray`.

abstract type AbstractBackend end
struct CPUBackend  <: AbstractBackend end
struct CUDABackend <: AbstractBackend end

# ── State and parameter aliases ──────────────────────────────────────────────
# `State{N,T}` is a stack-allocated, fixed-size, zero-overhead state vector.
# Per charter §15.2 (StaticArrays.jl): without this, the inner SDE loop would
# regress badly versus JAX which unrolls small arrays into registers.

const State{N,T} = SVector{N,T}

# `DynParams{T}` is a flat AbstractVector with dot-access (`p.k_FB`) provided
# by ComponentArrays.jl. AdvancedHMC.jl + Enzyme.jl accept this directly.
# The component layout is model-specific and supplied at construction.

const DynParams{T,A<:AbstractVector{T}} = ComponentVector{T,A}

# ── Particle cloud ───────────────────────────────────────────────────────────
# `Particle{N,T}` is a single particle in the inner filter. `ParticleCloud`
# stores them as a `StructArray` so the underlying memory is contiguous per
# field — coalesced GPU access, while `for p in particles` reads naturally.
# Charter §13 (StructArrays.jl).

struct Particle{N,T}
    state::SVector{N,T}
    log_weight::T
end

const ParticleCloud{N,T} = StructArray{Particle{N,T}}

# ── Hybrid-execution type boundaries (charter §14.4) ─────────────────────────
# These two parametric types make the GPU/CPU contract explicit. The inner
# particle filter and SDE step run with `GPUFilterState`; the outer SMC² /
# HMC rejuvenation runs with `CPUParameterCloud`. The boundary is `compute_
# likelihood`: it takes a CPU `theta` vector, transfers it to GPU, dispatches
# `run_particle_filter!`, and returns a scalar log-likelihood back to CPU.

struct GPUFilterState{T}
    particles::CuArray{T,2}    # (n_pf, d_state) — state particles, GPU-resident
    weights::CuArray{T,1}      # (n_pf,)         — log-weights, GPU-resident
end

struct CPUParameterCloud{T}
    theta::Matrix{T}           # (n_smc, d_theta) — outer parameter cloud
    log_lik::Vector{T}         # (n_smc,)         — per-particle log-likelihood
end

# ── Marker types for multiple dispatch (charter §15.1) ───────────────────────
# These replace Python's flag-driven conditionals. `bridge_init(::GaussianBridge,
# ...)` and `bridge_init(::SchrodingerFollmerBridge, ...)` are two methods of
# the same function; the dispatcher picks one at compile time and the agent
# cannot accidentally reach the wrong branch.

abstract type BridgeKind end
struct GaussianBridge          <: BridgeKind end
struct SchrodingerFollmerBridge <: BridgeKind end

abstract type ChanceConstraintMode end
struct SoftSurrogate <: ChanceConstraintMode end
struct HardIndicator <: ChanceConstraintMode end

# Bridge-construction sentinels for SF q1 sampling (charter has IS vs annealed).
abstract type SFQ1Mode end
struct SFQ1ImportanceSampling <: SFQ1Mode end
struct SFQ1AnnealedSMC        <: SFQ1Mode end
