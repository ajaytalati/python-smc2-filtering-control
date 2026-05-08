"""
    Simulator/Observations.jl

Per-channel observation sampling with topological-sort dependency
resolution. Port of `smc2fc/simulator/sde_observations.py`.
"""
module Observations

using Random: AbstractRNG, MersenneTwister

export ObsChannel, generate_all_channels

"""
    ObsChannel(name; depends_on=Symbol[], generate_fn)

A named observation channel with dependency metadata.

# Fields
- `name::Symbol`: channel identifier.
- `depends_on::Vector{Symbol}`: upstream channels that must be generated
    first; topologically sorted by `generate_all_channels`.
- `generate_fn::F`: callable
    `(trajectory, t_grid, params, aux, prior_channels, rng) -> NamedTuple`
    returning the channel's outputs (e.g. `(value=..., mask=...)`).
"""
Base.@kwdef struct ObsChannel{F}
    name::Symbol
    depends_on::Vector{Symbol} = Symbol[]
    generate_fn::F
end

"""
    generate_all_channels(channels, trajectory, t_grid, params, aux=nothing;
                          seed=0) -> Dict{Symbol,Any}

Generate every channel's outputs in dependency order.

# Arguments
- `channels::Vector{<:ObsChannel}`: channels to generate.
- `trajectory::AbstractArray`: simulated state trajectory.
- `t_grid::AbstractVector`: time grid for `trajectory`.
- `params`: parameter container.
- `aux`: auxiliary inputs forwarded to each `generate_fn`.

# Keyword arguments
- `seed::Integer = 0`: PRNG seed; per-channel sub-seeds are derived from it.

# Returns
- `Dict{Symbol,Any}`: `{channel_name => channel_output}`. Each
    `generate_fn` receives the dict of already-generated channels as
    `prior_channels`, so chained channels can read upstream values.

# Throws
- `ErrorException`: on circular or unresolvable dependencies.

# Notes
- Topological sort is implemented as a fixed-point loop; the maximum
    number of passes is bounded by the channel count.
"""
function generate_all_channels(channels::Vector,
                                 trajectory::AbstractArray,
                                 t_grid::AbstractVector,
                                 params,
                                 aux = nothing;
                                 seed::Integer = 0)
    rng_seeds = MersenneTwister(seed)
    channel_seeds = Dict{Symbol,Int}(
        ch.name => abs(rand(rng_seeds, Int)) % (2^31 - 1) for ch in channels
    )

    generated = Dict{Symbol,Any}()
    remaining = collect(channels)
    max_iter  = length(remaining) + 1

    for _ in 1:max_iter
        isempty(remaining) && break
        progress = false
        for ch in copy(remaining)
            if all(d -> haskey(generated, d), ch.depends_on)
                ch_rng = MersenneTwister(channel_seeds[ch.name])
                generated[ch.name] = ch.generate_fn(
                    trajectory, t_grid, params, aux, generated, ch_rng,
                )
                filter!(c -> c !== ch, remaining)
                progress = true
            end
        end
        if !progress
            error("Circular or unresolvable channel dependencies: " *
                  string([c.name for c in remaining]) *
                  ". Already generated: $(collect(keys(generated)))")
        end
    end

    return generated
end

end # module Observations
