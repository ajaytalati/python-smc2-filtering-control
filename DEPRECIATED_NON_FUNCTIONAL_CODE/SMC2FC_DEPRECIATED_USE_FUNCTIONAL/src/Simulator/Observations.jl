# Simulator/Observations.jl — per-channel observation sampling.
#
# Port of `smc2fc/simulator/sde_observations.py`. The Python implementation
# walks a dependency graph between named channels (e.g. HR depends on
# T-state, sleep depends on circadian phase) using a topological sort.
#
# Julia: same algorithm, plain dict + while loop. ~50 LOC.

module Observations

using Random: AbstractRNG, MersenneTwister

export ObsChannel, generate_all_channels

"""
    ObsChannel(name; depends_on=Symbol[], generate_fn)

A named observation channel. `generate_fn` is a callable
    `(trajectory, t_grid, params, aux, prior_channels, rng) -> NamedTuple`
that returns the channel's outputs (e.g. `(value=..., mask=...)`).

`depends_on` lists channels that must be generated first; topological-
sorted by `generate_all_channels`.
"""
Base.@kwdef struct ObsChannel{F}
    name::Symbol
    depends_on::Vector{Symbol} = Symbol[]
    generate_fn::F
end

"""
    generate_all_channels(channels::Vector{<:ObsChannel}, trajectory, t_grid,
                            params, aux=nothing; seed=0)
        -> Dict{Symbol, NamedTuple}

Generate every channel's outputs in dependency order. Returns a dict
`{channel_name → channel_output}`. Each channel's `generate_fn` is given
the dict of *already-generated* channels via the `prior_channels` argument,
so chained channels can read upstream values.
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
