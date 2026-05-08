"""
    _enzyme_inactive_rules.jl

A small "patch" that registers `Random.fill_array!` and (where it
exists) `Random.dsfmt_fill_array_close1_open2!` as **inactive** under
Enzyme.

# Why this is needed

The bootstrap PF inside `SMC2FC_functional` samples per-particle noise
on demand via `randn(rng, Float64)` against a closure-captured
`MersenneTwister`. When Enzyme reverse-mode AD walks into the PF, it
follows the call chain
`randn(rng,...) → rand(rng,...) → fill_array!(::MersenneTwister, ...)
→ ccall(:dsfmt_fill_array_close1_open2, ...)`
and aborts with

    EnzymeNoDerivativeError:
      No augmented forward pass found for
      ejlstr\$dsfmt_fill_array_close1_open2\$libdSFMT

because Enzyme has no rule for the libdSFMT C call.

# Why declaring it inactive is correct

Random samples drawn inside the PF are by construction **independent of
the AD variable** `u`: the gradient ∂(noise)/∂u is identically zero.
The samples flow into the particles via `x = base + σ(u) · √dt · noise`,
so `∂x/∂u` comes from `∂σ/∂u`, not from the noise. Telling Enzyme that
`fill_array!` (and the underlying C call) carries no active derivative
information is mathematically correct — it lets the rule-respecting
inactive marker take the place of the missing AD rule.

# Loading

Either include this file directly from a test, or `include` it from any
script that needs Enzyme support on a `bootstrap_log_likelihood` target.
The rules are global once registered.
"""

using Enzyme
import Random

@info "_enzyme_inactive_rules.jl: registering Random.fill_array! family as inactive under Enzyme"

# Declaration that fill_array! has no active derivative contribution.
Enzyme.EnzymeRules.inactive(::typeof(Random.fill_array!), args...) = nothing

# Belt-and-braces: the C-call wrapper itself, where it's exposed.
@static if isdefined(Random, :dsfmt_fill_array_close1_open2!)
    Enzyme.EnzymeRules.inactive(::typeof(Random.dsfmt_fill_array_close1_open2!),
                                 args...) = nothing
end

# Random.rand! and Random.randn! also fan out into fill_array!. Marking
# them inactive at the higher level catches any path Enzyme might take
# that bypasses fill_array!.
Enzyme.EnzymeRules.inactive(::typeof(Random.rand!),  args...) = nothing
Enzyme.EnzymeRules.inactive(::typeof(Random.randn!), args...) = nothing
