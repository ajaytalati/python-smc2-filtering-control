# SMC2/Tempering.jl — adaptive δλ via bisection.
#
# Port of `smc2fc/core/jax_native_smc.py:_solve_delta_for_ess`. The Python
# version is forced to use `jax.lax.scan` over a fixed bisection count so the
# whole `while_loop` body is JIT-traceable. Julia uses a plain `for` loop
# (charter §15.1).

module Tempering

using LogExpFunctions: logsumexp

export ess_at_delta, solve_delta_for_ess

"""
    ess_at_delta(loglikelihood_per_particle, delta) -> scalar

ESS produced by reweighting with `delta * loglikelihood`. The argument
`loglikelihood_per_particle` is the *vector* of likelihoods evaluated once
outside the bisection loop — the bisection only varies `delta`.
"""
function ess_at_delta(loglikelihood_per_particle::AbstractVector{T},
                       delta::Real) where {T<:Real}
    log_w      = T(delta) .* loglikelihood_per_particle
    log_w_norm = log_w .- logsumexp(log_w)
    log_ess    = -logsumexp(2.0 .* log_w_norm)
    return exp(log_ess)
end

"""
    solve_delta_for_ess(loglikelihood_per_particle, target_ess_frac, max_delta;
                         n_bisect_steps=30) -> delta

Bisection on `delta ∈ [0, max_delta]` so that `ESS(delta) ≈ target_ess_frac · N`.
Returns `max_delta` if `ESS(max_delta) ≥ target` (we can take the full step).
"""
function solve_delta_for_ess(loglikelihood_per_particle::AbstractVector{T},
                              target_ess_frac::Real,
                              max_delta::Real;
                              n_bisect_steps::Integer = 30) where {T<:Real}
    n          = length(loglikelihood_per_particle)
    target_ess = target_ess_frac * n

    # Quick path: full step still healthy → no bisection needed.
    ess_max = ess_at_delta(loglikelihood_per_particle, max_delta)
    if ess_max ≥ target_ess
        return T(max_delta)
    end

    lo, hi = T(0.0), T(max_delta)
    for _ in 1:n_bisect_steps
        mid = T(0.5) * (lo + hi)
        ess_mid = ess_at_delta(loglikelihood_per_particle, mid)
        if ess_mid > target_ess
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

end # module Tempering
