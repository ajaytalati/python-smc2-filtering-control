# FSAv5/PhiBurst.jl — sub-daily Φ-burst expansion.
#
# Maps line-by-line to:
#   • Python: `models/fsa_high_res/_phi_burst.py:46-92`
# (No LEAN counterpart — this is a deterministic envelope helper, not
# part of the SDE math; the LEAN reference covers the dynamics layer.)
#
# Per-day Φ envelope: morning-loaded Gamma(k=2) shape, peaks ~3h
# post-wake, normalised so the daily integrated Φ is preserved
# regardless of wake/sleep window.
#
# Used by:
#   - `Plant.jl:advance!` (closed-loop control hook)
#   - the bench drivers when expanding daily-rate decisions to per-bin
#     stimulus arrays for the simulator.

# ── Time-grid constants (15-min bins, FSA_STEP_MINUTES=15 default) ─────────

const STEP_MIN_DEFAULT = 15

const BINS_PER_DAY = (60 * 24) ÷ STEP_MIN_DEFAULT          # 96
const DT_BIN_DAYS  = 1.0 / BINS_PER_DAY                     # 1/96
const DT_BIN_HOURS = 24.0 / BINS_PER_DAY                    # 0.25 h

# ── Per-day Φ envelope ─────────────────────────────────────────────────────

"""
    build_per_day_envelope(; wake_hour=7.0, sleep_hour=23.0, tau_hours=3.0)::Vector{Float64}

Construct the per-day Φ envelope of length `BINS_PER_DAY` (96 by default).

Mirrors `_phi_burst._build_per_day_envelope`. Returns array `e[k]`
such that for any daily-Φ value `Φ_d`, the per-bin Φ(t_k) = `Φ_d ·
e[k]`. The envelope is normalised so the daily integral
`sum(e[k] * dt_hours) = 24`, preserving the slow-Banister daily load.
"""
function build_per_day_envelope(; wake_hour::Float64  = 7.0,
                                  sleep_hour::Float64 = 23.0,
                                  tau_hours::Float64  = 3.0)::Vector{Float64}
    h = collect((0:BINS_PER_DAY-1) .* DT_BIN_HOURS)
    out = zeros(Float64, BINS_PER_DAY)
    @inbounds for k in 1:BINS_PER_DAY
        h_k = h[k]
        if h_k >= wake_hour && h_k < sleep_hour
            t_post = h_k - wake_hour
            out[k] = t_post * exp(-t_post / tau_hours)
        end
    end
    daily_integral = sum(out) * DT_BIN_HOURS
    if daily_integral > 1e-12
        out .*= (24.0 / daily_integral)
    end
    return out
end

# ── Daily → sub-daily expansion ────────────────────────────────────────────

"""
    expand_daily_phi_to_subdaily(daily_phi::AbstractVector{Float64};
                                  wake_hour=7.0, sleep_hour=23.0,
                                  tau_hours=3.0)::Vector{Float64}

Expand a per-day Φ schedule of length `n_days` into a per-bin Φ(t)
schedule of length `n_days * BINS_PER_DAY`. Each daily Φ value is
multiplied by the per-day envelope.

Mirrors `_phi_burst.expand_daily_phi_to_subdaily`.
"""
function expand_daily_phi_to_subdaily(daily_phi::AbstractVector{Float64};
                                       wake_hour::Float64  = 7.0,
                                       sleep_hour::Float64 = 23.0,
                                       tau_hours::Float64  = 3.0)::Vector{Float64}
    envelope = build_per_day_envelope(; wake_hour, sleep_hour, tau_hours)
    n_days   = length(daily_phi)
    out      = Vector{Float64}(undef, n_days * BINS_PER_DAY)
    @inbounds for d in 1:n_days
        phi_d = daily_phi[d]
        base  = (d - 1) * BINS_PER_DAY
        for k in 1:BINS_PER_DAY
            out[base + k] = phi_d * envelope[k]
        end
    end
    return out
end

export BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS
export build_per_day_envelope, expand_daily_phi_to_subdaily
