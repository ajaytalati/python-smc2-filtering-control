# Sub-daily Φ-burst expansion — port of `version_2/models/fsa_high_res/_phi_burst.py`.
#
# Converts a per-day Φ schedule (`(n_days,)`) to a per-bin Φ(t) schedule
# (`(n_days·BINS_PER_DAY,)`) with a morning-loaded Gamma-shape activity
# profile per wake-window:
#
#     t = hour_of_day - wake_hour       (zero at wake)
#     shape(t) = t · exp(-t / τ)         (Gamma(k=2) shape)
#                                         peaks at t = τ (~3h post-wake)
#
# Normalised so each day's integrated Φ equals 24 · Φ_daily.
# Sleep hours [sleep_hour, wake_hour+24]: Φ = 0.
#
# Reads `FSA_STEP_MINUTES` env var at module-load time to set BINS_PER_DAY
# (matches the Python convention).

module PhiBurst

# Time-grid constants — read FSA_STEP_MINUTES at load time.
const _STEP_MIN = parse(Int, get(ENV, "FSA_STEP_MINUTES", "15"))
@assert (60 * 24) % _STEP_MIN == 0 "FSA_STEP_MINUTES=$(_STEP_MIN) must divide 1440"

const BINS_PER_DAY = (60 * 24) ÷ _STEP_MIN
const DT_BIN_DAYS  = 1.0 / BINS_PER_DAY
const DT_BIN_HOURS = 24.0 / BINS_PER_DAY

"""
    build_per_day_envelope(; wake_hour=7.0, sleep_hour=23.0, tau_hours=3.0)
        -> Vector{Float64}(BINS_PER_DAY)

Build the per-day Φ envelope. For any daily-Φ value `Φ_d`, the per-bin
Φ at bin `k` of the day is `Φ_d · e[k]`. Normalised so
`sum(e[k] * DT_BIN_HOURS) = 24` regardless of wake/sleep window.
"""
function build_per_day_envelope(; wake_hour::Real = 7.0,
                                  sleep_hour::Real = 23.0,
                                  tau_hours::Real = 3.0)
    h = collect(0:BINS_PER_DAY-1) .* DT_BIN_HOURS
    in_wake = (h .>= wake_hour) .& (h .< sleep_hour)
    t_post = ifelse.(in_wake, h .- wake_hour, 0.0)
    raw_shape = ifelse.(in_wake, t_post .* exp.(-t_post ./ tau_hours), 0.0)
    daily_integral = sum(raw_shape .* DT_BIN_HOURS)
    return raw_shape .* (24.0 / max(daily_integral, 1e-12))
end


"""
    expand_daily_phi_to_subdaily(daily_phi::AbstractVector;
                                 wake_hour=7.0, sleep_hour=23.0, tau_hours=3.0)
        -> Vector{Float32}(n_days · BINS_PER_DAY)

Expand a per-day Φ schedule to per-bin Φ via the morning-loaded Gamma
envelope. Result dtype Float32 (mirrors Python).
"""
function expand_daily_phi_to_subdaily(daily_phi::AbstractVector;
                                       wake_hour::Real = 7.0,
                                       sleep_hour::Real = 23.0,
                                       tau_hours::Real = 3.0)
    envelope = build_per_day_envelope(; wake_hour, sleep_hour, tau_hours)
    n_days = length(daily_phi)
    out = Vector{Float32}(undef, n_days * BINS_PER_DAY)
    @inbounds for d in 1:n_days, k in 1:BINS_PER_DAY
        out[(d - 1) * BINS_PER_DAY + k] = Float32(daily_phi[d] * envelope[k])
    end
    return out
end


"""
    sleep_mask_from_hours(n_days; sleep_hour_lo=23.0, sleep_hour_hi=7.0)
        -> Vector{Float32}

Deterministic a-priori sleep mask (1 if 'nominally asleep' at bin).
"""
function sleep_mask_from_hours(n_days::Integer;
                                sleep_hour_lo::Real = 23.0,
                                sleep_hour_hi::Real = 7.0)
    mask = zeros(Float32, n_days * BINS_PER_DAY)
    @inbounds for d in 1:n_days, k in 1:BINS_PER_DAY
        h = (k - 1) * DT_BIN_HOURS
        in_sleep = (h >= sleep_hour_lo) || (h < sleep_hour_hi)
        mask[(d - 1) * BINS_PER_DAY + k] = in_sleep ? 1.0f0 : 0.0f0
    end
    return mask
end


export BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS
export build_per_day_envelope, expand_daily_phi_to_subdaily, sleep_mask_from_hours

end # module PhiBurst
