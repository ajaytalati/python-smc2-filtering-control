using StaticArrays

const A_TYP = 0.10
const F_TYP = 0.20
const EPS_A_FROZEN = 1.0e-4
const EPS_B_FROZEN = 1.0e-4

# Bins-per-day grid: keep configurable to mirror FSA_STEP_MINUTES.
# Default 96 (15-min bins) matches the Python default.
const FSA_BINS_PER_DAY_DEFAULT = 96

"""
    FsaParams{T}

Plain isbits parameter struct for the FSA-v2 SDE drift + diffusion.
Mirrors `simulation.DEFAULT_PARAMS` (G1-reparametrized form). Field
order matters because it's passed by value into GPU kernels.

Use `FsaParams(d::Dict)` to build from a Python-style param dict.
"""
struct FsaParams{T<:AbstractFloat}
    tau_B::T
    tau_F::T
    kappa_B::T
    kappa_F::T
    epsilon_A::T
    lambda_A::T
    mu_0::T
    mu_B::T
    mu_F::T
    mu_FF::T
    eta::T
    sigma_B::T
    sigma_F::T
    sigma_A::T
end

function FsaParams{T}(d::AbstractDict) where {T<:AbstractFloat}
    FsaParams{T}(
        T(d["tau_B"]),     T(d["tau_F"]),
        T(d["kappa_B"]),   T(d["kappa_F"]),
        T(d["epsilon_A"]), T(d["lambda_A"]),
        T(d["mu_0"]),      T(d["mu_B"]),
        T(d["mu_F"]),      T(d["mu_FF"]),
        T(d["eta"]),
        T(d["sigma_B"]),   T(d["sigma_F"]),  T(d["sigma_A"]),
    )
end
FsaParams(d::AbstractDict) = FsaParams{Float64}(d)

"""
    FSAv2_DEFAULT_PARAMS(T=Float64)

Returns the G1-reparametrized truth parameters matching
`version_2/models/fsa_high_res/simulation.py:DEFAULT_PARAMS`.
"""
function FSAv2_DEFAULT_PARAMS(::Type{T}=Float64) where {T<:AbstractFloat}
    FsaParams{T}(
        T(42.0),                              # tau_B
        T(7.0 / (1.0 + 1.00 * A_TYP)),        # tau_F = 6.3636…
        T(0.012 * (1.0 + 0.40 * A_TYP)),      # kappa_B = 0.01248
        T(0.030),                             # kappa_F
        T(0.40),                              # epsilon_A
        T(1.00),                              # lambda_A
        T(0.02 + 0.40 * F_TYP^2),             # mu_0 = 0.036
        T(0.30),                              # mu_B
        T(0.10 + 2.0 * F_TYP * 0.40),         # mu_F = 0.26
        T(0.40),                              # mu_FF
        T(0.20),                              # eta
        T(0.010),                             # sigma_B
        T(0.012),                             # sigma_F
        T(0.020),                             # sigma_A
    )
end

const FSAv2_DEFAULT_INIT = (B=0.05, F=0.30, A=0.10)

"""
    drift(y, p, Phi_t)

FSA-v2 G1-reparametrized drift, returning d[B,F,A]/dt as an SVector.
Mirrors `simulation.py:drift_jax` and `_dynamics.py:drift_jax`.
"""
@inline function drift(y::SVector{3,T}, p::FsaParams{T}, Phi_t::T) where {T}
    B = y[1]; F = y[2]; A = y[3]
    F_dev = F - T(F_TYP)
    mu = p.mu_0 + p.mu_B * B - p.mu_F * F - p.mu_FF * F_dev * F_dev
    a_factor_B = (one(T) + p.epsilon_A * A) /
                 (one(T) + p.epsilon_A * T(A_TYP))
    a_factor_F = (one(T) + p.lambda_A * A) /
                 (one(T) + p.lambda_A * T(A_TYP))
    dB = p.kappa_B * a_factor_B * Phi_t - B / p.tau_B
    dF = p.kappa_F * Phi_t - a_factor_F / p.tau_F * F
    dA = mu * A - p.eta * A * A * A
    SVector{3,T}(dB, dF, dA)
end

"""
    noise_scale(y, p)

State-dependent sqrt-Itô diffusion scale `sigma .* g(y)`, matching
`simulation.py:noise_scale_fn_jax` (note the `+ EPS_A_FROZEN` on A).
Multiplies by `sigma_*` so the caller can do `diff = noise_scale * sqrt_dt * N(0,1)`.
"""
@inline function noise_scale(y::SVector{3,T}, p::FsaParams{T}) where {T}
    B = clamp(y[1], T(EPS_B_FROZEN), one(T) - T(EPS_B_FROZEN))
    F = max(y[2], zero(T))
    A = max(y[3], zero(T))
    SVector{3,T}(
        p.sigma_B * sqrt(B * (one(T) - B)),
        p.sigma_F * sqrt(F),
        p.sigma_A * sqrt(A + T(EPS_A_FROZEN)),
    )
end

"""
    bin_lookup(t_days, Phi_arr, dt_bin_days)

Index Phi_arr at `clip(floor(t/dt_bin), 0, len-1)` — exactly matching
`simulation.py:_bin_lookup` and `drift_jax`'s integer-clip lookup.
"""
@inline function bin_lookup(t_days::T, Phi_arr::AbstractVector{T},
                             dt_bin_days::T) where {T}
    k = trunc(Int, t_days / dt_bin_days)
    k = clamp(k, 0, length(Phi_arr) - 1)
    @inbounds Phi_arr[k + 1]   # 1-based indexing
end

"""
    expand_phi_lut(daily_phi; bins_per_day=96, wake_hour=7.0,
                   sleep_hour=23.0, tau_hours=3.0)

Deterministic Gamma(k=2) morning-loaded envelope, byte-equivalent to
`simulation.py:generate_phi_sub_daily(daily_phi, noise_frac=0)`.
Returns a `Vector{Float32}` of length `length(daily_phi)*bins_per_day`.
"""
function expand_phi_lut(daily_phi::AbstractVector{<:Real};
                         bins_per_day::Int=FSA_BINS_PER_DAY_DEFAULT,
                         wake_hour::Real=7.0,
                         sleep_hour::Real=23.0,
                         tau_hours::Real=3.0)
    n_days = length(daily_phi)
    dt_bin_hours = 24.0 / bins_per_day
    wake_duration = sleep_hour - wake_hour
    T_wake = wake_duration
    gamma_int = tau_hours^2 *
                 (1.0 - exp(-T_wake / tau_hours) * (1.0 + T_wake / tau_hours))
    gamma_int = max(gamma_int, 1e-12)

    out = zeros(Float32, n_days * bins_per_day)
    for d in 0:(n_days - 1)
        phi_d = Float64(daily_phi[d + 1])
        amp = phi_d * 24.0 / gamma_int
        for k in 0:(bins_per_day - 1)
            h = k * dt_bin_hours
            if h < wake_hour || h >= sleep_hour
                continue
            end
            t = h - wake_hour
            shape = t * exp(-t / tau_hours)
            base = amp * shape
            out[d * bins_per_day + k + 1] = Float32(max(base, 0.0))
        end
    end
    out
end
