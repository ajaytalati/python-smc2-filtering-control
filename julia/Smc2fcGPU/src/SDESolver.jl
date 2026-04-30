using StaticArrays

"""
    em_oracle_single(y0, p, t_grid, Phi_arr, noise; n_substeps=10,
                     bounds=((0.0,1.0),(0.0,10.0),(0.0,5.0)))

Hand-rolled fixed-step Euler-Maruyama for a single FSA-v2 trajectory,
byte-parity oracle for `smc2fc.simulator.sde_solver_diffrax.solve_sde_jax`.

Arguments
---------
- `y0` :: SVector{3,T}                    initial state [B, F, A]
- `p`  :: FsaParams{T}                    G1-reparametrized parameters
- `t_grid` :: AbstractVector{T}           outer time grid (length T_grid)
- `Phi_arr` :: AbstractVector{T}          per-bin Phi (e.g. from `expand_phi_lut`),
                                           must cover the integration window;
                                           lookup uses `dt_bin_days = t_grid[2]-t_grid[1]`
- `noise` :: AbstractArray{T, 3}          shape `(n_substeps, n_grid, 3)`,
                                           pre-generated N(0,1) draws (column-major,
                                           matching the JAX `(n_grid, n_substeps, n_states)`
                                           tensor after a Python-side transpose).
- `n_substeps` :: Int                     number of Euler-Maruyama substeps per outer
                                           grid interval (default 10).

Returns
-------
- `traj` :: Matrix{T} of shape `(T_grid, 3)` — same convention as
  `solve_sde_jax`: row 1 is `y0`, rows 2..end are post-substep states.

The numerical step matches `sde_solver_diffrax.py:inner_step` exactly:

    sigma_y = sigma .* g(y)
    diff    = sigma_y .* sqrt_dt .* noise_unit
    y_new   = y + dt_sub .* drift(t, y) + diff
    y_new   = clip(y_new, bounds)

with `g(y) = [sqrt(B(1-B)); sqrt(F); sqrt(A + EPS_A_FROZEN)]`,
B clipped to `[EPS_B_FROZEN, 1 - EPS_B_FROZEN]` before the sqrt.
"""
function em_oracle_single(y0::SVector{3,T},
                           p::FsaParams{T},
                           t_grid::AbstractVector{T},
                           Phi_arr::AbstractVector{T},
                           noise::AbstractArray{T, 3};
                           n_substeps::Int = 10,
                           bounds = ((zero(T),  one(T)),
                                     (zero(T),  T(10.0)),
                                     (zero(T),  T(5.0)))) where {T<:AbstractFloat}
    n_grid = length(t_grid) - 1
    @assert size(noise) == (n_substeps, n_grid, 3) "noise shape must be (n_substeps, n_grid, 3); got $(size(noise))"

    dt_grid = t_grid[2] - t_grid[1]
    dt_sub = dt_grid / T(n_substeps)
    sqrt_dt = sqrt(dt_sub)

    traj = Matrix{T}(undef, n_grid + 1, 3)
    traj[1, 1] = y0[1]; traj[1, 2] = y0[2]; traj[1, 3] = y0[3]

    lo1, hi1 = T(bounds[1][1]), T(bounds[1][2])
    lo2, hi2 = T(bounds[2][1]), T(bounds[2][2])
    lo3, hi3 = T(bounds[3][1]), T(bounds[3][2])

    y = y0
    t = t_grid[1]
    for k in 1:n_grid
        for s in 1:n_substeps
            Phi_t = bin_lookup(t, Phi_arr, T(dt_grid))
            dy = drift(y, p, Phi_t)
            sigma_y = noise_scale(y, p)
            n1 = noise[s, k, 1]; n2 = noise[s, k, 2]; n3 = noise[s, k, 3]
            y_new = SVector{3,T}(
                clamp(y[1] + dt_sub * dy[1] + sigma_y[1] * sqrt_dt * n1, lo1, hi1),
                clamp(y[2] + dt_sub * dy[2] + sigma_y[2] * sqrt_dt * n2, lo2, hi2),
                clamp(y[3] + dt_sub * dy[3] + sigma_y[3] * sqrt_dt * n3, lo3, hi3),
            )
            y = y_new
            t += dt_sub
        end
        traj[k + 1, 1] = y[1]; traj[k + 1, 2] = y[2]; traj[k + 1, 3] = y[3]
    end
    traj
end

"""
    run_em_oracle!(traj, y0, p, t_grid, Phi_arr, noise; n_substeps=10)

In-place version writing into pre-allocated `traj::Matrix{T}` of shape
`(length(t_grid), 3)`.
"""
function run_em_oracle!(traj::AbstractMatrix{T},
                         y0::SVector{3,T},
                         p::FsaParams{T},
                         t_grid::AbstractVector{T},
                         Phi_arr::AbstractVector{T},
                         noise::AbstractArray{T, 3};
                         n_substeps::Int = 10) where {T<:AbstractFloat}
    out = em_oracle_single(y0, p, t_grid, Phi_arr, noise; n_substeps=n_substeps)
    @assert size(traj) == size(out)
    traj .= out
    traj
end
