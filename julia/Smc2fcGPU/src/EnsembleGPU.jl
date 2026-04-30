using StaticArrays
using Adapt
using CUDA
using SciMLBase
using StochasticDiffEq
using DiffEqGPU

"""
    FsaSDEParams{N, T}

Kernel-friendly SDE parameter bundle for FSA-v2.

`p` carries the G1-reparametrized scalars; `Phi_lut::SVector{N, T}` is the
per-bin training strain (held inline, register-resident — required because
`EnsembleGPUKernel` ships the *whole* `prob` to the device as a single
isbits struct, and a heap-allocated CuArray field would not be inline);
`dt_bin_days` is the outer-grid spacing used for the integer bin lookup.

For our use-case `N` = `length(daily_phi) * 96` (96 bins/day default).
At 1 day, N=96 → 384 bytes / trajectory which is comfortable for L1/regs.
"""
struct FsaSDEParams{N, T<:AbstractFloat}
    p::FsaParams{T}
    Phi_lut::SVector{N, T}
    dt_bin_days::T
end

@inline function FsaSDEParams(p::FsaParams{T}, Phi_lut::AbstractVector,
                               dt_bin_days::T) where {T<:AbstractFloat}
    N = length(Phi_lut)
    sv = SVector{N, T}(ntuple(i -> T(Phi_lut[i]), N))
    FsaSDEParams{N, T}(p, sv, dt_bin_days)
end

@inline function bin_lookup(t_days::T, Phi_lut::SVector{N, T},
                             dt_bin_days::T) where {N, T}
    k = trunc(Int, t_days / dt_bin_days)
    k = clamp(k, 0, N - 1)
    @inbounds Phi_lut[k + 1]
end

"""
    fsa_sde_drift(u, p, t)

Out-of-place SDE drift for `EnsembleGPUKernel(GPUEM())`. Mirrors
`SDESolver.em_oracle_single`'s drift step exactly. Returns `SVector{3,T}`.
"""
@inline function fsa_sde_drift(u::SVector{3,T},
                                 p::FsaSDEParams{N,T}, t::T) where {N, T}
    Phi_t = bin_lookup(t, p.Phi_lut, p.dt_bin_days)
    drift(u, p.p, Phi_t)
end

"""
    fsa_sde_diffusion(u, p, t)

Out-of-place state-dependent diagonal diffusion (DIFFUSION_DIAGONAL_STATE).
Mirrors `simulation.py:noise_scale_fn_jax` × `sigma`. Returns `SVector{3,T}`.
"""
@inline function fsa_sde_diffusion(u::SVector{3,T},
                                     p::FsaSDEParams{N,T}, t::T) where {N, T}
    noise_scale(u, p.p)
end

"""
    run_em_ensemble_gpu(y0, p, Phi_lut, t_grid;
                        n_trajectories=16_384, n_substeps=10, seed=42,
                        backend=CUDA.CUDABackend())

Run `n_trajectories` independent FSA-v2 SDE trajectories on the GPU via
`EnsembleGPUKernel(GPUEM())` with fixed step `dt = (t_grid[2]-t_grid[1])/n_substeps`.

Returns an `Array{T,3}` of shape `(length(t_grid), 3, n_trajectories)` —
trajectories along dim 3, matching the convention of `EnsembleProblem`.
"""
function run_em_ensemble_gpu(y0::SVector{3,T},
                              p::FsaParams{T},
                              Phi_lut::AbstractVector{T},
                              t_grid::AbstractVector{T};
                              n_trajectories::Int=16_384,
                              n_substeps::Int=10,
                              seed::Int=42,
                              backend=CUDA.CUDABackend()) where {T<:AbstractFloat}

    dt_bin_days = T(t_grid[2] - t_grid[1])
    dt_sub = dt_bin_days / T(n_substeps)
    tspan = (T(t_grid[1]), T(t_grid[end]))

    # Phi LUT lives inline as SVector — register-resident inside the kernel.
    sde_params = FsaSDEParams(p, collect(T, Phi_lut), dt_bin_days)

    # SDEProblem with diagonal noise + out-of-place SVector functions
    # (DIFFUSION_DIAGONAL_STATE semantics matching the JAX path).
    # GPUEM requires SVector u0 for kernel-fused execution.
    u0 = SVector{3, T}(y0[1], y0[2], y0[3])
    prob = SDEProblem{false}(
        fsa_sde_drift, fsa_sde_diffusion, u0, tspan, sde_params,
        seed=UInt64(seed),
    )

    # Per-trajectory variation: vary the random seed only; same y0 / params.
    function prob_func(prob, i, repeat)
        remake(prob; seed=UInt64(seed) + UInt64(i))
    end

    eprob = EnsembleProblem(prob; prob_func=prob_func, safetycopy=false)

    sol = solve(
        eprob, GPUEM(),
        EnsembleGPUKernel(backend, 0.0);
        trajectories=n_trajectories,
        dt=dt_sub,
        adaptive=false,
        saveat=collect(T, t_grid),
        save_everystep=false,
    )

    # Stack per-trajectory matrices into `(n_t_out, 3, K)`.
    # EnsembleGPUKernel + saveat returns either an EnsembleSolution (CPU u
    # vectors) or a (state, time, trajectory) 3D Array depending on options.
    # The number of saved points may be `length(saveat)` OR `length(saveat)+1`
    # depending on whether t0 is included — both are accepted; we infer from
    # the first trajectory and prepend y0 if needed.
    if sol isa AbstractArray && ndims(sol) == 3
        host = Array(sol)
        n_t_kernel = size(host, 2)
        if n_t_kernel == length(t_grid) - 1
            out = Array{T, 3}(undef, length(t_grid), 3, n_trajectories)
            @inbounds for i in 1:n_trajectories
                out[1, 1, i] = y0[1]; out[1, 2, i] = y0[2]; out[1, 3, i] = y0[3]
                for k in 1:n_t_kernel, j in 1:3
                    out[k + 1, j, i] = host[j, k, i]
                end
            end
            return out
        else
            out = Array{T, 3}(undef, n_t_kernel, 3, n_trajectories)
            @inbounds for i in 1:n_trajectories, k in 1:n_t_kernel, j in 1:3
                out[k, j, i] = host[j, k, i]
            end
            return out
        end
    else
        # EnsembleSolution with per-trajectory u vectors of SVector{3,T}.
        first_u = sol[1].u
        n_t_kernel = length(first_u)
        n_t_out = (n_t_kernel == length(t_grid) - 1) ? length(t_grid) : n_t_kernel
        prepend_y0 = (n_t_kernel == length(t_grid) - 1)
        out = Array{T, 3}(undef, n_t_out, 3, n_trajectories)
        for (i, traj_sol) in enumerate(sol)
            u = traj_sol.u
            @assert length(u) == n_t_kernel "trajectory $i has $(length(u)) points, expected $n_t_kernel"
            if prepend_y0
                out[1, 1, i] = y0[1]; out[1, 2, i] = y0[2]; out[1, 3, i] = y0[3]
                @inbounds for k in 1:n_t_kernel
                    out[k + 1, 1, i] = u[k][1]
                    out[k + 1, 2, i] = u[k][2]
                    out[k + 1, 3, i] = u[k][3]
                end
            else
                @inbounds for k in 1:n_t_kernel
                    out[k, 1, i] = u[k][1]
                    out[k, 2, i] = u[k][2]
                    out[k, 3, i] = u[k][3]
                end
            end
        end
        return out
    end
end
