# Per-chain host-loop catalogue (the §2.12 / §2.14 ceiling)

> Read-only audit. If Phase 1's median GPU util plateaus at ~50% regardless
> of `(N, K)`, this is the source-level fix list. Each entry names the file,
> line range, what scales with `N_smc`, and what an on-device fix would look
> like. Catalogue is referenced from `RESULTS.md` once Phase 1 confirms the
> plateau.

## Summary table

| # | Location | Loop body | Scales with | CPU dependency that prevents batching | On-device fix |
|---|---|---|---|---|---|
| 1 | [bench_smc_full_mpc_fsa_gpu.jl:128-130](../../../tools/bench_smc_full_mpc_fsa_gpu.jl#L128-L130) | Cold-start prior sampling | n_smc · d | RNG per element | One vectorised `randn(rng, n_smc, d)` |
| 2 | [bench_smc_full_mpc_fsa_gpu.jl:205-207](../../../tools/bench_smc_full_mpc_fsa_gpu.jl#L205-L207) | Constraint transform per row | n_smc · d | LOGNORMAL_MASK conditional | Element-wise broadcast |
| 3 | [bench_smc_full_mpc_fsa_gpu.jl:346-358](../../../tools/bench_smc_full_mpc_fsa_gpu.jl#L346-L358) | State extraction `_extract_xhat` — triple loop over M × K × 3 | M_smc · K · 3 | `Array(view(...))` copy of particles + log_w from GPU; per-chain logsumexp on host | Fuse weighted-mean into the existing per-chain stats kernel; emit one (M, 3) tensor instead of two (M·K, ·) tensors |
| 4 | [models/fsa_high_res/gpu_pf.jl:369-372](../../../models/fsa_high_res/gpu_pf.jl#L369-L372) | `params_cpu` rebuild + copyto per `gpu_log_density_batched` call | M · d | Fresh allocation + per-row constraint transform; called once per tempering level + once per HMC stencil row → ≥ (1 + 2d) × num_mcmc × n_temp times per stride | Keep `params_per_chain` GPU-resident across calls within a stride; update only changed rows via a per-chain constraint kernel |
| 5 | [models/fsa_high_res/gpu_pf.jl:417-421](../../../models/fsa_high_res/gpu_pf.jl#L417-L421) | Per-chain log-lik accumulation each segment | M, repeated per segment (T_steps ÷ R times) | `Array(log_max[1:M])` + `Array(log_z[1:M])` per segment | Maintain `log_lik_acc` as a GPU array; one host copy at end of window |
| 6 | [models/fsa_high_res/gpu_pf.jl:462-471](../../../models/fsa_high_res/gpu_pf.jl#L462-L471) | FD-perturbation expansion M → M·(1+2d) = 61M rows | M · d² | Host loop building U_flat | Single kernel that reads U_unc and writes the perturbed rows directly on GPU |
| 7 | [models/fsa_high_res/gpu_pf.jl:477-483](../../../models/fsa_high_res/gpu_pf.jl#L477-L483) | Gradient assembly from FD stencil | M · d | Host indexing into lls_flat | (M, d) gradient kernel — central-difference on GPU, one copy back |
| 8 | [models/fsa_high_res/gpu_pf.jl:527-533](../../../models/fsa_high_res/gpu_pf.jl#L527-L533) | HMC accept/reject + posterior cloud update | M, runs every HMC move (num_mcmc × n_temp times per stride) | `rand(rng)` per chain on host; conditional `U[m, :] = U_new[m, :]` row copy | GPU kernel with pre-drawn RNG bits + atomic accumulator for `n_acc`; in-place row update |
| 9 | [julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl:216-256](../../../../julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl#L216-L256) | Per-chain OT-rescue blend | M, runs at every resample | `Array(ess_gpu)` full copy per check; per-chain Nyström anchor index read from CPU buffer; per-chain try/catch | Fuse OT blend into one batched kernel (one warp per chain); move anchors GPU-resident; replace ESS copy with a GPU-side mask |
| 10 | [julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl:352-362](../../../../julia/SMC2FC/src/Filtering/GPUSegmentedPF.jl#L352-L362) | Unconditional `KernelAbstractions.synchronize` after every segment kernel | n_segments = T_steps ÷ R | Sync blocks the host stream | Defer sync to window-end; pipeline segmented kernels via CUDA streams |

## What scales with what

The two host-loop categories that grow with `N_smc` and dominate at large N:

1. **Per-chain loops in the HMC update path** — entries 4, 6, 7, 8 in the table. These run
   `num_mcmc × n_temp` times per stride. At Python's reference `num_mcmc=10, n_temp≈10`, that's
   ~100 launches per stride, each with `O(M)` host work. Doubling `n_smc` doubles M, doubles
   the host work, and the GPU sits idle in between.

2. **Per-segment sync + per-chain copy-back in the inner-PF** — entries 5, 9, 10. These run
   `T_steps / R` times per filter call (~24 segments per window at h=60min). At each segment
   the bench pulls `(M, ·)` log_max + log_z back to host (entry 5) and full `ess_gpu` copy
   (entry 9), then accumulates per-chain on the host. This is what the writeup calls "small
   but per-chain `Array(view(...))` copy back" in §2.14.

## Order to attack (cheapest wins first)

If util plateaus at ~50%, the order I'd try the source fixes in is:

1. **Entry 4 (params_cpu rebuild)** — easiest. The constraint transform is a pure element-wise
   op on a fixed-shape matrix; moving `params_per_chain` to a `CuArray` and writing a
   per-chain constraint kernel is ~30 lines of code and removes one of the two largest
   per-call host costs.
2. **Entry 8 (HMC accept/reject)** — also cheap. Pre-draw RNG into a CuArray of `M` floats,
   compare on GPU, do the row copy on GPU. ~40 lines.
3. **Entry 6 + 7 (FD expansion + gradient assembly)** — slightly more work. The `(1 + 2d)`
   row layout is regular; a single kernel can read U_unc and write all 61 rows directly on
   device, then a paired kernel does the central-difference on the result.
4. **Entries 5 + 10 (per-segment sync + log-lik copy)** — the most invasive. Requires
   restructuring the inner-PF loop in `GPUSegmentedPF.jl` to keep the per-segment reductions
   GPU-resident, with one final host copy at window end. Probably 100+ lines.
5. **Entry 9 (OT rescue batching)** — last because Sinkhorn's iterative inner loop is harder
   to vectorise across chains. But this is also the one that benefits most from larger N
   because rank-eff anchor selection scales with K.

Entries 1, 2, 3 (in the bench, lines 128–130, 205–207, 346–358) are smaller and only run
once per stride — fix them last unless they show up in a profile.
