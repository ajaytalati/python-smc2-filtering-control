
You are a mathematical and statistically principled verification and debugging agent 

You will read the report of an 6 hr GPU experiment here -  /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09/docs/v15_horizon_sweep_writeup.pdf

You will read this (outdated guide for the technical details of the framework - which refers to the Python-JAX implementation - not the current Julia code base)

You will investigate the Julia code base to investigate how it works as needed - and create a semantic map of the functional Julia code (DONT JUST PATTERN MATCH ON DOCS STRING - READ THE CODE) - this exists in THREE places

general framework - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC_functional

fsa_v_1_5_specific - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res

benchmark-experiment specific - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl

I want you first to give me TOP THREE evidence and reasoning based HYPOTHESIS of WHY all the parameter trace plots in the report are biased - in particular WHY the bias still exists as the planning horizon grows (i.e. more observations) !!!!

Then give me a plan how you will test your TOP THREE hypothesis - this will gate whether I allow you GPU time !!!!


---

My hypothesis is that the Lui-west shrinkage was switch off (stupid default) - and that the study needs to be rerun with it switched on - ADD THIS TO YOUR PLAN - HYPOTHESIS 

optional Liu–West θ-cloud shrinkage File: same plus version 1 5 Julia/tools/bench smc full mpc fsa gpu.jl. Added a CLI flag --liu-west-a (default 0.0 = disabled). When set in (0, 1) — e.g. 0.97 — applies the standard Liu–West shrinkage between resample and HMC inside run outer smc: θi ← a · θi + (1 − a) · θ + p1 − a2 · column std · ξ, ξ ∼ N (0, I) Helps maintain θ-cloud diversity at low N , where pure tempered HMC can collapse. 

Off by default so existing reproducibility is preserved.

---

You wire these in now

## On Bridge.GaussianBridge + Kernels.smooth_resample

You're right — these are O(d³) Cholesky / O(K²·d) kernel — _microseconds_ compared to HMC's ~ms-per-call (HMC fires K_per_chain × T_steps × M chains worth of inner-PF). Not "next steps" — they should be on by default. I framed them wrong.

Concretely, what I'd wire in (to confirm before I touch source):

1. **Between-window propagation** ([`bench_smc_full_mpc_fsa_gpu.jl:632-640`](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L632)): replace the identity-copy of `acc.filter_post` with `bridge_init(GaussianBridge(), prev_post, …, cfg, rng)` from [`Bridge.jl:110-118`](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/julia/SMC2FC_functional/src/SMC2/Bridge.jl#L110). One-time cost per window: fit `(μ, Σ̂_reg)`, sample `n_smc` × Cholesky.
2. **Between-tempering-level rejuvenation** ([`bench_smc_full_mpc_fsa_gpu.jl:284-290`](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl#L284)): replace the per-dim Liu-West with a call to a θ-cloud version of `smooth_resample` from [`Kernels.jl:184-206`](vscode-webview://1ejb69m9th6lj792cfn63fb24rrdi6fv7sg24mrvl33ggif2c936/julia/SMC2FC_functional/src/Filtering/Kernels.jl#L184). Caveat: `smooth_resample` is currently signed for state-PF use (`stochastic_idx` argument); I'd either lift it or copy the Silverman+LW math (~20 lines) directly into the bench.

Both behind a CLI flag (default ON) so the existing bench reproducibility remains addressable. Want me to do this now? (Per the still-active plan-mode reminders I haven't started; want a clean go-ahead before I edit source.)

AND ALSO CHANGE THE LUI-WEST TO ALWAYS BE ON WITH 0.97

----

I WANT TO TEST THE HYPOTHESIS THAT THE HMC IS NOT NEEDED

Now I want you to rerun the 6 horizon study - but with 

1) the FILTER HMC turned OFF, .i.e. zero leapfrog steps etc,.. all the HMC flag set to zero -
2) the Lui-West shrinkage turned on (it is now by default), AND ALL THE OTHER TRICKS -silver man etc, Gaussian bridge
3) The faster (reduced) controller config - 

write a short plan to confirm to me you understand this 

---

write this up as a latex report - you will include all the paratrace plots - like this /home/ajay/Repos/python-smc2-filtering-control/compare_v15_julia_vs_python/example_run/julia_horizon_sweep_2026-05-09_no_filter_hmc/T14d_seed42/v15_T14d_param_traces.png Include all the tables in the markdown doc - add a section at the begining what the purpose of the study was - to gather data/a baseline run o f how the filter performs WITH THE HMC -but with all the rejuvenation tricks. The conclusion is EVEN FOR THIS VERY SIMPLE linear noise observation model - the filter with HMC CANNOT EXACTLY estimate the posterios / model parameters

---


OK so now we have collected evidence / data of the baseline - WITHOUT the filter's HMC - AND DOCUMENTED IT

I want you to now describe how the filters HMC is not working and the tempering parameter bug.

Then how you will code the fixes.

Then you will run FAST test experiments for t=14, t=28, t=42 - using the reduced controller config - to see if your coded fixes work !!!!

Write plan for this





