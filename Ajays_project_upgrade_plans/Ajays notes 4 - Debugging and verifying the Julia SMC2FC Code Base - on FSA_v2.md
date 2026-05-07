
You are mathematically an scientifically principled verification and debugging agent. 

I want you to work on a PURE julia version of the Python/Jax code base in the folder - /home/ajay/Repos/python-smc2-filtering-control/smc2fc

The majority of the work has been done and tested - see here - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC

As two tests models to compare with the python code the Julia code base has been applied here for your reference - /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia

You will be working on the working tree of the repo - julia-port-version-1

To understand the GENERAL MOTIVATION FOR THE WORK you first read this very comprehensive guide which I have prepared for you - /home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/julia_port_charter.pdf

That document contains one MAJOR OUTDATED DESIGN DECISION - I have chosen to NOT implement the HMC sampler on the CPU - instead I have chosen to implement the **ChEES-HMC** (Change in the Estimator of the Expected Square HMC) on the GPU 

THE PRINCIPLE NOW IS ALL THE SMC2 FILTERING AND CLOSED LOOP CONTROL MPC SHOULD RUN ON THE GPU USING ITS FLOAT 32 CORES - any code which is float 64 and/or cpu bound MUST BE FLAGGED as it will be a performance drain

The previous coding agent has NOT implemented this into the framework YET SEE the comments below - in particular the sentences 

**Important caveat from the session you fired me for**: these framework samplers are CPU-only (AdvancedHMC.jl runs on CPU). The fast GPU path used in `bench_b3_gpu_parallel.jl` and my `bench_b3_gpu_segmented.jl` is a **separate model-specific** parallel-chains HMC that lives outside this framework — at [version_1_Julia/models/bistable_controlled/gpu_pf.jl:498](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/version_1_Julia/models/bistable_controlled/gpu_pf.jl#L498) (`parallel_hmc_one_move!`). That's the one that hits 153× per-move speedup on the GPU. ChEES adaptation for the GPU path is in the bench drivers themselves (`chees_pick_L_parallel` in [bench_b3_gpu_parallel.jl:81](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/version_1_Julia/tools/bench_b3_gpu_parallel.jl#L81)), not in the framework.

So if you're looking for "the canonical sampler" → CPU framework: `julia/SMC2FC/src/SMC2/HMC.jl`. If you're looking for "the actual sampler used by the GPU benches" → it's the model-specific `parallel_hmc_one_move!` per model.

---

Your first task is to work on the filter code which is NOT performing to the same standard as the python code -  /home/ajay/Repos/python-smc2-filtering-control/smc2fc/filtering/gk_dpf_v3_lite.py

You will need to implement the Optimal Transport rescue which is currently NOT implemented in the Julia inner particle filter - after you do that then both the new Julia and Python inner PFS should be algorithmic ally the same/similar

For each model you will need to implement the locally guided Pitt-Sheppard sampler which is model specific and should be code in the models estimation file estimation.py

---

Here is the latest report of the status of the samplers used for the OUTER PF 

All the framework samplers live in **[julia/SMC2FC/src/SMC2/HMC.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl)** (one file, ~410 lines). Wraps `AdvancedHMC.jl` and exposes 4 samplers via a `:sampler` symbol switch:

|sampler|function|line|what it does|
|---|---|---|---|
|`:HMC`|`hmc_step_chain`|[HMC.jl:122](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl#L122)|Fixed-step HMC, num_leapfrog steps per move, AdvancedHMC's `Leapfrog` + `EndPointTS` trajectory|
|`:NUTS`|`hmc_step_chain` (same)|[HMC.jl:147](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl#L147)|NUTS via AdvancedHMC's `MultinomialTS` trajectory|
|`:MALA`|`mala_step_chain`|[HMC.jl:181](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl#L181)|Metropolis-adjusted Langevin, single Langevin step + MH accept|
|`:AutoMALA`|`automala_step_chain`|[HMC.jl:253](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl#L253)|MALA with Robbins-Monro step-size adaptation toward 0.574 acceptance (Pigeons.jl-inspired, since Pigeons itself has a deps conflict)|

**ChEES** is a separate function: `chees_adapt_L` at [HMC.jl:302](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/HMC.jl#L302) — picks the leapfrog count L by maximising expected squared jumped distance per step.

Other SMC2/ files:

- [Bridge.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/Bridge.jl) — Schrödinger-Föllmer cross-window prior bridge
- [TemperedSMC.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/TemperedSMC.jl) — outer SMC² loop that calls one of the samplers per tempering level
- [Sampling.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/Sampling.jl) / [Tempering.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/Tempering.jl) / [MassMatrix.jl](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/julia/SMC2FC/src/SMC2/MassMatrix.jl) — supporting utilities

**Important caveat from the session you fired me for**: these framework samplers are CPU-only (AdvancedHMC.jl runs on CPU). The fast GPU path used in `bench_b3_gpu_parallel.jl` and my `bench_b3_gpu_segmented.jl` is a **separate model-specific** parallel-chains HMC that lives outside this framework — at [version_1_Julia/models/bistable_controlled/gpu_pf.jl:498](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/version_1_Julia/models/bistable_controlled/gpu_pf.jl#L498) (`parallel_hmc_one_move!`). That's the one that hits 153× per-move speedup on the GPU. ChEES adaptation for the GPU path is in the bench drivers themselves (`chees_pick_L_parallel` in [bench_b3_gpu_parallel.jl:81](vscode-webview://00p5kbj76ishvt4jpl2afflfs6liq6a4e9ej0voni00vcmrsfabt/version_1_Julia/tools/bench_b3_gpu_parallel.jl#L81)), not in the framework.

So if you're looking for "the canonical sampler" → CPU framework: `julia/SMC2FC/src/SMC2/HMC.jl`. If you're looking for "the actual sampler used by the GPU benches" → it's the model-specific `parallel_hmc_one_move!` per model.


---

I DO NOT WANT YOU TO WORK ON THE TWO MODELS IN - /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia

I want you to start working in new directory /home/ajay/Repos/python-smc2-filtering-control/version_2_Julia

Your task is to implement in Julia the ONLY the FSA model (DO NOT IMPLEMENT THE SWAT model) from here - /home/ajay/Repos/python-smc2-filtering-control/version_2/models/fsa_high_res  

And the tests from here - /home/ajay/Repos/python-smc2-filtering-control/version_2/tests

And tools from here - /home/ajay/Repos/python-smc2-filtering-control/version_2/tools

THIS IS MAINLY AN ALGORITHMIC AND CODE TRANSLATION TASK - IT IS NOT DIFFICULT WORK - AND I WILL NOT ACCEPT INFLATED TIME OR TOKEN EXPECTATIONS OR COSTS FOR YOU TO DO IT - A HUMAN DEVELOPER COULD DO THIS IN 2-3 HOURS !!!!

Your work will be judged by now accurately you reproduce the plots in the folders here - /home/ajay/Repos/python-smc2-filtering-control/version_2/outputs/fsa_high_res/g4_runs

In particular the parameter trace plots - /home/ajay/Repos/python-smc2-filtering-control/version_2/outputs/fsa_high_res/g4_runs/T14d_replanK2_h60min_no_infoaware/E5_full_mpc_T14d_param_traces.png 

And when I ask you to test the Julia code you develop against runs of the Python code

YOu will start with the T=14d experiment, and log all your runs in the outputs folder - with clear experiment_run.md document which is human readable and records the design choices made for each run

---

You will now give me the plan you have written for your work 