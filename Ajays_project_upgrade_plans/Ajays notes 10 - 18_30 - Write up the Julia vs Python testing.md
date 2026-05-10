
Ok we are at the point where it makes sense to do a write up - especially as this comparison work is so nuanced and there are many things we have tested that are stored in the context of this interaction.

Thus, can you create a new director on the root dir called - compare_v15_julia_vs_python

And make copies of all the files that are need for carry of the comparison and development work, i.e.

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX/tools/compare_v15_julia_vs_python.py
/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX/tools/profile_gpu.py
/home/ajay/Repos/python-smc2-filtering-control/version_2_Julia_DEPRECIATED_NOT WORKING/tools/profile_gpu.jl

anything else (files) another agent would find helpful

then into that folder add a /docs folder -  and in it write a comprehensive technical report in Latex - for another agent who will be asked to carry on this work

At the minimum it should contain,

- what the version 1_5 FSA model is, what the SMC2FC repo does, the functional programming paradigm we now use, 
- describes the problem of comparing the two versions of fsa_v_1_5 implemented using the two different frameworks Python vs Julia - how you did that, what code you developed, etc
- what we have done to fix things
- where we are up to
- what comes next - speed up the julia code, and optimize it for gpu use and performance - test at longer horizons i.e. T=28
- anything else you want to add?
- the basics of how the two repos work
- how the models are implemented
- how to profile them

Also that we now have a function programmed version of the Julia SMC2FC library here - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC_functional

Proof of principle here  - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC_functional/benchmarks/bench_smc_full_mpc_fsa_v15_functional_gpu.jl

And that should now be the default used going forward - it (should be) a direct drop in replacement  


