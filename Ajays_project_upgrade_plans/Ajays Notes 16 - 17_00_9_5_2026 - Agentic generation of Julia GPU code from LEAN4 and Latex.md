
I want you to write me a plan for 

- Give the FSA v5 LaTex model definition pdf and tex doc

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.pdf
/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/LaTex_docs/FSA_version_5_technical_guide.tex

- And the formulation as LEAN code

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_LEAN/Fsa/V15

How you will write the the FSAv5 updates of the FSAv1_5 Julia code, which I can give you to see the patterns

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res

I also want you to write the diff_test.jl for this

I WANT YOU TO BE VERY METHODOLOGICAL ABOUT THIS AND DOCUMENT ALL THE CODE, AND THE PROCESS AND REASONING HOW YOU WROTE IT - SO IT IS REPRODUCIBLE 


---


We now need to write the "bench" script which is the final piece before we can actual run the new model v5! 

The v_1_5 script is here -/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/tools/bench_smc_full_mpc_fsa_gpu.jl 

See also the bench dir

It has been substantially refactored so that there is now a minimal model specific module - in the v_1_5 model dir

/home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia/models/fsa_high_res/bench_glue.jl

the rest is all model agnostic - you do not need to change it,? Just copy it 

I want you to make a new dir tools_v5 and put you new v5 bench script there - copy the bench folder  -and write the model specific module - which should be placed with the v5 model code. So the same format/pattern as the v_1_5 model is used

