
Ok so we are now abandoning python totally - we are switching to pure type safe languages i.e. LEAN4 and Julia. 

The smc2fc codebase is being ported to Julia, using the plan in part 2 here - read this - /home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/julia_port_charter.pdf 

The current state is fully coded and tested - here - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC 

To prepare for this you should begin planing how you will write the Julia versions of the FAS_v5 model -  /home/ajay/Repos/python-smc2-filtering-control/version_3/models/fsa_v5 - EXPLICIT using the LEAN4 primitive modelling files you have just written - this is non-negotiable !!!! 

This should be a MUCH more straight forward process than writing the python equivalents - 

I will also give you the responsibility of writing the Julia ports of the 

tools - /home/ajay/Repos/python-smc2-filtering-control/version_3/tools 
tests - /home/ajay/Repos/python-smc2-filtering-control/version_3/tests 

These are secondary priorities and again routine TRANSLATION !!! 

It would be sensible to thus to do this in a sister folder version_3_julia 

You should first plan this before writing a single line of Julia code !!!