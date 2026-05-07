
You are mathematically an scientifically principled verification and debugging agent. 

I want you to work on a PURE julia version of the Python/Jax code base in the folder - /home/ajay/Repos/python-smc2-filtering-control/smc2fc

The majority of the work has been done and tested - see here - /home/ajay/Repos/python-smc2-filtering-control/julia/SMC2FC

You will be working on the working tree of repo - julia-port-version-1

To understand the work you first read this very comprehensive guide which I have prepared for you - /home/ajay/Repos/python-smc2-filtering-control/LaTex_docs/julia_port_charter.pdf

That document contains one MAJOR OUTDATED DESIGN DECISION - I have chosen to NOT implement the HMC sampler on the CPU - instead I have chosen to implement the **ChEES-HMC** (Change in the Estimator of the Expected Square HMC)

You first is to work on the filter code which is NOT performing to the same standard as the python code -  