
Ok now I want you to plan another translation task!

Currently I do not have a python-Jax version of the simplified Julia closed loop control of the FSA_version 1_5 model

In order to more thoroughly test the Julia version - it would be a great help to develop a PythonJax version.

Can you do this by creating in new directory - /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Python_JAX

python versions of these Julia version1_5 files - version_1_5_Julia/models/fsa_high_res

This should very similar to the existing Python_JAX code for 

version_2 - (the plant is more complex and there is an elaborate observation model) -  version_2_Python_JAX/models/fsa_high_res

version_1 - there is no filter code or plant - this is purely open loop control - version_1_Python_JAX/models/fsa_high_res

---

As a follow on to this I now want you to plan the actual run testing of the Julia verse Python FSA model 1_5 code bases.

They should give all most exactly the same qualitative plots / results - what I have found though is that the Julia code-base seems to be failing for closed loop control? 

As I've already tested the filter and controller for open loop control - and both seem reasonable - my hypothesis are 

H1 - that there is a bug on the orchestration which is needed for closed loop control ???
H2 - that the Julia code is simply NOT giving the controller enough computational power - the Julia code severly under uses my RTX-5090, while the python code with the settings used for the verison 2 model maxes it out

Can you thus write a plan to do this side my side testing - and if LEAN can lets definitely use it too!!!

I think this is more sensible to do for t=28 days rather than shorter experiments - 
