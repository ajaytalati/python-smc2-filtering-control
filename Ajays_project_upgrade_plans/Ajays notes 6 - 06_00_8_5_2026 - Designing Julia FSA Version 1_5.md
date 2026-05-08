
I want you to plan a very simple project now that you have a good understanding of the Julia FSA version 2 Code base

First I want you to understand the very simple open loop FSA model version 1 model here - /home/ajay/Repos/python-smc2-filtering-control/version_1_Julia/models/fsa_high_res

It is a Julia GPU model which is exactly the same as FSA version two - except for re-parametrization to allow identifability.

I want you to setup a new directory -  /home/ajay/Repos/python-smc2-filtering-control/version_1_5_Julia

And do the following VERY SIMPLE things

1) Add the filtering code - the observation model is simple Gaussian noise to each of the latent variable - so this is the SIMPLEST OBSERVATION model possible 
2) Add a very simple plant code - is basically "glue" for closed loop MPC
3) For the control file use exactly the existing file from FSA version 1

So this code base will act as a bridge between FSA versions 1 and 2 - it is a very simple extension of FSA v1 from open to closed loop

Give me your reasoning what are the pros and cons of this plan - and how you will do it


---

NO !!! Remove the phi_burst


 
