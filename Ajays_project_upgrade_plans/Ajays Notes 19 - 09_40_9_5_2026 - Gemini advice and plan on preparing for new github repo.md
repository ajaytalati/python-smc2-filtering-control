
I have a research code base - 


which I now want to make more professional and ready for independent verification, and public github presentation

The main thing I want to do now is to simply move all the directories which are currently needed by the Julia and Lean code into a new Repo directory on my HD called Repos/LEAN4-JULIA-SM2 filtering and control

I don't not want to port over all the legacy python and Jax code - which is now mainly in the DEPRECIATED directories - most of that code was NOT pure functional and goes against the new code bases mandate of being purely functional and GPU optimized, and verifiable in LEAN4 !!!

I also want to simplify the organization and separate the 

- FSA version 1_5 model (much simpler with very simple linear observation model - used mainly for developing the SMC2 code base) 
- and new FSA version 5 (much more realistic physiological model, which complex nonlinear stability)

All testing code can be removed as can any version 1 code

All latex docs should be kept/not deleted - and flagged for review

Note this is purely an addictive and mainly copy task - YOU DO NOT HAVE PERMISSION TO EDIT ANY FILES IN THE MASTER REPO - 