# extended_sources branch: addition and modification to the original skylab code
- Made for stacking of extended spatial PDFs according to UHECR arrival directions
- Usage: see stack_multi_trials.py 
- Trial evaluation: see ts_trial_evaluation.py

# skylab

Skylab is a python based toolkit to perform unbinned likelihood analyses of
spatial clustering on the sphere, commonly used in astroparticle physics.

Documentation of the code can be found in ./doc

This python package provides an interface to do liklelihood maximisation searches
common in neutrino astronomy, but in general extendable to any clustering
search. All the magic happens within the framework of *numpy*, *scipy*, and
*healpy*.

http://github.com/coenders/skylab

Based on commit 363c999 (after pull request #4 by kkrings)

