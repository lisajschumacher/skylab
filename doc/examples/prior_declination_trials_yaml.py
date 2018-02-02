#!/bin/env python

# -*-coding:utf8-*-

import os
import logging
import time
from socket import gethostname
from argparse import ArgumentParser
from yaml import load as yload
import cPickle as pickle

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np
import numpy.lib.recfunctions

# skylab
from skylab.psLLH import MultiPointSourceLLH
from skylab.prior_generator import UhecrPriorGenerator
from skylab.prior_injector import PriorInjector

# local
# sorry but I had to change the names
# this is too confusing for me and the interpreter :D
import ic_utils as utils

level=logging.WARNING
logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(level)
logging.getLogger("skylab.prior_injector.PriorInjector").setLevel(level)
logging.getLogger("MixIn").setLevel(level)

pVal_func = None
ident = [
"nsamples",
"hecut",
"mdparams",
"ecut",
"mu",
"fixedgamma"
]

backend = "pdf"
extension = "_test_trials.png"

plt = utils.plotting(backend=backend)

parser = ArgumentParser()

parser.add_argument("yaml_file", 
                    type=str,  
                    help="yaml file for setting parameters"
                   )
parser.add_argument("job", 
                    type=int,  
                    help="Job ID"
                   )

if __name__=="__main__":
    # parse the yaml file
    inputs = parser.parse_args()
    yaml_file = inputs.yaml_file
    jobID = inputs.job
    args = yload(file(yaml_file))
    print("Loaded yaml args...:")
    print(args)
    # get the parameter args and get a string for saving later
    if not "test" in args["add"].lower():
        # Add time since epoch as additional unique label if we are not testing        
        identifier = args["add"]
        if identifier[-1]!="_": identifier+="_"
        for arg in ident:
            identifier+=arg+"-"+str(args[arg])+"_"
        # Remove last underscore
        if identifier[-1]=="_": identifier=identifier[:-1]
        seed = np.random.randint(2**32-jobID)
    else:
        identifier = args["add"]
        seed = jobID
        
    basepath, inipath, savepath, crpath, _ = utils.get_paths(gethostname())
    print "Data will be saved to: ", savepath
    print "With Identifier: ", identifier

    if not args["hecut"]:
        hemispheres = dict(North = np.radians(args["north_range"]), 
                           South = np.radians(args["south_range"]))
        sinDec_range =  [-1., 1.]
    else:
        hemispheres = dict(North = np.radians(args["north_range"]))
	sinDec_range = [np.sin(hemispheres["North"][0]), 1.]
    nside = 2**args["nsideparam"]

    # Other stuff
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 1
    else:
        ncpu = 1
    
    startup_dict = dict(basepath = basepath,
                        inipath = inipath,
                        seed = seed,
                        Nsrc = 0,
                        fixed_gamma = args["fixedgamma"],
			gamma_range = args["gammarange"],
                        add_prior = False,
                        src_gamma = args["srcgamma"],
                        fit_gamma = args["fitgamma"],
                        multi = True if args["nsamples"]>1 else False,
                        nside_param = args["nsideparam"],
                        burn = args["burn"],
                        ncpu = ncpu,
                        n_samples = args["nsamples"],
                        he_cut = args["hecut"],
			sinDec_range = sinDec_range,
                        mode = "box")

    llh, injector = utils.startup(**startup_dict)
    
    if injector==None:
        mu = None
    else:
        mu = injector.sample(args["mu"], poisson=True, position=True)
    ##################
    num = 60
    declinations = np.arcsin(np.linspace(-1., 1., num=num+1))
    declinations = (declinations[:-1]+declinations[1:])/2.
    
    start1 = time.time() 
    for i,src_dec in enumerate(declinations):
        if i%10==1:
	    print "Iteration", i, "of", num
        trials = llh.do_trials(src_ra=np.pi/2., src_dec=src_dec, mu=None, n_iter=args["niter"])
	plt.figure(i)
        n, bins, _ = plt.hist(trials["TS"], bins=20, normed=False)
	plt.semilogy(nonposy="clip")
	plt.savefig(("figures/TS_trials/logTS_{}"+identifier+extension).format(i))

    stop1 = time.time()
    mins, secs = divmod(stop1 - start1, 60)
    print("Trials finished after {0:2d}' {1:4.2f}''".format(int(mins), int(secs)))
    #~ pickle.dump(declinations, open("figures/TS_trials/declinations.pickle", "wb"))
    #~ ###################
#~ 
    #~ if "test" in args["add"].lower():
	#~ keys = hemispheres.keys()
	#~ keys.extend(['best', 'ra', 'dec', 'nsources', 'gamma', 'ra_inj', 'dec_inj', 'n_inj'])
        #~ print(keys)
        #~ print(best_hotspots[keys])
    #~ # Save the results
    #~ savepath = os.path.join(savepath, identifier)
    #~ utils.prepare_directory(savepath)
    #~ if jobID == 0:
        #~ utils.save_json_data(startup_dict, savepath, "startup_dict")
        #~ utils.save_json_data(trials_dict, savepath, "trials_dict")
        #~ utils.save_json_data(hemispheres.keys(), savepath, "hemispheres")
    #~ for i,hs in enumerate(best_hotspots):
        #~ hs = numpy.lib.recfunctions.append_fields(hs, 
                                                  #~ "energy", 
                                                  #~ data=energies,
                                                  #~ dtypes=np.float, 
                                                  #~ usemask=False)
        #~ np.savetxt(os.path.join(savepath,  "job"+str(jobID)+"_hotspots_"+str(i)+".txt"),
                   #~ hs,
                   #~ header=" ".join(hs.dtype.names),
                   #~ comments="")
    #~ print "Trials and Hotspots saved to:"
    #~ print savepath
    
