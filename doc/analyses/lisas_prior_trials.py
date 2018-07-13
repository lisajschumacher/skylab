#!/bin/env python

# -*-coding:utf8-*-

from os.path import join
import logging
from time import time
import datetime
from yaml import load as yload

# scipy
import numpy as np
import numpy.lib.recfunctions
from argparse import ArgumentParser

# skylab
#from skylab.priors import SpatialPrior
from skylab.test_statistics import TestStatisticNotSeparated

# additional personal imports
import ic_utils as utils
#from prior_generator import UhecrPriorGenerator
from prior_generator import UhecrSpatialPrior

# stuff to identify user
import getpass
from socket import gethostname
username = getpass.getuser()


import healpy as hp

# plotting
import seaborn as sns
plt = sns.mpl.pyplot

possible_filenames = ['IC40',
                      'IC59',
                      'IC79',
                      'IC86, 2011',
                      'IC86, 2012-2014',
                      #'IC86, 2012',
                      #'IC86, 2013',
                      #'IC86, 2014',
                      'ANTARES'
                     ]
  
parser = ArgumentParser()

parser.add_argument("yaml_file", 
                    type=str,  
                    help="yaml file for setting parameters"
                   )
parser.add_argument("job", 
                    type=int,  
                    help="Job ID"
                   )


# parse the yaml file
inputs = parser.parse_args()
yaml_file = inputs.yaml_file
jobID = inputs.job
args = yload(file(yaml_file))
#print("Loaded yaml args...:")
#print(args)

pVal_func = None
ident = [
"mdparams",
"ecut",
"mu",
"srcgamma",
"shift"
]

test_args = {"nsideparam": 4,
             "followupfactor": 0,
             "ecut": 0,
             "crfiles": ["Auger_test.txt"],
             "bunchsize": 10,
             "burn": True}
full_args = {"nsideparam": 6,
             "followupfactor": 2,
             "crfiles": ["AugerUHECR2014.txt", "TelArrayUHECR-2017.txt"],
             "bunchsize": 10,
             "burn": False}

#yaml_file = "/home/lschumacher/svn_repos/skylab/trunk/doc/analyses/lschumacher-UHECR/trial_test.yaml"
#jobID = 1

#args = yload(file(yaml_file))
print("Loaded yaml args...:")

if args["sample_names"]=="all":
    args["sample_names"] = possible_filenames
    
# get the parameter args and get a string for saving later
if not "test" in args["add"].lower():
    args.update(full_args)
    # Add time since epoch as additional unique label if we are not testing        
    identifier = args["add"]
    if identifier[-1]!="_": identifier+="_"
    for arg in ident:
        identifier+=arg+"-"+str(args[arg])+"_"
    identifier+="nsamples-"+str(len(args["sample_names"]))+"_"
    # Remove last underscore
    if identifier[-1]=="_": identifier=identifier[:-1]
    seed = np.random.randint(2**32-jobID)
else:
    args.update(test_args)
    identifier = args["add"]
    seed = jobID

savepath, crpath, figurepath = utils.get_paths(gethostname())
savepath = join(savepath, identifier)
utils.prepare_directory(savepath)

print "Data will be saved to: ", savepath
print "With Identifier: ", identifier

sinDec_range =  [-1., 1.]
hemispheres = dict(Full = [-np.pi/2., np.pi/2.]) 
nside = 2**args["nsideparam"]
print(args)


# Generate several templates for prior fitting
# One for each deflection hypothesis each
#pgen = UhecrPriorGenerator(args["nsideparam"])
#p_template = pgen.calc_template(np.radians(args["mdparams"]), args["ecut"], crpath, files=args["crfiles"])
prior = UhecrSpatialPrior(args["nsideparam"], 
                     np.radians(args["mdparams"]), 
                     args["ecut"], 
                     crpath, 
                     files=args["crfiles"]
                    )


## I want to be able to shift the injector template
## in order to test systematically shifted priors
## shift is given in sigma of the prior size
if args["shift"]!=0:
    prior_signal = UhecrSpatialPrior(args["nsideparam"], 
                                np.radians(args["mdparams"]), 
                                args["ecut"], 
                                crpath, 
                                files=args["crfiles"],
                                shift=args["shift"]
                               )
else:
    prior_signal = prior
                                
# free some memory
#del p_template
#del pgen

print("\n Setup:")
t0 = time()
startup_dict = dict(sample_names = args["sample_names"],
                    seed = seed,
                    Nsrc = args["mu"], 
                    gamma_range = args["gammarange"],
                    src_gamma = args["srcgamma"],
                    burn = args["burn"],
                    ncpu=1,
                    mode = "box",
                    bunchsize = args["bunchsize"]
                   )

llh, injector = utils.startup(prior = prior_signal, 
                              teststatistic = TestStatisticNotSeparated(),
                              **startup_dict
                             )
print("  - took %.2f sec" % (time()-t0))

trials_dict = dict(n_iter = args["niter"], 
                   nside = nside,
                   follow_up_factor = args["followupfactor"],
                   pVal = pVal_func,
                   return_only_grid = False
                  )
if injector is not None:
    trials_dict["mean_signal"] = args["mu"]*len(prior.p)
    trials_dict["return_position"] = True
    trials_dict["poisson"] = ("test" not in args["add"].lower())
if injector is not None:
    trials_dict["phi0"] = injector.mu2flux(args["mu"])
if ("test" in args["add"].lower()) or (jobID==0):
    utils.save_json_data(startup_dict, savepath, "startup_dict")
    utils.save_json_data(args, savepath, "args_dict")
    utils.save_json_data(trials_dict, savepath, "trials_dict")
    
trials_dict.pop("phi0", None)

start1 = time()
print("1st Scan started at", datetime.datetime.now().strftime("%y-%m-%d %H:%M"))
for i,(scan, hotspot) in enumerate(
    llh.do_allsky_trials(injector=injector,
                         hemispheres=hemispheres,
                         spatial_prior=prior,
                         **trials_dict                         
                        )):
    for hs in hotspot:
        pri = int(hs.split("_")[-1])
        if "inj" in hotspot[hs]:
            hotspot[hs]["inj"]["cr_e"] = prior.energy[pri]
        else:
            hotspot[hs]["cr_e"] = prior.energy[pri]
    utils.save_json_data(hotspot, savepath, "job-{}_hotspot-{}".format(jobID, i))
    spots = utils.get_spots(scan["TS"], cutoff_TS=1)
    np.save(join(savepath, "job-{}_warmspots-{}.npy".format(jobID, i)), spots)

if injector is not None:
    trials_dict["phi0"] = injector.mu2flux(args["mu"])
if ("test" in args["add"].lower()) or (jobID==0):
    utils.save_json_data(startup_dict, savepath, "startup_dict")
    utils.save_json_data(args, savepath, "args_dict")
    utils.save_json_data(trials_dict, savepath, "trials_dict")
    #np.save(join(savepath, "job-{}_scan-{}.npy".format(jobID, i)), scan)
    
print "Trials and Hotspots saved to:"
print savepath
stop1 = time()

mins, secs = divmod(stop1 - start1, 60)
hours, mins = divmod(mins, 60)
print("Full scan finished after {2:2d}h {0:2d}m {1:2d}s".format(int(mins), int(secs), int(hours)))

