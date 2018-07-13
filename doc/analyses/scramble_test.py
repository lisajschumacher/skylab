#!/bin/env python

# -*-coding:utf8-*-

from os.path import join
from time import time
from argparse import ArgumentParser
from yaml import load as yload

import numpy as np
import numpy.lib.recfunctions
import scipy
from healpy import npix2nside
import healpy as hp

import sys
from socket import gethostname
sys.path.append("/home/lschumacher/svn_repos/skylab/trunk/doc/analyses/lschumacher-UHECR/")


import ic_utils as utils
savepath, crpath, figurepath = utils.get_paths(gethostname())
print savepath
print figurepath


from prior_scan import ScrambledPriorScan
from prior_generator import UhecrSpatialPrior


print ":)"
  
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

pVal_func = None
ident = [
"mdparams",
"ecut"
]

full_args = {"nsideparam": 8,
             "crfiles": ["AugerUHECR2014.txt", "TelArrayUHECR-2017.txt"]
            }
test_args = {"nsideparam": 4,
             "niter": 2,
             "crfiles": ["AugerUHECR2014.txt", "TelArrayUHECR-2017.txt"]
            }

print("Loaded yaml args...:")

# get the parameter args and get a string for saving later
if not "test" in args["add"].lower():
    args.update(full_args)
    # Add time since epoch as additional unique label if we are not testing        
    identifier = args["add"]
    if identifier[-1]!="_": identifier+="_"
    for arg in ident:
        identifier+=arg+"-"+str(args[arg])+"_"
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


mdparams = args["mdparams"]
nsideparam = args["nsideparam"]
nside = 2**nsideparam
npix = hp.nside2npix(nside)

prepath = "/data/user/lschumacher/projects/stacking/hotspot_fitting/merged/signal_1531220158_ic-ant-scan_"
map_dtype = [('dec', '<f8'), ('theta', '<f8'), ('ra', '<f8'), ('TS', '<f8'), ('pVal', '<f8')]

wspot_dict = {70: {"TS": 1043, "mu": 2,
                     "filename": prepath+"1_mdparams-6_ecut-70_mu-2_srcgamma-2.19_shift-0_nsamples-2/job-0_warmspots-0.npy"
                     },
              85: {"TS": 314, "mu": 2,
                     "filename": prepath+"2_mdparams-6_ecut-85_mu-2_srcgamma-2.19_shift-0_nsamples-2/job-10_warmspots-0.npy"
                     },
              100: {"TS": 176, "mu": 4,
                     "filename": prepath+"3_mdparams-6_ecut-100_mu-4_srcgamma-2.19_shift-0_nsamples-2/job-0_warmspots-2.npy"
                     }
             }


prior = UhecrSpatialPrior(args["nsideparam"], 
                         np.radians(args["mdparams"]), 
                         args["ecut"], 
                         crpath, 
                         files=args["crfiles"]
                        )

filename = wspot_dict[args["ecut"]]["filename"]
warmspot = np.load(filename)
print "loaded warmspot information"
warm_spot_map = np.zeros(npix, dtype=map_dtype)
warm_spot_map["theta"], warm_spot_map["ra"] = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
warm_spot_map["dec"] = np.pi/2. - warm_spot_map["theta"]
for sp in warmspot:
    pix = hp.ang2pix(nside, sp["theta"], sp["phi"])
    warm_spot_map[pix]["TS"] = sp["TS"]
    warm_spot_map[pix]["pVal"] = sp["TS"]

print "instantiating a scrambler object"
wspot_scrambler = ScrambledPriorScan(warm_spot_map, prior, rs=np.random.RandomState(seed))
TS, original_TS_sum = wspot_scrambler.calculate_post_trial_pvalue(niter=args["niter"], verbose=True)
result = np.concatenate([[original_TS_sum], TS])
np.save(join(savepath, "job-{}_result_dict".format(jobID)), result)
