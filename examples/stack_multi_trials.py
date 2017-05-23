#!/bin/env python
from __future__ import print_function

import sys, os
import socket
import time
import numpy as np

from argparse import ArgumentParser

from skylab.psLLH import MultiPointSourceLLH, PointSourceLLH
from skylab.ps_model import ExtendedLLH, EnergyLLH
from skylab.ps_injector import UHECRSourceInjector
from skylab.utils import rotate

from my_functions import load_data, prepare_directory

_i=10
_job=0
_et=85
_det=["IC86", "IC79", "IC59", "IC40"]
_md=6
_mu=0
_gamma=2.

parser = ArgumentParser()

parser.add_argument("--job", 
                    dest="job", 
                    type=int, 
                    default=_job
                    )

parser.add_argument("--i", 
                    dest="i", 
                    type=int, 
                    default=_i, 
                    help="number of trial iterations"
                   )

parser.add_argument("--det", 
                    dest="det",
                    nargs="+",
                    default=_det, 
                    help="Detector configuration, variation of [IC40, IC59, IC79, IC86]"
                   )

parser.add_argument("--md", 
                    dest="md", 
                    type=int, 
                    default=_md, 
                    help="Magnetic deflection parameter in degree, 3-9"
                   )

parser.add_argument("--et", 
                    dest="et", 
                    type=int, 
                    default=_et, 
                    help="Energy threshold for UHECRs in EeV, 60-120"
                   )

parser.add_argument("--mu", 
                    dest="mu", 
                    type=float, 
                    default=_mu, 
                    help="Mean number of signal events per source, for injection"
                   )

parser.add_argument("--add", 
                    dest="add", 
                    type=str, 
                    default="", 
                    help="Additional string for saving path"
                   )

args = parser.parse_args()

# Check parameters
correct = True
for det in args.det:
	if not (det in ["IC40", "IC59", "IC79", "IC86"]):
		print("detector {} not in configuration".format(det))
		correct = False
if not correct:
	print("Chose {} instead".format(_det))
	args.det = _det
	
if args.et>150 or args.et<50:
     print("energy threshold {} not between 50 and 150, chose {} EeV instead".format(args.et, _et))
     args.et = _et

if args.md<0:
	print("magnetic deflection {} below zero, chose {} deg instead".format(args.md, _md))
	args.md = _md

if args.mu<0:
	print("Mu-per-source {} below zero, chose {} instead".format(args.mu, _mu))
	args.mu = _mu

if args.i<1:
	print("number of iterations {} too small, chose {} instead".format(args.i, _i))
	args.i = _i

if args.job<0:
	print("jobID {} too small, chose {} instead".format(args.job, _job))
	args.job = _job

# get the parameter args and get a string for saving later
add_save=args.add
if add_save[-1]!="_": add_save+="_"
for arg in vars(args):
	if arg!="job" and arg!="add":
		if arg!="det":
			add_save+=arg+str(getattr(args, arg))+"_"
		else:
			add_save+=arg
			for a in getattr(args,arg):
				add_save+=a+"-"
			if add_save[-1]=="-": add_save=add_save[:-1]+"_"
if add_save[-1]=="_": add_save=add_save[:-1] #remove last underscore


# Read/save data from/to ...
if "physik.rwth-aachen.de" in socket.gethostname():
    if "lx3b74" in socket.gethostname():
        # Local scratch
        basepath="/user/scratch/lschumacher/ps_data"
        inipath="/user/scratch/lschumacher/ps_data"
    else:
	basepath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub"
	inipath="/net/scratch_icecube4/user/lschumacher/projects/data/ps_sample/coenders_pub"
    savepath = "/net/scratch_icecube4/user/lschumacher/projects/stacking"
elif "M16" in socket.gethostname():
    raise("Not up to date! Probably... go check it first")
    savepath = "/home/icecube/Desktop/pScratch4/lschumacher/projects/stacking"
    basepath="/home/icecube/Desktop/pScratch4/lschumacher/projects/data/ps_sample/coenders_pub"
    inipath="/home/icecube/Desktop/pScratch4/lschumacher/projects/data/ps_sample/coenders_pub"
elif "icecube.wisc.edu" in socket.gethostname():
    basepath = "/data/user/coenders/data/MultiYearPointSource/npz"
    inipath = "/data/user/lschumacher/config_files_ps"
    savepath = "/data/user/lschumacher/projects/stacking"
    crpath = "/home/lschumacher/git_repos/general_code_repo/data"
else:
    raise("Unknown Host")
    
savepath = os.path.join(savepath, add_save)
print("Parameters: {}".format(add_save), "Starting!")
start = time.time()

#Initialize MultiLLH and dicts
multi_llh = MultiPointSourceLLH()
multi_llh.reset()
mcdict = dict()
ltdict = dict()
start1 = time.time()
for key,det in enumerate(args.det):	
	# Load data
	mc, exp, livetime = load_data(basepath, inipath, det)
	mc = np.rec.array(mc)
	exp = np.rec.array(exp)	
	
	# Initialize PS LLH
	print("LLH setup, detector ",det)
	ps_llh = PointSourceLLH(exp, mc, livetime, llh_model=ExtendedLLH(), mode="all") #, seed=1)
	multi_llh.add_sample(det, ps_llh)
	if args.mu>0:
		mcdict[key]=mc
		ltdict[key]=livetime
stop1 = time.time()
mins, secs = divmod(stop1 - start1, 60)
print("Setup finished after {0:2d}' {1:4.2f}''".format(int(mins), int(secs)))
print(crpath)
injector = UHECRSourceInjector(2., np.radians(args.md), e_thresh=args.et, path=crpath)

if args.mu>0:
	injector.fill(mcdict, ltdict)
	sampler = injector.sample(args.mu*len(injector.uhecr_dec))
	trials = multi_llh.do_trials(injector.uhecr_ra, injector.uhecr_dec, src_sigma=injector.uhecr_sigma, n_iter=args.i, mu=sampler)
else:
	trials = multi_llh.do_trials(injector.uhecr_ra, injector.uhecr_dec, src_sigma=injector.uhecr_sigma, n_iter=args.i)

# Save stuff
header=""
for i in trials.dtype.names:
	header+=i+" "
		 
savestring = os.path.join(savepath, "trials_job{}".format(args.job))
prepare_directory(os.path.join(savepath))
np.savetxt(	savestring,
						trials, 
						header=header,
						comments=""
					)
os.chmod(savestring, 0754)
stop = time.time()
mins, secs = divmod(stop - start, 60)
print("{} trials saved to {}".format(len(trials), savestring))
print("Finished after {0:2d}' {1:4.2f}''".format(int(mins), int(secs)))


