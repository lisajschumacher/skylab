#!/bin/env python

# -*-coding:utf8-*-

import os
import logging
import time
from socket import gethostname
from argparse import ArgumentParser
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
_niter = 10
_job = 0
_nsideparam = 6
_followupfactor = 2
_burn = False
_nsamples = 7
_mdparams = [6.]
_ecut = 70
_mu = 3.0

parser = ArgumentParser()

parser.add_argument("--job", 
                    dest="job", 
                    type=int, 
                    default=_job
                    )

parser.add_argument("--niter", 
                    dest="niter", 
                    type=int, 
                    default=_niter,
                    help="Number of trial iterations"
                   )

parser.add_argument("--mu", 
                    dest="mu", 
                    type=float, 
                    default=_mu,
                    help="Mean number of neutrinos per source, 0 will lead to background trials"
                   )

parser.add_argument("--nsideparam", 
                    dest="nsideparam", 
                    type=int, 
                    default=_nsideparam, 
                    help="Nside parameter for HealPy Maps, nside=2**nsideparam"
                   )

parser.add_argument("--ff", 
                    dest="followupfactor", 
                    type=int, 
                    default=_followupfactor, 
                    help="Follow-up factor for second iteration of sky scan"
                   )

parser.add_argument("--nsamples", 
                    dest="nsamples", 
                    type=int, 
                    default=_nsamples, 
                    help="Number of samples, from IC40 to latest IC86"
                   )

parser.add_argument("--burn", 
                    dest="burn", 
                    action="store_true" 
                   )
parser.add_argument("--full", 
                    dest="burn", 
                    action="store_false" 
                   )
parser.set_defaults(burn=_burn)

parser.add_argument("--add", 
                    dest="add", 
                    type=str, 
                    default="", 
                    help="Additional string for saving path"
                   )

parser.add_argument("--ecut", 
                    dest="ecut", 
                    type=int, 
                    default=_ecut, 
                    help="Cut on UHECR energy in EeV"
                   )

parser.add_argument("--mdparams", 
                    dest="mdparams",
                    type=float,
                    default=_mdparams, 
                    help="Magnetic deflection parameter, should range between 3 and 9 degree"
                   )

if __name__=="__main__":

    args = parser.parse_args()

    if args.niter<1:
        print("number of iterations {} too small, chose {} instead".format(args.niter, _niter))
        args.niter = _niter

    if args.job<0:
        print("jobID {} too small, chose {} instead".format(args.job, _job))
        args.job = _job

    if args.nsamples<=1 or args.nsamples>7:
        print("Number of samples {} not in correct range, chose {} instead".format(args.nsamples, _nsamples))
        args.nsamples = _nsamples

    if args.nsideparam>7 or args.nsideparam<3:
        print("nsideparam {} not in correct range, chose {} instead".format(args.nsideparam, _nsideparam))
        args.nsideparam = _nsideparam

    if args.followupfactor<0 or args.followupfactor>3:
        print("follow_up_factor {} not in correct range, chose {} instead".format(args.followupfactor, _followupfactor))
        args.followupfactor = _followupfactor

    if args.ecut<0 or args.ecut>180:
        print("ecut {} not in correct range, chose {} instead".format(args.ecut, _ecut))
        args.ecut = _ecut

    # get the parameter args and get a string for saving later
    if not "test" in args.add.lower():
        # Add time since epoch as additional unique label if we are not testing        
        identifier = str(int(time.time()))+"_"
        identifier += args.add
        if identifier[-1]!="_": identifier+="_"
        for arg in vars(args):
            if arg!="job" and arg!="add":
                identifier+=arg+str(getattr(args, arg))+"_"
        # Remove last underscore
        if identifier[-1]=="_": identifier=identifier[:-1]
        seed = np.random.randint(2**32-args.job)
    else:
        identifier = args.add
        seed = args.job
        
    basepath, inipath, savepath, crpath, _ = utils.get_paths(gethostname())
    print "Data will be saved to: ", savepath
    print "With Identifier: ", identifier

    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    nside = 2**args.nsideparam

    # Other stuff
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 1
    else:
        ncpu = 1

    # Generate several templates for prior fitting
    # One for each deflection hypothesis each
    pg = UhecrPriorGenerator(args.nsideparam)
    log_tm = pg.calc_template(np.radians(args.mdparams), pg._get_UHECR_positions(args.ecut, crpath))
    temp = np.exp(log_tm)
    tm = temp/temp.sum(axis=1)[np.newaxis].T
    energies = pg.energy
    
    startup_dict = dict(basepath = basepath,
                        inipath = inipath,
                        seed = seed,
                        Nsrc = args.mu, ### Signal ###
                        fixed_gamma = True,
                        add_prior = True,
                        src_gamma = 2.,
                        fit_gamma = 2.,
                        multi = True if args.nsamples>1 else False,
                        n_uhecr = pg.n_uhecr,
                        nside_param = args.nsideparam,
                        burn = args.burn,
                        ncpu = ncpu,
                        n_samples = args.nsamples,
                        mode = "box")

    llh, injector = utils.startup(prior = tm, **startup_dict)

    if injector==None:
        mu = None
    else:
        mu = injector.sample(args.mu, poisson=True)

    trials_dict = dict(n_iter = args.niter, 
                        nside = nside,
                        follow_up_factor = args.followupfactor,
                        pVal = pVal_func,
                        fit_gamma = 2.)
    start1 = time.time() 
    best_hotspots = llh.do_trials(prior = log_tm,
                        hemispheres = hemispheres,
                        mu = mu, 
                        **trials_dict)
    stop1 = time.time()

    mins, secs = divmod(stop1 - start1, 60)
    hours, mins = divmod(mins, 60)
    print("Full scan finished after {2:2d}h {0:2d}m {1:2d}s".format(int(mins), int(secs), int(hours)))
    print(best_hotspots)
    # Save the results
    savepath = os.path.join(savepath, identifier)
    utils.prepare_directory(savepath)
    if args.job == 0:
        utils.save_json_data(startup_dict, savepath, "startup_dict")
        utils.save_json_data(trials_dict, savepath, "trials_dict")
        utils.save_json_data(hemispheres.keys(), savepath, "hemispheres")
    for i,hs in enumerate(best_hotspots):
        hs = numpy.lib.recfunctions.append_fields(hs, 
                                                "energy", 
                                                data=energies,
                                                dtypes=np.float, 
                                                usemask=False)
        np.savetxt(os.path.join(savepath,  "job"+str(args.job)+"_hotspots_"+str(i)+".txt"),
                   hs,
                   header=" ".join(hs.dtype.names),
                   comments="")
    print "Trials and Hotspots saved to:"
    print savepath
    