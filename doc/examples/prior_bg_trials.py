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
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(level)
logging.getLogger("skylab.prior_injector.PriorInjector").setLevel(level)

pVal_func = None
_n_iter = 10
_job = 0
_nside_param = 6
_follow_up_factor = 2
_burn = False
_n_samples = 7


parser = ArgumentParser()

parser.add_argument("--job", 
                    dest="job", 
                    type=int, 
                    default=_job
                    )

parser.add_argument("--niter", 
                    dest="n_iter", 
                    type=int, 
                    default=_n_iter, 
                    help="Number of trial iterations"
                   )

parser.add_argument("--nsideparam", 
                    dest="nside_param", 
                    type=int, 
                    default=_nside_param, 
                    help="Nside parameter for HealPy Maps, nside=2**nside_param"
                   )

parser.add_argument("--ff", 
                    dest="follow_up_factor", 
                    type=int, 
                    default=_follow_up_factor, 
                    help="Follow-up factor for second iteration of sky scan"
                   )

parser.add_argument("--nsamples", 
                    dest="n_samples", 
                    type=int, 
                    default=_n_samples, 
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

if __name__=="__main__":

    args = parser.parse_args()

    if args.n_iter<1:
        print("number of iterations {} too small, chose {} instead".format(args.n_iter, _n_iter))
        args.n_iter = _n_iter

    if args.job<0:
        print("jobID {} too small, chose {} instead".format(args.job, _job))
        args.job = _job

    if args.n_samples<=1 or args.n_samples>7:
        print("Number of samples {} not in correct range, chose {} instead".format(args.n_samples, _n_samples))
        args.n_samples = _n_samples

    if args.nside_param>7 or args.nside_param<3:
        print("nside_param {} not in correct range, chose {} instead".format(args.nside_param, _nside_param))
        args.nside_param = _nside_param

    if args.follow_up_factor<0 or args.follow_up_factor>3:
        print("follow_up_factor {} not in correct range, chose {} instead".format(args.follow_up_factor, _follow_up_factor))
        args.follow_up_factor = _follow_up_factor

    # get the parameter args and get a string for saving later
    identifier=args.add
    if identifier[-1]!="_": identifier+="_"
    for arg in vars(args):
        if arg!="job" and arg!="add":
            identifier+=arg+str(getattr(args, arg))+"_"
    if identifier[-1]=="_": identifier=identifier[:-1] #remove last underscore

    basepath, inipath, savepath, crpath = utils.get_paths(gethostname())
    print "Data will be saved to: ", savepath
    print "With Identifier: ", identifier

    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    nside = 2**args.nside_param

    # Other stuff
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 4
    else:
        ncpu = 1

    # Generate several templates for prior fitting
    # One for each deflection hypothesis each
    md_params = [3., 6.]
    pg = UhecrPriorGenerator(args.nside_param)
    log_tm = []
    tm = []
    for md in md_params:
        temp = pg.calc_template(np.radians(md), pg._get_UHECR_positions(70, crpath))
        log_tm.extend(temp)
        temp = np.exp(temp)
        tm.extend(temp/temp.sum(axis=1)[np.newaxis].T)
    log_tm = np.array(log_tm)
    tm = np.array(tm)
    energies = pg.energy
    
    startup_dict = dict(basepath = basepath,
                        inipath = inipath,
                        Nsrc = 0, ### Background ###
                        fixed_gamma = True,
                        add_prior = True,
                        src_gamma = 2.,
                        fit_gamma = 2.,
                        multi = True if args.n_samples>1 else False,
                        n_uhecr = pg.n_uhecr,
                        # prior = tm1, # not needed for Background
                        nside_param = args.nside_param,
                        burn = args.burn,
                        ncpu = ncpu,
                        n_samples = args.n_samples,
                        mode = "box")

    llh, injector = utils.startup(**startup_dict)

    if injector==None:
        mu = None
    else:
        mu = injector.sample(Nsrc, poisson=True)

    trials_dict = dict(n_iter = args.n_iter, 
                        mu = mu,  
                        nside = nside,
                        follow_up_factor = args.follow_up_factor,
                        pVal = pVal_func,
                        fit_gamma = 2.)
    start1 = time.time() 
    best_hotspots = llh.do_trials(prior = log_tm,
                        hemispheres = hemispheres,
                        **trials_dict)
    stop1 = time.time()

    mins, secs = divmod(stop1 - start1, 60)
    hours, mins = divmod(mins, 60)
    print("Full scan finished after {2:2d}h {0:2d}m {1:2d}s".format(int(mins), int(secs), int(hours)))

    # Save the results
    savepath = os.path.join(savepath, identifier)
    utils.prepare_directory(savepath)
    utils.save_json_data(startup_dict, savepath, "startup_dict")
    utils.save_json_data(trials_dict, savepath, "trials_dict")
    utils.save_json_data(hemispheres.keys(), savepath, "hemispheres")
    for i,hs in enumerate(best_hotspots):
        hs = numpy.lib.recfunctions.append_fields(hs, 
            ["energy", "deflection"], 
            data=[np.tile(energies, len(md_params)), 
                  np.repeat(md_params, pg.n_uhecr)],
            dtypes=[np.float, np.float], 
            usemask=False)
        np.savetxt(os.path.join(savepath, "hotspots_"+str(i)+".txt"),
                   hs,
                   header=" ".join(hs.dtype.names),
                   comments="")
    print "Trials and Hotspots saved to:"
    print savepath
    
