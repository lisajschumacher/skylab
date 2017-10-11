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

level=logging.INFO
logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(level)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(level)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(level)
logging.getLogger("skylab.prior_injector.PriorInjector").setLevel(level)

label = dict(TS=r"$\mathcal{TS}$",
             postTS=r"post-prior $\mathcal{TS}$",
             preTS=r"pre-prior $\mathcal{TS}$",
             allPrior=r"Prior",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )
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

    if args.nsamples<1 or args.nsamples>7:
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
        identifier = args.add
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
        
    basepath, inipath, savepath, crpath, figurepath = utils.get_paths(gethostname())
    print "Data will be saved to: ", savepath
    print "With Identifier: ", identifier

    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    nside = 2**args.nsideparam
    mark_hotspots = False
    mark_injected = True
    plot_single = False
    extension = "_multi_with_hotspots.png"
    
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 4
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
        num, mu, src_ra, src_dec = injector.sample(args.mu, poisson=True, position=True).next()

    scan_dict = dict(nside = nside,
                        follow_up_factor = args.followupfactor,
                        pVal = pVal_func,
                        fit_gamma = 2.)

    if mu is not None:
		llh._add_injection(mu)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspots) in enumerate(llh.all_sky_scan(hemispheres=hemispheres,
                                prior=log_tm,
                                **scan_dict)
                                ):
        if args.followupfactor == 0: break # In case you don't want a follow-up
        if i > 0:
            # break after first follow up
            break
    for k in scan.dtype.names:
        scan[k] = hp.sphtfunc.smoothing(scan[k], sigma=np.radians(0.5))

    eps = 1.
    # Custom colormap using cubehelix from seaborn, see utils
    cmap = utils.cmap
    plt = utils.plotting("svg")
    # Looking at the hotspots and separating them into North and South
    hk = hemispheres.keys()
    print "Hemisphere keys:", hk
    best_hotspots = np.zeros(pg.n_uhecr,
                             dtype=[(p, np.float) for p in hk]
                                                +[("best", np.float)]
                                                +[("dec", np.float)]
                                                +[("ra", np.float)]
                                                +[("nsources", np.float)])

    for i,hi in enumerate(hotspots):
        for h in hk:
            best_hotspots[h][i] = hi[h]["best"]["TS"]
        if best_hotspots[hk[0]][i] >= best_hotspots[hk[1]][i]:
            best_hotspots["best"][i] = best_hotspots[hk[0]][i]
            best_hotspots["ra"][i] = hi[hk[0]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[0]]["best"]["dec"]
            best_hotspots["nsources"][i] = hi[hk[0]]["best"]["nsources"]
        else:
            best_hotspots["best"][i] = best_hotspots[hk[1]][i]
            best_hotspots["ra"][i] = hi[hk[1]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[1]]["best"]["dec"]
            best_hotspots["nsources"][i] = hi[hk[1]]["best"]["nsources"]
        
    print "Hotspots:"
    print best_hotspots.dtype.names
    print best_hotspots

    # Plotting
    #if not os.path.exists("figures"):
    #    os.makedirs("figures")

    what_to_plot = ["preTS", "allPrior"] 

    for key in what_to_plot + llh.params:
        if key == "gamma": continue # skip gamma
        eps = 0.1 if key not in what_to_plot else 0.0
        vmin, vmax = np.percentile(scan[key], [eps, 100. - eps])
        vmin = np.floor(max(0, vmin))
        vmax = min(8, np.ceil(vmax))
        q = np.ma.masked_array(scan[key])
        q.mask = ~(scan["nsources"] > 0.5) if key not in what_to_plot else np.zeros_like(q, dtype=np.bool)
        fig, ax = utils.skymap(plt, q, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               colorbar=dict(title=label[key]),
                               rasterized=True)
        if mark_hotspots:
            for bhi in best_hotspots:
                ax.scatter(np.pi - bhi["ra"], bhi["dec"], 1200,
                            marker="d",
                            facecolor='none',
                            edgecolors='orange',
                            label="Hotspot fit")
        if mark_injected and args.mu>0:
            ax.scatter(np.pi - src_ra, src_dec, 1200,
                        marker="o",
                        facecolor='none',
                        edgecolors='cyan',
                        label="Injected")
        fig.savefig(figurepath+"/skymap_" + key + extension, dpi=256)
        plt.close("all")
    # Now we look at the single results:
    if plot_single:
        c=0
        key="postTS"
        for i,s in enumerate(llh.postTS):
            vmin, vmax = np.percentile(s, [0., 100.])
            vmin = np.floor(max(0, vmin))
            vmax = min(8, np.ceil(vmax))
            q = np.ma.masked_array(s)
            q.mask = np.zeros_like(q, dtype=np.bool)
            fig, ax = utils.skymap(plt, q, cmap=cmap,
                                   vmin=vmin, vmax=vmax,
                                   colorbar=dict(title=label[key]),
                                   rasterized=True)
            if mark_hotspots:
                for bhi in best_hotspots:
                    ax.scatter(np.pi - bhi["ra"], bhi["dec"], 1200,
                                marker="d",
                                facecolor='none',
                                edgecolors='orange',
                                label="Hotspot fit")
            if mark_injected and args.mu>0:
                ax.scatter(np.pi - src_ra, src_dec, 1200,
                            marker="o",
                            facecolor='none',
                            edgecolors='cyan',
                            label="Injected")
            fig.savefig(figurepath+"/skymap_postTS_" + str(c) + extension, dpi=256)
            plt.close("all")
            c+=1
    else:
        key="postTS"
        pTS = []
        for i,s in enumerate(llh.postTS):
            pTS.append(s)
        s=np.array(pTS).sum(axis=0)
        vmin, vmax = np.percentile(s, [0., 100.])
        vmin = np.floor(max(0, vmin))
        vmax = min(8, np.ceil(vmax))
        q = np.ma.masked_array(s)
        q.mask = np.zeros_like(q, dtype=np.bool)
        fig, ax = utils.skymap(plt, q, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               colorbar=dict(title=label[key]),
                               rasterized=True)
        if mark_hotspots:
            for bhi in best_hotspots:
                ax.scatter(np.pi - bhi["ra"], bhi["dec"], 1200,
                            marker="d",
                            facecolor='none',
                            edgecolors='orange',
                            label="Hotspot fit")
        if mark_injected and args.mu>0:
            ax.scatter(np.pi - src_ra, src_dec, 1200,
                        marker="o",
                        facecolor='none',
                        edgecolors='cyan',
                        label="Injected")
        fig.savefig(figurepath+"/skymap_postTS_full" + extension, dpi=256)
        plt.close("all")
        

            
