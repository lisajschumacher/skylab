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

label = dict(TS=r"$\mathcal{TS}$",
             postTS=r"post-prior $\mathcal{TS}$",
             preTS=r"pre-prior $\mathcal{TS}$",
             allPrior=r"Prior",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
            )
	    
pVal_func = None
ident = [
"nsamples",
"hecut",
"mdparams",
"ecut",
"mu",
"fixedgamma"
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
        
    basepath, inipath, savepath, crpath, figurepath = utils.get_paths(gethostname())
    print "Data will be saved to: ", savepath
    print "With Identifier: ", identifier

    if not args["hecut"]:
        hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
	sinDec_range = [-1,1]
    else:
        hemispheres = dict(North = np.radians([-5., 90.]))
	sinDec_range = [np.sin(hemispheres["North"][0]), 1]
    nside = 2**args["nsideparam"]
    mark_hotspots = True
    mark_injected = True
    markersize = 250
    plot_single = False
    
    
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 4
    else:
        ncpu = 1

    # Generate several templates for prior fitting
    # One for each deflection hypothesis each
    pg = UhecrPriorGenerator(args["nsideparam"])
    log_tm = pg.calc_template(np.radians(args["mdparams"]), pg._get_UHECR_positions(args["ecut"], crpath))
    temp = np.exp(log_tm)
    tm = temp/temp.sum(axis=1)[np.newaxis].T
    energies = pg.energy
    if len(log_tm)>1: extension = "_multi.png"
    if len(log_tm)==1: extension = "_single_2.png"
    
    startup_dict = dict(basepath = basepath,
                        inipath = inipath,
                        seed = seed,
                        Nsrc = args["mu"], ### Signal ###
                        fixed_gamma = args["fixedgamma"],
			gamma_range = args["gammarange"],
                        add_prior = True,
                        src_gamma = args["srcgamma"],
                        fit_gamma = args["fitgamma"],
                        multi = True if args["nsamples"]>1 else False,
                        n_uhecr = pg.n_uhecr,
                        nside_param = args["nsideparam"],
                        burn = args["burn"],
                        ncpu = ncpu,
                        n_samples = args["nsamples"],
                        he_cut = args["hecut"],
			sinDec_range = sinDec_range,
                        mode = "box")

    llh, injector = utils.startup(prior = tm, **startup_dict)
        
    if injector==None:
        mu = None
    else:
        num, mu, src_ra, src_dec = injector.sample(args["mu"],
	                                           poisson=True,
						   position=True
						  ).next()

    scan_dict = dict(nside = nside,
                     follow_up_factor = args["followupfactor"],
                     pVal = pVal_func,
                     fit_gamma = 2.
		    )

    if mu is not None:
	llh._add_injection(mu)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspots) in enumerate(llh.all_sky_scan(hemispheres=hemispheres,
                                prior=log_tm,
                                **scan_dict)
                                ):
        if args["followupfactor"] == 0: break # In case you don't want a follow-up
        if i > 0:
            # break after first follow up
            break
    for k in scan.dtype.names:
        scan[k] = hp.sphtfunc.smoothing(scan[k], sigma=np.radians(0.5))

    eps = 1.
    # Custom colormap using cubehelix from seaborn, see utils
    cmap = utils.cmap
    plt = utils.plotting("pdf")
    # Looking at the hotspots and separating them into North and South
    hk = hemispheres.keys()
    print "Hemisphere keys:", hk
    print "injected: ", num
    print "at position: (ra, dec) ", src_ra, src_dec
    best_hotspots = np.zeros(pg.n_uhecr,
                             dtype=[(p, np.float) for p in hk]
                                                +[("best", np.float)]
                                                +[("dec", np.float)]
                                                +[("ra", np.float)]
                                                +[("nsources", np.float)]
                                                +[("gamma", np.float)])

    for i,hi in enumerate(hotspots):
        for h in hk:
            best_hotspots[h][i] = hi[h]["best"]["TS"]
        if best_hotspots[hk[0]][i] >= best_hotspots[hk[1]][i]:
            best_hotspots["best"][i] = best_hotspots[hk[0]][i]
            best_hotspots["ra"][i] = hi[hk[0]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[0]]["best"]["dec"]
            best_hotspots["nsources"][i] = hi[hk[0]]["best"]["nsources"]
            best_hotspots["gamma"][i] = hi[hk[0]]["best"]["gamma"]
        else:
            best_hotspots["best"][i] = best_hotspots[hk[1]][i]
            best_hotspots["ra"][i] = hi[hk[1]]["best"]["ra"]
            best_hotspots["dec"][i] = hi[hk[1]]["best"]["dec"]
            best_hotspots["nsources"][i] = hi[hk[1]]["best"]["nsources"]
            best_hotspots["gamma"][i] = hi[hk[1]]["best"]["gamma"]
        
    print "Hotspots:"
    print best_hotspots.dtype.names
    print best_hotspots

    # Plotting
    #if not os.path.exists("figures"):
    #    os.makedirs("figures")

    what_to_plot = ["preTS", "allPrior"] 

    for key in what_to_plot + llh.params:
        if key=="gamma" and args["fixedgamma"]==True: continue # skip gamma if fixed (boring!)
        eps = 0.1 if key not in what_to_plot else 0.0
        vmin, vmax = np.percentile(scan[key], [eps, 100. - eps])
        vmin = np.floor(max(0, vmin))
        vmax = min(30, np.ceil(vmax))
	if key=="gamma": vmin, vmax = args["gammarange"]
        q = np.ma.masked_array(scan[key])
        q.mask = ~(scan["nsources"] > 0.5) if key not in what_to_plot else np.zeros_like(q, dtype=np.bool)
        fig, ax = utils.skymap(plt, q, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               colorbar=dict(title=label[key]),
                               rasterized=True)
			       
	fig.savefig(figurepath+"/skymap_" + key + extension, dpi=256)
	if args["mu"]>0: #mark_injected and
            ax.scatter(np.pi - src_ra, src_dec, markersize,
                        marker="o",
                        facecolor='none',
                        edgecolors='cyan',
			alpha=0.5,
                        label="Injected")
	    fig.savefig(figurepath+"/skymap_inj_" + key + extension, dpi=256)
	    
        if mark_hotspots:
            for bhi in best_hotspots:
                ax.scatter(np.pi - bhi["ra"], bhi["dec"], markersize,
                            marker="h",
                            facecolor='none',
                            edgecolors='orange',
			    alpha=0.5,
                            label="Hotspot fit")
	    fig.savefig(figurepath+"/skymap_hsp_" + key + extension, dpi=256)
	
        
        plt.close("all")
    # Now we look at the single results:
    #if plot_single:
    c=0
    key="postTS"
    for c,s in enumerate(llh.postTS):
	vmin, vmax = np.percentile(s, [0., 100.])
	vmin = np.floor(max(0, vmin))
	vmax = min(30, np.ceil(vmax))
	q = np.ma.masked_array(s)
	q.mask = np.zeros_like(q, dtype=np.bool)
	fig, ax = utils.skymap(plt, q, cmap=cmap,
			       vmin=vmin, vmax=vmax,
			       colorbar=dict(title=label[key]),
			       rasterized=True)
			       
	fig.savefig(figurepath+"/skymap_" + str(c) + key + extension, dpi=256)
	if args["mu"]>0:
	    ax.scatter(np.pi - src_ra[c], src_dec[c], markersize,
			marker="o",
			facecolor='none',
			edgecolors='cyan',
			alpha=0.5,
			label="Injected")
	    fig.savefig(figurepath+"/skymap_inj_" + str(c) + extension, dpi=256)
	if mark_hotspots:
	    #for bhi in best_hotspots:
	    ax.scatter(np.pi - best_hotspots[c]["ra"], best_hotspots[c]["dec"], markersize,
			marker="h",
			facecolor='none',
			edgecolors='orange',
			alpha=0.5,
			label="Hotspot fit")
	    fig.savefig(figurepath+"/skymap_hsp_" + str(c) + key + extension, dpi=256)
	plt.close("all")
	#c+=1
    #else:
    key="postTS"
    pTS = []
    for i,s in enumerate(llh.postTS):
	pTS.append(s)
    s=np.array(pTS).sum(axis=0)
    vmin, vmax = np.percentile(s, [0., 100.])
    vmin = np.floor(max(0, vmin))
    vmax = min(30, np.ceil(vmax))
    q = np.ma.masked_array(s)
    q.mask = np.zeros_like(q, dtype=np.bool)
    fig, ax = utils.skymap(plt, q, cmap=cmap,
			   vmin=vmin, vmax=vmax,
			   colorbar=dict(title=label[key]),
			   rasterized=True)
    fig.savefig(figurepath+"/skymap_postTS_full" + extension, dpi=256)
    if args["mu"]>0:
	ax.scatter(np.pi - src_ra, src_dec, markersize,
		    marker="o",
		    facecolor='none',
		    edgecolors='cyan',
		    alpha=0.5,
		    label="Injected")
	fig.savefig(figurepath+"/skymap_postTS_full_inj" + extension, dpi=256)
    if mark_hotspots:
	for bhi in best_hotspots:
	    ax.scatter(np.pi - bhi["ra"], bhi["dec"], markersize,
			marker="h",
			facecolor='none',
			edgecolors='orange',
			alpha=0.5,
			label="Hotspot fit")
	fig.savefig(figurepath+"/skymap_postTS_full_hsp" + extension, dpi=256)
    plt.close("all")
