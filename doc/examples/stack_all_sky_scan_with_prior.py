# -*-coding:utf8-*-

import os
import logging
from socket import gethostname

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH
from skylab.prior_generator import UhecrPriorGenerator
from skylab.prior_injector import PriorInjector

# local
# sorry but I had to change the names
# this is too confusing for me and the interpreter :D
import ic_utils as utils


logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(logging.INFO)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(logging.INFO)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(logging.INFO)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(logging.INFO)
logging.getLogger("skylab.prior_injector.PriorInjector").setLevel(logging.INFO)

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

label = dict(TS=r"$\mathcal{TS}$",
             preTS=r"pre-prior $\mathcal{TS}$",
             postTS=r"post-prior $\mathcal{TS}$",
             prior=r"prior",
             allPrior=r"Combined prior",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )

basepath, inipath, savepath, crpath = utils.get_paths(gethostname())

if __name__=="__main__":
    
    backend = "svg"
    extension = "_testing_hemispheres.png"
    plt = utils.plotting(backend=backend)
    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    hcolor = dict(North = "cyan", South = "red")

    nside_param = 4
    nside = 2**nside_param
    
    multi = False # work with single or multiple different samples
    # LLH and Fitting parameters
    fixed_gamma = True
    add_prior = True
    src_gamma = 2.
    fit_gamma = 2.
    Nsrc = 0

    # Other stuff
    if "physik.rwth-aachen.de" in gethostname():
        ncpu = 4
    else:
        ncpu = 1
    burn = True
    inj_seed = 666
    save_res = True
    identifier = "test_"
    mark_hotspots = False

    # "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"
    # "/home/lschumacher/git_repos/general_code_repo/data"
    pg = UhecrPriorGenerator(nside_param, np.radians(6), 100, crpath)
    tm = np.exp(pg.template)
    tm = tm/tm.sum(axis=1)[np.newaxis].T


    llh, injector = utils.startup(basepath,
                                    inipath,
                                    Nsrc=Nsrc,
                                    fixed_gamma=fixed_gamma,
                                    add_prior=add_prior,
                                    src_gamma=src_gamma,
                                    multi=multi,
                                    n_uhecr=pg.n_uhecr,
                                    prior=tm,
                                    nside_param=nside_param,
                                    burn=burn,
                                    inj_seed=inj_seed,
                                    ncpu=ncpu
                                    )
    #~ print(llh)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspots) in enumerate(llh.all_sky_scan(
                                nside=nside, follow_up_factor=1,
                                pVal=pVal_func,
                                hemispheres=hemispheres,
                                prior=pg.template,
                                fit_gamma=fit_gamma)
                                ):

        if i > 0:
            # break after first follow up
            break
    #~ for k in scan.dtype.names:
        #~ scan[k] = hp.sphtfunc.smoothing(scan[k], sigma=np.radians(0.5))

    eps = 1.
    # Custom colormap using cubehelix from seaborn, see utils
    cmap = utils.cmap

    # Looking at the hotspots and separating them into North and South
    hk = hemispheres.keys()
    print "Hemisphere keys:", hk
    best_hotspots = np.zeros(pg.n_uhecr, dtype=[(p, np.float) for p in hk]
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
    if not os.path.exists("figures"):
        os.makedirs("figures")

    what_to_plot = ["preTS", "allPrior"] 

    for key in what_to_plot + llh.params:
        if fixed_gamma and key == "gamma": continue # skip gamma, if fixed
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
                ax.scatter(np.pi - bhi["ra"], bhi["dec"], 20,
                       marker="o",
                       color="cyan",
                       alpha=0.25,
                       label="Hotspot fit")
            if Nsrc>0:
                ax.scatter(np.pi - injector._src_ra, injector._src_dec, 20,
                           marker="d",
                           color="orange",
                           alpha=0.25,
                           label="Injected")
        fig.savefig("figures/skymap_" + key + extension, dpi=256)
        plt.close("all")
    # Now we look at the single results:
    if False:
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
                ax.scatter(np.pi - bhi["ra"], bhi["dec"], 20,
                       marker="o",
                       color="cyan",
                       alpha=0.25,
                       label="Hotspot fit")
            if Nsrc>0:
                ax.scatter(np.pi - injector._src_ra, injector._src_dec, 20,
                           marker="d",
                           color="orange",
                           alpha=0.25,
                           label="Injected")
        fig.savefig("figures/skymap_postTS_" + str(c) + extension, dpi=256)
        plt.close("all")
        c+=1
