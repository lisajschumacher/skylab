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
    hemispheres = dict(North = np.radians([-5., 90.]), South = np.radians([-90., -5.]))
    hcolor = dict(North = "cyan", South = "magenta")

    plt = utils.plotting(backend=backend)

    nside_param = 4
    nside = 2**nside_param
    
    multi = True # work with multiple different samples
    # This sets whether or not we choose the template fit with fixed gamma
    fixed_gamma = True
    add_prior = True

    # "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data"
    # "/home/lschumacher/git_repos/general_code_repo/data"
    pg = UhecrPriorGenerator(nside_param, np.radians(6), 100, crpath)
    tm = np.exp(pg.template)
    tm = tm/tm.sum(axis=1)[np.newaxis].T
    src_gamma = 2.
    fit_gamma = 2.

    llh, injector = utils.startup(basepath,
                            inipath,
                            Nsrc=20,
                            fixed_gamma=fixed_gamma,
                            add_prior=add_prior,
                            src_gamma=src_gamma,
                            multi=multi,
                            n_uhecr=pg.n_uhecr,
                            prior=tm,
                            nside_param=nside_param
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
    for k in scan.dtype.names:
        scan[k] = hp.sphtfunc.smoothing(scan[k], sigma=np.radians(0.5))

    eps = 1.
    """
    if hasattr(plt.cm, "magma"):
        cmap = plt.cm.magma
    else:
        cmap = None # plt.cm.brg
    """
    # Custom colormap using cubehelix from seaborn, see utils
    cmap = utils.cmap

    '''
    if isinstance(llh, MultiPointSourceLLH):
        for llh in llh._sams.itervalues():
            ax.scatter(np.pi - llh.exp["ra"], np.arcsin(llh.exp["sinDec"]), 1,
                       marker="x",
                       #color=plt.gca()._get_lines.color_cycle.next(),
                       alpha=0.2)#, rasterized=True)
    else:
        ax.scatter(np.pi - llh.exp["ra"], np.arcsin(llh.exp["sinDec"]), 10,
                   marker="o",
                   #color=plt.gca()._get_lines.color_cycle.next(),
                   alpha=0.05)#, rasterized=True)
    #'''

    if not os.path.exists("figures"):
        os.makedirs("figures")

    what_to_plot = ["preTS", "postTS", "allPrior"]

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
        if True:
            for hi in hotspots:
                for h in hemispheres.keys():
                    ax.scatter(np.pi - hi[h]["best"]["ra"], hi[h]["best"]["dec"], 20,
                           marker="o",
                           color=hcolor[h],
                           alpha=0.25,
                           label="Hotspot fit")
            ax.scatter(np.pi - injector._src_ra, injector._src_dec, 20,
                       marker="x",
                       color=hcolor[h],
                       alpha=0.5,
                       label="Injected")
        fig.savefig("figures/skymap_" + key + extension, dpi=256)
        plt.close("all")
    for h in hemispheres.keys():
        print h
        best_ts_hotspots = np.array([hi[h]["best"]["TS"] for hi in hotspots])
        print best_ts_hotspots, best_ts_hotspots.sum()
