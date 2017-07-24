# -*-coding:utf8-*-

import os
import logging

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH
from skylab.prior_generator import UhecrPriorGenerator

# local
import utils

logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(logging.INFO)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(logging.INFO)
logging.getLogger("skylab.stacking_priorllh.StackingPriorLLH").setLevel(logging.INFO)
logging.getLogger("skylab.stacking_priorllh.MultiStackingPriorLLH").setLevel(logging.INFO)

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

if __name__=="__main__":
    
    backend = "svg"
    extension = "_multi_sample_multi_prior.png"    
    plt = utils.plotting(backend=backend)

    nside_param = 4
    nside = 2**nside_param
    multi = True # work with multiple different samples
    # This sets whether or not we choose the template fit with fixed gamma
    fixed_gamma = True
    add_prior = True
    pg = UhecrPriorGenerator(nside_param, np.radians(6), 125, "/home/home2/institut_3b/lschumacher/phd_stuff/phd_code_git/data", multi=True)
    prior = pg.template
    fit_gamma = 2.
    # Source parameters for injection
    nUHECRs = 5
    src_dec = np.arcsin(np.random.uniform(-1,1,nUHECRs))
    src_ra = np.random.uniform(0., np.pi*2., nUHECRs)
    src_sigma = np.random.uniform(0.8, 1.2, nUHECRs) * np.radians(6.)
    src_gamma = 2.
    # For prior
    shift = 0.5
    pdec = src_dec + shift * src_sigma
    pra = src_ra + shift * src_sigma

    print "Important parameters: "
    print "fixed_gamma is ", fixed_gamma
    print "add_prior is ", add_prior
    
    #~ llh, mc = utils.startup(Nsrc=5, fixed_gamma=fixed_gamma, add_prior=add_prior,
                            #~ gamma_inj=src_gamma, mulit=multi,
                            #~ src_dec=src_dec, src_ra=src_ra
                            #~ )
    llh, mc = utils.startup(multi=True, fixed_gamma=fixed_gamma, add_prior=add_prior)                            
    print(llh)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspots) in enumerate(llh.all_sky_scan(
                                nside=nside, follow_up_factor=1,
                                pVal=pVal_func,
                                hemispheres=dict(Full=np.radians([-90., 90.])),
                                prior=prior,
                                pdec=pdec,
                                pra=pra,
                                psig=src_sigma,
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
        if False:
            for hi in hotspots:
                ax.scatter(np.pi - hi["Full"]["best"]["ra"], hi["Full"]["best"]["dec"], 10,
                       marker="x",
                       color='red',
                       alpha=0.85,
                       label="Hotspot fit")
        fig.savefig("figures/skymap_" + key + extension, dpi=256)
        plt.close("all")
    best_ts_hotspots = np.array([hi["Full"]["best"]["TS"] for hi in hotspots])
    print best_ts_hotspots, best_ts_hotspots.sum()
