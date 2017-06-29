# -*-coding:utf8-*-

import os
import logging

# scipy
from scipy.stats import chi2
import healpy as hp
import numpy as np

# skylab
from skylab.psLLH import MultiPointSourceLLH

# local
import utils

logging.getLogger("skylab.psLLH.PointSourceLLH").setLevel(logging.INFO)
logging.getLogger("skylab.priorllh.PriorLLH").setLevel(logging.INFO)

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

label = dict(TS=r"$\mathcal{TS}$",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )


if __name__=="__main__":

    plt = utils.plotting(backend="pdf")
    nside = 2**4
    # This sets whether or not we choose the template fit with fixed gamma
    fixed_gamma = True
    add_prior = True
    prior = None #np.zeros(hp.nside2npix(nside)) # None
    fit_gamma = 2.
    # Source parameters for injection
    src_dec = 0.
    src_ra = np.pi
    src_sigma = np.radians(6.)
    src_gamma = 2.
    print "Important parameters: "
    print "fixed_gamma is ", fixed_gamma
    print "add_prior is ", add_prior
    
    llh, mc = utils.startup(Nsrc=10, fixed_gamma=fixed_gamma,
                            gamma_inj=src_gamma,
                            src_dec=src_dec, src_ra=src_ra,
                            add_prior=add_prior
                            ) 
    print(llh)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(
                                nside=nside, follow_up_factor=1,
                                pVal=pVal_func,
                                hemispheres=dict(Full=np.radians([-90., 90.])),
                                prior=prior,
                                pdec=src_dec,
                                pra=src_ra,
                                psig=src_sigma,
                                fit_gamma=fit_gamma)
                                ):

        if i > 0:
            # break after first follow up
            break
    #print "hotspot data is: ", hotspot
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

    fig, ax = utils.skymap(plt, scan["pVal"], cmap=cmap,
                           vmin=0., vmax=np.ceil(hotspot["Full"]["best"]["pVal"]),
                           colorbar=dict(title=r"$-\log_{10}\rm p$"),
                           rasterized=True)

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
    #'''
    if isinstance(llh, MultiPointSourceLLH):
        for llh in llh._sams.itervalues():
            ax.scatter(np.pi - llh.exp["ra"], np.arcsin(llh.exp["sinDec"]), 1,
                       marker="x",
                       #color=plt.gca()._get_lines.color_cycle.next(),
                       alpha=0.2)#, rasterized=True)
    else:
        ax.scatter(np.pi - hotspot["Full"]["best"]["ra"], hotspot["Full"]["best"]["dec"], 10,
                   marker="x",
                   color='lightgreen',
                   alpha=0.45)
        ax.text(np.pi, np.pi/2.,
                "pVal: {:1.2f} \n TS: {:1.2f}".format(hotspot["Full"]["best"]["pVal"], hotspot["Full"]["best"]["TS"]))
        ax.scatter(np.pi - src_ra, src_dec, 10,
                   marker="x",
                   color='lightblue',
                   alpha=0.45)
    #'''
    
    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig.savefig("figures/skymap_pVal.pdf", dpi=256)

    if add_prior:
        fig, ax = utils.skymap(plt, scan['prior'], cmap=cmap,
                               rasterized=True)

        fig.savefig("figures/prior.pdf", dpi=256)

    for key in ["TS"] + llh.params:
        if fixed_gamma and key == "gamma": continue # skip gamma, if fixed
        eps = 0.1 if key != "TS" else 0.0
        vmin, vmax = np.percentile(scan[key], [eps, 100. - eps])
        vmin = np.floor(max(0, vmin))
        vmax = min(8, np.ceil(vmax))
        q = np.ma.masked_array(scan[key])
        q.mask = ~(scan["nsources"] > 0.5) if key != "TS" else np.zeros_like(q, dtype=np.bool)
        fig, ax = utils.skymap(plt, q, cmap=cmap,
                               vmin=vmin, vmax=vmax,
                               colorbar=dict(title=label[key]),
                               rasterized=True)

        fig.savefig("figures/skymap_" + key +".pdf", dpi=256)
        plt.close("all")
