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

# convert test statistic to a p-value for a given point
pVal_func = lambda TS, dec: -np.log10(0.5 * (chi2(len(llh.params)).sf(TS)
                                             + chi2(len(llh.params)).cdf(-TS)))

label = dict(TS=r"$\mathcal{TS}$",
             nsources=r"$n_S$",
             gamma=r"$\gamma$",
             )

if __name__=="__main__":
    
    backend = "svg"
    extension = "_multi.png"
    
    plt = utils.plotting(backend=backend)
    nside = 2**4
        
    llh, mc = utils.startup(multi=True) 
    print(llh)
    # iterator of all-sky scan with follow up scans of most interesting points
    for i, (scan, hotspot) in enumerate(llh.all_sky_scan(
                                nside=nside, follow_up_factor=1,
                                pVal=pVal_func,
                                hemispheres=dict(Full=np.radians([-90., 90.])))
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
    
    if not os.path.exists("figures"):
        os.makedirs("figures")

    fig.savefig("figures/skymap_pVal"+extension, dpi=256)

    for key in ["TS"] + llh.params:
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

        fig.savefig("figures/skymap_" + key + extension, dpi=256)
        plt.close("all")
